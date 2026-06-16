"""
Clickstream encoder training pipeline.

Multi-task objective: CE (archetype classification) + NT-Xent (individual identity).

Each participant's web sessions are grouped from the clickstream log (anonymous
sessions excluded). Sessions are tokenised per-event via ClickstreamVocabulary, then
each session is reduced to a fixed vector by the GRU (``ClickstreamEncoder.encode_session``),
and the per-customer session vectors are mean-pooled into one embedding
(``ClickstreamEncoder.forward``).

For NT-Xent (individual identity), each participant's sessions are split at the
median timestamp into two chronological halves; both halves are encoded and treated
as a positive pair. Archetype CE uses the full (all-sessions) customer embedding.

Training configuration (mirrors the transaction encoder):
    - Batch size: 64 (participant-level)
    - Learning rate: 5e-4
    - Epochs: 30
    - Optimiser: AdamW, weight_decay=1e-4
    - Train/val split: 80/20 by participant_id
    - Early stopping on val loss
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset

from schemas import CHECKPOINT_PATHS, EMBEDDING_DIM, PERSONA_TO_IDX
from schemas.clickstream import ClickstreamEvent
from encoders.clickstream.features import (
    MAX_EVENTS_PER_SESSION,
    MAX_SESSIONS,
    TOKEN_DIM,
    ClickstreamVocabulary,
)
from encoders.clickstream.model import ClickstreamEncoder

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/synthetic")
CLICKSTREAM_FILE = DATA_DIR / "clickstream.jsonl"
PSYCHOGRAPHICS_FILE = DATA_DIR / "psychographics.jsonl"

# Training defaults
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 5e-4
DEFAULT_EPOCHS = 30
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GRU_HIDDEN = 128
DEFAULT_GRU_LAYERS = 2
DEFAULT_GRU_DROPOUT = 0.1
DEFAULT_PROJECTION_DIM = 64
DEFAULT_PATIENCE = 5
RANDOM_SEED = 42
MIN_SESSIONS_FOR_SPLIT = (
    2  # participants below this threshold skip NT-Xent (CE still applied)
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


# Per-participant item:
#   (full_session_tokens, full_session_lens, full_session_mask,
#    h1_session_tokens, h1_session_lens, h1_session_mask,
#    h2_session_tokens, h2_session_lens, h2_session_mask,
#    persona_label, nt_xent_eligible)
_Item = tuple[
    list[torch.Tensor],  # full session token tensors, each (T, TOKEN_DIM)
    list[int],  # full session lengths
    int,  # number of real sessions (full)
    list[torch.Tensor],  # half1 session token tensors
    list[int],  # half1 session lengths
    int,  # number of real sessions (half1)
    list[torch.Tensor],  # half2 session token tensors
    list[int],  # half2 session lengths
    int,  # number of real sessions (half2)
    int,  # persona label idx
    bool,  # NT-Xent eligible
]


class ClickstreamCustomerDataset(Dataset):
    """Per-participant dataset with chronological session splits for NT-Xent.

    Sessions are tokenised up-front via ``ClickstreamVocabulary.encode_session``
    (one ``(T, TOKEN_DIM)`` tensor per session). Each item stores:

      - full session set (most-recent MAX_SESSIONS) for archetype CE
      - first (oldest) half and second (most-recent) half of sessions for NT-Xent
      - persona_label for archetype CE
      - nt_xent_eligible: False when fewer than MIN_SESSIONS_FOR_SPLIT sessions

    Sessions are ordered oldest-first so the chronological split at the median
    yields an oldest half and a most-recent half.
    """

    def __init__(
        self,
        events_by_participant: dict[str, list[list[ClickstreamEvent]]],
        vocab: ClickstreamVocabulary,
        max_sessions: int = MAX_SESSIONS,
        max_events_per_session: int = MAX_EVENTS_PER_SESSION,
    ) -> None:
        n_skipped = 0
        self.items: list[_Item] = []

        for _, sessions in events_by_participant.items():
            # Sort sessions oldest-first by first event timestamp.
            sessions_sorted = sorted(sessions, key=lambda s: s[0].event_ts if s else "")
            # Truncate to most-recent max_sessions (keep the tail).
            sessions_sorted = sessions_sorted[-max_sessions:]

            # Tokenise each session into a (T, TOKEN_DIM) tensor.
            full_tokens = [
                vocab.encode_session(s)[:max_events_per_session].detach()
                for s in sessions_sorted
            ]
            full_lens = [t.size(0) for t in full_tokens]
            n_full = len(full_tokens)

            persona_id = sessions_sorted[0][0].participant_id
            # Derive persona from participant_id prefix (participant_id is
            # "<persona>_<NNNN>"); fall back to psychographics map at load time.
            persona_label = PERSONA_TO_IDX.get(_persona_prefix(persona_id), 0)

            eligible = n_full >= MIN_SESSIONS_FOR_SPLIT
            if not eligible:
                n_skipped += 1
                h1_tokens = full_tokens
                h1_lens = full_lens
                h2_tokens = full_tokens
                h2_lens = full_lens
            else:
                mid = n_full // 2
                h1_tokens = full_tokens[:mid]
                h1_lens = full_lens[:mid]
                h2_tokens = full_tokens[mid:]
                h2_lens = full_lens[mid:]

            self.items.append(
                (
                    full_tokens,
                    full_lens,
                    n_full,
                    h1_tokens,
                    h1_lens,
                    len(h1_tokens),
                    h2_tokens,
                    h2_lens,
                    len(h2_tokens),
                    persona_label,
                    eligible,
                )
            )

        if n_skipped:
            logger.warning(
                "%d participants have fewer than %d sessions and will be "
                "excluded from the NT-Xent loss (CE loss still applied).",
                n_skipped,
                MIN_SESSIONS_FOR_SPLIT,
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> _Item:
        return self.items[idx]


def _persona_prefix(participant_id: str) -> str:
    """Extract the persona prefix from a participant_id like 'price_lex_0000'."""
    if "_" not in participant_id:
        return participant_id
    # Persona labels contain underscores (e.g. price_lex), so match against the
    # canonical PERSONA_TO_IDX keys by longest-prefix match.
    for label in sorted(PERSONA_TO_IDX, key=len, reverse=True):
        if participant_id.startswith(label + "_") or participant_id == label:
            return label
    # Fallback: strip the trailing numeric segment.
    return participant_id.rsplit("_", 1)[0]


def _pad_sessions(
    token_lists: list[list[torch.Tensor]],
    lens_lists: list[list[int]],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Pad a batch of customers' session-token lists into per-customer tensors.

    For each customer, sessions are right-padded to the batch-wide max event
    length, producing ``(N_sessions, T_max, TOKEN_DIM)`` token tensors and
    ``(N_sessions,)`` length tensors.

    Returns
    -------
    session_token_batches : list of (N_i, T_max, TOKEN_DIM) tensors
    session_len_batches   : list of (N_i,) long tensors
    """
    out_tokens: list[torch.Tensor] = []
    out_lens: list[torch.Tensor] = []
    for tokens, lens in zip(token_lists, lens_lists):
        n = len(tokens)
        if n == 0:
            out_tokens.append(torch.zeros(1, 1, TOKEN_DIM))
            out_lens.append(torch.tensor([1], dtype=torch.long))
            continue
        max_t = max(lens) if lens else 1
        padded = torch.zeros(n, max_t, TOKEN_DIM)
        for i, (tok, tlen) in enumerate(zip(tokens, lens)):
            padded[i, :tlen] = tok[:tlen]
        out_tokens.append(padded)
        out_lens.append(torch.tensor(lens, dtype=torch.long))
    return out_tokens, out_lens


def _collate_view(
    batch_items: list[_Item], view_idx: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate one view (full/h1/h2) across a batch.

    view_idx selects the (tokens, lens, n_sessions) triplet inside each item:
      full -> 0,1,2 ; h1 -> 3,4,5 ; h2 -> 6,7,8.

    Pads sessions to a batch-wide max session count (MAX_SESSIONS) and event
    length, producing session-level token tensors and masks for
    ``ClickstreamEncoder.encode_session`` / ``forward``.
    """
    token_lists: list[list[torch.Tensor]] = [
        item[view_idx]
        for item in batch_items  # type: ignore[assignment]
    ]
    lens_lists: list[list[int]] = [
        item[view_idx + 1]
        for item in batch_items  # type: ignore[assignment]
    ]
    n_sessions: list[int] = [
        int(item[view_idx + 2])  # type: ignore[arg-type]
        for item in batch_items
    ]
    max_n = max(n_sessions) if n_sessions else 1

    per_cust_tokens, per_cust_lens = _pad_sessions(token_lists, lens_lists)

    # Batch-wide max event length across all customers/sessions.
    global_max_t = max(int(t.size(1)) for t in per_cust_tokens)

    B = len(batch_items)
    session_tokens = torch.zeros(B, max_n, global_max_t, TOKEN_DIM)
    session_lens = torch.ones(B, max_n, dtype=torch.long)
    session_mask = torch.zeros(B, max_n, dtype=torch.bool)

    for i, (toks, lens, n) in enumerate(
        zip(per_cust_tokens, per_cust_lens, n_sessions)
    ):
        n_clamped = min(n, toks.size(0))
        for j in range(n_clamped):
            tlen = int(lens[j])
            tlen = min(tlen, toks.size(1), global_max_t)
            session_tokens[i, j, :tlen] = toks[j, :tlen]
            session_lens[i, j] = max(tlen, 1)
        session_mask[i, :n_clamped] = True

    return session_tokens, session_lens, session_mask


def collate_fn(
    batch: list[_Item],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[bool],
]:
    """Collate a batch of ClickstreamCustomerDataset items.

    Produces three customer-level view batches (full, half1, half2), each with
    session-token tensors, session lengths, and session masks.

    Returns
    -------
    full_session_tokens : (B, N_full, T_full, TOKEN_DIM)
    full_session_lens   : (B, N_full)
    full_session_mask   : (B, N_full)  bool
    h1_session_tokens   : (B, N_h1, T_h1, TOKEN_DIM)
    h1_session_lens     : (B, N_h1)
    h1_session_mask     : (B, N_h1)  bool
    h2_session_tokens   : (B, N_h2, T_h2, TOKEN_DIM)
    h2_session_lens     : (B, N_h2)
    h2_session_mask     : (B, N_h2)  bool
    labels              : (B,)  long
    eligible            : list[bool], length B
    """
    full_t, full_l, full_m = _collate_view(batch, 0)
    h1_t, h1_l, h1_m = _collate_view(batch, 3)
    h2_t, h2_l, h2_m = _collate_view(batch, 6)
    labels = torch.tensor([item[9] for item in batch], dtype=torch.long)
    eligible = [item[10] for item in batch]
    return (
        full_t,
        full_l,
        full_m,
        h1_t,
        h1_l,
        h1_m,
        h2_t,
        h2_l,
        h2_m,
        labels,
        eligible,
    )


# ---------------------------------------------------------------------------
# NT-Xent for split-view pairs
# ---------------------------------------------------------------------------


def nt_xent_views(
    emb_v1: torch.Tensor,
    emb_v2: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """NT-Xent for position-matched split-session participant pairs."""
    B = emb_v1.size(0)
    if B < 2:
        return torch.tensor(0.0, device=emb_v1.device, requires_grad=True)

    embs = F.normalize(torch.cat([emb_v1, emb_v2], dim=0), dim=1)
    sim = torch.mm(embs, embs.t()) / temperature

    labels = torch.cat(
        [
            torch.arange(B, 2 * B, device=emb_v1.device),
            torch.arange(0, B, device=emb_v1.device),
        ]
    )
    mask = torch.eye(2 * B, dtype=torch.bool, device=emb_v1.device)
    sim = sim.masked_fill(mask, float("-inf"))
    return F.cross_entropy(sim, labels)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_customer(
    encoder: ClickstreamEncoder,
    session_tokens: torch.Tensor,
    session_lens: torch.Tensor,
    session_mask: torch.Tensor,
) -> torch.Tensor:
    """Encode a batch of customers into (B, EMBEDDING_DIM) embeddings.

    Flattens the (B, N, T, TOKEN_DIM) token tensor to (B*N, T, TOKEN_DIM),
    runs the session GRU, reshapes back, and mean-pools with the mask.
    """
    B, N, T = session_tokens.size(0), session_tokens.size(1), session_tokens.size(2)
    flat_tokens = session_tokens.view(B * N, T, TOKEN_DIM)
    flat_lens = session_lens.view(B * N)
    # Guard against zero-length sessions (set to 1 for pack_padded_sequence).
    flat_lens = flat_lens.clamp(min=1)
    session_embs = encoder.encode_session(flat_tokens, flat_lens)  # (B*N, gru_hidden)
    session_embs = session_embs.view(B, N, -1)
    return encoder(session_embs, session_mask)  # (B, EMBEDDING_DIM)


def split_by_participant(
    events_by_participant: dict[str, list[list[ClickstreamEvent]]],
    val_fraction: float = 0.2,
    seed: int = RANDOM_SEED,
) -> tuple[
    dict[str, list[list[ClickstreamEvent]]],
    dict[str, list[list[ClickstreamEvent]]],
]:
    """Split sessions by participant_id (no leakage)."""
    participant_ids = sorted(events_by_participant.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(participant_ids)

    split = int((1 - val_fraction) * len(participant_ids))
    train_ids = set(participant_ids[:split])
    val_ids = set(participant_ids[split:])

    train_data = {pid: events_by_participant[pid] for pid in train_ids}
    val_data = {pid: events_by_participant[pid] for pid in val_ids}

    logger.info(
        "Split: %d train participants, %d val participants",
        len(train_ids),
        len(val_ids),
    )
    return train_data, val_data


def load_data(
    clickstream_path: Path | str | None = None,
    psychographics_path: Path | str | None = None,
) -> tuple[dict[str, list[list[ClickstreamEvent]]], dict[str, str]]:
    """Load clickstream events grouped by participant, and persona labels.

    Returns
    -------
    events_by_participant : dict[participant_id -> list[sessions]]
        Each session is a list of ClickstreamEvent. Anonymous sessions
        (customer_id == 'anonymous' / participant_id == '') are excluded.
    persona_map : dict[participant_id -> persona_id]
    """
    clickstream_path = Path(clickstream_path) if clickstream_path else CLICKSTREAM_FILE
    psychographics_path = (
        Path(psychographics_path) if psychographics_path else PSYCHOGRAPHICS_FILE
    )

    # Load persona_id mapping (participant_id -> persona_id).
    persona_map: dict[str, str] = {}
    with psychographics_path.open() as f:
        for line in f:
            rec = json.loads(line)
            pid = rec.get("participant_id", "")
            persona = rec.get("persona_id", "")
            if pid and persona:
                persona_map[pid] = persona

    # Load clickstream events, group by participant then session.
    raw_by_participant_session: dict[str, dict[str, list[ClickstreamEvent]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    n_anonymous = 0
    n_skipped_summary = 0
    with clickstream_path.open() as f:
        for line in f:
            rec = json.loads(line)
            pid = rec.get("participant_id", "")
            customer_id = rec.get("customer_id", "")
            if not pid or customer_id == "anonymous":
                n_anonymous += 1
                continue
            # The file interleaves ClickstreamEvent rows with SessionSummary
            # rows (carrying "n_events"). Only event rows carry "event_type";
            # skip summaries.
            if "event_type" not in rec:
                n_skipped_summary += 1
                continue
            event = ClickstreamEvent(**rec)
            sid = rec.get("session_id", "")
            raw_by_participant_session[pid][sid].append(event)

    events_by_participant: dict[str, list[list[ClickstreamEvent]]] = {}
    for pid, sessions in raw_by_participant_session.items():
        session_lists = list(sessions.values())
        # Sort events within each session by timestamp.
        session_lists = [sorted(s, key=lambda e: e.event_ts) for s in session_lists]
        events_by_participant[pid] = session_lists

    logger.info(
        "Loaded %d clickstream events across %d participants "
        "(%d anonymous rows, %d session-summary rows excluded)",
        sum(len(s) for sess in events_by_participant.values() for s in sess),
        len(events_by_participant),
        n_anonymous,
        n_skipped_summary,
    )
    return events_by_participant, persona_map


# ---------------------------------------------------------------------------
# Strategy recovery probe
# ---------------------------------------------------------------------------


def strategy_recovery(
    encoder: ClickstreamEncoder,
    dataset: ClickstreamCustomerDataset,
    device: torch.device,
) -> float:
    """Strategy recovery: LogisticRegression(embedding -> archetype) val accuracy.

    Encodes all customers in the dataset to 128-dim embeddings, then fits a
    logistic regression predicting the archetype label and returns the held-out
    (in-sample, given the dataset is the val split) accuracy.
    """
    encoder.eval()
    embeddings: list[np.ndarray] = []
    labels: list[int] = []

    with torch.no_grad():
        for item in dataset:
            (
                full_tokens,
                full_lens,
                _n_full,
                *_,
                label,
                _eligible,
            ) = item
            # Encode the single customer's sessions.
            n = len(full_tokens)
            if n == 0:
                embeddings.append(np.zeros(EMBEDDING_DIM, dtype=np.float32))
                labels.append(label)
                continue
            max_t = max(full_lens) if full_lens else 1
            tokens = torch.zeros(1, n, max_t, TOKEN_DIM)
            lens = torch.ones(n, dtype=torch.long)
            for j, (tok, tlen) in enumerate(zip(full_tokens, full_lens)):
                tlen_c = min(tlen, max_t)
                tokens[0, j, :tlen_c] = tok[:tlen_c]
                lens[j] = max(tlen_c, 1)
            mask = torch.ones(1, n, dtype=torch.bool)
            emb = _encode_customer(
                encoder,
                tokens.to(device),
                lens.unsqueeze(0).to(device),
                mask.to(device),
            )
            embeddings.append(emb.squeeze(0).cpu().numpy())
            labels.append(label)

    X = np.stack(embeddings)
    y = np.array(labels)
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    clf.fit(X, y)
    acc = float(clf.score(X, y))
    return acc


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    events_by_participant: dict[str, list[list[ClickstreamEvent]]] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    n_epochs: int = DEFAULT_EPOCHS,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    gru_hidden: int = DEFAULT_GRU_HIDDEN,
    gru_layers: int = DEFAULT_GRU_LAYERS,
    gru_dropout: float = DEFAULT_GRU_DROPOUT,
    projection_dim: int = DEFAULT_PROJECTION_DIM,
    patience: int = DEFAULT_PATIENCE,
    seed: int = RANDOM_SEED,
    device: str | None = None,
    lambda_contrastive: float = 0.5,
    nt_xent_temperature: float = 0.07,
    save_path: Path | None = None,
) -> ClickstreamEncoder:
    """Train the clickstream encoder with CE (archetype) + NT-Xent (individual identity).

    Split-session views: each participant's sessions are split at the median
    timestamp into oldest-half and most-recent-half. Both halves are encoded via
    the GRU session encoder + customer pooling; NT-Xent treats them as a
    positive pair. Archetype CE uses the full-sessions customer embedding.

    Parameters
    ----------
    events_by_participant
        Sessions grouped by participant. If None, loads from
        data/synthetic/clickstream.jsonl.
    batch_size
        Participant-level batch size.
    lr, weight_decay
        AdamW hyperparameters.
    n_epochs, patience
        Training budget and early stopping.
    gru_hidden, gru_layers, gru_dropout, projection_dim
        Encoder architecture (must match existing checkpoint dims if fine-tuning).
    seed
        Random seed for train/val split.
    device
        Device string. Defaults to "cuda" if available, else "cpu".
    lambda_contrastive
        Weight for NT-Xent loss (total = CE + lambda * NT-Xent).
    nt_xent_temperature
        Temperature for NT-Xent.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    if events_by_participant is None:
        events_by_participant, _persona_map = load_data()

    train_data, val_data = split_by_participant(events_by_participant, seed=seed)

    vocab = ClickstreamVocabulary()

    encoder = ClickstreamEncoder(
        vocab=vocab,
        projection_dim=projection_dim,
        gru_hidden=gru_hidden,
        gru_layers=gru_layers,
        gru_dropout=gru_dropout,
    ).to(torch_device)

    train_ds = ClickstreamCustomerDataset(train_data, vocab)
    val_ds = ClickstreamCustomerDataset(val_data, vocab)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    optimiser = torch.optim.AdamW(
        encoder.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    ce_criterion = nn.CrossEntropyLoss()

    with mlflow.start_run(run_name="clickstream_encoder_v1_contrastive"):
        mlflow.set_tag("modality", "clickstream")
        mlflow.log_params(
            {
                "lr": lr,
                "batch_size": batch_size,
                "gru_hidden": gru_hidden,
                "gru_layers": gru_layers,
                "gru_dropout": gru_dropout,
                "projection_dim": projection_dim,
                "weight_decay": weight_decay,
                "n_epochs": n_epochs,
                "patience": patience,
                "seed": seed,
                "lambda_contrastive": lambda_contrastive,
                "nt_xent_temperature": nt_xent_temperature,
                "objective": "ce+nt_xent_split_sessions",
            }
        )

        best_val_loss = float("inf")
        best_encoder_state: dict[str, torch.Tensor] | None = None
        patience_counter = 0
        first_epoch_ce: float | None = None
        last_epoch_ce: float | None = None

        for epoch in range(n_epochs):
            encoder.train()
            epoch_ce = 0.0
            epoch_nt = 0.0
            n_batches = 0

            for (
                full_t,
                full_l,
                full_m,
                h1_t,
                h1_l,
                h1_m,
                h2_t,
                h2_l,
                h2_m,
                labels,
                eligible,
            ) in train_loader:
                full_t = full_t.to(torch_device)
                full_l = full_l.to(torch_device)
                full_m = full_m.to(torch_device)
                h1_t = h1_t.to(torch_device)
                h1_l = h1_l.to(torch_device)
                h1_m = h1_m.to(torch_device)
                h2_t = h2_t.to(torch_device)
                h2_l = h2_l.to(torch_device)
                h2_m = h2_m.to(torch_device)
                labels = labels.to(torch_device)

                # Archetype CE on full customer embedding (forward_with_logits).
                _, full_logits = encoder.forward_with_logits(
                    _session_embeddings(encoder, full_t, full_l),
                    full_m,
                )
                ce_loss = ce_criterion(full_logits, labels)

                # NT-Xent on split-session views (eligible participants only).
                elig_mask = torch.tensor(eligible, device=torch_device)
                nt_loss = torch.tensor(0.0, device=torch_device)
                if int(elig_mask.sum()) >= 2:
                    emb_h1 = _encode_customer(encoder, h1_t, h1_l, h1_m)
                    emb_h2 = _encode_customer(encoder, h2_t, h2_l, h2_m)
                    idx = elig_mask.nonzero(as_tuple=True)[0]
                    nt_loss = nt_xent_views(
                        emb_h1[idx], emb_h2[idx], nt_xent_temperature
                    )

                loss = ce_loss + lambda_contrastive * nt_loss

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                epoch_ce += ce_loss.item()
                epoch_nt += nt_loss.item()
                n_batches += 1

            avg_ce = epoch_ce / max(n_batches, 1)
            avg_nt = epoch_nt / max(n_batches, 1)
            last_epoch_ce = avg_ce
            if first_epoch_ce is None:
                first_epoch_ce = avg_ce

            # Validation: CE loss + accuracy on full-sessions archetype prediction.
            encoder.eval()
            val_ce = 0.0
            val_correct = 0
            val_total = 0
            n_val_batches = 0

            with torch.no_grad():
                for full_t, full_l, full_m, *_rest in val_loader:
                    full_t = full_t.to(torch_device)
                    full_l = full_l.to(torch_device)
                    full_m = full_m.to(torch_device)
                    labels = _rest[-2].to(torch_device)

                    full_session_embs = _session_embeddings(encoder, full_t, full_l)
                    _, full_logits = encoder.forward_with_logits(
                        full_session_embs, full_m
                    )

                    val_ce += ce_criterion(full_logits, labels).item()
                    val_correct += (full_logits.argmax(dim=1) == labels).sum().item()
                    val_total += labels.size(0)
                    n_val_batches += 1

            avg_val_ce = val_ce / max(n_val_batches, 1)
            val_acc = val_correct / max(val_total, 1)

            mlflow.log_metrics(
                {
                    "train_ce_loss": avg_ce,
                    "train_nt_loss": avg_nt,
                    "val_ce_loss": avg_val_ce,
                    "val_acc": val_acc,
                },
                step=epoch,
            )

            logger.info(
                "Epoch %d/%d  ce=%.4f  nt=%.4f  val_ce=%.4f  val_acc=%.4f",
                epoch + 1,
                n_epochs,
                avg_ce,
                avg_nt,
                avg_val_ce,
                val_acc,
            )

            if avg_val_ce < best_val_loss:
                best_val_loss = avg_val_ce
                patience_counter = 0
                best_encoder_state = {
                    k: v.clone() for k, v in encoder.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        if best_encoder_state is not None:
            encoder.load_state_dict(best_encoder_state)

        mlflow.log_metric("best_val_loss", best_val_loss)

    _save_path = save_path if save_path is not None else CHECKPOINT_PATHS["clickstream"]
    _save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), _save_path)
    logger.info("Checkpoint saved to %s", _save_path)
    logger.info("Training complete. Best val loss: %.4f", best_val_loss)

    # Strategy recovery probe (on val split).
    rec = strategy_recovery(encoder, val_ds, torch_device)
    logger.info("Strategy recovery (val, LogisticRegression): %.4f", rec)
    logger.info(
        "CE first=%s -> final=%s (decreased=%s)",
        f"{first_epoch_ce:.4f}" if first_epoch_ce is not None else "n/a",
        f"{last_epoch_ce:.4f}" if last_epoch_ce is not None else "n/a",
        first_epoch_ce is not None
        and last_epoch_ce is not None
        and last_epoch_ce < first_epoch_ce,
    )

    return encoder


def _session_embeddings(
    encoder: ClickstreamEncoder,
    session_tokens: torch.Tensor,
    session_lens: torch.Tensor,
) -> torch.Tensor:
    """Encode a (B, N, T, TOKEN_DIM) batch of sessions into (B, N, gru_hidden).

    Helper that flattens, runs the session GRU, and reshapes — returns the
    session-level embeddings (before customer pooling) so that
    ``forward_with_logits`` can be applied.
    """
    B, N, T = session_tokens.size(0), session_tokens.size(1), session_tokens.size(2)
    flat_tokens = session_tokens.view(B * N, T, TOKEN_DIM)
    flat_lens = session_lens.view(B * N).clamp(min=1)
    session_embs = encoder.encode_session(flat_tokens, flat_lens)
    return session_embs.view(B, N, -1)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(override=True)
    logging.basicConfig(level=logging.INFO)
    train()
