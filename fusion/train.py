"""
fusion/train.py

Training script for the late fusion meta-learner.

Loads all four frozen modality encoders, generates cached embeddings,
and trains the LateFusionMetaLearner on persona classification.

Usage:
    uv run python -m fusion.train
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import mlflow

from schemas import CHECKPOINT_PATHS, EMBEDDING_DIM, PERSONA_TO_IDX
from fusion.meta_learner import LateFusionMetaLearner


# ---------------------------------------------------------------------------
# Encoder loading
# ---------------------------------------------------------------------------


def load_encoders(
    modalities: list[str] | None = None,
    device: str = "cpu",
) -> dict[str, nn.Module]:
    """Load a configurable set of encoder checkpoints (frozen).

    Parameters
    ----------
    modalities : list[str] | None
        Modality names to load. If None, loads every key in
        ``schemas.CHECKPOINT_PATHS`` except ``"fusion"`` (trace, transaction,
        text, psychographic, clickstream, campaign). Pass an explicit subset
        (e.g. excluding ``"text"`` for the 5-modality dry run) to load only
        those encoders.
    device : str
        Target device for models ("cpu" or "cuda").

    Returns
    -------
    dict[str, nn.Module]
        Dictionary mapping modality names to frozen encoder models.
    """
    from encoders.campaign.model import CampaignEncoder
    from encoders.clickstream.model import ClickstreamEncoder
    from encoders.psychographic.model import PsychographicEncoder
    from encoders.text.embed import TextEncoder
    from encoders.trace.model import TraceEncoder
    from encoders.transaction.model import TransactionEncoder

    if modalities is None:
        modalities = [m for m in CHECKPOINT_PATHS if m != "fusion"]

    encoders: dict[str, nn.Module] = {}

    def _freeze(model: nn.Module) -> None:
        for param in model.parameters():
            param.requires_grad = False

    for name in modalities:
        if name == "trace":
            encoder = TraceEncoder(n_classes=7).to(device)
        elif name == "transaction":
            encoder = TransactionEncoder().to(device)
        elif name == "text":
            encoder = TextEncoder(n_classes=7).to(device)
        elif name == "psychographic":
            encoder = PsychographicEncoder(n_classes=7).to(device)
        elif name == "clickstream":
            # vocab=None builds the default ClickstreamVocabulary.
            encoder = ClickstreamEncoder().to(device)
        elif name == "campaign":
            # vocab=None builds the default CampaignVocabulary.
            encoder = CampaignEncoder().to(device)
        else:
            raise ValueError(f"Unknown modality for load_encoders: {name!r}")

        state = torch.load(
            CHECKPOINT_PATHS[name], map_location=device, weights_only=True
        )
        # strict=False: checkpoints save the backbone only (classifier head
        # dropped after training); missing keys are expected.
        encoder.load_state_dict(state, strict=False)
        _freeze(encoder)
        encoders[name] = encoder

    return encoders


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_psychographics(
    path: Path = Path("data/synthetic/psychographics.jsonl"),
) -> list[dict]:
    """Load psychographic records to get participant IDs and labels.

    Parameters
    ----------
    path : Path
        Path to psychographics.jsonl.

    Returns
    -------
    list[dict]
        List of psychographic records with participant_id and persona_id.
    """
    records = []
    for line in path.read_text().strip().splitlines():
        record = json.loads(line)
        records.append(record)
    return records


def split_by_participant(
    participant_ids: list[str],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Split participant IDs into train/val sets.

    Uses the same seed and ratio as encoder probes to ensure identical val sets.

    Parameters
    ----------
    participant_ids : list[str]
        All participant IDs.
    train_ratio : float
        Fraction of participants assigned to train set.
    seed : int
        Random seed for reproducible splits.

    Returns
    -------
    tuple[list[str], list[str]]
        (train_ids, val_ids) — participant ID lists for train/val splits.
    """
    rng = np.random.default_rng(seed=seed)
    shuffled_ids = rng.permutation(participant_ids)
    split_idx = int(train_ratio * len(shuffled_ids))
    train_ids = shuffled_ids[:split_idx].tolist()
    val_ids = shuffled_ids[split_idx:].tolist()
    return train_ids, val_ids


def load_modality_data(
    modality: str,
    participant_ids: list[str],
) -> dict[str, dict]:
    """Load modality data and index by participant_id.

    Parameters
    ----------
    modality : str
        Modality name ("traces", "trials", "transactions", "narratives").
    participant_ids : list[str]
        Canonical participant ID ordering.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary mapping participant_id to modality data tensor.
    """
    data_path = Path(f"data/synthetic/{modality}.jsonl")
    participant_to_data = {}

    for line in data_path.read_text().strip().splitlines():
        record = json.loads(line)
        participant_id = record.get("participant_id")
        if participant_id in participant_ids:
            participant_to_data[participant_id] = record

    return participant_to_data


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------


def generate_embeddings(
    encoders: dict[str, nn.Module],
    psychographics: list[dict],
    modality_data: dict[str, dict[str, dict]],
    device: str = "cpu",
) -> dict:
    """Generate embeddings for all participants using frozen encoders.

    Produces a per-modality ``[N, EMBEDDING_DIM]`` tensor for every modality in
    ``encoders`` (trace/transaction/text/psychographic/clickstream/campaign as
    loaded), plus the ``labels`` and ``participant_ids`` bookkeeping keys.
    Modalities absent from ``encoders`` are simply not produced.
    """
    from collections import defaultdict

    from encoders.psychographic.features import to_feature_vector
    from encoders.text.embed import TextEncoder as TxtEncoder
    from encoders.transaction.model import TransactionEncoder as TxEncoder
    from schemas.psychographic import PsychographicVector
    from schemas.trace import AcquisitionEvent, TrialRecord
    from schemas.transaction import TransactionRecord

    n_participants = len(psychographics)

    # Dynamic embeddings dict — one zero tensor per loaded modality, plus the
    # bookkeeping keys. No hardcoded modality count.
    embeddings: dict = {}
    for name in encoders:
        embeddings[name] = torch.zeros(
            n_participants,
            EMBEDDING_DIM,
            device=device,  # type: ignore[reportPrivateImportUsage]
        )
    embeddings["labels"] = torch.zeros(
        n_participants,
        dtype=torch.long,
        device=device,  # type: ignore[reportPrivateImportUsage]
    )
    embeddings["participant_ids"] = []

    # Build pid → index map
    pid_to_idx = {p["participant_id"]: i for i, p in enumerate(psychographics)}

    # ── Trace: events + trials grouped by participant ─────────────────────────
    events_by_pid: dict[str, list] = defaultdict(list)
    trials_by_pid: dict[str, list] = defaultdict(list)

    if "trace" in encoders:
        from encoders.trace.tokeniser import build_vocab, tokenise_trial

        traces_path = Path("data/synthetic/traces.jsonl")
        if traces_path.exists():
            for line in traces_path.read_text().strip().splitlines():
                r = json.loads(line)
                pid = r.get("participant_id", "")
                if pid in pid_to_idx:
                    # Filter to known dataclass fields: traces.jsonl carries extra
                    # keys (e.g. "month") added by the schema-update epic that the
                    # immutable AcquisitionEvent does not accept. Same pattern as
                    # encoders/trace/train.py load_data().
                    events_by_pid[pid].append(
                        AcquisitionEvent(
                            **{
                                k: v
                                for k, v in r.items()
                                if k in AcquisitionEvent.__dataclass_fields__
                            }
                        )
                    )

        trials_path = Path("data/synthetic/trials.jsonl")
        if trials_path.exists():
            for line in trials_path.read_text().strip().splitlines():
                r = json.loads(line)
                pid = r.get("participant_id", "")
                if pid in pid_to_idx:
                    trials_by_pid[pid].append(
                        TrialRecord(
                            **{
                                k: v
                                for k, v in r.items()
                                if k in TrialRecord.__dataclass_fields__
                            }
                        )
                    )

        all_events = [ev for evs in events_by_pid.values() for ev in evs]
        trace_vocab = build_vocab(all_events)
    else:
        trace_vocab = None

    # ── Transaction records grouped by participant ────────────────────────────
    # sort_transactions_most_recent_first is imported lazily in the per-participant
    # loop below (where it is used), so no top-level import is needed here.
    tx_by_pid: dict[str, list] = defaultdict(list)
    if "transaction" in encoders:
        tx_path = Path("data/synthetic/transactions.jsonl")
        if tx_path.exists():
            for line in tx_path.read_text().strip().splitlines():
                r = json.loads(line)
                pid = r.get("participant_id", "")
                if pid in pid_to_idx:
                    tx_by_pid[pid].append(r)

    # ── Clickstream sessions grouped by participant ───────────────────────────
    # Mirrors encoders/clickstream/train.py load_data(): anonymous sessions
    # (customer_id == "anonymous") are excluded, and only event rows (those
    # carrying "event_type") are kept — the file interleaves SessionSummary rows.
    click_sessions_by_pid: dict[str, list[list]] = defaultdict(list)
    if "clickstream" in encoders:
        from schemas.clickstream import ClickstreamEvent

        click_path = Path("data/synthetic/clickstream.jsonl")
        raw_by_pid_sid: dict[str, dict[str, list]] = defaultdict(
            lambda: defaultdict(list)
        )
        if click_path.exists():
            for line in click_path.read_text().strip().splitlines():
                r = json.loads(line)
                pid = r.get("participant_id", "")
                customer_id = r.get("customer_id", "")
                if not pid or customer_id == "anonymous":
                    continue
                if "event_type" not in r:  # skip interleaved SessionSummary rows
                    continue
                if pid in pid_to_idx:
                    raw_by_pid_sid[pid][r.get("session_id", "")].append(
                        ClickstreamEvent(**r)
                    )
        for pid, sessions in raw_by_pid_sid.items():
            click_sessions_by_pid[pid] = [
                sorted(s, key=lambda e: e.event_ts) for s in sessions.values()
            ]

    # ── Campaign events grouped by participant ────────────────────────────────
    # Mirrors encoders/campaign/train.py load_campaign_events(): chronologically
    # sorted by sent_ts so encode_sequence's most-recent-last truncation keeps
    # the correct tail.
    campaign_events_by_pid: dict[str, list] = defaultdict(list)
    if "campaign" in encoders:
        from schemas.campaign import CampaignEvent

        camp_path = Path("data/synthetic/campaigns.jsonl")
        if camp_path.exists():
            for line in camp_path.read_text().strip().splitlines():
                r = json.loads(line)
                pid = r.get("participant_id", "")
                if pid in pid_to_idx:
                    campaign_events_by_pid[pid].append(CampaignEvent(**r))
            for pid in campaign_events_by_pid:
                campaign_events_by_pid[pid].sort(key=lambda e: e.sent_ts)

    # ── Encode per participant ─────────────────────────────────────────────────
    for i, psycho in enumerate(psychographics):
        pid = psycho["participant_id"]
        embeddings["participant_ids"].append(pid)
        embeddings["labels"][i] = PERSONA_TO_IDX[psycho["persona_id"]]

        with torch.no_grad():
            # Trace: mean-pool embeddings across all trials for this participant
            if "trace" in encoders and trace_vocab is not None:
                trial_embs = []
                for trial in trials_by_pid.get(pid, []):
                    trial_events = events_by_pid.get(pid, [])
                    tid_events = [
                        e for e in trial_events if e.trial_id == trial.trial_id
                    ]
                    if not tid_events:
                        continue
                    from encoders.trace.tokeniser import tokenise_trial

                    tokens, mask = tokenise_trial(tid_events, trial, trace_vocab)
                    tokens_b = tokens.unsqueeze(0).to(device)
                    mask_b = mask.unsqueeze(0).to(device) if mask is not None else None
                    emb = encoders["trace"](tokens_b, mask_b)
                    trial_embs.append(emb.squeeze(0))
                if trial_embs:
                    embeddings["trace"][i] = torch.stack(trial_embs).mean(0)

            # Transaction: encode sequence, get participant embedding
            if "transaction" in encoders:
                raw_txs = tx_by_pid.get(pid, [])
                if raw_txs:
                    from encoders.transaction.features import (
                        sort_transactions_most_recent_first,
                    )

                    tx_records = [
                        TransactionRecord(
                            participant_id=r.get("participant_id", ""),
                            persona_id=r.get("persona_id", ""),
                            transaction_id=r.get("transaction_id", ""),
                            days_before_session=r.get("days_before_session", 0),
                            category=r.get("category", ""),
                            product_id=r.get("product_id", ""),
                            brand_tier=r.get("brand_tier", "value"),
                            price_paid_normalised=r.get("price_paid_normalised", 0.0),
                            quantity=r.get("quantity", 1),
                            channel=r.get("channel", "online"),
                            purchase_type=r.get("purchase_type", "planned"),
                            on_promotion=r.get("on_promotion", False),
                            loyalty_card=r.get("loyalty_card"),
                        )
                        for r in raw_txs
                    ]
                    sorted_tx = sort_transactions_most_recent_first(tx_records)
                    tx_enc = encoders["transaction"]
                    assert isinstance(tx_enc, TxEncoder)
                    token_seq = tx_enc.vocab.encode_sequence(sorted_tx)
                    token_seq_b = token_seq.unsqueeze(0).to(device)
                    lengths = torch.tensor([len(sorted_tx)], device=device)  # type: ignore[reportPrivateImportUsage]
                    embeddings["transaction"][i] = tx_enc(token_seq_b, lengths).squeeze(
                        0
                    )

            # Text: sentence-transformer encode then project (conditional — the
            # 5-modality dry run excludes text because narratives.jsonl is empty)
            if "text" in encoders:
                narrative = modality_data.get("narratives", {}).get(pid)
                if narrative:
                    text = narrative.get("text", "")
                    if text:
                        txt_enc = encoders["text"]
                        assert isinstance(txt_enc, TxtEncoder)
                        sent_emb = txt_enc.encode_texts([text]).to(device)
                        embeddings["text"][i] = txt_enc(sent_emb).squeeze(0)

            # Psychographic: feature vector → MLP
            if "psychographic" in encoders:
                psycho_vec = PsychographicVector(
                    **{
                        k: v
                        for k, v in psycho.items()
                        if k in PsychographicVector.__dataclass_fields__
                    }
                )
                raw_vec = to_feature_vector(psycho_vec).to(device)
                embeddings["psychographic"][i] = encoders["psychographic"](
                    raw_vec.unsqueeze(0)
                ).squeeze(0)

            # Clickstream: per-session encode_session → customer-level forward
            # Mirrors encoders/clickstream/train.py (ClickstreamCustomerDataset
            # + _encode_customer). Sessions truncated to most-recent
            # MAX_SESSIONS, events to MAX_EVENTS_PER_SESSION, oldest-first so
            # the model sees chronological order.
            if "clickstream" in encoders:
                from encoders.clickstream.features import (
                    MAX_EVENTS_PER_SESSION,
                    MAX_SESSIONS,
                    TOKEN_DIM,
                )
                from encoders.clickstream.model import ClickstreamEncoder

                click_enc = encoders["clickstream"]
                assert isinstance(click_enc, ClickstreamEncoder)
                sessions = click_sessions_by_pid.get(pid, [])
                if sessions:
                    sessions_sorted = sorted(
                        sessions, key=lambda s: s[0].event_ts if s else ""
                    )
                    sessions_sorted = sessions_sorted[-MAX_SESSIONS:]
                    full_tokens = [
                        click_enc.vocab.encode_session(sess)[
                            :MAX_EVENTS_PER_SESSION
                        ].detach()
                        for sess in sessions_sorted
                    ]
                    n_sessions = len(full_tokens)
                    if n_sessions > 0:
                        max_t = max(t.size(0) for t in full_tokens)
                        tokens = torch.zeros(
                            1, n_sessions, max_t, TOKEN_DIM, device=device
                        )
                        lens = torch.ones(n_sessions, dtype=torch.long, device=device)
                        for j, tok in enumerate(full_tokens):
                            tlen = min(tok.size(0), max_t)
                            tokens[0, j, :tlen] = tok[:tlen]
                            lens[j] = max(tlen, 1)
                        mask = torch.ones(
                            1, n_sessions, dtype=torch.bool, device=device
                        )
                        # flatten → encode_session → reshape → forward (mean-pool)
                        flat_tokens = tokens.view(n_sessions, max_t, TOKEN_DIM)
                        session_embs = click_enc.encode_session(
                            flat_tokens, lens
                        )  # (N, gru_hidden)
                        session_embs = session_embs.unsqueeze(0)  # (1, N, gru_hidden)
                        embeddings["clickstream"][i] = click_enc(
                            session_embs, mask
                        ).squeeze(0)

            # Campaign: encode_sequence (10-dim) → zero-pad to TOKEN_DIM=11 →
            # prepend CLS row → forward. Mirrors encoders/campaign/train.py
            # _build_token_sequence(). The 11th dim is reserved-but-unpopulated
            # by the tokeniser (documented contract deviation); the model's
            # input_proj expects 11 inputs, so zero-padding is mandatory.
            if "campaign" in encoders:
                from encoders.campaign.features import TOKEN_DIM
                from encoders.campaign.model import CampaignEncoder

                camp_enc = encoders["campaign"]
                assert isinstance(camp_enc, CampaignEncoder)
                events = campaign_events_by_pid.get(pid, [])
                if events:
                    raw = camp_enc.vocab.encode_sequence(events).detach()  # (S, 10)
                    seq_len = raw.size(0)
                    if raw.size(1) < TOKEN_DIM:
                        pad = torch.zeros(
                            seq_len, TOKEN_DIM - raw.size(1), dtype=raw.dtype
                        )
                        raw = torch.cat([raw, pad], dim=1)  # (S, 11)
                    cls_row = torch.zeros(1, TOKEN_DIM, dtype=raw.dtype, device=device)
                    tokens = torch.cat([cls_row, raw.to(device)], dim=0).unsqueeze(
                        0
                    )  # (1, S+1, 11)
                    mask = torch.ones(
                        1, tokens.size(1), dtype=torch.bool, device=device
                    )
                    embeddings["campaign"][i] = camp_enc(tokens, mask).squeeze(0)

        if (i + 1) % 100 == 0:
            print(f"  Embedded {i + 1}/{n_participants} participants")

    return embeddings


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


def build_cache(
    encoders: dict[str, nn.Module],
    cache_path: Path,
    device: str = "cpu",
) -> dict:
    """Build or load embedding cache.

    Parameters
    ----------
    encoders : dict[str, nn.Module]
        Frozen encoder models.
    cache_path : Path
        Path to cache file.
    device : str
        Target device for computation.

    Returns
    -------
    dict
        Cached embeddings — one ``[N, EMBEDDING_DIM]`` tensor per loaded
        modality, plus ``labels`` and ``participant_ids``.
    """
    # Expected modality keys in the cache: exactly the loaded encoders.
    expected_modalities = set(encoders.keys())

    # Check if cache is valid: exists, newer than all loaded encoder
    # checkpoints, AND contains exactly the expected modality set.
    cache_valid = cache_path.exists()
    if cache_valid:
        cache_mtime = cache_path.stat().st_mtime
        for modality, checkpoint_path in CHECKPOINT_PATHS.items():
            if modality == "fusion" or modality not in encoders:
                continue
            if checkpoint_path.stat().st_mtime > cache_mtime:
                cache_valid = False
                break
        if cache_valid:
            # Validate the cache modality set matches the loaded encoders.
            # A stale 4-modality cache must not be reused for a 5-modality run.
            # Note: weights_only=False required for cache (dict with tensors +
            # list). Safe because cache is created by our own code.
            cached = torch.load(cache_path)
            cached_modalities = {
                k for k in cached if k not in ("labels", "participant_ids")
            }
            if cached_modalities != expected_modalities:
                print(
                    f"Cache modality set {sorted(cached_modalities)} != expected "
                    f"{sorted(expected_modalities)}; rebuilding."
                )
                cache_valid = False
            else:
                print(f"Loading cached embeddings from {cache_path}")
                return cached

    # Build cache
    print("Generating embeddings from encoder checkpoints...")

    # Load psychographics to get participant ordering
    psychographics = load_psychographics()
    participant_ids = [p["participant_id"] for p in psychographics]

    # Narratives loaded here; traces and transactions loaded inside generate_embeddings
    modality_data = {
        "narratives": load_modality_data("narratives", participant_ids),
    }

    # Generate embeddings
    embeddings = generate_embeddings(encoders, psychographics, modality_data, device)

    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, cache_path)
    print(f"Saved embeddings cache to {cache_path}")

    return embeddings


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def nt_xent_fusion(
    emb_v1: torch.Tensor,
    emb_v2: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """NT-Xent (SimCLR-style) for a matched pair of fused embedding views.

    emb_v1[i] and emb_v2[i] are two modality-dropout augmented views of the same
    participant i. All other cross-participant pairs are negatives.

    Parameters
    ----------
    emb_v1, emb_v2 : Tensor, shape (B, D)
        L2-normalised CDT embeddings from two dropout-augmented forward passes.
    temperature : float

    Returns
    -------
    Scalar loss tensor.
    """
    B = emb_v1.size(0)
    if B < 2:
        return torch.tensor(0.0, device=emb_v1.device, requires_grad=True)

    embs = F.normalize(torch.cat([emb_v1, emb_v2], dim=0), dim=1)  # (2B, D)
    sim = torch.mm(embs, embs.t()) / temperature  # (2B, 2B)

    labels = torch.cat(
        [
            torch.arange(B, 2 * B, device=emb_v1.device),
            torch.arange(0, B, device=emb_v1.device),
        ]
    )
    mask = torch.eye(2 * B, dtype=torch.bool, device=emb_v1.device)
    sim = sim.masked_fill(mask, float("-inf"))
    return F.cross_entropy(sim, labels)


def _apply_modality_dropout(
    batch_embs: torch.Tensor, p_dropout: float, device: str
) -> torch.Tensor:
    """Apply independent per-modality dropout to a [B, M, 128] embedding batch."""
    B, M, _ = batch_embs.shape
    masks = [torch.rand(B, 1, device=device) >= p_dropout for _ in range(M)]
    result = batch_embs.clone()
    for i in range(M):
        result[:, i] = result[:, i] * masks[i].float()
    return result


def train(
    *,
    modalities: list[str] | None = None,
    cache_path: Optional[Path] = None,
    n_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    p_dropout: float = 0.2,
    device: str = "cpu",
    log_mlflow: bool = True,
    phase: str = "2",
    lambda_contrastive: float = 0.5,
    nt_xent_temperature: float = 0.07,
) -> LateFusionMetaLearner:
    """Train the fusion meta-learner with NT-Xent + CE multi-task objective.

    Two modality-dropout augmented views of each participant's fused embedding
    are used as NT-Xent positive pairs. Other participants in the batch are
    negatives. A CE auxiliary head retains archetype separability (Tier 1 gate).

    Parameters
    ----------
    modalities : list[str] | None
        Modality names to fuse. If None, loads every modality in
        ``CHECKPOINT_PATHS`` except fusion. Pass a subset (e.g. excluding
        ``"text"`` when narratives are unavailable) to run a reduced modality
        set; the meta-learner is sized to match.
    cache_path : Path | None
        Path to embedding cache. Default: models/fusion_embeddings_cache.pt.
    n_epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size.
    lr : float
        Learning rate.
    p_dropout : float
        Per-modality dropout probability for augmentation views (default 0.2).
    device : str
        Target device ("cpu" or "cuda").
    log_mlflow : bool
        Whether to log to MLflow.
    phase : str
        Meta-learner phase ("1" or "2").
    lambda_contrastive : float
        Weight for NT-Xent loss. Total = CE + lambda * NT-Xent.
    nt_xent_temperature : float
        NT-Xent temperature (default 0.07).

    Returns
    -------
    LateFusionMetaLearner
        Trained meta-learner model.
    """
    if cache_path is None:
        cache_path = Path("models/fusion_embeddings_cache.pt")

    # Load encoders
    print("Loading encoder checkpoints...")
    encoders = load_encoders(modalities=modalities, device=device)
    print(f"Encoders loaded and frozen: {sorted(encoders.keys())}")

    # Build or load embedding cache
    embeddings = build_cache(encoders, cache_path, device)

    # Split participants
    participant_ids: list[str] = embeddings["participant_ids"]  # type: ignore[assignment]
    train_ids, val_ids = split_by_participant(participant_ids)

    # Create train/val indices
    participant_to_idx = {pid: i for i, pid in enumerate(participant_ids)}
    train_indices = torch.tensor([participant_to_idx[pid] for pid in train_ids])  # type: ignore[reportPrivateImportUsage]
    val_indices = torch.tensor([participant_to_idx[pid] for pid in val_ids])  # type: ignore[reportPrivateImportUsage]

    # Extract embeddings and labels.
    # Modalities = every cache key that is an actual modality tensor. Both
    # "labels" and "participant_ids" (a list, not a tensor) MUST be excluded or
    # torch.stack would corrupt the concat with a non-tensor / wrong-dim entry.
    _MODALITIES = [k for k in embeddings if k not in ("labels", "participant_ids")]
    n_modalities = len(_MODALITIES)
    # Loud guard: the modality count must match the loaded encoders (catches a
    # 4-modality regression or a stale cache) and the stacked tensor must be
    # [N, n_modalities, 128] (catches a participant_ids leak into the concat).
    assert n_modalities == len(encoders), (
        f"n_modalities={n_modalities} != len(encoders)={len(encoders)}; "
        f"modalities={_MODALITIES}, encoders={sorted(encoders.keys())}"
    )
    train_embs = {mod: embeddings[mod][train_indices] for mod in _MODALITIES}
    train_labels = embeddings["labels"][train_indices]

    val_embs = {mod: embeddings[mod][val_indices] for mod in _MODALITIES}
    val_labels = embeddings["labels"][val_indices]

    def make_dataset(embs_dict, labels):
        all_embs = torch.stack(
            [embs_dict[mod] for mod in _MODALITIES], dim=1
        )  # [N, M, 128]
        assert all_embs.shape[1:] == (n_modalities, EMBEDDING_DIM), (
            f"Expected stacked shape [N, {n_modalities}, {EMBEDDING_DIM}], "
            f"got {tuple(all_embs.shape)} — a non-modality key may have leaked "
            f"into the concat."
        )
        return TensorDataset(all_embs, labels)

    train_ds = make_dataset(train_embs, train_labels)
    val_ds = make_dataset(val_embs, val_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Fusing {n_modalities} modalities: {_MODALITIES}")

    # Initialize model with the correct modality count
    model = LateFusionMetaLearner(phase=phase, n_modalities=n_modalities).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 10

    for epoch in range(n_epochs):
        model.train()
        epoch_ce = 0.0
        epoch_nt = 0.0
        n_batches = 0

        for batch_embs, batch_labels in train_loader:
            batch_embs = batch_embs.to(device)
            batch_labels = batch_labels.to(device)

            # View 1: modality-dropout augmented fusion input
            v1 = _apply_modality_dropout(batch_embs, p_dropout, device)
            norm_v1 = [F.normalize(v1[:, i], p=2, dim=-1) for i in range(n_modalities)]
            fusion_v1 = torch.cat(norm_v1, dim=-1)  # [B, M*128]

            # View 2: independent modality-dropout augmented fusion input
            v2 = _apply_modality_dropout(batch_embs, p_dropout, device)
            norm_v2 = [F.normalize(v2[:, i], p=2, dim=-1) for i in range(n_modalities)]
            fusion_v2 = torch.cat(norm_v2, dim=-1)

            # CE loss on view 1 (archetype auxiliary head)
            logits, emb_v1 = model.forward_with_embedding(fusion_v1)
            ce_loss = criterion(logits, batch_labels)

            # NT-Xent: same participant across two dropout views
            _, emb_v2 = model.forward_with_embedding(fusion_v2)
            nt_loss = nt_xent_fusion(emb_v1, emb_v2, nt_xent_temperature)

            loss = ce_loss + lambda_contrastive * nt_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_ce += ce_loss.item()
            epoch_nt += nt_loss.item()
            n_batches += 1

        avg_ce = epoch_ce / n_batches
        avg_nt = epoch_nt / n_batches

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0.0

        with torch.no_grad():
            for batch_embs, batch_labels in val_loader:
                batch_embs = batch_embs.to(device)
                batch_labels = batch_labels.to(device)

                norm_embs = [
                    F.normalize(batch_embs[:, i], p=2, dim=-1)
                    for i in range(n_modalities)
                ]
                fusion_input = torch.cat(norm_embs, dim=-1)

                logits, _ = model.forward_with_embedding(fusion_input)
                val_loss_sum += criterion(logits, batch_labels).item()

                val_correct += (logits.argmax(dim=-1) == batch_labels).sum().item()
                val_total += batch_labels.shape[0]

        avg_val_loss = val_loss_sum / len(val_loader)
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch + 1}/{n_epochs}: "
            f"ce={avg_ce:.4f}  nt={avg_nt:.4f}  "
            f"val_loss={avg_val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        if log_mlflow:
            mlflow.log_metrics(
                {
                    "train_ce_loss": avg_ce,
                    "train_nt_loss": avg_nt,
                    "val_loss": avg_val_loss,
                    "val_acc": val_acc,
                },
                step=epoch,
            )

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            checkpoint_path = CHECKPOINT_PATHS["fusion"]
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  → New best model saved to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(
                    f"  → Early stopping at epoch {epoch + 1} (patience={max_patience})"
                )
                break

    print(f"\nTraining complete. Best val_acc: {best_val_acc:.4f}")
    return model


def _select_modalities() -> list[str]:
    """Determine the modality set for this run.

    Defaults to every modality in ``CHECKPOINT_PATHS`` except fusion. Drops
    ``"text"`` when ``data/synthetic/narratives.jsonl`` is missing or empty
    (the 5-modality dry run path — full 6-modality run follows once narratives
    are regenerated).
    """
    modalities = [m for m in CHECKPOINT_PATHS if m != "fusion"]
    narratives_path = Path("data/synthetic/narratives.jsonl")
    if not narratives_path.exists() or not narratives_path.read_text().strip():
        if "text" in modalities:
            modalities.remove("text")
            print(
                "narratives.jsonl is empty/missing — dropping 'text' modality "
                f"(running {len(modalities)} modalities: {modalities})"
            )
    return modalities


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(override=True)
    train(modalities=_select_modalities())
