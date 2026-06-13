"""
Campaign encoder training loop.

Per-customer campaign interaction sequences, grouped by participant_id.
Trains with supervised cross-entropy (7-class archetype) + NT-Xent
contrastive loss (individual identity via per-epoch split-view of each
customer's campaign sequence). Mirrors the structure of
``encoders/trace/train.py`` but operates on one campaign Transformer per
customer rather than per-trial.

Contract deviation (documented):
    ``features.py`` produces 10-dim event tokens (5 campaign_type_embed +
    1 discount + 4 funnel flags), but ``TOKEN_DIM = 11`` and
    ``CampaignEncoder.input_proj`` expects 11 input features. The 11th
    dimension is reserved but unpopulated by the tokeniser, so we zero-pad
    the 10-dim ``encode_sequence`` output to ``TOKEN_DIM`` before feeding
    the model. This is the only way to honour both immutable contracts
    (model.py and features.py) simultaneously without modifying either.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import mlflow
import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from schemas import CHECKPOINT_PATHS, EMBEDDING_DIM, PERSONA_TO_IDX
from schemas.campaign import CampaignEvent

from encoders.campaign.features import (
    MAX_EVENTS,
    TOKEN_DIM,
    CampaignVocabulary,
)
from encoders.campaign.model import CampaignEncoder

logger = logging.getLogger(__name__)

# Pinned hyperparameters (match trace encoder exactly)
DEFAULT_BATCH_SIZE: int = 256
DEFAULT_LR: float = 1e-3
DEFAULT_EPOCHS: int = 50
DEFAULT_WEIGHT_DECAY: float = 1e-4
DEFAULT_LAMBDA_CONTRASTIVE: float = 0.5
DEFAULT_TEMPERATURE: float = 0.07
DEFAULT_SEED: int = 42
TRAIN_FRACTION: float = 0.8
N_CLASSES: int = 7
# Dimension actually emitted by features.encode_sequence (see module docstring).
FEATURE_EMIT_DIM: int = 10

DATA_DIR = Path("data/synthetic")
CAMPAIGNS_PATH = DATA_DIR / "campaigns.jsonl"
PSYCHOGRAPHICS_PATH = DATA_DIR / "psychographics.jsonl"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class CampaignDataset(Dataset):
    """Per-customer campaign sequences with persona labels.

    Each item: (tokens_with_cls, mask, persona_label, participant_id).
    ``tokens_with_cls`` has shape (S+1, TOKEN_DIM) — a zero CLS placeholder
    row prepended at index 0, mirroring the trace tokeniser convention.
    """

    def __init__(
        self,
        sequences: list[Tensor],
        persona_labels: list[int],
        participant_ids: list[str],
    ) -> None:
        self.sequences = sequences
        self.persona_labels = persona_labels
        self.participant_ids = participant_ids

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[Tensor, int, str]:
        return (
            self.sequences[idx],
            self.persona_labels[idx],
            self.participant_ids[idx],
        )


# ---------------------------------------------------------------------------
# StratifiedSampler — guarantees >=2 samples per persona per batch
# (required for NT-Xent split-view to have valid positive pairs)
# ---------------------------------------------------------------------------


class StratifiedSampler:
    """Batch sampler ensuring >=2 samples per persona per batch.

    Identical algorithm to the trace encoder's sampler: samples by persona
    first, oversampling with replacement when a persona is under-saturated,
    and drops tail batches that would violate the min-per-persona constraint.
    """

    def __init__(
        self,
        persona_labels: list[int],
        batch_size: int,
        seed: int = DEFAULT_SEED,
        min_per_persona: int = 2,
    ) -> None:
        self.persona_labels = persona_labels
        self.batch_size = batch_size
        self.min_per_persona = min_per_persona
        self.base_seed = seed
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

        self.label_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, lbl in enumerate(persona_labels):
            self.label_to_indices[lbl].append(idx)
        self.unique_labels = sorted(self.label_to_indices.keys())

        self._batches: list[list[int]] = []
        self._build_batches()

    def _build_batches(self) -> None:
        self._batches = []
        shuffled: dict[int, list[int]] = {}
        for lbl in self.unique_labels:
            indices = list(self.label_to_indices[lbl])
            perm = torch.randperm(len(indices), generator=self.rng)
            shuffled[lbl] = [indices[p.item()] for p in perm]

        n_labels = len(self.unique_labels)
        if n_labels == 0:
            return

        per_persona = max(self.min_per_persona, self.batch_size // n_labels)
        max_iter = max(len(v) for v in shuffled.values())
        n_batches = max_iter // per_persona
        if n_batches == 0 and max_iter > 0:
            n_batches = 1

        for batch_idx in range(n_batches):
            batch: list[int] = []
            for lbl in self.unique_labels:
                indices = shuffled[lbl]
                start = batch_idx * per_persona
                end = start + per_persona
                if end <= len(indices):
                    batch.extend(indices[start:end])
                elif start < len(indices):
                    available = indices[start:]
                    batch.extend(available)
                    needed = per_persona - len(available)
                    for _ in range(needed):
                        rand_idx = torch.randint(
                            len(indices), (1,), generator=self.rng
                        ).item()
                        batch.append(indices[rand_idx])
                else:
                    for _ in range(per_persona):
                        rand_idx = torch.randint(
                            len(indices), (1,), generator=self.rng
                        ).item()
                        batch.append(indices[rand_idx])

            if len(batch) >= self.min_per_persona * n_labels:
                self._batches.append(batch)

    def set_epoch(self, epoch: int) -> None:
        self.rng.manual_seed(self.base_seed + epoch)
        self._build_batches()

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def cross_entropy_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Supervised cross-entropy over the 7-class archetype head."""
    return F.cross_entropy(logits, labels)


def nt_xent_views(
    emb_v1: Tensor,
    emb_v2: Tensor,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Tensor:
    """NT-Xent for position-matched split-view pairs.

    ``emb_v1[i]`` and ``emb_v2[i]`` are two half-sequence mean-pools of the
    same customer's campaign history (positive pair). All other
    cross-customer pairs are negatives. Identical to the trace encoder's
    split-view contrastive objective.
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


# ---------------------------------------------------------------------------
# Data loading and tokenisation
# ---------------------------------------------------------------------------


def _build_token_sequence(
    events: list[CampaignEvent],
    vocab: CampaignVocabulary,
) -> Tensor:
    """Tokenise one customer's campaign history into (S+1, TOKEN_DIM).

    Steps:
      1. ``vocab.encode_sequence`` -> (S, 10) most-recent-last, truncated to
         MAX_EVENTS=50.
      2. Zero-pad to TOKEN_DIM=11 (see module docstring for the contract
         deviation).
      3. Prepend a zero CLS placeholder row -> (S+1, 11). The model's
         learned positional encoding at index 0 turns this into the CLS
         representation (same convention as the trace tokeniser).
      4. ``.detach()`` so the token tensor is a fixed feature input, not a
         graph-intermediate of ``vocab.campaign_type_embed``. The vocabulary
         is treated as a static featurizer; the encoder's ``input_proj`` is
         where representation learning happens. Detaching is also required
         because sequence tensors are built once and reused across batches
         (StratifiedSampler oversampling), and a graph-linked tensor cannot
         survive a second ``backward()``.
    """
    raw = vocab.encode_sequence(events).detach()  # (S, 10)
    seq_len = raw.size(0)
    if raw.size(1) < TOKEN_DIM:
        pad = torch.zeros(seq_len, TOKEN_DIM - raw.size(1), dtype=raw.dtype)
        raw = torch.cat([raw, pad], dim=1)  # (S, TOKEN_DIM)
    cls_row = torch.zeros(1, TOKEN_DIM, dtype=raw.dtype)
    return torch.cat([cls_row, raw], dim=0)  # (S+1, TOKEN_DIM)


def load_persona_labels(path: Path = PSYCHOGRAPHICS_PATH) -> dict[str, int]:
    """participant_id -> persona class index (PERSONA_TO_IDX).

    ``psychographics.jsonl`` carries one row per participant per month; we
    deduplicate on participant_id, keeping the persona_id from the first row
    (it is stable across months for the same participant).
    """
    pid_to_label: dict[str, int] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = rec["participant_id"]
            if pid in pid_to_label:
                continue
            persona = rec["persona_id"]
            if persona not in PERSONA_TO_IDX:
                logger.warning(
                    "Unknown persona_id %r for participant %r — skipping",
                    persona,
                    pid,
                )
                continue
            pid_to_label[pid] = PERSONA_TO_IDX[persona]
    return pid_to_label


def load_campaign_events(
    path: Path = CAMPAIGNS_PATH,
) -> dict[str, list[CampaignEvent]]:
    """Group all campaign events by participant_id, chronologically sorted."""
    by_pid: dict[str, list[CampaignEvent]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ev = CampaignEvent(**json.loads(line))
            by_pid[ev.participant_id].append(ev)
    # Chronological order by sent_ts so encode_sequence's most-recent-last
    # truncation keeps the right tail.
    for pid in by_pid:
        by_pid[pid].sort(key=lambda e: e.sent_ts)
    return by_pid


def build_dataset(
    events_by_pid: dict[str, list[CampaignEvent]],
    pid_to_label: dict[str, int],
    vocab: CampaignVocabulary,
    participant_ids: list[str],
) -> CampaignDataset:
    """Tokenise campaign histories for a set of participants."""
    sequences: list[Tensor] = []
    labels: list[int] = []
    pids: list[str] = []
    for pid in participant_ids:
        events = events_by_pid.get(pid, [])
        if not events:
            continue
        sequences.append(_build_token_sequence(events, vocab))
        labels.append(pid_to_label[pid])
        pids.append(pid)
    return CampaignDataset(sequences, labels, pids)


def split_participants(
    participant_ids: list[str],
    train_fraction: float = TRAIN_FRACTION,
    seed: int = DEFAULT_SEED,
) -> tuple[list[str], list[str]]:
    """Deterministic 80/20 split of participant ids."""
    rng = np.random.default_rng(seed)
    shuffled = sorted(participant_ids)
    rng.shuffle(shuffled)
    split_idx = int(train_fraction * len(shuffled))
    return shuffled[:split_idx], shuffled[split_idx:]


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def collate_fn(
    batch: list[tuple[Tensor, int, str]],
) -> tuple[Tensor, Tensor, Tensor, list[str]]:
    """Pad variable-length sequences to the batch max length.

    Returns (tokens, mask, labels, participant_ids) where tokens has shape
    (B, max_S, TOKEN_DIM) and mask True marks real (non-padding) positions.
    """
    seqs = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    pids = [item[2] for item in batch]

    max_len = max(s.size(0) for s in seqs)
    B = len(seqs)
    tokens = torch.zeros(B, max_len, TOKEN_DIM, dtype=seqs[0].dtype)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, s in enumerate(seqs):
        L = s.size(0)
        tokens[i, :L] = s
        mask[i, :L] = True
    return tokens, mask, labels, pids


# ---------------------------------------------------------------------------
# Strategy recovery probe
# ---------------------------------------------------------------------------


def strategy_recovery(
    encoder: CampaignEncoder,
    dataset: CampaignDataset,
    device: str = "cpu",
) -> float:
    """Logistic-regression archetype recovery on held-out val embeddings.

    Trains a sklearn LogisticRegression on the encoder embeddings and
    reports accuracy — the "strategy recovery" metric used across all
    encoder probes.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    encoder.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)
    all_embs: list[np.ndarray] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for tokens, mask, labels, _ in loader:
            tokens = tokens.to(device)
            mask = mask.to(device)
            emb = encoder(tokens, mask)
            all_embs.append(emb.cpu().numpy())
            all_labels.extend(labels.tolist())
    X = np.vstack(all_embs)
    y = np.array(all_labels)

    n_classes = int(y.max()) + 1
    # Use up to 5 folds, clamped to the number of samples per class so tiny
    # validation sets don't blow up sklearn's splitter.
    class_counts = np.bincount(y, minlength=n_classes)
    min_class_count = int(class_counts.min())
    if len(y) < 2 or min_class_count < 2:
        # Not enough samples for cross-validation; return neutral score.
        logger.warning(
            "strategy_recovery: too few val samples (%d, min class %d) for CV",
            len(y),
            min_class_count,
        )
        return 0.0
    n_splits = min(5, min_class_count)
    clf = LogisticRegression(max_iter=2000, random_state=DEFAULT_SEED)
    scores = cross_val_score(
        clf,
        X,
        y,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=DEFAULT_SEED),
        scoring="accuracy",
    )
    return float(scores.mean())


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    campaigns_path: Path = CAMPAIGNS_PATH,
    psychographics_path: Path = PSYCHOGRAPHICS_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    n_epochs: int = DEFAULT_EPOCHS,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    seed: int = DEFAULT_SEED,
    device: str = "cpu",
    save_path: Path | None = None,
    lambda_contrastive: float = DEFAULT_LAMBDA_CONTRASTIVE,
    nt_xent_temperature: float = DEFAULT_TEMPERATURE,
) -> CampaignEncoder:
    """Train the campaign encoder with CE + NT-Xent split-view.

    Saves the backbone (everything except the classifier head) to
    ``save_path``, defaulting to ``CHECKPOINT_PATHS["campaign"]``. The
    checkpoint is loadable via
    ``CampaignEncoder().load_state_dict(torch.load(path), strict=False)``.
    """
    torch.manual_seed(seed)

    # Load data
    pid_to_label = load_persona_labels(psychographics_path)
    events_by_pid = load_campaign_events(campaigns_path)
    all_pids = [pid for pid in pid_to_label if pid in events_by_pid]
    logger.info(
        "Participants with campaigns: %d, persona classes: %d",
        len(all_pids),
        len(set(pid_to_label[p] for p in all_pids)),
    )

    train_pids, val_pids = split_participants(all_pids, seed=seed)
    logger.info(
        "Train participants: %d, Val participants: %d", len(train_pids), len(val_pids)
    )

    # One shared vocabulary so train/val embeddings are consistent
    vocab = CampaignVocabulary()
    train_dataset = build_dataset(events_by_pid, pid_to_label, vocab, train_pids)
    val_dataset = build_dataset(events_by_pid, pid_to_label, vocab, val_pids)
    logger.info(
        "Train samples: %d, Val samples: %d", len(train_dataset), len(val_dataset)
    )

    # Build model (constants are module-level in model.py — do not override)
    encoder = CampaignEncoder(vocab=vocab).to(device)
    optimizer = torch.optim.AdamW(
        encoder.parameters(), lr=lr, weight_decay=weight_decay
    )

    # Stratified sampler ensures >=2 per persona for NT-Xent positive pairs
    stratified_sampler = StratifiedSampler(
        persona_labels=train_dataset.persona_labels,
        batch_size=batch_size,
        seed=seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=stratified_sampler,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # MLflow tracking
    mlflow.set_tag("modality", "campaign")
    mlflow.log_params(
        {
            "lr": lr,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "weight_decay": weight_decay,
            "n_classes": N_CLASSES,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "objective": "ce+nt_xent_split_view",
            "lambda_contrastive": lambda_contrastive,
            "nt_xent_temperature": nt_xent_temperature,
            "max_events": MAX_EVENTS,
            "token_dim": TOKEN_DIM,
            "embedding_dim": EMBEDDING_DIM,
        }
    )

    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    for epoch in range(n_epochs):
        stratified_sampler.set_epoch(epoch)

        # --- Training ---
        encoder.train()
        epoch_cls_loss = 0.0
        epoch_nt_loss = 0.0
        n_batches = 0

        for tokens, mask, labels, _participant_ids_batch in train_loader:
            tokens = tokens.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # CE on per-customer embeddings
            customer_embs, logits = encoder.forward_with_logits(tokens, mask)
            cls_loss = cross_entropy_loss(logits, labels)

            # NT-Xent split-view: each customer produces ONE embedding, so
            # we build contrastive views at the persona level — group the
            # batch's customer embeddings by persona label, then mean-pool a
            # random half into view1 and the complementary half into view2.
            # Same archetype -> positive pair; different archetype ->
            # negative. This mirrors the trace encoder's split-view (which
            # groups per-trial embeddings by persona), and shares the CE
            # forward graph so a single backward pass suffices.
            label_to_embs: dict[int, list[Tensor]] = defaultdict(list)
            for emb, lbl in zip(customer_embs, labels.tolist()):
                label_to_embs[lbl].append(emb)

            v1_list: list[Tensor] = []
            v2_list: list[Tensor] = []
            for lbl, embs in label_to_embs.items():
                if len(embs) < 2:
                    continue
                stacked = torch.stack(embs)  # (n_persona, D)
                perm = torch.randperm(stacked.size(0), device=device)
                half = stacked.size(0) // 2
                if half < 1:
                    continue
                v1_list.append(stacked[perm[:half]].mean(0))
                v2_list.append(stacked[perm[half : half * 2]].mean(0))

            if len(v1_list) >= 2:
                nt_loss = nt_xent_views(
                    torch.stack(v1_list),
                    torch.stack(v2_list),
                    nt_xent_temperature,
                )
            else:
                nt_loss = torch.tensor(0.0, device=device)

            loss = cls_loss + lambda_contrastive * nt_loss
            loss.backward()
            optimizer.step()

            epoch_cls_loss += cls_loss.item()
            epoch_nt_loss += nt_loss.item()
            n_batches += 1

        avg_train_cls = epoch_cls_loss / max(n_batches, 1)
        avg_train_nt = epoch_nt_loss / max(n_batches, 1)

        # --- Validation ---
        encoder.eval()
        val_cls_loss = 0.0
        val_correct = 0
        val_total = 0
        val_batches = 0

        with torch.no_grad():
            for tokens, mask, labels, _ in val_loader:
                tokens = tokens.to(device)
                mask = mask.to(device)
                labels = labels.to(device)
                _, logits = encoder.forward_with_logits(tokens, mask)
                loss = cross_entropy_loss(logits, labels)
                val_cls_loss += loss.item()
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)
                val_batches += 1

        avg_val_loss = val_cls_loss / max(val_batches, 1)
        val_acc = val_correct / max(val_total, 1)

        mlflow.log_metrics(
            {
                "train_cls_loss": avg_train_cls,
                "train_nt_loss": avg_train_nt,
                "val_cls_loss": avg_val_loss,
                "val_acc": val_acc,
            },
            step=epoch,
        )

        logger.info(
            "Epoch %d/%d — ce=%.4f nt=%.4f val_loss=%.4f val_acc=%.4f",
            epoch + 1,
            n_epochs,
            avg_train_cls,
            avg_train_nt,
            avg_val_loss,
            val_acc,
        )

        # Early stopping on validation loss; save backbone (no classifier)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            _save_path = (
                save_path if save_path is not None else CHECKPOINT_PATHS["campaign"]
            )
            _save_path.parent.mkdir(parents=True, exist_ok=True)
            backbone_state = {
                k: v
                for k, v in encoder.state_dict().items()
                if not k.startswith("classifier")
            }
            torch.save(backbone_state, _save_path)
            logger.info("Saved best model to %s", _save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(
                    "Early stopping at epoch %d (patience=%d)",
                    epoch + 1,
                    patience,
                )
                break

    mlflow.log_metric("best_val_loss", best_val_loss)

    # Strategy recovery probe on held-out val
    recovery = strategy_recovery(encoder, val_dataset, device=device)
    mlflow.log_metric("strategy_recovery", recovery)
    logger.info("Strategy recovery (val, 5-fold LR): %.4f", recovery)

    logger.info("Training complete. Best val loss: %.4f", best_val_loss)
    return encoder


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(override=True)
    logging.basicConfig(level=logging.INFO)
    with mlflow.start_run(run_name="campaign_encoder_v1"):
        train()
