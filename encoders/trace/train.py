"""
Trace encoder training loop.

Splits by participant_id, uses StratifiedSampler for contrastive batch
construction, trains with NT-Xent contrastive loss + auxiliary classification
head (weight=0.3), and logs everything to MLflow.
"""

from __future__ import annotations

import json
import logging
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import mlflow
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from schemas.trace import AcquisitionEvent, TrialRecord

from encoders.trace.model import TraceEncoder
from encoders.trace.tokeniser import (
    build_vocab,
    collate_batch,
    tokenise_trial,
)

logger = logging.getLogger(__name__)

# Default hyperparameters (from SPEC.md)
DEFAULT_BATCH_SIZE: int = 256
DEFAULT_LR: float = 1e-3
DEFAULT_EPOCHS: int = 50
DEFAULT_WEIGHT_DECAY: float = 1e-4
DEFAULT_TEMPERATURE: float = 0.07
DEFAULT_AUX_WEIGHT: float = 0.3
DEFAULT_SEED: int = 42
TRAIN_FRACTION: float = 0.8

DATA_DIR = Path("data/synthetic")
TRACES_PATH = DATA_DIR / "traces.jsonl"
TRIALS_PATH = DATA_DIR / "trials.jsonl"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TraceDataset(Dataset):
    """
    Holds tokenised trial sequences with metadata for contrastive training.

    Each item is a (tokens, mask, persona_label) tuple.
    """

    def __init__(
        self,
        tokens_list: list[tuple[Tensor, Tensor]],
        persona_labels: list[int],
        persona_ids: list[str],
        trial_ids: list[str],
        participant_ids: list[str],
    ) -> None:
        self.tokens_list = tokens_list
        self.persona_labels = persona_labels
        self.persona_ids = persona_ids
        self.trial_ids = trial_ids
        self.participant_ids = participant_ids

    def __len__(self) -> int:
        return len(self.tokens_list)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, int, str]:
        tokens, mask = self.tokens_list[idx]
        return tokens, mask, self.persona_labels[idx], self.persona_ids[idx]


# ---------------------------------------------------------------------------
# StratifiedSampler
# ---------------------------------------------------------------------------


class StratifiedSampler:
    """
    Batch sampler that guarantees >=2 trials per persona_id per batch.

    Samples by persona first, then by participant within persona.
    Uses drop_last=True semantics: tail batches that would violate the
    >=2 constraint are dropped.

    Parameters
    ----------
    persona_ids:
        List of persona_id strings, one per sample (same length as dataset).
    batch_size:
        Number of samples per batch.
    seed:
        Random seed for reproducibility.
    min_per_persona:
        Minimum number of samples per persona per batch (default 2).
    """

    def __init__(
        self,
        persona_ids: list[str],
        batch_size: int,
        seed: int = DEFAULT_SEED,
        min_per_persona: int = 2,
    ) -> None:
        self.persona_ids = persona_ids
        self.batch_size = batch_size
        self.min_per_persona = min_per_persona
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)
        self.epoch = 0

        # Build persona -> list of indices
        self.persona_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, pid in enumerate(persona_ids):
            self.persona_to_indices[pid].append(idx)

        self.unique_personas = sorted(self.persona_to_indices.keys())

        # Pre-compute batches
        self._batches: list[list[int]] = []
        self._build_batches()

    def _build_batches(self) -> None:
        """Build batches that satisfy the >=2 per persona constraint."""
        self._batches = []

        # Shuffle within each persona group using the generator
        shuffled_indices: dict[str, list[int]] = {}
        for persona in self.unique_personas:
            indices = list(self.persona_to_indices[persona])
            perm = torch.randperm(len(indices), generator=self.rng)
            shuffled_indices[persona] = [indices[p.item()] for p in perm]

        n_personas = len(self.unique_personas)
        if n_personas == 0:
            return

        # How many samples per persona per batch: batch_size // n_personas
        # Must be >= min_per_persona
        per_persona = max(self.min_per_persona, self.batch_size // n_personas)

        # For personas with fewer samples than per_persona, we need to
        # oversample with replacement
        max_iterations = max(len(v) for v in shuffled_indices.values())
        # Number of complete batches we can form
        n_batches = max_iterations // per_persona

        if n_batches == 0 and max_iterations > 0:
            # We have fewer samples per persona than per_persona requires.
            # Oversample with replacement to fill at least one batch.
            n_batches = 1

        for batch_idx in range(n_batches):
            batch: list[int] = []
            for persona in self.unique_personas:
                indices = shuffled_indices[persona]
                start = batch_idx * per_persona
                end = start + per_persona
                if end <= len(indices):
                    batch.extend(indices[start:end])
                elif start < len(indices):
                    # Partial: take what's available and oversample the rest
                    available = indices[start:]
                    batch.extend(available)
                    # Oversample with replacement from the full persona pool
                    needed = per_persona - len(available)
                    for _ in range(needed):
                        rand_idx = torch.randint(
                            len(indices), (1,), generator=self.rng
                        ).item()
                        batch.append(indices[rand_idx])
                else:
                    # No more natural samples; oversample entire quota
                    for _ in range(per_persona):
                        rand_idx = torch.randint(
                            len(indices), (1,), generator=self.rng
                        ).item()
                        batch.append(indices[rand_idx])

            # Verify the constraint
            if len(batch) >= self.min_per_persona * n_personas:
                self._batches.append(batch)

    def set_epoch(self, epoch: int) -> None:
        """Re-shuffle for a new epoch."""
        self.epoch = epoch
        self.rng.manual_seed(DEFAULT_SEED + epoch)
        self._build_batches()

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


# ---------------------------------------------------------------------------
# NT-Xent contrastive loss
# ---------------------------------------------------------------------------


def nt_xent_loss(
    embeddings: Tensor,
    persona_ids: list[str],
    temperature: float = DEFAULT_TEMPERATURE,
) -> Tensor:
    """
    NT-Xent (Normalised Temperature-scaled Cross-Entropy) contrastive loss.

    Positive pairs: same persona_id.  Negative pairs: different persona_id.

    Parameters
    ----------
    embeddings:
        (B, D) normalised embeddings (L2-normalised to unit sphere).
    persona_ids:
        Persona labels for each sample in the batch.
    temperature:
        Temperature scaling parameter (tau).

    Returns
    -------
    Scalar loss tensor.
    """
    if embeddings.size(0) < 2:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    # L2-normalise embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Cosine similarity matrix: (B, B)
    sim_matrix = torch.mm(embeddings, embeddings.t()) / temperature

    # Mask: 1 where same persona, 0 otherwise (excluding self)
    n = embeddings.size(0)
    device = embeddings.device
    positive_mask = torch.zeros(n, n, device=device, dtype=torch.bool)
    for i in range(n):
        for j in range(n):
            if i != j and persona_ids[i] == persona_ids[j]:
                positive_mask[i, j] = True

    # Check that each row has at least one positive
    # If not, skip those rows (they contribute 0 loss)
    has_positives = positive_mask.any(dim=1)

    if not has_positives.any():
        # No valid contrastive pairs — return zero loss
        return torch.tensor(0.0, device=device, requires_grad=True)

    # For numerical stability, subtract max per row
    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    logits = sim_matrix - logits_max.detach()

    # Log-sum-exp of all negatives + positives (denominator)
    # Mask out self-similarity (diagonal)
    self_mask = torch.eye(n, device=device, dtype=torch.bool)
    exp_logits = torch.exp(logits)
    exp_logits = exp_logits.masked_fill(self_mask, 0.0)
    log_sum_exp = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    # Log probability of positive pairs
    log_prob = logits - log_sum_exp

    # Mean log-prob over positive pairs
    # Only compute for rows that have positives
    mean_log_prob_pos = (log_prob * positive_mask.float()).sum(dim=1)
    n_positives = positive_mask.float().sum(dim=1).clamp(min=1.0)
    mean_log_prob_pos = mean_log_prob_pos / n_positives

    # Loss is negative mean log prob, only for rows with positives
    loss = -mean_log_prob_pos[has_positives].mean()

    return loss


def cross_entropy_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Supervised cross-entropy loss for persona archetype classification."""
    return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------


def load_data(
    traces_path: Path = TRACES_PATH,
    trials_path: Path = TRIALS_PATH,
) -> tuple[list[AcquisitionEvent], dict[str, TrialRecord]]:
    """Load traces and trials from JSONL files."""
    events = [
        AcquisitionEvent(**json.loads(line))
        for line in traces_path.read_text().strip().split("\n")
        if line.strip()
    ]
    trials = {
        r.trial_id: r
        for line in trials_path.read_text().strip().split("\n")
        if line.strip()
        for r in [TrialRecord(**json.loads(line))]
    }
    return events, trials


def split_by_participant(
    trials: dict[str, TrialRecord],
    train_fraction: float = TRAIN_FRACTION,
    seed: int = DEFAULT_SEED,
) -> tuple[dict[str, TrialRecord], dict[str, TrialRecord]]:
    """
    Split trials into train/val by participant_id (never by trial_id).

    Returns (train_trials, val_trials) dicts keyed by trial_id.
    """
    import numpy as np

    participant_ids = sorted(set(r.participant_id for r in trials.values()))
    rng = np.random.default_rng(seed)
    rng.shuffle(participant_ids)

    split_idx = int(train_fraction * len(participant_ids))
    train_participants = set(participant_ids[:split_idx])
    val_participants = set(participant_ids[split_idx:])

    train_trials = {
        tid: r for tid, r in trials.items() if r.participant_id in train_participants
    }
    val_trials = {
        tid: r for tid, r in trials.items() if r.participant_id in val_participants
    }

    return train_trials, val_trials


def build_dataset(
    events: list[AcquisitionEvent],
    trials: dict[str, TrialRecord],
    persona_to_label: dict[str, int] | None = None,
    vocab: dict[str, dict[str, int]] | None = None,
) -> tuple[TraceDataset, dict[str, int]]:
    """
    Tokenise all trials and build a TraceDataset.

    Returns (dataset, persona_to_label_map).
    """
    # Group events by trial_id
    events_by_trial: dict[str, list[AcquisitionEvent]] = defaultdict(list)
    for ev in events:
        events_by_trial[ev.trial_id].append(ev)

    # Sort events by event_index within each trial
    for tid in events_by_trial:
        events_by_trial[tid].sort(key=lambda e: e.event_index)

    # Build vocab from all events if not provided
    if vocab is None:
        vocab = build_vocab(
            events, cache_path=Path(tempfile.mkdtemp()) / "trace_vocab.json"
        )

    # Build persona label map if not provided
    if persona_to_label is None:
        all_personas = sorted(set(r.persona_id for r in trials.values()))
        persona_to_label = {p: i for i, p in enumerate(all_personas)}

    tokens_list: list[tuple[Tensor, Tensor]] = []
    persona_labels: list[int] = []
    persona_ids: list[str] = []
    trial_ids: list[str] = []
    participant_ids: list[str] = []

    for tid, trial in trials.items():
        trial_events = events_by_trial.get(tid, [])
        tokens, mask = tokenise_trial(trial_events, trial, vocab)
        tokens_list.append((tokens, mask))
        persona_labels.append(persona_to_label[trial.persona_id])
        persona_ids.append(trial.persona_id)
        trial_ids.append(trial.trial_id)
        participant_ids.append(trial.participant_id)

    dataset = TraceDataset(
        tokens_list, persona_labels, persona_ids, trial_ids, participant_ids
    )
    return dataset, persona_to_label


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def collate_fn(
    batch: list[tuple[Tensor, Tensor, int, str]],
) -> tuple[Tensor, Tensor, Tensor, list[str]]:
    """Collate a batch of (tokens, mask, label, persona_id) tuples."""
    tokens_masks = [(t, m) for t, m, _, _ in batch]
    labels = torch.tensor([lbl for _, _, lbl, _ in batch], dtype=torch.long)
    persona_ids = [p for _, _, _, p in batch]

    padded_tokens, padded_mask = collate_batch(tokens_masks)
    return padded_tokens, padded_mask, labels, persona_ids


def train(
    traces_path: Path = TRACES_PATH,
    trials_path: Path = TRIALS_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    n_epochs: int = DEFAULT_EPOCHS,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    seed: int = DEFAULT_SEED,
    device: str = "cpu",
    save_dir: Path = Path("models"),
) -> TraceEncoder:
    """
    Train the trace encoder with supervised cross-entropy classification.

    Saves encoder backbone weights (NOT classification head) to save_dir.

    Parameters
    ----------
    traces_path:
        Path to traces JSONL file.
    trials_path:
        Path to trials JSONL file.
    batch_size:
        Batch size for training.
    lr:
        Learning rate.
    n_epochs:
        Number of training epochs.
    weight_decay:
        AdamW weight decay.
    temperature:
        NT-Xent temperature (tau).
    aux_weight:
        Weight for auxiliary classification loss.
    seed:
        Random seed.
    device:
        Device string ("cpu" or "cuda").
    save_dir:
        Directory to save model weights.

    Returns
    -------
    Trained TraceEncoder model.
    """
    torch.manual_seed(seed)

    # Load data
    events, all_trials = load_data(traces_path, trials_path)

    # Build persona label map from ALL trials (before split) so that val
    # personas are always covered even when the split is tiny.
    all_personas = sorted(set(r.persona_id for r in all_trials.values()))
    persona_to_label = {p: i for i, p in enumerate(all_personas)}

    # Split by participant
    train_trials, val_trials = split_by_participant(all_trials, seed=seed)
    logger.info(
        "Train trials: %d, Val trials: %d",
        len(train_trials),
        len(val_trials),
    )

    # Build datasets
    train_events = [e for e in events if e.trial_id in train_trials]
    val_events = [e for e in events if e.trial_id in val_trials]

    # Build vocab from ALL events so both train and val use the same mapping
    vocab = build_vocab(events, cache_path=DATA_DIR / "trace_vocab.json")

    train_dataset, _ = build_dataset(
        train_events, train_trials, persona_to_label, vocab
    )
    val_dataset, _ = build_dataset(val_events, val_trials, persona_to_label, vocab)

    n_classes = len(persona_to_label)
    logger.info("Personas: %d classes: %s", n_classes, persona_to_label)

    # Build model
    # Determine vocab sizes from data
    n_attributes = (
        max(int(t[:, 0].max().item()) for t, _ in train_dataset.tokens_list) + 1
    )
    n_alternatives = (
        max(int(t[:, 1].max().item()) for t, _ in train_dataset.tokens_list) + 1
    )

    encoder = TraceEncoder(
        n_attributes=n_attributes,
        n_alternatives=n_alternatives,
        n_classes=n_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(
        encoder.parameters(), lr=lr, weight_decay=weight_decay
    )

    # Build stratified batch sampler for training
    stratified_sampler = StratifiedSampler(
        persona_ids=train_dataset.persona_ids,
        batch_size=batch_size,
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=stratified_sampler,
        collate_fn=collate_fn,
    )

    # Simple sequential loader for validation
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # MLflow tracking
    mlflow.set_tag("modality", "trace")
    mlflow.log_params(
        {
            "lr": lr,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "weight_decay": weight_decay,
            "n_classes": n_classes,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "objective": "supervised_cross_entropy",
        }
    )

    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    for epoch in range(n_epochs):
        # Re-shuffle sampler each epoch
        stratified_sampler.set_epoch(epoch)

        # --- Training ---
        encoder.train()
        epoch_cls_loss = 0.0
        n_batches = 0

        for tokens, mask, labels, _ in train_loader:
            tokens = tokens.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            _, logits = encoder.forward_with_logits(tokens, mask)
            loss = cross_entropy_loss(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_cls_loss += loss.item()
            n_batches += 1

        avg_train_cls = epoch_cls_loss / max(n_batches, 1)

        # --- Validation ---
        encoder.eval()
        val_cls_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for tokens, mask, labels, _ in val_loader:
                tokens = tokens.to(device)
                mask = mask.to(device)
                labels = labels.to(device)

                _, logits = encoder.forward_with_logits(tokens, mask)
                loss = cross_entropy_loss(logits, labels)

                val_cls_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_cls_loss / max(val_batches, 1)

        mlflow.log_metrics(
            {
                "train_cls_loss": avg_train_cls,
                "val_cls_loss": avg_val_loss,
            },
            step=epoch,
        )

        logger.info(
            "Epoch %d/%d — train_cls_loss=%.4f val_cls_loss=%.4f",
            epoch + 1,
            n_epochs,
            avg_train_cls,
            avg_val_loss,
        )

        # Early stopping on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best encoder backbone weights
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / "trace_encoder.pt"
            # Save only backbone (exclude classification head)
            backbone_state = {
                k: v
                for k, v in encoder.state_dict().items()
                if not k.startswith("classifier")
            }
            torch.save(backbone_state, save_path)
            logger.info("Saved best model to %s", save_path)
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
    logger.info("Training complete. Best val loss: %.4f", best_val_loss)

    return encoder


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with mlflow.start_run(run_name="trace_encoder_v1"):
        train()
