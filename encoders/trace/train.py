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

from schemas import CHECKPOINT_PATHS
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
CHOICE_SETS_PATH = DATA_DIR / "choice_sets.jsonl"
PRODUCTS_PATH = DATA_DIR / "products.jsonl"

# bead b8b: auxiliary choice-prediction loss weight (pinned, declared).
LAMBDA_CHOICE: float = 0.5


# §0.1 brand_tier ordinal — inlined here (encoders must not import generator or
# applications) to mirror generator.choice_model / applications.choice.data.
_BRAND_TIER_LEVEL: dict[str, float] = {
    "premium": 1.0,
    "mid": 0.66,
    "value": 0.33,
    "own_label": 0.0,
}


def _product_features(product: dict) -> Tensor:
    """§0.1 board encoding of a catalogue product → 8-dim float tensor."""
    feats = [
        float(product["price_normalised"]),
        _BRAND_TIER_LEVEL[product["brand_tier"]],
        float(product["quality_score"]),
        float(product["warranty_score"]),
        float(product["rating"]) / 5.0,
        float(product["features_score"]),
        1.0 if product["availability"] else 0.0,
        float(product["design_score"]),
    ]
    return torch.tensor(feats, dtype=torch.float)


def load_choice_data(
    choice_sets_path: Path = CHOICE_SETS_PATH,
    products_path: Path = PRODUCTS_PATH,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Load choice_sets.jsonl (keyed by choice_set_id) + products.jsonl (by product_id)."""
    choice_sets: dict[str, dict] = {}
    with open(choice_sets_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cs = json.loads(line)
            choice_sets[cs["choice_set_id"]] = cs

    products: dict[str, dict] = {}
    with open(products_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            product = json.loads(line)
            products[product["product_id"]] = product
    return choice_sets, products


def _trial_choice_tensors(
    trial_id: str,
    choice_sets: dict[str, dict],
    products: dict[str, dict],
) -> tuple[Tensor, Tensor] | None:
    """Build (slot_features [n_slots, 8], chosen_labels [n_slots]) for a trial.

    Returns None if the trial has no matching ChoiceSet (choice loss is skipped
    for that trial; strategy-CE / NT-Xent still apply).
    """
    cs = choice_sets.get(trial_id)
    if cs is None:
        return None
    alt_products = cs.get("alternative_products", {})
    chosen = cs.get("chosen_alternative")
    feats: list[Tensor] = []
    labels: list[float] = []
    for slot, product_id in alt_products.items():
        product = products.get(product_id)
        if product is None:
            continue
        feats.append(_product_features(product))
        labels.append(1.0 if slot == chosen else 0.0)
    if not feats:
        return None
    return torch.stack(feats), torch.tensor(labels, dtype=torch.float)



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
        choice_feats: list[Tensor | None] | None = None,
        choice_labels: list[Tensor | None] | None = None,
    ) -> None:
        self.tokens_list = tokens_list
        self.persona_labels = persona_labels
        self.persona_ids = persona_ids
        self.trial_ids = trial_ids
        self.participant_ids = participant_ids
        self.choice_feats = choice_feats or []
        self.choice_labels = choice_labels or []

    def __len__(self) -> int:
        return len(self.tokens_list)

    def __getitem__(self, idx: int):
        tokens, mask = self.tokens_list[idx]
        cf = self.choice_feats[idx] if idx < len(self.choice_feats) else None
        cl = self.choice_labels[idx] if idx < len(self.choice_labels) else None
        return (
            tokens,
            mask,
            self.persona_labels[idx],
            self.persona_ids[idx],
            self.participant_ids[idx],
            cf,
            cl,
        )


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


def nt_xent_views(
    emb_v1: Tensor,
    emb_v2: Tensor,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Tensor:
    """NT-Xent for position-matched participant split-view pairs.

    emb_v1[i] and emb_v2[i] are the positive pair (two trial-subset mean-pools
    for the same participant). All other cross-participant pairs are negatives.
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
        AcquisitionEvent(
            **{
                k: v
                for k, v in json.loads(line).items()
                if k in AcquisitionEvent.__dataclass_fields__
            }
        )
        for line in traces_path.read_text().strip().split("\n")
        if line.strip()
    ]
    trials = {
        r.trial_id: r
        for line in trials_path.read_text().strip().split("\n")
        if line.strip()
        for r in [
            TrialRecord(
                **{
                    k: v
                    for k, v in json.loads(line).items()
                    if k in TrialRecord.__dataclass_fields__
                }
            )
        ]
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
    choice_sets: dict[str, dict] | None = None,
    products: dict[str, dict] | None = None,
) -> tuple[TraceDataset, dict[str, int]]:
    """
    Tokenise all trials and build a TraceDataset.

    If ``choice_sets`` and ``products`` are provided (bead b8b), each trial is
    also attached to its ChoiceSet's per-slot product features + chosen labels
    for the auxiliary choice-prediction loss.

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
    choice_feats: list[Tensor | None] = []
    choice_labels: list[Tensor | None] = []

    for tid, trial in trials.items():
        trial_events = events_by_trial.get(tid, [])
        tokens, mask = tokenise_trial(trial_events, trial, vocab)
        tokens_list.append((tokens, mask))
        persona_labels.append(persona_to_label[trial.persona_id])
        persona_ids.append(trial.persona_id)
        trial_ids.append(trial.trial_id)
        participant_ids.append(trial.participant_id)
        if choice_sets is not None and products is not None:
            ct = _trial_choice_tensors(trial.trial_id, choice_sets, products)
            if ct is None:
                choice_feats.append(None)
                choice_labels.append(None)
            else:
                cf, cl = ct
                choice_feats.append(cf)
                choice_labels.append(cl)

    dataset = TraceDataset(
        tokens_list,
        persona_labels,
        persona_ids,
        trial_ids,
        participant_ids,
        choice_feats,
        choice_labels,
    )
    return dataset, persona_to_label


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def collate_fn(batch):
    """Collate a batch of (tokens, mask, label, persona_id, participant_id, choice_feats, choice_labels).

    Returns padded tokens/mask + labels + ids + flattened choice tensors
    (batch_choice_feats [total_slots, 8], batch_choice_labels [total_slots],
    batch_choice_trial_idx [total_slots] mapping each slot to its batch row).
    """
    tokens_masks = [(t, m) for t, m, *_ in batch]
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    persona_ids = [item[3] for item in batch]
    participant_ids = [item[4] for item in batch]

    # Flatten per-trial choice tensors across the batch (bead b8b).
    feats_list, labels_list, trial_idx_list = [], [], []
    for i, item in enumerate(batch):
        cf, cl = item[5], item[6]
        if cf is None or cl is None or cf.numel() == 0:
            continue
        feats_list.append(cf)
        labels_list.append(cl)
        trial_idx_list.append(torch.full((cf.size(0),), i, dtype=torch.long))
    if feats_list:
        batch_choice_feats = torch.cat(feats_list, dim=0)
        batch_choice_labels = torch.cat(labels_list, dim=0)
        batch_choice_trial_idx = torch.cat(trial_idx_list, dim=0)
    else:
        batch_choice_feats = torch.empty((0, 8), dtype=torch.float)
        batch_choice_labels = torch.empty((0,), dtype=torch.float)
        batch_choice_trial_idx = torch.empty((0,), dtype=torch.long)

    padded_tokens, padded_mask = collate_batch(tokens_masks)
    return (
        padded_tokens,
        padded_mask,
        labels,
        persona_ids,
        participant_ids,
        batch_choice_feats,
        batch_choice_labels,
        batch_choice_trial_idx,
    )


def train(
    traces_path: Path = TRACES_PATH,
    trials_path: Path = TRIALS_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    n_epochs: int = DEFAULT_EPOCHS,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    seed: int = DEFAULT_SEED,
    device: str = "cpu",
    save_path: Path | None = None,
    lambda_contrastive: float = 0.5,
    nt_xent_temperature: float = DEFAULT_TEMPERATURE,
) -> TraceEncoder:
    """
    Train the trace encoder with supervised cross-entropy classification.

    Saves encoder backbone weights (NOT classification head) to
    save_path, defaulting to CHECKPOINT_PATHS["trace"].

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
    save_path:
        Path to save backbone weights. Defaults to CHECKPOINT_PATHS["trace"].

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

    # bead b8b: load choice sets + products for the auxiliary choice loss.
    # Skipped (None) if the files are absent — e.g. in tests without data.
    try:
        choice_sets, products = load_choice_data()
        logger.info("Loaded %d choice sets for auxiliary choice loss", len(choice_sets))
    except FileNotFoundError:
        choice_sets, products = None, None
        logger.warning("choice_sets.jsonl not found — auxiliary choice loss disabled")

    train_dataset, _ = build_dataset(
        train_events, train_trials, persona_to_label, vocab, choice_sets, products
    )
    val_dataset, _ = build_dataset(
        val_events, val_trials, persona_to_label, vocab, choice_sets, products
    )

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
            "objective": "ce+nt_xent+choice_b8b",
            "lambda_choice": LAMBDA_CHOICE,
            "lambda_contrastive": lambda_contrastive,
            "nt_xent_temperature": nt_xent_temperature,
        }
    )

    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    for epoch in range(n_epochs):
        # Re-shuffle sampler each epoch (also re-randomises trial splits below)
        stratified_sampler.set_epoch(epoch)

        # --- Training ---
        encoder.train()
        epoch_cls_loss = 0.0
        epoch_nt_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            tokens, mask, labels = batch[0], batch[1], batch[2]
            participant_ids_batch = batch[4]
            ch_feats, ch_labels, ch_trial_idx = batch[5], batch[6], batch[7]
            tokens = tokens.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # CE loss on individual trial embeddings
            trial_embs, logits = encoder.forward_with_logits(tokens, mask)
            cls_loss = cross_entropy_loss(logits, labels)

            # NT-Xent: per-epoch random 50/50 split of trials per participant
            # Group trial embeddings by participant_id
            pid_to_embs: dict[str, list[Tensor]] = defaultdict(list)
            for emb, pid in zip(trial_embs, participant_ids_batch):
                pid_to_embs[pid].append(emb)

            v1_list: list[Tensor] = []
            v2_list: list[Tensor] = []
            for pid, embs in pid_to_embs.items():
                if len(embs) < 2:
                    continue
                perm = torch.randperm(len(embs), device=device)
                half = len(embs) // 2
                v1_list.append(torch.stack([embs[i] for i in perm[:half]]).mean(0))
                v2_list.append(torch.stack([embs[i] for i in perm[half:]]).mean(0))

            if len(v1_list) >= 2:
                nt_loss = nt_xent_views(
                    torch.stack(v1_list),
                    torch.stack(v2_list),
                    nt_xent_temperature,
                )
            else:
                nt_loss = torch.tensor(0.0, device=device)

            # bead b8b: auxiliary choice-prediction loss. For each (trial, slot)
            # row, predict P(chosen) from concat(trial_emb, 8-dim product vector).
            if ch_feats.numel() > 0:
                ch_feats = ch_feats.to(device)
                ch_labels = ch_labels.to(device)
                ch_trial_idx = ch_trial_idx.to(device)
                slot_trial_embs = trial_embs[ch_trial_idx]  # (total_slots, 128)
                choice_in = torch.cat([slot_trial_embs, ch_feats], dim=1)
                choice_logits = encoder.choice_head(choice_in).squeeze(-1)
                choice_loss = F.binary_cross_entropy_with_logits(
                    choice_logits, ch_labels
                )
            else:
                choice_loss = torch.tensor(0.0, device=device)

            loss = cls_loss + lambda_contrastive * nt_loss + LAMBDA_CHOICE * choice_loss
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

        with torch.no_grad():
            for batch in val_loader:
                tokens, mask, labels = batch[0], batch[1], batch[2]
                tokens = tokens.to(device)
                mask = mask.to(device)
                labels = labels.to(device)

                _, logits = encoder.forward_with_logits(tokens, mask)
                loss = cross_entropy_loss(logits, labels)
                val_cls_loss += loss.item()
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_cls_loss / max(n_batches, 1)
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

        # Early stopping on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            _save_path = (
                save_path if save_path is not None else CHECKPOINT_PATHS["trace"]
            )
            _save_path.parent.mkdir(parents=True, exist_ok=True)
            backbone_state = {
                k: v
                for k, v in encoder.state_dict().items()
                if not k.startswith("classifier") and not k.startswith("choice_head")
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
    logger.info("Training complete. Best val loss: %.4f", best_val_loss)

    return encoder


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with mlflow.start_run(run_name="trace_encoder_v1"):
        train()
