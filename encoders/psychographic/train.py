"""
Training script for the psychographic encoder.

Multi-task objective: CE (archetype classification) + NT-Xent (individual identity).
Two augmented views per participant are created inline via independent feature dropout
(p=0.1 per feature, applied in the training loop — not an nn.Module).
Split by participant_id to prevent data leakage.
Logs to MLflow with tag ``modality=psychographic``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import mlflow

from schemas import CHECKPOINT_PATHS, EMBEDDING_DIM, PERSONA_LABELS, PERSONA_TO_IDX
from schemas.psychographic import PsychographicVector

from encoders.psychographic.features import (
    batch_to_feature_matrix,
    save_vocab,
)
from encoders.psychographic.model import PsychographicEncoder


# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------


def load_psychographics(
    path: Path = Path("data/synthetic/psychographics.jsonl"),
) -> list[PsychographicVector]:
    """Load psychographic records from JSONL."""
    return [
        PsychographicVector(**json.loads(line))
        for line in path.read_text().strip().splitlines()
    ]


def split_by_participant(
    records: list[PsychographicVector],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[PsychographicVector], list[PsychographicVector]]:
    """Split records into train/val sets by participant_id."""
    participant_ids = sorted(set(r.participant_id for r in records))
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(participant_ids)
    split_idx = int(train_ratio * len(participant_ids))
    train_ids = set(participant_ids[:split_idx])
    val_ids = set(participant_ids[split_idx:])

    train_records = [r for r in records if r.participant_id in train_ids]
    val_records = [r for r in records if r.participant_id in val_ids]
    return train_records, val_records


def records_to_tensors(
    records: list[PsychographicVector],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert records to (features, labels) tensors."""
    features = batch_to_feature_matrix(records)
    labels = torch.tensor(
        [PERSONA_TO_IDX[r.persona_id] for r in records], dtype=torch.long
    )
    return features, labels


# ---------------------------------------------------------------------------
# NT-Xent for augmented view pairs (SimCLR-style)
# ---------------------------------------------------------------------------


def nt_xent_views(
    emb_v1: torch.Tensor,
    emb_v2: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """NT-Xent contrastive loss for a matched pair of augmented views.

    emb_v1[i] and emb_v2[i] are the positive pair for sample i.
    All other cross-sample pairs are negatives.

    Parameters
    ----------
    emb_v1, emb_v2 : Tensor, shape (B, D)
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

    # Positive for row i (0..B-1) is row i+B; for row i (B..2B-1) is row i-B
    labels = torch.cat(
        [
            torch.arange(B, 2 * B, device=emb_v1.device),
            torch.arange(0, B, device=emb_v1.device),
        ]
    )

    # Mask out self-similarity (diagonal)
    mask = torch.eye(2 * B, dtype=torch.bool, device=emb_v1.device)
    sim = sim.masked_fill(mask, float("-inf"))

    return F.cross_entropy(sim, labels)


# ---------------------------------------------------------------------------
# Post-training diagnostic: within-archetype similarity delta
# ---------------------------------------------------------------------------


def compute_similarity_delta(
    model: PsychographicEncoder,
    records: list[PsychographicVector],
    feat_dropout_p: float,
    device: str,
    n_samples: int = 500,
) -> float:
    """Mean pairwise cosine similarity for same-participant augmented views
    minus the same-archetype cross-participant baseline.

    Returns delta > 0 when same-participant views are more similar than
    randomly chosen same-archetype pairs.
    """
    model.eval()
    features, _ = records_to_tensors(records)
    features = features.to(device)

    # Build persona_id → indices map
    by_persona: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_persona[r.persona_id].append(i)

    rng = torch.Generator(device=device)
    rng.manual_seed(0)

    same_participant_sims: list[float] = []
    cross_participant_sims: list[float] = []

    with torch.no_grad():
        for idx in range(min(n_samples, len(records))):
            feat = features[idx].unsqueeze(0)
            # Two augmented views
            m1 = (torch.rand_like(feat, generator=rng) > feat_dropout_p).float()
            m2 = (torch.rand_like(feat, generator=rng) > feat_dropout_p).float()
            e1 = F.normalize(model(feat * m1), dim=1)
            e2 = F.normalize(model(feat * m2), dim=1)
            same_participant_sims.append((e1 * e2).sum().item())

            # Cross-participant baseline: pick a different participant in same archetype
            pid = records[idx].persona_id
            same_arch = [j for j in by_persona[pid] if j != idx]
            if not same_arch:
                continue
            j = same_arch[torch.randint(len(same_arch), (1,), generator=rng).item()]  # type: ignore[arg-type]
            feat_j = features[j].unsqueeze(0)
            m3 = (torch.rand_like(feat_j, generator=rng) > feat_dropout_p).float()
            m4 = (torch.rand_like(feat_j, generator=rng) > feat_dropout_p).float()
            ej1 = F.normalize(model(feat_j * m3), dim=1)
            ej2 = F.normalize(model(feat_j * m4), dim=1)
            cross_participant_sims.append((e1 * ej1).sum().item())
            cross_participant_sims.append((e2 * ej2).sum().item())

    if not same_participant_sims or not cross_participant_sims:
        return 0.0
    return float(np.mean(same_participant_sims) - np.mean(cross_participant_sims))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    records: Optional[list[PsychographicVector]] = None,
    *,
    n_epochs: int = 40,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    train_ratio: float = 0.8,
    seed: int = 42,
    device: str = "cpu",
    log_mlflow: bool = True,
    vocab_path: Optional[Path] = None,
    lambda_contrastive: float = 0.5,
    nt_xent_temperature: float = 0.07,
    feat_dropout_p: float = 0.1,
) -> PsychographicEncoder:
    """Train the psychographic encoder with CE + NT-Xent multi-task objective.

    Two augmented views per sample are created inline in each training batch
    via independent feature dropout (p=feat_dropout_p). NT-Xent treats the
    two views of the same sample as a positive pair and all other samples as
    negatives (SimCLR-style).

    Parameters
    ----------
    records
        Pre-loaded records. If ``None``, loads from ``data/synthetic/``.
    n_epochs
        Number of training epochs.
    batch_size
        Mini-batch size.
    lr
        Learning rate.
    weight_decay
        AdamW weight decay.
    train_ratio
        Fraction of participants assigned to train set.
    seed
        Random seed for participant splitting.
    device
        ``"cpu"`` or ``"cuda"``.
    log_mlflow
        Whether to log the run to MLflow.
    vocab_path
        Path to persist vocabulary JSON. Defaults to ``data/synthetic/psych_vocab.json``.
    lambda_contrastive
        Weight for NT-Xent loss (total = CE + lambda * NT-Xent).
    nt_xent_temperature
        Temperature parameter for NT-Xent (default 0.07).
    feat_dropout_p
        Per-feature zero probability for augmentation views.

    Returns
    -------
    PsychographicEncoder
        Trained encoder (with classification head still attached).
    """
    if records is None:
        records = load_psychographics()

    # Persist vocabulary
    save_vocab(vocab_path)

    # Split by participant
    train_records, val_records = split_by_participant(
        records, train_ratio=train_ratio, seed=seed
    )

    train_features, train_labels = records_to_tensors(train_records)
    val_features, val_labels = records_to_tensors(val_records)

    train_ds = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Model, optimiser, loss
    n_classes = len(PERSONA_LABELS)
    model = PsychographicEncoder(n_classes=n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    def _evaluate() -> tuple[float, float]:
        """Returns (val_loss, val_acc)."""
        model.eval()
        with torch.no_grad():
            feats = val_features.to(device)
            labels = val_labels.to(device)
            _, logits = model.forward_with_logits(feats)
            loss = criterion(logits, labels).item()
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean().item()
        return loss, acc

    def _train_loop() -> None:
        best_val_loss = float("inf")
        patience, patience_counter = 10, 0

        for epoch in range(n_epochs):
            model.train()
            epoch_ce = 0.0
            epoch_nt = 0.0
            n_batches = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                # CE loss on original features
                _, logits = model.forward_with_logits(batch_x)
                ce_loss = criterion(logits, batch_y)

                # NT-Xent: two feature-dropout augmented views (inline, not an nn.Module)
                mask1 = (torch.rand_like(batch_x) > feat_dropout_p).float()
                mask2 = (torch.rand_like(batch_x) > feat_dropout_p).float()
                emb_v1 = model(batch_x * mask1)
                emb_v2 = model(batch_x * mask2)
                nt_loss = nt_xent_views(emb_v1, emb_v2, nt_xent_temperature)

                loss = ce_loss + lambda_contrastive * nt_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_ce += ce_loss.item()
                epoch_nt += nt_loss.item()
                n_batches += 1

            avg_ce = epoch_ce / max(n_batches, 1)
            avg_nt = epoch_nt / max(n_batches, 1)
            val_loss, val_acc = _evaluate()

            if log_mlflow:
                mlflow.log_metrics(
                    {
                        "train_ce_loss": avg_ce,
                        "train_nt_loss": avg_nt,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    },
                    step=epoch,
                )

            print(
                f"Epoch {epoch + 1}/{n_epochs}  "
                f"ce={avg_ce:.4f}  nt={avg_nt:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                checkpoint_path = CHECKPOINT_PATHS["psychographic"]
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # Restore best checkpoint
        checkpoint_path = CHECKPOINT_PATHS["psychographic"]
        if checkpoint_path.exists():
            model.load_state_dict(
                torch.load(checkpoint_path, map_location=device, weights_only=True)
            )

    if log_mlflow:
        with mlflow.start_run(run_name="psychographic_encoder_v2_contrastive"):
            mlflow.set_tag("modality", "psychographic")
            mlflow.log_params(
                {
                    "lr": lr,
                    "n_epochs": n_epochs,
                    "batch_size": batch_size,
                    "weight_decay": weight_decay,
                    "train_ratio": train_ratio,
                    "embedding_dim": EMBEDDING_DIM,
                    "lambda_contrastive": lambda_contrastive,
                    "nt_xent_temperature": nt_xent_temperature,
                    "feat_dropout_p": feat_dropout_p,
                    "objective": "ce+nt_xent",
                }
            )
            _train_loop()
            final_val_loss, final_val_acc = _evaluate()
            mlflow.log_metric("final_val_loss", final_val_loss)
            mlflow.log_metric("final_val_acc", final_val_acc)

            # Diagnostic: within-archetype similarity delta
            sim_delta = compute_similarity_delta(
                model, val_records, feat_dropout_p, device
            )
            mlflow.log_metric("similarity_delta_within_vs_cross_archetype", sim_delta)
            print(
                f"Similarity delta (same-participant vs cross-participant): {sim_delta:.4f}"
            )
            print(f"Final val_acc: {final_val_acc:.4f}")
    else:
        _train_loop()

    print(f"Saved checkpoint to {CHECKPOINT_PATHS['psychographic']}")
    return model


if __name__ == "__main__":
    trained_model = train()
    print("Training complete.")
    print(f"Encoder output dim: {EMBEDDING_DIM}")
