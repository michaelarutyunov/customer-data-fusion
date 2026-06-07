"""
Training script for the psychographic encoder.

Supervised classification objective (predict persona archetype).
Split by participant_id to prevent data leakage.
Logs to MLflow with tag ``modality=psychographic``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
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
) -> PsychographicEncoder:
    """Train the psychographic encoder.

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

    def _evaluate(features: torch.Tensor, labels: torch.Tensor) -> float:
        model.eval()
        with torch.no_grad():
            features = features.to(device)
            labels = labels.to(device)
            _, logits = model.forward_with_logits(features)
            loss = criterion(logits, labels).item()
        return loss

    if log_mlflow:
        mlflow.set_tag("modality", "psychographic")

    def _train_loop() -> None:
        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                _, logits = model.forward_with_logits(batch_x)
                loss = criterion(logits, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            val_loss = _evaluate(val_features, val_labels)

            if log_mlflow:
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)

    if log_mlflow:
        with mlflow.start_run(run_name="psychographic_encoder_v1"):
            mlflow.set_tag("modality", "psychographic")
            mlflow.log_params(
                {
                    "lr": lr,
                    "n_epochs": n_epochs,
                    "batch_size": batch_size,
                    "weight_decay": weight_decay,
                    "train_ratio": train_ratio,
                    "embedding_dim": EMBEDDING_DIM,
                }
            )
            _train_loop()
            # Log final val loss as a summary metric
            final_val_loss = _evaluate(val_features, val_labels)
            mlflow.log_metric("final_val_loss", final_val_loss)
    else:
        _train_loop()

    # Save checkpoint
    checkpoint_path = CHECKPOINT_PATHS["psychographic"]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    return model


if __name__ == "__main__":
    trained_model = train()
    print("Training complete.")
    print(f"Encoder output dim: {EMBEDDING_DIM}")
