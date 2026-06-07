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


def load_encoders(device: str = "cpu") -> dict[str, nn.Module]:
    """Load all four encoder checkpoints (frozen).

    Parameters
    ----------
    device : str
        Target device for models ("cpu" or "cuda").

    Returns
    -------
    dict[str, nn.Module]
        Dictionary mapping modality names to frozen encoder models.
    """
    from encoders.trace.model import TraceEncoder
    from encoders.transaction.model import TransactionEncoder
    from encoders.text.embed import TextEncoder
    from encoders.psychographic.model import PsychographicEncoder

    encoders = {}

    # Trace encoder
    trace_encoder = TraceEncoder(n_classes=7).to(device)
    trace_state = torch.load(
        CHECKPOINT_PATHS["trace"], map_location=device, weights_only=True
    )
    trace_encoder.load_state_dict(trace_state, strict=False)
    for param in trace_encoder.parameters():
        param.requires_grad = False
    encoders["trace"] = trace_encoder

    # Transaction encoder
    tx_encoder = TransactionEncoder(n_classes=7).to(device)
    tx_state = torch.load(
        CHECKPOINT_PATHS["transaction"], map_location=device, weights_only=True
    )
    tx_encoder.load_state_dict(tx_state)
    for param in tx_encoder.parameters():
        param.requires_grad = False
    encoders["transaction"] = tx_encoder

    # Text encoder
    text_encoder = TextEncoder(n_classes=7).to(device)
    # Text encoder checkpoint contains only trainable parameters
    text_state = torch.load(
        CHECKPOINT_PATHS["text"], map_location=device, weights_only=True
    )
    text_encoder.load_state_dict(text_state, strict=False)
    for param in text_encoder.parameters():
        param.requires_grad = False
    encoders["text"] = text_encoder

    # Psychographic encoder
    psycho_encoder = PsychographicEncoder(n_classes=7).to(device)
    psycho_state = torch.load(
        CHECKPOINT_PATHS["psychographic"], map_location=device, weights_only=True
    )
    psycho_encoder.load_state_dict(psycho_state)
    for param in psycho_encoder.parameters():
        param.requires_grad = False
    encoders["psychographic"] = psycho_encoder

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
) -> dict[str, torch.Tensor]:
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
    modality_data: dict[str, dict[str, torch.Tensor]],
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Generate embeddings for all participants using frozen encoders.

    Parameters
    ----------
    encoders : dict[str, nn.Module]
        Frozen encoder models.
    psychographics : list[dict]
        Psychographic records with participant_id and persona_id.
    modality_data : dict[str, dict[str, torch.Tensor]]
        Modality data indexed by modality name and participant_id.
    device : str
        Target device for computation.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with keys "trace", "transaction", "text", "psychographic",
        "labels", "participant_ids". Each embedding tensor has shape [N, 128].
    """
    n_participants = len(psychographics)
    embeddings = {
        "trace": torch.zeros(n_participants, EMBEDDING_DIM, device=device),
        "transaction": torch.zeros(n_participants, EMBEDDING_DIM, device=device),
        "text": torch.zeros(n_participants, EMBEDDING_DIM, device=device),
        "psychographic": torch.zeros(n_participants, EMBEDDING_DIM, device=device),
        "labels": torch.zeros(n_participants, dtype=torch.long, device=device),
        "participant_ids": [],
    }

    for i, psycho in enumerate(psychographics):
        participant_id = psycho["participant_id"]
        embeddings["participant_ids"].append(participant_id)
        embeddings["labels"][i] = PERSONA_TO_IDX[psycho["persona_id"]]

        # Generate trace embedding
        trace_data = modality_data["traces"].get(participant_id, [])
        trace_emb = encoders["trace"].embed_trace(trace_data)
        embeddings["trace"][i] = trace_emb

        # Generate transaction embedding
        tx_data = modality_data["transactions"].get(participant_id, [])
        tx_emb = encoders["transaction"].embed_transactions(tx_data)
        embeddings["transaction"][i] = tx_emb

        # Generate text embedding
        narrative_data = modality_data["narratives"].get(participant_id)
        if narrative_data:
            text_emb = encoders["text"].encode_texts([narrative_data["text"]])
            embeddings["text"][i] = text_emb.squeeze(0)

        # Generate psychographic embedding
        psycho_emb = encoders["psychographic"].embed_psychographic(psycho)
        embeddings["psychographic"][i] = psycho_emb

    return embeddings


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


def build_cache(
    encoders: dict[str, nn.Module],
    cache_path: Path,
    device: str = "cpu",
) -> dict[str, torch.Tensor | list[str]]:
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
    dict[str, torch.Tensor | list[str]]
        Cached embeddings with keys "trace", "transaction", "text",
        "psychographic", "labels", "participant_ids".
    """
    # Check if cache is valid (newer than all encoder checkpoints)
    cache_valid = cache_path.exists()
    if cache_valid:
        cache_mtime = cache_path.stat().st_mtime
        for modality, checkpoint_path in CHECKPOINT_PATHS.items():
            if modality == "fusion":
                continue
            if checkpoint_path.stat().st_mtime > cache_mtime:
                cache_valid = False
                break

    if cache_valid:
        print(f"Loading cached embeddings from {cache_path}")
        # Note: weights_only=False required for cache (dict with tensors + list)
        # Safe because cache is created by our own code in same script
        return torch.load(cache_path)

    # Build cache
    print("Generating embeddings from encoder checkpoints...")

    # Load psychographics to get participant ordering
    psychographics = load_psychographics()
    participant_ids = [p["participant_id"] for p in psychographics]

    # Load modality data
    modality_data = {
        "traces": load_modality_data("traces", participant_ids),
        "transactions": load_modality_data("transactions", participant_ids),
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


def train(
    *,
    cache_path: Optional[Path] = None,
    n_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    p_dropout: float = 0.2,
    device: str = "cpu",
    log_mlflow: bool = True,
    phase: str = "2",
) -> LateFusionMetaLearner:
    """Train the fusion meta-learner.

    Parameters
    ----------
    cache_path : Path | None
        Path to embedding cache. Default: models/fusion_embeddings_cache.pt.
    n_epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size.
    lr : float
        Learning rate.
    p_dropout : float
        Modality dropout probability during training.
    device : str
        Target device ("cpu" or "cuda").
    log_mlflow : bool
        Whether to log to MLflow.
    phase : str
        Meta-learner phase ("1" or "2").

    Returns
    -------
    LateFusionMetaLearner
        Trained meta-learner model.

    Notes
    -----
    - Encoders are frozen during fusion training.
    - L2-normalisation is applied to each modality embedding before concatenation.
    - Modality dropout (p=0.2) is applied independently to each modality during training.
    - Partial-modality inference is supported: zero the 128-dim slice for missing modalities.
    """
    if cache_path is None:
        cache_path = Path("models/fusion_embeddings_cache.pt")

    # Load encoders
    print("Loading encoder checkpoints...")
    encoders = load_encoders(device)
    print("Encoders loaded and frozen.")

    # Build or load embedding cache
    embeddings = build_cache(encoders, cache_path, device)

    # Split participants
    participant_ids = embeddings["participant_ids"]
    train_ids, val_ids = split_by_participant(participant_ids)

    # Create train/val indices
    participant_to_idx = {pid: i for i, pid in enumerate(participant_ids)}
    train_indices = torch.tensor([participant_to_idx[pid] for pid in train_ids])
    val_indices = torch.tensor([participant_to_idx[pid] for pid in val_ids])

    # Extract embeddings and labels
    train_embs = {
        mod: embeddings[mod][train_indices]
        for mod in ["trace", "transaction", "text", "psychographic"]
    }
    train_labels = embeddings["labels"][train_indices]

    val_embs = {
        mod: embeddings[mod][val_indices]
        for mod in ["trace", "transaction", "text", "psychographic"]
    }
    val_labels = embeddings["labels"][val_indices]

    # Create dataset
    def make_dataset(embs_dict, labels):
        emb_list = [
            embs_dict[mod] for mod in ["trace", "transaction", "text", "psychographic"]
        ]
        all_embs = torch.stack(emb_list, dim=1)  # [N, 4, 128]
        return TensorDataset(all_embs, labels)

    train_ds = make_dataset(train_embs, train_labels)
    val_ds = make_dataset(val_embs, val_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = LateFusionMetaLearner(phase=phase).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True
    )

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 10

    for epoch in range(n_epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_embs, batch_labels in train_loader:
            batch_embs = batch_embs.to(device)
            batch_labels = batch_labels.to(device)

            # Apply modality dropout
            if model.training:
                batch_size = batch_embs.shape[0]
                dropout_masks = [
                    torch.rand(batch_size, 1, device=device) >= p_dropout
                    for _ in range(4)
                ]
                for i in range(4):
                    batch_embs[:, i] = batch_embs[:, i] * dropout_masks[i].float()

            # L2-normalise each modality
            norm_embs = [F.normalize(batch_embs[:, i], p=2, dim=-1) for i in range(4)]
            fusion_input = torch.cat(norm_embs, dim=-1)  # [B, 512]

            # Forward pass
            logits = model(fusion_input)
            loss = criterion(logits, batch_labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for batch_embs, batch_labels in val_loader:
                batch_embs = batch_embs.to(device)
                batch_labels = batch_labels.to(device)

                # No dropout during validation
                norm_embs = [
                    F.normalize(batch_embs[:, i], p=2, dim=-1) for i in range(4)
                ]
                fusion_input = torch.cat(norm_embs, dim=-1)

                logits = model(fusion_input)
                loss = criterion(logits, batch_labels)
                val_loss += loss.item()

                predictions = logits.argmax(dim=-1)
                val_correct += (predictions == batch_labels).sum().item()
                val_total += batch_labels.shape[0]

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch + 1}/{n_epochs}: "
            f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if log_mlflow:
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)

        # Learning rate scheduling
        scheduler.step(val_acc)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            # Save checkpoint
            checkpoint_path = CHECKPOINT_PATHS["fusion"]
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  → New best model! Saved to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"  → Early stopping triggered (patience={max_patience})")
                break

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    return model


if __name__ == "__main__":
    train()
