"""
Transaction encoder training pipeline.

Self-supervised training via next-brand_tier prediction. Given transaction
history t1..tn, predict the brand_tier of t_{n+1}. Uses cross-entropy loss
on a 4-class classification task.

Training configuration (from SPEC.md):
    - Batch size: 128 (participant-level)
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
from typing import Sequence

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

from schemas import CHECKPOINT_PATHS
from schemas.transaction import TransactionRecord
from encoders.transaction.features import (
    MAX_SEQ_LEN,
    TOKEN_DIM,
    TxVocabulary,
    sort_transactions_most_recent_first,
)
from encoders.transaction.model import NextBrandTierHead, TransactionEncoder

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/synthetic")
TRANSACTIONS_FILE = DATA_DIR / "transactions.jsonl"

# Training defaults
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 5e-4
DEFAULT_EPOCHS = 30
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GRU_HIDDEN = 128
DEFAULT_GRU_LAYERS = 2
DEFAULT_GRU_DROPOUT = 0.1
DEFAULT_PROJECTION_DIM = 64
DEFAULT_PATIENCE = 5
RANDOM_SEED = 42


class TransactionSequenceDataset(Dataset):
    """Dataset that groups TransactionRecords by participant.

    Each item is a tuple of:
      (token_seq, brand_tier_targets, length)
    where:
      - token_seq: (T, 20) tensor of token vectors (input = t1..tn)
      - brand_tier_targets: (T,) long tensor of brand_tier indices (target = t2..t_{n+1})
      - length: actual sequence length (int)
    """

    def __init__(
        self,
        records: Sequence[TransactionRecord],
        vocab: TxVocabulary,
        max_seq_len: int = MAX_SEQ_LEN,
    ) -> None:
        self.vocab = vocab
        self.max_seq_len = max_seq_len

        # Group by participant, sort most-recent-first
        by_participant: dict[str, list[TransactionRecord]] = defaultdict(list)
        for r in records:
            by_participant[r.participant_id].append(r)

        self.sequences: list[tuple[torch.Tensor, torch.Tensor, int]] = []
        for _, txs in by_participant.items():
            txs = sort_transactions_most_recent_first(txs)[:max_seq_len]

            # Build brand_tier target indices for all records
            brand_tier_indices = [vocab.brand_tier_index(r.brand_tier) for r in txs]

            # Input: t1..t(n-1), Target: t2..tn (next brand_tier prediction)
            # For input, we need token vectors for t1..t(n-1)
            if len(txs) < 2:
                # Not enough transactions for next-step prediction, but
                # include for embedding — target is just the last one
                input_txs = txs
                targets = brand_tier_indices
            else:
                input_txs = txs[:-1]
                targets = brand_tier_indices[1:]

            token_seq = vocab.encode_sequence(input_txs)  # (T, 20)
            target_tensor = torch.tensor(targets, dtype=torch.long)

            # Detach — dataset tensors must not carry gradient history.
            # token_seq inherits requires_grad=True from vocab embedding
            # lookups, which causes "double backward" errors across epochs.
            self.sequences.append((token_seq.detach(), target_tensor, len(input_txs)))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return self.sequences[idx]


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate variable-length sequences into a padded batch.

    Sorts by descending length for pack_padded_sequence compatibility.

    Returns
    -------
    token_seqs : Tensor, shape (B, T_max, 20)
    targets : Tensor, shape (B, T_max)
    lengths : Tensor, shape (B,)
    """
    # Sort by descending length
    batch.sort(key=lambda x: x[2], reverse=True)

    token_seqs, targets, lengths = zip(*batch)
    lengths_t = torch.tensor(lengths, dtype=torch.long)
    max_len = max(lengths)

    B = len(batch)
    padded_tokens = torch.zeros(B, max_len, TOKEN_DIM)
    padded_targets = torch.zeros(B, max_len, dtype=torch.long)

    for i, (seq, tgt, length) in enumerate(zip(token_seqs, targets, lengths)):
        padded_tokens[i, :length] = seq
        padded_targets[i, :length] = tgt

    return padded_tokens, padded_targets, lengths_t


def split_by_participant(
    records: Sequence[TransactionRecord],
    val_fraction: float = 0.2,
    seed: int = RANDOM_SEED,
) -> tuple[list[TransactionRecord], list[TransactionRecord]]:
    """Split records into train/val by participant_id (no leakage).

    Returns (train_records, val_records).
    """
    participant_ids = sorted({r.participant_id for r in records})
    rng = np.random.default_rng(seed)
    rng.shuffle(participant_ids)

    split = int((1 - val_fraction) * len(participant_ids))
    train_ids = set(participant_ids[:split])
    val_ids = set(participant_ids[split:])

    train_records = [r for r in records if r.participant_id in train_ids]
    val_records = [r for r in records if r.participant_id in val_ids]

    logger.info(
        "Split: %d train participants (%d records), %d val participants (%d records)",
        len(train_ids),
        len(train_records),
        len(val_ids),
        len(val_records),
    )
    return train_records, val_records


def load_records(path: Path | str | None = None) -> list[TransactionRecord]:
    """Load transaction records from JSONL."""
    path = Path(path) if path else TRANSACTIONS_FILE
    records = []
    for line in path.open():
        records.append(TransactionRecord(**json.loads(line)))
    logger.info("Loaded %d transaction records from %s", len(records), path)
    return records


def train_epoch(
    encoder: TransactionEncoder,
    prediction_head: NextBrandTierHead,
    dataloader: DataLoader,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch. Returns mean loss."""
    encoder.train()
    prediction_head.train()
    total_loss = 0.0
    n_batches = 0

    for token_seqs, targets, lengths in dataloader:
        token_seqs = token_seqs.to(device)
        targets = targets.to(device)
        lengths = lengths.to(device)

        # Project tokens and run through GRU
        projected = torch.relu(encoder.token_proj(token_seqs))

        packed = pack_padded_sequence(
            projected, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        gru_output_packed, _ = encoder.gru(packed)

        # Unpack GRU outputs to get hidden state at each step
        gru_output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            gru_output_packed, batch_first=True
        )

        # Predict next brand_tier at each step
        logits = prediction_head(gru_output)  # (B, T, 4)

        # Build mask for valid positions only
        B, T, _ = logits.shape
        mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)

        # Compute loss only on valid positions
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        losses = loss_fn(
            logits.view(-1, 4),
            targets.view(-1),
        )
        losses = losses.view(B, T) * mask
        loss = losses.sum() / mask.sum()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_epoch(
    encoder: TransactionEncoder,
    prediction_head: NextBrandTierHead,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate on validation set. Returns mean loss."""
    encoder.eval()
    prediction_head.eval()
    total_loss = 0.0
    n_batches = 0

    for token_seqs, targets, lengths in dataloader:
        token_seqs = token_seqs.to(device)
        targets = targets.to(device)
        lengths = lengths.to(device)

        projected = torch.relu(encoder.token_proj(token_seqs))
        packed = pack_padded_sequence(
            projected, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        gru_output_packed, _ = encoder.gru(packed)
        gru_output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            gru_output_packed, batch_first=True
        )

        logits = prediction_head(gru_output)

        B, T, _ = logits.shape
        mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)

        loss_fn = nn.CrossEntropyLoss(reduction="none")
        losses = loss_fn(logits.view(-1, 4), targets.view(-1))
        losses = losses.view(B, T) * mask
        loss = losses.sum() / mask.sum()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train(
    records: list[TransactionRecord] | None = None,
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
) -> TransactionEncoder:
    """Train the transaction encoder with next-brand_tier prediction.

    Parameters
    ----------
    records : list[TransactionRecord] or None
        Transaction records. If None, loads from data/synthetic/transactions.jsonl.
    batch_size : int
        Participant-level batch size.
    lr : float
        Learning rate.
    n_epochs : int
        Maximum training epochs.
    weight_decay : float
        AdamW weight decay.
    gru_hidden : int
        GRU hidden size.
    gru_layers : int
        Number of GRU layers.
    gru_dropout : float
        Dropout between GRU layers.
    projection_dim : int
        Token projection dimension.
    patience : int
        Early stopping patience (epochs without improvement).
    seed : int
        Random seed for train/val split.
    device : str or None
        Device string. Defaults to "cuda" if available, else "cpu".

    Returns
    -------
    TransactionEncoder
        Trained encoder (without prediction head).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    # Load data
    if records is None:
        records = load_records()

    # Split by participant
    train_records, val_records = split_by_participant(records, seed=seed)

    # Build vocabulary and encoder
    vocab = TxVocabulary()
    vocab.save_vocab()

    encoder = TransactionEncoder(
        vocab=vocab,
        projection_dim=projection_dim,
        gru_hidden=gru_hidden,
        gru_layers=gru_layers,
        gru_dropout=gru_dropout,
    ).to(torch_device)

    # Prediction head (trained jointly, discarded after)
    prediction_head = NextBrandTierHead(gru_hidden=gru_hidden).to(torch_device)

    # Datasets and dataloaders
    train_ds = TransactionSequenceDataset(train_records, vocab)
    val_ds = TransactionSequenceDataset(val_records, vocab)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Optimiser — joint params from encoder + prediction head + vocab embeddings
    params = list(encoder.parameters()) + list(prediction_head.parameters())
    optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # MLflow tracking
    with mlflow.start_run(run_name="transaction_encoder_v1"):
        mlflow.set_tag("modality", "transaction")
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
            }
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(n_epochs):
            train_loss = train_epoch(
                encoder, prediction_head, train_loader, optimiser, torch_device
            )
            val_loss = eval_epoch(encoder, prediction_head, val_loader, torch_device)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            logger.info(
                "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f",
                epoch + 1,
                n_epochs,
                train_loss,
                val_loss,
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best weights
                best_state = {
                    "encoder": encoder.state_dict(),
                    "vocab": vocab.state_dict(),
                }
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        # Restore best weights
        if best_state is not None:
            encoder.load_state_dict(best_state["encoder"])

        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.pytorch.log_model(encoder, "transaction_encoder")

    # Save local checkpoint for probe evaluation
    _save_path = CHECKPOINT_PATHS["transaction"]
    _save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), _save_path)
    logger.info("Checkpoint saved to %s", _save_path)

    logger.info("Training complete. Best val loss: %.4f", best_val_loss)
    return encoder


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
