"""
Transaction encoder training pipeline.

Multi-task objective: CE (archetype classification) + NT-Xent (individual identity).

Each participant's transaction history is split at the median timestamp into two
chronological halves. Both halves are encoded via the GRU to produce participant-level
embeddings; NT-Xent treats the two halves of the same participant as a positive pair.
Archetype CE uses the full-sequence embedding.

Participants with fewer than 4 transactions are excluded from the NT-Xent loss
(CE loss still applied); a warning is logged at dataset load time.

Training configuration:
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
from typing import Sequence

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

from schemas import CHECKPOINT_PATHS, PERSONA_TO_IDX
from schemas.transaction import TransactionRecord
from encoders.transaction.features import (
    MAX_SEQ_LEN,
    TOKEN_DIM,
    TxVocabulary,
)
from encoders.transaction.model import TransactionEncoder

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/synthetic")
TRANSACTIONS_FILE = DATA_DIR / "transactions.jsonl"

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
MIN_TX_FOR_SPLIT = (
    4  # participants below this threshold skip NT-Xent (CE still applied)
)
N_ARCHETYPE_CLASSES = 7


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


# Per-participant item: (full_tokens, full_len, half1_tokens, half1_len,
#                        half2_tokens, half2_len, persona_label, nt_xent_eligible)
_Item = tuple[
    torch.Tensor,
    int,
    torch.Tensor,
    int,
    torch.Tensor,
    int,
    int,
    bool,
]


class SplitTransactionDataset(Dataset):
    """Per-participant dataset with chronological split views for NT-Xent.

    Each item stores:
      - full sequence (oldest-first) for archetype CE
      - first (oldest) half and second (most-recent) half for NT-Xent
      - persona_label for archetype CE
      - nt_xent_eligible: False when < MIN_TX_FOR_SPLIT transactions

    Transactions are sorted oldest-first (ascending days_before_session) so
    the GRU reads the history in chronological order. The split is at the
    median: half1 = oldest transactions, half2 = most-recent transactions.
    """

    def __init__(
        self,
        records: Sequence[TransactionRecord],
        vocab: TxVocabulary,
        max_seq_len: int = MAX_SEQ_LEN,
    ) -> None:
        by_participant: dict[str, list[TransactionRecord]] = defaultdict(list)
        for r in records:
            by_participant[r.participant_id].append(r)

        n_skipped = 0
        self.items: list[_Item] = []

        for _, txs in by_participant.items():
            # Sort oldest-first (largest days_before_session = furthest in the past)
            txs_sorted = sorted(txs, key=lambda r: r.days_before_session, reverse=True)
            txs_sorted = txs_sorted[:max_seq_len]
            n = len(txs_sorted)

            full_tokens = vocab.encode_sequence(txs_sorted).detach()
            persona_label = PERSONA_TO_IDX[txs_sorted[0].persona_id]

            eligible = n >= MIN_TX_FOR_SPLIT
            if not eligible:
                n_skipped += 1
                # Dummy half tensors (won't be used; eligible=False guards usage)
                half1_tokens = full_tokens
                half1_len = n
                half2_tokens = full_tokens
                half2_len = n
            else:
                mid = n // 2
                half1 = txs_sorted[:mid]  # oldest
                half2 = txs_sorted[mid:]  # most recent
                half1_tokens = vocab.encode_sequence(half1).detach()
                half2_tokens = vocab.encode_sequence(half2).detach()
                half1_len = len(half1)
                half2_len = len(half2)

            self.items.append(
                (
                    full_tokens,
                    n,
                    half1_tokens,
                    half1_len,
                    half2_tokens,
                    half2_len,
                    persona_label,
                    eligible,
                )
            )

        if n_skipped:
            logger.warning(
                "%d participants have fewer than %d transactions and will be "
                "excluded from the NT-Xent loss (CE loss still applied).",
                n_skipped,
                MIN_TX_FOR_SPLIT,
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> _Item:
        return self.items[idx]


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
    list[bool],
]:
    """Collate a batch of SplitTransactionDataset items.

    Produces three padded tensors (full, half1, half2), each sorted by
    descending length for pack_padded_sequence compatibility.

    Returns
    -------
    full_tokens   : (B, T_full, TOKEN_DIM)
    full_lengths  : (B,)
    h1_tokens     : (B, T_h1, TOKEN_DIM)
    h1_lengths    : (B,)
    h2_tokens     : (B, T_h2, TOKEN_DIM)
    h2_lengths    : (B,)
    labels        : (B,)  long
    eligible      : list[bool], length B
    """
    (full_seqs, full_lens, h1_seqs, h1_lens, h2_seqs, h2_lens, labels, eligible) = zip(
        *batch
    )

    def _pad(
        seqs: tuple[torch.Tensor, ...], lens: tuple[int, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_l = max(lens)
        B = len(seqs)
        out = torch.zeros(B, max_l, TOKEN_DIM)
        for i, (s, seq_len) in enumerate(zip(seqs, lens)):
            out[i, :seq_len] = s[:seq_len]
        return out, torch.tensor(lens, dtype=torch.long)

    full_t, full_l = _pad(full_seqs, full_lens)
    h1_t, h1_l = _pad(h1_seqs, h1_lens)
    h2_t, h2_l = _pad(h2_seqs, h2_lens)

    # Sort by descending full length for pack_padded_sequence
    order = torch.argsort(full_l, descending=True)
    eligible_list = [eligible[i] for i in order.tolist()]

    return (
        full_t[order],
        full_l[order],
        h1_t[order],
        h1_l[order],
        h2_t[order],
        h2_l[order],
        torch.tensor([labels[i] for i in order.tolist()], dtype=torch.long),
        eligible_list,
    )


# ---------------------------------------------------------------------------
# NT-Xent for split-view pairs
# ---------------------------------------------------------------------------


def nt_xent_views(
    emb_v1: torch.Tensor,
    emb_v2: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """NT-Xent for position-matched split-history participant pairs."""
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


def _encode_sequence(
    encoder: TransactionEncoder,
    token_seqs: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Run the GRU encoder and return (B, EMBEDDING_DIM) participant embeddings."""
    return encoder(token_seqs, lengths)


def split_by_participant(
    records: Sequence[TransactionRecord],
    val_fraction: float = 0.2,
    seed: int = RANDOM_SEED,
) -> tuple[list[TransactionRecord], list[TransactionRecord]]:
    """Split records into train/val by participant_id (no leakage)."""
    participant_ids = sorted({r.participant_id for r in records})
    rng = np.random.default_rng(seed)
    rng.shuffle(participant_ids)

    split = int((1 - val_fraction) * len(participant_ids))
    train_ids = set(participant_ids[:split])
    val_ids = set(participant_ids[split:])

    train_records = [r for r in records if r.participant_id in train_ids]
    val_records = [r for r in records if r.participant_id in val_ids]

    logger.info(
        "Split: %d train participants, %d val participants",
        len(train_ids),
        len(val_ids),
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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


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
    lambda_contrastive: float = 0.5,
    nt_xent_temperature: float = 0.07,
) -> TransactionEncoder:
    """Train the transaction encoder with CE (archetype) + NT-Xent (individual identity).

    Split-history views: each participant's transactions are split at the
    median timestamp into oldest-half and most-recent-half. Both halves are
    encoded via the GRU; NT-Xent treats them as a positive pair.
    Archetype CE uses the full-sequence GRU embedding.

    Parameters
    ----------
    records
        Transaction records. If None, loads from data/synthetic/transactions.jsonl.
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

    if records is None:
        records = load_records()

    train_records, val_records = split_by_participant(records, seed=seed)

    vocab = TxVocabulary()
    vocab.save_vocab()

    encoder = TransactionEncoder(
        vocab=vocab,
        projection_dim=projection_dim,
        gru_hidden=gru_hidden,
        gru_layers=gru_layers,
        gru_dropout=gru_dropout,
    ).to(torch_device)

    # Archetype classification head (trained jointly, discarded after)
    archetype_head = nn.Linear(gru_hidden, N_ARCHETYPE_CLASSES).to(torch_device)

    train_ds = SplitTransactionDataset(train_records, vocab)
    val_ds = SplitTransactionDataset(val_records, vocab)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    optimiser = torch.optim.AdamW(
        list(encoder.parameters()) + list(archetype_head.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    ce_criterion = nn.CrossEntropyLoss()

    with mlflow.start_run(run_name="transaction_encoder_v2_contrastive"):
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
                "lambda_contrastive": lambda_contrastive,
                "nt_xent_temperature": nt_xent_temperature,
                "objective": "ce+nt_xent_split_history",
            }
        )

        best_val_loss = float("inf")
        best_encoder_state: dict[str, torch.Tensor] | None = None
        patience_counter = 0

        for epoch in range(n_epochs):
            encoder.train()
            archetype_head.train()
            epoch_ce = 0.0
            epoch_nt = 0.0
            n_batches = 0

            for (
                full_t,
                full_l,
                h1_t,
                h1_l,
                h2_t,
                h2_l,
                labels,
                eligible,
            ) in train_loader:
                full_t = full_t.to(torch_device)
                full_l = full_l.to(torch_device)
                h1_t = h1_t.to(torch_device)
                h1_l = h1_l.to(torch_device)
                h2_t = h2_t.to(torch_device)
                h2_l = h2_l.to(torch_device)
                labels = labels.to(torch_device)

                # Archetype CE: recompute via encoder internals to get the pre-projection
                # gru_hidden-dim hidden state (archetype_head expects gru_hidden, not EMBEDDING_DIM).
                projected_full = torch.relu(encoder.token_proj(full_t))
                packed_full = pack_padded_sequence(
                    projected_full, full_l.cpu(), batch_first=True, enforce_sorted=True
                )
                _, hidden_full = encoder.gru(packed_full)
                gru_final_full = hidden_full[-1]  # (B, gru_hidden)
                arch_logits = archetype_head(gru_final_full)
                ce_loss = ce_criterion(arch_logits, labels)

                # NT-Xent on split-history views (eligible participants only)
                elig_mask = torch.tensor(eligible, device=torch_device)
                nt_loss = torch.tensor(0.0, device=torch_device)
                if elig_mask.sum() >= 2:
                    h1_e = h1_t[elig_mask]
                    h1_l_e = h1_l[elig_mask]
                    h2_e = h2_t[elig_mask]
                    h2_l_e = h2_l[elig_mask]

                    # Sort each half by descending length for pack_padded_sequence
                    o1 = torch.argsort(h1_l_e, descending=True)
                    o2 = torch.argsort(h2_l_e, descending=True)

                    emb_h1_sorted = _encode_sequence(encoder, h1_e[o1], h1_l_e[o1])
                    emb_h2_sorted = _encode_sequence(encoder, h2_e[o2], h2_l_e[o2])

                    # Re-align to original eligible order before NT-Xent
                    inv1 = torch.argsort(o1)
                    inv2 = torch.argsort(o2)
                    emb_h1 = emb_h1_sorted[inv1]
                    emb_h2 = emb_h2_sorted[inv2]

                    nt_loss = nt_xent_views(emb_h1, emb_h2, nt_xent_temperature)

                loss = ce_loss + lambda_contrastive * nt_loss

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                epoch_ce += ce_loss.item()
                epoch_nt += nt_loss.item()
                n_batches += 1

            avg_ce = epoch_ce / max(n_batches, 1)
            avg_nt = epoch_nt / max(n_batches, 1)

            # Validation: CE loss + accuracy on full-sequence archetype prediction
            encoder.eval()
            archetype_head.eval()
            val_ce = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for full_t, full_l, _, _, _, _, labels, _ in val_loader:
                    full_t = full_t.to(torch_device)
                    full_l = full_l.to(torch_device)
                    labels = labels.to(torch_device)

                    projected_full = torch.relu(encoder.token_proj(full_t))
                    packed_full = pack_padded_sequence(
                        projected_full,
                        full_l.cpu(),
                        batch_first=True,
                        enforce_sorted=True,
                    )
                    _, hidden_full = encoder.gru(packed_full)
                    gru_final = hidden_full[-1]
                    arch_logits = archetype_head(gru_final)

                    val_ce += ce_criterion(arch_logits, labels).item()
                    val_correct += (arch_logits.argmax(dim=1) == labels).sum().item()
                    val_total += labels.size(0)

            avg_val_ce = val_ce / max(n_batches, 1)
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

    _save_path = CHECKPOINT_PATHS["transaction"]
    _save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), _save_path)
    logger.info("Checkpoint saved to %s", _save_path)
    logger.info("Training complete. Best val loss: %.4f", best_val_loss)
    return encoder


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
