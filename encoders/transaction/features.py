"""
Transaction feature tokenisation.

Converts TransactionRecord objects into 20-dim token vectors using
learned embeddings for categorical features and normalised continuous features.

Token dimension breakdown (8 + 4 + 4 + 1 + 1 + 1 + 1 = 20):
  brand_tier_embed     8  (learned, vocab = {premium, mid, value, own_label})
  channel_embed        4  (learned, vocab = Channel enum values)
  purchase_type_embed  4  (learned, vocab = PurchaseType enum values)
  price_paid_norm      1  (direct from record)
  on_promotion         1  (0.0 or 1.0)
  quantity_norm        1  (min(quantity / 5.0, 1.0))
  recency_norm         1  (1.0 - days_before_session / 365.0)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

from schemas.transaction import Channel, PurchaseType, TransactionRecord

# Canonical vocabularies — stable across train/test
BRAND_TIER_VOCAB: list[str] = ["premium", "mid", "value", "own_label"]
CHANNEL_VOCAB: list[str] = [c.value for c in Channel]
PURCHASE_TYPE_VOCAB: list[str] = [p.value for p in PurchaseType]

TOKEN_DIM: int = 20  # 8 + 4 + 4 + 1 + 1 + 1 + 1
MAX_SEQ_LEN: int = 80

DATA_DIR = Path("data/synthetic")
VOCAB_FILE = DATA_DIR / "tx_vocab.json"


class TxVocabulary(nn.Module):
    """Learned embedding tables for transaction categorical features.

    Holds nn.Embedding modules for brand_tier, channel, and purchase_type.
    The vocabulary mapping (string -> index) is fixed; the embedding weights
    are trained jointly with the GRU encoder.
    """

    def __init__(self) -> None:
        super().__init__()
        self.brand_tier_to_idx = {v: i for i, v in enumerate(BRAND_TIER_VOCAB)}
        self.channel_to_idx = {v: i for i, v in enumerate(CHANNEL_VOCAB)}
        self.purchase_type_to_idx = {v: i for i, v in enumerate(PURCHASE_TYPE_VOCAB)}

        self.brand_tier_embed = nn.Embedding(len(BRAND_TIER_VOCAB), 8)
        self.channel_embed = nn.Embedding(len(CHANNEL_VOCAB), 4)
        self.purchase_type_embed = nn.Embedding(len(PURCHASE_TYPE_VOCAB), 4)

    # ---- index lookups ---------------------------------------------------

    def brand_tier_index(self, value: str) -> int:
        return self.brand_tier_to_idx[value]

    def channel_index(self, value: Channel | str) -> int:
        key = value.value if isinstance(value, Channel) else value
        return self.channel_to_idx[key]

    def purchase_type_index(self, value: PurchaseType | str) -> int:
        key = value.value if isinstance(value, PurchaseType) else value
        return self.purchase_type_to_idx[key]

    # ---- tokenisation ----------------------------------------------------

    def to_token_vector(self, record: TransactionRecord) -> torch.Tensor:
        """Convert a single TransactionRecord into a 20-dim float tensor.

        Applies learned embeddings for categorical features and normalisation
        for continuous features.
        """
        brand_idx = torch.tensor(self.brand_tier_index(record.brand_tier))
        channel_idx = torch.tensor(self.channel_index(record.channel))
        ptype_idx = torch.tensor(self.purchase_type_index(record.purchase_type))

        brand_vec = self.brand_tier_embed(brand_idx)  # (8,)
        channel_vec = self.channel_embed(channel_idx)  # (4,)
        ptype_vec = self.purchase_type_embed(ptype_idx)  # (4,)

        price = torch.tensor([record.price_paid_normalised])
        promo = torch.tensor([float(record.on_promotion)])
        qty = torch.tensor([min(record.quantity / 5.0, 1.0)])
        recency = torch.tensor([1.0 - record.days_before_session / 365.0])

        return torch.cat(
            [brand_vec, channel_vec, ptype_vec, price, promo, qty, recency]
        )

    # ---- batch helpers ---------------------------------------------------

    def encode_sequence(self, records: Sequence[TransactionRecord]) -> torch.Tensor:
        """Encode a sorted sequence of records into a (T, 20) tensor.

        Records should already be sorted most-recent-first (descending
        days_before_session).
        """
        tokens = [self.to_token_vector(r) for r in records]
        return torch.stack(tokens)  # (T, 20)

    # ---- persistence -----------------------------------------------------

    def save_vocab(self, path: Path | str | None = None) -> Path:
        """Save vocabulary mappings (not embedding weights) to JSON."""
        path = Path(path) if path else VOCAB_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "brand_tier": self.brand_tier_to_idx,
            "channel": self.channel_to_idx,
            "purchase_type": self.purchase_type_to_idx,
        }
        path.write_text(json.dumps(data, indent=2))
        return path

    @classmethod
    def load_vocab(cls, path: Path | str | None = None) -> TxVocabulary:
        """Load vocabulary mappings from JSON.

        Returns a TxVocabulary with the saved index mappings; embedding
        weights are initialised randomly (to be loaded from a checkpoint
        separately).
        """
        path = Path(path) if path else VOCAB_FILE
        data = json.loads(path.read_text())
        vocab = cls()
        vocab.brand_tier_to_idx = data["brand_tier"]
        vocab.channel_to_idx = data["channel"]
        vocab.purchase_type_to_idx = data["purchase_type"]
        return vocab


def sort_transactions_most_recent_first(
    records: Sequence[TransactionRecord],
) -> list[TransactionRecord]:
    """Sort transactions so that index 0 is the most recent.

    days_before_session is 1-indexed where 1 = yesterday, 365 = a year ago.
    Ascending sort puts the smallest value first = most recent first.
    """
    return sorted(records, key=lambda r: r.days_before_session)
