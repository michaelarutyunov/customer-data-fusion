"""
Trace encoder model — Transformer sequence encoder for MouseLab acquisition traces.

Architecture:
    Input sequence:  [T x 31]
        -> Token embedding (attribute/alternative/event_type learned embeddings + continuous features)
        -> Linear projection: [T x 64]
        -> + Positional encoding (learned, max_len=200)
        -> Transformer encoder: 4 heads, 3 layers, d_model=64, d_ff=256, dropout=0.1
        -> CLS token output: [64]
        -> Linear projection: [EMBEDDING_DIM]
        -> e_trace: [EMBEDDING_DIM]

The CLS token is prepended at index 0 using a learned embedding.
An auxiliary classification head (7-class softmax) is included for training
and discarded after training.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from schemas import EMBEDDING_DIM

from encoders.trace.tokeniser import MAX_SEQ_LEN, N_EVENT_TYPES

# Model hyperparameters (from SPEC.md)
D_MODEL: int = 64
D_FF: int = 256
N_HEADS: int = 4
N_LAYERS: int = 3
DROPOUT: float = 0.1

# Embedding dimensions per SPEC.md tokenisation
ATTR_EMBED_DIM: int = 16
ALT_EMBED_DIM: int = 8
EVENT_TYPE_EMBED_DIM: int = 4

# Number of continuous scalar features per token
# timestamp_norm + dwell_zscore + is_reinspection = 3
N_CONTINUOUS: int = 3

# §0.1 board width — the product-feature vector dimension for the choice head
# (bead b8b). Mirrors applications/choice/data.PRODUCT_DIM.
PRODUCT_FEATURE_DIM: int = 8


class _PositionalEncoding(nn.Module):
    """Learned positional encoding for sequences up to *max_len* positions."""

    def __init__(self, d_model: int, max_len: int = MAX_SEQ_LEN + 1) -> None:
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, S, D) — add positional embedding to each position."""
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pos_embed(positions)


class TraceEncoder(nn.Module):
    """
    Transformer encoder for MouseLab acquisition trace sequences.

    Takes tokenised input of shape (B, S, 31) where:
      - token[:, 0] = attribute vocab index (int -> 16-dim embedding)
      - token[:, 1] = alternative vocab index (int -> 8-dim embedding)
      - token[:, 2] = event_type vocab index (int -> 4-dim embedding)
      - token[:, 3] = timestamp_norm (float)
      - token[:, 4] = dwell_zscore (float)
      - token[:, 5] = is_reinspection (float)

    The first position (index 0) is the CLS token placeholder.  Its vocab
    indices are 0, which maps to a dedicated CLS embedding row.
    """

    def __init__(
        self,
        n_attributes: int = 9,  # 8 attrs + 1 for CLS/unseen
        n_alternatives: int = 8,  # 7 alts + 1 for CLS/unseen
        n_event_types: int = N_EVENT_TYPES,  # 5 types + 1 for CLS/unseen
        d_model: int = D_MODEL,
        d_ff: int = D_FF,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        dropout: float = DROPOUT,
        max_seq_len: int = MAX_SEQ_LEN,
        n_classes: int = 7,  # persona archetypes
    ) -> None:
        super().__init__()

        self.d_model = d_model

        # Learned embeddings for discrete features
        self.attribute_embed = nn.Embedding(n_attributes, ATTR_EMBED_DIM)
        self.alternative_embed = nn.Embedding(n_alternatives, ALT_EMBED_DIM)
        self.event_type_embed = nn.Embedding(n_event_types, EVENT_TYPE_EMBED_DIM)

        # Projection from token features to d_model
        # ATTR_EMBED_DIM + ALT_EMBED_DIM + EVENT_TYPE_EMBED_DIM + N_CONTINUOUS
        # = 16 + 8 + 4 + 3 = 31
        self.input_proj = nn.Linear(
            ATTR_EMBED_DIM + ALT_EMBED_DIM + EVENT_TYPE_EMBED_DIM + N_CONTINUOUS,
            d_model,
        )

        # Positional encoding
        self.pos_enc = _PositionalEncoding(d_model, max_len=max_seq_len + 1)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # CLS projection head
        self.cls_proj = nn.Linear(d_model, EMBEDDING_DIM)

        # Auxiliary classification head (discarded after training)
        self.classifier = nn.Linear(EMBEDDING_DIM, n_classes)

        # Auxiliary choice-prediction head (bead b8b, discarded after training).
        # For each (trial, slot): Linear(concat(trial_emb, 8-dim product vector))
        # -> scalar logit -> P(chosen). Trained jointly so the trial embedding
        # carries choice-relevant signal into the fused CDT.
        self.choice_head = nn.Linear(EMBEDDING_DIM + PRODUCT_FEATURE_DIM, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialisation for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def embed_tokens(self, tokens: Tensor) -> Tensor:
        """
        Convert raw token features into embedded representation.

        Parameters
        ----------
        tokens: (B, S, 31) — raw token features from tokeniser.

        Returns
        -------
        (B, S, d_model) — projected token embeddings.
        """
        # Extract discrete indices and continuous features
        attr_idx = tokens[:, :, 0].long()  # (B, S)
        alt_idx = tokens[:, :, 1].long()  # (B, S)
        event_type_idx = tokens[:, :, 2].long()  # (B, S)
        continuous = tokens[:, :, 3:6]  # (B, S, 3) — timestamp, dwell, reinspect

        # Lookup embeddings
        attr_emb = self.attribute_embed(attr_idx)  # (B, S, 16)
        alt_emb = self.alternative_embed(alt_idx)  # (B, S, 8)
        event_type_emb = self.event_type_embed(event_type_idx)  # (B, S, 4)

        # Concatenate all features
        combined = torch.cat(
            [attr_emb, alt_emb, event_type_emb, continuous], dim=-1
        )  # (B, S, 31)

        # Project to d_model
        return self.input_proj(combined)  # (B, S, d_model)

    def forward(
        self,
        tokens: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """
        Encode a batch of trace sequences into fixed-dimension embeddings.

        Parameters
        ----------
        tokens:
            FloatTensor (B, S, 31) — tokenised input from tokeniser.
        mask:
            BoolTensor (B, S) — True for real positions, False for padding.
            Converted to ``~mask`` for the ``src_key_padding_mask`` argument
            of the Transformer (True = ignore).

        Returns
        -------
        e_trace: (B, EMBEDDING_DIM) — trial-level embeddings.
        """
        # Embed tokens
        x = self.embed_tokens(tokens)  # (B, S, d_model)

        # Scale by sqrt(d_model) (standard transformer practice)
        x = x * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.pos_enc(x)

        # Transformer encoder with padding mask
        # PyTorch mask convention: True = position to IGNORE
        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = ~mask  # (B, S)

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Extract CLS token (position 0)
        cls_output = x[:, 0, :]  # (B, d_model)

        # Project to embedding dimension
        e_trace = self.cls_proj(cls_output)  # (B, EMBEDDING_DIM)

        return e_trace

    def forward_with_logits(
        self,
        tokens: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass that also returns classification logits.

        Returns
        -------
        e_trace: (B, EMBEDDING_DIM)
        logits: (B, n_classes) — auxiliary classification head output.
        """
        e_trace = self.forward(tokens, mask)
        logits = self.classifier(e_trace)
        return e_trace, logits
