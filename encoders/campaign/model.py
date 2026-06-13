"""
Campaign encoder model.

Self-attention encoder that processes campaign interaction sequences into
fixed-dimension customer embeddings. Uses a CLS token to aggregate the
sequence (same pattern as the trace Transformer).

Architecture:
    Input sequence: [T x 11]
        → Linear projection: [T x 32]
        → + Positional encoding (learned)
        → Transformer encoder: 2 heads, 2 layers, d_model=32, d_ff=128, dropout=0.1
        → CLS token output: [32]
        → Linear projection: [EMBEDDING_DIM]
        → e_campaign: [EMBEDDING_DIM]

Training: CE (archetype) + NT-Xent (individual identity).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from schemas import EMBEDDING_DIM
from encoders.campaign.features import MAX_EVENTS, TOKEN_DIM, CampaignVocabulary

D_MODEL: int = 32
D_FF: int = 128
N_HEADS: int = 2
N_LAYERS: int = 2
DROPOUT: float = 0.1


class _PositionalEncoding(nn.Module):
    """Learned positional encoding."""

    def __init__(self, d_model: int, max_len: int = MAX_EVENTS + 1) -> None:
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pos_embed(positions)


class CampaignEncoder(nn.Module):
    """Self-attention encoder for campaign interaction sequences.

    Parameters
    ----------
    vocab : CampaignVocabulary
        Learned embedding tables for campaign token features.
    d_model : int
        Transformer hidden dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer encoder layers.
    """

    def __init__(
        self,
        vocab: CampaignVocabulary | None = None,
        d_model: int = D_MODEL,
        d_ff: int = D_FF,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        dropout: float = DROPOUT,
        max_seq_len: int = MAX_EVENTS,
        n_classes: int = 7,
    ) -> None:
        super().__init__()
        self.vocab = vocab or CampaignVocabulary()
        self.d_model = d_model

        # Token projection: 11 -> 32
        self.input_proj = nn.Linear(TOKEN_DIM, d_model)

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

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def embed_tokens(self, tokens: Tensor) -> Tensor:
        """Project raw token features to d_model."""
        return self.input_proj(tokens)  # (B, S, d_model)

    def forward(
        self,
        tokens: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Encode a batch of campaign sequences into embeddings.

        Parameters
        ----------
        tokens : Tensor, shape (B, S, TOKEN_DIM)
            Tokenised campaign sequences (CLS already prepended at index 0).
        mask : Tensor, shape (B, S), optional
            True for real positions, False for padding.

        Returns
        -------
        Tensor, shape (B, EMBEDDING_DIM)
        """
        x = self.embed_tokens(tokens)
        x = x * math.sqrt(self.d_model)
        x = self.pos_enc(x)

        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = ~mask

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        cls_output = x[:, 0, :]  # (B, d_model)
        return self.cls_proj(cls_output)  # (B, EMBEDDING_DIM)

    def forward_with_logits(
        self,
        tokens: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass that also returns classification logits."""
        e_campaign = self.forward(tokens, mask)
        logits = self.classifier(e_campaign)
        return e_campaign, logits
