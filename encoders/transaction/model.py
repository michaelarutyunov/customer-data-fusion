"""
Transaction encoder model.

GRU-based encoder that processes 12-month purchase histories into
fixed-dimension participant embeddings. Uses the final hidden state of
the last GRU layer (not mean pooling) to preserve recency bias.

Architecture:
    Input sequence: [B x T x 20]
           ↓
    Linear projection: [B x T x 64]
           ↓
    GRU: hidden_size=128, num_layers=2, dropout=0.1, batch_first=True
           ↓
    Final hidden state (last layer): [B x 128]
           ↓
    Linear projection + LayerNorm: [B x 128]
           ↓
    e_transaction: [B x EMBEDDING_DIM]
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from schemas import EMBEDDING_DIM
from encoders.transaction.features import TOKEN_DIM, TxVocabulary


class TransactionEncoder(nn.Module):
    """GRU encoder for participant transaction histories.

    Parameters
    ----------
    vocab : TxVocabulary
        Learned embedding tables for categorical token features.
    projection_dim : int
        Intermediate dimension after the linear projection of token vectors.
    gru_hidden : int
        GRU hidden size per direction.
    gru_layers : int
        Number of stacked GRU layers.
    gru_dropout : float
        Dropout between GRU layers (applied when gru_layers > 1).
    """

    def __init__(
        self,
        vocab: TxVocabulary | None = None,
        projection_dim: int = 64,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        gru_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab = vocab or TxVocabulary()
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers

        # Token projection: 20 -> 64
        self.token_proj = nn.Linear(TOKEN_DIM, projection_dim)

        # GRU encoder
        self.gru = nn.GRU(
            input_size=projection_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
        )

        # Output projection: gru_hidden -> EMBEDDING_DIM with LayerNorm
        self.output_proj = nn.Sequential(
            nn.Linear(gru_hidden, EMBEDDING_DIM),
            nn.LayerNorm(EMBEDDING_DIM),
        )

    def forward(
        self,
        token_seqs: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Encode batched transaction sequences.

        Parameters
        ----------
        token_seqs : Tensor, shape (B, T, 20)
            Pre-tokenised transaction sequences. Must be sorted by
            descending length (requirement of pack_padded_sequence).
        lengths : Tensor, shape (B,)
            Actual sequence length per participant (before padding).

        Returns
        -------
        Tensor, shape (B, EMBEDDING_DIM)
            Participant-level transaction embeddings.
        """
        B, T, _ = token_seqs.shape

        # Project tokens: (B, T, 20) -> (B, T, 64)
        projected = torch.relu(self.token_proj(token_seqs))

        # Pack sequences to avoid processing padding tokens
        packed = pack_padded_sequence(
            projected,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=True,
        )

        # GRU forward
        _, hidden = self.gru(packed)
        # hidden shape: (num_layers, B, gru_hidden)
        # Take final hidden state of the last layer
        final_hidden = hidden[-1]  # (B, gru_hidden)

        # Project to embedding dimension
        return self.output_proj(final_hidden)  # (B, EMBEDDING_DIM)

    def get_brand_tier_prediction_head(self) -> nn.Linear:
        """Create a next-brand_tier prediction head.

        Returns an nn.Linear that maps gru_hidden -> 4 (brand_tier classes).
        This head is trained with the self-supervised objective and
        discarded after training.
        """
        return nn.Linear(self.gru_hidden, 4)


class NextBrandTierHead(nn.Module):
    """Prediction head for next brand_tier self-supervised training.

    Takes GRU hidden states and predicts the brand_tier of the next
    transaction in the sequence.
    """

    def __init__(self, gru_hidden: int = 128, n_classes: int = 4) -> None:
        super().__init__()
        self.head = nn.Linear(gru_hidden, n_classes)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Predict next brand_tier from GRU hidden states.

        Parameters
        ----------
        hidden : Tensor, shape (B, gru_hidden)
            GRU hidden states at each step.

        Returns
        -------
        Tensor, shape (B, n_classes)
            Raw logits for brand_tier classes.
        """
        return self.head(hidden)
