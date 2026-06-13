"""
Clickstream encoder model.

GRU-based encoder that processes web session event sequences into fixed-dimension
customer embeddings. Sessions are encoded individually, then mean-pooled to the
customer level.

Architecture:
    Per session: [T_events x 19] -> GRU -> session hidden state [128]
    Customer:    mean-pool session embeddings -> output_proj -> e_clickstream [EMBEDDING_DIM]

Training: CE (archetype) + NT-Xent (individual identity), same multi-task objective
as other encoders.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from schemas import EMBEDDING_DIM
from encoders.clickstream.features import TOKEN_DIM, ClickstreamVocabulary


class ClickstreamEncoder(nn.Module):
    """GRU encoder for clickstream session sequences.

    Parameters
    ----------
    vocab : ClickstreamVocabulary
        Learned embedding tables for categorical token features.
    projection_dim : int
        Intermediate dimension after token projection.
    gru_hidden : int
        GRU hidden size per layer.
    gru_layers : int
        Number of stacked GRU layers.
    """

    def __init__(
        self,
        vocab: ClickstreamVocabulary | None = None,
        projection_dim: int = 64,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        gru_dropout: float = 0.1,
        n_classes: int = 7,
    ) -> None:
        super().__init__()
        self.vocab = vocab or ClickstreamVocabulary()
        self.gru_hidden = gru_hidden

        # Token projection: 19 -> 64
        self.token_proj = nn.Linear(TOKEN_DIM, projection_dim)

        # GRU encoder for sessions
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

        # Auxiliary classification head (discarded after training)
        self.classifier = nn.Linear(EMBEDDING_DIM, n_classes)

    def encode_session(
        self, session_tokens: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Encode a batch of sessions into session-level embeddings.

        Parameters
        ----------
        session_tokens : Tensor, shape (B_sessions, T_events, TOKEN_DIM)
        lengths : Tensor, shape (B_sessions,)
            Actual event count per session (before padding).

        Returns
        -------
        Tensor, shape (B_sessions, gru_hidden)
        """
        from torch.nn.utils.rnn import pack_padded_sequence

        projected = torch.relu(self.token_proj(session_tokens))
        packed = pack_padded_sequence(
            projected, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        return hidden[-1]  # (B_sessions, gru_hidden)

    def forward(
        self,
        session_embeddings: torch.Tensor,
        session_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Aggregate session embeddings to customer-level embedding.

        Parameters
        ----------
        session_embeddings : Tensor, shape (B_customers, N_sessions, gru_hidden)
            Pre-encoded session embeddings per customer.
        session_mask : Tensor, shape (B_customers, N_sessions), optional
            True for real sessions, False for padding.

        Returns
        -------
        Tensor, shape (B_customers, EMBEDDING_DIM)
        """
        if session_mask is not None:
            mask_f = session_mask.float().unsqueeze(-1)  # (B, N, 1)
            summed = (session_embeddings * mask_f).sum(dim=1)
            counts = mask_f.sum(dim=1).clamp(min=1.0)
            pooled = summed / counts  # masked mean
        else:
            pooled = session_embeddings.mean(dim=1)

        return self.output_proj(pooled)  # (B, EMBEDDING_DIM)

    def forward_with_logits(
        self,
        session_embeddings: torch.Tensor,
        session_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass that also returns classification logits."""
        e_clickstream = self.forward(session_embeddings, session_mask)
        logits = self.classifier(e_clickstream)
        return e_clickstream, logits
