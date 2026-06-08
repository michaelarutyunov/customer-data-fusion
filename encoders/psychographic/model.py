"""
Psychographic encoder — MLP projector for fixed-width survey vectors.

Architecture:
    Input(19) -> Linear(19, 64) -> ReLU -> Dropout(0.2)
              -> Linear(64, EMBEDDING_DIM) -> ReLU
              -> LayerNorm(EMBEDDING_DIM)

Includes a supervised classification head for persona archetype prediction
(used during training, discarded at inference).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from schemas import EMBEDDING_DIM

from encoders.psychographic.features import FEATURE_DIM


class PsychographicEncoder(nn.Module):
    """MLP encoder that maps 19-dim psychographic feature vectors to
    ``EMBEDDING_DIM``-dimensional embeddings.

    The forward pass returns the embedding tensor. Use ``forward_with_logits``
    during training to also get persona classification logits.
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        n_classes: int = 7,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, EMBEDDING_DIM),
            nn.ReLU(),
            nn.LayerNorm(EMBEDDING_DIM),
        )
        # Classification head — used for supervised training only
        self.classifier = nn.Linear(EMBEDDING_DIM, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return embedding of shape ``(batch_size, EMBEDDING_DIM)``."""
        return self.encoder(x)

    def forward_with_logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (embedding, classification_logits).

        ``embedding`` has shape ``(batch_size, EMBEDDING_DIM)``.
        ``logits`` has shape ``(batch_size, n_classes)``.
        """
        embedding = self.encoder(x)
        logits = self.classifier(embedding)
        return embedding, logits
