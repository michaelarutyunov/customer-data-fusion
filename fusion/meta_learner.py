"""
fusion/meta_learner.py

Late fusion meta-learner for Consumer Digital Twin embedding.

Architecture (Phase 2, default v0.1):
    [B, 512] → Linear(512, 256) → LayerNorm(256) → GELU → Dropout(0.2)
           → Linear(256, 128) → LayerNorm(128) → GELU → Dropout(0.1)
           → Linear(128, 7) [classification head]

Phase 1 (logistic baseline):
    [B, 512] → Linear(512, 7) [single layer, no hidden layers]

The 128-dim output of the second hidden layer (Phase 2) is the CDT embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LateFusionMetaLearner(nn.Module):
    """Late fusion meta-learner for multimodal behavioural embeddings.

    Combines four frozen modality encoder outputs (each 128-dim) into a
    single consumer behavioural embedding (CDT embedding, 128-dim) and
    predicts persona archetype (7-class classification).

    Parameters
    ----------
    input_dim : int
        Dimension of concatenated modality embeddings (default: 512 = 4 × 128).
    hidden_dim : int
        Dimension of first hidden layer (default: 256).
    embed_dim : int
        Dimension of CDT embedding output (default: 128).
    n_classes : int
        Number of persona archetypes (default: 7).
    dropout : float
        Dropout probability for hidden layers (default: 0.2).
    phase : {"1", "2"}
        Architecture phase: "1" = logistic baseline, "2" = shallow MLP (default).
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        n_classes: int = 7,
        dropout: float = 0.2,
        phase: str = "2",
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.n_classes = n_classes
        self.dropout = dropout
        self.phase = phase

        if phase == "1":
            # Phase 1: Logistic regression baseline
            # [B, 512] → Linear(512, 7) → [B, 7]
            self.classifier = nn.Linear(input_dim, n_classes)
        elif phase == "2":
            # Phase 2: Shallow MLP meta-learner (default)
            # [B, 512] → Linear(512, 256) → LayerNorm(256) → GELU → Dropout(0.2)
            #         → Linear(256, 128) → LayerNorm(128) → GELU → Dropout(0.1)
            #         → Linear(128, 7) [classification head]

            # First hidden layer
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.dropout1 = nn.Dropout(dropout)

            # Second hidden layer (produces CDT embedding)
            self.fc2 = nn.Linear(hidden_dim, embed_dim)
            self.ln2 = nn.LayerNorm(embed_dim)
            self.dropout2 = nn.Dropout(dropout / 2)  # 0.1 if dropout=0.2

            # Classification head
            self.classifier = nn.Linear(embed_dim, n_classes)
        else:
            raise ValueError(f"Invalid phase: {phase}. Must be '1' or '2'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for training.

        Parameters
        ----------
        x : torch.Tensor
            Concatenated modality embeddings, shape [B, 512].

        Returns
        -------
        torch.Tensor
            Classification logits, shape [B, 7].
        """
        if self.phase == "1":
            # Phase 1: Logistic regression (single linear layer)
            return self.classifier(x)
        else:
            # Phase 2: Shallow MLP
            # First hidden layer
            h1 = self.fc1(x)
            h1 = self.ln1(h1)
            h1 = F.gelu(h1)
            h1 = self.dropout1(h1)

            # Second hidden layer (CDT embedding)
            h2 = self.fc2(h1)
            h2 = self.ln2(h2)
            h2 = F.gelu(h2)
            h2 = self.dropout2(h2)

            # Classification head
            logits = self.classifier(h2)
            return logits

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CDT embedding (no classification head).

        Parameters
        ----------
        x : torch.Tensor
            Concatenated modality embeddings, shape [B, 512].

        Returns
        -------
        torch.Tensor
            CDT embedding, shape [B, 128].
            For Phase 1 (logistic regression), returns the input.
            For Phase 2 (MLP), returns the second hidden layer.
        """
        if self.phase == "1":
            # Phase 1 has no embedding layer — return input as-is
            # Note: this is not a true embedding, but provides interface compatibility
            return x
        else:
            # Phase 2: Extract second hidden layer
            h1 = self.fc1(x)
            h1 = self.ln1(h1)
            h1 = F.gelu(h1)
            h1 = self.dropout1(h1)

            h2 = self.fc2(h1)
            h2 = self.ln2(h2)
            h2 = F.gelu(h2)
            # No dropout before returning embedding
            return h2

    def forward_with_embedding(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return both logits and CDT embedding.

        Parameters
        ----------
        x : torch.Tensor
            Concatenated modality embeddings, shape [B, 512].

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (logits, embedding) where logits has shape [B, 7] and
            embedding has shape [B, 128].
        """
        if self.phase == "1":
            logits = self.classifier(x)
            return logits, x
        else:
            h1 = self.fc1(x)
            h1 = self.ln1(h1)
            h1 = F.gelu(h1)
            h1 = self.dropout1(h1)

            h2 = self.fc2(h1)
            h2 = self.ln2(h2)
            h2 = F.gelu(h2)
            embedding = h2  # No dropout for embedding

            logits = self.classifier(self.dropout2(embedding))
            return logits, embedding
