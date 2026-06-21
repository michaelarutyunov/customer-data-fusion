"""
fusion/meta_learner.py

Late fusion meta-learner for Consumer Digital Twin embedding.

Architecture (Phase 2, default v0.1):
    [B, n_modalities*128] → Linear(→256) → LayerNorm → GELU → Dropout(0.2)
           → Linear(→128) → LayerNorm → GELU → Dropout(0.1)
           → Linear(→7) [classification head]

The 128-dim output of the second hidden layer (Phase 2) is the CDT embedding.

Supports variable modality count (default 6: trace + transaction + text +
psychographic + clickstream + campaign). Missing modalities (e.g., no traces
for customers outside the coverage subset) are replaced with a learned MISSING
embedding before concatenation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# §0.1 board width — product-feature vector dimension for the auxiliary choice
# head (experiment: choice loss at the fusion level). Mirrors
# applications/choice/data.PRODUCT_DIM / encoders/trace/model.PRODUCT_FEATURE_DIM.
PRODUCT_FEATURE_DIM: int = 8


class LateFusionMetaLearner(nn.Module):
    """Late fusion meta-learner for multimodal behavioural embeddings.

    Combines modality encoder outputs (each 128-dim) into a single consumer
    behavioural embedding (CDT embedding, 128-dim) and predicts persona
    archetype (7-class classification).

    Parameters
    ----------
    n_modalities : int
        Number of modality encoder inputs (default: 6). The concatenated input
        dimension is ``n_modalities * per_modality_dim``.
    per_modality_dim : int
        Dimension of each modality encoder output (default: 128 = EMBEDDING_DIM).
    input_dim : int, optional
        Explicit concatenated input dimension. Overrides the
        ``n_modalities * per_modality_dim`` computation if given.
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
    use_missing_embedding : bool
        If True, register a learnable MISSING embedding vector for absent
        modalities (partial coverage). Default: True.
    """

    def __init__(
        self,
        n_modalities: int = 6,
        per_modality_dim: int = 128,
        input_dim: int | None = None,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        n_classes: int = 7,
        dropout: float = 0.2,
        phase: str = "2",
        use_missing_embedding: bool = True,
    ) -> None:
        super().__init__()

        # Compute input_dim from n_modalities if not explicitly given
        if input_dim is None:
            input_dim = n_modalities * per_modality_dim

        self.input_dim = input_dim
        self.n_modalities = n_modalities
        self.per_modality_dim = per_modality_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.n_classes = n_classes
        self.dropout = dropout
        self.phase = phase

        # Learnable MISSING embedding: replaces absent modality outputs.
        # One vector per modality slot so each slot learns its own "missing" signal.
        self.use_missing_embedding = use_missing_embedding
        if use_missing_embedding:
            self.missing_embedding = nn.Parameter(
                torch.zeros(n_modalities, per_modality_dim)
            )
            nn.init.normal_(self.missing_embedding, mean=0.0, std=0.02)

        # Learnable TEMPORAL MISSING embedding: pads missing monthly observations.
        # Used when participant has < 12 months of transaction/clickstream data.
        self.temporal_missing_embedding = nn.Parameter(torch.zeros(per_modality_dim))
        nn.init.normal_(self.temporal_missing_embedding, mean=0.0, std=0.02)

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

        # Auxiliary choice-prediction head (experiment: choice loss at the
        # fusion level). For each (participant, slot): Linear(concat(CDT, 8-dim
        # product vector)) -> scalar logit -> P(chosen). Trained jointly so the
        # CDT itself carries choice-relevant signal. Discarded at inference
        # (only embed() is used downstream).
        self.choice_head = nn.Linear(embed_dim + PRODUCT_FEATURE_DIM, 1)

    def apply_missing_mask(
        self,
        modality_embeddings: list[torch.Tensor],
        presence_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Concatenate modality embeddings, replacing absent ones with MISSING.

        Parameters
        ----------
        modality_embeddings : list of Tensor, each [B, per_modality_dim]
            One tensor per modality (length must equal n_modalities).
        presence_mask : Tensor, shape [B, n_modalities], optional
            True where the modality is present, False where absent (use MISSING).
            If None, all modalities assumed present.

        Returns
        -------
        Tensor, shape [B, input_dim]
            Concatenated embeddings with MISSING substitution applied.
        """
        B = modality_embeddings[0].size(0)
        slots: list[torch.Tensor] = []
        for i, emb in enumerate(modality_embeddings):
            if presence_mask is not None:
                # Broadcast mask: [B, 1] for selecting per-modality
                present = presence_mask[:, i].unsqueeze(-1)  # [B, 1]
                missing_vec = self.missing_embedding[i].unsqueeze(0).expand(B, -1)
                slot = torch.where(present, emb, missing_vec)
            else:
                slot = emb
            slots.append(slot)
        return torch.cat(slots, dim=-1)  # [B, n_modalities * per_modality_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for training.

        Parameters
        ----------
        x : torch.Tensor
            Concatenated modality embeddings, shape [B, input_dim].

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
