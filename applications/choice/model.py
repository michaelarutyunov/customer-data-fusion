"""Choice model module — two-tower CDT + Product → P(choose)."""

from __future__ import annotations

import torch
import torch.nn as nn

from schemas import EMBEDDING_DIM


class ChoiceModel(nn.Module):
    """
    Two-tower choice model: [CDT, Product] → P(choose_this_product).

    Consumer tower: CDT[128] → Linear(128, 64) → ReLU → Dropout(0.1)
    Product tower: Product[D] → Linear(D, 64) → ReLU → Dropout(0.1)
    Joint: Concat[128] → Linear(128, 1) → Sigmoid → P(choose)

    This architecture enables M1 to predict consumer choices from their CDT embedding
    and product features, with the critical CDT lift gate proving that decision process
    (trace-dominated CDT) predicts choices better than product features alone.
    """

    def __init__(
        self,
        cdt_dim: int = EMBEDDING_DIM,  # 128
        product_dim: int = 8,  # full §0.1 board (bead 6ca)
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.consumer_tower = nn.Sequential(
            nn.Linear(cdt_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.product_tower = nn.Sequential(
            nn.Linear(product_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.joint = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, cdt_embedding, product_features):
        """
        Args:
            cdt_embedding: [B, 128] CDT embedding from frozen fusion
            product_features: [B, D] product features (price, quality, etc.)

        Returns:
            [B, 1] probability of choosing this product
        """
        cdt_feat = self.consumer_tower(cdt_embedding)  # [B, 64]
        prod_feat = self.product_tower(product_features)  # [B, 64]

        joint = torch.cat([cdt_feat, prod_feat], dim=1)  # [B, 128]
        prob = self.joint(joint)  # [B, 1]

        return prob