"""Campaign feature tokenisation.

Converts CampaignEvent sequences into fixed-dim token vectors using
learned embeddings for categorical features and normalised continuous features.

Token dimension breakdown (5 + 1 + 4 + 1 = 11):
  campaign_type_embed  5  (learned, vocab = 5 CampaignType values)
  discount_pct         1  (direct, 0.0–0.5)
  funnel_flags         4  (opened, clicked, converted, unsub as 0.0/1.0)
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from schemas.campaign import CampaignEvent, CampaignType

# Canonical vocabulary — stable across train/test
CAMPAIGN_TYPE_VOCAB: list[str] = [c.value for c in CampaignType]

TOKEN_DIM: int = 11  # 5 + 1 + 4
MAX_EVENTS: int = 50  # truncate to most recent campaigns per customer


class CampaignVocabulary(nn.Module):
    """Learned embedding tables for campaign categorical features."""

    def __init__(self) -> None:
        super().__init__()
        self.campaign_type_to_idx = {v: i for i, v in enumerate(CAMPAIGN_TYPE_VOCAB)}
        self.campaign_type_embed = nn.Embedding(len(CAMPAIGN_TYPE_VOCAB), 5)

    def _idx(self, value: object) -> int:
        key = value.value if hasattr(value, "value") else str(value)
        return self.campaign_type_to_idx.get(key, 0)

    def to_event_token(self, event: CampaignEvent) -> torch.Tensor:
        """Convert a single CampaignEvent into an 11-dim float tensor."""
        ct_idx = torch.tensor(self._idx(event.campaign_type))
        ct_vec = self.campaign_type_embed(ct_idx)  # (5,)

        discount = torch.tensor([float(event.discount_pct)])
        funnel = torch.tensor(
            [
                float(event.opened),
                float(event.clicked),
                float(event.converted),
                float(event.unsub),
            ]
        )

        return torch.cat([ct_vec, discount, funnel])  # (11,)

    def encode_sequence(self, events: Sequence[CampaignEvent]) -> torch.Tensor:
        """Encode a customer's campaign history into a (T, 11) tensor.

        Truncates to most recent MAX_EVENTS campaigns (chronological order).
        """
        # Most-recent-last ordering; keep last MAX_EVENTS
        truncated = list(events)[-MAX_EVENTS:]
        if not truncated:
            return torch.zeros(1, TOKEN_DIM)
        tokens = [self.to_event_token(e) for e in truncated]
        return torch.stack(tokens)  # (T, 11)
