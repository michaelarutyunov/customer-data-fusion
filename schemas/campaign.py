"""
Campaign schema — email/push marketing interaction log.

Provides intervention-response signal for the campaign encoder.
Each CampaignEvent records whether a customer opened, clicked, converted,
or unsubscribed from a dispatched campaign.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CampaignType(str, Enum):
    """Campaign dispatch categories."""

    PROMOTION = "promotion"
    NEWSLETTER = "newsletter"
    RE_ENGAGEMENT = "re_engagement"
    LOYALTY = "loyalty"
    NEW_PRODUCT = "new_product"


@dataclass(frozen=True)
class CampaignEvent:
    """
    A single campaign interaction event for one customer.

    Funnel: sent -> opened -> clicked -> converted.
    Unsub is a terminal action — once True, no further campaigns are sent.

    discount_pct: fraction off (0.0–0.5). Non-zero only for PROMOTION and LOYALTY.
    sent_ts: ISO 8601 datetime string (e.g. "2025-03-15T09:30:00Z").
    month: 1-based month index for temporal partitioning.
    """

    customer_id: str
    campaign_id: str
    sent_ts: str  # ISO datetime string
    campaign_type: CampaignType
    discount_pct: float  # 0.0–0.5
    category: str
    opened: bool = False
    clicked: bool = False
    converted: bool = False
    unsub: bool = False
    month: int = 0
    participant_id: str = ""  # individual consumer this campaign targeted
