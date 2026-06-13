"""
Transaction schema — purchase history records.

Provides preference magnitude and price sensitivity calibration signal
that process traces alone cannot supply.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Channel(str, Enum):
    ONLINE = "online"
    IN_STORE = "in_store"
    CLICK_AND_COLLECT = "click_and_collect"


class PurchaseType(str, Enum):
    PLANNED = "planned"  # intended purchase
    IMPULSE = "impulse"  # unplanned
    HABITUAL = "habitual"  # repeat without deliberation
    DEAL_DRIVEN = "deal_driven"  # triggered by promotion


class PaymentMethod(str, Enum):
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    PAYPAL = "paypal"
    CASH = "cash"
    BNPL = "bnpl"  # buy now, pay later


@dataclass(frozen=True)
class TransactionRecord:
    """
    A single purchase event in a participant's transaction history.
    Lookback window: 12 months from session date.

    price_paid_normalised: price as a percentile (0–1) within category
    range observed in the synthetic market, enabling cross-category
    comparison without raw price leakage.
    """

    participant_id: str
    transaction_id: str
    days_before_session: int  # 1–365; relative to process trace session date
    category: str
    product_id: str  # anonymised product identifier
    brand_tier: str  # "premium", "mid", "value", "own_label"
    price_paid_normalised: float  # 0–1 percentile within category
    quantity: int
    channel: Channel
    purchase_type: PurchaseType
    on_promotion: bool
    persona_id: str  # ground truth archetype (synthetic data only)
    sku: str = ""  # realistic order-line identifier: SKU-{category}-{tier}-{seq}; populated by Phase 2c
    unit_price: float = 0.0  # absolute price; populated by Phase 2c
    discount_applied: Optional[float] = (
        None  # 0.0–0.3 discount fraction; None if no discount
    )
    payment_method: PaymentMethod = PaymentMethod.CREDIT_CARD
    loyalty_card: Optional[bool] = None  # retailer loyalty programme membership
