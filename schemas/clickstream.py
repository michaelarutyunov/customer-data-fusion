"""
Clickstream schema — web session event log and session summaries.

Provides dense temporal signal on browsing behaviour. Each ClickstreamEvent is
one token in the clickstream encoder input. SessionSummary aggregates per-session
statistics for auxiliary tasks.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ClickstreamEventType(str, Enum):
    """Event types recorded in a web clickstream session."""

    PAGE_VIEW = "page_view"
    PRODUCT_VIEW = "product_view"
    ADD_TO_CART = "add_to_cart"
    REMOVE_FROM_CART = "remove_from_cart"
    SEARCH = "search"
    FILTER_APPLY = "filter_apply"
    CHECKOUT_START = "checkout_start"
    PURCHASE = "purchase"


class PageType(str, Enum):
    """Page categories in the web session navigation graph."""

    HOME = "home"
    CATEGORY = "category"
    PRODUCT = "product"
    SEARCH_RESULTS = "search_results"
    CART = "cart"
    CHECKOUT = "checkout"


class ReferrerType(str, Enum):
    """Traffic source for the session."""

    DIRECT = "direct"
    ORGANIC = "organic"
    PAID_SEARCH = "paid_search"
    EMAIL = "email"
    SOCIAL = "social"


class DeviceType(str, Enum):
    """Device used for the session."""

    DESKTOP = "desktop"
    MOBILE = "mobile"
    TABLET = "tablet"


class SessionIntent(str, Enum):
    """Inferred browsing intent for a web session."""

    BROWSE = "browse"
    RESEARCH = "research"
    BUY = "buy"


@dataclass(frozen=True)
class ClickstreamEvent:
    """
    A single event in a web clickstream session.

    One token in the clickstream sequence encoder input. Events within a session
    are ordered by event_ts. Dwell time follows a log-normal distribution.
    """

    customer_id: str
    session_id: str
    event_ts: str  # ISO 8601 datetime string
    event_type: ClickstreamEventType
    page_type: PageType
    sku_viewed: Optional[str] = None
    referrer: ReferrerType = ReferrerType.DIRECT
    device: DeviceType = DeviceType.DESKTOP
    dwell_ms: float = 0.0
    month: int = 0


@dataclass(frozen=True)
class SessionSummary:
    """
    Aggregated statistics for a single web session.

    Links to ClickstreamEvent records via session_id. Used for auxiliary
    prediction tasks and session-level analysis.
    """

    customer_id: str
    session_id: str
    n_events: int
    session_duration_s: float
    intent: SessionIntent
    device: DeviceType
    month: int = 0
