"""
Campaign generator — dispatches marketing campaigns with sigmoid response model.

Generates a sequence of CampaignEvent records per customer over n_months.
Campaign dispatch frequency is 4-8 per month. Type distribution is weighted
by customer archetype (e.g. price_lex gets more PROMOTION, brand_affect gets
more LOYALTY).

Response model uses sigmoid(logit(base_rate) + beta * z_relevant) with
pinned base rates per campaign type. The funnel is sequential: opened ->
clicked -> converted. Each stage is conditionally dependent on the previous.

Unsub feedback: once unsub=True, no further campaigns are sent. Unsub
probability is 0.5-3% per campaign, conditioned on cumulative campaign
frequency (high-frequency customers are slightly more likely to unsub).

Public API:
    simulate_campaigns(config, n_months, month, random_seed) -> list[CampaignEvent]
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import structlog

from generator.persona_sampler import LatentDeviation
from schemas.campaign import CampaignEvent, CampaignType
from schemas.persona import PersonaConfig

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Base response rates per campaign type (pinned from bead spec)
# ---------------------------------------------------------------------------

_BASE_RATES: dict[CampaignType, dict[str, float]] = {
    CampaignType.PROMOTION: {
        "base_open": 0.30,
        "base_click": 0.10,
        "base_convert": 0.03,
    },
    CampaignType.NEWSLETTER: {
        "base_open": 0.25,
        "base_click": 0.05,
        "base_convert": 0.01,
    },
    CampaignType.RE_ENGAGEMENT: {
        "base_open": 0.20,
        "base_click": 0.08,
        "base_convert": 0.02,
    },
    CampaignType.LOYALTY: {
        "base_open": 0.40,
        "base_click": 0.08,
        "base_convert": 0.02,
    },
    CampaignType.NEW_PRODUCT: {
        "base_open": 0.25,
        "base_click": 0.12,
        "base_convert": 0.02,
    },
}

# ---------------------------------------------------------------------------
# z conditioning: which latent axis drives which funnel stage per campaign type
# ---------------------------------------------------------------------------

# opened: z_relevant mapping per campaign type
_Z_OPEN: dict[CampaignType, str] = {
    CampaignType.LOYALTY: "brand_lean",
    CampaignType.NEWSLETTER: "brand_lean",
    CampaignType.PROMOTION: "price_lean",
    # RE_ENGAGEMENT and NEW_PRODUCT use unconditioned base rate
}

# clicked: z_relevant mapping per campaign type
_Z_CLICK: dict[CampaignType, str] = {
    CampaignType.NEW_PRODUCT: "openness",
    CampaignType.LOYALTY: "brand_lean",
}

# converted: z_relevant mapping per campaign type
_Z_CONVERT: dict[CampaignType, str] = {
    CampaignType.PROMOTION: "price_lean",
    CampaignType.LOYALTY: "brand_lean",
}

_BETA: float = 0.5  # z conditioning strength

# ---------------------------------------------------------------------------
# Discount distributions per campaign type
# ---------------------------------------------------------------------------

_DISCOUNT_RANGE: dict[CampaignType, tuple[float, float]] = {
    CampaignType.PROMOTION: (0.10, 0.30),
    CampaignType.LOYALTY: (0.05, 0.15),
    # All others: 0.0 (no discount)
}

# ---------------------------------------------------------------------------
# Archetype-weighted campaign type dispatch
# ---------------------------------------------------------------------------

# Weights: how likely each archetype is to receive each campaign type.
# Higher weight = more campaigns of that type.
# E.g. price_lex gets more PROMOTION, brand_affect gets more LOYALTY.
_DISPATCH_WEIGHTS: dict[str, dict[CampaignType, float]] = {
    "price_lex": {
        CampaignType.PROMOTION: 0.40,
        CampaignType.NEWSLETTER: 0.15,
        CampaignType.RE_ENGAGEMENT: 0.10,
        CampaignType.LOYALTY: 0.05,
        CampaignType.NEW_PRODUCT: 0.30,
    },
    "quality_lex": {
        CampaignType.PROMOTION: 0.10,
        CampaignType.NEWSLETTER: 0.25,
        CampaignType.RE_ENGAGEMENT: 0.10,
        CampaignType.LOYALTY: 0.20,
        CampaignType.NEW_PRODUCT: 0.35,
    },
    "compensatory": {
        CampaignType.PROMOTION: 0.25,
        CampaignType.NEWSLETTER: 0.20,
        CampaignType.RE_ENGAGEMENT: 0.10,
        CampaignType.LOYALTY: 0.15,
        CampaignType.NEW_PRODUCT: 0.30,
    },
    "satisficer": {
        CampaignType.PROMOTION: 0.20,
        CampaignType.NEWSLETTER: 0.20,
        CampaignType.RE_ENGAGEMENT: 0.20,
        CampaignType.LOYALTY: 0.15,
        CampaignType.NEW_PRODUCT: 0.25,
    },
    "brand_affect": {
        CampaignType.PROMOTION: 0.05,
        CampaignType.NEWSLETTER: 0.15,
        CampaignType.RE_ENGAGEMENT: 0.10,
        CampaignType.LOYALTY: 0.50,
        CampaignType.NEW_PRODUCT: 0.20,
    },
    "adaptive": {
        CampaignType.PROMOTION: 0.20,
        CampaignType.NEWSLETTER: 0.20,
        CampaignType.RE_ENGAGEMENT: 0.15,
        CampaignType.LOYALTY: 0.15,
        CampaignType.NEW_PRODUCT: 0.30,
    },
    "low_involve": {
        CampaignType.PROMOTION: 0.30,
        CampaignType.NEWSLETTER: 0.15,
        CampaignType.RE_ENGAGEMENT: 0.25,
        CampaignType.LOYALTY: 0.05,
        CampaignType.NEW_PRODUCT: 0.25,
    },
}

# Product categories (shared with transaction simulator)
_CATEGORIES = ["electronics", "household", "personal"]

# ---------------------------------------------------------------------------
# Helper: sigmoid response conditioned on z
# ---------------------------------------------------------------------------


def _get_z_axis(z: Optional[LatentDeviation], axis_name: str) -> float:
    """Extract a named latent axis from z. Returns 0.0 if z is None."""
    if z is None:
        return 0.0
    return getattr(z, axis_name, 0.0)


def _sigmoid_response(
    base_rate: float,
    z_relevant: float,
    beta: float = _BETA,
) -> float:
    """Compute sigmoid(logit(base_rate) + beta * z_relevant).

    Clamps base_rate to [0.001, 0.999] to avoid log(0).
    """
    base_clamped = max(0.001, min(0.999, base_rate))
    logit_base = math.log(base_clamped / (1.0 - base_clamped))
    prob = 1.0 / (1.0 + math.exp(-(logit_base + beta * z_relevant)))
    return float(max(0.0, min(1.0, prob)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_campaigns(
    config: PersonaConfig,
    n_months: int = 12,
    month: int = 1,
    random_seed: Optional[int] = None,
) -> list[CampaignEvent]:
    """Generate campaign interaction events for one customer.

    Args:
        config: PersonaConfig for this customer (contains latent z).
        n_months: Number of months to simulate.
        month: Starting month index (1-based).
        random_seed: Seed for reproducibility.

    Returns:
        List of CampaignEvent records. Empty if customer unsubs early.
    """
    rng = np.random.default_rng(random_seed or config.random_seed or 42)
    customer_id = config.persona_id
    z = config.latent

    # Resolve archetype dispatch weights (fallback to uniform if unknown archetype)
    weights_dict = _DISPATCH_WEIGHTS.get(config.persona_id)
    if weights_dict is None:
        weights_dict = {ct: 1.0 for ct in CampaignType}

    campaign_types = list(weights_dict.keys())
    weights = np.array([weights_dict[ct] for ct in campaign_types])
    weights = weights / weights.sum()  # normalise

    events: list[CampaignEvent] = []
    has_unsubscribed = False
    campaign_counter = 0

    # Base date for timestamp generation (start of simulation)
    base_date = datetime(2025, 1, 1, tzinfo=timezone.utc)

    for m in range(month, month + n_months):
        if has_unsubscribed:
            break

        # 4-8 campaigns per month (uniform integer)
        n_campaigns_this_month = int(rng.integers(4, 9))

        for _ in range(n_campaigns_this_month):
            if has_unsubscribed:
                break

            campaign_counter += 1

            # Dispatch campaign type from archetype-weighted distribution
            ct_idx = rng.choice(len(campaign_types), p=weights)
            campaign_type = campaign_types[ct_idx]

            # Sample category uniformly
            category = rng.choice(_CATEGORIES)

            # Discount
            discount_range = _DISCOUNT_RANGE.get(campaign_type)
            if discount_range is not None:
                discount_pct = float(rng.uniform(discount_range[0], discount_range[1]))
            else:
                discount_pct = 0.0

            # Timestamp: random day/time within the month
            day_offset = int(rng.integers(0, 28))
            hour = int(rng.integers(6, 22))
            minute = int(rng.integers(0, 60))
            sent_dt = base_date + timedelta(
                days=(m - 1) * 30 + day_offset, hours=hour, minutes=minute
            )
            sent_ts = sent_dt.isoformat()

            # Campaign ID
            campaign_id = f"CAMP-{campaign_type.value}-{campaign_counter:04d}"

            # --- Response model (sequential funnel) ---
            rates = _BASE_RATES[campaign_type]

            # Opened
            z_open_axis = _Z_OPEN.get(campaign_type)
            z_open = _get_z_axis(z, z_open_axis) if z_open_axis else 0.0
            p_open = _sigmoid_response(rates["base_open"], z_open)
            opened = bool(rng.random() < p_open)

            # Clicked (only if opened)
            clicked = False
            if opened:
                z_click_axis = _Z_CLICK.get(campaign_type)
                z_click = _get_z_axis(z, z_click_axis) if z_click_axis else 0.0
                p_click = _sigmoid_response(rates["base_click"], z_click)
                clicked = bool(rng.random() < p_click)

            # Converted (only if clicked)
            converted = False
            if clicked:
                z_convert_axis = _Z_CONVERT.get(campaign_type)
                z_convert = _get_z_axis(z, z_convert_axis) if z_convert_axis else 0.0
                p_convert = _sigmoid_response(rates["base_convert"], z_convert)
                converted = bool(rng.random() < p_convert)

            # Unsub: 0.5-3% base, slightly higher for high-frequency customers
            # Scale unsub probability with cumulative campaign count
            freq_factor = min(1.0, campaign_counter / 50.0)  # ramps up over time
            p_unsub = 0.005 + 0.025 * freq_factor  # 0.5% → 3%
            unsub = bool(rng.random() < p_unsub)

            if unsub:
                has_unsubscribed = True

            events.append(
                CampaignEvent(
                    customer_id=customer_id,
                    campaign_id=campaign_id,
                    sent_ts=sent_ts,
                    campaign_type=campaign_type,
                    discount_pct=round(discount_pct, 4),
                    category=category,
                    opened=opened,
                    clicked=clicked,
                    converted=converted,
                    unsub=unsub,
                    month=m,
                )
            )

    n_months_simulated = 0
    if events:
        n_months_simulated = events[-1].month - month + 1

    logger.debug(
        "campaigns_simulated",
        customer_id=customer_id,
        n_events=len(events),
        n_months_simulated=n_months_simulated,
        has_unsubscribed=has_unsubscribed,
    )
    return events
