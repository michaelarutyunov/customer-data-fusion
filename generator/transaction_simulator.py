"""
Transaction simulator — generates 12-month synthetic purchase history from PersonaConfig.

Price percentiles (price_paid_normalised) are drawn from a Beta distribution whose
parameters are tuned to persona price_sensitivity:
  - High price_sensitivity → Beta(1, 4) → concentrates mass near 0 (cheap end)
  - Low price_sensitivity  → Beta(4, 1) → concentrates mass near 1 (premium end)
  - Mid sensitivity        → Beta(2, 2) → roughly uniform / bell near 0.5

Channel is sampled proportionally from config.transactions.channel_mix.

Phase 2c additions:
  - Non-homogeneous Poisson process for inter-purchase timing (seasonality + z-conditioning)
  - Product catalog with 27 SKUs across 3 categories × 3 brand tiers
  - SKU choice via conditional logit utility model
  - Population of order-line fields: sku, unit_price, discount_applied, payment_method
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import structlog

from schemas.persona import PersonaConfig
from schemas.transaction import Channel, PaymentMethod, PurchaseType, TransactionRecord

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Product catalog: 3 categories × 3 tiers = 27 SKUs
# ---------------------------------------------------------------------------

_CATEGORIES = ["electronics", "household", "personal"]
_CATALOG_TIERS = ["budget", "mid", "premium"]  # tier names used in SKU IDs

# Base prices per (category, tier) — used to derive unit_price from price_paid_normalised
_BASE_PRICES: dict[str, dict[str, float]] = {
    "electronics": {"budget": 29.99, "mid": 79.99, "premium": 199.99},
    "household": {"budget": 9.99, "mid": 24.99, "premium": 59.99},
    "personal": {"budget": 4.99, "mid": 14.99, "premium": 34.99},
}

# Pre-built SKU catalog: list of (sku_id, category, tier, base_price)
_SKU_CATALOG: list[tuple[str, str, str, float]] = []
for _cat in _CATEGORIES:
    for _tier in _CATALOG_TIERS:
        for _seq in range(1, 4):  # 3 SKUs per (category, tier)
            _sku_id = f"SKU-{_cat}-{_tier}-{_seq:03d}"
            _base = _BASE_PRICES[_cat][_tier]
            _SKU_CATALOG.append((_sku_id, _cat, _tier, _base))

# Category indices for fast lookup
_CATALOG_ARR = np.array([(s, c, t, p) for s, c, t, p in _SKU_CATALOG], dtype=object)
_CATALOG_BASE_PRICES = np.array([item[3] for item in _SKU_CATALOG], dtype=float)
_CATALOG_TIERS_ARR = np.array([item[2] for item in _SKU_CATALOG])
_CATALOG_CATEGORIES_ARR = np.array([item[1] for item in _SKU_CATALOG])
_N_SKUS = len(_SKU_CATALOG)

# Tier indicator for conditional logit: 0=budget, 1=mid, 2=premium
_TIER_INDICATOR = np.array(
    [{"budget": 0, "mid": 1, "premium": 2}[t] for _, _, t, _ in _SKU_CATALOG],
    dtype=float,
)

# ---------------------------------------------------------------------------
# Brand tiers (legacy compatibility — still used for brand_tier field)
# ---------------------------------------------------------------------------

_BRAND_TIERS = ["premium", "mid", "value", "own_label"]

# Map channel_mix string keys → Channel enum
_CHANNEL_MAP: dict[str, Channel] = {
    "online": Channel.ONLINE,
    "in_store": Channel.IN_STORE,
    "click_and_collect": Channel.CLICK_AND_COLLECT,
}

# Payment method distributions per channel
_PAYMENT_DISTRIBUTIONS: dict[Channel, list[tuple[PaymentMethod, float]]] = {
    Channel.ONLINE: [
        (PaymentMethod.CREDIT_CARD, 0.60),
        (PaymentMethod.PAYPAL, 0.20),
        (PaymentMethod.DEBIT_CARD, 0.15),
        (PaymentMethod.BNPL, 0.05),
    ],
    Channel.IN_STORE: [
        (PaymentMethod.CASH, 0.40),
        (PaymentMethod.DEBIT_CARD, 0.35),
        (PaymentMethod.CREDIT_CARD, 0.25),
    ],
    Channel.CLICK_AND_COLLECT: [
        (PaymentMethod.CREDIT_CARD, 0.50),
        (PaymentMethod.DEBIT_CARD, 0.30),
        (PaymentMethod.PAYPAL, 0.20),
    ],
}

# ---------------------------------------------------------------------------
# Hazard-model constants
# ---------------------------------------------------------------------------

_LAMBDA_0 = 2.5  # base rate: purchases/month
_SEASONALITY_NORMAL = 1.0  # months 1-10
_SEASONALITY_HOLIDAY = 1.8  # months 11-12 (holiday peak)


def _price_beta_params(price_sensitivity: float) -> tuple[float, float]:
    """Return Beta(a, b) params so higher sensitivity → lower price percentile."""
    # Interpolate: sensitivity 0 → (4, 1), sensitivity 1 → (1, 4)
    a = 4.0 - 3.0 * price_sensitivity
    b = 1.0 + 3.0 * price_sensitivity
    return a, b


def _sample_brand_tier(rng: np.random.Generator, brand_loyalty: float) -> str:
    """
    High brand_loyalty → concentrate in 1–2 tiers (premium or mid).
    Low brand_loyalty  → spread across all 4 tiers uniformly.
    """
    loyal_weights = np.array([0.55, 0.35, 0.07, 0.03])
    flat_weights = np.array([0.25, 0.25, 0.25, 0.25])
    weights = brand_loyalty * loyal_weights + (1.0 - brand_loyalty) * flat_weights
    weights /= weights.sum()
    return str(rng.choice(_BRAND_TIERS, p=weights))


def _sample_purchase_type(
    rng: np.random.Generator,
    brand_loyalty: float,
    on_promotion: bool,
) -> PurchaseType:
    """
    High brand_loyalty → more HABITUAL.
    on_promotion=True  → more DEAL_DRIVEN.
    Otherwise          → mix of PLANNED / IMPULSE.
    """
    w = np.array([0.35, 0.20, 0.30, 0.15])
    w[2] += brand_loyalty * 0.30
    if on_promotion:
        w[3] += 0.25
    w = np.clip(w, 0, None)
    w /= w.sum()
    types = [
        PurchaseType.PLANNED,
        PurchaseType.IMPULSE,
        PurchaseType.HABITUAL,
        PurchaseType.DEAL_DRIVEN,
    ]
    return types[int(rng.choice(len(types), p=w))]


def _sample_payment_method(
    rng: np.random.Generator,
    channel: Channel,
) -> PaymentMethod:
    """Sample payment method from channel-dependent distribution."""
    dist = _PAYMENT_DISTRIBUTIONS[channel]
    methods = [m for m, _ in dist]
    probs = np.array([p for _, p in dist], dtype=float)
    probs /= probs.sum()
    return methods[int(rng.choice(len(methods), p=probs))]


def _involvement_score(config: PersonaConfig) -> float:
    """
    Derive involvement score from z.thoroughness (latent) or psychographic.involvement_score.
    Falls back to psychographic.involvement_score if no latent is set.
    """
    if config.latent is not None:
        # Project thoroughness (standardised deviation) into [0, 1] via logistic
        raw = config.latent.thoroughness
        return float(1.0 / (1.0 + math.exp(-raw)))
    return config.psychographic.involvement_score


def _seasonality(month: int) -> float:
    """Return seasonal multiplier for 1-indexed month (1=Jan, ..., 12=Dec)."""
    return _SEASONALITY_HOLIDAY if month in (11, 12) else _SEASONALITY_NORMAL


def _generate_purchase_occasions(
    rng: np.random.Generator,
    n_months: int,
    base_rate: float,
    involvement: float,
) -> list[int]:
    """
    Generate purchase occasions via non-homogeneous Poisson process.

    For each month, the rate is:
        lambda(t) = lambda_base * lambda_seasonal(t)
    where lambda_base = base_rate * (0.5 + involvement_score).

    Parameters
    ----------
    rng:
        NumPy random generator.
    n_months:
        Lookback window in months.
    base_rate:
        Persona's configured purchase_frequency_per_month (lambda_0).
    involvement:
        Involvement score in [0, 1] from z.thoroughness or psychographic.

    Returns a list of day-offsets (1–365) for each purchase occasion.
    """
    lambda_base = base_rate * (0.5 + involvement)
    occasions: list[int] = []
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    cumulative_day = 0
    for month_idx in range(min(n_months, 12)):
        month = month_idx + 1  # 1-indexed
        rate = lambda_base * _seasonality(month)
        n_in_month = int(rng.poisson(rate))
        days_in_month = days_per_month[month_idx]

        for _ in range(n_in_month):
            day_in_month = int(rng.integers(1, days_in_month + 1))
            absolute_day = cumulative_day + day_in_month
            occasions.append(absolute_day)

        cumulative_day += days_in_month

    # Convert absolute days to days_before_session (session is at end of window)
    total_days = sum(days_per_month[:n_months])
    days_before = [max(1, total_days - d + 1) for d in occasions]
    return days_before


def _choose_sku(
    rng: np.random.Generator,
    category: str,
    price_sensitivity: float,
    brand_loyalty: float,
) -> tuple[str, str, float]:
    """
    Choose a SKU via conditional logit model.

    Utility = beta_price * log(unit_price) + beta_brand * tier_indicator + epsilon

    beta_price is negative (from price_sensitivity), beta_brand is positive (from brand_loyalty).
    Returns (sku_id, tier, base_price).
    """
    # Filter catalog to the requested category
    cat_mask = _CATALOG_CATEGORIES_ARR == category
    indices = np.where(cat_mask)[0]

    if len(indices) == 0:
        # Fallback: use first 9 SKUs (first category)
        indices = np.arange(9)

    base_prices = _CATALOG_BASE_PRICES[indices]
    tier_indicators = _TIER_INDICATOR[indices]

    # Conditional logit coefficients
    beta_price = -(
        0.5 + price_sensitivity * 2.0
    )  # negative: higher price → lower utility
    beta_brand = (
        brand_loyalty * 1.5
    )  # positive: premium tier preferred by loyal customers

    # Deterministic utility
    log_prices = np.log(base_prices)
    utility = beta_price * log_prices + beta_brand * tier_indicators

    # Add Gumbel noise (epsilon) → softmax choice probabilities
    noise = rng.gumbel(0, 1, size=len(indices))
    perturbed_utility = utility + noise

    chosen_local = int(np.argmax(perturbed_utility))
    chosen_idx = indices[chosen_local]

    sku_id = _SKU_CATALOG[chosen_idx][0]
    tier = _SKU_CATALOG[chosen_idx][2]
    base_price = _SKU_CATALOG[chosen_idx][3]

    return sku_id, tier, base_price


def simulate_transactions(
    config: PersonaConfig,
    category: str = "electronics",
    n_months: int = 12,
    participant_id: str | None = None,
) -> list[TransactionRecord]:
    """
    Generate a synthetic 12-month purchase history for one participant.

    Uses a non-homogeneous Poisson process for inter-purchase timing with:
    - z-conditioned base rate (involvement from thoroughness)
    - Seasonal multiplier (1.8x holiday peak in months 11-12)

    Each purchase is populated with order-line fields (SKU, unit_price,
    discount_applied, payment_method) drawn from a conditional logit SKU
    choice model and channel-dependent distributions.

    Parameters
    ----------
    config:
        PersonaConfig instance — the generative root for this participant.
    category:
        Product category string embedded in each TransactionRecord.
    n_months:
        Lookback window in months; default 12.
    participant_id:
        Unique participant identifier. Defaults to config.persona_id
        (the archetype label) when None.

    Returns
    -------
    List of TransactionRecord instances, one per simulated purchase event.
    """
    if participant_id is None:
        participant_id = config.persona_id

    rng = np.random.default_rng(config.random_seed)
    params = config.transactions

    # Generate purchase occasions via hazard model
    involvement = _involvement_score(config)
    days_before_occasions = _generate_purchase_occasions(
        rng, n_months, params.purchase_frequency_per_month, involvement
    )
    n_transactions = len(days_before_occasions)

    logger.debug(
        "simulating_transactions",
        persona_id=config.persona_id,
        n_transactions=n_transactions,
        category=category,
        involvement=involvement,
    )

    channel_keys = list(params.channel_mix.keys())
    channel_probs = np.array([params.channel_mix[k] for k in channel_keys], dtype=float)
    channel_probs /= channel_probs.sum()
    channels = [_CHANNEL_MAP[k] for k in channel_keys]

    alpha, beta = _price_beta_params(params.price_sensitivity)

    records: list[TransactionRecord] = []
    for i in range(n_transactions):
        days_before = days_before_occasions[i]
        price_pct = float(np.clip(rng.beta(alpha, beta), 0.0, 1.0))
        quantity = max(1, int(rng.poisson(params.basket_size_mean)))
        on_promo = bool(rng.random() < params.price_sensitivity * 0.4)
        channel = channels[int(rng.choice(len(channels), p=channel_probs))]
        brand_tier = _sample_brand_tier(rng, params.brand_loyalty)
        purchase_type = _sample_purchase_type(rng, params.brand_loyalty, on_promo)
        product_id = f"prod_{category}_{brand_tier}_{int(rng.integers(1000, 9999))}"

        # --- Phase 2c: SKU choice via conditional logit ---
        sku_id, sku_tier, base_price = _choose_sku(
            rng, category, params.price_sensitivity, params.brand_loyalty
        )
        unit_price = round(price_pct * base_price, 2)

        # --- Phase 2c: discount_applied ---
        discount_applied: Optional[float] = None
        trigger_prob = 0.3 * params.price_sensitivity
        if rng.random() < trigger_prob:
            discount_applied = round(float(rng.uniform(0.05, 0.3)), 3)

        # --- Phase 2c: payment_method ---
        payment_method = _sample_payment_method(rng, channel)

        records.append(
            TransactionRecord(
                participant_id=participant_id,
                transaction_id=f"tx_{participant_id}_{i:04d}",
                days_before_session=days_before,
                category=category,
                product_id=product_id,
                brand_tier=brand_tier,
                price_paid_normalised=price_pct,
                quantity=quantity,
                channel=channel,
                purchase_type=purchase_type,
                on_promotion=on_promo,
                persona_id=config.persona_id,
                sku=sku_id,
                unit_price=unit_price,
                discount_applied=discount_applied,
                payment_method=payment_method,
                loyalty_card=None,
            )
        )

    return records
