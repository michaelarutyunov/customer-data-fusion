"""
Transaction simulator — generates 12-month synthetic purchase history from PersonaConfig.

Price percentiles (price_paid_normalised) are drawn from a Beta distribution whose
parameters are tuned to persona price_sensitivity:
  - High price_sensitivity → Beta(1, 4) → concentrates mass near 0 (cheap end)
  - Low price_sensitivity  → Beta(4, 1) → concentrates mass near 1 (premium end)
  - Mid sensitivity        → Beta(2, 2) → roughly uniform / bell near 0.5

Channel is sampled proportionally from config.transactions.channel_mix.
"""

from __future__ import annotations

import numpy as np
import structlog

from schemas.persona import PersonaConfig
from schemas.transaction import Channel, PurchaseType, TransactionRecord

logger = structlog.get_logger(__name__)

_BRAND_TIERS = ["premium", "mid", "value", "own_label"]

# Map channel_mix string keys → Channel enum
_CHANNEL_MAP: dict[str, Channel] = {
    "online": Channel.ONLINE,
    "in_store": Channel.IN_STORE,
    "click_and_collect": Channel.CLICK_AND_COLLECT,
}


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
    types = [PurchaseType.PLANNED, PurchaseType.IMPULSE, PurchaseType.HABITUAL, PurchaseType.DEAL_DRIVEN]
    return types[int(rng.choice(len(types), p=w))]


def simulate_transactions(
    config: PersonaConfig,
    category: str = "electronics",
    n_months: int = 12,
) -> list[TransactionRecord]:
    """
    Generate a synthetic 12-month purchase history for one participant.

    Parameters
    ----------
    config:
        PersonaConfig instance — the generative root for this participant.
    category:
        Product category string embedded in each TransactionRecord.
    n_months:
        Lookback window in months; default 12.

    Returns
    -------
    List of TransactionRecord instances, one per simulated purchase event.
    """
    rng = np.random.default_rng(config.random_seed)
    params = config.transactions

    lam = params.purchase_frequency_per_month * n_months
    n_transactions = int(rng.poisson(lam))

    logger.debug(
        "simulating_transactions",
        persona_id=config.persona_id,
        n_transactions=n_transactions,
        category=category,
    )

    channel_keys = list(params.channel_mix.keys())
    channel_probs = np.array([params.channel_mix[k] for k in channel_keys], dtype=float)
    channel_probs /= channel_probs.sum()
    channels = [_CHANNEL_MAP[k] for k in channel_keys]

    alpha, beta = _price_beta_params(params.price_sensitivity)

    records: list[TransactionRecord] = []
    for i in range(n_transactions):
        days_before = int(rng.integers(1, 366))  # 1–365 inclusive
        price_pct = float(np.clip(rng.beta(alpha, beta), 0.0, 1.0))
        quantity = max(1, int(rng.poisson(params.basket_size_mean)))
        on_promo = bool(rng.random() < params.price_sensitivity * 0.4)
        channel = channels[int(rng.choice(len(channels), p=channel_probs))]
        brand_tier = _sample_brand_tier(rng, params.brand_loyalty)
        purchase_type = _sample_purchase_type(rng, params.brand_loyalty, on_promo)
        product_id = f"prod_{category}_{brand_tier}_{int(rng.integers(1000, 9999))}"

        records.append(
            TransactionRecord(
                participant_id=config.persona_id,
                transaction_id=f"tx_{config.persona_id}_{i:04d}",
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
                loyalty_card=None,
            )
        )

    return records
