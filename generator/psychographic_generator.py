"""
generator/psychographic_generator.py

Generates a PsychographicVector from a PersonaConfig by mapping archetype
parameters to survey-scale values with calibrated Gaussian noise.

Public API:
    generate_psychographic(config: PersonaConfig, category: str = "electronics") -> PsychographicVector
"""

from __future__ import annotations

import numpy as np
import structlog

from schemas.persona import PersonaConfig, PriceConsciousness, Strategy
from schemas.psychographic import PsychographicVector

log = structlog.get_logger(__name__)

_PRICE_CONSCIOUSNESS_MAP: dict[PriceConsciousness, float] = {
    PriceConsciousness.LOW: 0.2,
    PriceConsciousness.MEDIUM: 0.5,
    PriceConsciousness.HIGH: 0.85,
}

_STRATEGY_TO_DECISION_STYLE: dict[Strategy, str] = {
    Strategy.LEXICOGRAPHIC: "analytical",
    Strategy.COMPENSATORY: "analytical",
    Strategy.SATISFICING: "dependent",
    Strategy.AFFECT_HEURISTIC: "intuitive",
    Strategy.RANDOM: "spontaneous",
    Strategy.ADAPTIVE: "avoidant",
}

_AGE_BANDS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
_AGE_BAND_EDGES = [18, 25, 35, 45, 55, 65, 200]

_EMPLOYMENT_STATUSES = [
    "full_time",
    "part_time",
    "self_employed",
    "not_employed",
    "retired",
]
_EMPLOYMENT_WEIGHTS = [0.5, 0.15, 0.15, 0.10, 0.10]


def _age_to_band(age: int) -> str:
    for i, edge in enumerate(_AGE_BAND_EDGES[1:], start=0):
        if age < edge:
            return _AGE_BANDS[i]
    return _AGE_BANDS[-1]


def _noisy(rng: np.random.Generator, base: float, std_factor: float = 0.05) -> float:
    noise = rng.normal(0.0, std_factor * base)
    return float(np.clip(base + noise, 0.0, 1.0))


def generate_psychographic(
    config: PersonaConfig,
    category: str = "electronics",
) -> PsychographicVector:
    """
    Generate a PsychographicVector from a PersonaConfig.

    Applies Gaussian noise (std = 0.05 * base) to continuous fields and
    derives categorical fields from strategy and narrative parameters.

    Args:
        config: PersonaConfig archetype root.
        category: Product category for the vector.

    Returns:
        PsychographicVector with all fields populated.
    """
    rng = np.random.default_rng(config.random_seed)

    psych = config.psychographic
    txn = config.transactions
    narrative = config.narrative
    strategy = config.strategy

    # Continuous fields with noise
    involvement_score = _noisy(rng, psych.involvement_score)
    maximiser_score = _noisy(rng, psych.maximiser_score)
    risk_tolerance = _noisy(rng, psych.risk_tolerance)
    openness_to_new = _noisy(rng, psych.openness_to_new)

    # Derived continuous fields
    price_consciousness_base = _PRICE_CONSCIOUSNESS_MAP[psych.price_consciousness]
    price_consciousness = _noisy(rng, price_consciousness_base)

    brand_sensitivity = _noisy(rng, txn.brand_loyalty)

    # Decision style from primary strategy
    decision_style_dominant = _STRATEGY_TO_DECISION_STYLE[strategy.primary_strategy]

    # Age band: sample uniformly from age_range
    age_lo, age_hi = narrative.age_range
    age = int(rng.integers(age_lo, age_hi + 1))
    age_band = _age_to_band(age)

    # Employment status: weighted sample
    employment_status = str(rng.choice(_EMPLOYMENT_STATUSES, p=_EMPLOYMENT_WEIGHTS))

    # Purchase frequency band
    freq = txn.purchase_frequency_per_month
    if freq >= 4:
        purchase_frequency_band = "weekly"
    elif freq >= 1:
        purchase_frequency_band = "monthly"
    elif freq >= 0.25:
        purchase_frequency_band = "quarterly"
    else:
        purchase_frequency_band = "annually_or_less"

    log.debug(
        "generated_psychographic",
        persona_id=config.persona_id,
        category=category,
        decision_style=decision_style_dominant,
        age_band=age_band,
    )

    return PsychographicVector(
        participant_id=config.persona_id,
        persona_id=config.persona_id,
        involvement_score=involvement_score,
        maximiser_score=maximiser_score,
        risk_tolerance=risk_tolerance,
        price_consciousness=price_consciousness,
        brand_sensitivity=brand_sensitivity,
        openness_to_new=openness_to_new,
        decision_style_dominant=decision_style_dominant,
        age_band=age_band,
        household_type=narrative.household_type,
        employment_status=employment_status,
        category=category,
        purchase_frequency_band=purchase_frequency_band,
        years_buying_category=None,
    )
