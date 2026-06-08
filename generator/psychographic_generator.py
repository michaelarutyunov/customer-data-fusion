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

from generator.persona_sampler import project
from schemas.persona import PersonaConfig, PriceConsciousness
from schemas.psychographic import PsychographicVector

log = structlog.get_logger(__name__)

_PRICE_CONSCIOUSNESS_BASES: dict[PriceConsciousness, float] = {
    PriceConsciousness.LOW: 0.2,
    PriceConsciousness.MEDIUM: 0.5,
    PriceConsciousness.HIGH: 0.85,
}

_DECISION_STYLES = [
    "analytical",
    "dependent",
    "intuitive",
    "spontaneous",
    "avoidant",
    "deliberate",
]

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


def generate_psychographic(
    config: PersonaConfig,
    category: str = "electronics",
    participant_id: str | None = None,
) -> PsychographicVector:
    """
    Generate a PsychographicVector from a PersonaConfig.

    Applies Gaussian noise (std = 0.05 * base) to continuous fields and
    derives categorical fields from strategy and narrative parameters.

    Args:
        config: PersonaConfig archetype root.
        category: Product category for the vector.
        participant_id: Unique participant identifier. Defaults to
            config.persona_id (the archetype label) when None.

    Returns:
        PsychographicVector with all fields populated.
    """
    if participant_id is None:
        participant_id = config.persona_id

    rng = np.random.default_rng(config.random_seed)

    psych = config.psychographic
    txn = config.transactions
    narrative = config.narrative

    z = config.latent
    if z is None:
        from schemas.persona import LatentDeviation

        z = LatentDeviation()

    # Continuous fields sourced directly from z-driven PersonaConfig values
    involvement_score = psych.involvement_score
    maximiser_score = psych.maximiser_score
    risk_tolerance = psych.risk_tolerance
    openness_to_new = psych.openness_to_new

    # price_consciousness: continuous, driven by z.price_lean
    price_consciousness = project(
        z.price_lean,
        base=_PRICE_CONSCIOUSNESS_BASES[psych.price_consciousness],
        sigma=0.8,
    )

    # brand_sensitivity: same z.brand_lean as transaction brand_loyalty
    brand_sensitivity = txn.brand_loyalty

    # Decision style: z-conditioned softmax sample
    logits = np.array(
        [
            1.5 * z.thoroughness - 1.0 * z.impulsivity,  # analytical
            0.0,  # dependent
            0.3 * z.impulsivity,  # intuitive
            2.0 * z.impulsivity,  # spontaneous
            0.5 * z.impulsivity,  # avoidant
            1.0 * z.thoroughness,  # deliberate
        ]
    )
    probs = np.exp(logits) / np.sum(np.exp(logits))
    decision_style_dominant = str(rng.choice(_DECISION_STYLES, p=probs))

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
        participant_id=participant_id,
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
