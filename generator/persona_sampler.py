"""
generator/persona_sampler.py

Loads config/personas.yaml and instantiates PersonaConfig with per-participant
stochastic noise around the archetype base parameters.

Public API:
    sample_persona(archetype_id, random_seed) -> PersonaConfig
    list_archetype_ids() -> list[str]
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import structlog
import yaml

from schemas.persona import (
    InspectionDepth,
    LatentDeviation,
    NarrativeParams,
    PersonaConfig,
    PriceConsciousness,
    PsychographicParams,
    Strategy,
    StrategyParams,
    TransactionParams,
)

log = structlog.get_logger()

_YAML_PATH = Path(__file__).parent.parent / "config" / "personas.yaml"

# Per-participant noise scale (fraction of base value as std dev)
_NOISE_SCALE = 0.15

# Calibration parameters: scale sigma in project() calls per modality.
# Reduce GENERATOR_SPREAD to increase archetype signal in trace/transaction.
# Increase PSYCHOGRAPHIC_SPREAD to add individual noise in psychographic.
GENERATOR_SPREAD: float = float(os.environ.get("GENERATOR_SPREAD", "1.0"))
PSYCHOGRAPHIC_SPREAD: float = float(os.environ.get("PSYCHOGRAPHIC_SPREAD", "1.0"))


@lru_cache(maxsize=1)
def _load_yaml() -> dict:
    path = Path(os.environ.get("PERSONAS_YAML", str(_YAML_PATH)))
    with open(path) as f:
        data = yaml.safe_load(f)
    log.info(
        "personas_yaml_loaded", path=str(path), n_archetypes=len(data["archetypes"])
    )
    return data


def list_archetype_ids() -> list[str]:
    return list(_load_yaml()["archetypes"].keys())


def sample_persona(
    archetype_id: str, random_seed: Optional[int] = None
) -> PersonaConfig:
    """
    Instantiate a PersonaConfig for one participant from the named archetype.

    Per-participant noise is applied to continuous float parameters using a
    clipped normal distribution (std = _NOISE_SCALE * base_value), so each
    participant is a distinct sample from the archetype distribution while
    remaining semantically consistent with it.
    """
    data = _load_yaml()
    archetypes = data["archetypes"]
    if archetype_id not in archetypes:
        raise ValueError(
            f"Unknown archetype '{archetype_id}'. Valid ids: {list(archetypes.keys())}"
        )

    rng = np.random.default_rng(random_seed)
    raw = archetypes[archetype_id]

    z_arr = rng.standard_normal(5)
    z = LatentDeviation(
        price_lean=float(z_arr[0]),
        brand_lean=float(z_arr[1]),
        thoroughness=float(z_arr[2]),
        impulsivity=float(z_arr[3]),
        openness=float(z_arr[4]),
    )

    strategy = _build_strategy(raw["strategy"], rng)
    transactions = _build_transactions(raw["transactions"], rng, z)
    psychographic = _build_psychographic(raw["psychographic"], rng, z)
    narrative = _build_narrative(raw["narrative"], z)

    config = PersonaConfig(
        persona_id=archetype_id,
        label=raw["label"],
        strategy=strategy,
        transactions=transactions,
        psychographic=psychographic,
        narrative=narrative,
        latent=z,
        random_seed=random_seed,
    )
    log.debug(
        "persona_sampled",
        archetype_id=archetype_id,
        random_seed=random_seed,
        strategy=strategy.primary_strategy.value,
    )
    return config


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------


def _noisy(
    rng: np.random.Generator, base: float, lo: float = 0.0, hi: float = 1.0
) -> float:
    """Apply Gaussian noise to a base value, clipped to [lo, hi]."""
    std = _NOISE_SCALE * base if base > 0 else _NOISE_SCALE
    return float(np.clip(rng.normal(base, std), lo, hi))


def project(
    z_axis: float, base: float, sigma: float = 1.0, lo: float = 0.0, hi: float = 1.0
) -> float:
    """Logit-normal projection of a latent ``z`` axis onto a bounded parameter.

    theta = sigmoid(logit(base) + sigma * z_axis), clipped to [lo, hi].
    ``base`` is clamped to [0.001, 0.999] before ``logit`` to avoid log(0).
    """
    base_clamped = max(0.001, min(0.999, base))
    logit_base = math.log(base_clamped / (1.0 - base_clamped))
    theta = 1.0 / (1.0 + math.exp(-(logit_base + sigma * z_axis)))
    return float(max(lo, min(hi, theta)))


def _build_strategy(raw: dict, rng: np.random.Generator) -> StrategyParams:
    strategy = Strategy(raw["primary_strategy"])
    depth = InspectionDepth(raw["inspection_depth"])

    attribute_weights: Optional[dict[str, float]] = None
    if "attribute_weights" in raw:
        weights = {
            k: _noisy(rng, v, lo=0.01) for k, v in raw["attribute_weights"].items()
        }
        total = sum(weights.values())
        attribute_weights = {k: v / total for k, v in weights.items()}

    aspiration_levels: Optional[dict[str, float]] = None
    if "aspiration_levels" in raw:
        aspiration_levels = {
            k: _noisy(rng, v) for k, v in raw["aspiration_levels"].items()
        }

    return StrategyParams(
        primary_strategy=strategy,
        inspection_depth=depth,
        first_attribute=raw.get("first_attribute"),
        rejection_threshold_pct=(
            _noisy(rng, raw["rejection_threshold_pct"])
            if "rejection_threshold_pct" in raw
            else None
        ),
        attribute_weights=attribute_weights,
        aspiration_levels=aspiration_levels,
        p_reinspect=_noisy(rng, raw["p_reinspect"], lo=0.0, hi=1.0),
        p_strategy_lapse=_noisy(rng, raw["p_strategy_lapse"], lo=0.0, hi=1.0),
        time_pressure_multiplier=_noisy(
            rng, raw["time_pressure_multiplier"], lo=0.1, hi=1.0
        ),
    )


def _build_transactions(
    raw: dict, rng: np.random.Generator, z: LatentDeviation
) -> TransactionParams:
    channel_mix_raw = {
        k: _noisy(rng, v, lo=0.01) for k, v in raw["channel_mix"].items()
    }
    total = sum(channel_mix_raw.values())
    channel_mix = {k: v / total for k, v in channel_mix_raw.items()}

    _s = GENERATOR_SPREAD
    return TransactionParams(
        price_sensitivity=project(
            z.price_lean, raw["price_sensitivity"], sigma=1.0 * _s
        ),
        brand_loyalty=project(z.brand_lean, raw["brand_loyalty"], sigma=1.0 * _s),
        purchase_frequency_per_month=_noisy(
            rng, raw["purchase_frequency_per_month"], lo=0.1, hi=30.0
        ),
        basket_size_mean=max(
            1, int(round(_noisy(rng, float(raw["basket_size_mean"]), lo=1.0, hi=20.0)))
        ),
        channel_mix=channel_mix,
        price_variance_tolerance=_noisy(rng, raw["price_variance_tolerance"]),
    )


def _build_psychographic(
    raw: dict, _rng: np.random.Generator, z: LatentDeviation
) -> PsychographicParams:
    _s = PSYCHOGRAPHIC_SPREAD
    return PsychographicParams(
        involvement_score=project(
            z.thoroughness, raw["involvement_score"], sigma=1.0 * _s
        ),
        maximiser_score=project(z.thoroughness, raw["maximiser_score"], sigma=1.0 * _s),
        risk_tolerance=project(z.openness, raw["risk_tolerance"], sigma=0.8 * _s),
        price_consciousness=PriceConsciousness(raw["price_consciousness"].lower()),
        openness_to_new=project(z.openness, raw["openness_to_new"], sigma=1.0 * _s),
    )


# ---------------------------------------------------------------------------
# Narrative style lexicon — z-graded natural-language descriptors.
# Composed descriptions must never contain archetype labels, persona_ids, or
# Strategy enum values; only the phrases below are emitted.
# ---------------------------------------------------------------------------

_STYLE_LEXICON: dict[str, dict[str, str]] = {
    "price_lean": {
        "low": "largely indifferent to price differences",
        "mid": "mindful of value for money",
        "high": "acutely price-conscious, always seeking the best deal",
    },
    "brand_lean": {
        "low": "open to any brand that meets their needs",
        "mid": "with a soft preference for familiar names",
        "high": "fiercely loyal to one or two trusted brands",
    },
    "thoroughness": {
        "low": "who decides quickly with minimal comparison",
        "mid": "who weighs a few key factors before choosing",
        "high": "who researches options thoroughly before committing",
    },
    "impulsivity": {
        "low": "rarely makes unplanned purchases",
        "mid": "occasionally acts on impulse when something catches their eye",
        "high": "frequently makes spontaneous purchases driven by the moment",
    },
    "openness": {
        "low": "sticking to tried-and-tested options",
        "mid": "occasionally willing to try something new",
        "high": "actively seeking out new and unfamiliar products",
    },
}


def _bin_axis(value: float) -> str:
    if value < -0.5:
        return "low"
    elif value > 0.5:
        return "high"
    else:
        return "mid"


def _compose_style_description(z: LatentDeviation) -> str:
    thoroughness_phrase = _STYLE_LEXICON["thoroughness"][_bin_axis(z.thoroughness)]
    impulsivity_phrase = _STYLE_LEXICON["impulsivity"][_bin_axis(z.impulsivity)]
    brand_phrase = _STYLE_LEXICON["brand_lean"][_bin_axis(z.brand_lean)]
    price_phrase = _STYLE_LEXICON["price_lean"][_bin_axis(z.price_lean)]
    openness_phrase = _STYLE_LEXICON["openness"][_bin_axis(z.openness)]

    sentence1 = (
        f"A shopper {thoroughness_phrase}, {brand_phrase}, and {impulsivity_phrase}."
    )
    sentence2 = f"They are {price_phrase} and {openness_phrase}."
    return f"{sentence1} {sentence2}"


def _build_narrative(raw: dict, z: LatentDeviation) -> NarrativeParams:
    age_range_raw = raw["age_range"]
    return NarrativeParams(
        age_range=(int(age_range_raw[0]), int(age_range_raw[1])),
        household_type=raw["household_type"],
        category_relationship=raw["category_relationship"],
        decision_style_description=_compose_style_description(z),
        price_attitude=raw["price_attitude"],
        narrative_length_words=int(raw.get("narrative_length_words", 300)),
    )
