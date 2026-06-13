"""
generator/persona_sampler.py

Loads config/personas.yaml and instantiates PersonaConfig with per-participant
stochastic noise around the archetype base parameters.

Public API:
    sample_persona(archetype_id, random_seed) -> PersonaConfig
    sample_temporal_trajectory(config, n_months, random_seed) -> list[PersonaConfig]
    list_archetype_ids() -> list[str]
"""

from __future__ import annotations

import math
import os
from dataclasses import replace
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

    z_arr = rng.standard_normal(7)
    z = LatentDeviation(
        price_lean=float(z_arr[0]),
        brand_lean=float(z_arr[1]),
        thoroughness=float(z_arr[2]),
        impulsivity=float(z_arr[3]),
        search_orientation=float(z_arr[4]),
        attentional_bias=float(z_arr[5]),
        openness=float(z_arr[6]),
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
# Temporal z evolution — AR(1) drift over simulated months
# ---------------------------------------------------------------------------

# Drift applied only to loyalty/churn dimensions and attentional_bias.
# thoroughness, impulsivity, openness, search_orientation are stable traits.
_DRIFT_AXES = {0, 1, 5}  # price_lean, brand_lean, attentional_bias indices

# Default AR(1) parameters (configurable via env vars)
_AR1_ALPHA: float = float(os.environ.get("AR1_ALPHA", "0.85"))  # persistence
_AR1_SIGMA: float = float(os.environ.get("AR1_SIGMA", "0.15"))  # drift noise
_REGIME_SHIFT_PCT: float = float(
    os.environ.get("REGIME_SHIFT_PCT", "0.12")
)  # fraction of cohort


def sample_temporal_trajectory(
    config: PersonaConfig,
    n_months: int = 12,
    random_seed: Optional[int] = None,
) -> list[PersonaConfig]:
    """Generate a month-by-month trajectory of PersonaConfig snapshots.

    Applies AR(1) drift to the loyalty/churn dimensions (price_lean,
    brand_lean) and attentional_bias. Stable traits (thoroughness,
    impulsivity, openness, search_orientation) remain fixed.

    For ~12% of participants (controlled by _REGIME_SHIFT_PCT), a regime
    shift is injected at a random month (6-10): a sudden large negative
    shock to brand_lean (loyalty decay) or shift in attentional_bias.

    Returns a list of PersonaConfig with month=0..n_months, where month 0
    is the original config (baseline). Generators should use month 1..n_months.
    """
    rng = np.random.default_rng(random_seed or config.random_seed or 42)
    z0 = config.latent or LatentDeviation()
    z_arr = list(z0.as_tuple())  # 7 elements

    # Decide if this participant gets a regime shift
    has_regime_shift = rng.random() < _REGIME_SHIFT_PCT
    shift_month: int = 0
    if has_regime_shift:
        shift_month = int(rng.integers(6, 11))  # month 6-10

    trajectory: list[PersonaConfig] = [_with_month(config, z0, 0)]

    for month in range(1, n_months + 1):
        z_new = list(z_arr)  # copy previous

        # AR(1) drift on drift axes only
        for idx in _DRIFT_AXES:
            z_new[idx] = (
                _AR1_ALPHA * z_arr[idx]
                + (1.0 - _AR1_ALPHA) * 0.0  # mu=0 (centered at archetype mean)
                + _AR1_SIGMA * rng.standard_normal()
            )

        # Regime shift: sudden loyalty decay or attention shift
        if has_regime_shift and month == shift_month:
            # Large negative shock to brand_lean (loyalty decay)
            z_new[1] -= 1.5 + 0.5 * rng.standard_normal()
            # Attentional bias shift
            z_new[5] += 0.8 * rng.standard_normal()

        z_month = LatentDeviation(
            price_lean=z_new[0],
            brand_lean=z_new[1],
            thoroughness=z_new[2],
            impulsivity=z_new[3],
            search_orientation=z_new[4],
            attentional_bias=z_new[5],
            openness=z_new[6],
        )
        trajectory.append(_with_month(config, z_month, month))
        z_arr = z_new  # carry forward for next month

    log.debug(
        "temporal_trajectory_sampled",
        persona_id=config.persona_id,
        n_months=n_months,
        has_regime_shift=has_regime_shift,
        shift_month=shift_month if has_regime_shift else None,
    )
    return trajectory


def get_drift_metadata(
    config: PersonaConfig,
    n_months: int = 12,
    random_seed: Optional[int] = None,
) -> tuple[bool, int | None]:
    """Return ground-truth regime shift metadata for a participant's trajectory.

    Replays the same RNG decisions as sample_temporal_trajectory to determine
    whether this participant has an injected regime shift and at which month.
    This is the authoritative drift label — superior to post-hoc threshold
    detection on brand_lean deviations, which conflates regime shifts with
    ordinary AR(1) stochastic drift.

    Returns:
        (has_regime_shift, shift_month) where shift_month is None if no shift.
    """
    rng = np.random.default_rng(random_seed or config.random_seed or 42)
    has_regime_shift = rng.random() < _REGIME_SHIFT_PCT
    shift_month: int | None = None
    if has_regime_shift:
        shift_month = int(rng.integers(6, 11))  # month 6-10
    return has_regime_shift, shift_month


def _with_month(config: PersonaConfig, z: LatentDeviation, month: int) -> PersonaConfig:
    """Create a copy of config with updated latent z and month index."""
    return replace(config, latent=z, month=month)


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
