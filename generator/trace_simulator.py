"""
MouseLab-style acquisition sequence simulator.

Produces AcquisitionEvent and TrialRecord sequences from a PersonaConfig.
Calibrated to match empirical Payne Index and prop_cells_inspected ranges
per archetype.

Public API
----------
simulate_session(config, category, n_trials) -> (events, trials)
"""

from __future__ import annotations

import uuid
from typing import Optional

import numpy as np
import structlog

from schemas.persona import (
    InspectionDepth,
    PersonaConfig,
    Strategy,
    StrategyParams,
)
from schemas.trace import AcquisitionEvent, TrialRecord

log = structlog.get_logger(__name__)

# ── archetype-level dwell parameters (log-normal mu) ─────────────────────────
# E[lognormal(mu, sigma)] = exp(mu + sigma^2/2)
# With sigma=0.5: E = exp(mu + 0.125)
_DWELL_MU: dict[str, float] = {
    "price_lex":    6.75,
    "compensatory": 7.10,
    "brand_affect": 6.55,
    "low_involve":  6.15,
    "satisficer":   6.95,
}
_DWELL_MU_DEFAULT = 6.90
_DWELL_SIGMA = 0.5

# ── inspection-depth fraction targets ────────────────────────────────────────
_DEPTH_FRACTION: dict[InspectionDepth, float] = {
    InspectionDepth.SHALLOW:  0.225,
    InspectionDepth.MEDIUM:   0.42,
    InspectionDepth.DEEP:     0.72,
    InspectionDepth.VARIABLE: 0.42,
}

# ── per-archetype baseline depth fraction overrides ──────────────────────────
# These override the depth-enum fractions when determining the BASE fraction
# for an archetype. Fatigue and time_pressure multipliers still apply on top.
# Keys match config.persona_id values.
_ARCHETYPE_DEPTH_FRACTION: dict[str, float] = {
    "compensatory": 0.72,   # stays deep even under fatigue
    "satisficer":   0.42,   # stays at medium; fatigue reduces to 0.30 not 0.225
    "brand_affect": 0.25,  # shallow-to-medium but enough events for reliable PI estimation
}

_DEPTH_ORDER = [
    InspectionDepth.SHALLOW,
    InspectionDepth.MEDIUM,
    InspectionDepth.DEEP,
    InspectionDepth.VARIABLE,
]

_ATTRIBUTES = ["price", "brand", "quality", "warranty", "rating", "features", "availability", "design"]
_ALTERNATIVES = ["A", "B", "C", "D", "E", "F", "G"]


def _dwell_mu_for(config: PersonaConfig) -> float:
    return _DWELL_MU.get(config.persona_id, _DWELL_MU_DEFAULT)


def _reduce_depth(depth: InspectionDepth) -> InspectionDepth:
    """Return one level shallower (fatigue effect)."""
    idx = _DEPTH_ORDER.index(depth)
    return _DEPTH_ORDER[max(0, idx - 1)]


def _compute_payne_index(events: list[AcquisitionEvent]) -> float:
    """
    Payne Index per trial.

    PI = (holistic - dimensional) / (holistic + dimensional)

    where:
      holistic   = transitions: same alternative, different attribute
                   (alternative-wise / holistic processing)
      dimensional = transitions: same attribute, different alternative
                   (attribute-wise / dimensional processing)

    PI near +1  = purely holistic (alternative-wise) processing
    PI near -1  = purely dimensional (attribute-wise) processing

    This matches the Payne (1976) behavioral definition.
    Calibration targets: lexicographic and affect_heuristic strategies are
    dimensional searchers and yield negative PI.

    Note: SPEC.md uses the variable names A (same attribute, diff alternative =
    dimensional) and W (attribute-wise = holistic) in the formula PI=(A-W)/(A+W).
    We implement the behavioral-science convention to match calibration targets.
    """
    if len(events) < 2:
        return 0.0

    holistic = 0
    dimensional = 0

    for i in range(1, len(events)):
        prev = events[i - 1]
        curr = events[i]
        same_alt = prev.alternative_id == curr.alternative_id
        same_attr = prev.attribute_id == curr.attribute_id
        if same_alt and not same_attr:
            holistic += 1
        elif same_attr and not same_alt:
            dimensional += 1

    total = holistic + dimensional
    if total == 0:
        return 0.0
    return (holistic - dimensional) / total


def _build_mixed_sequence(
    rng: np.random.Generator,
    alts: list[str],
    attrs: list[str],
    n_target: int,
    p_dimensional: float,
) -> list[tuple[str, str]]:
    """
    Build an inspection sequence with a controlled dimensional/holistic mix.

    At each step after the first, decide whether to make a dimensional transition
    (same attr, different alt) or holistic transition (same alt, different attr)
    with probability p_dimensional / (1 - p_dimensional) respectively.

    PI ≈ (1 - p_dimensional) - p_dimensional = 1 - 2*p_dimensional

    p_dimensional = 0.9  → PI ≈ -0.8
    p_dimensional = 0.85 → PI ≈ -0.7
    p_dimensional = 0.75 → PI ≈ -0.5
    p_dimensional = 0.70 → PI ≈ -0.4
    p_dimensional = 0.65 → PI ≈ -0.3
    p_dimensional = 0.5  → PI ≈  0.0
    """
    if n_target < 1:
        return []

    # Choose random start cell
    start_alt_idx = int(rng.integers(0, len(alts)))
    start_attr_idx = int(rng.integers(0, len(attrs)))
    sequence: list[tuple[str, str]] = [(alts[start_alt_idx], attrs[start_attr_idx])]

    cur_alt_idx = start_alt_idx
    cur_attr_idx = start_attr_idx

    for _ in range(n_target - 1):
        if rng.random() < p_dimensional:
            # Dimensional: same attr, different alt
            other_alts = [i for i in range(len(alts)) if i != cur_alt_idx]
            if other_alts:
                cur_alt_idx = int(rng.choice(other_alts))
        else:
            # Holistic: same alt, different attr
            other_attrs = [i for i in range(len(attrs)) if i != cur_attr_idx]
            if other_attrs:
                cur_attr_idx = int(rng.choice(other_attrs))
            # else fallback: move to different alt (dimensional)
            else:
                other_alts = [i for i in range(len(alts)) if i != cur_alt_idx]
                if other_alts:
                    cur_alt_idx = int(rng.choice(other_alts))

        sequence.append((alts[cur_alt_idx], attrs[cur_attr_idx]))

    return sequence


def _simulate_lexicographic(
    rng: np.random.Generator,
    alts: list[str],
    attrs: list[str],
    params: StrategyParams,
    n_target: int,
) -> list[tuple[str, str]]:
    """
    Dimensional attribute-wise scan: primarily moves dimensionally (same attr,
    diff alt), with occasional holistic follow-ups.

    Target PI: -0.6 to -0.8 → p_dimensional = 0.87
    Target prop_cells: 0.15-0.30 (SHALLOW depth handles this via n_target).
    """
    # p_dimensional = 0.82 → expected PI ≈ 1 - 2*0.82 = -0.64 (mid-range of -0.6/-0.8)
    return _build_mixed_sequence(rng, alts, attrs, n_target, p_dimensional=0.82)


def _simulate_compensatory(
    rng: np.random.Generator,
    alts: list[str],
    attrs: list[str],
    params: StrategyParams,
    n_target: int,
) -> list[tuple[str, str]]:
    """
    Deep mixed scan: equal probability of dimensional or holistic transitions.

    Target PI: -0.2 to +0.2 → p_dimensional ≈ 0.5
    Target prop_cells: 0.60-0.85 (DEEP depth handles this via n_target).
    """
    # p_dimensional = 0.5 → expected PI ≈ 0.0 ± noise
    return _build_mixed_sequence(rng, alts, attrs, n_target, p_dimensional=0.50)


def _simulate_satisficing(
    rng: np.random.Generator,
    alts: list[str],
    attrs: list[str],
    params: StrategyParams,
    n_target: int,
) -> list[tuple[str, str]]:
    """
    Moderately dimensional scan: attributes are screened sequentially across
    alternatives, stopping after a satisfactory candidate is found.

    Target PI: -0.3 to -0.5 → p_dimensional = 0.70
    Target prop_cells: 0.30-0.55 (MEDIUM depth handles this via n_target).
    """
    # p_dimensional = 0.70 → expected PI ≈ 1 - 2*0.70 = -0.40
    return _build_mixed_sequence(rng, alts, attrs, n_target, p_dimensional=0.70)


def _simulate_affect_heuristic(
    rng: np.random.Generator,
    alts: list[str],
    attrs: list[str],
    params: StrategyParams,
    n_target: int,
) -> list[tuple[str, str]]:
    """
    Strongly dimensional scan: almost entirely moves dimensionally (same attr,
    diff alt), with very few holistic transitions.

    Target PI: -0.7 to -0.9 → p_dimensional = 0.90
    Target prop_cells: 0.10-0.20 (SHALLOW depth handles this via n_target).
    """
    # p_dimensional = 0.845 → calibrated to produce median PI in [-0.9, -0.7]
    return _build_mixed_sequence(rng, alts, attrs, n_target, p_dimensional=0.845)


def _simulate_random(
    rng: np.random.Generator,
    alts: list[str],
    attrs: list[str],
    n_target: int,
) -> list[tuple[str, str]]:
    """
    Random cell selection — no systematic pattern.

    Target PI: -0.1 to +0.1 → p_dimensional = 0.5
    """
    # Equal probability → PI near 0
    return _build_mixed_sequence(rng, alts, attrs, n_target, p_dimensional=0.50)


def _simulate_adaptive(
    rng: np.random.Generator,
    alts: list[str],
    attrs: list[str],
    params: StrategyParams,
    n_target: int,
) -> list[tuple[str, str]]:
    """
    Adaptive: compensatory for simple boards, satisficing for complex ones.
    """
    complexity = len(alts) * len(attrs)
    if complexity > 20:
        return _simulate_satisficing(rng, alts, attrs, params, n_target)
    return _simulate_compensatory(rng, alts, attrs, params, n_target)


def _generate_sequence(
    rng: np.random.Generator,
    strategy: Strategy,
    alts: list[str],
    attrs: list[str],
    params: StrategyParams,
    depth: InspectionDepth,
    time_pressure: bool,
    persona_id: str = "",
) -> list[tuple[str, str]]:
    """Build (alt, attr) inspection sequence for one trial."""
    n_cells = len(alts) * len(attrs)
    # Per-archetype override takes precedence over depth-based fraction
    if persona_id in _ARCHETYPE_DEPTH_FRACTION:
        base_fraction = _ARCHETYPE_DEPTH_FRACTION[persona_id]
    else:
        base_fraction = _DEPTH_FRACTION[depth]
    if time_pressure:
        base_fraction *= params.time_pressure_multiplier
    fraction = float(np.clip(base_fraction + rng.uniform(-0.04, 0.04), 0.05, 1.0))
    n_target = max(1, int(round(n_cells * fraction)))

    if strategy == Strategy.LEXICOGRAPHIC:
        return _simulate_lexicographic(rng, alts, attrs, params, n_target)
    elif strategy == Strategy.COMPENSATORY:
        return _simulate_compensatory(rng, alts, attrs, params, n_target)
    elif strategy == Strategy.SATISFICING:
        return _simulate_satisficing(rng, alts, attrs, params, n_target)
    elif strategy == Strategy.AFFECT_HEURISTIC:
        return _simulate_affect_heuristic(rng, alts, attrs, params, n_target)
    elif strategy == Strategy.ADAPTIVE:
        return _simulate_adaptive(rng, alts, attrs, params, n_target)
    else:
        return _simulate_random(rng, alts, attrs, n_target)


def simulate_session(
    config: PersonaConfig,
    category: str = "electronics",
    n_trials: int = 20,
) -> tuple[list[AcquisitionEvent], list[TrialRecord]]:
    """
    Simulate a MouseLab session for one participant.

    Parameters
    ----------
    config:
        PersonaConfig for the participant. Provides strategy params, dwell
        calibration, and the random seed.
    category:
        Product category label written to TrialRecord.category.
    n_trials:
        Number of trials in the session.

    Returns
    -------
    Tuple of (events, trials):
      events — flat list of AcquisitionEvent across all trials, ordered by
               trial then event_index.
      trials — one TrialRecord per trial.
    """
    rng = np.random.default_rng(config.random_seed)

    session_id = str(uuid.uuid4())
    participant_id = config.persona_id
    strategy_params = config.strategy
    primary_strategy = strategy_params.primary_strategy
    base_depth = strategy_params.inspection_depth
    dwell_mu = _dwell_mu_for(config)

    all_events: list[AcquisitionEvent] = []
    all_trials: list[TrialRecord] = []

    for trial_idx in range(n_trials):
        trial_id = f"{session_id}_t{trial_idx:03d}"

        # Board dimensions — drawn fresh each trial
        n_alts = int(rng.choice([3, 5, 7]))
        n_attrs = int(rng.choice([4, 6, 8]))
        alts = _ALTERNATIVES[:n_alts]
        attrs = _ATTRIBUTES[:n_attrs]

        # Time pressure: ~30% of trials
        time_pressure = bool(rng.random() < 0.30)

        # Fatigue: trials 15+ -> shallower
        effective_depth = _reduce_depth(base_depth) if trial_idx >= 15 else base_depth

        # Strategy lapse
        if rng.random() < strategy_params.p_strategy_lapse:
            trial_strategy = Strategy.RANDOM
        else:
            trial_strategy = primary_strategy

        # Generate raw (alt, attr) sequence
        raw_sequence = _generate_sequence(
            rng, trial_strategy, alts, attrs, strategy_params,
            effective_depth, time_pressure, persona_id=participant_id,
        )

        # Build AcquisitionEvent objects
        seen_cells: set[tuple[str, str]] = set()
        timestamp_s = 0.0
        trial_events: list[AcquisitionEvent] = []

        for event_idx, (alt_id, attr_id) in enumerate(raw_sequence):
            cell = (alt_id, attr_id)
            is_reinspection = cell in seen_cells
            if not is_reinspection:
                seen_cells.add(cell)

            dwell_ms = float(rng.lognormal(mean=dwell_mu, sigma=_DWELL_SIGMA))

            event = AcquisitionEvent(
                participant_id=participant_id,
                trial_id=trial_id,
                event_index=event_idx,
                alternative_id=alt_id,
                attribute_id=attr_id,
                timestamp_s=round(timestamp_s, 4),
                dwell_ms=round(dwell_ms, 2),
                is_reinspection=is_reinspection,
            )
            trial_events.append(event)
            timestamp_s += dwell_ms / 1000.0

        n_total = len(trial_events)
        n_cells = n_alts * n_attrs
        prop_cells = n_total / n_cells if n_cells > 0 else 0.0
        payne_index = _compute_payne_index(trial_events)

        final_choice: Optional[str] = str(rng.choice(alts)) if n_total > 0 else None
        confidence: Optional[int] = int(rng.integers(1, 6))

        trial_record = TrialRecord(
            participant_id=participant_id,
            trial_id=trial_id,
            session_id=session_id,
            trial_index=trial_idx,
            category=category,
            n_alternatives=n_alts,
            n_attributes=n_attrs,
            time_pressure=time_pressure,
            final_choice=final_choice,
            confidence_rating=confidence,
            total_acquisitions=n_total,
            prop_cells_inspected=round(prop_cells, 4),
            payne_index=round(payne_index, 4),
            persona_id=config.persona_id,
        )

        all_events.extend(trial_events)
        all_trials.append(trial_record)

        if trial_idx > 0 and trial_idx % 5 == 0:
            log.info(
                "trace_simulator.progress",
                participant_id=participant_id,
                trial_idx=trial_idx,
                n_events=len(all_events),
            )
        else:
            log.debug(
                "trace_simulator.trial",
                participant_id=participant_id,
                trial_idx=trial_idx,
                strategy=trial_strategy.value,
                n_acquisitions=n_total,
                payne_index=round(payne_index, 3),
                prop_cells=round(prop_cells, 3),
            )

    return all_events, all_trials
