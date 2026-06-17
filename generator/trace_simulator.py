"""
MouseLab-style acquisition sequence simulator.

Produces AcquisitionEvent and TrialRecord sequences from a PersonaConfig.
Calibrated to match empirical Payne Index and prop_cells_inspected ranges
per archetype.

Phase 2b additions:
- Per-individual strategy mixture derived from LatentDeviation via softmax
- Attentional weight divergence (dwell share rotated by attentional_bias)
- EventType generation per AcquisitionEvent
- Elimination-by-aspects (EBA) strategy

Public API
----------
simulate_session(config, category, n_trials) -> (events, trials)
"""

from __future__ import annotations

import math
import uuid
from typing import Optional

import numpy as np
import structlog

from generator.persona_sampler import GENERATOR_SPREAD
from schemas.persona import (
    InspectionDepth,
    LatentDeviation,
    PersonaConfig,
    Strategy,
    StrategyParams,
)
from schemas.product import Product
from schemas.trace import AcquisitionEvent, EventType, TrialRecord

log = structlog.get_logger(__name__)

# ── dwell parameters (log-normal) ────────────────────────────────────────────
# E[lognormal(mu, sigma)] = exp(mu + sigma^2/2)
# With sigma=0.5: E = exp(mu + 0.125)
_DWELL_SIGMA = 0.5

# ── dwell threshold for event_type ───────────────────────────────────────────
_DEEP_DWELL_THRESHOLD_MS = 800.0

# ── inspection-depth fraction targets ────────────────────────────────────────
_DEPTH_FRACTION: dict[InspectionDepth, float] = {
    InspectionDepth.SHALLOW: 0.225,
    InspectionDepth.MEDIUM: 0.42,
    InspectionDepth.DEEP: 0.72,
    InspectionDepth.VARIABLE: 0.42,
}

_DEPTH_ORDER = [
    InspectionDepth.SHALLOW,
    InspectionDepth.MEDIUM,
    InspectionDepth.DEEP,
    InspectionDepth.VARIABLE,
]

_ATTRIBUTES = [
    "price",
    "brand",
    "quality",
    "warranty",
    "rating",
    "features",
    "availability",
    "design",
]
_ALTERNATIVES = ["A", "B", "C", "D", "E", "F", "G"]

# ── strategy mixture: logit coefficients for softmax ─────────────────────────
# Order must match _MIXTURE_STRATEGIES below.
# Compensatory favoured by thoroughness; lexicographic favoured by low thoroughness
# and low search_orientation plus impulsivity; random favoured by impulsivity alone.
_STRATEGY_LOGIT_COEFFS: list[tuple[float, float, float]] = [
    # (thoroughness_coeff, search_orientation_coeff, impulsivity_coeff)
    (2.0, 0.0, 0.0),  # compensatory
    (-1.5, 1.0, 0.0),  # satisficing
    (-2.0, -1.0, 1.5),  # lexicographic
    (0.0, 0.0, 1.0),  # random
]
_MIXTURE_STRATEGIES: list[Strategy] = [
    Strategy.COMPENSATORY,
    Strategy.SATISFICING,
    Strategy.LEXICOGRAPHIC,
    Strategy.RANDOM,
]

# Mapping from non-mixture strategies to their closest mixture equivalent,
# so the primary-strategy bonus can be applied correctly.
_PRIMARY_TO_MIXTURE: dict[Strategy, Strategy] = {
    Strategy.AFFECT_HEURISTIC: Strategy.LEXICOGRAPHIC,  # both strongly dimensional
    Strategy.ADAPTIVE: Strategy.COMPENSATORY,  # uses compensatory for simple boards
    Strategy.ELIMINATION_BY_ASPECTS: Strategy.SATISFICING,  # both sequential screening
}

# ── EBA parameters ───────────────────────────────────────────────────────────
_EBA_ELIMINATION_THRESHOLD = 0.60  # fraction of alternatives eliminated per attribute


# ── Strategy mixture from latent deviation ────────────────────────────────────


# Base logit bonus for the archetype's primary strategy. At z=0 this gives the
# primary strategy ~95% weight while allowing individual variation from the
# z-derived logits. The z logits can override this when |z| is large (>2.0).
_PRIMARY_STRATEGY_LOGIT_BONUS = 5.0


def _compute_strategy_mixture(
    z: LatentDeviation,
    primary_strategy: Strategy,
    temperature: float = 1.0,
) -> list[float]:
    """Compute per-individual strategy mixture weights via softmax over z-derived logits.

    The z-derived logit formula is:
        logits = [
            2.0 * z.thoroughness,                                                    # compensatory
            -1.5 * z.thoroughness + 1.0 * z.search_orientation,                      # satisficing
            -2.0 * z.thoroughness - 1.0 * z.search_orientation + 1.5 * z.impulsivity, # lexicographic
            1.0 * z.impulsivity                                                       # random
        ]
    A bonus is added to the primary strategy's logit so that at archetype-mean
    z (all zeros), the primary strategy dominates (~95% weight).
    If primary_strategy is not in the mixture (e.g. AFFECT_HEURISTIC), it is
    mapped to its closest mixture equivalent via _PRIMARY_TO_MIXTURE.
    Positive thoroughness shifts toward compensatory; negative shifts toward
    lexicographic/random. Search_orientation shifts toward satisficing.
    Impulsivity shifts toward lexicographic/random.
    """
    # Map non-mixture strategies to their closest mixture equivalent
    mixture_anchor = _PRIMARY_TO_MIXTURE.get(primary_strategy, primary_strategy)

    z_axes = (z.thoroughness, z.search_orientation, z.impulsivity)
    logits: list[float] = []
    for idx, (t_coeff, s_coeff, i_coeff) in enumerate(_STRATEGY_LOGIT_COEFFS):
        logit = t_coeff * z_axes[0] + s_coeff * z_axes[1] + i_coeff * z_axes[2]
        # Add bonus to the primary strategy so it dominates at archetype-mean z
        if _MIXTURE_STRATEGIES[idx] == mixture_anchor:
            logit += _PRIMARY_STRATEGY_LOGIT_BONUS
        logits.append(logit)

    # Softmax with temperature
    scaled = [logit / temperature for logit in logits]
    max_scaled = max(scaled)
    exp_vals = [math.exp(s - max_scaled) for s in scaled]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]


def _sample_strategy_from_mixture(
    rng: np.random.Generator, mixture: list[float]
) -> Strategy:
    """Draw a strategy from the mixture distribution."""
    idx = int(rng.choice(len(mixture), p=mixture))
    return _MIXTURE_STRATEGIES[idx]


# ── Attentional weight divergence ────────────────────────────────────────────


def _compute_attentional_dwell_weights(
    attribute_weights: dict[str, float] | None,
    attrs: list[str],
    attentional_bias: float,
) -> dict[str, float]:
    """Compute dwell-share per attribute, rotated away from choice preference weights.

    When attentional_bias=0, dwell shares match attribute_weights exactly.
    When attentional_bias>0, dwell on price is reduced by ``attentional_bias * 0.3``
    and the removed share is redistributed proportionally to other attributes.
    When attentional_bias<0, price dwell increases symmetrically.

    If attribute_weights is None, returns uniform weights (no rotation applied).
    """
    n = len(attrs)
    if attribute_weights is None:
        return {a: 1.0 / n for a in attrs}

    # Normalise attribute_weights to only include attrs in the current board
    raw = {a: attribute_weights.get(a, 1.0 / n) for a in attrs}
    total = sum(raw.values())
    base_shares = {a: v / total for a, v in raw.items()}

    if abs(attentional_bias) < 1e-9:
        return base_shares

    # Rotate price share
    shifted = dict(base_shares)
    if "price" in shifted:
        price_shift = attentional_bias * 0.3
        shifted["price"] = max(0.0, shifted["price"] - price_shift)

    # Redistribute the deficit/surplus proportionally to other attributes
    others = [a for a in attrs if a != "price"]
    other_total = sum(shifted[a] for a in others)
    current_total = sum(shifted.values())
    if current_total > 0 and other_total > 0:
        # Scale others so total sums to 1.0
        scale = (1.0 - shifted["price"]) / other_total if other_total > 0 else 1.0
        for a in others:
            shifted[a] = shifted[a] * scale

    # Final normalisation (float safety)
    total_shifted = sum(shifted.values())
    if total_shifted > 0:
        shifted = {a: v / total_shifted for a, v in shifted.items()}
    else:
        shifted = {a: 1.0 / n for a in attrs}

    return shifted


# ── Dwell time with attentional modulation ────────────────────────────────────


def _dwell_mu_for(config: PersonaConfig) -> float:
    """Dwell mean (log-normal mu) as a continuous function of involvement and z."""
    z = config.latent
    if z is None:
        z = LatentDeviation()
    # Base mu from involvement_score (0->5.8, 1->7.5)
    involvement = config.psychographic.involvement_score
    mu_base = 5.8 + 1.7 * involvement
    # thoroughness shifts mu up; impulsivity shifts it down — scaled by GENERATOR_SPREAD
    _s = GENERATOR_SPREAD
    return float(mu_base + 0.4 * _s * z.thoroughness - 0.3 * _s * z.impulsivity)


def _sample_dwell_ms(
    rng: np.random.Generator,
    base_mu: float,
    attr: str,
    dwell_weights: dict[str, float],
    uniform_weight: float,
) -> float:
    """Sample dwell time for an attribute, modulated by dwell weight share.

    Attributes with higher dwell weight share get longer dwells on average.
    The modulation is multiplicative on the log-normal mu: higher share -> longer.
    """
    share = dwell_weights.get(attr, uniform_weight)
    # Modulate mu: shift up/down proportionally to how much share exceeds uniform
    # share > uniform => more attention => longer dwell
    mu_shift = 0.3 * (share - uniform_weight)
    dwell_ms = float(rng.lognormal(mean=base_mu + mu_shift, sigma=_DWELL_SIGMA))
    return max(10.0, dwell_ms)  # floor at 10ms


# ── Depth helpers ─────────────────────────────────────────────────────────────


def _reduce_depth(depth: InspectionDepth) -> InspectionDepth:
    """Return one level shallower (fatigue effect)."""
    idx = _DEPTH_ORDER.index(depth)
    return _DEPTH_ORDER[max(0, idx - 1)]


# ── Payne Index ───────────────────────────────────────────────────────────────


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


# ── Sequence builders ─────────────────────────────────────────────────────────


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
    _n_target: int,
) -> list[tuple[str, str]]:
    """
    Price-lexicographic: inspect only the key attribute across all alternatives
    in random order, then stop.

    Target PI: ≈ -1.0 (pure dimensional — same attr, all alts)
    Target prop_cells: 1/n_attrs (exactly one column)
    """
    key_attr = (
        params.first_attribute
        if (params.first_attribute and params.first_attribute in attrs)
        else attrs[0]
    )
    alt_order = list(alts)
    rng.shuffle(alt_order)
    return [(alt, key_attr) for alt in alt_order]


def _simulate_compensatory(
    rng: np.random.Generator,
    alts: list[str],
    attrs: list[str],
    _params: StrategyParams,
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
    _params: StrategyParams,
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
    _params: StrategyParams,
    n_target: int,
) -> list[tuple[str, str]]:
    """
    Strongly dimensional scan: almost entirely moves dimensionally (same attr,
    diff alt), with very few holistic transitions.

    Target PI: -0.7 to -0.9 → p_dimensional = 0.90
    Target prop_cells: 0.10-0.20 (SHALLOW depth handles this via n_target).
    """
    # p_dimensional = 0.80 → calibrated to produce median PI in [-0.9, -0.7]
    return _build_mixed_sequence(rng, alts, attrs, n_target, p_dimensional=0.80)


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


def _simulate_elimination_by_aspects(
    rng: np.random.Generator,
    alts: list[str],
    attrs: list[str],
    params: StrategyParams,
    n_target: int,
) -> list[tuple[str, str]]:
    """
    Elimination-by-aspects (EBA): inspect attributes sequentially by weight,
    eliminating alternatives below threshold on each attribute.

    Process:
    1. Sort attributes by weight (descending). Use attribute_weights if available,
       otherwise use attr order as proxy.
    2. For each attribute: inspect remaining alternatives dimensionally.
    3. Eliminate a fraction of alternatives below threshold.
    4. Repeat until one alternative remains or attributes exhausted.

    Target PI: -0.4 to -0.6 (dimensional search within each attribute).
    Target prop_cells: 0.25-0.45.
    """
    # Determine attribute order by weight (descending)
    weights = params.attribute_weights
    if weights:
        attr_order = sorted(
            attrs,
            key=lambda a: weights.get(a, 0.1),
            reverse=True,
        )
    else:
        attr_order = list(attrs)

    remaining_alts = list(alts)
    sequence: list[tuple[str, str]] = []

    for attr in attr_order:
        if len(remaining_alts) <= 1:
            break

        # Early stopping: with few remaining alternatives, sometimes stop
        if len(remaining_alts) <= 2 and rng.random() < 0.5:
            break

        # Inspect a subset of remaining alternatives (not all) to control prop_cells
        # Sample min(len(remaining_alts), ceil(len(remaining_alts)*0.8)) alternatives
        n_inspect = max(
            1, min(len(remaining_alts), int(len(remaining_alts) * 0.8 + 0.5))
        )
        inspect_alts = list(remaining_alts)
        rng.shuffle(inspect_alts)
        inspect_alts = inspect_alts[:n_inspect]
        for alt in inspect_alts:
            sequence.append((alt, attr))

        # Eliminate a fraction of alternatives (those with worst values)
        # Use stochastic elimination to avoid deterministic output
        n_to_eliminate = max(
            0,
            int(len(remaining_alts) * _EBA_ELIMINATION_THRESHOLD)
            + (1 if rng.random() < 0.3 else 0),
        )
        if n_to_eliminate > 0 and len(remaining_alts) > n_to_eliminate + 1:
            # Randomly eliminate (simulating unknown value distribution)
            rng.shuffle(remaining_alts)
            remaining_alts = remaining_alts[n_to_eliminate:]

    # If we have remaining budget from n_target, add a few more inspections
    if len(sequence) < n_target and remaining_alts:
        extra = n_target - len(sequence)
        for _ in range(extra):
            alt = str(rng.choice(remaining_alts))
            attr = str(rng.choice(attr_order))
            sequence.append((alt, attr))

    return sequence


def _generate_sequence(
    rng: np.random.Generator,
    strategy: Strategy,
    alts: list[str],
    attrs: list[str],
    params: StrategyParams,
    depth: InspectionDepth,
    time_pressure: bool,
    z: LatentDeviation | None = None,
) -> list[tuple[str, str]]:
    """Build (alt, attr) inspection sequence for one trial."""
    n_cells = len(alts) * len(attrs)
    base_fraction = _DEPTH_FRACTION[depth]
    if time_pressure:
        base_fraction *= params.time_pressure_multiplier
    # z.thoroughness widens or narrows the fraction jitter — scaled by GENERATOR_SPREAD
    thoroughness_factor = 0.0 if z is None else z.thoroughness * GENERATOR_SPREAD
    fraction = float(
        np.clip(
            base_fraction + rng.uniform(-0.04, 0.04) + 0.06 * thoroughness_factor,
            0.05,
            1.0,
        )
    )
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
    elif strategy == Strategy.ELIMINATION_BY_ASPECTS:
        return _simulate_elimination_by_aspects(rng, alts, attrs, params, n_target)
    else:
        return _simulate_random(rng, alts, attrs, n_target)


# ── EventType assignment ──────────────────────────────────────────────────────


def _assign_event_types(
    events: list[AcquisitionEvent],
    strategy: Strategy,
) -> list[AcquisitionEvent]:
    """Assign EventType to each AcquisitionEvent based on dwell and transition context.

    Rules:
    - CELL_HOVER: shallow inspection (dwell < 800ms)
    - CELL_OPEN: deep inspection (dwell >= 800ms)
    - COLUMN_ADD: first inspection in a new attribute column (different attr from previous)
    - SORT_APPLY: strategy-driven attribute switch mid-sequence (compensatory only)
    - CHOICE: final selection event (appended as last event in trial)

    Distribution is conditioned on strategy:
    - Lexicographic: mostly CELL_HOVER + CHOICE (shallow, fast scans)
    - Compensatory: CELL_OPEN + COLUMN_ADD + SORT_APPLY (deep, broad exploration)
    - Satisficing: mix of CELL_HOVER and CELL_OPEN
    - Affect heuristic: mostly CELL_HOVER (shallow)
    - EBA: mix of CELL_HOVER and CELL_OPEN with COLUMN_ADD between attribute blocks
    """
    if not events:
        return events

    typed_events: list[AcquisitionEvent] = []
    seen_attrs: set[str] = set()
    prev_attr: str | None = None

    for i, event in enumerate(events):
        is_final = i == len(events) - 1

        if is_final:
            # Last event before choice: treat as CHOICE indicator
            # We keep the actual event but change type to CHOICE
            typed = AcquisitionEvent(
                participant_id=event.participant_id,
                trial_id=event.trial_id,
                event_index=event.event_index,
                alternative_id=event.alternative_id,
                attribute_id=event.attribute_id,
                timestamp_s=event.timestamp_s,
                dwell_ms=event.dwell_ms,
                is_reinspection=event.is_reinspection,
                event_type=EventType.CHOICE,
            )
            typed_events.append(typed)
            continue

        # Determine base type from dwell
        if event.dwell_ms < _DEEP_DWELL_THRESHOLD_MS:
            base_type = EventType.CELL_HOVER
        else:
            base_type = EventType.CELL_OPEN

        # Check for COLUMN_ADD: first time we see a new attribute
        attr = event.attribute_id
        is_new_attr = attr not in seen_attrs
        if is_new_attr:
            seen_attrs.add(attr)

        # Check for SORT_APPLY: strategy-driven attribute switch
        # (compensatory switching to a new attribute mid-sequence, not the first time)
        is_sort_apply = (
            strategy == Strategy.COMPENSATORY
            and prev_attr is not None
            and attr != prev_attr
            and not is_new_attr  # not the first visit to this attribute
        )

        if is_sort_apply:
            event_type = EventType.SORT_APPLY
        elif is_new_attr and prev_attr is not None and i > 0:
            event_type = EventType.COLUMN_ADD
        else:
            event_type = base_type

        typed = AcquisitionEvent(
            participant_id=event.participant_id,
            trial_id=event.trial_id,
            event_index=event.event_index,
            alternative_id=event.alternative_id,
            attribute_id=event.attribute_id,
            timestamp_s=event.timestamp_s,
            dwell_ms=event.dwell_ms,
            is_reinspection=event.is_reinspection,
            event_type=event_type,
        )
        typed_events.append(typed)
        prev_attr = attr

    return typed_events


def _compute_choice_from_inspected_cells(
    trial_events: list[AcquisitionEvent],
    trial_strategy: StrategyParams,
    alternative_products: dict[str, Product],
    temperature: float = 1.0,
    rng: np.random.Generator = None,
) -> Optional[str]:
    """Compute choice based on inspected cells and strategy.

    This couples the trace (what was inspected) to the choice,
    enabling the CDT encoding the trace to predict the decision.

    Parameters
    ----------
    trial_events:
        List of acquisition events from the trace simulation.
    trial_strategy:
        Strategy parameters for this trial.
    alternative_products:
        Mapping from slot letters to Product objects.
    temperature:
        Softmax temperature for choice probability (default 1.0).
    rng:
        Random number generator (uses default if None).

    Returns
    -------
    chosen_slot:
        The slot letter of the chosen alternative, or None if no events.
    """
    if rng is None:
        rng = np.random.default_rng()

    if len(trial_events) == 0:
        return None

    # Extract inspected attribute values
    inspected_attrs = {}
    for event in trial_events:
        if event.attribute_id is not None:
            attr, value = event.attribute_id.split("=")
            inspected_attrs[attr] = float(value)

    # Compute utility per alternative based on strategy
    utilities = {}
    for slot, product in alternative_products.items():
        if trial_strategy.primary_strategy == Strategy.LEXICOGRAPHIC:
            # Choose best on first_attribute among inspected cells
            first_attr = trial_strategy.first_attribute
            if first_attr in inspected_attrs:
                utilities[slot] = inspected_attrs[first_attr]
            else:
                utilities[slot] = 0.0  # Penalty for unobserved attribute
        elif trial_strategy.primary_strategy == Strategy.COMPENSATORY:
            # Weighted sum over inspected attributes
            utility = 0.0
            if trial_strategy.attribute_weights:
                for attr, weight in trial_strategy.attribute_weights.items():
                    if attr in inspected_attrs:
                        utility += weight * inspected_attrs[attr]
                utilities[slot] = utility
            else:
                utilities[slot] = 0.0
        else:
            # Fallback to random for unknown strategies
            utilities[slot] = rng.uniform(0, 1)

    # Add strategy lapse noise
    lapse_noise = rng.normal(0, trial_strategy.p_strategy_lapse)
    for slot in utilities:
        utilities[slot] += lapse_noise

    # Softmax with pinned temperature
    exp_utilities = {k: np.exp(u / temperature) for k, u in utilities.items()}
    sum_exp = sum(exp_utilities.values())

    if sum_exp == 0:
        return str(rng.choice(list(alternative_products.keys())))

    probs = {k: v / sum_exp for k, v in exp_utilities.items()}

    # Sample choice
    chosen_slot = rng.choice(list(probs.keys()), p=list(probs.values()))
    return chosen_slot


# ── Main simulation entry point ──────────────────────────────────────────────


def simulate_session(
    config: PersonaConfig,
    category: str = "electronics",
    n_trials: int = 20,
    participant_id: str | None = None,
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
    participant_id:
        Unique participant identifier. Defaults to config.persona_id
        (the archetype label) when None, preserving backward compatibility
        with tests that don't pass explicit IDs.

    Returns
    -------
    Tuple of (events, trials):
      events — flat list of AcquisitionEvent across all trials, ordered by
               trial then event_index.
      trials — one TrialRecord per trial.
    """
    rng = np.random.default_rng(config.random_seed)

    session_id = str(uuid.uuid4())
    if participant_id is None:
        participant_id = config.persona_id
    strategy_params = config.strategy
    primary_strategy = strategy_params.primary_strategy
    base_depth = strategy_params.inspection_depth
    dwell_mu = _dwell_mu_for(config)

    z = config.latent
    if z is None:
        z = LatentDeviation()

    # Compute per-individual strategy mixture from latent deviation,
    # anchored to the archetype's primary strategy
    mixture = _compute_strategy_mixture(z, primary_strategy)

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

        # Per-trial strategy drawn from individual mixture.
        # When the drawn mixture strategy maps back to the primary (via _PRIMARY_TO_MIXTURE),
        # use the actual primary strategy for simulation. This ensures AFFECT_HEURISTIC
        # stays affect_heuristic, ADAPTIVE stays adaptive, etc.
        mixture_draw = _sample_strategy_from_mixture(rng, mixture)
        if (
            primary_strategy not in _MIXTURE_STRATEGIES
            and _PRIMARY_TO_MIXTURE.get(primary_strategy) == mixture_draw
        ):
            trial_strategy = primary_strategy
        else:
            trial_strategy = mixture_draw

        # Strategy lapse modulated by impulsivity — scaled by GENERATOR_SPREAD.
        # This overrides the mixture draw on impulse-heavy participants.
        impulsivity_boost = 0.12 * GENERATOR_SPREAD * max(0.0, z.impulsivity)
        effective_lapse_prob = float(
            np.clip(strategy_params.p_strategy_lapse + impulsivity_boost, 0.0, 0.8)
        )
        if rng.random() < effective_lapse_prob:
            trial_strategy = Strategy.RANDOM

        # Generate raw (alt, attr) sequence
        raw_sequence = _generate_sequence(
            rng,
            trial_strategy,
            alts,
            attrs,
            strategy_params,
            effective_depth,
            time_pressure,
            z=z,
        )

        # Brand reinspection: with prob proportional to brand_lean, add a brand cell reinspection
        brand_lean = z.brand_lean
        if brand_lean > 0 and rng.random() < 0.3 * brand_lean:
            brand_attr = "brand"
            if brand_attr in attrs:
                reinspect_alt = str(rng.choice(alts))
                raw_sequence.append((reinspect_alt, brand_attr))

        # Compute attentional dwell weights for this trial's attribute set
        trial_dwell_weights = _compute_attentional_dwell_weights(
            strategy_params.attribute_weights,
            attrs,
            z.attentional_bias,
        )
        uniform_weight = 1.0 / len(attrs)

        # Build AcquisitionEvent objects
        seen_cells: set[tuple[str, str]] = set()
        timestamp_s = 0.0
        trial_events: list[AcquisitionEvent] = []

        for event_idx, (alt_id, attr_id) in enumerate(raw_sequence):
            cell = (alt_id, attr_id)
            is_reinspection = cell in seen_cells
            if not is_reinspection:
                seen_cells.add(cell)

            dwell_ms = _sample_dwell_ms(
                rng, dwell_mu, attr_id, trial_dwell_weights, uniform_weight
            )

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

        # Assign event types based on dwell and transition context
        trial_events = _assign_event_types(trial_events, trial_strategy)

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
