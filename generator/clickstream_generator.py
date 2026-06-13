"""
Clickstream generator — Markov session model conditioned on latent z.

Generates web clickstream events for one participant over one or more months.
Session navigation follows intent-specific Markov transition matrices with
z-conditioned perturbations (brand_lean, impulsivity, involvement).

Public API:
    simulate_clickstream(config, month, random_seed) -> (events, summaries)
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Optional, TypeVar

import numpy as np
import structlog

from generator.persona_sampler import LatentDeviation, project
from schemas.clickstream import (
    ClickstreamEvent,
    ClickstreamEventType,
    DeviceType,
    PageType,
    ReferrerType,
    SessionIntent,
    SessionSummary,
)
from schemas.persona import PersonaConfig

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Markov transition matrices by session intent
# ---------------------------------------------------------------------------
# States are ClickstreamEventType values. ``"end"`` is a sentinel indicating
# session termination. Transitions are stored as {state -> {next_state -> prob}}.

_TransitionMatrix = dict[str, dict[str, float]]

_BASE_TRANSITIONS: dict[SessionIntent, _TransitionMatrix] = {
    SessionIntent.BROWSE: {
        "HOME": {"CATEGORY": 0.6, "PAGE_VIEW": 0.4},
        "CATEGORY": {"PRODUCT_VIEW": 0.5, "CATEGORY": 0.3, "PAGE_VIEW": 0.2},
        "PRODUCT_VIEW": {"CATEGORY": 0.6, "PRODUCT_VIEW": 0.3, "end": 0.1},
        # Fallback for any unrecognised state
        "PAGE_VIEW": {"CATEGORY": 0.5, "end": 0.5},
    },
    SessionIntent.RESEARCH: {
        "HOME": {"SEARCH": 0.8, "CATEGORY": 0.2},
        "SEARCH": {"SEARCH_RESULTS": 1.0},
        "SEARCH_RESULTS": {"PRODUCT_VIEW": 0.7, "SEARCH_RESULTS": 0.2, "end": 0.1},
        "PRODUCT_VIEW": {
            "PRODUCT_VIEW": 0.4,
            "SEARCH_RESULTS": 0.4,
            "CATEGORY": 0.2,
        },
    },
    SessionIntent.BUY: {
        "HOME": {"CATEGORY": 0.4, "SEARCH": 0.3, "PRODUCT_VIEW": 0.3},
        "CATEGORY": {"PRODUCT_VIEW": 0.7, "ADD_TO_CART": 0.2, "end": 0.1},
        "PRODUCT_VIEW": {"PRODUCT_VIEW": 0.3, "ADD_TO_CART": 0.5, "CATEGORY": 0.2},
        "ADD_TO_CART": {
            "CHECKOUT_START": 0.70,
            "REMOVE_FROM_CART": 0.15,
            "PRODUCT_VIEW": 0.10,
            "end": 0.05,
        },
        "REMOVE_FROM_CART": {"CATEGORY": 0.5, "SEARCH": 0.3, "end": 0.2},
        "CHECKOUT_START": {"PURCHASE": 0.85, "end": 0.15},
        "SEARCH": {"SEARCH_RESULTS": 1.0},
        "SEARCH_RESULTS": {
            "PRODUCT_VIEW": 0.6,
            "ADD_TO_CART": 0.3,
            "SEARCH_RESULTS": 0.1,
        },
    },
}

# All event types available in the simulation
_ALL_EVENT_TYPES = list(ClickstreamEventType)

# Device distribution for sampling
_DEVICE_WEIGHTS: dict[DeviceType, float] = {
    DeviceType.DESKTOP: 0.50,
    DeviceType.MOBILE: 0.35,
    DeviceType.TABLET: 0.15,
}

# Referrer distribution for sampling
_REFERRER_WEIGHTS: dict[ReferrerType, float] = {
    ReferrerType.DIRECT: 0.30,
    ReferrerType.ORGANIC: 0.30,
    ReferrerType.PAID_SEARCH: 0.20,
    ReferrerType.EMAIL: 0.10,
    ReferrerType.SOCIAL: 0.10,
}

# Anonymous session probability range
_ANONYMOUS_PROB_MIN = 0.10
_ANONYMOUS_PROB_MAX = 0.15

# Short bounce session probability range (1-3 events, no purchase)
_BOUNCE_PROB_MIN = 0.30
_BOUNCE_PROB_MAX = 0.60

# Log-normal dwell time parameters
_DWELL_MEDIAN_MS = 5000.0  # median dwell in ms
_DWELL_SIGMA = 0.8  # log-normal sigma

_T = TypeVar("_T")


def _project_involvement(z: Optional[LatentDeviation]) -> float:
    """Project z onto involvement_score in [0, 1]."""
    if z is None:
        return 0.5
    return project(z.thoroughness, 0.5, sigma=1.0, lo=0.0, hi=1.0)


def _sample_enum(rng: np.random.Generator, weights: dict[_T, float]) -> _T:
    """Sample an enum value from a {value: weight} dict."""
    items = list(weights.keys())
    probs = np.array([weights[k] for k in items], dtype=float)
    probs /= probs.sum()
    return items[int(rng.choice(len(items), p=probs))]


def _perturb_transitions(
    base: _TransitionMatrix,
    brand_lean: float,
    impulsivity: float,
    involvement: float,
    intent: SessionIntent,
) -> _TransitionMatrix:
    """Apply z-conditioned perturbations to a base transition matrix.

    - brand_lean increases PRODUCT_VIEW -> PRODUCT_VIEW transition by 0.2 * brand_lean
    - impulsivity increases exit probability by 0.15 * impulsivity
    - involvement scales session length via base_n_events = 3 + 5 * involvement
    """
    perturbed: _TransitionMatrix = {}
    for state, transitions in base.items():
        new_trans = dict(transitions)

        # brand_lean: increase self-loop on PRODUCT_VIEW
        if state == "PRODUCT_VIEW" and "PRODUCT_VIEW" in new_trans:
            delta = 0.2 * brand_lean
            new_trans["PRODUCT_VIEW"] = new_trans.get("PRODUCT_VIEW", 0.0) + delta
            # Renormalise by reducing other transitions proportionally
            other_keys = [k for k in new_trans if k != "PRODUCT_VIEW"]
            if other_keys:
                reduction = delta / len(other_keys)
                for k in other_keys:
                    new_trans[k] = max(0.0, new_trans[k] - reduction)

        # impulsivity: increase exit probability
        if impulsivity > 0:
            exit_boost = 0.15 * impulsivity
            if "end" in new_trans:
                new_trans["end"] = new_trans["end"] + exit_boost
            else:
                new_trans["end"] = exit_boost
            # Reduce the largest non-end transition to compensate
            non_end = {k: v for k, v in new_trans.items() if k != "end"}
            if non_end:
                largest = max(non_end, key=lambda k: non_end[k])
                new_trans[largest] = max(0.0, new_trans[largest] - exit_boost)

        # Clip all probabilities to [0, 1] and renormalise
        total = sum(max(0.0, v) for v in new_trans.values())
        if total > 0:
            new_trans = {k: max(0.0, v) / total for k, v in new_trans.items()}

        perturbed[state] = new_trans

    return perturbed


def _generate_single_session(
    rng: np.random.Generator,
    transitions: _TransitionMatrix,
    intent: SessionIntent,
    device: DeviceType,
    referrer: ReferrerType,
    customer_id: str,
    session_id: str,
    session_start: datetime,
    month: int,
    base_n_events: int,
    is_bounce: bool = False,
) -> tuple[list[ClickstreamEvent], SessionSummary]:
    """Generate events for a single clickstream session using a Markov walk."""
    events: list[ClickstreamEvent] = []
    current_state = "HOME"
    event_ts = session_start

    if is_bounce:
        # Short bounce session: 1-3 events from HOME, then end
        n_bounce = int(rng.integers(1, 4))  # 1-3 events
        bounce_states = ["HOME", "CATEGORY", "PAGE_VIEW", "PRODUCT_VIEW"]
        for i in range(n_bounce):
            state = bounce_states[min(i, len(bounce_states) - 1)]
            event_type = _et_map.get(state, ClickstreamEventType.PAGE_VIEW)
            page_type = _state_to_page(state)
            dwell = float(rng.lognormal(math.log(_DWELL_MEDIAN_MS), _DWELL_SIGMA))
            events.append(
                ClickstreamEvent(
                    customer_id=customer_id,
                    session_id=session_id,
                    event_ts=event_ts.isoformat(),
                    event_type=event_type,
                    page_type=page_type,
                    sku_viewed=_maybe_sku(rng, state),
                    referrer=referrer,
                    device=device,
                    dwell_ms=dwell,
                    month=month,
                )
            )
            event_ts += timedelta(milliseconds=dwell)
        duration = (event_ts - session_start).total_seconds()
        summary = SessionSummary(
            customer_id=customer_id,
            session_id=session_id,
            n_events=len(events),
            session_duration_s=duration,
            intent=intent,
            device=device,
            month=month,
        )
        return events, summary

    # Normal Markov session
    max_events = max(base_n_events * 3, 30)  # safety cap
    step = 0
    while current_state != "end" and step < max_events:
        # Map state name to ClickstreamEventType
        if current_state not in _et_map:
            break

        event_type = _et_map[current_state]
        page_type = _state_to_page(current_state)
        dwell = float(rng.lognormal(math.log(_DWELL_MEDIAN_MS), _DWELL_SIGMA))

        events.append(
            ClickstreamEvent(
                customer_id=customer_id,
                session_id=session_id,
                event_ts=event_ts.isoformat(),
                event_type=event_type,
                page_type=page_type,
                sku_viewed=_maybe_sku(rng, current_state),
                referrer=referrer,
                device=device,
                dwell_ms=dwell,
                month=month,
            )
        )
        event_ts += timedelta(milliseconds=dwell)
        step += 1

        # Transition to next state
        if current_state not in transitions:
            break
        next_states = transitions[current_state]
        keys = list(next_states.keys())
        probs = np.array([next_states[k] for k in keys], dtype=float)
        probs /= probs.sum()
        current_state = keys[int(rng.choice(len(keys), p=probs))]

    duration = (event_ts - session_start).total_seconds()
    summary = SessionSummary(
        customer_id=customer_id,
        session_id=session_id,
        n_events=len(events),
        session_duration_s=duration,
        intent=intent,
        device=device,
        month=month,
    )
    return events, summary


# Mapping of state strings to ClickstreamEventType.
# Navigation states (HOME, CATEGORY, SEARCH_RESULTS) map to PAGE_VIEW
# since visiting these pages is a page view event.
_et_map: dict[str, ClickstreamEventType] = {
    "HOME": ClickstreamEventType.PAGE_VIEW,
    "CATEGORY": ClickstreamEventType.PAGE_VIEW,
    "SEARCH_RESULTS": ClickstreamEventType.PAGE_VIEW,
    "PAGE_VIEW": ClickstreamEventType.PAGE_VIEW,
    "PRODUCT_VIEW": ClickstreamEventType.PRODUCT_VIEW,
    "ADD_TO_CART": ClickstreamEventType.ADD_TO_CART,
    "REMOVE_FROM_CART": ClickstreamEventType.REMOVE_FROM_CART,
    "SEARCH": ClickstreamEventType.SEARCH,
    "FILTER_APPLY": ClickstreamEventType.FILTER_APPLY,
    "CHECKOUT_START": ClickstreamEventType.CHECKOUT_START,
    "PURCHASE": ClickstreamEventType.PURCHASE,
}

# Mapping of state strings to PageType
_page_map: dict[str, PageType] = {
    "HOME": PageType.HOME,
    "CATEGORY": PageType.CATEGORY,
    "PRODUCT": PageType.PRODUCT,
    "SEARCH_RESULTS": PageType.SEARCH_RESULTS,
    "CART": PageType.CART,
    "CHECKOUT": PageType.CHECKOUT,
}


def _state_to_page(state: str) -> PageType:
    """Map a Markov state name to its corresponding PageType."""
    return _page_map.get(state, PageType.HOME)


def _maybe_sku(rng: np.random.Generator, state: str) -> Optional[str]:
    """Generate a SKU for product-related states."""
    if state == "PRODUCT_VIEW":
        return f"SKU-{int(rng.integers(1000, 9999))}"
    if state == "ADD_TO_CART":
        return f"SKU-{int(rng.integers(1000, 9999))}"
    return None


def simulate_clickstream(
    config: PersonaConfig,
    month: int = 1,
    random_seed: Optional[int] = None,
) -> tuple[list[ClickstreamEvent], list[SessionSummary]]:
    """
    Generate synthetic clickstream sessions for one participant over one month.

    Parameters
    ----------
    config:
        PersonaConfig instance — the generative root for this participant.
    month:
        Month index for temporal partitioning (1-12).
    random_seed:
        Seed for reproducible generation. Defaults to config.random_seed.

    Returns
    -------
    Tuple of (events, summaries):
        - events: flat list of all ClickstreamEvent across all sessions
        - summaries: list of SessionSummary, one per session
    """
    seed = random_seed if random_seed is not None else config.random_seed
    rng = np.random.default_rng(seed)

    z = config.latent or LatentDeviation()

    # Project z axes to bounded parameters
    involvement = _project_involvement(z)
    brand_lean = (
        z.brand_lean
    )  # raw z, not projected — perturbation uses signed deviation
    impulsivity = max(0.0, z.impulsivity)  # only positive impulsivity increases bounce

    # Sessions per month: Poisson(lambda = 3 + 4 * involvement_score)
    lam = 3.0 + 4.0 * involvement
    n_sessions = max(1, int(rng.poisson(lam)))

    # Determine anonymous session fraction
    anonymous_prob = float(rng.uniform(_ANONYMOUS_PROB_MIN, _ANONYMOUS_PROB_MAX))
    # Determine bounce session fraction
    bounce_prob = float(rng.uniform(_BOUNCE_PROB_MIN, _BOUNCE_PROB_MAX))

    # Base session length: 3 + 5 * involvement
    base_n_events = 3 + int(5 * involvement)

    all_events: list[ClickstreamEvent] = []
    all_summaries: list[SessionSummary] = []

    # Month baseline for timestamps
    month_start = datetime(2025, month, 1)
    # Max days in the month
    if month == 12:
        next_month_start = datetime(2026, 1, 1)
    else:
        next_month_start = datetime(2025, month + 1, 1)
    days_in_month = (next_month_start - month_start).days

    for sess_idx in range(n_sessions):
        session_id = f"sess_{config.persona_id}_{month:02d}_{sess_idx:04d}"

        # Decide if anonymous session
        is_anonymous = rng.random() < anonymous_prob
        customer_id = "anonymous" if is_anonymous else config.persona_id

        # Decide if bounce session
        is_bounce = rng.random() < bounce_prob

        # Sample session intent
        intent = _sample_intent(rng)

        # Sample device and referrer
        device = _sample_enum(rng, _DEVICE_WEIGHTS)
        referrer = _sample_enum(rng, _REFERRER_WEIGHTS)

        # Random session start within the month
        day_offset = int(rng.integers(0, days_in_month))
        hour = int(rng.integers(0, 24))
        minute = int(rng.integers(0, 60))
        session_start = month_start + timedelta(
            days=day_offset, hours=hour, minutes=minute
        )

        # Get transition matrix and apply z perturbations
        base_matrix = _BASE_TRANSITIONS[intent]
        perturbed_matrix = _perturb_transitions(
            base_matrix, brand_lean, impulsivity, involvement, intent
        )

        events, summary = _generate_single_session(
            rng=rng,
            transitions=perturbed_matrix,
            intent=intent,
            device=device,
            referrer=referrer,
            customer_id=customer_id,
            session_id=session_id,
            session_start=session_start,
            month=month,
            base_n_events=base_n_events,
            is_bounce=is_bounce,
        )

        all_events.extend(events)
        all_summaries.append(summary)

    logger.debug(
        "clickstream_simulated",
        persona_id=config.persona_id,
        month=month,
        n_sessions=n_sessions,
        n_events=len(all_events),
        n_anonymous=sum(1 for s in all_summaries if s.customer_id == "anonymous"),
        n_bounce=sum(1 for s in all_summaries if s.n_events <= 3),
    )

    return all_events, all_summaries


def _sample_intent(rng: np.random.Generator) -> SessionIntent:
    """Sample session intent: BROWSE 50%, RESEARCH 30%, BUY 20%."""
    r = rng.random()
    if r < 0.50:
        return SessionIntent.BROWSE
    elif r < 0.80:
        return SessionIntent.RESEARCH
    else:
        return SessionIntent.BUY
