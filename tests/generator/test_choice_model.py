"""Tests for the Phase 0 choice model (SPEC §0.1–§0.5) — bead 5bz.

Covers the pure weight-resolution logic (default-weight rule, 'other' catch-all,
§0.4 z.brand_lean coupling), per-strategy utilities, slot exclusion, and RANDOM
uniformity.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pytest

from generator.choice_model import (
    BOARD_ATTRIBUTES,
    GAIN,
    TAU,
    compensatory_utility,
    compensatory_weights,
    encode_board,
    goodness_of,
)
from generator.trace_simulator import _compute_choice_from_inspected_cells
from schemas.persona import InspectionDepth, LatentDeviation, Strategy, StrategyParams
from schemas.product import Product
from schemas.trace import AcquisitionEvent

BOARD4 = BOARD_ATTRIBUTES[:4]
_DEPTH = InspectionDepth.MEDIUM


def _product(
    price=0.5,
    brand_tier="mid",
    quality=0.5,
    warranty=0.5,
    rating=3.0,
    features=0.5,
    availability=True,
    design=0.5,
) -> Product:
    return Product(
        product_id="t_00",
        category="electronics",
        price_normalised=price,
        brand_tier=brand_tier,
        quality_score=quality,
        warranty_score=warranty,
        rating=rating,
        features_score=features,
        availability=availability,
        design_score=design,
        on_promotion=False,
    )


def _evt(alt: str, attr: str, idx: int = 0) -> AcquisitionEvent:
    return AcquisitionEvent(
        participant_id="p",
        trial_id="t",
        event_index=idx,
        alternative_id=alt,
        attribute_id=attr,
        timestamp_s=0.0,
        dwell_ms=100.0,
        is_reinspection=False,
    )


def _run(
    strategy: StrategyParams,
    events: list[AcquisitionEvent],
    n_attrs: int = 4,
    z: Optional[LatentDeviation] = None,
    n_slots: int = 3,
    seed: int = 0,
) -> tuple[object, dict[str, float]]:
    prods = {chr(65 + k): _product() for k in range(n_slots)}
    rng = np.random.default_rng(seed)
    return _compute_choice_from_inspected_cells(
        events, strategy, prods, n_attrs, z or LatentDeviation(),
        price_sensitivity=0.5, brand_loyalty=0.5, rng=rng,
    )


# ── §0.1/§0.2 encoding & goodness ───────────────────────────────────────────


def test_encode_board_uses_section_0_1_encodings() -> None:
    p = _product(price=0.8, brand_tier="premium", quality=0.6, rating=4.0, availability=False)
    board = encode_board(p, 8)
    assert board["price"] == 0.8
    assert board["brand"] == 1.0  # premium ordinal
    assert board["quality"] == 0.6
    assert board["rating"] == pytest.approx(4.0 / 5.0)
    assert board["availability"] == 0.0  # False → 0.0


def test_goodness_inverts_price_only() -> None:
    p = _product(price=0.8, quality=0.6, brand_tier="premium")
    assert goodness_of(p, "price") == pytest.approx(0.2)  # 1 - 0.8
    assert goodness_of(p, "quality") == pytest.approx(0.6)
    assert goodness_of(p, "brand") == pytest.approx(1.0)  # not price → as-is


# ── §0.3 compensatory weight resolution ──────────────────────────────────────


def test_other_catchall_splits_over_unnamed_inspected() -> None:
    aw = {"price": 0.5, "quality": 0.2, "brand": 0.15, "other": 0.15}
    w = compensatory_weights(aw, ["price", "warranty", "rating"], BOARD4, 0.5, 0.5, 0.0)
    # price named (0.5); warranty+rating unnamed → other/2 = 0.075 each
    assert w == {"price": 0.5, "warranty": 0.075, "rating": 0.075}


def test_other_dropped_when_all_inspected_named() -> None:
    aw = {"price": 0.5, "brand": 0.2, "other": 0.1}
    w = compensatory_weights(aw, ["price", "brand"], BOARD4, 0.5, 0.5, 0.0)
    assert set(w) == {"price", "brand"}
    assert w["price"] == 0.5 and w["brand"] == 0.2


def test_brand_lean_shifts_w_brand_pre_renormalisation() -> None:
    aw = {"price": 0.5, "brand": 0.2, "other": 0.0}
    w = compensatory_weights(aw, ["price", "brand"], BOARD4, 0.5, 0.5, brand_lean=1.0)
    assert w["brand"] == pytest.approx(0.45)  # 0.2 + 0.25·1.0
    assert w["price"] == 0.5


def test_brand_lean_clamped_at_zero() -> None:
    aw = {"brand": 0.1}
    w = compensatory_weights(aw, ["brand"], BOARD4, 0.5, 0.5, brand_lean=-1.0)
    assert w["brand"] == 0.0  # max(0, 0.1 - 0.25)


def test_default_weight_rule_scales_price_and_brand() -> None:
    w = compensatory_weights(None, ["price", "quality"], BOARD4, 0.5, 0.5, 0.0)
    # board uniform 0.25; price·1.5=0.375, brand·1.5=0.375, others 0.25 → 1.25
    # renorm: price=0.3, brand=0.3, quality=0.2, warranty=0.2; restrict inspected
    assert w["price"] == pytest.approx(0.3)
    assert w["quality"] == pytest.approx(0.2)


def test_compensatory_utility_weighted_goodness() -> None:
    p = _product(price=0.8, quality=0.6)  # g_price=0.2, g_quality=0.6
    u = compensatory_utility(p, ["price", "quality"], None, BOARD4, 0.5, 0.5, 0.0)
    # weights price=0.3, quality=0.2 → (0.3·0.2 + 0.2·0.6)/0.5 = 0.36
    assert u == pytest.approx(0.36)


# ── §0.3 strategy routing via _compute_choice_from_inspected_cells ───────────


def test_random_strategy_is_uniform() -> None:
    strat = StrategyParams(primary_strategy=Strategy.RANDOM, inspection_depth=_DEPTH)
    events = [_evt("A", "price"), _evt("B", "price")]
    _, probs = _run(strat, events)
    n = len(probs)
    assert all(abs(p - 1.0 / n) < 1e-9 for p in probs.values())


def test_degenerate_trial_is_uniform() -> None:
    # Only one slot inspected → |C| < 2 → effective lapse (uniform over all)
    strat = StrategyParams(primary_strategy=Strategy.COMPENSATORY, inspection_depth=_DEPTH)
    events = [_evt("A", "price")]
    _, probs = _run(strat, events, n_attrs=4)
    n = len(probs)
    assert all(abs(p - 1.0 / n) < 1e-9 for p in probs.values())


def test_excluded_uninspected_slot_gets_zero_probability() -> None:
    strat = StrategyParams(primary_strategy=Strategy.COMPENSATORY, inspection_depth=_DEPTH)
    # A & C inspected, B not → B excluded (prob 0), A+C share the mass
    events = [_evt("A", "price"), _evt("C", "price")]
    _, probs = _run(strat, events, n_attrs=4)
    assert probs["B"] == 0.0
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_satisficing_runs_aspiration_rule() -> None:
    strat = StrategyParams(
        primary_strategy=Strategy.SATISFICING,
        inspection_depth=_DEPTH,
        aspiration_levels={"price": 0.5},
    )
    events = [_evt("A", "price"), _evt("B", "price")]
    _, probs = _run(strat, events, n_attrs=4)
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_affect_heuristic_neutral_when_brand_uninspected() -> None:
    strat = StrategyParams(primary_strategy=Strategy.AFFECT_HEURISTIC, inspection_depth=_DEPTH)
    # No brand inspected → ū=0.5 for both → equal softmax
    events = [_evt("A", "price"), _evt("B", "price")]
    _, probs = _run(strat, events, n_attrs=4)
    assert abs(probs["A"] - probs["B"]) < 1e-9


def test_adaptive_resolves_by_n_attrs() -> None:
    strat = StrategyParams(primary_strategy=Strategy.ADAPTIVE, inspection_depth=_DEPTH)
    events = [_evt("A", "price"), _evt("B", "price")]
    for n_attrs in (4, 6, 8):
        _, probs = _run(strat, events, n_attrs=n_attrs)
        assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_constants_are_pinned() -> None:
    assert GAIN == 8.0
    assert TAU == 1.0
    assert math.isclose(GAIN / TAU, 8.0)
