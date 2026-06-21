"""Phase 0 choice-model constants and pure helpers (SPEC §0.1–§0.5).

Every value here is a **declared, fixed** constant — not a tunable. This module
is the single source of truth shared by the catalogue generator, the trace
simulator's board display and choice logic, and (eventually) the evaluation
oracle. See ``.claude/context/new-capabilities.md`` "Phase 0 Generator SPEC".

This module depends only on ``schemas.product`` — it never imports other
generator modules, so it cannot create a cycle.
"""

from __future__ import annotations

from schemas.product import Product

# ── §0.1 board ──────────────────────────────────────────────────────────────
# The board shows the first ``n_attrs`` (n_attrs ∈ {4, 6, 8}) of these names,
# in this fixed order. Slot letters are chr(65 + k) for k = 0 .. n_alts-1.
BOARD_ATTRIBUTES: list[str] = [
    "price",
    "brand",
    "quality",
    "warranty",
    "rating",
    "features",
    "availability",
    "design",
]

# §0.1 brand_tier → ordinal level. This is both the brand display encoding and
# the source of brand goodness (g_brand = level, since brand is not price).
BRAND_TIER_LEVEL: dict[str, float] = {
    "premium": 1.0,
    "mid": 0.66,
    "value": 0.33,
    "own_label": 0.0,
}

# ── §0.5 decisiveness gain + softmax temperature (both fixed) ───────────────
# GAIN sharpens preference choices (a 0.3 goodness gap → ≈11:1 odds at GAIN=8).
# TAU is pinned at 1.0 — never tuned to hit M1 calibration (no-circularity).
GAIN: float = 8.0
TAU: float = 1.0


def displayed_value(product: Product, attr: str) -> float:
    """§0.1 encoding: map a board attribute to its displayed float in [0, 1].

    ``price`` is shown as-is; ``brand`` as its tier ordinal; ``rating`` scaled
    by 5; ``availability`` as 1.0/0.0; every other attribute as-is.

    Raises ``KeyError`` for a non-board attribute — callers that may receive
    ad-hoc keys (e.g. the ``other`` weight) must guard with ``attr in
    BOARD_ATTRIBUTES`` first.
    """
    if attr == "price":
        return float(product.price_normalised)
    if attr == "brand":
        return BRAND_TIER_LEVEL[product.brand_tier]
    if attr == "quality":
        return float(product.quality_score)
    if attr == "warranty":
        return float(product.warranty_score)
    if attr == "rating":
        return float(product.rating) / 5.0
    if attr == "features":
        return float(product.features_score)
    if attr == "availability":
        return 1.0 if product.availability else 0.0
    if attr == "design":
        return float(product.design_score)
    raise KeyError(f"Not a board attribute: {attr!r}")


def goodness(attr: str, value: float) -> float:
    """§0.2 goodness ``g_a ∈ [0,1]`` from a displayed value: price is inverted
    (lower price is better), every other attribute is used as-is."""
    if attr == "price":
        return 1.0 - value
    return value


def encode_board(product: Product, n_attrs: int) -> dict[str, float]:
    """Build ``displayed_attributes`` for the first ``n_attrs`` board attributes."""
    return {attr: displayed_value(product, attr) for attr in BOARD_ATTRIBUTES[:n_attrs]}


def goodness_of(product: Product, attr: str) -> float:
    """§0.2 goodness ``g_a`` for a board attribute of ``product``."""
    return goodness(attr, displayed_value(product, attr))


# ── §0.3 compensatory weight resolution ─────────────────────────────────────


def _renormalise(weights: dict[str, float]) -> None:
    """Scale a weight dict to sum to 1 in place (no-op if total is 0)."""
    total = sum(weights.values())
    if total > 0:
        for attr in weights:
            weights[attr] /= total


def compensatory_weights(
    attribute_weights: dict[str, float] | None,
    inspected_attrs: list[str],
    board_attrs: list[str],
    price_sensitivity: float,
    brand_loyalty: float,
    brand_lean: float,
) -> dict[str, float]:
    """§0.3 per-inspected-attribute weights for the compensatory utility.

    Two modes (the caller then computes ū = Σ w_a·g_a / Σ w_a over inspected):

    - ``attribute_weights`` present: explicitly-named inspected attributes use
      their stated weight; the ``other`` mass is split equally across inspected
      attributes that are NOT explicitly named. If every inspected attribute is
      named, ``other`` is dropped. §0.4 ``brand_lean`` shifts ``w_brand`` before
      the caller's renormalisation.
    - ``attribute_weights`` absent (default-weight rule): uniform ``1/n`` over
      the board, ``w_price`` scaled by ``(1+price_sensitivity)``, ``w_brand`` by
      ``(1+brand_loyalty)``, §0.4 ``brand_lean`` shift on ``w_brand``, board-
      renormalised to sum 1, then restricted to the inspected attributes.

    Returns a weight dict keyed by ``inspected_attrs``.
    """
    if attribute_weights:
        named = {a: attribute_weights[a] for a in inspected_attrs if a in attribute_weights}
        unnamed = [a for a in inspected_attrs if a not in attribute_weights]
        other_mass = attribute_weights.get("other", 0.0)
        weights = dict(named)
        if unnamed:  # split 'other' across inspected unnamed attrs
            share = other_mass / len(unnamed)
            for attr in unnamed:
                weights[attr] = share
        # §0.4 z.brand_lean shift on w_brand before the caller renormalises
        if "brand" in weights:
            weights["brand"] = max(0.0, weights["brand"] + 0.25 * brand_lean)
        return weights

    # Default-weight rule (§0.3), evaluated over the full board then restricted.
    n = len(board_attrs)
    board_weights = {attr: 1.0 / n for attr in board_attrs}
    if "price" in board_weights:
        board_weights["price"] *= 1.0 + price_sensitivity
    if "brand" in board_weights:
        board_weights["brand"] *= 1.0 + brand_loyalty
    if "brand" in board_weights:  # §0.4 z.brand_lean shift before board renorm
        board_weights["brand"] = max(0.0, board_weights["brand"] + 0.25 * brand_lean)
    _renormalise(board_weights)
    return {attr: board_weights[attr] for attr in inspected_attrs}


def compensatory_utility(
    product: Product,
    inspected_attrs: list[str],
    attribute_weights: dict[str, float] | None,
    board_attrs: list[str],
    price_sensitivity: float,
    brand_loyalty: float,
    brand_lean: float,
) -> float:
    """§0.3 compensatory utility ū = Σ w_a·g_a / Σ w_a over inspected attributes."""
    weights = compensatory_weights(
        attribute_weights,
        inspected_attrs,
        board_attrs,
        price_sensitivity,
        brand_loyalty,
        brand_lean,
    )
    numerator = sum(weights[a] * goodness_of(product, a) for a in inspected_attrs)
    denominator = sum(weights[a] for a in inspected_attrs)
    return numerator / denominator if denominator > 0 else 0.0

