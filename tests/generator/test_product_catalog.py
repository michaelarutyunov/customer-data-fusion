"""Tests for generator/product_catalog.py (Phase 0 SPEC §0.6).

Covers the acceptance criteria for bead 7c1: per-category count, exact tier
quota, stable IDs, byte-identical regeneration, and generate-once idempotency.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from generator.product_catalog import (
    CATALOGUE_SEED,
    N_PRODUCTS_PER_CATEGORY,
    generate_product_catalog,
)
from generator.choice_model import BRAND_TIER_LEVEL
from schemas.product import Product

CATEGORIES = ("electronics", "fashion", "home_goods")
EXPECTED_QUOTA = {"premium": 3, "mid": 7, "value": 7, "own_label": 3}


def _generate(category: str, tmp_path: Path, force: bool = False) -> list[Product]:
    return generate_product_catalog(category, tmp_path / "products.jsonl", force=force)


@pytest.mark.parametrize("category", CATEGORIES)
def test_count_per_category(category: str, tmp_path: Path) -> None:
    products = _generate(category, tmp_path)
    assert len(products) == N_PRODUCTS_PER_CATEGORY == 20


@pytest.mark.parametrize("category", CATEGORIES)
def test_exact_tier_quota(category: str, tmp_path: Path) -> None:
    products = _generate(category, tmp_path)
    counts = {tier: 0 for tier in EXPECTED_QUOTA}
    for p in products:
        assert p.brand_tier in EXPECTED_QUOTA
        counts[p.brand_tier] += 1
    assert counts == EXPECTED_QUOTA


@pytest.mark.parametrize("category", CATEGORIES)
def test_stable_ids(category: str, tmp_path: Path) -> None:
    products = _generate(category, tmp_path)
    assert [p.product_id for p in products] == [
        f"{category}_{i:02d}" for i in range(N_PRODUCTS_PER_CATEGORY)
    ]


@pytest.mark.parametrize("category", CATEGORIES)
def test_byte_identical_across_regenerations(category: str, tmp_path: Path) -> None:
    """Same category → same products, independent of the pipeline --seed."""
    first = _generate(category, tmp_path)
    # A second generate (forced) on a fresh file must reproduce byte-identical rows.
    other_path = tmp_path / "other.jsonl"
    second = generate_product_catalog(category, other_path, force=True)
    assert [p.__dict__ for p in first] == [p.__dict__ for p in second]


@pytest.mark.parametrize("category", CATEGORIES)
def test_categories_are_independent(category: str, tmp_path: Path) -> None:
    """Different categories must (almost surely) produce different product sets."""
    target = _generate(category, tmp_path)
    other = _generate("fashion", tmp_path)
    prices_a = [p.price_normalised for p in target]
    prices_b = [p.price_normalised for p in other]
    if category != "fashion":
        assert prices_a != prices_b


def test_idempotent_noop_when_present(tmp_path: Path) -> None:
    first = _generate("electronics", tmp_path)
    # Second call without --force is a no-op: same rows, file untouched.
    second = _generate("electronics", tmp_path)
    assert [p.__dict__ for p in first] == [p.__dict__ for p in second]


def test_force_overwrites(tmp_path: Path) -> None:
    first = _generate("electronics", tmp_path)
    # Force must rewrite deterministically (same seed → same products).
    forced = _generate("electronics", tmp_path, force=True)
    assert [p.__dict__ for p in first] == [p.__dict__ for p in forced]


def test_accumulates_other_categories(tmp_path: Path) -> None:
    """Generating a second category preserves the first category's rows."""
    generate_product_catalog("electronics", tmp_path / "products.jsonl")
    generate_product_catalog("fashion", tmp_path / "products.jsonl")
    products = generate_product_catalog("home_goods", tmp_path / "products.jsonl")
    # All three categories present in the returned home_goods set + on disk there
    # are 60 rows total.
    rows = (tmp_path / "products.jsonl").read_text().strip().split("\n")
    assert len(rows) == 60
    cats = {p.category for p in products}
    assert cats == {"home_goods"}


@pytest.mark.parametrize("category", CATEGORIES)
def test_field_ranges(category: str, tmp_path: Path) -> None:
    products = _generate(category, tmp_path)
    for p in products:
        for attr in (
            "price_normalised",
            "quality_score",
            "warranty_score",
            "features_score",
            "design_score",
        ):
            assert 0.0 <= getattr(p, attr) <= 1.0
        assert 0.0 <= p.rating <= 5.0
        assert p.brand_tier in BRAND_TIER_LEVEL
        assert isinstance(p.availability, bool)
        assert isinstance(p.on_promotion, bool)


def test_seed_is_independent_of_pipeline() -> None:
    """The catalogue seed is a fixed constant, not derived from a pipeline seed."""
    assert CATALOGUE_SEED == 20260101
