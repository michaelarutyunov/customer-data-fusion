"""Product catalogue generator — Phase 0 SPEC §0.6.

One run produces **one category's** catalogue (matching the pipeline's single
``--category`` contract). Output ``data/synthetic/products.jsonl`` holds one
record per line, accumulated across categories.

Reproducibility & idempotency (required):
  - Generated from a fixed ``CATALOGUE_SEED`` (independent of the pipeline
    ``--seed``), so a category's products are byte-identical across regenerations
    and across H1 waves.
  - Generate-once per category: if ``products.jsonl`` already holds rows for the
    target category, the run is a no-op. ``--force`` overwrites that category's
    rows deterministically (same seed → same products).
  - The product RNG is seeded from ``CATALOGUE_SEED + stable_hash(category)`` so
    categories are independent but each is stable. We use ``hashlib`` (not the
    builtin ``hash``) because the builtin is salted per process and would break
    byte-identical regeneration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from schemas.product import Product
from generator.choice_model import BRAND_TIER_LEVEL

# Fixed, independent of the pipeline --seed.
CATALOGUE_SEED: int = 20260101
N_PRODUCTS_PER_CATEGORY: int = 20

# Brand-tier quota at N=20 → exactly 3 / 7 / 7 / 3 (assigned by quota, not sampled).
_TIER_QUOTA: dict[str, int] = {"premium": 3, "mid": 7, "value": 7, "own_label": 3}

# Fixed tier order for product indices 0..19 (deterministic given only the index).
_TIER_ORDER: tuple[str, ...] = ("premium", "mid", "value", "own_label")

CATEGORIES: tuple[str, ...] = ("electronics", "fashion", "home_goods")


def _stable_category_hash(category: str) -> int:
    """Process-independent hash of a category name (builtin hash() is salted)."""
    digest = hashlib.sha256(category.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _tier_sequence() -> list[str]:
    """Per-product tier labels for indices 0..N-1, in fixed _TIER_ORDER."""
    seq: list[str] = []
    for tier in _TIER_ORDER:
        seq.extend([tier] * _TIER_QUOTA[tier])
    assert len(seq) == N_PRODUCTS_PER_CATEGORY
    return seq


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(min(max(x, lo), hi))


def _generate_category_products(category: str) -> list[Product]:
    """Generate N_PRODUCTS_PER_CATEGORY products for one category (SPEC §0.6).

    The draw order (q, price, quality, features, design, warranty, rating,
    availability, on_promotion) is fixed, so given the category-seeded RNG the
    output is byte-identical across runs.
    """
    rng = np.random.default_rng(CATALOGUE_SEED + _stable_category_hash(category))
    products: list[Product] = []

    for i, tier in enumerate(_tier_sequence()):
        tier_level = BRAND_TIER_LEVEL[tier]
        q = float(rng.uniform(0.0, 1.0))  # latent quality factor

        price = _clip(0.15 + 0.7 * tier_level + 0.15 * q + rng.normal(0.0, 0.05))
        quality = _clip(0.4 * q + 0.4 * tier_level + rng.normal(0.0, 0.1))
        features = _clip(0.4 * q + 0.4 * tier_level + rng.normal(0.0, 0.1))
        design = _clip(0.4 * q + 0.4 * tier_level + rng.normal(0.0, 0.1))
        warranty = _clip(0.5 * tier_level + rng.normal(0.0, 0.15))
        rating = _clip(2.5 + 2.0 * q + rng.normal(0.0, 0.3), 0.0, 5.0)
        availability = bool(rng.random() < 0.95)
        on_promotion = bool(rng.random() < 0.15)
        if on_promotion:
            price = _clip(price * 0.85)

        products.append(
            Product(
                product_id=f"{category}_{i:02d}",
                category=category,
                price_normalised=round(price, 6),
                brand_tier=tier,
                quality_score=round(quality, 6),
                warranty_score=round(warranty, 6),
                rating=round(rating, 6),
                features_score=round(features, 6),
                availability=availability,
                design_score=round(design, 6),
                on_promotion=on_promotion,
            )
        )

    return products


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=False) + "\n")


def generate_product_catalog(
    category: str,
    output_path: Path = Path("data/synthetic/products.jsonl"),
    force: bool = False,
) -> list[Product]:
    """Generate (or no-op) one category's catalogue per SPEC §0.6.

    Generate-once: if ``output_path`` already contains rows for ``category`` and
    ``force`` is False, this is a no-op and returns those existing products.
    With ``force`` (or if the category is absent), the category's rows are
    regenerated deterministically; other categories already in the file are
    preserved untouched.

    Returns the category's products (freshly generated, or existing on a no-op).
    """
    existing = _read_rows(output_path)
    own_rows = [r for r in existing if r.get("category") == category]

    if own_rows and not force:
        return [Product(**r) for r in own_rows]

    new_products = _generate_category_products(category)
    other_rows = [r for r in existing if r.get("category") != category]
    _write_rows(output_path, other_rows + [asdict(p) for p in new_products])
    return new_products


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate the product catalogue (§0.6)"
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Single category to (re)generate. Default: all of electronics, fashion, home_goods.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the category's existing rows deterministically.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic/products.jsonl"),
    )
    args = parser.parse_args(argv)

    categories = [args.category] if args.category else list(CATEGORIES)
    for category in categories:
        products = generate_product_catalog(category, args.output, force=args.force)
        print(
            f"{category}: {len(products)} products ({products[0].product_id} … {products[-1].product_id})"
        )


if __name__ == "__main__":
    _main()
