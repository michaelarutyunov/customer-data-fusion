"""Product schema — products in the synthetic catalogue (Phase 0 SPEC §0.1).

A stable product identity shared across all participants. The first ``n_attrs``
board attributes — in the fixed §0.1 order ``price, brand, quality, warranty,
rating, features, availability, design`` — are encoded to ``[0, 1]`` floats for
``ChoiceSet.displayed_attributes`` at display time (see
``generator/choice_model.py``). ``on_promotion`` is catalogue metadata that
influences ``price_normalised`` at generation time; it is NOT a board attribute.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Product:
    """A product in the synthetic catalogue.

    Field ranges follow Phase 0 SPEC §0.1/§0.6. The choice model never reads
    these fields directly; it goes through the §0.1 display encoding, which
    keeps the schema a pure data contract.
    """

    product_id: str  # f"{category}_{i:02d}" — stable, category-scoped, unique
    category: str  # "electronics" | "fashion" | "home_goods"
    price_normalised: float  # 0.0 (cheapest) to 1.0 (most expensive); promo-adjusted
    brand_tier: str  # one of: premium, mid, value, own_label
    quality_score: float  # 0.0 (low) to 1.0 (high)
    warranty_score: float  # 0.0 (low) to 1.0 (high)
    rating: float  # 0.0 to 5.0 (star rating)
    features_score: float  # 0.0 (sparse) to 1.0 (feature-rich)
    availability: bool  # True if the product is in stock
    design_score: float  # 0.0 (basic) to 1.0 (premium design)
    on_promotion: bool  # catalogue metadata (not a board attribute)
