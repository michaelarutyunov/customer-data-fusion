"""Product schema — products in the synthetic catalogue."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Product:
    """Product in the synthetic catalogue.

    Stable product identity shared across all participants.
    Attributes are normalized to [0, 1] for decision processes.
    """

    product_id: str  # e.g. "prod_electronics_premium_001"
    category: str  # "electronics", "fashion", "home_goods"
    price_normalised: float  # 0.0 (cheapest) to 1.0 (most expensive)
    brand: str  # "Sony", "Samsung", "Nike", "Adidas", etc.
    quality_normalised: float  # 0.0 (low quality) to 1.0 (high quality)
    features: List[str]  # ["4K", "HDR", "Smart TV", "NoiseCancel"]
