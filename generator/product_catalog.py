"""Product catalog generator for synthetic choice trials.

Generates ~81 products across 3 categories with varying brands,
quality levels, and feature sets. Products are the stable choice set
for all participants in M1 choice model experiments.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np

from schemas.product import Product


def _sample_features_for_category(category: str, quality: float) -> List[str]:
    """Sample category-specific features based on quality level.

    Args:
        category: One of "electronics", "fashion", "home_goods"
        quality: Normalised quality 0.0-1.0

    Returns:
        List of feature strings (subset for lower quality)
    """
    # Define all possible features by category
    all_features = {
        "electronics": ["4K", "HDR", "Smart TV", "NoiseCancel"],
        "fashion": ["breathable", "stretch", "moisture-wicking", "sustainable"],
        "home_goods": ["durable", "easy-assembly", "waterproof", "eco-friendly"],
    }

    # Get base feature list for category
    features = all_features.get(category, [])

    # Sample based on quality: higher quality = more features
    # quality=0.0 -> 1 feature, quality=1.0 -> all features
    n_features = int(1 + quality * (len(features) - 1))

    rng = np.random.default_rng()
    selected = rng.choice(features, size=n_features, replace=False).tolist()

    return selected


def generate_product_catalog(
    output_path: Path = Path("data/synthetic/products.jsonl"),
    seed: int = 42,
) -> List[Product]:
    """Generate product catalog for synthetic choice trials.

    Creates ~81 products across:
    - 3 categories (electronics, fashion, home_goods)
    - 3 brands per category
    - 3 quality levels (0.0, 0.5, 1.0)
    - 3 alternative set configurations

    Args:
        output_path: Where to write products.jsonl
        seed: Random seed for reproducibility

    Returns:
        List of generated Product objects
    """
    rng = np.random.default_rng(seed)

    # Define brands by category
    brands_by_category = {
        "electronics": ["Sony", "Samsung", "LG"],
        "fashion": ["Nike", "Adidas", "Under Armour"],
        "home_goods": ["IKEA", "Target", "Walmart"],
    }

    categories = list(brands_by_category.keys())
    n_quality_levels = 3
    n_alternative_configs = 3

    products: List[Product] = []

    # Generate products
    for category in categories:
        for brand in brands_by_category[category]:
            for quality_idx in range(n_quality_levels):
                for alt_idx in range(n_alternative_configs):
                    # Normalised quality: 0.0, 0.5, 1.0
                    quality_normalised = quality_idx / (n_quality_levels - 1)

                    # Sample features based on quality
                    features = _sample_features_for_category(
                        category, quality_normalised
                    )

                    # Random normalised price 0.1-0.9
                    price_normalised = float(rng.uniform(0.1, 0.9))

                    # Unique product ID
                    product_id = f"prod_{category}_{brand}_{quality_idx}_{alt_idx}"

                    product = Product(
                        product_id=product_id,
                        category=category,
                        price_normalised=price_normalised,
                        brand=brand,
                        quality_normalised=quality_normalised,
                        features=features,
                    )
                    products.append(product)

    # Write to JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for product in products:
            f.write(json.dumps(asdict(product)) + "\n")

    return products


if __name__ == "__main__":
    products = generate_product_catalog()
    print(f"Generated {len(products)} products")
    print(f"Sample product: {products[0].product_id}")
