"""Choice data loading module — builds flat training table from ChoiceSet + CDT cache."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

# §0.1 board — the full 8-attribute product vector for the product tower (bead
# 6ca). Inlined here rather than imported from generator.choice_model to respect
# the applications↛generator module boundary; the encodings mirror that module.
_BOARD_ATTRIBUTES: tuple[str, ...] = (
    "price",
    "brand",
    "quality",
    "warranty",
    "rating",
    "features",
    "availability",
    "design",
)
_BRAND_TIER_LEVEL: dict[str, float] = {
    "premium": 1.0,
    "mid": 0.66,
    "value": 0.33,
    "own_label": 0.0,
}
PRODUCT_DIM: int = len(_BOARD_ATTRIBUTES)  # 8


def _encode_product_features(product: dict) -> list[float]:
    """§0.1 board encoding of a product → 8 floats in board order.

    This is the product's full displayed-attribute vector (the inspected
    ``displayed_attributes`` of a trial are a prefix of it), so the product tower
    sees every attribute that can drive a choice — not just price + quality.
    """
    return [
        float(product["price_normalised"]),
        _BRAND_TIER_LEVEL[product["brand_tier"]],
        float(product["quality_score"]),
        float(product["warranty_score"]),
        float(product["rating"]) / 5.0,
        float(product["features_score"]),
        1.0 if product["availability"] else 0.0,
        float(product["design_score"]),
    ]



def build_choice_training_table(
    choice_sets_path: Path = Path("data/synthetic/choice_sets.jsonl"),
    products_path: Path = Path("data/synthetic/products.jsonl"),
    cdt_cache_path: Path = Path("applications/_cache/cdt_embeddings.parquet"),
    trace_coverage_path: Path = Path("applications/choice/trace_coverage_participants.txt"),
    output_path: Path = Path("applications/choice/choice_training.parquet"),
) -> None:
    """
    Build flat training table: (participant, cdt, product_features, chosen).

    Creates N rows per trial (1 chosen + N-1 rejected) for binary classification.
    Loads CDT embeddings from cache and product features from choice sets.
    Only includes participants with trace coverage to ensure CDT embeddings contain decision process information.
    """
    print(f"Loading trace coverage participants from {trace_coverage_path}")
    with open(trace_coverage_path) as f:
        trace_coverage_participants = set(line.strip() for line in f)
    print(f"Found {len(trace_coverage_participants)} participants with trace coverage")

    print(f"Loading CDT embeddings from {cdt_cache_path}")
    cdt_table = pq.read_table(cdt_cache_path)
    cdt_df = cdt_table.to_pandas()
    print(f"Loaded {len(cdt_df)} CDT embeddings")

    print(f"Loading products from {products_path}")
    products = {}
    with open(products_path) as f:
        for line in f:
            product = json.loads(line)
            products[product["product_id"]] = product
    print(f"Loaded {len(products)} products")

    print(f"Loading choice sets from {choice_sets_path}")
    rows = []

    with open(choice_sets_path) as f:
        for line in f:
            choice_set = json.loads(line)
            # Remove month field if present (injected by pipeline)
            choice_set.pop("month", None)

            # Only include participants with trace coverage
            if choice_set["participant_id"] not in trace_coverage_participants:
                continue

            # Get CDT embedding for this participant
            cdt_row = cdt_df[cdt_df["participant_id"] == choice_set["participant_id"]]

            if len(cdt_row) == 0:
                # Skip if no CDT embedding found (shouldn't happen with correct data)
                continue

            cdt_embedding = cdt_row.iloc[0]["cdt"]

            # Create N rows per trial (1 chosen + N-1 rejected)
            for slot, product_id in choice_set["alternative_products"].items():
                product = products.get(product_id)
                if product is None:
                    continue

                # §0.1 full-board product features (8 attrs, board order).
                product_features = _encode_product_features(product)

                # Binary label: chosen or not?
                chosen = (slot == choice_set["chosen_alternative"])

                rows.append({
                    "participant_id": choice_set["participant_id"],
                    "cdt_embedding": cdt_embedding,
                    "product_features": product_features,
                    "chosen": chosen,
                    "choice_set_id": choice_set["choice_set_id"],
                    "slot": slot,
                    "product_id": product_id,
                })

    df = pd.DataFrame(rows)
    print(f"Built training table: {len(df)} rows ({df['chosen'].sum()} chosen, {(len(df) - df['chosen'].sum())} rejected)")

    # Save as Parquet (convert DataFrame to pyarrow Table)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {output_path}")
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path, compression="snappy")
    print("✅ Choice training table saved successfully")


if __name__ == "__main__":
    build_choice_training_table()