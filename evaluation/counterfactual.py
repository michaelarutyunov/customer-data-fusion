"""
evaluation/counterfactual.py

Archetype-level counterfactual redistribution simulation (Option A).

For each of three scenarios, applies redistribution rules derived directly
from personas.yaml archetype parameters to predict how each consumer archetype
would shift their choices under the changed choice set.

Approach (Option A): classify existing CDT embeddings → get archetype label →
apply per-archetype redistribution formula.  All rules are grounded in
personas.yaml float params; no invented magic numbers.

Option B (generator re-run) is tracked in bead sei and deferred.

Scenarios
---------
1. price_increase_20pct  — uniform 20% price rise across all alternatives
2. new_entrant           — new option with best-in-class quality, mid-price, unknown brand
3. brand_removal         — the archetype's preferred brand is withdrawn from the choice set

Output
------
dict with keys: scenario_name -> {persona_id -> {stay_share, defect_share, defect_to, n_participants}}

Depends on
----------
models/fusion_embeddings_cache.pt  (written by fusion/train.py)
models/fusion_meta_learner.pt      (written by fusion/train.py)
config/personas.yaml               (archetype definitions)
"""

from __future__ import annotations

from pathlib import Path

import yaml
import torch

from schemas import CHECKPOINT_PATHS
from fusion.meta_learner import LateFusionMetaLearner

CACHE_PATH = Path("models/fusion_embeddings_cache.pt")
FUSION_CHECKPOINT = CHECKPOINT_PATHS.get(
    "fusion", Path("models/fusion_meta_learner.pt")
)
PERSONAS_YAML = Path("config/personas.yaml")

PERSONA_IDS = [
    "price_lex",
    "quality_lex",
    "compensatory",
    "satisficer",
    "brand_affect",
    "low_involve",
    "adaptive",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_cache(cache_path: Path) -> dict:
    # weights_only=False required: cache contains list[str] participant_ids
    return torch.load(cache_path, map_location="cpu", weights_only=False)  # type: ignore[reportPrivateImportUsage]


def _load_model(checkpoint_path: Path) -> LateFusionMetaLearner:
    model = LateFusionMetaLearner()
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


def _load_archetype_params(yaml_path: Path) -> dict[str, dict]:
    """Load personas.yaml and flatten the params needed for redistribution rules."""
    with yaml_path.open() as f:
        raw = yaml.safe_load(f)

    params: dict[str, dict] = {}
    for persona_id, arch in raw["archetypes"].items():
        strategy = arch.get("strategy", {})
        transactions = arch.get("transactions", {})
        psychographic = arch.get("psychographic", {})

        params[persona_id] = {
            "label": arch.get("label", persona_id),
            "primary_strategy": strategy.get("primary_strategy", "unknown"),
            "first_attribute": strategy.get("first_attribute", None),
            "inspection_depth": strategy.get("inspection_depth", "medium"),
            "rejection_threshold_pct": strategy.get("rejection_threshold_pct", 0.5),
            "price_sensitivity": transactions.get("price_sensitivity", 0.5),
            "brand_loyalty": transactions.get("brand_loyalty", 0.5),
            "price_variance_tolerance": transactions.get(
                "price_variance_tolerance", 0.25
            ),
            "involvement_score": psychographic.get("involvement_score", 0.5),
            "openness_to_new": psychographic.get("openness_to_new", 0.5),
        }
    return params


def _participant_counts_by_archetype(cache: dict) -> dict[str, int]:
    """Count participants per archetype from CDT embedding cache labels."""
    labels: torch.Tensor = cache["labels"]  # type: ignore[reportPrivateImportUsage]
    counts: dict[str, int] = {}
    for idx, persona_id in enumerate(PERSONA_IDS):
        counts[persona_id] = int((labels == idx).sum().item())
    return counts


# ---------------------------------------------------------------------------
# Redistribution rules (all derived from personas.yaml params)
# ---------------------------------------------------------------------------


def _scenario_price_increase(params: dict) -> dict:
    """
    Scenario: uniform 20% price increase across all alternatives.

    defect_share = clamp(price_sensitivity * (0.20 / price_variance_tolerance), 0.95)

    A consumer with price_variance_tolerance=0.10 treats a 20% rise as 2x their
    normal tolerance. Multiplied by price_sensitivity this gives the fraction who
    exit their current choice. Consumers with tolerance >= 0.20 defect only weakly.
    """
    ps = params["price_sensitivity"]
    pvt = params["price_variance_tolerance"]
    strategy = params["primary_strategy"]
    first_attr = params.get("first_attribute")

    defect_share = min(ps * (0.20 / pvt), 0.95)

    if strategy == "lexicographic" and first_attr == "price":
        destination = "cheapest_remaining"
    elif strategy == "lexicographic" and first_attr == "quality":
        destination = "best_quality_remaining"
    elif strategy == "affect_heuristic":
        destination = "preferred_brand_regardless_of_price"
    elif strategy in ("compensatory", "adaptive"):
        destination = "best_overall_remaining"
    elif strategy == "satisficing":
        destination = "first_satisfactory_remaining"
    else:
        destination = "random_remaining"

    return {
        "stay_share": round(1.0 - defect_share, 3),
        "defect_share": round(defect_share, 3),
        "defect_to": destination,
    }


def _scenario_new_entrant(params: dict) -> dict:
    """
    Scenario: new entrant with best-in-class quality, mid-price, unknown brand.

    Rule varies by primary_strategy — the new entrant is attractive only if
    the consumer's decision heuristic is compatible with its offer profile.

      price_lex:    mid-price = not cheapest -> low consideration
      quality_lex:  best quality -> high consideration
      brand_affect: unknown brand = disqualified -> openness_to_new * (1 - brand_loyalty)
      compensatory/adaptive: inspect all attributes -> openness_to_new * involvement_score
      satisficing:  meets aspiration levels -> openness_to_new * 0.50
      random:       low engagement -> openness_to_new * 0.40

    All defectors go to the new entrant.
    """
    otn = params["openness_to_new"]
    inv = params["involvement_score"]
    bl = params["brand_loyalty"]
    strategy = params["primary_strategy"]
    first_attr = params.get("first_attribute")

    if strategy == "lexicographic" and first_attr == "price":
        defect_share = otn * 0.15
    elif strategy == "lexicographic" and first_attr == "quality":
        defect_share = otn * 0.80
    elif strategy == "affect_heuristic":
        defect_share = otn * (1.0 - bl)
    elif strategy in ("compensatory", "adaptive"):
        defect_share = otn * inv
    elif strategy == "satisficing":
        defect_share = otn * 0.50
    else:
        defect_share = otn * 0.40

    defect_share = min(defect_share, 0.95)

    return {
        "stay_share": round(1.0 - defect_share, 3),
        "defect_share": round(defect_share, 3),
        "defect_to": "new_entrant",
    }


def _scenario_brand_removal(params: dict) -> dict:
    """
    Scenario: the archetype's preferred brand is removed from the choice set.

    defect_share = brand_loyalty

    The fraction that was anchored to the removed brand must switch. Low-loyalty
    archetypes are barely affected; brand_affect (loyalty=0.85) is severely disrupted.
    """
    bl = params["brand_loyalty"]
    strategy = params["primary_strategy"]
    first_attr = params.get("first_attribute")

    defect_share = min(bl, 0.95)

    if strategy == "lexicographic" and first_attr == "price":
        destination = "cheapest_remaining"
    elif strategy == "lexicographic" and first_attr == "quality":
        destination = "best_quality_remaining"
    elif strategy == "affect_heuristic":
        destination = "no_preferred_alternative"
    elif strategy in ("compensatory", "adaptive"):
        destination = "best_overall_remaining"
    elif strategy == "satisficing":
        destination = "first_satisfactory_remaining"
    else:
        destination = "random_remaining"

    return {
        "stay_share": round(1.0 - defect_share, 3),
        "defect_share": round(defect_share, 3),
        "defect_to": destination,
    }


SCENARIO_RULES: dict[str, object] = {
    "price_increase_20pct": _scenario_price_increase,
    "new_entrant_best_in_class": _scenario_new_entrant,
    "brand_removal": _scenario_brand_removal,
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def simulate(
    cache_path: Path = CACHE_PATH,
    checkpoint_path: Path = FUSION_CHECKPOINT,
    personas_path: Path = PERSONAS_YAML,
) -> dict[str, dict[str, dict]]:
    """
    Run archetype-level counterfactual redistribution simulation.

    Returns
    -------
    dict[scenario_name, dict[persona_id, result_dict]]
    result_dict keys: stay_share, defect_share, defect_to, n_participants, label
    """
    cache = _load_cache(cache_path)
    _load_model(checkpoint_path)  # validates checkpoint is loadable
    archetype_params = _load_archetype_params(personas_path)
    n_by_archetype = _participant_counts_by_archetype(cache)

    results: dict[str, dict[str, dict]] = {}

    for scenario_name, rule_fn in SCENARIO_RULES.items():
        scenario_result: dict[str, dict] = {}
        for persona_id in PERSONA_IDS:
            params = archetype_params[persona_id]
            redistribution = rule_fn(params)  # type: ignore[operator]
            redistribution["n_participants"] = n_by_archetype.get(persona_id, 0)
            redistribution["label"] = params["label"]
            scenario_result[persona_id] = redistribution
        results[scenario_name] = scenario_result

    return results


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_SCENARIO_HEADERS = {
    "price_increase_20pct": "Scenario 1: +20% price (uniform across all alternatives)",
    "new_entrant_best_in_class": "Scenario 2: New entrant (best quality, mid-price, unknown brand)",
    "brand_removal": "Scenario 3: Preferred brand removed from choice set",
}


def format_results(results: dict[str, dict[str, dict]]) -> str:
    lines = []

    for scenario_name, persona_results in results.items():
        lines.append("")
        lines.append("=" * 78)
        lines.append(_SCENARIO_HEADERS.get(scenario_name, scenario_name))
        lines.append("=" * 78)
        header = (
            f"{'Archetype':<26} {'Stay':>6} {'Defect':>8}  {'Defect to':<32} {'N':>4}"
        )
        lines.append(header)
        lines.append("-" * 78)
        for persona_id in PERSONA_IDS:
            r = persona_results[persona_id]
            lines.append(
                f"{r['label']:<26} {r['stay_share']:>6.1%} {r['defect_share']:>8.1%}"
                f"  {r['defect_to']:<32} {r['n_participants']:>4}"
            )

    lines.append("")
    lines.append("─" * 78)
    lines.append("Option A: archetype-level rules derived from personas.yaml params.")
    lines.append("Redistribution is archetype-uniform — individual variation within")
    lines.append(
        "archetype is not modelled. See bead sei for Option B (generator re-run)."
    )
    lines.append("─" * 78)

    return "\n".join(lines)


if __name__ == "__main__":
    results = simulate()
    print(format_results(results))
