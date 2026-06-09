"""
generator/pipeline.py

Orchestrates all modality generators for N participants.

For each participant:
  1. sample_persona(archetype_id, seed)
  2. simulate_session(config)
  3. simulate_transactions(config)
  4. generate_psychographic(config)
  5. generate_narrative(config)
  6. validate_participant(...)
  7. append JSONL rows for all 5 modalities

Output: data/synthetic/{traces,trials,transactions,psychographics,narratives}.jsonl

Usage:
    uv run python -m generator.pipeline --n 100
    uv run python -m generator.pipeline --n 100 --archetypes price_lex compensatory
    uv run python -m generator.pipeline --n 10 --seed 42 --category electronics
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from enum import Enum
from pathlib import Path

import structlog

from schemas.persona import InspectionDepth, PersonaConfig
from generator.persona_sampler import list_archetype_ids, sample_persona
from generator.psychographic_generator import generate_psychographic
from generator.text_generator import generate_narrative
from generator.trace_simulator import simulate_session
from generator.transaction_simulator import simulate_transactions
from generator.validate import validate_participant

log = structlog.get_logger(__name__)

_OUTPUT_DIR = Path("data/synthetic")

# ---------------------------------------------------------------------------
# Counterfactual overrides — flat name → (parent_attr, child_attr) mapping
# ---------------------------------------------------------------------------

COUNTERFACTUAL_FIELDS: dict[str, tuple[str, str]] = {
    "price_sensitivity": ("transactions", "price_sensitivity"),
    "brand_loyalty": ("transactions", "brand_loyalty"),
    "p_strategy_lapse": ("strategy", "p_strategy_lapse"),
    "risk_tolerance": ("psychographic", "risk_tolerance"),
    "maximiser_score": ("psychographic", "maximiser_score"),
    "involvement_score": ("psychographic", "involvement_score"),
    # inspection_depth is NOT overridable: it's an InspectionDepth enum, not a float.
}


def _apply_overrides(
    config: PersonaConfig,
    overrides: dict[str, float],
) -> PersonaConfig:
    """Apply counterfactual overrides to a frozen PersonaConfig.

    Parameters
    ----------
    config : PersonaConfig
        Original (frozen) persona configuration.
    overrides : dict[str, float]
        Flat field name → new float value.  Must be in COUNTERFACTUAL_FIELDS.

    Returns
    -------
    PersonaConfig
        New frozen PersonaConfig with overridden values.

    Raises
    ------
    ValueError
        If an unknown field name is supplied.
    """
    # Validate all keys first
    for field_name in overrides:
        if field_name not in COUNTERFACTUAL_FIELDS:
            raise ValueError(f"Unknown PersonaConfig field: {field_name}")

    # Group overrides by parent attribute to minimise replace() calls
    grouped: dict[str, dict[str, float]] = {}
    for field_name, new_value in overrides.items():
        parent_attr, child_attr = COUNTERFACTUAL_FIELDS[field_name]
        grouped.setdefault(parent_attr, {})[child_attr] = new_value

    # Apply each group: replace nested object, then replace PersonaConfig
    for parent_attr, child_overrides in grouped.items():
        nested_obj = getattr(config, parent_attr)
        new_nested = dataclasses.replace(nested_obj, **child_overrides)
        config = dataclasses.replace(config, **{parent_attr: new_nested})

    return config


def _inspection_depth_to_float(depth: InspectionDepth) -> float:
    """Convert InspectionDepth enum to continuous float for evaluation."""
    mapping = {
        InspectionDepth.SHALLOW: 0.33,
        InspectionDepth.MEDIUM: 0.66,
        InspectionDepth.DEEP: 1.0,
        InspectionDepth.VARIABLE: 0.5,  # average when variable
    }
    return mapping.get(depth, 0.5)


def _to_json(obj: object) -> str:
    """Serialise a dataclass to a JSON line. Enum values use .value."""

    def _default(o: object) -> object:
        if isinstance(o, Enum):
            return o.value
        raise TypeError(f"Object of type {type(o)} is not JSON serialisable")

    return json.dumps(dataclasses.asdict(obj), default=_default)  # type: ignore[arg-type]


def run_pipeline(
    n: int,
    archetypes: list[str] | None = None,
    category: str = "electronics",
    base_seed: int = 0,
    n_trials: int = 20,
    n_months: int = 12,
    output_dir: Path = _OUTPUT_DIR,
    skip_narratives: bool = False,
    n_per_archetype: int | None = None,
    only_narratives: bool = False,
    counterfactual_overrides: dict[str, dict[str, float]] | None = None,
) -> dict[str, int]:
    """
    Generate synthetic data for `n` participants.

    Parameters
    ----------
    n:
        Total participants to generate.
    archetypes:
        Archetype IDs to cycle over. Defaults to all 7.
    category:
        Product category passed to all generators.
    base_seed:
        Random seed offset. Participant i gets seed base_seed+i.
    n_trials:
        Number of MouseLab trials per participant.
    n_months:
        Transaction history window.
    output_dir:
        Directory for JSONL outputs.
    skip_narratives:
        Skip LLM narrative generation (useful for fast dry-runs).
    n_per_archetype:
        If set, derives n = n_per_archetype * len(active_archetypes),
        overriding the `n` argument. Produces a balanced dataset.
    only_narratives:
        Generate only narratives.jsonl, leaving all other output files untouched.
        Participant IDs are derived deterministically (same logic as a full run) so
        they will match existing traces/transactions/psychographics produced with the
        same --n / --n-per-archetype and --seed values.
    counterfactual_overrides:
        Optional dict mapping participant_id → {field_name: new_value}.  Applied
        after PersonaConfig construction but before any modality generators.  Only
        the 6 float fields in COUNTERFACTUAL_FIELDS are supported; raises ValueError
        for unknown field names.  Participants not listed are unaffected.

    Returns
    -------
    counts: dict mapping file stem → number of rows written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_archetypes = list_archetype_ids()
    active_archetypes = archetypes if archetypes else all_archetypes

    if n_per_archetype is not None:
        n = n_per_archetype * len(active_archetypes)

    if only_narratives:
        handles = {
            "narratives": open(output_dir / "narratives.jsonl", "w"),
        }
    else:
        handles = {
            "traces": open(output_dir / "traces.jsonl", "w"),
            "trials": open(output_dir / "trials.jsonl", "w"),
            "transactions": open(output_dir / "transactions.jsonl", "w"),
            "psychographics": open(output_dir / "psychographics.jsonl", "w"),
            "narratives": open(output_dir / "narratives.jsonl", "w"),
            "participant_configs": open(output_dir / "participant_configs.jsonl", "w"),
        }
    counts: dict[str, int] = {k: 0 for k in handles}
    counts["narrative_failures"] = 0
    n_validation_failures = 0

    try:
        per_archetype_counter: dict[str, int] = {a: 0 for a in active_archetypes}
        for i in range(n):
            archetype_id = active_archetypes[i % len(active_archetypes)]
            seed = base_seed + i
            archetype_idx = per_archetype_counter[archetype_id]
            participant_id = f"{archetype_id}_{archetype_idx:04d}"
            per_archetype_counter[archetype_id] += 1

            config = sample_persona(archetype_id, random_seed=seed)

            if counterfactual_overrides and participant_id in counterfactual_overrides:
                config = _apply_overrides(
                    config, counterfactual_overrides[participant_id]
                )
                log.debug(
                    "pipeline.counterfactual_override",
                    participant_id=participant_id,
                    overrides=counterfactual_overrides[participant_id],
                )

            events: list = []
            trials: list = []
            transactions: list = []
            psychographic = None
            narrative = None

            if not only_narratives:
                events, trials = simulate_session(
                    config,
                    category=category,
                    n_trials=n_trials,
                    participant_id=participant_id,
                )
                transactions = simulate_transactions(
                    config,
                    category=category,
                    n_months=n_months,
                    participant_id=participant_id,
                )
                psychographic = generate_psychographic(
                    config,
                    category=category,
                    participant_id=participant_id,
                )

            if only_narratives or not skip_narratives:
                try:
                    narrative = generate_narrative(
                        config,
                        category=category,
                        participant_id=participant_id,
                    )
                except Exception as exc:
                    log.warning(
                        "pipeline.narrative_failed",
                        participant_id=participant_id,
                        archetype=archetype_id,
                        participant_index=i,
                        error=str(exc),
                    )
                    counts["narrative_failures"] += 1
                    narrative = None

            if narrative is not None and not only_narratives:
                assert psychographic is not None
                report = validate_participant(
                    config,
                    trials,
                    transactions,
                    psychographic,
                    narrative,
                    participant_id=participant_id,
                )
                if not report.passed:
                    n_validation_failures += 1
                    log.warning(
                        "pipeline.validation_failure",
                        participant_id=participant_id,
                        archetype=archetype_id,
                        participant_index=i,
                        n_failures=len(report.failures),
                        checks=[f[0] for f in report.failures],
                    )

            if not only_narratives:
                for event in events:
                    handles["traces"].write(_to_json(event) + "\n")
                    counts["traces"] += 1

                for trial in trials:
                    handles["trials"].write(_to_json(trial) + "\n")
                    counts["trials"] += 1

                for tx in transactions:
                    handles["transactions"].write(_to_json(tx) + "\n")
                    counts["transactions"] += 1

                handles["psychographics"].write(_to_json(psychographic) + "\n")
                counts["psychographics"] += 1

                # Write participant config continuous latent variables
                participant_config = {
                    "participant_id": participant_id,
                    "price_sensitivity": config.transactions.price_sensitivity,
                    "brand_loyalty": config.transactions.brand_loyalty,
                    "inspection_depth": _inspection_depth_to_float(
                        config.strategy.inspection_depth
                    ),
                    "maximiser_score": config.psychographic.maximiser_score,
                    "involvement_score": config.psychographic.involvement_score,
                    "risk_tolerance": config.psychographic.risk_tolerance,
                    "p_strategy_lapse": config.strategy.p_strategy_lapse,
                }
                handles["participant_configs"].write(
                    json.dumps(participant_config) + "\n"
                )
                counts["participant_configs"] += 1

            if narrative is not None:
                handles["narratives"].write(_to_json(narrative) + "\n")
                counts["narratives"] += 1

            # Flush after each participant so partial progress is visible on disk
            for fh in handles.values():
                fh.flush()

            if (i + 1) % 100 == 0:
                log.info(
                    "pipeline.progress",
                    completed=i + 1,
                    total=n,
                    validation_failures=n_validation_failures,
                )
            else:
                log.debug("pipeline.participant_done", index=i, archetype=archetype_id)

    finally:
        for fh in handles.values():
            fh.close()

    log.info(
        "pipeline.complete",
        n_participants=n,
        validation_failures=n_validation_failures,
        **{f"n_{k}": v for k, v in counts.items()},
    )
    return counts


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CDT synthetic dataset")
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Number of participants (required if --n-per-archetype not set)",
    )
    parser.add_argument(
        "--archetypes",
        nargs="+",
        default=None,
        help="Archetype IDs to cycle over (default: all 7)",
    )
    parser.add_argument("--category", default="electronics")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-months", type=int, default=12)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_OUTPUT_DIR,
        help="Output directory (default: data/synthetic/)",
    )
    parser.add_argument(
        "--skip-narratives",
        action="store_true",
        help="Skip LLM narrative generation (for fast dry-runs)",
    )
    parser.add_argument(
        "--only-narratives",
        action="store_true",
        help="Generate only narratives.jsonl; leaves traces/transactions/psychographics untouched",
    )
    parser.add_argument(
        "--n-per-archetype",
        type=int,
        default=None,
        help="Participants per archetype (e.g. 143 → 1001 total). Overrides --n.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    if args.n is None and args.n_per_archetype is None:
        print("Error: either --n or --n-per-archetype is required", file=sys.stderr)
        sys.exit(2)
    counts = run_pipeline(
        n=args.n or 0,  # 0 when n_per_archetype derives n
        archetypes=args.archetypes,
        category=args.category,
        base_seed=args.seed,
        n_trials=args.n_trials,
        n_months=args.n_months,
        output_dir=args.output_dir,
        skip_narratives=args.skip_narratives,
        n_per_archetype=args.n_per_archetype,
        only_narratives=args.only_narratives,
    )
    for modality, count in counts.items():
        print(f"  {modality}: {count} rows")
    sys.exit(0)
