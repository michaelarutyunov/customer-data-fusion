"""
generator/pipeline.py

Orchestrates all modality generators for N participants over a temporal horizon.

For each participant:
  1. sample_persona(archetype_id, seed)          — baseline PersonaConfig
  2. sample_temporal_trajectory(config, n_months) — month-by-month drifted configs
  3. For each month 1..n_months:
     a. Get month-specific config from trajectory
     b. Generate transactions, clickstream, campaigns (event-stream, every month)
     c. Generate traces ONLY if in coverage subset AND month in (1, 2)
     d. Generate psychographics ONLY at months 1 and 7
     e. Generate narrative ONLY at month 1
  4. validate_participant(...) with baseline (month 1) data
  5. Write merged + month-partitioned JSONL outputs

Output:
  Merged:   data/synthetic/{traces,trials,transactions,psychographics,narratives,
            clickstream,campaigns,participant_configs}.jsonl
  Monthly:  data/synthetic/{transactions,clickstream,campaigns}_month_{MM}.jsonl

Usage:
    uv run python -m generator.pipeline --n 100
    uv run python -m generator.pipeline --n 100 --archetypes price_lex compensatory
    uv run python -m generator.pipeline --n 10 --seed 42 --category electronics
    uv run python -m generator.pipeline --n 1000 --trace-coverage 250
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from enum import Enum
from pathlib import Path

import numpy as np
import structlog

from schemas.persona import InspectionDepth, PersonaConfig
from generator.campaign_generator import simulate_campaigns
from generator.clickstream_generator import simulate_clickstream
from generator.persona_sampler import (
    get_drift_metadata,
    list_archetype_ids,
    sample_persona,
    sample_temporal_trajectory,
)
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


def _to_json_with_month(obj: object, month: int) -> str:
    """Serialise a dataclass to JSON and inject a ``month`` field."""
    data: dict = dataclasses.asdict(obj)  # type: ignore[arg-type]

    def _convert_enums(o: object) -> object:
        if isinstance(o, Enum):
            return o.value
        if isinstance(o, dict):
            return {k: _convert_enums(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_convert_enums(v) for v in o]
        return o

    data = _convert_enums(data)  # type: ignore[assignment]
    data["month"] = month
    return json.dumps(data)


def _select_trace_coverage_subset(
    participant_ids: list[str],
    archetype_ids: list[str],
    trace_coverage: int,
    rng: np.random.Generator,
) -> set[str]:
    """Select participants for trace coverage via stratified sampling.

    Distributes ``trace_coverage`` slots equally across archetypes,
    distributing the remainder across the first few archetypes.
    """
    unique_archetypes = list(dict.fromkeys(archetype_ids))  # preserve order
    n_archetypes = len(unique_archetypes)
    base_per = trace_coverage // n_archetypes
    remainder = trace_coverage - base_per * n_archetypes

    # Group participant IDs by archetype
    by_archetype: dict[str, list[str]] = {a: [] for a in unique_archetypes}
    for pid, aid in zip(participant_ids, archetype_ids):
        by_archetype[aid].append(pid)

    selected: set[str] = set()
    for idx, archetype in enumerate(unique_archetypes):
        n_select = base_per + (1 if idx < remainder else 0)
        pool = by_archetype[archetype]
        if len(pool) <= n_select:
            selected.update(pool)
        else:
            chosen = rng.choice(pool, size=n_select, replace=False)
            selected.update(chosen)
    return selected


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
    trace_coverage: int = 250,
    counterfactual_overrides: dict[str, dict[str, float]] | None = None,
) -> dict[str, int]:
    """
    Generate synthetic data for ``n`` participants over a temporal horizon.

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
        Temporal horizon in months (default 12).
    output_dir:
        Directory for JSONL outputs.
    skip_narratives:
        Skip LLM narrative generation (useful for fast dry-runs).
    n_per_archetype:
        If set, derives n = n_per_archetype * len(active_archetypes),
        overriding the ``n`` argument. Produces a balanced dataset.
    only_narratives:
        Generate only narratives.jsonl, leaving all other output files untouched.
        Participant IDs are derived deterministically (same logic as a full run) so
        they will match existing traces/transactions/psychographics produced with the
        same --n / --n-per-archetype and --seed values.
    trace_coverage:
        Number of participants who receive process traces (months 1-2 only).
        Remaining participants have traces modality missing (natural missingness).
        Selected via stratified sampling for equal archetype representation.
    counterfactual_overrides:
        Optional dict mapping participant_id to {field_name: new_value}.  Applied
        after PersonaConfig construction but before any modality generators.  Only
        the 6 float fields in COUNTERFACTUAL_FIELDS are supported; raises ValueError
        for unknown field names.  Participants not listed are unaffected.

    Returns
    -------
    counts: dict mapping file stem to number of rows written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_archetypes = list_archetype_ids()
    active_archetypes = archetypes if archetypes else all_archetypes

    if n_per_archetype is not None:
        n = n_per_archetype * len(active_archetypes)

    # ------------------------------------------------------------------
    # Phase 1: determine participant IDs and trace-coverage subset
    # ------------------------------------------------------------------
    per_archetype_counter: dict[str, int] = {a: 0 for a in active_archetypes}
    participant_ids: list[str] = []
    participant_archetype_ids: list[str] = []

    for i in range(n):
        archetype_id = active_archetypes[i % len(active_archetypes)]
        archetype_idx = per_archetype_counter[archetype_id]
        pid = f"{archetype_id}_{archetype_idx:04d}"
        per_archetype_counter[archetype_id] += 1
        participant_ids.append(pid)
        participant_archetype_ids.append(archetype_id)

    rng = np.random.default_rng(base_seed)
    trace_subset = (
        _select_trace_coverage_subset(
            participant_ids, participant_archetype_ids, trace_coverage, rng
        )
        if not only_narratives
        else set()
    )

    # ------------------------------------------------------------------
    # Phase 2: open output file handles
    # ------------------------------------------------------------------
    if only_narratives:
        handles: dict[str, object] = {
            "narratives": open(output_dir / "narratives.jsonl", "w"),
        }
        month_handles: dict[str, object] = {}
    else:
        handles = {
            "traces": open(output_dir / "traces.jsonl", "w"),
            "trials": open(output_dir / "trials.jsonl", "w"),
            "transactions": open(output_dir / "transactions.jsonl", "w"),
            "psychographics": open(output_dir / "psychographics.jsonl", "w"),
            "participant_configs": open(output_dir / "participant_configs.jsonl", "w"),
            "clickstream": open(output_dir / "clickstream.jsonl", "w"),
            "campaigns": open(output_dir / "campaigns.jsonl", "w"),
        }
        # Only open narratives.jsonl when actually generating. With
        # --skip-narratives we leave any existing file untouched instead of
        # truncating it (the write path is guarded by `not skip_narratives`).
        if not skip_narratives:
            handles["narratives"] = open(output_dir / "narratives.jsonl", "w")
        month_handles = {}
        for month in range(1, n_months + 1):
            mm = f"{month:02d}"
            month_handles[f"transactions_{mm}"] = open(
                output_dir / f"transactions_month_{mm}.jsonl", "w"
            )
            month_handles[f"clickstream_{mm}"] = open(
                output_dir / f"clickstream_month_{mm}.jsonl", "w"
            )
            month_handles[f"campaigns_{mm}"] = open(
                output_dir / f"campaigns_month_{mm}.jsonl", "w"
            )

    counts: dict[str, int] = {k: 0 for k in handles}
    counts["narrative_failures"] = 0
    # Keep the narratives count key present in the report even when generation
    # is skipped (no handle opened in that case).
    counts.setdefault("narratives", 0)
    n_validation_failures = 0

    # ------------------------------------------------------------------
    # Phase 3: generate data per participant
    # ------------------------------------------------------------------
    try:
        for i, (participant_id, archetype_id) in enumerate(
            zip(participant_ids, participant_archetype_ids)
        ):
            seed = base_seed + i
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

            # Temporal trajectory: list[PersonaConfig] with index 0=baseline
            trajectory = sample_temporal_trajectory(
                config, n_months=n_months, random_seed=seed
            )

            # Drift detection — ground-truth regime shift metadata from trajectory
            drift_label, drift_month = get_drift_metadata(
                config, n_months=n_months, random_seed=seed
            )

            has_traces = participant_id in trace_subset
            baseline_config = trajectory[0]

            # Accumulators for baseline-month validation
            baseline_trials: list = []
            baseline_transactions: list = []
            baseline_psychographic = None
            narrative = None

            if only_narratives:
                # Narrative-only mode: generate at month 1
                try:
                    narrative = generate_narrative(
                        baseline_config,
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

                if narrative is not None:
                    handles["narratives"].write(  # type: ignore[union-attr]
                        _to_json_with_month(narrative, 1) + "\n"
                    )
                    counts["narratives"] += 1
            else:
                for month in range(1, n_months + 1):
                    month_config = trajectory[month]

                    # --- Transactions (event-stream: every month) ---
                    month_transactions = simulate_transactions(
                        month_config,
                        category=category,
                        n_months=1,
                        participant_id=participant_id,
                    )
                    mm = f"{month:02d}"
                    for tx in month_transactions:
                        line = _to_json_with_month(tx, month)
                        handles["transactions"].write(line + "\n")  # type: ignore[union-attr]
                        month_handles[f"transactions_{mm}"].write(line + "\n")  # type: ignore[index]
                        counts["transactions"] += 1
                    if month == 1:
                        baseline_transactions = month_transactions

                    # --- Clickstream (event-stream: every month) ---
                    click_events, click_summaries = simulate_clickstream(
                        month_config,
                        participant_id=participant_id,
                        month=month,
                        random_seed=seed + month,
                    )
                    for cev in click_events:
                        line = _to_json(cev)
                        handles["clickstream"].write(line + "\n")  # type: ignore[union-attr]
                        month_handles[f"clickstream_{mm}"].write(line + "\n")  # type: ignore[index]
                        counts["clickstream"] += 1
                    for summ in click_summaries:
                        line = _to_json(summ)
                        handles["clickstream"].write(line + "\n")  # type: ignore[union-attr]
                        month_handles[f"clickstream_{mm}"].write(line + "\n")  # type: ignore[index]
                        counts["clickstream"] += 1

                    # --- Campaigns (event-stream: every month) ---
                    campaign_events = simulate_campaigns(
                        month_config,
                        participant_id=participant_id,
                        n_months=n_months,
                        month=month,
                        random_seed=seed + month,
                    )
                    for cev in campaign_events:
                        line = _to_json(cev)
                        handles["campaigns"].write(line + "\n")  # type: ignore[union-attr]
                        month_handles[f"campaigns_{mm}"].write(line + "\n")  # type: ignore[index]
                        counts["campaigns"] += 1

                    # --- Traces (snapshot: months 1-2, coverage subset only) ---
                    if has_traces and month in (1, 2):
                        events, trials = simulate_session(
                            month_config,
                            category=category,
                            n_trials=n_trials,
                            participant_id=participant_id,
                        )
                        for event in events:
                            handles["traces"].write(  # type: ignore[union-attr]
                                _to_json_with_month(event, month) + "\n"
                            )
                            counts["traces"] += 1
                        for trial in trials:
                            handles["trials"].write(  # type: ignore[union-attr]
                                _to_json_with_month(trial, month) + "\n"
                            )
                            counts["trials"] += 1
                        if month == 1:
                            baseline_trials = trials

                    # --- Psychographics (snapshot: months 1 and 7) ---
                    if month in (1, 7):
                        psychographic = generate_psychographic(
                            month_config,
                            category=category,
                            participant_id=participant_id,
                        )
                        handles["psychographics"].write(  # type: ignore[union-attr]
                            _to_json_with_month(psychographic, month) + "\n"
                        )
                        counts["psychographics"] += 1
                        if month == 1:
                            baseline_psychographic = psychographic

                    # --- Narrative (snapshot: month 1 only) ---
                    if month == 1 and not skip_narratives:
                        try:
                            narrative = generate_narrative(
                                month_config,
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

                        if narrative is not None:
                            handles["narratives"].write(  # type: ignore[union-attr]
                                _to_json_with_month(narrative, 1) + "\n"
                            )
                            counts["narratives"] += 1

                # --- Participant config with drift labels ---
                for month in range(1, n_months + 1):
                    month_config = trajectory[month]
                    participant_config = {
                        "participant_id": participant_id,
                        "month": month,
                        "random_seed": seed,
                        "drift_label": drift_label,
                        "drift_month": drift_month,
                        "price_sensitivity": month_config.transactions.price_sensitivity,
                        "brand_loyalty": month_config.transactions.brand_loyalty,
                        "inspection_depth": _inspection_depth_to_float(
                            month_config.strategy.inspection_depth
                        ),
                        "maximiser_score": month_config.psychographic.maximiser_score,
                        "involvement_score": month_config.psychographic.involvement_score,
                        "risk_tolerance": month_config.psychographic.risk_tolerance,
                        "p_strategy_lapse": month_config.strategy.p_strategy_lapse,
                    }
                    handles["participant_configs"].write(  # type: ignore[union-attr]
                        json.dumps(participant_config) + "\n"
                    )
                    counts["participant_configs"] += 1

                # --- Validation (baseline month only) ---
                if narrative is not None and baseline_psychographic is not None:
                    report = validate_participant(
                        baseline_config,
                        baseline_trials,
                        baseline_transactions,
                        baseline_psychographic,
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

            # Flush after each participant so partial progress is visible on disk
            for fh in handles.values():
                fh.flush()  # type: ignore[union-attr]
            for fh in month_handles.values():
                fh.flush()  # type: ignore[union-attr]

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
            fh.close()  # type: ignore[union-attr]
        for fh in month_handles.values():
            fh.close()  # type: ignore[union-attr]

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
        help="Participants per archetype (e.g. 143 -> 1001 total). Overrides --n.",
    )
    parser.add_argument(
        "--trace-coverage",
        type=int,
        default=250,
        help="Number of participants who receive process traces (default: 250). "
        "Remaining participants have traces modality missing.",
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
        trace_coverage=args.trace_coverage,
    )
    for modality, count in counts.items():
        print(f"  {modality}: {count} rows")
    sys.exit(0)
