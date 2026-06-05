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

from generator.persona_sampler import list_archetype_ids, sample_persona
from generator.psychographic_generator import generate_psychographic
from generator.text_generator import generate_narrative
from generator.trace_simulator import simulate_session
from generator.transaction_simulator import simulate_transactions
from generator.validate import validate_participant

log = structlog.get_logger(__name__)

_OUTPUT_DIR = Path("data/synthetic")


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

    Returns
    -------
    counts: dict mapping file stem → number of rows written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_archetypes = list_archetype_ids()
    active_archetypes = archetypes if archetypes else all_archetypes

    handles = {
        "traces":         open(output_dir / "traces.jsonl", "w"),
        "trials":         open(output_dir / "trials.jsonl", "w"),
        "transactions":   open(output_dir / "transactions.jsonl", "w"),
        "psychographics": open(output_dir / "psychographics.jsonl", "w"),
        "narratives":     open(output_dir / "narratives.jsonl", "w"),
    }
    counts: dict[str, int] = {k: 0 for k in handles}
    n_validation_failures = 0

    try:
        for i in range(n):
            archetype_id = active_archetypes[i % len(active_archetypes)]
            seed = base_seed + i

            config = sample_persona(archetype_id, random_seed=seed)

            events, trials = simulate_session(config, category=category, n_trials=n_trials)
            transactions = simulate_transactions(config, category=category, n_months=n_months)
            psychographic = generate_psychographic(config, category=category)

            if skip_narratives:
                narrative = None
            else:
                narrative = generate_narrative(config, category=category)

            if narrative is not None:
                report = validate_participant(config, trials, transactions, psychographic, narrative)
                if not report.passed:
                    n_validation_failures += 1
                    log.warning(
                        "pipeline.validation_failure",
                        participant_id=config.persona_id,
                        archetype=archetype_id,
                        participant_index=i,
                        n_failures=len(report.failures),
                        checks=[f[0] for f in report.failures],
                    )

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

            if narrative is not None:
                handles["narratives"].write(_to_json(narrative) + "\n")
                counts["narratives"] += 1

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
    parser.add_argument("--n", type=int, required=True, help="Number of participants")
    parser.add_argument(
        "--archetypes", nargs="+", default=None,
        help="Archetype IDs to cycle over (default: all 7)"
    )
    parser.add_argument("--category", default="electronics")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-months", type=int, default=12)
    parser.add_argument(
        "--output-dir", type=Path, default=_OUTPUT_DIR,
        help="Output directory (default: data/synthetic/)"
    )
    parser.add_argument(
        "--skip-narratives", action="store_true",
        help="Skip LLM narrative generation (for fast dry-runs)"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    counts = run_pipeline(
        n=args.n,
        archetypes=args.archetypes,
        category=args.category,
        base_seed=args.seed,
        n_trials=args.n_trials,
        n_months=args.n_months,
        output_dir=args.output_dir,
        skip_narratives=args.skip_narratives,
    )
    for modality, count in counts.items():
        print(f"  {modality}: {count} rows")
    sys.exit(0)
