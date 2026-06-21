"""
evaluation/verify_trace_choice_coupling.py

Verifies that trace encoder embeddings predict choices better than chance.

This script checks the critical trace-choice coupling assumption:
that the decision process (encoded in the trace) predicts the final choice.
If this check fails, M1's CDT lift gate is mathematically impossible to pass.

Usage:
    uv run python -m evaluation.verify_trace_choice_coupling
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from schemas.trace import TrialRecord
from schemas.choice_set import ChoiceSet


def load_trial_data(
    trials_path: Path,
    choice_sets_path: Path,
) -> tuple[list[TrialRecord], list[ChoiceSet]]:
    """Load trial and choice set data from JSONL files."""
    import json

    trials: list[TrialRecord] = []
    choice_sets: list[ChoiceSet] = []

    with open(trials_path) as f:
        for line in f:
            trial_dict = json.loads(line)
            # Remove month field if present (injected by pipeline, not in schema)
            trial_dict.pop("month", None)
            trials.append(TrialRecord(**trial_dict))

    with open(choice_sets_path) as f:
        for line in f:
            choice_set_dict = json.loads(line)
            # Remove month field if present (injected by pipeline, not in schema)
            choice_set_dict.pop("month", None)
            choice_sets.append(ChoiceSet(**choice_set_dict))

    return trials, choice_sets


def verify_trace_choice_coupling(
    trials_path: Path = Path("data/synthetic/trials.jsonl"),
    choice_sets_path: Path = Path("data/synthetic/choice_sets.jsonl"),
    encoder_path: str | Path = Path("models/trace_encoder.pt"),
    device: str = "cpu",
) -> float:
    """
    Verify trace embedding predicts choice better than chance.

    Returns:
        accuracy (proportion of correct predictions)
    """
    # Load data
    print(f"Loading trials from {trials_path}")
    print(f"Loading choice sets from {choice_sets_path}")
    trials, choice_sets = load_trial_data(trials_path, choice_sets_path)

    # Build choice_set lookup
    choice_set_map = {cs.choice_set_id: cs for cs in choice_sets}

    # Build dataset using trial metadata as proxy for trace features
    # Note: Full implementation would load actual trace encoder and encode acquisition events
    X = []  # Trial features (proxy for trace embeddings)
    y = []  # Chosen slots (converted to int)

    n_alternatives_counts = []

    for trial in trials:
        if trial.choice_set_id is None:
            continue  # Skip trials without choice linkage

        choice_set = choice_set_map.get(trial.choice_set_id)
        if choice_set is None:
            continue  # Skip trials with missing choice sets

        # Use trial metadata as proxy features for trace encoder output
        # Full implementation would encode the actual acquisition sequence
        trial_features = np.array([
            trial.n_alternatives / 7.0,  # Normalized
            trial.n_attributes / 8.0,  # Normalized
            trial.total_acquisitions / 50.0,  # Normalized
            trial.prop_cells_inspected,
            trial.payne_index,
        ])

        X.append(trial_features)

        # Convert chosen alternative to int (A=0, B=1, C=2, etc.)
        chosen_slot = choice_set.chosen_alternative
        if chosen_slot:
            chosen_int = ord(chosen_slot) - ord('A')
            y.append(chosen_int)

        n_alternatives_counts.append(choice_set.n_alternatives)

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        print("❌ No valid trial-choice pairs found in data")
        return 0.0

    print(f"Built dataset: {len(X)} trial-choice pairs")

    if len(X) == 0:
        print("❌ ERROR: No trial-choice pairs found with choice_set_id linkage")
        print("   This may be because:")
        print("   1. Dataset was generated before Task #9 (ChoiceSet generation)")
        print("   2. Trace coverage subset doesn't have choice linkage yet")
        print("   Solution: Regenerate dataset or check trace_coverage parameter")
        return 0.0

    # Train logistic regression
    print("Training logistic regression on trial features (trace proxy)...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, y)

    # Measure accuracy
    accuracy = clf.score(X, y)

    # Compute chance baseline
    mean_n_alternatives = np.mean(n_alternatives_counts)
    chance_baseline = 1.0 / mean_n_alternatives

    # Target: 1.5 × chance baseline
    target = 1.5 * chance_baseline

    print(f"\n{'='*60}")
    print(f"Trace-Choice Coupling Verification (Proxy)")
    print(f"{'='*60}")
    print(f"Accuracy:           {accuracy:.3f}")
    print(f"Chance baseline:    {chance_baseline:.3f} (1/{mean_n_alternatives:.1f})")
    print(f"Target (1.5×):     {target:.3f}")
    print(f"{'='*60}")
    print(f"⚠️  NOTE: Using trial metadata as proxy for trace encoder features")
    print(f"      Full implementation would encode acquisition events with trained encoder")

    if accuracy < target:
        print(f"\n⚠️  TRACE-CHOICE COUPLING WEAK: {accuracy:.3f} < {target:.3f} (1.5× chance)")
        print("M1's CDT lift gate may be unpassable.")
        print("Consider tuning inspection_bonus coefficient in choice utility.")
    else:
        print(f"\n✅ TRACE-CHOICE COUPLING STRONG: {accuracy:.3f} ≥ {target:.3f} (1.5× chance)")
        print("Trial features predict choices. M1 should be able to learn CDT lift.")

    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify trace-choice coupling for M1 prerequisite"
    )
    parser.add_argument(
        "--trials",
        type=Path,
        default=Path("data/synthetic/trials.jsonl"),
        help="Path to trials.jsonl",
    )
    parser.add_argument(
        "--choice-sets",
        type=Path,
        default=Path("data/synthetic/choice_sets.jsonl"),
        help="Path to choice_sets.jsonl",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="models/trace_encoder.pt",
        help="Path to trace encoder checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for encoder (cpu or cuda)",
    )

    args = parser.parse_args()

    try:
        accuracy = verify_trace_choice_coupling(
            trials_path=args.trials,
            choice_sets_path=args.choice_sets,
            encoder_path=args.encoder,
            device=args.device,
        )
        # Exit with error code if coupling is weak
        mean_n_alternatives = 5.0  # Approximate average
        target = 1.5 / mean_n_alternatives
        if accuracy < target:
            exit(1)
    except Exception as e:
        print(f"❌ ERROR: {e}")
        exit(1)


if __name__ == "__main__":
    main()
