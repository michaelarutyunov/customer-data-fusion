"""
evaluation/check_retraining_trigger.py

Checks if acquisition-token distribution has shifted significantly.

This script monitors the acquisition token distribution in generated traces.
If the KL divergence exceeds a threshold (0.05 nats), it signals that the
trace encoder may need retraining.

Usage:
    uv run python -m evaluation.check_retraining_trigger
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_baseline(baseline_path: Path) -> dict:
    """Load baseline acquisition token distribution."""
    try:
        with open(baseline_path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def save_baseline(baseline_path: Path, distribution: dict) -> None:
    """Save current distribution as new baseline."""
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with open(baseline_path, "w") as f:
        json.dump(distribution, f, indent=2)
    print(f"Saved new baseline to {baseline_path}")


def compute_token_distribution(
    traces_path: Path = Path("data/synthetic/traces.jsonl"),
) -> dict[str, int]:
    """Compute acquisition token distribution from traces."""
    from schemas.trace import AcquisitionEvent

    token_counts: dict[str, int] = {}
    total_events = 0

    print(f"Computing token distribution from {traces_path}")
    with open(traces_path) as f:
        for line in f:
            event_dict = json.loads(line)
            event = AcquisitionEvent(**event_dict)

            # Extract token from attribute_id (format: "attribute_name=value")
            if event.attribute_id:
                token = event.attribute_id.split("=")[0]
                token_counts[token] = token_counts.get(token, 0) + 1
                total_events += 1

    print(f"Processed {total_events} acquisition events")
    print(f"Found {len(token_counts)} unique tokens")

    return token_counts


def compute_symmetric_kl_divergence(
    p: dict[str, int],
    q: dict[str, int],
) -> float:
    """Compute symmetric KL divergence between two token distributions."""
    # Normalize to probabilities
    total_p = sum(p.values())
    total_q = sum(q.values())

    if total_p == 0 or total_q == 0:
        return 0.0

    p_probs = {k: v / total_p for k, v in p.items()}
    q_probs = {k: v / total_q for k, v in q.items()}

    # Get all tokens
    all_tokens = set(list(p_probs.keys()) + list(q_probs.keys()))

    # Compute symmetric KL divergence
    kl_div = 0.0
    for token in all_tokens:
        p_i = p_probs.get(token, 1e-10)  # Small epsilon for unseen tokens
        q_i = q_probs.get(token, 1e-10)

        if p_i > 0 and q_i > 0:
            kl_div += p_i * np.log(p_i / q_i)

    # Symmetric: add the reverse direction
    for token in all_tokens:
        p_i = p_probs.get(token, 1e-10)
        q_i = q_probs.get(token, 1e-10)

        if p_i > 0 and q_i > 0:
            kl_div += q_i * np.log(q_i / p_i)

    return kl_div / 2.0


def check_retraining_trigger(
    traces_path: Path = Path("data/synthetic/traces.jsonl"),
    baseline_path: Path = Path("data/synthetic/acquisition_token_baseline.json"),
    threshold: float = 0.05,
) -> float:
    """
    Check if acquisition-token distribution has shifted significantly.

    Returns:
        kl_divergence (symmetric KL divergence in nats)
    """
    # Compute current distribution
    current_dist = compute_token_distribution(traces_path)

    # Try to load baseline
    baseline_dist = load_baseline(baseline_path)

    if baseline_dist is None:
        print("⚠️  Baseline not found. Creating baseline from current distribution.")
        save_baseline(baseline_path, current_dist)
        return 0.0

    # Compute symmetric KL divergence
    kl_div = compute_symmetric_kl_divergence(current_dist, baseline_dist)

    print(f"\n{'='*60}")
    print(f"Acquisition Token Drift Detection")
    print(f"{'='*60}")
    print(f"KL divergence: {kl_div:.4f} nats")
    print(f"Threshold:      {threshold:.4f} nats")
    print(f"{'='*60}")

    if kl_div > threshold:
        print(f"⚠️  ACQUISITION-TOKEN KL DIVERGENCE: {kl_div:.4f} > {threshold:.4f}")
        print("Retrain trace encoder recommended.")
        print("Option: Save current distribution as new baseline with --save-baseline")
    else:
        print(f"✅ ACQUISITION-TOKEN DISTRIBUTION STABLE: {kl_div:.4f} ≤ {threshold:.4f}")
        print("No retraining needed.")

    return kl_div


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check acquisition token drift for retraining trigger"
    )
    parser.add_argument(
        "--traces",
        type=Path,
        default=Path("data/synthetic/traces.jsonl"),
        help="Path to traces.jsonl",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("data/synthetic/acquisition_token_baseline.json"),
        help="Path to baseline token distribution",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="KL divergence threshold for retraining trigger (default: 0.05 nats)",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save current distribution as new baseline (overwrite existing)",
    )

    args = parser.parse_args()

    try:
        if args.save_baseline:
            current_dist = compute_token_distribution(args.traces)
            save_baseline(args.baseline, current_dist)
            print("✅ Baseline updated successfully")
        else:
            kl_div = check_retraining_trigger(
                traces_path=args.traces,
                baseline_path=args.baseline,
                threshold=args.threshold,
            )
            # Exit with error code if retraining is needed
            if kl_div > args.threshold:
                exit(1)
    except Exception as e:
        print(f"❌ ERROR: {e}")
        exit(1)


if __name__ == "__main__":
    main()
