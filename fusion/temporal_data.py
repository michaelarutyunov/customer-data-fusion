"""
fusion/temporal_data.py

Generate monthly temporal embeddings cache for fusion training.

For each participant, generates 12 monthly embedding sequences using the
frozen encoders. These are used as temporal contrastive loss targets during
fusion training.

Usage:
    uv run python -m fusion.temporal_data
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from schemas import EMBEDDING_DIM


def generate_monthly_temporal_embeddings(
    monthly_data_path: Path,
    output_path: Path,
    device: str = "cpu",
) -> None:
    """Generate temporal embeddings cache from monthly data.

    Args:
        monthly_data_path: Path to monthly observations JSONL
        output_path: Where to save the temporal embeddings cache
        device: Device to run on
    """
    print(f"Loading monthly data from {monthly_data_path}...")
    with open(monthly_data_path) as f:
        monthly_records = [json.loads(line) for line in f]

    # Group by participant_id
    from collections import defaultdict

    monthly_by_participant = defaultdict(list)
    for record in monthly_records:
        monthly_by_participant[record["participant_id"]].append(record)

    # Sort by month and verify we have 12 months for each participant
    participant_ids = []
    monthly_embeddings_list = []

    for pid in sorted(monthly_by_participant.keys()):
        records = sorted(monthly_by_participant[pid], key=lambda r: r["month"])
        if len(records) != 12:
            print(f"⚠️  Participant {pid} has {len(records)} months, skipping")
            continue

        participant_ids.append(pid)
        # TODO: Extract embeddings for each month (Task 8 will complete this)
        # For now: placeholder zeros
        monthly_embeddings_list.append(torch.zeros(12, EMBEDDING_DIM))

    monthly_embeddings = torch.stack(monthly_embeddings_list)  # [N, 12, 128]

    # Save cache
    temporal_cache = {
        "monthly_embeddings": monthly_embeddings,
        "participant_ids": participant_ids,
        "n_participants": len(participant_ids),
        "embedding_dim": EMBEDDING_DIM,
        "n_months": 12,
    }

    print(f"Saving temporal cache to {output_path}...")
    torch.save(temporal_cache, output_path)
    print(
        f"✅ Saved {len(participant_ids)} participants × 12 months × {EMBEDDING_DIM} dim"
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate temporal embeddings cache")
    parser.add_argument(
        "--monthly-data",
        type=str,
        required=True,
        help="Path to monthly observations JSONL",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/temporal/monthly_embeddings.pt",
        help="Output path for temporal embeddings cache",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on",
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generate_monthly_temporal_embeddings(
        monthly_data_path=Path(args.monthly_data),
        output_path=output_path,
        device=args.device,
    )


if __name__ == "__main__":
    main()
