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
from collections import defaultdict

import torch

import torch
import torch.nn.functional as F

from schemas import CHECKPOINT_PATHS, EMBEDDING_DIM
from fusion.meta_learner import LateFusionMetaLearner


def load_monthly_features_for_participant(
    participant_id: str,
    month: int,
    device: str,
) -> dict[str, torch.Tensor]:
    """Load modality features for a specific participant-month.

    Args:
        participant_id: Participant identifier
        month: Month index (0-11)
        device: Device to load tensors on

    Returns:
        Dictionary with modality tensors (all 6 modalities)
    """
    # TODO: This is a placeholder - needs actual data loading logic
    # For now, return random embeddings to test the pipeline
    # Must return all 6 modalities for fusion model compatibility
    _ = participant_id, month  # Mark as used for future implementation
    return {
        "trace": torch.randn(1, EMBEDDING_DIM, device=device),
        "transaction": torch.randn(1, EMBEDDING_DIM, device=device),
        "text": torch.randn(1, EMBEDDING_DIM, device=device),
        "psychographic": torch.randn(1, EMBEDDING_DIM, device=device),
        "clickstream": torch.randn(1, EMBEDDING_DIM, device=device),
        "campaign": torch.randn(1, EMBEDDING_DIM, device=device),
    }


def generate_monthly_temporal_embeddings(
    monthly_data_path: Path,
    output_path: Path,
    device: str = "cpu",
    phase: str = "2",
    checkpoint_path: Path | None = None,
) -> None:
    """Generate temporal embeddings cache from monthly data.

    Args:
        monthly_data_path: Path to monthly observations JSONL
        output_path: Where to save the temporal embeddings cache
        device: Device to run on
        phase: Fusion phase (default "2")
        checkpoint_path: Path to frozen fusion checkpoint (if None, loads latest)
    """
    print(f"Loading monthly data from {monthly_data_path}...")
    with open(monthly_data_path) as f:
        monthly_records = [json.loads(line) for line in f]

    # Group by participant_id

    monthly_by_participant = defaultdict(list)
    for record in monthly_records:
        monthly_by_participant[record["participant_id"]].append(record)

    # Load frozen fusion model (for extracting monthly embeddings)
    print("Loading frozen fusion model...")
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_PATHS["fusion"]
    model = LateFusionMetaLearner(phase=phase, n_modalities=6).to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()

    # Sort by month and verify we have 12 months for each participant
    participant_ids = []
    monthly_embeddings_list = []

    for pid in sorted(monthly_by_participant.keys()):
        records = sorted(monthly_by_participant[pid], key=lambda r: r["month"])
        if len(records) != 12:
            print(f"⚠️  Participant {pid} has {len(records)} months, skipping")
            continue

        participant_ids.append(pid)

        # Extract embeddings for each month
        participant_monthly_embeddings = []
        for month_record in records:
            # Load modality features for this month
            monthly_features = load_monthly_features_for_participant(
                participant_id=pid,
                month=month_record["month"],
                device=device,
            )

            # Stack modality embeddings [1, n_modalities, 128]
            modality_embeddings = torch.stack(
                [monthly_features[mod] for mod in sorted(monthly_features.keys())],
                dim=1,
            )

            # Normalize each modality
            norm_embs = [
                F.normalize(modality_embeddings[:, i], p=2, dim=-1)
                for i in range(modality_embeddings.shape[1])
            ]
            fusion_input = torch.cat(norm_embs, dim=-1)

            # Get CDT embedding from frozen fusion model
            with torch.no_grad():
                _, cdt_embedding = model.forward_with_embedding(fusion_input)
                # Squeeze batch dim: [1, 128] -> [128]
                participant_monthly_embeddings.append(cdt_embedding.squeeze(0).cpu())

        # Stack: [12, 128]
        monthly_embeddings_list.append(torch.stack(participant_monthly_embeddings))

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
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to frozen fusion checkpoint (default: latest from CHECKPOINT_PATHS)",
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generate_monthly_temporal_embeddings(
        monthly_data_path=Path(args.monthly_data),
        output_path=output_path,
        device=args.device,
        phase="2",
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
    )


if __name__ == "__main__":
    main()
