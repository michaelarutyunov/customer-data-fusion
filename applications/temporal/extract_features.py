"""
Extract drift-detection features from monthly CDT embedding trajectories.

Computes L2 distances between consecutive monthly embeddings, then extracts
summary statistics (mean, std, max, argmax, slope) that serve as features
for drift detection.

Output: applications/temporal/features.parquet
Columns: [participant_id, dist_mean, dist_std, dist_max, max_dist_month, dist_slope]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from scipy import stats
from tqdm import tqdm

log = structlog.get_logger(__name__)


def _compute_l2_distances(participant_data: pd.DataFrame) -> list[float]:
    """Compute L2 distances between consecutive monthly embeddings.

    Parameters
    ----------
    participant_data : pd.DataFrame
        DataFrame for a single participant, sorted by month ascending

    Returns
    -------
    list of float
        L2 distances for months 2-12 relative to previous month
        (length = n_months - 1)
    """
    embeddings = participant_data.sort_values("month")["cdt"].tolist()

    distances: list[float] = []
    for i in range(1, len(embeddings)):
        prev_emb = np.array(embeddings[i - 1])
        curr_emb = np.array(embeddings[i])
        dist = float(np.linalg.norm(curr_emb - prev_emb))
        distances.append(dist)

    return distances


def _extract_trajectory_features(
    participant_data: pd.DataFrame,
) -> dict[str, float | int]:
    """Extract drift features from a single participant's embedding trajectory.

    Parameters
    ----------
    participant_data : pd.DataFrame
        DataFrame for a single participant with columns [month, cdt]

    Returns
    -------
    dict
        Feature dictionary with keys:
        - dist_mean: mean L2 distance across months
        - dist_std: std of L2 distances
        - dist_max: maximum L2 distance
        - max_dist_month: month with largest distance (argmax)
        - dist_slope: linear regression slope of distances over time
    """
    distances = _compute_l2_distances(participant_data)

    if not distances:
        # Edge case: participant has only 1 month (no distances)
        return {
            "dist_mean": 0.0,
            "dist_std": 0.0,
            "dist_max": 0.0,
            "max_dist_month": participant_data.iloc[0]["month"],
            "dist_slope": 0.0,
        }

    # Summary statistics
    dist_mean = float(np.mean(distances))
    dist_std = float(np.std(distances, ddof=1))  # Sample std
    dist_max = float(np.max(distances))

    # Month of maximum distance (argmax)
    # distances[i] is distance for month i+2 (since distances starts at month 2)
    max_idx = int(np.argmax(distances))
    max_dist_month = participant_data.iloc[max_idx + 1][
        "month"
    ]  # +1 because distances is 0-indexed for month 2

    # Linear slope of distances over time
    months = np.arange(2, 2 + len(distances))  # Months 2-12
    slope, _, _, _, _ = stats.linregress(months, distances)
    dist_slope = float(slope)

    return {
        "dist_mean": dist_mean,
        "dist_std": dist_std,
        "dist_max": dist_max,
        "max_dist_month": max_dist_month,
        "dist_slope": dist_slope,
    }


def extract_features(
    input_path: Path | str = "applications/_cache/cdt_embeddings_monthly.parquet",
    output_path: Path | str = "applications/temporal/features.parquet",
) -> None:
    """Extract drift features from monthly CDT embeddings.

    Parameters
    ----------
    input_path : Path or str
        Input parquet file with columns [participant_id, month, cdt]
    output_path : Path or str
        Output parquet file with drift features
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    log.info("feature_extraction.loading_embeddings", input_path=str(input_path))
    embeddings_df = pd.read_parquet(input_path)

    # Validate input
    required_cols = {"participant_id", "month", "cdt"}
    if not required_cols.issubset(set(embeddings_df.columns)):
        missing = required_cols - set(embeddings_df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    # Extract features per participant
    features_list: list[dict] = []
    for participant_id, participant_data in tqdm(
        embeddings_df.groupby("participant_id"), desc="Extracting features"
    ):
        features = _extract_trajectory_features(participant_data)
        features["participant_id"] = participant_id
        features_list.append(features)

    features_df = pd.DataFrame(features_list)

    # Validate output
    expected_cols = {
        "participant_id",
        "dist_mean",
        "dist_std",
        "dist_max",
        "max_dist_month",
        "dist_slope",
    }
    assert expected_cols == set(features_df.columns)

    # Check for NaN/Inf
    for col in ["dist_mean", "dist_std", "dist_max", "dist_slope"]:
        if features_df[col].isna().any():
            log.warning("feature_extraction.nan_detected", column=col)
        if not np.isfinite(features_df[col]).all():
            log.warning("feature_extraction.inf_detected", column=col)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(output_path, index=False)

    log.info(
        "feature_extraction.complete",
        output_path=str(output_path),
        n_participants=len(features_df),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract drift features from embeddings"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="applications/_cache/cdt_embeddings_monthly.parquet",
        help="Input embeddings parquet file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="applications/temporal/features.parquet",
        help="Output features parquet file",
    )
    args = parser.parse_args()

    extract_features(input_path=args.input, output_path=args.output)
