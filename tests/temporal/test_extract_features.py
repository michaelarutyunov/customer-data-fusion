"""Tests for drift feature extraction from embedding trajectories."""

import numpy as np
import pandas as pd
import pytest

from applications.temporal.extract_features import (
    _compute_l2_distances,
    _extract_trajectory_features,
    extract_features,
)


@pytest.fixture
def sample_embeddings():
    """Create sample monthly embeddings for two participants."""
    # Participant A: gradual drift (small changes each month)
    emb_a = []
    for month in range(1, 13):
        base = np.random.randn(128) * 0.1  # Small random variation
        base[0] += month * 0.05  # Gradual drift in first dimension
        emb_a.append({"participant_id": "participant_A", "month": month, "cdt": base})

    # Participant B: sudden drift in month 7 (large jump)
    emb_b = []
    for month in range(1, 13):
        if month < 7:
            base = np.random.randn(128) * 0.1
        else:
            base = np.random.randn(128) * 0.1 + 2.0  # Large shift
        emb_b.append({"participant_id": "participant_B", "month": month, "cdt": base})

    df = pd.DataFrame(emb_a + emb_b)
    return df


def test_compute_l2_distances(sample_embeddings):
    """Test L2 distance computation across consecutive months."""
    participant_data = sample_embeddings[
        sample_embeddings["participant_id"] == "participant_A"
    ].sort_values("month")

    distances = _compute_l2_distances(participant_data)

    # Should have 11 distances (months 2-12 relative to previous month)
    assert len(distances) == 11
    assert all(d >= 0 for d in distances)  # L2 distances are non-negative

    # First distance is month 2 - month 1
    # We planted a gradual drift of 0.05 per month in dimension 0
    # So distance should be approximately 0.05 (plus random noise)
    assert distances[0] > 0


def test_extract_trajectory_features(sample_embeddings):
    """Test feature extraction for a single participant trajectory."""
    participant_data = sample_embeddings[
        sample_embeddings["participant_id"] == "participant_B"
    ].sort_values("month")

    features = _extract_trajectory_features(participant_data)

    # Check required feature keys
    assert "dist_mean" in features
    assert "dist_std" in features
    assert "dist_max" in features
    assert "max_dist_month" in features
    assert "dist_slope" in features

    # Participant B has a large jump at month 7
    # max_dist_month should be 7 (or 8, since we compute distance to previous month)
    assert features["max_dist_month"] in [7, 8]

    # dist_max should be large (we planted a 2.0 shift)
    assert features["dist_max"] > 1.5


def test_extract_features_integration(sample_embeddings, tmp_path):
    """Integration test for full feature extraction pipeline."""
    input_path = tmp_path / "embeddings.parquet"
    output_path = tmp_path / "features.parquet"

    # Write sample embeddings
    sample_embeddings.to_parquet(input_path, index=False)

    # Extract features
    extract_features(input_path=input_path, output_path=output_path)

    # Verify output exists and has correct structure
    assert output_path.exists()

    df = pd.read_parquet(output_path)
    assert "participant_id" in df.columns
    assert "dist_mean" in df.columns
    assert "dist_std" in df.columns
    assert "dist_max" in df.columns
    assert "max_dist_month" in df.columns
    assert "dist_slope" in df.columns

    # Should have 2 participants
    assert len(df) == 2
    assert set(df["participant_id"]) == {"participant_A", "participant_B"}

    # Verify all features are finite (no NaN/Inf)
    for col in ["dist_mean", "dist_std", "dist_max", "dist_slope"]:
        assert df[col].notna().all()
        assert np.isfinite(df[col]).all()
