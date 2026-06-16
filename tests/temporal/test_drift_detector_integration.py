"""End-to-end integration tests for H1 Temporal Dynamics pipeline."""

from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.integration
def test_full_pipeline_end_to_end(tmp_path, fusion_model_exists):
    """Test the complete pipeline: embeddings → features → detector → evaluation.

    This is a slow integration test that requires:
    1. Frozen fusion model at models/fusion_metalearner.pt
    2. Monthly modality files in data/synthetic/

    It will be skipped in CI if dependencies are missing.
    """
    from applications.temporal.generate_monthly_embeddings import generate_monthly_embeddings
    from applications.temporal.extract_features import extract_features
    from applications.temporal.train_drift_detector import train_drift_detector

    # Skip if fusion model doesn't exist
    if not fusion_model_exists:
        pytest.skip("Frozen fusion model not found")

    # Paths
    data_dir = Path("data/synthetic")
    embeddings_path = tmp_path / "cdt_embeddings_monthly.parquet"
    features_path = tmp_path / "features.parquet"
    labels_path = data_dir / "participant_configs.jsonl"
    classifier_path = tmp_path / "drift_classifier.joblib"
    month_estimator_path = tmp_path / "drift_month_estimator.joblib"

    # Step 1: Generate monthly embeddings (3 months for speed)
    generate_monthly_embeddings(
        n_months=3,  # Use fewer months for test speed
        data_dir=data_dir,
        output_path=embeddings_path,
        fusion_model_path=Path("models/fusion_metalearner.pt"),
        force=True,
    )

    assert embeddings_path.exists()
    embeddings_df = pd.read_parquet(embeddings_path)
    assert len(embeddings_df) > 0
    assert set(embeddings_df.columns) >= {"participant_id", "month", "cdt"}

    # Step 2: Extract features
    extract_features(input_path=embeddings_path, output_path=features_path)

    assert features_path.exists()
    features_df = pd.read_parquet(features_path)
    assert len(features_df) > 0
    assert set(features_df.columns) >= {
        "participant_id",
        "dist_mean",
        "dist_std",
        "dist_max",
        "max_dist_month",
        "dist_slope",
    }

    # Step 3: Train drift detector
    # Skip if labels don't exist
    if not labels_path.exists():
        pytest.skip("Ground truth labels not found")

    metrics = train_drift_detector(
        features_path=features_path,
        labels_path=labels_path,
        classifier_output_path=classifier_path,
        month_estimator_output_path=month_estimator_path,
        test_size=0.3,
        random_state=42,
    )

    # Verify metrics are returned
    assert "stage1_recall" in metrics
    assert "stage1_precision" in metrics
    assert "stage1_f1" in metrics
    # stage2_mae may be None if no drift cases in test set

    # Verify model files exist
    assert classifier_path.exists()
    assert month_estimator_path.exists()


@pytest.fixture
def fusion_model_exists():
    """Pytest fixture for checking fusion model availability."""
    return Path("models/fusion_metalearner.pt").exists()
