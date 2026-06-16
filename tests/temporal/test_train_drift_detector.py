"""Tests for drift detector training."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from applications.temporal.train_drift_detector import (
    DriftMonthEstimator,
    train_drift_detector,
)


@pytest.fixture
def sample_features_and_labels(tmp_path):
    """Create sample features and ground truth labels."""
    # Create 100 participants with drift features
    features_list = []
    labels_list = []

    for i in range(100):
        # 20% drift participants
        has_drift = i < 20

        if has_drift:
            # Large dist_max, random drift month in 6-10
            dist_max = np.random.uniform(1.5, 3.0)
            drift_month = np.random.randint(6, 11)
            max_dist_month = drift_month
            dist_mean = dist_max * 0.4
            dist_std = dist_max * 0.2
        else:
            # Small distances, no drift
            dist_max = np.random.uniform(0.1, 0.5)
            max_dist_month = np.random.randint(1, 13)
            dist_mean = dist_max * 0.5
            dist_std = dist_max * 0.1

        features_list.append(
            {
                "participant_id": f"participant_{i}",
                "dist_mean": dist_mean,
                "dist_std": dist_std,
                "dist_max": dist_max,
                "max_dist_month": max_dist_month,
                "dist_slope": np.random.uniform(-0.01, 0.01),
            }
        )

        labels_list.append(
            {
                "participant_id": f"participant_{i}",
                "drift_label": has_drift,
                "drift_month": drift_month if has_drift else None,
            }
        )

    features_df = pd.DataFrame(features_list)

    # Write to temporary files
    features_path = tmp_path / "features.parquet"
    labels_path = tmp_path / "labels.jsonl"

    features_df.to_parquet(features_path, index=False)

    with open(labels_path, "w") as f:
        for label in labels_list:
            f.write(json.dumps(label) + "\n")

    return features_path, labels_path


def test_train_drift_detector_integration(sample_features_and_labels, tmp_path):
    """Integration test for drift detector training."""
    features_path, labels_path = sample_features_and_labels
    classifier_output = tmp_path / "drift_classifier.joblib"
    month_estimator_output = tmp_path / "drift_month_estimator.joblib"

    # Train detector
    train_drift_detector(
        features_path=features_path,
        labels_path=labels_path,
        classifier_output_path=classifier_output,
        month_estimator_output_path=month_estimator_output,
        test_size=0.3,
        random_state=42,
    )

    # Verify model files exist
    assert classifier_output.exists()
    assert month_estimator_output.exists()

    # Load models and verify types
    import joblib

    classifier = joblib.load(classifier_output)
    month_estimator = joblib.load(month_estimator_output)

    assert isinstance(classifier, LogisticRegression)
    assert isinstance(month_estimator, DriftMonthEstimator)

    # Verify classifier has the expected number of features
    assert classifier.n_features_in_ == 4  # dist_mean, dist_std, dist_max, dist_slope


def test_drift_month_estimator_validation():
    """Test that DriftMonthEstimator validates drift_month range."""
    estimator = DriftMonthEstimator()

    # Valid drift month
    assert estimator._validate_drift_month(7) == 7

    # Invalid drift month (outside 6-10)
    assert estimator._validate_drift_month(3) is None  # Too early
    assert estimator._validate_drift_month(11) is None  # Too late
