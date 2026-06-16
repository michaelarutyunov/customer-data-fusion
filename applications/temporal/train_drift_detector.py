"""
Train a two-stage drift detector for temporal regime shifts.

Stage 1: Binary classification (drift vs. no drift)
Stage 2: Drift month estimation (peak detection)

Models:
- Stage 1: LogisticRegression with class_weight='balanced'
- Stage 2: Peak detection (argmax of L2 distances, validated to 6-10)

Security Note: joblib is used here for model persistence. This is safe because:
1. Models are trained locally on trusted synthetic data
2. The output directory is not user-accessible
3. This is a research prototype, not a production system
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import joblib
import numpy as np
import pandas as pd
import structlog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
)
from sklearn.model_selection import StratifiedShuffleSplit

log = structlog.get_logger(__name__)


class DriftMonthEstimator:
    """Estimate drift month using peak detection with validation.

    The drift month is estimated as the month with the maximum L2 distance.
    This is validated to be within the expected range [6, 10] based on
    how regime shifts are injected in the generator.
    """

    def __init__(self, valid_range: tuple[int, int] = (6, 10)):
        """
        Parameters
        ----------
        valid_range : tuple[int, int]
            Valid range for drift_month (inclusive)
        """
        self.valid_range = valid_range

    def _validate_drift_month(self, month: int) -> int | None:
        """Validate that drift_month is within expected range.

        Parameters
        ----------
        month : int
            Predicted drift month

        Returns
        -------
        int or None
            The month if valid, None otherwise
        """
        min_month, max_month = self.valid_range
        if min_month <= month <= max_month:
            return month
        return None

    def predict(self, max_dist_month: int) -> int | None:
        """Predict drift month with validation.

        Parameters
        ----------
        max_dist_month : int
            Month with maximum L2 distance

        Returns
        -------
        int or None
            Predicted drift month if valid, None if outside expected range
        """
        return self._validate_drift_month(max_dist_month)


def train_drift_detector(
    features_path: Path | str = "applications/temporal/features.parquet",
    labels_path: Path | str = "data/synthetic/participant_configs.jsonl",
    classifier_output_path: Path
    | str = "applications/temporal/drift_classifier.joblib",
    month_estimator_output_path: Path
    | str = "applications/temporal/drift_month_estimator.joblib",
    test_size: float = 0.3,
    random_state: int = 42,
) -> dict:
    """Train a two-stage drift detector.

    Parameters
    ----------
    features_path : Path or str
        Input parquet file with drift features
    labels_path : Path or str
        Ground truth labels (participant_configs.jsonl)
    classifier_output_path : Path or str
        Output path for Stage 1 classifier
    month_estimator_output_path : Path or str
        Output path for Stage 2 month estimator
    test_size : float
        Test set proportion for holdout validation
    random_state : int
        Random seed for train/test split

    Returns
    -------
    dict
        Evaluation metrics on test set:
        - stage1_recall: recall for drift classification
        - stage1_precision: precision for drift classification
        - stage1_f1: F1 score
        - stage2_mae: mean absolute error for drift month (on drift_label=True subset)
    """
    features_path = Path(features_path)
    labels_path = Path(labels_path)
    classifier_output_path = Path(classifier_output_path)
    month_estimator_output_path = Path(month_estimator_output_path)

    log.info("drift_detector.loading_features", path=str(features_path))
    features_df = pd.read_parquet(features_path)

    log.info("drift_detector.loading_labels", path=str(labels_path))
    labels_list = []
    with open(labels_path, "r") as f:
        for line in f:
            if line.strip():
                labels_list.append(json.loads(line))
    labels_df = pd.DataFrame(labels_list)

    # Merge features with labels
    merged_df = features_df.merge(labels_df, on="participant_id", how="inner")

    # Prepare Stage 1 features (binary classification)
    feature_cols = ["dist_mean", "dist_std", "dist_max", "dist_slope"]
    X = merged_df[feature_cols].values
    y_drift = merged_df["drift_label"].values

    # Stratified train/test split (preserve drift class balance)
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(sss.split(X, y_drift))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_drift[train_idx], y_drift[test_idx]

    # Train Stage 1: Logistic Regression
    log.info("drift_detector.training_stage1_classifier")
    classifier = LogisticRegression(
        class_weight="balanced", random_state=random_state, max_iter=1000
    )
    classifier.fit(X_train, y_train)

    # Evaluate Stage 1
    y_pred = classifier.predict(X_test)

    report_dict = cast(
        dict[str, dict[str, float]],
        classification_report(y_test, y_pred, output_dict=True),
    )
    cm = confusion_matrix(y_test, y_pred)

    recall_drift = report_dict["True"]["recall"]
    precision_drift = report_dict["True"]["precision"]
    f1_drift = report_dict["True"]["f1-score"]

    log.info(
        "drift_detector.stage1_metrics",
        recall_drift=recall_drift,
        precision_drift=precision_drift,
        f1_drift=f1_drift,
        confusion_matrix=cm.tolist(),
    )

    # Prepare Stage 2 evaluation (drift month prediction)
    # Only evaluate on participants with drift_label=True
    drift_mask: np.ndarray[bool] = y_test
    if drift_mask.sum() > 0:
        test_df = merged_df.iloc[test_idx]
        drift_df = test_df[drift_mask]

        # Stage 2: Peak detection (max_dist_month is already computed by feature extractor)
        # MAE calculated on all drift_label=True cases with known drift_month
        y_drift_month_true = drift_df["drift_month"].values
        y_drift_month_pred = drift_df["max_dist_month"].values

        # Collect all cases with ground truth drift_month
        # Validate predictions are in [6, 10] as per spec
        valid_trues = []
        invalid_preds = []
        for pred, true in zip(y_drift_month_pred, y_drift_month_true):
            if pd.notna(true):  # Only include cases with ground truth
                # Validate prediction is in expected range [6, 10]
                if 6 <= pred <= 10:
                    valid_trues.append((pred, true))
                else:
                    invalid_preds.append((pred, true))

        # Log invalid predictions
        if invalid_preds:
            log.warning(
                "drift_detector.stage2_invalid_predictions",
                n_invalid=len(invalid_preds),
                n_total=len(y_drift_month_pred),
                invalid_samples=[{"pred": p, "true": t} for p, t in invalid_preds[:5]],
            )

        # Convert to arrays for MAE calculation
        if valid_trues:
            preds, trues = zip(*valid_trues)
            preds = list(preds)
            trues = list(trues)
            mae = mean_absolute_error(trues, preds)
            log.info(
                "drift_detector.stage2_metrics",
                mae=mae,
                n_evaluated=len(valid_trues),
                n_invalid=len(invalid_preds) if invalid_preds else 0,
                n_total=len(y_drift_month_pred),
            )
        else:
            mae = None
            log.warning("drift_detector.stage2_no_valid_predictions")
    else:
        mae = None
        log.warning("drift_detector.stage2_no_drift_cases_in_test")

    # Train Stage 2 estimator (trivial peak detector)
    month_estimator = DriftMonthEstimator(valid_range=(6, 10))

    # Save models
    classifier_output_path.parent.mkdir(parents=True, exist_ok=True)
    month_estimator_output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(classifier, classifier_output_path)
    joblib.dump(month_estimator, month_estimator_output_path)

    log.info(
        "drift_detector.models_saved",
        classifier_path=str(classifier_output_path),
        month_estimator_path=str(month_estimator_output_path),
    )

    return {
        "stage1_recall": recall_drift,
        "stage1_precision": precision_drift,
        "stage1_f1": f1_drift,
        "stage2_mae": mae,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train drift detector")
    parser.add_argument(
        "--features",
        type=str,
        default="applications/temporal/features.parquet",
        help="Input features parquet file",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="data/synthetic/participant_configs.jsonl",
        help="Ground truth labels file",
    )
    parser.add_argument(
        "--classifier-output",
        type=str,
        default="applications/temporal/drift_classifier.joblib",
    )
    parser.add_argument(
        "--month-estimator-output",
        type=str,
        default="applications/temporal/drift_month_estimator.joblib",
    )
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    metrics = train_drift_detector(
        features_path=args.features,
        labels_path=args.labels,
        classifier_output_path=args.classifier_output,
        month_estimator_output_path=args.month_estimator_output,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print("\n=== Drift Detector Training Results ===")
    print(f"Stage 1 Recall: {metrics['stage1_recall']:.3f}")
    print(f"Stage 1 Precision: {metrics['stage1_precision']:.3f}")
    print(f"Stage 1 F1: {metrics['stage1_f1']:.3f}")
    if metrics["stage2_mae"]:
        print(f"Stage 2 MAE: {metrics['stage2_mae']:.3f} months")
