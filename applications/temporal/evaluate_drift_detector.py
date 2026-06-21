"""
Evaluate drift detector and print detailed metrics report.

Loads a trained drift detector, runs it on a test set, and prints:
- Stage 1: Classification metrics (recall, precision, F1, confusion matrix)
- Stage 2: Drift month MAE (on drift_label=True subset)
- Calibration plot (predicted probability vs. actual drift rate)
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import structlog
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
)

log = structlog.get_logger(__name__)


def evaluate_drift_detector(
    features_path: Path | str,
    labels_path: Path | str,
    classifier_path: Path | str,
    month_estimator_path: Path | str,
    output_dir: Path | str | None = None,
) -> dict:
    """Evaluate a trained drift detector.

    Parameters
    ----------
    features_path : Path or str
        Test set features parquet file
    labels_path : Path or str
        Ground truth labels
    classifier_path : Path or str
        Trained Stage 1 classifier
    month_estimator_path : Path or str
        Trained Stage 2 month estimator
    output_dir : Path or str, optional
        Directory to save evaluation plots (if None, don't save)

    Returns
    -------
    dict
        Evaluation metrics (same structure as train_drift_detector return value)
    """
    features_path = Path(features_path)
    labels_path = Path(labels_path)
    classifier_path = Path(classifier_path)
    month_estimator_path = Path(month_estimator_path)

    # Load data
    log.info(
        "evaluation.loading_data", features=str(features_path), labels=str(labels_path)
    )
    features_df = pd.read_parquet(features_path)

    labels_list = []
    with open(labels_path, "r") as f:
        for line in f:
            if line.strip():
                labels_list.append(json.loads(line))
    labels_df = pd.DataFrame(labels_list)

    # Merge features with labels
    merged_df = features_df.merge(labels_df, on="participant_id", how="inner")

    # Load models
    classifier = joblib.load(classifier_path)

    # Stage 1 evaluation
    feature_cols = ["dist_mean", "dist_std", "dist_max", "dist_slope"]
    X = merged_df[feature_cols].values
    y_drift = merged_df["drift_label"].values

    y_pred = classifier.predict(X)
    y_pred_proba = classifier.predict_proba(X)[:, 1]

    # Print classification report
    print("\n=== Stage 1: Drift Classification ===")
    print(classification_report(y_drift, y_pred, target_names=["No Drift", "Drift"]))

    # Confusion matrix
    cm = confusion_matrix(y_drift, y_pred)
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("                No Drift  Drift")
    print(f"Actual No Drift   {cm[0, 0]:6d}  {cm[0, 1]:5d}")
    print(f"Actual Drift      {cm[1, 0]:6d}  {cm[1, 1]:5d}")

    # Extract metrics
    report = classification_report(y_drift, y_pred, output_dict=True)
    recall_drift = report["1"]["recall"]
    precision_drift = report["1"]["precision"]
    f1_drift = report["1"]["f1-score"]

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_drift, y_pred_proba, n_bins=10)

    print("\nCalibration (Predicted Prob vs. Actual Rate):")
    print("Bin Pred_Prob  Actual_Rate  n_samples")
    for i, (pred, true) in enumerate(zip(prob_pred, prob_true)):
        n_samples = ((y_pred_proba >= (i / 10)) & (y_pred_proba < ((i + 1) / 10))).sum()
        print(f"{i:2d}    {pred:.2f}        {true:.2f}      {n_samples:6d}")

    # Stage 2 evaluation (drift month)
    drift_mask = y_drift
    if drift_mask.sum() > 0:
        drift_df = merged_df[drift_mask]

        y_drift_month_true = drift_df["drift_month"].to_numpy()
        y_drift_month_pred = drift_df["max_dist_month"].to_numpy()

        # Filter to valid predictions (6-10)
        valid_preds = []
        valid_trues = []
        for pred, true in zip(y_drift_month_pred, y_drift_month_true):
            if 6 <= pred <= 10 and pd.notna(true):
                valid_preds.append(pred)
                valid_trues.append(true)

        if len(valid_preds) > 0:
            mae = mean_absolute_error(valid_trues, valid_preds)
            print("\n=== Stage 2: Drift Month Estimation ===")
            print(f"MAE: {mae:.3f} months (on {len(valid_preds)} valid predictions)")

            # Accuracy and ±1 tolerance
            exact_match = np.mean([p == t for p, t in zip(valid_preds, valid_trues)])
            within_one = np.mean(
                [abs(p - t) <= 1 for p, t in zip(valid_preds, valid_trues)]
            )
            print(f"Exact match: {exact_match:.3f}")
            print(f"Within ±1 month: {within_one:.3f}")
        else:
            mae = None
            print("\n=== Stage 2: No valid predictions ===")
    else:
        mae = None
        print("\n=== Stage 2: No drift cases in test set ===")

    # Save calibration plot if output_dir specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(
            prob_pred, prob_true, marker="o", linewidth=2, label="Calibration curve"
        )
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("True drift rate")
        ax.set_title("Calibration Plot (Stage 1)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_path = output_dir / "calibration_plot.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"\nCalibration plot saved to: {plot_path}")

    return {
        "stage1_recall": recall_drift,
        "stage1_precision": precision_drift,
        "stage1_f1": f1_drift,
        "stage2_mae": mae,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate drift detector")
    parser.add_argument(
        "--features",
        type=str,
        default="applications/temporal/features.parquet",
        help="Test set features parquet file",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="data/synthetic/participant_configs.jsonl",
        help="Ground truth labels file",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="applications/temporal/drift_classifier.joblib",
        help="Trained Stage 1 classifier",
    )
    parser.add_argument(
        "--month-estimator",
        type=str,
        default="applications/temporal/drift_month_estimator.joblib",
        help="Trained Stage 2 month estimator",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation plots",
    )
    args = parser.parse_args()

    evaluate_drift_detector(
        features_path=args.features,
        labels_path=args.labels,
        classifier_path=args.classifier,
        month_estimator_path=args.month_estimator,
        output_dir=args.output_dir,
    )
