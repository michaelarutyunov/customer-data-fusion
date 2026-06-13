"""
Engineered-metrics baseline for trace encoder ablation.

Per docs/modalities/mouselab.md: 'the engineered-metrics version may well match
the sequence encoder; show both.' This script trains a logistic regression on
aggregated trace features and reports strategy recovery + individual identity,
for comparison against the Transformer trace encoder.

Usage:
    uv run python -m evaluation.trace_baseline
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import structlog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from schemas import PERSONA_TO_IDX

log = structlog.get_logger(__name__)

DATA_DIR = Path("data/synthetic")
TRIALS_PATH = DATA_DIR / "trials.jsonl"
TRACES_PATH = DATA_DIR / "traces.jsonl"
RESULTS_DIR = Path("evaluation/results")
RESULTS_PATH = RESULTS_DIR / "trace_baseline_comparison.json"


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _ensure_data() -> None:
    """Generate trace data if not present."""
    if TRIALS_PATH.exists() and TRACES_PATH.exists():
        return
    log.info("trace_baseline.generating_data")
    from generator.pipeline import run_pipeline

    run_pipeline(n=100, skip_narratives=True)


def extract_customer_features(
    trials: list[dict], traces: list[dict]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract per-customer aggregated trace features.

    Returns (X, y, customer_ids) where X is (n_customers, 14) — mean and std of
    7 metrics across trials.
    """
    # Index traces by trial_id
    traces_by_trial: dict[str, list[dict]] = {}
    for ev in traces:
        traces_by_trial.setdefault(ev["trial_id"], []).append(ev)

    # Group trials by participant
    trials_by_participant: dict[str, list[dict]] = {}
    for trial in trials:
        trials_by_participant.setdefault(trial["participant_id"], []).append(trial)

    features: list[list[float]] = []
    labels: list[int] = []
    customer_ids: list[str] = []

    for pid, pid_trials in trials_by_participant.items():
        per_trial_metrics: list[list[float]] = []
        for trial in pid_trials:
            trial_events = traces_by_trial.get(trial["trial_id"], [])
            if not trial_events:
                continue

            dwell_times = [e["dwell_ms"] for e in trial_events]
            total_dwell = sum(dwell_times) or 1.0
            price_dwell = sum(
                e["dwell_ms"] for e in trial_events if e["attribute_id"] == "price"
            )
            brand_dwell = sum(
                e["dwell_ms"] for e in trial_events if e["attribute_id"] == "brand"
            )
            n_reinspect = sum(1 for e in trial_events if e.get("is_reinspection"))

            per_trial_metrics.append(
                [
                    trial["payne_index"],
                    trial["prop_cells_inspected"],
                    float(np.mean(dwell_times)),
                    price_dwell / total_dwell,
                    brand_dwell / total_dwell,
                    n_reinspect / len(trial_events),
                    len(trial_events),
                ]
            )

        if not per_trial_metrics:
            continue

        arr = np.array(per_trial_metrics)  # (n_trials, 7)
        # Aggregate: mean and std across trials → 14-dim
        feat = np.concatenate([arr.mean(axis=0), arr.std(axis=0)])
        features.append(feat.tolist())
        labels.append(PERSONA_TO_IDX.get(pid_trials[0]["persona_id"], 0))
        customer_ids.append(pid)

    return np.array(features), np.array(labels), customer_ids


def compute_identity_recall(
    X: np.ndarray, customer_ids: list[str], train_idx: np.ndarray, test_idx: np.ndarray
) -> float:
    """Compute recall@1 in feature space (cosine similarity).

    For each test customer, find the nearest train customer by cosine
    similarity. recall@1 = fraction where the nearest neighbour is the
    same archetype.
    """
    X_train = X[train_idx]
    X_test = X[test_idx]
    ids_train = [customer_ids[i] for i in train_idx]

    # Normalise for cosine similarity
    def _norm(v: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return v / norms

    X_train_n = _norm(X_train)
    X_test_n = _norm(X_test)
    sims = X_test_n @ X_train_n.T  # (n_test, n_train)

    # Extract archetype from customer_id prefix (e.g., "price_lex_0001" -> "price_lex")
    def _archetype(cid: str) -> str:
        return "_".join(cid.split("_")[:-1])

    correct = 0
    for i, nn_idx in enumerate(sims.argmax(axis=1)):
        test_arch = _archetype(customer_ids[test_idx[i]])
        train_arch = _archetype(ids_train[nn_idx])
        if test_arch == train_arch:
            correct += 1

    return correct / len(test_idx) if len(test_idx) > 0 else 0.0


def run_baseline() -> dict:
    """Run the engineered-metrics baseline and return results."""
    _ensure_data()

    trials = _load_jsonl(TRIALS_PATH)
    traces = _load_jsonl(TRACES_PATH)
    log.info("trace_baseline.loaded", n_trials=len(trials), n_traces=len(traces))

    X, y, customer_ids = extract_customer_features(trials, traces)
    log.info(
        "trace_baseline.features", n_customers=len(customer_ids), n_features=X.shape[1]
    )

    if len(customer_ids) < 10:
        log.warning("trace_baseline.insufficient_data", n=len(customer_ids))
        return {"error": "insufficient data", "n_customers": len(customer_ids)}

    # Train/test split (stratified by archetype)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
    )

    # Train logistic regression (multinomial is default for multiclass in sklearn>=1.5)
    clf = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = float((y_pred == y_test).mean())
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))

    # Individual identity recall@1
    recall_at_1 = compute_identity_recall(X, customer_ids, idx_train, idx_test)

    results = {
        "engineered_features": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "recall_at_1": recall_at_1,
            "n_customers": len(customer_ids),
            "n_features": X.shape[1],
        },
        "transformer_placeholder": {
            "note": "Fill with Transformer trace encoder results after Phase 4b retrain",
            "accuracy": None,
            "macro_f1": None,
            "recall_at_1": None,
        },
        "feature_names": [
            "payne_index_mean",
            "prop_cells_mean",
            "mean_dwell_mean",
            "dwell_share_price_mean",
            "dwell_share_brand_mean",
            "reinspection_rate_mean",
            "n_events_mean",
            "payne_index_std",
            "prop_cells_std",
            "mean_dwell_std",
            "dwell_share_price_std",
            "dwell_share_brand_std",
            "reinspection_rate_std",
            "n_events_std",
        ],
    }

    # Print comparison table
    print("\n" + "=" * 60)
    print("TRACE ENCODER ABLATION: Engineered Features vs Transformer")
    print("=" * 60)
    print(f"{'Metric':<25} {'Engineered':>15} {'Transformer':>15}")
    print("-" * 60)
    print(f"{'Accuracy':<25} {accuracy:>15.3f} {'(pending)':>15}")
    print(f"{'Macro F1':<25} {macro_f1:>15.3f} {'(pending)':>15}")
    print(f"{'Individual recall@1':<25} {recall_at_1:>15.3f} {'(pending)':>15}")
    print("=" * 60)
    print(f"\nCustomers: {len(customer_ids)}, Features: {X.shape[1]}")
    print("\nPer-class report (engineered features):")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    log.info("trace_baseline.saved", path=str(RESULTS_PATH))

    return results


if __name__ == "__main__":
    run_baseline()
