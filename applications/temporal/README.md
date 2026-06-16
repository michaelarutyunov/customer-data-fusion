# H1 Temporal Dynamics — Usage Guide

## Overview

The H1 Temporal Dynamics capability detects regime shifts in consumer decision processes across 12 monthly observation windows using frozen CDT embeddings.

## Quick Start

### 1. Generate Monthly Embeddings

```bash
uv run python applications/temporal/generate_monthly_embeddings.py
```

This reads month-partitioned modality files (`{modality}_month_MM.jsonl`) and writes frozen CDT embeddings to `applications/_cache/cdt_embeddings_monthly.parquet`.

**Options:**
- `--n-months`: Number of months to encode (default: 12)
- `--data-dir`: Data directory (default: `data/synthetic`)
- `--output`: Output parquet path (default: `applications/_cache/cdt_embeddings_monthly.parquet`)
- `--fusion-model`: Path to frozen fusion model (default: `models/fusion_metalearner.pt`)
- `--force`: Overwrite existing cache (default: skip if exists)

**Output:** Parquet file with columns `[participant_id, month, cdt]`

### 2. Extract Drift Features

```bash
uv run python applications/temporal/extract_features.py
```

This reads monthly embeddings and computes L2 distance statistics across consecutive months.

**Options:**
- `--input`: Input embeddings parquet (default: `applications/_cache/cdt_embeddings_monthly.parquet`)
- `--output`: Output features parquet (default: `applications/temporal/features.parquet`)

**Output:** Parquet file with columns `[participant_id, dist_mean, dist_std, dist_max, max_dist_month, dist_slope]`

### 3. Train Drift Detector

```bash
uv run python applications/temporal/train_drift_detector.py
```

This trains a two-stage drift detector and prints evaluation metrics.

**Options:**
- `--features`: Input features parquet (default: `applications/temporal/features.parquet`)
- `--labels`: Ground truth labels (default: `data/synthetic/participant_configs.jsonl`)
- `--classifier-output`: Stage 1 model output (default: `applications/temporal/drift_classifier.joblib`)
- `--month-estimator-output`: Stage 2 model output (default: `applications/temporal/drift_month_estimator.joblib`)
- `--test-size`: Test set proportion (default: 0.3)
- `--random-state`: Random seed (default: 42)

**Output:** Two model files (`.joblib`) + printed metrics report

### 4. Evaluate Drift Detector

```bash
uv run python applications/temporal/evaluate_drift_detector.py
```

This loads a trained detector and prints detailed evaluation metrics.

**Options:**
- `--features`: Test set features (default: `applications/temporal/features.parquet`)
- `--labels`: Ground truth labels (default: `data/synthetic/participant_configs.jsonl`)
- `--classifier`: Trained Stage 1 classifier (default: `applications/temporal/drift_classifier.joblib`)
- `--month-estimator`: Trained Stage 2 estimator (default: `applications/temporal/drift_month_estimator.joblib`)
- `--output-dir`: Directory to save calibration plot (optional)

**Output:** Printed metrics report + optional calibration plot

## Pipeline Script

To run the full pipeline in one command:

```bash
# Generate embeddings
uv run python applications/temporal/generate_monthly_embeddings.py --force

# Extract features
uv run python applications/temporal/extract_features.py

# Train detector
uv run python applications/temporal/train_drift_detector.py

# Evaluate
uv run python applications/temporal/evaluate_drift_detector.py
```

## Success Criteria

The capability passes when:
- **Stage 1 Recall@1 ≥ 0.80**: Detect 80% of true regime shifts
- **Stage 2 MAE ≤ 1.5 months**: Predict drift month within ±2 months on average
- **Precision ≥ 0.60**: Avoid excessive false alarms

## Expected Runtime

- **Monthly embeddings:** 5-10 minutes (12 months × single-session encode time)
- **Feature extraction:** 1-2 minutes
- **Detector training:** <1 minute
- **Total:** ~10-15 minutes

## Troubleshooting

### "Fusion model not found"

The frozen fusion model must exist at `models/fusion_metalearner.pt`. If missing:
1. Train the fusion meta-learner: `uv run python -m fusion.train`
2. Verify the checkpoint path

### "Missing month files"

The pipeline expects month-partitioned files: `{modality}_month_01.jsonl` to `{modality}_month_12.jsonl`. If missing:
1. Verify data was generated with temporal fielding
2. Check `data/synthetic/` for the expected files

### "NaN detected in features"

This warning appears when some participants have invalid embeddings (e.g., from missing modality data). These participants are included but may have `dist_* = 0.0`.

### "Stage 2: No valid predictions"

This occurs when the test set has no drift cases or all predicted drift months are outside [6, 10]. It's expected on small test sets.

## Architecture

```
Monthly Data → Frozen Fusion → 12 Embeddings per Participant → Delta Features → Two-Stage Drift Detector
                                                                               ↓
                                                                    Stage 1: Drift? (binary)
                                                                    Stage 2: When? (month)
```

## Next Steps

After v0.1 (drift detection) is validated, consider:
- **v0.2:** Stateful streaming model (Option B2) for incremental drift detection
- **v0.3:** Counterfactual analysis ("what if" scenarios)
- **v0.4:** Real-time drift monitoring with streaming pipeline

See `docs/superpowers/specs/2026-06-16-h1-temporal-dynamics-design.md` for details.
