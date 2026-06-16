# H1 Temporal Dynamics — Capability Specification

## Purpose

Detect regime shifts in consumer decision processes across 12 monthly observation windows using frozen CDT embeddings.

## Components

### 1. Monthly Re-Encoder

**Script:** `generate_monthly_embeddings.py`

**Input:** Month-partitioned modality files (`{modality}_month_MM.jsonl`)

**Output:** `applications/_cache/cdt_embeddings_monthly.parquet` with columns:
- `participant_id`: str
- `month`: int (1-12)
- `cdt`: array<float128> (frozen 128-dim fusion embedding)

**Invariant:** Uses the frozen fusion meta-learner. No retraining. No parameter changes.

### 2. Feature Extractor

**Script:** `extract_features.py`

**Input:** `cdt_embeddings_monthly.parquet`

**Output:** `applications/temporal/features.parquet` with columns:
- `participant_id`: str
- `dist_mean`: float (mean L2 distance across months)
- `dist_std`: float (std of L2 distances)
- `dist_max`: float (maximum L2 distance)
- `max_dist_month`: int (month of largest jump)
- `dist_slope`: float (linear regression slope of distances)

### 3. Drift Detector

**Script:** `train_drift_detector.py`

**Stage 1 (Binary Classification):**
- Model: LogisticRegression with class_weight='balanced'
- Features: [dist_mean, dist_std, dist_max, dist_slope]
- Target: participant_configs.drift_label (binary)
- Output: drift_classifier.joblib

**Stage 2 (Month Estimation):**
- Model: Peak detection (argmax of L2 distances)
- Validation: max_dist_month must be in [6, 10]
- Target: participant_configs.drift_month (when drift_label=True)
- Output: drift_month_estimator.joblib

## Success Criteria

- Stage 1 Recall@1 ≥ 0.80 (detect 80% of true drift cases)
- Stage 2 MAE ≤ 1.5 months (predict drift month within ±2 months)
- Precision ≥ 0.60 (avoid excessive false alarms)

## Data Contracts

**Frozen Fusion Model:** Loaded from `models/fusion_metalearner.pt` (hardcoded path, validated on startup)

**Ground Truth Labels:** Loaded from `data/synthetic/participant_configs.jsonl` with fields:
- `participant_id`: str
- `drift_label`: bool (True if regime shift injected)
- `drift_month`: int | null (month of regime shift, 6-10, only when drift_label=True)

## Cache Strategy

- `cdt_embeddings_monthly.parquet`: Skip generation if exists; use `--force` to overwrite
- `features.parquet`: Regenerated if embeddings change
- Model files: Regenerated on each training run
