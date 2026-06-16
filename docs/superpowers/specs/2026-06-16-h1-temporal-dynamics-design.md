# H1 Temporal Dynamics — Design Specification

> Date: 2026-06-16
> Capability: H1 (Temporal Dynamics / Sequential CDT)
> Version: v0.1 (Drift Detection)
> Status: Design Approved, Pending Implementation

## Overview

H1 treats the Consumer Digital Twin (CDT) embedding as a time series and detects **regime shifts** (sudden changes in decision process) across 12 monthly observation windows per participant. The capability validates that CDT embeddings capture temporal dynamics — not just static identity.

**Core claim:** A person's decision process evolves over time, and the CDT embedding can detect when that evolution represents a regime shift (e.g., loyalty decay, attentional bias change).

## Architecture

```
Monthly Data → Frozen Fusion → 12 Embeddings per Participant → Delta Features → Two-Stage Drift Detector
                                                                               ↓
                                                                    Stage 1: Drift? (binary)
                                                                    Stage 2: When? (month)
```

**Three main components:**
1. **Monthly Re-Encoder** — Generate CDT embeddings for each month
2. **Feature Extractor** — Convert embedding trajectories into drift features
3. **Two-Stage Drift Detector** — Binary classification + drift month estimation

## Components

### 1. Monthly Re-Encoder

**Purpose:** Generate one frozen CDT embedding per participant per month (12 total).

**Implementation:**
- **Script:** `applications/temporal/generate_monthly_embeddings.py`
- **Process:**
  1. Loop months 1-12
  2. For each month, load month-partitioned data files: `{modality}_month_{MM}.jsonl`
  3. Pass through the **frozen** fusion meta-learner (no retraining, no parameter changes)
  4. Accumulate embeddings in `[participant_id, month, cdt]` format
- **Output:** `applications/_cache/cdt_embeddings_monthly.parquet`
- **Columns:** `[participant_id, month, cdt]` where `cdt` is a 128-float vector
- **Cache behavior:** Skip generation if file exists; `--force` flag overwrites
- **Runtime:** ~5-10 minutes (12 × single-session encode time)

**Key invariant:** Uses the **frozen** fusion meta-learner. The encoder and fusion training logic remain untouched. No new architecture is trained for v0.1.

### 2. Feature Extractor

**Purpose:** Convert 12 monthly embeddings into drift-detection features.

**Feature set (per participant):**

1. **Monthly L2 distances:** `dist_t = ||embedding_t - embedding_{t-1}||` for t = 2..12
2. **Summary statistics:**
   - `dist_mean`: Mean of L2 distances across months
   - `dist_std`: Standard deviation of L2 distances
   - `dist_max`: Maximum L2 distance (largest single-month jump)
   - `max_dist_month`: `argmax(dist_t)` — the month with largest jump
3. **Trend features:**
   - `dist_slope`: Linear regression slope of distances over time (increasing/decreasing volatility)

**Implementation:**
- **Script:** `applications/temporal/extract_features.py`
- **Input:** `cdt_embeddings_monthly.parquet`
- **Output:** `applications/temporal/features.parquet`
- **Columns:** `[participant_id, dist_mean, dist_std, dist_max, max_dist_month, dist_slope]`

**Stage 2 shortcut:** `max_dist_month` IS the drift month prediction. This is validated against ground truth `participant_configs.drift_month` (only exists when `drift_label=True`).

### 3. Two-Stage Drift Detector

**Stage 1: Binary Classification (Drift or No Drift?)**

**Model:** Logistic Regression with class-balanced weights
```
features → Linear(n_features, 1) → sigmoid → P(drift)
```

**Input:** `[dist_mean, dist_std, dist_max, dist_slope]` (summary statistics)

**Target:** `participant_configs.drift_label` (binary ground truth: `True` if regime shift injected, `False` otherwise)

**Decision threshold:** Default 0.5; can tune for precision/recall tradeoff

**Why Logistic Regression:** Interpretable, fast, less data-hungry. You can explain "we flag them because their embedding changed X% on average, with a max jump of Y in month Z."

**Stage 2: Drift Month Estimation (When Did It Happen?)**

**Model:** Peak detection with validation
```
drift_month = argmax(L2 distances)  # month with largest embedding jump
```

**Validation rule:**
- If `max_dist_month` is in `[6, 10]` (ground truth range), accept as prediction
- Otherwise, flag as "detected but month uncertain"

**Target:** `participant_configs.drift_month` (integer ground truth, only exists when `drift_label=True`)

**Implementation:**
- **Script:** `applications/temporal/train_drift_detector.py`
- **Outputs:**
  - `applications/temporal/drift_classifier.joblib` (Stage 1: LogisticRegression)
  - `applications/temporal/drift_month_estimator.joblib` (Stage 2: PeakDetector wrapper)

## Data Flow

```
generator pipeline (already run)
    ↓
{modality}_month_01.jsonl .. {modality}_month_12.jsonl (12 files per modality)
    ↓
generate_monthly_embeddings.py
    ↓
cdt_embeddings_monthly.parquet [participant_id, month, cdt]
    ↓
extract_features.py
    ↓
features.parquet [participant_id, dist_mean, dist_std, dist_max, max_dist_month, dist_slope, ...]
    ↓
train_drift_detector.py
    ↓
Stage 1: drift_classifier.joblib (LogisticRegression)
Stage 2: drift_month_estimator.joblib (PeakDetector)
```

## Evaluation Strategy

**Train/Test Split:**
- Participant-level holdout (70/30)
- Stratified by `drift_label` to preserve class balance
- Use `sklearn.model_selection.StratifiedShuffleSplit`

**Stage 1 Metrics (Binary Classification):**
- **Primary:** **Recall@1 ≥ 0.80** (detect 80% of true drift cases)
- **Secondary:** Precision, F1, AUC
- **Confusion matrix:** Show false positives/negatives (who was incorrectly flagged?)

**Stage 2 Metrics (Drift Month Estimation):**
- **Primary:** **Mean Absolute Error (MAE)** on `drift_month` predictions
- **Secondary:** Accuracy (exact month match), ±1 month tolerance
- Only evaluated on `drift_label=True` subset (participants who actually experienced drift)

**Reporting:**
- Per-class metrics (drift vs. no drift)
- Calibration plot (predicted probability vs. actual drift rate)
- Error analysis: Which drift cases are missed? (e.g., small magnitude shifts vs. large loyalty decay)

## Success Criteria

**v0.1 Gate (Drift Detection):**
- ✅ **Stage 1 Recall@1 ≥ 0.80** (detect 80% of true regime shifts)
- ✅ **Stage 2 MAE ≤ 1.5 months** (predict drift month within ±2 months on average)
- ✅ **Precision ≥ 0.60** (avoid excessive false alarms)

If all gates pass, the core claim is validated: **CDT embeddings can detect temporal regime shifts in decision processes.**

## Future Enhancements (v0.2+)

**v0.2: Trajectory Modeling (Option B2)**
- **Stateful streaming model** (GRU/hidden Markov) for incremental drift detection
- Maintain a running hidden state that updates each month
- No monthly retraining — train once, then process embeddings incrementally
- More suitable for production deployment than batch retraining

**v0.3: Counterfactual Analysis**
- **"What if" scenarios:** Predict how CDT would evolve under different interventions
- Example: "How would this customer's embedding change if we sent a retention offer in month 8?"
- Requires causal extension beyond correlation

**v0.4: Real-Time Drift Monitoring**
- **Streaming pipeline** for real-time drift alerts
- Integration with production CDT embedding generation
- Dashboard for monitoring drift across customer segments

## Dependencies

**Existing Prerequisites (Already Built):**
- `PersonaConfig.month` field exists
- `persona_sampler.sample_temporal_trajectory()` generates 12 monthly snapshots with AR(1) drift
- `participant_configs.jsonl` carries `drift_label`/`drift_month` ground truth
- All modalities fielded monthly (trace/transaction/psychographic/clickstream/campaign all carry `month`)
- `evaluation/temporal_split.py` defines months 1-8 train / 9-12 eval split

**New Dependencies (To Be Built):**
- None — H1 v0.1 is self-contained within `applications/temporal/`

## Non-Goals (Explicitly Out of Scope)

- **Retraining encoders or fusion:** The frozen fusion model is invariant. H1 reads embeddings; it does not modify how they are produced.
- **P0 (Schema Foundation):** H1 does not require `Product`/`ChoiceSet`/`MarketState` schemas. It operates purely on the frozen CDT embedding.
- **Causal claims:** H1 detects correlation (embedding changes coincide with regime shifts), not causation (regime shifts cause embedding changes).
- **Real data integration (H2):** v0.1 is synthetic-only. Real data validation is a separate capability.

## Risk Mitigation

**Risk 1: Drift detection fails (Recall@1 < 0.80)**
- **Mitigation:** Report honestly. A negative result is still informative — it means CDT embeddings do NOT capture regime shifts under the current encoding. This guides future research (e.g., maybe we need explicit temporal encoders).

**Risk 2: Embedding deltas reflect noise, not signal**
- **Mitigation:** Compare against a baseline classifier that uses raw persona parameters (e.g., `price_sensitivity`, `brand_loyalty` drift) instead of embeddings. If CDT embeddings beat the baseline, signal is real.

**Risk 3: Class imbalance (drift is rare: ~12% of participants)**
- **Mitigation:** Use `class_weight='balanced'` in logistic regression. Evaluate per-class metrics, not just overall accuracy.

**Risk 4: Temporal leakage in evaluation**
- **Mitigation:** Strict participant-level holdout. Never split at month level — the same participant must not appear in both train and test with overlapping months.

## Implementation Notes

**Operational realism (future refactoring):**
- v0.1 uses **batch encoding** (single-pass over 12 months) for rapid prototyping
- Production deployment would use **incremental per-month encoding** (Option B) — encode each month as it arrives, not batch-process a full year
- The outputs are identical: concatenating 12 per-month files produces the single batch file
- Refactor note for v0.2: Switch to incremental encoding when operational patterns are clarified

**Cache management:**
- `cdt_embeddings_monthly.parquet` is a read-once, write-many cache
- Regeneration occurs only if the upstream synthetic dataset is regenerated
- Add a `--force` flag to overwrite existing cache (useful after schema updates)

**Testing:**
- Unit tests for feature extraction (verify L2 distance computation)
- Integration test: End-to-end pipeline on a small participant subset (N=50)
- Validate against ground truth: Ensure extracted `max_dist_month` matches manual computation on a held-out participant

## References

- **new-capabilities.md** — Capability roadmap, H1 specification
- **persona_sampler.py** — `sample_temporal_trajectory()` function, AR(1) drift dynamics
- **temporal_split.py** — Temporal train/test split (months 1-8 train, 9-12 eval)
- **test-isolation-postmortem.md** — Checkpoint corruption fix (encoder `save_path` param)

---

**Next Step:** Create implementation plan via `writing-plans` skill.
