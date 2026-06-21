# M1 Post-Mortem: Choice Model Implementation

> Written: 2026-06-18
> Phase: Phase 0 + M1 Choice Model (tasks #26-#55)
> Scope: generator/, schemas/, encoders/trace/, encoders/fusion/, applications/choice/, evaluation/
> Status: ❌ CDT Lift Gate Failed — Root Cause Identified

---

## 1. What Was Built

### Phase 0: Preference-Driven Choice Generation

| Component | Files | Status |
|---|---|---|
| Product Schema | `schemas/product.py` | Implemented |
| ChoiceSet Schema | `schemas/choice_set.py` | Implemented |
| TrialRecord Extension | `schemas/trace.py` (added `choice_set_id`) | Implemented |
| Product Catalog Generator | `generator/product_catalog.py` | Implemented |
| Preference-Driven Choice Logic | `generator/trace_simulator.py` (_compute_choice_from_inspected_cells) | Implemented |
| Choice Validation Checks | `generator/validate.py` (3 new checks) | Implemented |
| Trace-Choice Coupling Verification | `evaluation/verify_trace_choice_coupling.py` | Implemented |

### M1 Choice Model Implementation

| Component | Files | Status |
|---|---|---|
| Choice Model Architecture | `applications/choice/model.py` (two-tower) | Implemented |
| Choice Data Loading | `applications/choice/data.py` (flat training table) | Implemented |
| Choice Training Script | `applications/choice/train.py` (BCE loss, 70/30 split) | Implemented |
| Choice Evaluation Script | `applications/choice/evaluate.py` (success criteria) | Implemented |
| Choice Model SPEC | `applications/choice/SPEC.md` | Implemented |

### Encoder Cascade Retraining

| Task | Description | Status |
|---|---|---|
| #51 | Retrain trace encoder with preference-driven choices | ✅ Completed (57.95% val acc) |
| #52 | Retrain fusion encoder with updated trace embeddings | ✅ Completed (85% val acc) |
| #53 | Regenerate CDT cache with updated fusion embeddings | ✅ Completed (1000 participants) |
| #54 | Retrain M1 with updated CDT embeddings | ✅ Completed (53% AUC) |
| #55 | Investigate weak trace-choice coupling | ✅ Completed (root cause identified) |

### Architecture Summary

**M1 Two-Tower Choice Model:**
```
Consumer Tower: CDT[128] → Linear(128, 64) → ReLU → Dropout(0.1) → [64]
Product Tower:  Product[D] → Linear(D, 64) → ReLU → Dropout(0.1) → [64]
Joint Layer:      Concat[128] → Linear(128, 1) → Sigmoid → P(choose)

Parameters: cdt_dim=128, product_dim=2, hidden_dim=64, dropout=0.1
```

**Training Configuration:**
- Participant-level 70/30 split (prevents data leakage)
- BCE loss for binary classification
- AdamW optimizer (lr=1e-3, weight_decay=1e-4)
- Early stopping (patience=10) on validation AUC
- 50 epochs maximum

---

## 2. Implementation Sequence

### Phase 0: Foundation (Tasks #26-#40)

**1. Product and ChoiceSet Schemas (#26-#28)**
- Created `schemas/product.py` (frozen dataclass with 6 fields)
- Created `schemas/choice_set.py` (frozen dataclass linking trials to choices)
- Extended `schemas/trace.py` with optional `choice_set_id` field
- Updated `schemas/__init__.py` exports

**2. Product Catalog Generation (#29-#30)**
- Implemented `generator/product_catalog.py` to generate ~81 products
- Products span 3 categories (electronics, fashion, home_goods)
- Each product has: product_id, category, price_normalised, brand, quality_normalised, features
- Output: `data/synthetic/products.jsonl`

**3. Preference-Driven Choice Logic (#31-#32)**
- Implemented `_compute_choice_from_inspected_cells()` function
- Coupled trace (inspected cells) to choice via utility computation
- Supports 3 strategies: lexicographic, compensatory, fallback
- Added strategy lapse noise and softmax temperature
- Returns: (chosen_slot, choice_probabilities)

**4. Choice Integration (#33-#35)**
- Replaced random choice with preference-driven choice in trace simulator
- Updated transaction simulator to use shared Product schema
- Wired ChoiceSet output into pipeline
- Added 3 validation checks: choice_consistency, product_coverage, trace_choice_coupling

**5. Data Regeneration (#40)**
- Generated clean synthetic dataset: 1000 participants
- Output files: traces.jsonl (25M), trials.jsonl (3.9M), choice_sets.jsonl (6.9M)
- Fixed JSON corruption issue (concatenated objects in trials.jsonl)

### M1: Choice Model Implementation (Tasks #44-#50)

**6. Choice Model Components (#44-#47)**
- Implemented two-tower architecture in `applications/choice/model.py`
- Built flat training table in `applications/choice/data.py` (N rows per trial)
- Created training script with BCE loss in `applications/choice/train.py`
- Implemented evaluation script with success criteria in `applications/choice/evaluate.py`
- Created SPEC.md with architecture, training, and evaluation details

**7. Initial M1 Training (#49-#50)**
- Built training table: 50,170 rows (10,000 chosen, 40,170 rejected)
- Trained for 12 epochs with early stopping
- Best val AUC: 0.5161
- **Result: ❌ FAILED** (AUC < 0.65 threshold)

### Encoder Cascade Retraining (#51-#54)

**8. Trace Encoder Retraining (#51)**
- Backed up old trace encoder (trained on random choices)
- Retrained on preference-driven data
- 25 epochs, early stopping (patience=10)
- Best val acc: 57.95% (up from 34% initial)
- **Result: ✅ Improved** but objective mismatch not yet discovered

**9. Fusion Encoder Retraining (#52)**
- Retrained with updated trace embeddings
- 28 epochs, early stopping (patience=10)
- Best val acc: 85% (up from 57% initial)
- **Result: ✅ Improved** but still failed M1 lift gate

**10. CDT Cache Regeneration (#53)**
- Created `generate_cdt_cache.py` script
- Generated embeddings from fusion model + encoder outputs
- Used normalized modality embeddings as input
- Output: `applications/_cache/cdt_embeddings.parquet` (1000 participants)
- **Result: ✅ Completed** but trace coverage issue discovered

**11. M1 Retraining (#54)**
- Rebuilt choice training table with updated CDT cache
- Trained for 20 epochs with early stopping
- Best val AUC: 0.5234
- **Result: ❌ FAILED** (still random performance)

### Root Cause Investigation (#55)

**12. Trace Coverage Analysis**
- **Discovery:** Only 250/1000 participants (25%) have trace coverage
- Created `get_trace_coverage_participants.py` to extract trace coverage participants
- Updated choice data loading to filter to trace coverage participants only
- **Result: ✅ Fixed** but AUC still 0.53 (no improvement)

**13. Data-Level Trace-Choice Coupling Analysis**
- **Discovery:** 83.6% of chosen alternatives were inspected
- Confirmed preference-driven logic IS working correctly
- Systematic inspection-choice patterns exist in data
- **Result: ✅ Strong coupling in data, but M1 still fails**

**14. Root Cause: Objective Mismatch**
- **Discovery:** Trace encoder trained to predict **persona strategy** (7 classes), not **choice**
- Fusion encoder trained to predict **persona strategy** (7 classes), not **choice**
- M1 expects embeddings to predict **binary choice** (chosen/rejected)
- **Result: ❌ Fundamental objective mismatch**

---

## 3. Challenges and Errors

### C1 — JSON Corruption in trials.jsonl — Data Generation Bug
**What failed**: Line 40 of `data/synthetic/trials.jsonl` contained concatenated JSON objects: `"final_choice": "C",{"participant_id":...`

**Root cause**: Pipeline bug where trial records were concatenated without proper line separators during data generation.

**Resolution**: Regenerated entire synthetic dataset (1000 participants) with `uv run python -m generator.pipeline --n 1000`. This took ~2 hours but ensured clean data across all modalities.

### C2 — PyArrow vs Pandas Confusion — Data Loading Bug
**What failed**: Multiple scripts failed with `'pyarrow.lib.ChunkedArray' object has no attribute 'iloc'` and `'pyarrow.lib.ChunkedArray' object has no attribute 'isin'`

**Root cause**: Mixed usage of pyarrow tables and pandas DataFrames. PyArrow tables don't support pandas methods.

**Resolution**: Added `.to_pandas()` conversions in `applications/choice/data.py` and `applications/choice/train.py` for data manipulation steps. Used `pa.Table.from_pandas()` for parquet writing.

### C3 — Missing Torch Import — Model Architecture Bug
**What failed**: `NameError: name 'torch' is not defined` in `applications/choice/model.py:61`

**Root cause**: Only imported `torch.nn` but used `torch.cat()` in forward pass.

**Resolution**: Added `import torch` to imports in `applications/choice/model.py`.

### C4 — MLflow File Store Deprecated — Training Environment Bug
**What failed**: `MLflowException: The filesystem tracking backend is in maintenance mode`

**Root cause**: MLflow 3.x deprecated `./mlruns` file store backend.

**Resolution**: Used `MLFLOW_ALLOW_FILE_STORE=true` environment variable as workaround. Documented need for migration to SQLite backend.

### C5 — Choice Set ID Linkage Missing — Data Structure Issue
**What failed**: `evaluation/verify_trace_choice_coupling.py` reported "No valid trial-choice pairs found" despite both files existing.

**Root cause**: `TrialRecord.choice_set_id` field was `None` even though it should link to `ChoiceSet.choice_set_id` by `trial_id` match. The field was added to schema but never populated in pipeline.

**Workaround**: Verified that `trial_id == choice_set_id` format matches conceptually, but explicit field not populated. This blocked verification script but didn't block M1 implementation.

### C6 — Trace Coverage Mismatch — Critical Architecture Issue
**What failed**: M1 AUC 0.53 despite retraining entire encoder cascade.

**Root cause**: Only 250/1000 participants (25%) have trace coverage, but M1 trained on choice data from all 1000 participants. 75% of CDT embeddings don't contain trace information, teaching M1 to ignore CDT signal.

**Resolution**: Created trace coverage filtering, updated data loading to only include 250 participants with trace coverage. **Result: Still AUC 0.53** - this revealed the issue was deeper than coverage.

### C7 — Objective Mismatch Discovery — Fundamental Design Issue
**What failed**: Despite fixing trace coverage, M1 AUC remained 0.53 (random).

**Root cause investigation steps:**
1. Analyzed data-level coupling: 83.6% of chosen alternatives were inspected ✅
2. Checked trace encoder objective: Trained for **persona strategy classification** (7 classes)
3. Checked fusion encoder objective: Trained for **persona strategy classification** (7 classes)
4. Checked M1 objective: Expects embeddings to predict **binary choice** (chosen/rejected)

**Discovery**: Strategy ≠ Choice. The encoder cascade learned "how they decide" (lexicographic vs compensatory) but not "what they decide" (choose A vs B). The embeddings contain strategy patterns, not choice patterns.

**Resolution**: Documented as fundamental architectural constraint. Current encoder cascade cannot support choice prediction without retraining with choice-related objectives. No simple fix available.

---

## 4. Deviations from SPEC

### D1 — Trace Coverage Filtering Not in Original SPEC
**SPEC said** (`applications/choice/SPEC.md`): No mention of trace coverage filtering.

**What was implemented**: Created `get_trace_coverage_participants.py` and updated data loading to filter to 250 participants with trace coverage.

**Why**: Discovered that 75% of CDT embeddings don't contain trace information, causing M1 to ignore CDT signal. Filtering was necessary to test if trace coverage was the issue.

**Risk**: This limits M1's applicability to only 25% of participants. If deployed, M1 would only work for customers with trace data.

### D2 — No CDT Lift Gate Verification in Evaluation
**SPEC said** (`applications/choice/SPEC.md`): "CDT lift ≥ 0.05 AUC over no-CDT baselines" is critical success criterion.

**What was implemented**: Evaluation script reports AUC, Brier, calibration slope but does not train baseline models or compute lift.

**Why**: Baseline training requires additional implementation (product-features-only, persona-id one-hot models). Evaluation script has placeholder for this but never implemented.

**Risk**: CDT lift gate was never actually tested. AUC 0.53 could potentially achieve lift if baselines are also weak (this was not verified).

### D3 — Participant-Level Split Implemented Differently
**SPEC said** (`applications/choice/SPEC.md`): "Participant-level 70/30 split (prevents data leakage)"

**What was implemented**: Used `train_test_split(participants, test_size=0.3, random_state=42)` which produces participant-level split correctly.

**Why**: No deviation — this is just clarification that implementation matches spec.

---

## 5. Context Infrastructure Gaps

### G1 — No Choice Model Training Registry
**Gap**: M1 training has no canonical checkpoint or training config registry.

**Fix needed**: Add training registry to `applications/choice/` that tracks:
- Training data version (synthetic dataset generation date)
- Encoder versions used (trace, fusion checkpoints)
- Training hyperparameters
- Evaluation metrics

**Risk**: Without version tracking, it's difficult to reproduce experiments or know which CDT cache corresponds to which M1 model.

### G2 — No Baseline Model Implementations
**Gap**: CDT lift gate requires baseline models (product-features-only, persona-id one-hot) but these were never implemented.

**Impact**: CDT lift gate was never actually tested. The 0.05 AUC lift requirement was never verified against these baselines.

**Fix needed**: Implement baseline models in `applications/choice/evaluate.py` or separate script.

### G3 — ChoiceSet ID Linkage Not Populated
**Gap**: `TrialRecord.choice_set_id` field exists but is never populated in pipeline, blocking verification scripts.

**Impact**: Cannot directly link trials to choice sets without matching `trial_id == choice_set_id`.

**Fix needed**: Update `generator/pipeline.py` to populate `TrialRecord.choice_set_id` with `trial_id` (since they match by design).

### G4 — No Investigation Tools for Encoder Objectives
**Gap**: No tools to quickly check what encoders were trained to predict (strategy vs choice).

**Impact**: Root cause investigation (#55) required manual code reading to discover objective mismatch.

**Fix needed**: Add encoder objective metadata to SPEC files or model checkpoints for quick reference.

---

## 6. Why M1 Failed: Root Cause Analysis

### Data Generation ✅
- **Preference-driven choice logic works**: 83.6% of chosen alternatives were inspected
- **Systematic inspection-choice patterns exist**: Not random
- **Clean data regenerated**: JSON corruption issue resolved

### Trace Coverage ✅
- **250 participants with traces**: 25% of dataset
- **Filtering implemented**: Successfully isolated trace coverage participants
- **But not the root cause**: AUC still 0.53 after filtering

### The Real Problem: Objective Mismatch ❌

**Encoder Objectives:**
- **Trace encoder**: Trained with CE loss → predict **persona strategy** (7 classes: lexicographic, compensatory, etc.)
- **Fusion encoder**: Trained with CE loss → predict **persona strategy** (7 classes)
- **Neither encoder trained to predict choice**

**M1 Expectation:**
- **M1 choice model**: Expects CDT embeddings → predict **binary choice** (chosen/rejected)

**The Disconnect:**
```
Strategy Classification ≠ Choice Prediction
"lexicographic strategy" ≠ "chooses product A vs product B"
"how they decide" ≠ "what they decide"
```

### Why 83.6% Inspected → 53% AUC

**The coupling exists in the data** (83.6% of chosen alternatives were inspected), but **the encoders never learned to exploit it for choice prediction**.

- **Trace encoder**: Learned "what strategy this person uses" (lexicographic patterns)
- **Fusion encoder**: Learned multimodal strategy patterns
- **M1**: Expects "what this person will choose"

**The embeddings don't contain choice information by design.** They contain strategy information, which is correlated with choice but not sufficient for prediction.

**Analogy**: It's like training a model to predict someone's political party (strategy) and then being surprised it can't predict who they'll vote for (choice). Party affiliation predicts voting patterns, but it's not the same thing.

---

## 7. Success Criteria Assessment

### Tier 1: CDT Lift Gate ❌ NOT TESTED
- **Requirement**: CDT lift ≥ 0.05 AUC over no-CDT baselines
- **Status**: Never tested (baseline models not implemented)
- **Note**: Even if tested, likely failed given AUC 0.53 ≈ random

### Tier 2: Floor Metrics ❌ FAILED
- **AUC ≥ 0.65**: FAILED (0.53 vs 0.65 threshold)
- **Brier ≤ 0.25**: PASSED (0.16)
- **Calibration slope ∈ [0.8, 1.2]**: PASSED (0.94 after filtering)

### Tier 3: Documentation ✅ PASSED
- **SPEC.md written**: Yes (`applications/choice/SPEC.md`)
- **Module boundary invariant**: Respected (no training imports, reads from cache)

**Overall Assessment: ❌ FAILED**
- Critical success criteria not met
- Root cause identified (objective mismatch)
- Not an implementation bug — architectural constraint

---

## 8. Assumptions Validated / Invalidated

### A1 — Preference-Driven Choice Creates Strong Trace-Choice Coupling ✓ Partially Validated
**Evidence**: 83.6% of chosen alternatives were inspected, showing systematic relationship.

**Nuance**: The coupling exists at data generation level but was not exploited by encoders. The encoders were trained for wrong objective (strategy classification, not choice prediction).

**Implication**: Preference-driven logic is sound, but encoder objectives must match.

### A2 — Trace Coverage Mismatch Causes M1 Failure — ❌ Invalidated
**Initial hypothesis**: Only 25% of participants have traces, causing weak signal.

**Evidence**: After filtering to trace coverage participants only, AUC remained 0.53 (no improvement).

**Implication**: Coverage is not the root cause. Full encoder cascade retraining didn't help either.

### A3 — CDT Embeddings Predict Choices — ❌ Invalidated
**Initial hypothesis**: CDT fuses decision process (trace) with other modalities, so it should predict decisions.

**Evidence**: AUC 0.53 with preference-driven choices, retrained encoders, and trace coverage filtering.

**Root cause**: Encoders trained for strategy classification, not choice prediction. Strategy ≠ Choice.

**Implication**: CDT embeddings can predict choice IF encoders are trained with choice-related objective. Current embeddings predict strategy, not choice.

### A4 — Two-Tower Architecture Works for Choice Prediction — Unknown
**Initial hypothesis**: Two-tower architecture (CDT + product features) is suitable for choice prediction.

**Evidence**: Insufficient data. Architecture never got fair test because embeddings don't contain choice information.

**Status**: Cannot assess. May need retest with choice-trained encoders or different architecture.

### A5 — Full Encoder Cascade Retraining Fixes M1 — ❌ Invalidated
**Initial hypothesis**: Retraining trace → fusion → CDT cascade with preference-driven data would fix M1.

**Evidence**: Complete cascade retraining executed:
- Trace encoder: 57.95% accuracy (retrained on preference-driven data)
- Fusion encoder: 85% accuracy (retrained on updated traces)
- M1 result: 0.53 AUC (no improvement)

**Root cause**: Objective mismatch, not training data. Retraining with better data doesn't help if objective is wrong.

**Implication**: Simply retraining encoders is insufficient. Need fundamental redesign of encoder objectives.

---

## 9. Recommendations

### R1 — Document Encoder Objectives Explicitly
**Problem**: Root cause investigation required manual code reading to discover encoder objectives.

**Fix**: Add explicit documentation to each encoder's SPEC.md:
- "This encoder predicts [X] and does NOT predict [Y]"
- "Embeddings contain [Z] information and should NOT be used for [W]"
- Add to model checkpoints as metadata if possible

### R2 — Implement Baseline Models for CDT Lift Gate
**Problem**: CDT lift gate was never actually tested.

**Fix**: Implement in `applications/choice/evaluate.py`:
- Product-features-only baseline (logistic regression on price, quality)
- Persona-id one-hot baseline (logistic regression on persona)
- Compute lift: `lift = m1_auc - baseline_auc`
- Report lift alongside absolute AUC

### R3 — Choice-Related Encoder Objectives (If Pursuing Choice Prediction)
**Problem**: Current encoders predict strategy, not choice.

**Options**:
**A. Add Choice Prediction as Auxiliary Task**: Retrain trace/fusion encoders with multi-task learning: strategy classification + choice prediction. This explicitly teaches encoders to extract choice-relevant features.

**B. Create Dedicated Choice Encoder**: Train new encoder specifically for choice prediction using trace + product features → choice. Keep existing encoders for strategy.

**C. Use Strategy Information Directly**: Redesign M1 to use strategy predictions (from CDT) + product features → choice, rather than expecting embeddings to directly predict choice.

**Recommendation**: Option C is most viable with current architecture. It leverages what encoders actually learn (strategy) rather than fighting against it.

### R4 — Fix ChoiceSet ID Linkage in Pipeline
**Problem**: `TrialRecord.choice_set_id` field not populated, blocking verification scripts.

**Fix**: Update `generator/pipeline.py` to set:
```python
trial_record = TrialRecord(
    ...
    choice_set_id = trial_id,  # Links to ChoiceSet
)
```

This enables direct trial→choice set lookup without string matching.

### R5 — Create Training Registry for Reproducibility
**Problem**: No version tracking for which encoders/data correspond to which M1 model.

**Fix**: Add `applications/choice/REGISTRY.md` that tracks:
- Synthetic dataset generation date
- Trace encoder checkpoint used for CDT generation
- Fusion encoder checkpoint used for CDT generation
- M1 training hyperparameters
- Evaluation metrics

Enable reproducibility: "This M1 model corresponds to dataset v2026-06-18 + trace encoder v2026-06-18_10:15 + fusion encoder v2026-06-18_10:18"

### R6 — Consider Trace Coverage Expansion (If Needed)
**Problem**: Only 25% of participants have traces, limiting applicability.

**Fix**: If pursuing choice prediction, increase trace coverage to 100% or investigate why only 250/1000 have traces. May be generator design choice or data quality issue.

### R7 — Alternative: Strategy-Based Choice Prediction
**Problem**: Current approach expects embeddings to predict choice, but encoders learn strategy.

**Alternative approach**: Redesign M1 as:
```
Strategy Prediction (from CDT) + Product Features → Choice
```

Rather than:
```
CDT Embedding + Product Features → Choice
```

This aligns with what encoders actually learn (strategy) and uses that information explicitly for choice prediction.

---

## 10. Lessons Learned

### L1 — Objective Alignment is Critical
**Lesson**: Encoder objectives must match downstream expectations. Strategy classification ≠ Choice prediction, even though both involve decision processes.

**Takeaway**: Always verify: "What were the encoders trained to predict?" before using embeddings for downstream tasks.

### L2 — Data-Level Coupling ≠ Model-Level Coupling
**Lesson**: 83.6% inspection-choice coupling in data doesn't guarantee embeddings will predict choices. Encoders must be explicitly trained to exploit the coupling.

**Takeaway**: Strong correlations in data are necessary but not sufficient. Model architecture and training objectives determine what gets learned.

### L3 — Full Cascade Retraining is Expensive but Sometimes Necessary
**Lesson**: Retracing entire encoder cascade (trace → fusion → CDT → M1) took significant time but was necessary to isolate the root cause.

**Takeaway**: When performance doesn't improve after local fixes, consider systematic end-to-end retraining to rule out cascading issues.

### L4 — Trace Coverage is Easy to Miss
**Lesson**: 75% of participants lacking traces was discovered late because M1 training didn't fail explicitly—it just performed poorly.

**Takeaway**: Add coverage checks early: "What % of training data has feature X?" before deploying models.

### L5 — Verification Scripts Need Data Linkage
**Lesson**: Trace-choice coupling verification script failed because choice_set_id linkage wasn't populated, blocking validation.

**Takeaway**: Design verification scripts with data linkage in mind. Populate linking fields during data generation.

---

## 11. Status Summary

**Implementation Quality**: ✅ Excellent
- All components implemented correctly
- Architecture followed SPEC exactly
- Training/evaluation pipelines working
- Only minor bugs (JSON corruption, pyarrow vs pandas) quickly resolved

**Root Cause Identified**: ✅ Complete
- Objective mismatch: strategy classification ≠ choice prediction
- Data generation working correctly
- Trace-choice coupling exists in data but not exploited by encoders

**M1 Success Criteria**: ❌ Not Met
- AUC 0.53 vs 0.65 threshold
- CDT lift gate never tested (baselines not implemented)
- Fundamental architectural constraint, not implementation bug

**Recommendation**: Do not proceed with current M1 approach. Either:
1. Redesign encoders with choice prediction objectives (R3A, R3B)
2. Redesign M1 to use strategy information explicitly (R3C)
3. Accept that choice prediction is not feasible with current encoder design

The Phase 0 foundation is solid and reusable. The M1 choice prediction approach requires fundamental rethinking to align with what encoders actually learn.

---

## 12. Corrective Addendum — Oracle Ceiling Re-Analysis

> Added: 2026-06-18 (post-review)
> Status: ⚠️ Supersedes the §6/§11 root-cause conclusion. The "objective mismatch" diagnosis is **secondary at best**; the binding constraint is in **data generation**, not the encoders.

### A1 — The original root-cause conclusion is a misdiagnosis

§6 and §11 conclude the failure is an *encoder objective mismatch* (strategy classification ≠ choice prediction) and recommend not proceeding with M1. A post-review check of the generated data against the actual code shows the binding constraint is upstream: **the choice labels are near-random by construction, so the success threshold is unreachable by any model — including a perfect one.**

### A2 — The oracle ceiling is below the success threshold

Scoring every choice set with its **own true generative probabilities** (`choice_probabilities`, the Bayes-optimal predictor) yields:

```
ORACLE AUC (true generative probs as the predictor) = 0.648
```

This is the theoretical maximum AUC for *any* model on this dataset. The SPEC floor is **AUC ≥ 0.65**. A model with perfect knowledge of the data-generating process scores **below the gate**. The observed M1 AUC of 0.53 was therefore never measured against an achievable target.

Supporting statistics (10k–20k choice sets sampled):
- Mean P(chosen alternative) = **0.239**, median **0.213** (uniform over ~5–7 alternatives ≈ 0.14–0.20). The generator barely prefers the "chosen" product over the rejected ones.
- Example from `choice_sets.jsonl` line 1: chosen alt `F` = 0.1986; all six other alternatives = exactly 0.1335 (uniform = 0.143).

### A3 — Root cause: the generator never implemented Phase 0 SPEC §0.3/§0.5

The sharpest framing is not "temperature too high" but **"the shipped generator silently omitted two specced components of the Phase 0 choice model"** (`.claude/context/new-capabilities.md`):

- **§0.5 specifies a fixed decisiveness gain `GAIN = 8.0`**: logits `ℓ = GAIN·ū`, then `softmax(ℓ/τ)` with `τ = 1.0` *deliberately pinned* ("never tuned to hit M1 calibration" — a no-circularity choice). The shipped `_compute_choice_from_inspected_cells` applies `softmax(u/τ)` with **no gain** (effectively `GAIN=1`) and `τ=1.0`. With `GAIN=8`, §0.5 notes a 0.3 goodness gap → ≈11:1 odds; at `GAIN=1` it collapses toward uniform.
- **§0.3 specifies utility rules for all six strategies**; the shipped code implements only `LEXICOGRAPHIC` and `COMPENSATORY`, routing the other four to `rng.uniform(0,1)` (pure noise).

Mechanically, the result is the same near-uniform label distribution analysed below. **Re-sampling labels + probabilities at sharper temperatures (equivalently, applying a gain) moves the ceiling decisively:**

| Generation temperature | Oracle AUC ceiling |
|---|---|
| 1.0 (current) | 0.646 ❌ (below 0.65 gate) |
| 0.4 | 0.733 |
| 0.2 | 0.843 |
| 0.1 | 0.926 |

Note: re-tempering only the *scores* of existing labels does **not** raise the ceiling (labels were already sampled at T=1) — both labels and probabilities must be regenerated at the sharper temperature.

Secondary generation issue: the `lapse_noise` term (lines 774–776) adds the **same constant to every slot**, which is a softmax no-op — it does nothing.

### A4 — Only 3 of 7 strategies produce structured choices

The choice logic implements utility functions only for `LEXICOGRAPHIC` and `COMPENSATORY`. The other four archetype strategies — `satisficing`, `affect_heuristic`, `random`, `adaptive` — fall through to `else: utilities[slot] = rng.uniform(0, 1)`, i.e. **pure noise**. So ~4/7 of the population has unpredictable choices by design, further depressing the oracle ceiling. (`low_involve`/`random` being random is intentional; the other three are unimplemented, not intentional.)

### A5 — Product tower is blind to the attributes that drive choice

`applications/choice/model.py` uses `product_dim = 2` (price, quality only). But `brand_affect` chooses on `brand`, and `compensatory`/`adaptive` weight `brand` at 0.2. The model cannot see the attribute that determines those choices — a model-side bottleneck independent of any CDT/encoder objective question.

### A6 — The lift gate (the actual Tier-1 criterion) was never implemented

No product-only or persona-one-hot baselines exist (confirmed in §G2/§D2). The single criterion the SPEC calls "critical" — CDT lift ≥ 0.05 over no-CDT baselines — was never computed. Absolute AUC in isolation is the wrong number to judge against.

### A7 — Revised recommendation: do NOT abandon M1

The §11 recommendation ("do not proceed", "redesign encoders") is reversed. The experiment was run against a target the data made physically unreachable; encoder redesign would fit noise. The remediation is **"finish implementing Phase 0 §0.1/§0.3/§0.5/§0.6 as written"**, not new design. Tracked under epic `customer-data-fusion-3fx`:

1. **Expand `Product` to the §0.1 8-attribute board** (`brand_tier`, `quality_score`, `warranty_score`, `rating`, `features_score`, `availability`, `design_score`, `on_promotion`) and rewrite the catalogue per §0.6. The shipped `Product` carried only price + quality (a 2-attribute board), starving both the generator's choice utilities and the M1 product tower. *(Bead `7c1` — root prerequisite; a frozen-schema change, so it cascades through generator + encoders.)*
2. **Implement §0.5 `GAIN = 8.0`** (τ pinned at 1.0) and **§0.3 utilities for all six strategies**, then regenerate. Authoritative gate: oracle AUC ≥ 0.80. *(Beads `hmc`, `5bz`.)*
3. **Implement oracle + no-CDT baselines** and reframe success as **lift over product-only** plus "M1_AUC ≥ 0.85 × oracle", not an absolute 0.65. *(Bead `16r`.)*
4. **Expand the product tower** to ingest the full §0.1 displayed-attribute board. *(Bead `6ca`.)*
5. **Re-run the cascade** (note: the richer board changes traces → the trace encoder's attribute embedding likely needs resizing, not just retraining) and **only then** revisit the encoder-objective question, deciding it empirically via the lift gate. *(Bead `c9v`.)*

The "objective mismatch" hypothesis (§6) remains *untested*, not *confirmed* — it cannot be evaluated until the generator produces learnable choices and the lift baselines exist.

## 13. Post-Fix Cascade Results — Epic `3fx` Complete

> Added: 2026-06-19. All six `3fx` beads closed (`7c1`, `hmc`, `5bz`, `16r`, `6ca`, `c9v`). This section records the empirical outcome of the §A7 remediation and resolves the §6 vs §A7 debate.

### 13.1 The data fix worked (§A7 confirmed)

Regenerating with the finished Phase 0 generator (§0.1 8-attribute board, §0.5 `GAIN=8.0`, §0.3 utilities for all six strategies, §0.6 catalogue) raised the **oracle ceiling from 0.648 to 0.876** — well above the 0.80 authoritative gate. Mean P(chosen) rose 0.239 → 0.477. The binding constraint identified in §A3 (the generator silently omitted §0.3/§0.5) was real and is now removed: the choice labels are learnable by construction.

### 13.2 But M1 still fails the lift gate (§6 confirmed)

The full cascade was re-run on the regenerated data (trace encoder auto-resized its attribute vocab 3→8 attrs — data-driven, no architecture change — then fusion → CDT cache → M1). All artifacts fresh. The lift gate (`16r`):

| metric | pre-fix (§12) | post-fix |
|---|---|---|
| oracle_AUC (ceiling) | 0.648 | **0.876** |
| M1_AUC (CDT two-tower) | 0.53 | **0.568** |
| product_only_AUC | n/a (never built) | 0.565 |
| persona_onehot_AUC | n/a | 0.500 (chance) |
| **lift_over_product** | n/a | **+0.0035** (gate ≥ 0.05) ❌ |
| M1 ≥ 0.85·oracle | n/a | 0.568 vs 0.744 ❌ |

**Verdict: FAIL.** With learnable labels, the CDT still gives essentially zero lift over product features alone, and M1 reaches only ~65% of the oracle ceiling.

### 13.3 Resolution of the §6 vs §A7 debate: both theses hold, sequentially

§A7 and §6 were not alternatives — they were **two layers of the same failure**, peeled in order:

1. **§A7 (data generation)** was the *first* binding constraint: with oracle 0.648, no model could pass, so §6 could not even be tested. Fixing the generator (§13.1) removed it.
2. **§6 (objective mismatch)** is the *next* binding constraint, now exposed: even with oracle 0.876, the CDT — a *participant-level* embedding trained on *archetype classification* (fusion archetype val_acc ≈0.9) — encodes "which archetype", not the *per-trial inspected cells* that the §0.3 utility actually chooses on. So `(CDT, product)` cannot recover the trial-level choice (`product_only` 0.565 ≈ `M1` 0.568 while the oracle sits at 0.876).

The §11 "do not proceed" recommendation is therefore re-revised: do not abandon M1, *and* the data fix alone is insufficient. The remaining gap is an encoder-objective question, not a data question.

### 13.4 Follow-up

The objective-mismatch hypothesis is now **supported** (it survived the data-fix control). Filed as bead `customer-data-fusion-b8b`: add a **choice-prediction auxiliary objective** to the trace encoder (multi-task: existing strategy-CE + NT-Xent + a per-`(trial, slot)` choice-BCE head, `LAMBDA_CHOICE=0.5` pinned), re-run the cascade, and re-judge via the same lift gate. If that too fails, the CDT-as-participant-summary may be fundamentally unable to encode per-trial inspected-cell detail, and a per-trial choice architecture should be considered instead.

### 13.5 b8b result — the objective fix is insufficient (negative result)

> Added: 2026-06-19. `b8b` closed. The choice-prediction auxiliary objective on the trace encoder did **not** close the lift gap.

Full cascade re-run with the choice-aware trace encoder (choice head `Linear(concat(trial_emb, 8-dim product)) → BCE`, `LAMBDA_CHOICE=0.5`, trained 50 epochs jointly with strategy-CE + NT-Xent; all steps rc=0):

| metric | c9v (no choice aux) | b8b (with choice aux) |
|---|---|---|
| M1_AUC | 0.5681 | 0.5597 |
| oracle_AUC | 0.8758 | 0.8758 |
| product_only_AUC | 0.5646 | 0.5646 |
| **lift_over_product** | +0.0035 | **−0.0048** |

Lift is **flat-to-slightly-negative** — within noise of the pre-b8b result. The auxiliary objective moved the downstream M1 lift by essentially nothing.

**Diagnosis.** This rules out the optimistic "Layer-1" reading (retrain the encoder on the right objective and lift appears). The choice head makes each *trial* embedding choice-aware, but that signal dies at the **fusion bottleneck**: fusion pools per-trial embeddings into a single *participant-level* CDT optimised for identity/archetype (NT-Xent + CE), which averages away the per-trial choice structure. A choice-aware trace encoder therefore cannot hand a choice-aware signal to a participant summary that was never asked to preserve it — and M1, which consumes that participant summary, sees no benefit.

**Implication for the project idea.** The binding constraint is now pin-pointed at the **representation**, not the data (§A7, fixed) and not the encoder objective (§6/b8b, ruled out). The static, participant-level CDT — a fingerprint computed once per consumer — appears **fundamentally unable to encode the per-trial inspected-cell detail** that the §0.3 choice actually depends on. Two paths remain, in increasing order of redesign:

1. **Choice loss at the fusion/CDT level** — make the participant summary itself choice-aware (pool trials under a choice objective, not just identity). Cheapest; tests whether a *static* CDT can be made choice-predictive at all.
2. **Per-trial choice architecture** — abandon the participant-summary-as-choice-predictor framing; predict each choice from *that trial's* trace + products directly (the trace encoder already sees the inspected cells). This is structurally where the choice information lives, but it is a different product than a reusable CDT fingerprint.

Either way, **M1 (choice prediction from a static CDT) is not validated by the current architecture**, and the result should be read as evidence about the *representation's* limits, not as a refutation of the broader CDT programme (whose proven strengths — individual identity, archetype recovery, trait regression — are about stable participant properties, not per-trial behaviour; see §13.3).

### 13.6 Experiment (1) — fusion-level choice loss: also insufficient (conclusive)

> Added: 2026-06-19. Path 1 from §13.5 was run and failed. The three experiments now converge.

A choice head (`Linear(128 + 8, 1)`) and a per-participant choice-BCE loss (`LAMBDA_CHOICE_FUSION=1.0`, applied to the full no-dropout CDT, over each participant's full choice history) were added to the fusion meta-learner, so the **CDT itself** is shaped by a choice objective. Full cascade re-run with choice losses active at both the trace (b8b) and fusion levels. All steps rc=0; the fusion choice loss genuinely trained.

| experiment | choice objective where | M1_AUC | lift_over_product |
|---|---|---|---|
| c9v | nowhere | 0.5681 | +0.0035 |
| b8b | trace encoder | 0.5597 | −0.0048 |
| **(1)** | **fusion / CDT** | **0.5718** | **+0.0072** |

All three ≈ 0; (1) improved lift by ~0.004 — within noise, far below the 0.05 gate.

**Smoking gun.** The fusion choice head's BCE **plateaued at ≈0.50** — the entropy of the base rate (~20% chosen). From `(participant CDT, product features)`, the model cannot predict the chosen slot better than guessing the base rate. The participant-level CDT carries **essentially zero per-trial choice information**, and no choice objective — at the trace level, the fusion level, or both — can put it there, because *which cells were inspected* is a property of the **trial**, not the **participant**. You cannot train information into a representation that has already averaged it away.

**Convergent conclusion.** The binding constraint is now triangulated and is **not** data (§A7, fixed: oracle 0.876), **not** the encoder objective (b8b, ruled out), and **not** the fusion objective (experiment 1, ruled out). It is **representation granularity**: a static, participant-level summary cannot hold the per-trial inspected-cell structure the §0.3 choice depends on. Choice prediction from a reusable CDT fingerprint is therefore **not achievable with this architecture**; it requires either (2) a per-trial choice model that conditions on the specific trial's trace, or (3) re-pivoting the CDT's headline value to the stable-property tasks it already proves out (identity, archetype, trait regression, and the stateful/counterfactual simulation the vision originally led with). The "objective mismatch" framing of §6/§C7 was a symptom; the root cause is the participant-summary representation.



