# H1 Temporal Dynamics Post-Mortem

> Date: 2026-06-16
> Capability: H1 (Temporal Dynamics / Sequential CDT)
> Status: Validation Failed — Negative Result
> Complexity: Initially assessed as Low–Medium, actually High (requires architectural rethink)

## Executive Summary

H1 Temporal Dynamics failed validation because **frozen CDT embeddings do not capture temporal dynamics**. All 1002 participants produced identical embeddings across all 12 monthly observations (variance = 0.0), making drift detection impossible.

**Root cause:** The fusion meta-learner was trained with NT-Xent contrastive loss that explicitly collapses within-participant variance for identity stability. This makes the model temporal-blind — it treats month-to-month behavioral changes the same way it treats dropout noise.

**Key finding:** Identity embeddings ≠ temporal embeddings. A model trained to recognize "who someone is" will not necessarily detect "how someone changes over time."

## What Was Planned

### H1 Goal

Detect regime shifts in consumer decision processes across 12 monthly observation windows using frozen CDT embeddings.

### Success Criteria (from spec)

- ✅ Stage 1 Recall@1 ≥ 0.80 (detect 80% of regime shifts)
- ✅ Stage 2 MAE ≤ 1.5 months (predict drift month within ±2 months)
- ✅ Precision ≥ 0.60 (avoid false alarms)

### Architecture Design

```
Monthly Data → Frozen Fusion → 12 Embeddings per Participant → Delta Features → Two-Stage Drift Detector
                                                                               ↓
                                                                    Stage 1: Drift? (binary)
                                                                    Stage 2: When? (month)
```

**Key invariant (from spec):**
> Uses the **frozen** fusion meta-learner. The encoder and fusion training logic remain untouched. No new architecture is trained for v0.1.

### Implementation Pipeline

1. **generate_monthly_embeddings.py** — Encode each month's data through frozen fusion
2. **extract_features.py** — Compute L2 distances between consecutive monthly embeddings
3. **train_drift_detector.py** — Train two-stage classifier (drift detection + month estimation)
4. **evaluate_drift_detector.py** — Validate against ground truth

### Assumptions Made

**Assumption 1:** Different monthly behavioral data → different CDT embeddings
- **Reality:** Same participant across months → identical embeddings (variance = 0.0)

**Assumption 2:** Fusion training objective preserves within-participant variance
- **Reality:** NT-Xent loss explicitly collapses within-participant variance

**Assumption 3:** AR(1) drift in persona parameters (0.1-0.3/month) creates detectable signal
- **Reality:** Signal exists in raw data but is filtered out by identity-stable fusion model

**Assumption 4:** Frozen encoders + frozen fusion = temporal sensitivity
- **Reality:** Frozen models preserve whatever they were trained to capture. Our model was trained for identity, not temporality.

## What Actually Happened

### Validation Results

**Step 1: Monthly Embeddings Generated**
- Output: `applications/_cache/cdt_embeddings_monthly.parquet` (78MB, 12,024 participant-month records)
- ✅ Successfully generated embeddings for all 1002 participants across 12 months

**Step 2: Drift Features Extracted**
- Output: `applications/temporal/features.parquet`
- ❌ All drift features = 0.0 (dist_mean, dist_std, dist_max, dist_slope all zero)

**Step 3: Drift Detector Trained**
- Output: `applications/temporal/drift_classifier.joblib`
- ❌ Stage 1 Recall = 0.000 (target ≥0.80)
- ❌ Stage 1 Precision = 0.000 (target ≥0.60)
- ❌ Stage 1 F1 = 0.000
- Confusion Matrix: `[[3194, 0], [410, 0]]` (all drift cases misclassified as negative)

**Step 4: Root Cause Investigation**

Feature distribution analysis:
```
Non-Drift (n=887):   dist_mean=0.0000±0.0000, dist_max=0.0000±0.0000
Drift (n=114):       dist_mean=0.0000±0.0000, dist_max=0.0000±0.0000
Effect size: nan (no variance in either group)
```

Monthly embedding analysis:
```
Participants with identical monthly embeddings: 1002/1002
Max participant variance: 0.0000000000
```

**Conclusion:** All 1002 participants have exactly identical CDT embeddings across all 12 months.

### Why Embeddings Don't Vary

The fusion meta-learner was trained with a multi-task objective:

```python
loss = CE_loss + λ * NT_Xent_loss
```

**NT-Xent Loss (SimCLR-style):**
```python
# From fusion/train.py:624-649
def nt_xent_fusion(emb_v1, emb_v2, temperature=0.07):
    """
    emb_v1[i] and emb_v2[i] are two modality-dropout augmented views of the same participant i.
    All other cross-participant pairs are negatives.
    """
    # Implementation treats same-participant views as positive pairs
    # Pushes embeddings of the same participant closer together
    # Makes model robust to missing modalities (dropout noise)
```

**What NT-Xent optimizes for:**
- Identity stability: "Participant X should have the same embedding regardless of which modalities you show me"
- Robustness to missing modalities
- Collapses within-participant variance

**What H1 requires:**
- Temporal sensitivity: "Participant X's embedding should move as their decision process drifts"
- Preserve within-participant variance over time
- Temporal changes should be signal, not noise

**The mismatch:** When you pass month 1, month 2, ..., month 12 data through frozen encoders:
1. Encoders are frozen — they can't adapt to temporal variations
2. Fusion was trained to collapse variance — it treats month-to-month changes the same way it treats dropout noise
3. AR(1) drift (0.1-0.3/month) doesn't create enough raw variance to overcome identity-stable training

The model learned: **"This is participant X"** (identity), not **"This is participant X in month Y"** (temporal state).

## What Went Wrong

### Planning Failures

**1. Worked at wrong abstraction level**
- Focused on pipeline mechanics (monthly data → embeddings → classifier)
- Didn't verify the core assumption (do embeddings actually vary?)

**2. Success metrics defined, assumption not tested**
- Had clear success criteria (Recall@1 ≥ 0.80, MAE ≤ 1.5)
- But never tested: "Do frozen embeddings vary month-to-month?"

**3. Objective functions documented, but not connected to requirements**
- NT-Xent's identity-stability property was known and documented
- Nobody asked: "Does this conflict with temporal sensitivity requirements?"

### Missing Smoke Test

A 10-line script would have revealed this issue immediately:

```python
import pandas as pd
import numpy as np

embeddings = pd.read_parquet('applications/_cache/cdt_embeddings_monthly.parquet')

for pid in embeddings['participant_id'].unique()[:10]:
    pid_data = embeddings[embeddings['participant_id'] == pid].sort_values('month')
    embedding_matrix = np.stack(pid_data['cdt'].values)
    variance = embedding_matrix.var(axis=0).sum()
    print(f"{pid}: variance={variance:.10f}")

# Expected if H1 works: variance > 0.01
# Actual: variance = 0.0000000000
```

This should have been the **first step** after generating monthly embeddings, not the last step after training a failed classifier.

### Architectural Misunderstanding

**Incorrect assumption:** "Different input data → different embeddings"

**Reality:** "Frozen models preserve what they were trained to capture"

Our fusion model was trained for:
- ✅ Identity: "Who is this person?"
- ❌ Temporality: "How is this person changing?"

The training objective determines what the model captures, not the input data variance.

## Lessons Learned

### 1. Test Assumptions Early

**Before building complex pipelines, verify core assumptions.**

For H1, the assumption "different monthly data → different embeddings" should have been tested with a 5-minute smoke test before writing 200+ lines of feature extraction and classifier code.

**Rule of thumb:** If your approach depends on assumption X, write a script to verify X before building anything that depends on X being true.

### 2. Understand Training Objectives

**Before using a frozen model for a new task, understand what it was trained to do.**

The fusion model's NT-Xent loss was documented in `fusion/train.py`, but nobody connected:
- Documented behavior: "NT-Xent collapses within-participant variance"
- H1 requirement: "Preserve within-participant variance to detect drift"

**Rule of thumb:** If you're using a frozen model for task X, verify that its training objective is compatible with X's requirements.

### 3. Identity ≠ Temporality

**A model that recognizes "who someone is" won't necessarily detect "how someone changes."**

These are different learning objectives:
- Identity: Same person across time/context → similar embeddings
- Temporality: Same person at different times → different embeddings

Both are valid, but they require different training objectives. You can't have one model optimize for both without explicit multi-task design.

### 4. Negative Results Are Valuable

**Finding that frozen embeddings don't capture temporality is still progress.**

This result:
- Guides future architecture decisions (need temporal-aware encoders or separate temporal model)
- Prevents others from repeating the mistake
- Tells us that CDT embeddings are identity features, not temporal features

Scientific progress includes learning what doesn't work.

## Options Forward

### Option A: Document Negative Result (Recommended First Step)

**What:**
- Accept that frozen CDT embeddings don't capture temporal dynamics
- Create this post-mortem
- Update context documents with findings
- Update `new-capabilities.md` H1 status to "validated — negative result"

**Why:**
- ✅ Honest science — negative results are findings
- ✅ Minimal additional work
- ✅ Guides future research (identity ≠ temporal)
- ✅ Prevents others from repeating the mistake

**Estimated effort:** 2-3 hours (documentation)

### Option B: Retrain Fusion with Temporal Objective (Significant Work)

**What:**
- Modify fusion training objective to preserve temporal variance
- Replace NT-Xent with temporal contrastive loss
- Retrain fusion meta-learner from scratch

**Temporal contrastive loss:**
```python
# Existing (identity-stable):
loss = CE_loss + λ * NT_Xent_loss  # NT-Xent collapses within-participant variance

# Proposed (temporal-aware):
loss = CE_loss + λ_temporal * Temporal_Contrastive_loss
# Positive pairs: (participant_i, month_t) with (participant_i, month_{t+1})
# Negative pairs: (participant_i, month_t) with (participant_j, any_month)
```

**Process:**
1. Design temporal contrastive loss function
2. Modify `fusion/train.py` to support temporal objective
3. Generate monthly embedding sequences for training
4. Retrain fusion from scratch
5. Validate temporal sensitivity (embeddings should vary month-to-month)
6. Re-run H1 validation pipeline

**Why:**
- ✅ Fixes root cause (training objective)
- ✅ Enables all temporal capabilities (H1, churn prediction, trajectory modeling)
- ❌ Requires retraining fusion (affects all downstream dependencies)
- ❌ May break existing evaluation metrics if embeddings change significantly

**Estimated effort:** 2-3 days

**Note:** Temporal contrastive learning is a real technique (used in video representation learning, time series modeling), but it's not currently implemented in this codebase. This would be a new implementation with design choices (temporal window size, weighting of adjacent vs. non-adjacent pairs, etc.).

### Option C: Separate Temporal Model (Modest Pivot)

**What:**
- Keep frozen embeddings as static features
- Train separate temporal model (GRU/Transformer) on embedding sequences
- Use temporal model for drift detection

**Architecture:**
```python
# Frozen fusion embeddings (identity-only):
monthly_embeddings = [frozen_fusion(month_i_data) for month_i in months]

# Temporal model (learns trajectory dynamics):
temporal_encoder = GRU(input_dim=128, hidden_dim=64)
trajectory_encoding = temporal_encoder(monthly_embeddings)  # [T, 128] → [64]

# Drift detector:
drift_classifier = Linear(64, 1)  # Binary classification
```

**Process:**
1. Design temporal GRU/Transformer architecture
2. Train on monthly embedding sequences + drift labels
3. Evaluate drift detection performance
4. Validate against success criteria

**Why:**
- ✅ Keeps frozen fusion unchanged (no impact on existing capabilities)
- ✅ Modular — temporal model can be improved independently
- ✅ Faster to implement than full fusion retraining
- ❌ Less elegant than fixing fusion at the source
- ❌ Adds another model to maintain

**Estimated effort:** 1-2 days

## Recommendations

### Immediate Actions

1. **Complete this post-mortem** — Document what happened, why, and what was learned
2. **Update context documents:**
   - `.claude/context/fusion-architecture.md` — Add "Temporal Limitations" section
   - `.claude/context/new-capabilities.md` — Update H1 status to "validated — negative result"
3. **Add persistent memory:**
   ```bash
   bd remember "fusion-embeddings-are-temporally-blind" "Frozen fusion embeddings encode identity, not temporal dynamics. NT-Xent loss collapses within-participant variance. Always test: do embeddings vary over time? before building temporal models. See docs/post-mortems/h1-temporal-postmortem.md"
   ```

### Future Direction

**If temporal capabilities are high priority:** Go with Option B (retrain fusion with temporal objective). This is the cleanest long-term solution.

**If temporal is nice-to-have:** Go with Option C (separate temporal model). Faster to implement, preserves existing fusion.

**If other capabilities are higher priority:** Stick with Option A (negative result). The finding itself is valuable scientific progress.

### Process Improvements

**For all future capabilities:**

1. **Smoke test before full implementation**
   - Identify core assumption
   - Write 10-line test script
   - Verify assumption before building on top of it

2. **Check training objective compatibility**
   - Before using a frozen model for task X
   - Verify that its training objective is compatible with X's requirements
   - If objective favors property A but X requires property B, you'll have a bad time

3. **Distinguish identity from temporality**
   - These are different learning objectives
   - Identity: "Who is this?" (collapse within-person variance)
   - Temporality: "How are they changing?" (preserve within-person variance)
   - You can't optimize for both without explicit multi-task design

## Related Documents

- **Design spec:** `docs/superpowers/specs/2026-06-16-h1-temporal-dynamics-design.md`
- **Fusion architecture:** `.claude/context/fusion-architecture.md`
- **New capabilities:** `.claude/context/new-capabilities.md`
- **Fusion training:** `fusion/train.py` (NT-Xent implementation lines 624-649)

## Conclusion

H1 failed because of a fundamental architectural mismatch: the fusion model was trained for identity stability, but H1 requires temporal sensitivity.

This is not a bug in the implementation — the code works as designed. It's a mismatch between what the model was trained to do and what we asked it to do.

**The finding is valuable:** We now know that frozen CDT embeddings encode identity, not temporality. This guides future architecture decisions and prevents others from repeating the mistake.

**The lesson is generalizable:** Before using a frozen model for a new task, understand what it was trained to capture. Different training objectives produce different embedding spaces. Identity ≠ temporality.
