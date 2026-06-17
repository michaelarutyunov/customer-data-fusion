# Temporal-Aware Fusion Retraining — Design Specification

> Date: 2026-06-17
> Status: Design Approved, Pending Implementation
> Complexity: High (fusion retraining + H1 validation)

## Overview

Retrain the fusion meta-learner with a three-term loss function that balances archetype classification, identity stability, and temporal dynamics. This enables H1 (Temporal Dynamics) while preserving existing fusion capabilities.

**Current state:** Fusion model trained with NT-Xent + CE optimizes for identity stability, producing identical monthly embeddings (variance = 0.0). See `docs/post-mortems/h1-temporal-postmortem.md`.

**Goal:** Add temporal contrastive loss to preserve within-participant variance across months, enabling regime shift detection.

## Architecture

### Current Training Objective (Identity-Only)

```python
loss = CE_loss + 0.5 * NT_Xent_loss
```

- **CE loss:** Archetype classification (7 classes)
- **NT-Xent loss:** Identity stability — two dropout-augmented views of same participant pushed together

**Result:** Embeddings encode "who is this?" but not "how are they changing?"

### Proposed Training Objective (Identity + Temporal)

```python
loss = CE_loss + 0.3 * NT_Xent_loss + 0.3 * Temporal_loss
```

**Three-term balanced multi-task:**

1. **CE loss** — Archetype classification (preserves Tier-1 gate)
2. **NT-Xent loss** (λ₁ = 0.3) — Identity stability (reduced from 0.5 to make room for temporal)
3. **Temporal loss** (λ₂ = 0.3) — Temporal variance (new term)

### Temporal Contrastive Loss Function

```python
def temporal_contrastive_loss(monthly_embeddings, temperature=0.07):
    """
    Temporal contrastive loss for monthly embedding sequences.

    Parameters
    ----------
    monthly_embeddings : Tensor, shape [B, 12, 128]
        B participants, 12 monthly observations each, 128-dim embeddings
    temperature : float
        NT-Xent temperature parameter (default 0.07)

    Returns
    -------
    Scalar loss tensor

    Design
    -------
    Positive pairs: (participant_i, month_t) with (participant_i, month_{t+1})
    Negative pairs: (participant_i, month_t) with (participant_j, any_month)

    Total positive pairs per batch: B * 11 (11 adjacent month pairs per participant)
    Total negative pairs: B * 11 * (B - 1) * 12 (all cross-participant pairs)

    This encourages the model to preserve temporal variance — same participant at
    different months should have different embeddings that still maintain identity.
    """
    B, T, D = monthly_embeddings.shape

    # Flatten to [B*T, D] for SimCLR-style contrastive
    embeddings_flat = monthly_embeddings.reshape(-1, D)  # [B*T, D]
    embeddings_norm = F.normalize(embeddings_flat, dim=1)

    # Create positive pair indices: (i, t) pairs with (i, t+1)
    # Shape: [B * (T-1), 2] where each row is [pos_idx, neg_idx]
    positive_pairs = []
    for i in range(B):
        for t in range(T - 1):
            idx1 = i * T + t      # (participant_i, month_t)
            idx2 = i * T + t + 1  # (participant_i, month_{t+1})
            positive_pairs.append([idx1, idx2])

    positive_pairs = torch.tensor(positive_pairs, device=monthly_embeddings.device)

    # Compute similarity matrix: [B*T, B*T]
    sim_matrix = torch.mm(embeddings_norm, embeddings_norm.t()) / temperature

    # NT-Xent loss: positive pairs have high similarity, negative pairs low
    # Implementation follows standard SimCLR pattern
    # ... (full implementation in fusion/temporal_loss.py)
```

**Design choices:**

- **Adjacent months only** (not all-pairs): Captures essential temporal signal without collapsing all time-steps to similarity
- **SimCLR-style formulation**: Reuses proven contrastive learning pattern from NT-Xent
- **L2 normalization**: Removes scale bias, focuses on angular similarity

## Components

### 1. Temporal Data Generator (`fusion/temporal_data.py` — new)

**Purpose:** Generate monthly embedding sequences for temporal training.

**Input:**
- Month-partitioned modality files: `{modality}_month_MM.jsonl` (M = 1..12)
- Frozen modality encoders (loaded from checkpoints)

**Output:**
- `models/temporal_embeddings_cache.pt` — tensor dict with keys:
  - `monthly_embeddings`: [N_participants, 12, 128]
  - `participant_ids`: list of participant IDs
  - `months`: [12] tensor (month indices)

**Process:**
```python
# Pseudocode
for month in 1..12:
    load all 6 modality files for this month
    pass through frozen encoders
    accumulate by participant_id

# Stack into [N, 12, 128] tensor
# Each row = one participant, 12 monthly embeddings
```

**Error handling:**
- Missing months: Pad with `temporal_missing_embedding` (learnable parameter)
- Missing participants: Skip (not all participants appear in all months)
- Validation: Log warnings if > 10% data missing

### 2. Temporal Loss Function (`fusion/temporal_loss.py` — new)

**Purpose:** Implement temporal contrastive loss.

**Interface:**
```python
def temporal_contrastive_loss(
    monthly_embeddings: Tensor,  # [B, 12, 128]
    temperature: float = 0.07,
    missing_mask: Tensor | None = None,  # [B, 12] bool tensor
) -> Tensor:
    """Compute temporal contrastive loss."""
```

**Implementation details:**
- Create positive pair indices for adjacent months
- Compute similarity matrix
- Apply NT-Xent contrastive loss formula
- Handle missing months via mask (exclude from loss)

### 3. Modified Training Loop (`fusion/train.py` — modify)

**New CLI arguments:**
```python
parser.add_argument("--temporal-weight", type=float, default=0.3,
                   help="Weight for temporal contrastive loss")
parser.add_argument("--temporal-data", type=str,
                   default="models/temporal_embeddings_cache.pt",
                   help="Path to temporal embeddings cache")
```

**Security note:** All `torch.load()` calls must use `weights_only=True` (or set `TORCH_FORCE_WEIGHTS_ONLY_LOAD=1`) to prevent arbitrary code execution through pickle. The temporal cache contains only tensors and is safe to load with `weights_only=True`.

**Training loop changes:**
```python
# Load temporal data
temporal_cache = torch.load(args.temporal_data)
monthly_embeddings = temporal_cache["monthly_embeddings"]  # [N, 12, 128]
participant_ids = temporal_cache["participant_ids"]

# Create dataset that samples (participant, month_t, month_{t+1}) triples
# ... (dataloader setup)

# Training loop
for batch in dataloader:
    # Three loss terms
    ce_loss = CE_loss(logits, labels)
    nt_loss = nt_xent_fusion(emb_v1, emb_v2, temperature)
    temp_loss = temporal_contrastive_loss(monthly_batch, temperature)

    loss = ce_loss + args.lambda_contrastive * nt_loss + args.temporal_weight * temp_loss

    loss.backward()
    optimizer.step()
```

**Backward compatibility:**
- If `--temporal-data` not provided, fall back to current single-session training
- Existing `fusion.train` CLI unchanged (new args are optional)

### 4. H1 Validation Update (`applications/temporal/` — modify)

**Changes to existing scripts:**

1. **`generate_monthly_embeddings.py`**
   - Update to use new checkpoint: `fusion_metalearner_temporal.pt`
   - Verify embeddings now vary month-to-month (variance > 0.01)

2. **`extract_features.py`**
   - No changes needed (just reads embeddings)

3. **`train_drift_detector.py`**
   - Should now pass success criteria (drift features non-zero)

## Data Flow

### Training Pipeline

```
Monthly Modality Files (12 months × 6 modalities)
    ↓
fusion/temporal_data.py (new script)
    ↓
models/temporal_embeddings_cache.pt [N_participants, 12, 128]
    ↓
fusion/train.py --temporal-weight=0.3 --temporal-data=<cache>
    ↓
fusion_metalearner_temporal.pt (new checkpoint)
    ↓
Validation: H1 temporal pipeline (existing scripts)
```

### Inference Pipeline (Unchanged)

```
Single-session modality data
    ↓
Frozen encoders
    ↓
Fusion metalearner (new checkpoint)
    ↓
CDT embedding [128]
```

**Key invariant:** Inference path doesn't change. Users still call `fusion_model(data)` and get a 128-dim embedding. Only the training objective and weights change.

### Cache Management

**Two separate caches:**

1. **Existing cache:** `models/fusion_embeddings_cache.pt` — single-session embeddings
   - Used for: Identity-only fusion training, encoder retraining validation

2. **New cache:** `models/temporal_embeddings_cache.pt` — monthly sequences
   - Used for: Temporal fusion training

**Why separate:** Temporal cache has different shape [N, 12, 128] vs [N, 128]. Keeping them separate avoids breaking existing pipelines.

## Error Handling & Edge Cases

### Missing Months

**Scenario:** Participant has gaps (e.g., months 1-8, skip 9, months 10-12)

**Solution:** Pad missing entries with learnable `temporal_missing_embedding` (similar to existing modality MISSING embedding)

**Implementation:**
```python
# In temporal_data.py
class LateFusionMetaLearner:
    def __init__(self, ...):
        self.temporal_missing_embedding = nn.Parameter(torch.zeros(128))

# In temporal_contrastive_loss()
missing_mask = (monthly_embeddings != MISSING_MARKER)  # [B, 12] bool
monthly_embeddings_clean = torch.where(
    missing_mask.unsqueeze(-1),
    monthly_embeddings,
    model.temporal_missing_embedding.unsqueeze(0).unsqueeze(0)
)
```

### Insufficient Temporal Data

**Scenario:** Participant has < 2 months (can't form positive pairs)

**Solution:** Skip in temporal loss computation, still include in CE loss

**Implementation:**
```python
# In temporal_data.py
valid_participants = []
for pid in participant_ids:
    if participant_months[pid] >= 2:
        valid_participants.append(pid)
    else:
        log.warning(f"Participant {pid} has < 2 months, skipping temporal loss")
```

**Validation:** Log warning if > 10% of participants skipped

### Identity-Temporal Loss Divergence

**Scenario:** One loss goes up while other goes down (conflicting gradients)

**Solution:** Monitor both losses separately, add early stopping if identity recall@1 drops below 70%

**Implementation:**
```python
# In training loop
identity_recall = compute_recall_at1(val_embeddings)
if identity_recall < 0.70:
    log.warning(f"Identity degraded: {identity_recall:.3f} < 0.70")
    # Option: reduce temporal weight, continue training
    args.temporal_weight *= 0.8
```

**Fallback:** If identity degrades too much, reduce λ_temporal to 0.2

## Testing Strategy

### Unit Tests

**File:** `tests/fusion/test_temporal_loss.py`

1. **Positive pair construction:**
```python
def test_temporal_positive_pairs():
    embeddings = torch.randn(4, 12, 128)  # 4 participants, 12 months
    pairs = get_positive_pairs(embeddings)
    assert len(pairs) == 4 * 11  # 11 adjacent pairs per participant
```

2. **Missing month handling:**
```python
def test_temporal_missing_padding():
    embeddings = create_embeddings_with_missing(months=[5, 9])
    loss = temporal_contrastive_loss(embeddings)
    assert not torch.isnan(loss)
```

3. **Shape validation:**
```python
def test_temporal_shapes():
    embeddings = torch.randn(10, 12, 128)
    loss = temporal_contrastive_loss(embeddings)
    assert loss.dim() == 0  # scalar
```

### Integration Tests

**File:** `tests/fusion/test_temporal_training_integration.py`

1. **Full training run on 100 participants:**
```python
def test_temporal_training_converges():
    # Generate small temporal dataset
    cache = generate_temporal_cache(n_participants=100)
    
    # Train with temporal objective
    model = train_fusion(temporal_data=cache, epochs=5)
    
    # Verify convergence
    assert model.training_loss[-1] < model.training_loss[0]
```

2. **Monthly embeddings vary:**
```python
def test_monthly_embeddings_vary():
    cache = load_temporal_cache()
    embeddings = cache["monthly_embeddings"]  # [N, 12, 128]
    
    variance_per_participant = embeddings.var(dim=1).sum(dim=1)  # [N]
    assert (variance_per_participant > 0.01).all()  # all vary
```

3. **H1 validation passes:**
```python
def test_h1_temporal_gate():
    model = load_temporal_fusion_model()
    embeddings = generate_monthly_embeddings(model)
    features = extract_drift_features(embeddings)
    
    # Should now detect drift
    drift_recall = evaluate_drift_detector(features)
    assert drift_recall >= 0.60
```

### Regression Tests

**File:** `tests/fusion/test_temporal_regression.py`

Verify existing capabilities aren't broken:

1. **Identity recall@1 ≥ 70%:**
```python
def test_identity_preservation():
    model = load_temporal_fusion_model()
    recall = compute_dropout_recall(model, val_split)
    assert recall >= 0.70  # accepts degradation from 0.82
```

2. **Archetype recovery ≥ 85%:**
```python
def test_archetype_recovery():
    model = load_temporal_fusion_model()
    accuracy = compute_archetype_accuracy(model, val_split)
    assert accuracy >= 0.85  # current metric: 0.90
```

3. **PersonaConfig R² ≥ 0.70:**
```python
def test_config_regression():
    model = load_temporal_fusion_model()
    r2 = compute_config_r2(model, val_split)
    assert r2.mean() >= 0.70  # current metric: 0.75
```

## Success Criteria (Dual-Objective)

### Primary Gates (Both Must Pass)

**1. Temporal Capability — H1 Validation**
- Stage 1 Recall@1 ≥ 0.60 (down from original 0.80 to account for multi-task tradeoff)
- Stage 2 MAE ≤ 1.5 months (unchanged)
- Precision ≥ 0.60 (unchanged)

**2. Identity Preservation — Existing Fusion Capabilities**
- Dropout-view recall@1 ≥ 70% (down from current 82%, accepts some degradation)
- Archetype recovery ≥ 85% (unchanged from current 90%)
- PersonaConfig R² ≥ 0.70 (unchanged from current 0.75)

### Secondary Gates (Informative)

- **Monthly embedding variance:** `mean(variance across months) > 0.01`
  - Verifies temporal signal is present (not all zero like before)

- **Training stability:** Both losses converge (no divergence)
  - CE loss: Should decrease from ~0.5 to ~0.1
  - Temporal loss: Should decrease from ~5.0 to ~1.0
  - NT-Xent loss: Should stabilize around ~1.0

### Failure Mode Analysis

| Scenario | Diagnosis | Remedy |
|----------|-----------|--------|
| Temporal gate fails (Recall@1 < 0.60) | Insufficient temporal signal | Increase λ_temporal to 0.4 |
| Identity gate fails (Recall@1 < 0.70) | Temporal signal overwhelming identity | Decrease λ_temporal to 0.2, increase λ_nt_xent to 0.4 |
| Both gates fail | Fundamental design issue | Revisit window design (add 2-month pairs to positives) |
| Training diverges | Loss weights unbalanced | Reduce both λ values, or add gradient clipping |

## Implementation Phasing

### Phase 1: Core Temporal Loss (2 days)

**Tasks:**
1. Implement `fusion/temporal_loss.py` with `temporal_contrastive_loss()`
2. Add unit tests for positive pair construction, missing handling
3. Modify `fusion/train.py` to add temporal loss term (lambda args)
4. Smoke test: train on 10 participants for 1 epoch, verify no crashes

**Deliverables:**
- `fusion/temporal_loss.py` (new)
- `tests/fusion/test_temporal_loss.py` (new)
- `fusion/train.py` (modified)

**Success criteria:** All unit tests pass, smoke test converges

### Phase 2: Full Training Run (1 day)

**Tasks:**
1. Implement `fusion/temporal_data.py` to generate monthly cache
2. Run `uv run python -m fusion.temporal_data` to generate cache
3. Train full model: `uv run python -m fusion.train --temporal-weight=0.3 --temporal-data=models/temporal_embeddings_cache.pt`
4. Validate convergence: check loss curves, verify both losses decreasing

**Deliverables:**
- `fusion/temporal_data.py` (new)
- `models/temporal_embeddings_cache.pt` (new cache)
- `models/fusion_metalearner_temporal.pt` (new checkpoint)

**Success criteria:** Training converges in ~15 epochs, both losses stable

### Phase 3: H1 Validation (1 day)

**Tasks:**
1. Update `applications/temporal/generate_monthly_embeddings.py` to use new checkpoint
2. Run monthly embedding generation: `uv run python applications/temporal/generate_monthly_embeddings.py --fusion-model=models/fusion_metalearner_temporal.pt`
3. Extract features: `uv run python applications/temporal/extract_features.py`
4. Train drift detector: `uv run python applications/temporal/train_drift_detector.py`
5. Check success criteria (Recall@1 ≥ 0.60, MAE ≤ 1.5, Precision ≥ 0.60)

**Deliverables:**
- Validation report showing H1 gates passing

**Success criteria:** All 3 H1 gates pass

### Phase 4: Tuning & Iteration (1 day)

**Tasks:**
1. Run full evaluation suite: identity recall@1, archetype recovery, PersonaConfig R²
2. If gates not met, tune λ values:
   - Temporal gate fails → λ_temporal = 0.4
   - Identity gate fails → λ_temporal = 0.2, λ_nt_xent = 0.4
3. Ablation study: What if we remove NT-Xent entirely? (CE + Temporal only)
4. Final validation with best λ configuration

**Deliverables:**
- Final checkpoint with optimal λ values
- Evaluation report showing all gates passing

**Success criteria:** Both primary gates (temporal + identity) pass

### Total Effort

~5 days (2 + 1 + 1 + 1), assuming single developer working sequentially.

## Migration & Compatibility

### Checkpoint Compatibility

**New checkpoint:** `fusion_metalearner_temporal.pt`
- Architecture: Same `LateFusionMetaLearner` class (no structural changes)
- Weights: Different due to new training objective
- Compatibility: Drop-in replacement for existing checkpoint

**Inference:** No code changes needed. Same API:
```python
model = LateFusionMetaLearner()
model.load_state_dict(torch.load("fusion_metalearner_temporal.pt"))
embedding = model(modality_embeddings)  # [B, 128]
```

### Backward Compatibility

**Existing training:** Unchanged if temporal args not provided
```bash
# Old way still works:
uv run python -m fusion.train

# New way for temporal:
uv run python -m fusion.train --temporal-weight=0.3 --temporal-data=models/temporal_embeddings_cache.pt
```

**Evaluation scripts:** No changes needed, just use new checkpoint

### Rollback Plan

If temporal fusion fails validation:
1. Keep existing `fusion_metalearner.pt` as backup
2. Document failure in post-mortem
3. Fall back to Option C (separate temporal model)

## Alternatives Considered

### Alternative 1: Replace NT-Xent with Temporal

**Design:** `loss = CE + λ * Temporal` (no NT-Xent term)

**Pros:** Simpler loss function, stronger temporal signal

**Cons:** Loses identity stability (recall@1 would drop significantly from 82% to ~60-70%)

**Rejected:** User chose balanced multi-task (Option B) over temporal-primary

### Alternative 2: Sliding Window (k=3)

**Design:** Positive pairs = (t, t+1), (t, t+2), (t, t+3) with decaying weights

**Pros:** Captures multi-scale temporal dynamics

**Cons:** More complex, harder to tune, risk of collapsing all time-steps

**Rejected:** User chose simpler adjacent-months approach (Option A)

### Alternative 3: All-Pairs Within Participant

**Design:** All (month_i, month_j) for same participant are positive pairs

**Pros:** Maximizes temporal signal

**Cons:** Likely collapses identity (all time-steps become too similar), defeats purpose

**Rejected:** Too aggressive, would lose individual identity

## Dependencies

### Existing Prerequisites (Already Built)

All required for H1 validation (from H1 design spec):
- `PersonaConfig.month` field exists
- `persona_sampler.sample_temporal_trajectory()` generates 12 monthly snapshots
- `participant_configs.jsonl` carries `drift_label`/`drift_month` ground truth
- All modalities fielded monthly (trace/transaction/psychographic/clickstream/campaign)
- Monthly modality files: `{modality}_month_MM.jsonl` for M = 1..12

### New Dependencies

**Python packages:** None (uses existing PyTorch, numpy, etc.)

**Data files:**
- `models/temporal_embeddings_cache.pt` (generated by temporal_data.py)

**Scripts:**
- `fusion/temporal_data.py` (new)
- `fusion/temporal_loss.py` (new)
- Modified `fusion/train.py`

## Risks & Mitigations

### Risk 1: Identity Degrades Too Much (Recall@1 < 70%)

**Probability:** Medium (multi-objective tradeoff is real)

**Impact:** High (loses core fusion capability)

**Mitigation:**
- Start with conservative λ values (0.3, 0.3)
- Monitor identity recall@1 during training
- Early stopping if recall drops below 70%
- Fallback: reduce λ_temporal, increase λ_nt_xent

### Risk 2: Temporal Signal Still Too Weak (Recall@1 < 0.60)

**Probability:** Low (adjacent-month contrastive should work)

**Impact:** High (H1 fails again)

**Mitigation:**
- Verify monthly embeddings vary during Phase 2 (variance > 0.01)
- If variance too low, increase λ_temporal to 0.4 or 0.5
- Fallback: expand to sliding window (k=2 or k=3)

### Risk 3: Training Diverges

**Probability:** Low (three-term loss is standard multi-task)

**Impact:** Medium (wasted compute time)

**Mitigation:**
- Start with small λ values (0.3 each)
- Add gradient clipping if needed
- Monitor loss curves separately
- Fallback: reduce all λ values proportionally

### Risk 4: Regresses Other Capabilities

**Probability:** Medium (embedding space shifts affect all probes)

**Impact:** Medium (may break persona regression, geometry)

**Mitigation:**
- Run full evaluation suite in Phase 4
- If PersonaConfig R² < 0.70, reduce λ_temporal
- Document any capability changes in post-mortem

## Non-Goals (Explicitly Out of Scope)

- **Changing fusion architecture:** No structural changes to `LateFusionMetaLearner`, only training objective
- **Modality encoder retraining:** Encoders remain frozen, only fusion meta-learner retrains
- **Real-data integration:** Validation remains synthetic-only
- **Production deployment:** Prototype-only, no ops consideration

## References

- **H1 Temporal Dynamics Design:** `docs/superpowers/specs/2026-06-16-h1-temporal-dynamics-design.md`
- **H1 Post-Mortem:** `docs/post-mortems/h1-temporal-postmortem.md`
- **Fusion Architecture:** `.claude/context/fusion-architecture.md`
- **Current Training:** `fusion/train.py` (NT-Xent implementation lines 624-649, 685-837)
- **Temporal Contrastive Learning:** CPC (Contrastive Predictive Coding), Time-Contrastive Learning (TCL) literature

---

**Next Steps:**

1. Review this spec for clarity/completeness
2. If approved, invoke `writing-plans` skill to create implementation plan
3. Create beads for epic + child tasks
4. Execute per implementation plan
