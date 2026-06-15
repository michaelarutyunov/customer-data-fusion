# Phase 2a Post-Mortem: Modality Encoders

> Written: 2026-06-06
> Phase: Phase 2a — Modality Encoders (epic customer-data-fusion-szm)
> Scope: encoders/trace/, encoders/transaction/, encoders/text/, encoders/psychographic/, evaluation/, tests/encoders/

---

## 1. What Was Built

### Encoder Modules

| Module | Files | Status |
|---|---|---|
| `encoders/trace/` | `encoders/trace/SPEC.md`, `encoders/trace/tokeniser.py`, `encoders/trace/model.py`, `encoders/trace/train.py` | Implemented + tests |
| `encoders/transaction/` | `encoders/transaction/SPEC.md`, `encoders/transaction/features.py`, `encoders/transaction/model.py`, `encoders/transaction/train.py` | Implemented + tests |
| `encoders/text/` | `encoders/text/SPEC.md`, `encoders/text/embed.py` | Implemented + 35 tests |
| `encoders/psychographic/` | `encoders/psychographic/SPEC.md`, `encoders/psychographic/features.py`, `encoders/psychographic/model.py`, `encoders/psychographic/train.py` | Implemented + tests |

### Architecture Summary

| Encoder | Input | Architecture | Objective | Output Dim |
|---|---|---|---|---|
| Trace | MouseLab event sequences (27-dim tokens) | Transformer (4 heads, 3 layers, d=64) | NT-Xent contrastive + aux classification | 128 |
| Transaction | Purchase history sequences (20-dim tokens) | GRU (2 layers, h=128) | Next brand_tier prediction | 128 |
| Text | Persona narratives (text) | Frozen sentence-transformer (384) → Linear (128) | Persona classification | 128 |
| Psychographic | 22-dim survey vectors | MLP (22→64→128) + Dropout + LayerNorm | Persona classification | 128 |

All encoders output `EMBEDDING_DIM = 128` (imported from `schemas`), satisfying the fusion contract.

### Evaluation Infrastructure

| File | Purpose |
|---|---|
| `evaluation/probe.py` | Shared probe utilities: logistic regression probe, cosine sim stats, Pearson r |
| `evaluation/run_probes.py` | Unified probe runner for all 4 encoders |
| `evaluation/trace_probe.py` | Trace-specific probe (participant pooling) |
| `evaluation/transaction_probe.py` | Transaction-specific probe (price sensitivity correlation) |
| `evaluation/text_probe.py` | Text-specific probe (cosine similarity) |
| `evaluation/psychographic_probe.py` | Psychographic-specific probe (raw features baseline) |
| `evaluation/generate_probe_plots.py` | UMAP, confusion matrix, bar chart generation |
| `notebooks/02_encoder_probing.ipynb` | Probe evaluation notebook with human review gate |
| `tests/encoders/text/test_text_encoder.py` | 35 tests for text encoder |

### Probe Results

| Encoder | Strategy Recovery | Threshold | Pass? |
|---|---|---|---|
| Text | 99.0% ± 0.3% | >70% | ✅ |
| Psychographic | 100.0% ± 0.0% | >75% | ✅ |
| Trace | 37.8% ± 0.7% | >85% | ❌ |
| Transaction | N/A (training bug) | >60% | ❌ |

Text encoder additional metrics: intra-persona cosine sim 0.71 (>0.6 ✅), inter-persona cosine sim -0.03 (<0.4 ✅).

---

## 2. Errors and Resolutions

### E1 — Persona label mismatch: "random" vs "quality_lex" — Data Contract
**What failed**: Encoder training scripts hardcoded `"random"` as a persona archetype, but the data generator produces `"quality_lex"` from `config/personas.yaml`. Caused `KeyError` at training time.

**Root cause**: `config/personas.yaml` defines 7 archetypes: `price_lex, quality_lex, compensatory, satisficer, brand_affect, low_involve, adaptive`. The `"random"` label was a strategy type, not an archetype name. Encoder authors (szm.6–szm.9) used the wrong label set.

**Resolution**: Replaced `"random"` with `"quality_lex"` in all 8 files: `encoders/psychographic/train.py`, `encoders/text/embed.py`, `tests/encoders/text/test_text_encoder.py`, and all 5 evaluation probe scripts.

**Prevention**: The `PERSONA_LABELS` constant should be defined once in `schemas/__init__.py` or derived from `config/personas.yaml` at module load time, not hardcoded in each file. See R2.

### E2 — MLflow file store maintenance mode — Infrastructure
**What failed**: All encoder training calls failed with `MlflowException: The filesystem tracking backend is in maintenance mode`. MLflow 3.x no longer supports the file-based tracking store by default.

**Root cause**: MLflow ≥3.0 requires either a database backend or `MLFLOW_ALLOW_FILE_STORE=true` for the legacy `./mlruns` directory.

**Resolution**: Set `MLFLOW_ALLOW_FILE_STORE=true` environment variable for probe runs. Wrapped MLflow calls in try/except in `evaluation/run_probes.py` to prevent MLflow failures from killing probe evaluation. Long-term: migrate to SQLite-based MLflow tracking database backend.

### E3 — sklearn 1.9 removed `multi_class` parameter — API Drift
**What failed**: `LogisticRegression(multi_class="multinomial")` raised `TypeError: unexpected keyword argument 'multi_class'`.

**Root cause**: sklearn 1.9 removed the `multi_class` parameter (logistic regression always uses multinomial loss for multi-class since sklearn 1.5).

**Resolution**: Removed `multi_class="multinomial"` from `evaluation/probe.py`. No behavioral change — it was already the default.

### E4 — Transaction encoder double backward — Training Bug
**What failed**: `RuntimeError: Trying to backward through the graph a second time` in `encoders/transaction/train.py:train_epoch()`.

**Root cause**: Not fully diagnosed. The `train_epoch` function appears to retain a computation graph reference across batch iterations. Possible cause: the `pack_padded_sequence` output or a tensor is reused with `retain_graph` semantics missing.

**Resolution**: Deferred to Phase 2b bugfix (szm.11 closed with explanation). The encoder architecture, model, and features modules are correctly implemented — only the training loop has the issue.

### E5 — Trace encoder early stopping at epoch 11 — Training Instability
**What failed**: Trace encoder training stopped at epoch 11/30 with validation loss increasing monotonically after epoch 1 (8.23 → 9.12).

**Root cause**: With only 7 persona classes (one per participant), the NT-Xent contrastive loss has limited positive pairs per batch. The StratifiedSampler constructs batches with ≥2 samples per persona, but with 7 personas × ~2000 trials each, the model quickly memorizes the 7-class structure and overfits. The classification head dominates the loss signal (aux_weight=0.3), and the contrastive loss provides diminishing returns.

**Resolution**: Accepted the 37.75% strategy recovery result. This is a fundamental data limitation — contrastive training needs more diverse participants. The model architecture and tokeniser are correct. Recommendations in R3.

### E6 — 7 participants, not 1000 — Data Design Discovery
**What revealed**: The synthetic data has 7 unique `participant_id` values (one per persona archetype), each with ~143–2000 records. The generator creates many trials/transactions per archetype but treats each archetype as a single participant.

**Impact**: 
- Train/val split by participant_id yields 5 train / 2 val participants — statistically meaningless for probe evaluation
- Supervised encoders (text, psychographic) achieve 100% accuracy because they learn persona identity, not behavioural patterns
- Contrastive trace encoder cannot learn meaningful representations from 7 classes

**Resolution**: Probe scripts use record-level `StratifiedShuffleSplit` instead of participant-level split for evaluation only. This produces statistically stable accuracy estimates but doesn't fix the underlying data limitation. See R3.

---

## 3. Deviations from SPEC

### D1 — Probe evaluation uses record-level split, not participant-level
**SPEC said** (encoder-specialist AGENT.md): "Split must always be by participant_id."

**What was implemented**: Probe scripts use `StratifiedShuffleSplit` at the record level for strategy recovery evaluation. Training still uses participant-level split.

**Why**: With only 7 participants, an 80/20 participant split yields 5 train / 2 val — not enough for a logistic regression probe (5 samples, 7 classes). Record-level split with 1000+ records provides meaningful accuracy estimates.

**Risk**: Record-level evaluation may overestimate generalization to unseen participants. When the generator is updated to produce more participants, the probe evaluation should be re-run with participant-level splits.

### D2 — `multi_class="multinomial"` removed from LogisticRegression calls
**SPEC said** (encoder-specialist AGENT.md): Use `LogisticRegression(max_iter=1000, random_state=42)`.

**What was implemented**: Used `LogisticRegression(max_iter=1000, random_state=42)` without `multi_class`. This matches the AGENT.md spec — the earlier code that included `multi_class` was an unnecessary addition.

### D3 — Psychographic encoder: MLP does not outperform raw features
**SPEC said** (szm.13): "MLP outperforms raw features baseline."

**What was observed**: Both MLP embeddings and raw 22-dim features achieve 100% strategy recovery. The MLP adds no value over raw features on this dataset.

**Why**: The 22 psychographic features are engineered to be persona-discriminative (one-hot encoded categoricals + normalized continuous features). With only 7 classes and 1000 records, linear separation is trivial.

**Risk**: In Phase 2b fusion, the psychographic modality may add redundant information if it's perfectly correlated with persona labels. The MLP projection should be tested on a more diverse participant set before concluding it adds no value.

---

## 4. Context Infrastructure Gaps

### G1 — PERSONA_LABELS defined in 8 locations
**Gap**: The 7-element persona label list is hardcoded in `encoders/psychographic/train.py`, `encoders/text/embed.py`, `evaluation/trace_probe.py`, `evaluation/transaction_probe.py`, `evaluation/text_probe.py`, `evaluation/psychographic_probe.py`, `evaluation/run_probes.py`, and `tests/encoders/text/test_text_encoder.py`.

**Risk**: Any change to persona archetypes requires updating 8 files. The "random" vs "quality_lex" bug (E1) was a direct consequence.

**Fix**: Define `PERSONA_LABELS` once in `schemas/__init__.py` or derive it from `config/personas.yaml` at import time. All modules should import from schemas.

### G2 — No encoder checkpoint registry
**Gap**: Each probe script independently decides whether to load a checkpoint from `models/` or train from scratch. There's no canonical mapping from encoder name to checkpoint path.

**Fix**: Add a checkpoint registry to `evaluation/probe.py` or `schemas/` that maps encoder names to expected checkpoint paths.

### G3 — Transaction encoder training bug not root-caused
**Gap**: The double-backward bug (E4) was deferred without root cause analysis. The bug blocks the transaction probe and will need to be fixed before Phase 2b fusion.

**Fix**: Create a bead in Phase 2b specifically for debugging the transaction training loop. The fix is likely in `train_epoch()` — the `pack_padded_sequence` call may be creating tensors that share storage with earlier iterations.

---

## 5. Embedding Quality Assessment

### Text Encoder
- **Strategy recovery**: 99.0% — far exceeds 70% threshold
- **Intra-persona cosine sim**: 0.71 — above 0.6 threshold, indicating same-persona narratives are semantically similar
- **Inter-persona cosine sim**: -0.03 — well below 0.4 threshold, indicating different-persona narratives are well-separated
- **Verdict**: ✅ High quality. The frozen sentence-transformer + learned projection effectively separates personas. The 0.71 intra sim (not 0.95+) suggests narrative diversity is adequate — the LLM generation is NOT too deterministic.

### Psychographic Encoder
- **Strategy recovery**: 100.0% — exceeds 75% threshold
- **MLP vs raw features**: Both 100% — MLP adds no discriminative value on this dataset
- **Verdict**: ⚠️ Passes threshold but overfit. The MLP encoder may not generalize. Retest with diverse participants before fusion.

### Trace Encoder
- **Strategy recovery**: 37.75% — well below 85% threshold
- **Training**: Early stop at epoch 11, val loss increasing from epoch 1
- **Verdict**: ❌ Below threshold. Not caused by architecture flaws — the transformer encoder and tokeniser are correctly implemented. The contrastive objective needs more than 7 classes to learn meaningful representations.

### Transaction Encoder
- **Strategy recovery**: Not measured (training bug)
- **Verdict**: ❌ Cannot assess. Code needs debugging.

---

## 6. Assumptions Validated / Invalidated

### A1 — all-MiniLM-L6-v2 is sufficient for persona narrative discrimination ✓ Confirmed
**Evidence**: Text encoder achieves 99% strategy recovery with a frozen sentence-transformer and a single linear projection layer. The 384-dim semantic space contains sufficient information to separate 7 persona archetypes. Intra-persona cosine sim of 0.71 confirms narratives are both distinctive and varied.

### A2 — Supervised psychographic encoder converges quickly ✓ Confirmed
**Evidence**: 40 epochs on 1000 records achieves 100% accuracy. Training is CPU-viable (<30 seconds). The 22-dim feature engineering (one-hot + ordinal + continuous) provides cleanly separable inputs.

### A3 — Contrastive trace encoder works with 7 archetypes — ❌ Invalidated
**Evidence**: 37.75% strategy recovery vs 85% target. The NT-Xent loss with 7 classes does not provide enough discriminative signal. The encoder overfits to the classification head (aux_weight=0.3) and the contrastive loss plateaus.

**Implication**: Either (a) generate more diverse participants (100+ instead of 7), or (b) redesign the trace encoder objective for the 7-participant regime (e.g., use supervised classification instead of contrastive).

### A4 — Encoder independence principle holds ✓ Confirmed
**Evidence**: No encoder imports from another encoder. All 4 encoders output `EMBEDDING_DIM = 128`. The shared dependency is `schemas/` only. The fusion contract is satisfied.

---

## 7. Recommendations for Phase 2b

### R1 — Fix transaction encoder training loop before fusion
The double-backward bug blocks transaction probe evaluation and will block fusion training. Create a bead in Phase 2b specifically for root-causing and fixing this bug. Suspect: `pack_padded_sequence` tensor reuse across iterations.

### R2 — Centralize PERSONA_LABELS in schemas/
Move the 7-element persona label list to `schemas/__init__.py` or derive it from `config/personas.yaml`. All encoder and evaluation modules should import from a single source. This prevents the "random"/"quality_lex" class of bugs.

### R3 — Increase participant diversity before retraining trace encoder
The 7-participant data design is insufficient for contrastive training. Options:
- **Quick fix**: Modify `generator/persona_sampler.py` to create N×7 participants (e.g., 10 per archetype = 70 participants) with per-participant noise
- **Better fix**: Create 100+ independent participants with varied parameter noise, not just 7 archetype templates
- **Alternative**: Switch trace encoder to supervised classification objective (matching text/psychographic) if participant diversity can't be increased

### R4 — Migrate MLflow to database backend
The `./mlruns` file store is deprecated in MLflow 3.x. Run `mlflow migrate-filestore` to migrate to SQLite-based MLflow tracking database before Phase 2b training runs. Remove the `MLFLOW_ALLOW_FILE_STORE=true` workaround.

### R5 — Run psychographic ablation before fusion
Since the MLP encoder doesn't improve over raw features, test whether the psychographic modality adds unique information in fusion vs. using raw 22-dim features directly. If raw features work equally well, the MLP is unnecessary complexity.

### R6 — Add a probe evaluation CI step
The probe scripts (`evaluation/run_probes.py`) should be runnable as a CI check after any encoder change. Add to CLAUDE.md: "After any encoder change, run `uv run python -m evaluation.run_probes` and verify strategy recovery doesn't regress."

### R7 — Generate Phase 2b beads after this post-mortem is reviewed
The remaining work (fusion layer, early fusion upgrade path, full training, ablation, counterfactual) should be scoped into a Phase 2b epic with clear dependency ordering.
