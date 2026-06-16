# Fusion Architecture

> Tier 3 context document — see `codified-context-principles.md` for governance.
> Authoritative spec: `fusion/SPEC.md`. This document explains intent and rationale.
> Updated: schema-update epic (2026-06-13) — 6-modality fusion, learned MISSING embedding, variable n_modalities.
> Verified 2026-06-15 (bead `2io` + fixes `yy7`/`7t6`/`fkx`, full 6-modality run on dedup'd 1001-participant data): archetype recovery **90.0%** (>85% Tier-1 floor), dropout-view recall@1 **82.1%** (165× chance — improved over the 4-modality prototype's 70.4%), PersonaConfig R² **0.751 mean** (gate ≥0.70, accepted in `fkx` as the expected identity-vs-linearity tradeoff). The earlier 98%/0.66 figures were on a 2×-duplicated cache (bug `yy7`, fixed); the dedup'd numbers here are authoritative.

---

## What the Fusion Layer Does

The fusion layer takes independent per-participant embeddings (one per modality) and combines them into a single **CDT embedding** — the Consumer Digital Twin's compressed behavioural representation. It is the only place in the architecture where cross-modal information interacts.

Each modality encoder produces `[B, 128]`. With the schema-update epic, the modality count is **6** (trace, transaction, text, psychographic, clickstream, campaign), so the concatenation is `[B, 768]` → shallow MLP → `[B, 128]`. The classification head (`[B, 7]`) is a training proxy — the 128-dim hidden state is the actual output.

**Variable-modality loader (bead `hcx`, 2026-06-14).** `load_encoders(modalities=...)` and `generate_embeddings(...)` take a modality set (default: every key in `CHECKPOINT_PATHS`). `_MODALITIES` is derived as `[k for k in embeddings if k not in ("labels", "participant_ids")]` and guarded by two asserts — `n_modalities == len(encoders)` and stacked shape `[N, n_modalities, 128]`. The `"participant_ids"` exclusion is load-bearing: it is a list (not a tensor) and would corrupt `torch.stack`/`torch.cat` if it leaked into `_MODALITIES` (a latent bug the assert closes). `LateFusionMetaLearner` is instantiated with the *actual* `n_modalities`. `main()` drops `"text"` when `narratives.jsonl` is empty, yielding the 5-modality dry run. A 4-modality regression now fails loudly at the assert.

---

## Why Late Fusion

The architecture is deliberately **late fusion** (combine embeddings, not intermediate features):

1. **Modality independence**: Each encoder trains on its own objective (trace → strategy recovery, transaction → next brand tier, text → persona classification, psychographic → trait projection). Coupling them during training would contaminate objectives and complicate debugging.
2. **Swappability**: Any encoder can be retrained or replaced without touching the others. The fusion layer just re-caches embeddings and retrains.
3. **Interpretability**: Ablation tests (zero out one modality's embedding slice) directly measure each modality's contribution to the fused result.

Early fusion (fusing at the feature level, e.g. trace + transaction token streams) is a designed upgrade path for the trace + transaction pair specifically. Not implemented in v0.1.

---

## Why the CDT Embedding is 128-dim

Three reasons:

1. **Parity with individual encoder outputs**: Each modality encoder already produces 128-dim embeddings. Keeping the CDT embedding at 128-dim allows direct comparison — UMAP plots, cosine similarity, and probe accuracy can be computed identically on single-modality and fused embeddings.
2. **Information compression from 768**: The MLP's job is to compress 6 × 128 correlated signals (all derived from the same 7-archetype generative process) into a representation that discards inter-modality redundancy and retains discriminative signal.
3. **`schemas.EMBEDDING_DIM = 128`**: This constant governs all encoder output dims. Keeping fusion output at the same dim means the fusion layer is a drop-in replacement for any single encoder in downstream evaluation code.

---

## Modality Dropout + MISSING Embedding

Each modality is zeroed out independently with p=0.2 during training. This serves two purposes:

1. **Missing-data robustness**: Real CDT deployments may not have all modalities for every consumer. A fusion layer trained without dropout would produce garbage embeddings when a modality is absent.
2. **Preventing co-adaptation**: Without dropout, the MLP can learn to rely entirely on the strongest modality (trace, 95% probe accuracy) and ignore the weaker ones. Dropout forces the model to maintain utility from every modality subset.

**Learned MISSING embedding (schema-update epic):** For *natural* missingness — customers outside the trace coverage subset (250/1000 have traces) — the fusion layer now holds a trainable `missing_embedding` parameter (one `[per_modality_dim]` vector per slot). `apply_missing_mask()` replaces absent modality outputs with the MISSING vector before concatenation, rather than zero-filling. This is distinct from training-time dropout: dropout zeros random slots for augmentation; MISSING handles structurally absent modalities with a learned signal.

At inference, all modalities are active unless genuinely absent (partial coverage). The MISSING embedding handles that case.

---

## The Embedding Cache

Encoder forward passes are expensive relative to meta-learner training (especially the transformer trace encoder). The training script caches all `[1001, 128]` embedding matrices (one per loaded modality — 4, 5, or 6 depending on the run) to `models/fusion_embeddings_cache.pt` before training begins.

Cache invalidation: if any encoder checkpoint is newer than the cache file (mtime comparison), the cache is regenerated. This means retraining an encoder automatically triggers cache regeneration on the next fusion training run.

The cache is a deterministic function of the frozen encoder weights and the dataset — it is safe to reuse across fusion training runs as long as neither changes.

---

## Relationship to Individual Encoder Probes

Phase 2a encoder probe results (val split, 201 participants):

| Modality | Strategy recovery |
|---|---|
| Trace | 95.02% |
| Transaction | 62.59% |
| Text | 100% |
| Psychographic | 100% |
| **Fused target** | **>85%** |

Text and psychographic encoders are near-sufficient statistics for the latent `PersonaConfig` — 100% accuracy is expected given the generative design (psychographics are a near-direct transcription of config params; narratives are LLM realisations of config params). Fusion will not exceed 100%. Do not use "beats best single modality" as an acceptance criterion.

If fusion accuracy is lower than 85%, investigate ablation deltas to identify which modality is degrading the result. If all ablation deltas are near zero and accuracy is ≥85%, the modalities are encoding correlated information — this is a finding to report, not a failure (see `docs/post-mortems/phase2a-fix-postmortem.md` R7).

### Schema-update epic — encoder + fusion results (1001 participants, 2026-06-14)

Strategy recovery after retraining on the 1001-participant dataset (beads `syu`/`33x`/`fso` + stale-encoder retrain):

| Modality | Strategy recovery | Notes |
|---|---|---|
| Trace | ~0.51 | retrained on 1001 |
| Transaction | ~0.37 | retrained on 1001 |
| Psychographic | ~0.68 | retrained on 1001 |
| Text | — | not retrained this epic (narratives empty); checkpoint from prototype |
| Campaign | 0.71 | new encoder (bead `33x`) |
| Clickstream | 0.52 | new encoder (bead `syu`), after archetype-keying (`fso`); was 0.23 before |
| **Fused (6-modality, bead `2io`)** | **0.90** | dedup'd (`yy7`); above 85% Tier-1 floor |

**6-modality individual-identity + regression (bead `2io`, dedup'd, 201 val participants):**
- Dropout-view recall@1: **0.821** (165× chance) — **improved** over the 4-modality prototype's 0.704. Adding clickstream + campaign strengthened individual identity.
- PersonaConfig R² (fused): **0.751 mean** (price_sensitivity 0.79, brand_loyalty 0.75, involvement 0.79, maximiser 0.77, risk_tolerance 0.77, p_strategy_lapse 0.72, inspection_depth 0.68). Gate ≥0.70 (met). Psychographic-alone R² is higher for 5/7 params (0.84–0.86) — psychographics directly transcribe config params; fusion trades that linearity for individual identity.

**The identity-vs-linearity tradeoff (resolved in `fkx` — accepted):** giving the meta-learner more modalities improved the metrics it is directly optimised for (CE archetype, NT-Xent individual identity) but reduced how linearly the continuous PersonaConfig parameters are encoded in the CDT — for 5/7 params psychographic-alone reads the dials better. This is the *expected* behavior of an identity-optimized twin, not a defect. R² gate set to ≥0.70 (met at 0.751); recovering higher R² via an auxiliary regression head or lower `lambda_contrastive` was considered and rejected — it would trade away recall@1, the primary metric.

**Key finding — clickstream archetype signal:** clickstream's archetype signal is weak by *generator design* (transitions perturbed by within-archetype `config.latent`, not archetype-keyed). Bead `fso` added archetype-keyed session-intent priors, lifting raw-baseline recovery from 0.15 (chance) to 0.40 and encoder recovery 0.23 → 0.52. The 0.60 bead target was an encoder-capacity question, not a data one. Clickstream's real fusion value is individual identity (its NT-Xent term decreases), tested via recall@1 in the full 6-modality run (`2io`). See `docs/post-mortems/schema-update-postmortem.md`.

## Multi-Task Training Objective (v0.2 — NT-Xent + CE)

As of bead 0if (2026-06-09), the fusion training objective is multi-task:

```
total_loss = CE_loss + lambda_contrastive * NT_Xent_loss
```

**CE auxiliary head** — retains archetype separability (Tier 1 gate). The `forward_with_embedding()` method returns `(logits, cdt_embedding)`. CE is computed on `logits`.

**NT-Xent contrastive loss** — teaches the meta-learner that two modality-dropout-augmented views of the same participant should map to the same CDT embedding neighbourhood, regardless of which modalities are present.

Positive pairs: two forward passes of the same participant through the MLP, each with independent per-modality dropout (p=0.2). This reuses the existing modality-dropout mechanism — no new data structure needed. Other participants in the batch are negatives (SimCLR-style, temperature=0.07).

**Why modality-dropout as the augmentation:** the meta-learner is trained for robustness to missing modalities. Two dropout views of the same participant are a natural positive pair — they're the same person seen through different data availability. The NT-Xent forces the CDT embedding to be stable under this variation, which is exactly what individual-level identity requires.

### Results (val split, 210 participants)
- recall@1 (dropout-view CDT retrieval): **70.4%** (criterion >0.1)
- recall@10: **88.5%**
- Tier 1 archetype recovery: **100%**
- W/B variance ratio: **0.94** (within-archetype variance ≈ between-archetype — CDT is individual-discriminative, not archetype-clustered)

### Key parameter
`lambda_contrastive=0.5` (default). This produces roughly equal CE and NT-Xent gradient contributions at epoch 1, converging to NT-Xent ≈ 1.4 and CE ≈ 0.3 by epoch 15.

---

## Temporal Limitations

**Critical constraint:** The fusion meta-learner is trained for **identity stability**, not temporal sensitivity.

### What the Model Captures

**Identity (what it's trained for):**
- "Who is this person?" — collapses within-participant variance
- Same participant across months/sessions → similar embeddings
- Robust to missing modalities (dropout augmentation)

**Not temporality (what it's NOT trained for):**
- "How is this person changing?" — would require preserving within-participant variance
- Same participant at different times → different embeddings (does not happen)

### Why Embeddings Don't Vary Over Time

The NT-Xent loss explicitly teaches the model to ignore within-participant variance:

```python
# From fusion/train.py
def nt_xent_fusion(emb_v1, emb_v2):
    """
    emb_v1[i] and emb_v2[i] are two dropout-augmented views of participant i.
    Pushes them closer together → identity stability.
    """
```

When you pass month 1, month 2, ..., month 12 data through frozen encoders:
1. Encoders are frozen — they can't adapt to temporal variations
2. Fusion was trained to collapse variance — treats month-to-month changes as noise
3. Result: identical embeddings across all months (variance = 0.0)

### Evidence from H1 Validation Failed (2026-06-16)

H1 Temporal Dynamics attempted to detect regime shifts using monthly frozen embeddings. All 1002 participants produced identical embeddings across 12 months, making drift detection impossible.

- Drift features: all 0.0 (dist_mean, dist_std, dist_max, dist_slope)
- Stage 1 Recall: 0.000 (target ≥0.80)
- Stage 1 Precision: 0.000 (target ≥0.60)

**Root cause:** NT-Xent optimizes for identity, not temporality. See `docs/post-mortems/h1-temporal-postmortem.md` for full analysis.

### Implications for Temporal Capabilities

**What won't work with frozen fusion:**
- ❌ Regime shift detection from monthly embeddings (H1)
- ❌ Churn prediction from embedding trajectories
- ❌ "How is this customer changing?" queries

**What does work:**
- ✅ "Who is this customer?" (individual identification)
- ✅ "What archetype is this?" (classification)
- ✅ "Which customers are similar?" (retrieval)

### If You Need Temporal Capabilities

Two options:

**Option 1: Retrain fusion with temporal objective**
- Replace NT-Xent with temporal contrastive loss
- Positive pairs: (participant_i, month_t) with (participant_i, month_{t+1})
- Preserves temporal variance while maintaining identity signal
- Requires full retraining (affects all downstream dependencies)

**Option 2: Separate temporal model**
- Keep frozen embeddings as static features
- Train GRU/Transformer on embedding sequences
- Faster to implement, modular
- Less elegant than fixing fusion at the source

### Design Principle

**Frozen models preserve what they were trained to capture.**

Before using a frozen model for a new task, verify that its training objective is compatible with the task's requirements. If the model optimizes for property A (identity) but the task requires property B (temporality), you have a fundamental mismatch.

---

## Evaluation Beyond Archetype Recovery

The NT-Xent training objective explicitly optimises cross-view individual retrieval. The correct evaluation metric is **dropout-view CDT retrieval recall@1**: given two random modality-dropout views of the same participant, what fraction of the time does the correct participant rank #1 in a gallery of N participants?

- N=210 val participants, random chance = 0.005 (1/210)
- Post-bead-0if result: **70.4%** (140× over chance)

**Important:** the existing `evaluation/retrieval.py` computes CDT-vs-single-modality-encoder retrieval. This tests alignment between two completely different representation spaces (meta-learner output vs. individual encoder output) — spaces that were never trained to align. That metric will remain near-zero regardless of NT-Xent and is not a useful diagnostic for individual identity.

Three Phase 2b evaluations probe the CDT embedding more broadly:

1. **Dropout-view retrieval**: recall@1 using two dropout-augmented CDT views (primary individual-identity diagnostic — see above)
2. **PersonaConfig regression probe** (`evaluation/config_probe.py`): does the CDT embedding recover continuous latent parameters (price_sensitivity, inspection_depth, etc.)?
3. **Within-persona UMAP** (`evaluation/geometry.py`): does colouring the UMAP by a continuous param show a gradient within archetype clusters?

See `.claude/context/prd-validation.md` for quantitative results on all criteria.

---

## Counterfactual Evaluation Baseline

**Defined by bead c11 (2026-06-09).**

The baseline CDT embedding for each participant is computed from the **original** (pre-override) generator run using the frozen fusion model. Counterfactual shift is measured as cosine distance between this baseline and the re-generated embedding.

### Threshold

A counterfactual shift is considered **meaningful** if `cosine_distance_shift >= 0.27`.

**Rationale:** Intra-archetype pairwise cosine distance across all 7 archetypes (150 participants each):

| Metric | Value |
|--------|-------|
| Overall mean intra-archetype cosine distance | 0.3997 |
| Overall SD | 0.1332 |
| **Threshold (2×SD, rounded up)** | **0.27** |

Per-archetype breakdown:

| Archetype | n | Mean | SD |
|-----------|---|------|-----|
| price_lex | 150 | 0.3223 | 0.1020 |
| quality_lex | 150 | 0.3526 | 0.1075 |
| compensatory | 150 | 0.4266 | 0.1248 |
| satisficer | 150 | 0.4730 | 0.1352 |
| brand_affect | 150 | 0.3435 | 0.1091 |
| adaptive | 150 | 0.4507 | 0.1389 |
| low_involve | 150 | 0.4290 | 0.1278 |

The original threshold of 0.1 was rejected — it is below the within-archetype noise floor (2×SD = 0.27). A shift of 0.1 would be indistinguishable from natural variation between two participants of the same archetype.

**Note on seed confound:** Option B counterfactuals re-run the generator with `n=1`, producing a new participant ID (`{archetype}_0000`) with a different random seed than the original participant. The counterfactual embedding therefore reflects both the PersonaConfig override and a different noise realization. This is acceptable for the prototype — the test is whether the CDT embedding *responds* to parameter changes, not the precise magnitude of that response.
