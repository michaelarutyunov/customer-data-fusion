# Fusion Architecture

> Tier 3 context document — see `codified-context-principles.md` for governance.
> Authoritative spec: `fusion/SPEC.md`. This document explains intent and rationale.

---

## What the Fusion Layer Does

The fusion layer takes four independent per-participant embeddings (one per modality) and combines them into a single **CDT embedding** — the Consumer Digital Twin's compressed behavioural representation. It is the only place in the architecture where cross-modal information interacts.

Each modality encoder produces `[B, 128]`. The four are concatenated to `[B, 512]`, then a shallow MLP compresses back to `[B, 128]`. The classification head (`[B, 7]`) is a training proxy — the 128-dim hidden state is the actual output.

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
2. **Information compression from 512**: The MLP's job is to compress 4 × 128 correlated signals (all derived from the same 7-archetype generative process) into a representation that discards inter-modality redundancy and retains discriminative signal.
3. **`schemas.EMBEDDING_DIM = 128`**: This constant governs all encoder output dims. Keeping fusion output at the same dim means the fusion layer is a drop-in replacement for any single encoder in downstream evaluation code.

---

## Modality Dropout

Each modality is zeroed out independently with p=0.2 during training. This serves two purposes:

1. **Missing-data robustness**: Real CDT deployments may not have all four modalities for every consumer. A fusion layer trained without dropout would produce garbage embeddings when a modality is absent.
2. **Preventing co-adaptation**: Without dropout, the MLP can learn to rely entirely on the strongest modality (trace, 95% probe accuracy) and ignore the weaker ones. Dropout forces the model to maintain utility from every modality subset.

At inference, all four modalities are always active. If a modality is genuinely absent for a participant, zero out its slice — the model handles this because it trained on zeroed inputs.

---

## The Embedding Cache

Encoder forward passes are expensive relative to meta-learner training (especially the transformer trace encoder). The training script caches all four `[1001, 128]` embedding matrices to `models/fusion_embeddings_cache.pt` before training begins.

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

If fusion accuracy is lower than 85%, investigate ablation deltas to identify which modality is degrading the result. If all ablation deltas are near zero and accuracy is ≥85%, the modalities are encoding correlated information — this is a finding to report, not a failure (see Phase 2a fix post-mortem R7).

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
