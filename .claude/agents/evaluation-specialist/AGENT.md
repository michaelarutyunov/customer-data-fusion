# evaluation-specialist

## Role
Owns all code in `evaluation/` — strategy recovery, ablation, geometry, cross-modal retrieval, and PersonaConfig regression probes.

## Trigger Conditions
- Any edit to files in `evaluation/`
- Any task involving model evaluation, embedding geometry, ablation, or probe analysis

## Evaluation Framework

Phase 2b evaluation operates at two distinct levels. Keep them separate — they answer different questions.

### Tier 1 — Archetype Recovery (pass/fail gate)

| File | Task | Key metric | Gate |
|---|---|---|---|
| `strategy_recovery.py` | Classify archetype from fused CDT embedding | Top-1 accuracy on val split (201 participants) | >85% |
| `ablation.py` | Leave-one-out per-modality ablation | Accuracy delta when each modality's 128-dim slice is zeroed | Diagnostic only |

**Ablation procedure**: zero out one modality's 128-dim slice of the 512-dim fusion input (not drop the encoder — zero the slice). 4 tests total. A delta <5% is a reportable finding, not a failure. Text and psychographic encoders each reach 100% individually; low ablation deltas for correlated modalities are expected.

**Single-modality baselines for comparison table** (Phase 2a results):
- Trace: 95.02%
- Transaction: 62.59%
- Text: 100%
- Psychographic: 100%

Fused accuracy may not exceed 100% — do not use "beats best single modality" as a pass criterion.

### Tier 2 — CDT Embedding Quality (diagnostic — no pass/fail gates)

These probe whether the CDT embedding captures participant-level behavioural structure beyond the archetype label. All operate on the full dataset (1050 participants, 150 per archetype) using the embedding cache.

| File | Task | Key metric |
|---|---|---|
| `geometry.py` | UMAP of CDT embeddings [N, 128] | Silhouette score; within-persona gradient (see below) |
| `retrieval.py` | Cross-modal nearest-neighbour retrieval | recall@1, recall@10, within-archetype recall@1 |
| `config_probe.py` | Ridge regression for 7 PersonaConfig float params | R² per param per modality (7 × 5 matrix) |

**Primary individual-identity metric:** Dropout-view CDT retrieval recall@1 = 70.4% (140× over random chance, bead 0if). This is the correct metric for the NT-Xent-trained fusion model. See `.claude/context/prd-validation.md` for details.

**geometry.py** produces two UMAP views: (a) coloured by archetype — tests between-persona separation; (b) coloured by a continuous PersonaConfig param (e.g. `price_sensitivity`) within each cluster — tests within-persona variation. If (b) shows a gradient inside clusters, the CDT embedding preserves individual deviation. If flat, it has collapsed within-persona variation to the archetype label. Saves UMAP coordinates + all 7 PersonaConfig float params to `umap_fused.json` in `data/synthetic/` (written when `geometry.py` runs).

**retrieval.py** runs two evaluations:
- CDT-vs-single: fused CDT embedding as query → find nearest neighbour in each single-modality embedding space. 4 tests.
- Single-vs-single (baseline): 6 modality-pair tests.

> **⚠️ Important:** CDT-vs-encoder retrieval measures alignment between two *different representation spaces* never trained to align. Those recall@1 values will remain near-zero regardless of NT-Xent. The correct retrieval metric for individual identity is **dropout-view CDT recall@1** (computed during fusion training, see `.claude/context/prd-validation.md`).

**config_probe.py** trains `sklearn.Ridge(alpha=1.0)` on each of 5 embedding sets (fused, trace, transaction, text, psychographic) to predict each of 7 PersonaConfig float params. Output dict: `{param_name: {modality_name: r2_val}}`. A fused R² higher than all four individual modalities is the clearest evidence fusion combines complementary information. Negative R² is valid output for a poor fit — not an error.

## Data Dependencies

| File | Source |
|---|---|
| Embeddings | `models/fusion_embeddings_cache.pt` — written by `fusion/train.py` |
| PersonaConfig floats | `data/synthetic/participant_configs.jsonl` — written by generator pipeline to `output_dir` |
| Encoder checkpoints | `schemas.CHECKPOINT_PATHS` — never hardcode paths |

`participant_configs.jsonl` schema: one JSON record per participant with `participant_id` + 7 float fields: `price_sensitivity`, `brand_loyalty`, `inspection_depth`, `maximiser_score`, `involvement_score`, `risk_tolerance`, `p_strategy_lapse`. Note: `inspection_depth` is stored as a float (converted from `InspectionDepth` enum via `_inspection_depth_to_float()`).

## Key Constraints

- Evaluation never modifies model weights — read-only access to trained encoders and cached embeddings
- Ground truth archetype label: `schemas.PERSONA_TO_IDX` — use this for all integer label lookups
- Train/val split: always `split_participants(seed=42)` — same as encoder probes and fusion training
- Ablation zeros slices of the concatenated 512-dim input, it does not re-run forward passes with missing modalities
- Log all metrics to MLflow with appropriate tags (`stage=strategy_recovery`, `stage=ablation`, etc.)
- All output dicts must be serialisable (no Tensor values) — convert to Python floats before returning

## Context Documents

- `fusion/SPEC.md` — evaluation metrics spec and interpretation guidance (read first)
- `.claude/context/fusion-architecture.md` — why the architecture is designed the way it is
- `.claude/context/project-vision.md` — can/cannot-claim framing; what the evaluations are trying to prove
- `.claude/context/data-contracts.md` — schema field specifications
