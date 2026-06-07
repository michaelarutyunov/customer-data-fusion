# Fusion Layer Specification

## Current Version: 0.1

## Purpose

Combine four independent modality embeddings into a single consumer behavioural embedding (CDT embedding). The fusion layer is a late-fusion meta-learner: it receives frozen encoder outputs and learns to weight and combine them. Encoders are never updated during fusion training.

## Input Contract

Each modality encoder produces a per-participant embedding of shape `[batch_size, 128]`. The fusion layer receives all four:

| Modality | Encoder checkpoint | Embedding shape |
|---|---|---|
| Trace | `models/trace_encoder.pt` | `[B, 128]` |
| Transaction | `models/transaction_encoder.pt` | `[B, 128]` |
| Text | `models/text_encoder.pt` | `[B, 128]` |
| Psychographic | `models/psychographic_encoder.pt` | `[B, 128]` |

Checkpoint paths are canonical — always load from `schemas.CHECKPOINT_PATHS`. Never hardcode.

The four embeddings are concatenated to form the fusion input:

```python
fusion_input = torch.cat([trace_emb, tx_emb, text_emb, psycho_emb], dim=-1)
# shape: [B, 512]
```

All encoders must be frozen (`requires_grad=False`) during fusion training. Only the meta-learner parameters are updated.

## Architecture

### Phase 1 — Logistic Regression Baseline

A single linear layer from 512 → 7 (number of persona archetypes). No hidden layers, no activation. Establishes a ceiling for linear separability of the concatenated embedding space.

```
[B, 512] → Linear(512, 7) → [B, 7]
```

### Phase 2 — Shallow MLP Meta-learner (default for v0.1)

```
[B, 512]
    ↓
Linear(512, 256) → LayerNorm(256) → GELU → Dropout(0.2)
    ↓
Linear(256, 128) → LayerNorm(128) → GELU → Dropout(0.1)
    ↓
Linear(128, 7)   # classification head — removed at inference for embedding output
    ↓
[B, 7] logits
```

The 128-dim output of the second hidden layer is the **CDT embedding** — the compressed consumer representation used downstream. The classification head is detached at inference time.

### Phase 3 — Attention-based Combination (upgrade path, not in v0.1)

Replace concatenation + MLP with a cross-modal attention block that learns inter-modality alignment. Deferred — Phase 1/2 establishes whether the simpler architecture is sufficient.

## Output Contract

| Mode | Output | Shape | Use |
|---|---|---|---|
| Training | Logits | `[B, 7]` | Cross-entropy loss |
| Inference (embedding) | CDT embedding | `[B, 128]` | Downstream tasks, visualisation |
| Inference (classification) | Predicted persona | `[B]` int labels | Evaluation only |

The CDT embedding is the 128-dim hidden state before the final classification head. It is the primary deliverable of the fusion layer — the logits are a training proxy.

## Modality Dropout

During training, each modality's embedding is independently zeroed out with probability `p_dropout = 0.2` per sample. This trains the meta-learner to produce useful outputs even when one or more modalities are absent.

```python
embs = [trace_emb, tx_emb, text_emb, psycho_emb]  # each [B, 128]
if training:
    embs = [
        emb * (torch.rand(emb.shape[0], 1, device=emb.device) >= p_dropout).float()
        for emb in embs
    ]
fusion_input = torch.cat(embs, dim=-1)  # [B, 512]
```

Each column of the Bernoulli mask is drawn independently — different samples in the same batch can have different modalities active. During evaluation, all four modalities are always active (no dropout).

## Training Objective

Supervised classification: 7 persona archetypes (same label set as individual encoder probes).

```
Loss = CrossEntropyLoss(logits, persona_idx)
```

Labels from `schemas.PERSONA_TO_IDX`. 80/20 train/val split using `split_participants(seed=42)` — the same function and seed used by all encoder probes. This guarantees the val set is identical across all probes and fusion, making single-encoder vs fused accuracy comparisons valid.

Optimiser: Adam, lr=1e-3, weight_decay=1e-4. Scheduler: ReduceLROnPlateau (patience=5, factor=0.5). Early stopping: patience=10 epochs on val accuracy. Max epochs: 100.

## Data Loading

The fusion training script must:
1. Load all four encoder checkpoints (frozen).
2. For each participant, run all four encoders forward to produce embeddings.
3. Cache embeddings to disk — never re-run encoders during training epochs.
4. Train the meta-learner on cached embeddings only.

Embedding cache path: `models/fusion_embeddings_cache.pt`. Format: a dict with keys `"trace"`, `"transaction"`, `"text"`, `"psychographic"` each mapping to a `[N, 128]` tensor, plus `"labels"` as a `[N]` int tensor and `"participant_ids"` as a list of N strings (canonical ordering).

**Alignment invariant**: all five tensors/lists must share the same participant ordering. Row *i* across all four embedding tensors must correspond to the same participant. The cache build script establishes the canonical ordering from `psychographics.jsonl` (the only modality without duplicates), then indexes all other modalities by `participant_id` lookup. For the text modality, if a participant has more than one narrative (e.g. duplicate LLM outputs), keep only the first record by file order.

Cache is invalidated if any encoder checkpoint is newer than the cache file (check mtime). Training script must detect and regenerate automatically.

## Evaluation Metrics

All metrics computed on the val split (201 participants) unless stated otherwise.

### Archetype recovery (pass/fail gate)

| Metric | Definition | Threshold |
|---|---|---|
| Strategy recovery accuracy | Top-1 accuracy of persona classification | >85% — only hard gate |
| Per-modality ablation delta | Accuracy drop when one modality is zeroed out | Diagnostic — no pass/fail |
| Modality importance weights | Mean absolute output change per modality zeroing test | Diagnostic — no threshold |

**Ablation procedure**: for each modality, zero out its 128-dim slice of the 512-dim input and re-evaluate accuracy on the val split. Report the delta from the full-modality baseline. A delta < 5% is a finding worth investigating, not a failure — text and psychographic encoders both achieve 100% individual probe accuracy and may encode correlated information, making low deltas an expected and reportable result (see Phase 2a fix post-mortem R7). The only hard gate is overall strategy recovery >85%.

Note: text and psychographic encoders are near-sufficient statistics for the latent `PersonaConfig` — 100% archetype accuracy from a single modality is expected and does not imply fusion has nothing to add. The three evaluations below probe what fusion adds beyond archetype recovery.

### CDT embedding quality (diagnostic — no pass/fail gates)

These evaluate whether the CDT embedding captures participant-level behavioural structure beyond the 7-class archetype label. All operate on the full dataset (1001 participants).

| Metric | Definition | Implementation |
|---|---|---|
| CDT geometry | UMAP coloured by archetype (between-persona separation) + coloured by continuous PersonaConfig param within cluster (within-persona variation) | `evaluation/geometry.py` |
| Cross-modal retrieval | recall@1 and recall@10: CDT embedding → each single-modality space (4 tests); single-modality → single-modality (6 pairs). Within-archetype recall@1 vs chance baseline (1/143) | `evaluation/retrieval.py` |
| PersonaConfig regression | Ridge R² for each of 7 latent params × 5 embedding sets (fused + 4 individual). Tests whether fusion recovers continuous latent variables better than any single modality | `evaluation/config_probe.py` |

**Interpreting the geometry evaluation**: the within-persona UMAP view (coloured by a continuous param such as `price_sensitivity`) tests whether the CDT embedding preserves individual deviation from the archetype centre. If participants cluster by archetype but show no gradient within clusters, the embedding has collapsed within-persona variation — it is a persona classifier, not a participant-level twin. A visible gradient is evidence that the embedding encodes individual behavioural structure beyond the label.

**Interpreting retrieval**: within-archetype recall@1 measures whether the CDT embedding identifies the same participant across modalities, not just the same archetype. Chance is ≈1/143. Recall significantly above chance is the primary evidence that fusion learns a shared participant-level representation.

**Interpreting the regression probe**: a fused R² higher than all four individual modalities for a given parameter is the clearest evidence fusion is combining complementary information. For parameters like `inspection_depth` (strongly encoded in traces) or `price_sensitivity` (strongly encoded in transactions), single-modality probes will score high — fusion should match or exceed them. For parameters that no single modality encodes well, fusion may add most.

These three evaluations together determine whether the prototype supports the strong CDT claim or only the weaker archetype-recovery claim — see `.claude/context/project-vision.md`.

## File Structure

```
fusion/
  SPEC.md             # this file
  __init__.py
  meta_learner.py     # LateFusionMetaLearner (Phase 1 + 2)
  early_fusion.py     # placeholder for Phase 3
  train.py            # fusion training + embedding cache generation

evaluation/
  strategy_recovery.py  # archetype classification accuracy + comparison table
  ablation.py           # leave-one-out modality ablation (4 tests)
  geometry.py           # UMAP projection + within-persona colouring
  retrieval.py          # cross-modal nearest-neighbour retrieval
  config_probe.py       # Ridge regression probes for PersonaConfig params

data/synthetic/
  participant_configs.jsonl  # PersonaConfig float params per participant
                             # written by generator/pipeline.py (bead c33)
                             # required by geometry.py and config_probe.py
```

### `meta_learner.py` interface

```python
class LateFusionMetaLearner(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        n_classes: int = 7,
        dropout: float = 0.2,
    ): ...

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, 512] concatenated modality embeddings
        # returns: [B, 7] logits

    def embed(self, x: Tensor) -> Tensor:
        # x: [B, 512]
        # returns: [B, 128] CDT embedding (no classification head)
```

### `train.py` interface

```python
def train(
    *,
    cache_path: Path | None = None,
    n_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    p_dropout: float = 0.2,
    device: str = "cpu",
    log_mlflow: bool = True,
) -> LateFusionMetaLearner: ...
```

Checkpoint saved to `models/fusion_meta_learner.pt` (state dict of meta-learner only, no encoder weights). Add `"fusion"` key to `schemas.CHECKPOINT_PATHS` before implementing `train.py`.

## Constraints

- Encoders are **always frozen** during fusion training. No joint fine-tuning in v0.1.
- The fusion layer trains only on persona classification. No unsupervised objective.
- CDT embedding dimension is fixed at **128** — matches individual encoder output dim, enabling direct comparison between single-encoder and fused embeddings in evaluation.
- `fusion/` imports encoder model classes (e.g. `from encoders.trace.model import TraceEncoder`) to reconstruct architectures for checkpoint loading, but never couples to encoder training logic. The prohibition in CLAUDE.md is generator↔encoders, not fusion→encoders.
- `fusion/` imports from `schemas/` for CHECKPOINT_PATHS, PERSONA_TO_IDX, and EMBEDDING_DIM. No circular imports.

## Resolved Design Decisions (resolved 2026-06-07, subject to human review gate 67a.2)

1. **Normalisation before concat**: **Yes — L2-normalise each 128-dim embedding before concatenation.**
   Each encoder was trained with a different objective and may produce embeddings with different magnitude scales. Without normalisation, the MLP will bias toward the highest-variance modality. Normalise each `[B, 128]` slice to unit norm before `torch.cat`:
   ```python
   embs = [F.normalize(e, dim=-1) for e in [trace_emb, tx_emb, text_emb, psycho_emb]]
   fusion_input = torch.cat(embs, dim=-1)  # [B, 512], each slice unit-normed
   ```
   Apply normalisation consistently at both training time (on cached embeddings) and inference time.

2. **Partial-modality inference**: **Yes — support missing modalities at inference by zeroing the absent slice.**
   Modality dropout at training time (p=0.2) already trains the model to produce useful outputs when any modality is absent. At inference, if a modality is unavailable for a participant, zero its 128-dim slice. Document this in `train.py` docstring. The output distribution will shift slightly from full-modality inference — note this as a known limitation, not a bug.
