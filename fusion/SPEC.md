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

During training, each modality's embedding is independently zeroed out with probability `p_dropout = 0.2`. This trains the meta-learner to produce useful outputs even when one or more modalities are absent.

```python
for i, emb in enumerate([trace_emb, tx_emb, text_emb, psycho_emb]):
    if training and torch.rand(1).item() < p_dropout:
        emb = torch.zeros_like(emb)
fusion_input = torch.cat([trace_emb, tx_emb, text_emb, psycho_emb], dim=-1)
```

Dropout is applied independently per sample, not per batch. During evaluation, all four modalities are always active (no dropout).

## Training Objective

Supervised classification: 7 persona archetypes (same label set as individual encoder probes).

```
Loss = CrossEntropyLoss(logits, persona_idx)
```

Labels from `schemas.PERSONA_TO_IDX`. 80/20 train/val split stratified by persona. Same split indices as encoder probes for consistency.

Optimiser: Adam, lr=1e-3, weight_decay=1e-4. Scheduler: ReduceLROnPlateau (patience=5, factor=0.5). Early stopping: patience=10 epochs on val accuracy. Max epochs: 100.

## Data Loading

The fusion training script must:
1. Load all four encoder checkpoints (frozen).
2. For each participant, run all four encoders forward to produce embeddings.
3. Cache embeddings to disk — never re-run encoders during training epochs.
4. Train the meta-learner on cached embeddings only.

Embedding cache path: `models/fusion_embeddings_cache.pt`. Format: a dict with keys `"trace"`, `"transaction"`, `"text"`, `"psychographic"` each mapping to a `[1001, 128]` tensor, plus `"labels"` as a `[1001]` int tensor and `"participant_ids"` as a list of strings.

Cache is invalidated if any encoder checkpoint is newer than the cache file (check mtime). Training script must detect and regenerate automatically.

## Evaluation Metrics

All metrics computed on the val split (201 participants) unless stated otherwise.

| Metric | Definition | Threshold |
|---|---|---|
| Strategy recovery accuracy | Top-1 accuracy of persona classification | >85% |
| Per-modality ablation delta | Accuracy drop when one modality is zeroed out | Each modality ≥5% (else redundant) |
| Modality importance weights | Mean absolute output change per modality zeroing test | Logged, no threshold |
| CDT embedding geometry | UMAP coloured by persona — visual cluster separation | Qualitative |

**Ablation test procedure**: for each modality, zero out its 128-dim slice of the 512-dim input and re-evaluate accuracy on the val split. Report the delta from the full-modality baseline. A delta < 5% flags potential redundancy (see Phase 2a fix post-mortem R7 — text and psychographic encoders both achieve 100% probe accuracy and may encode correlated information).

## File Structure

```
fusion/
  SPEC.md             # this file
  __init__.py
  meta_learner.py     # LateFusionMetaLearner (Phase 1 + 2)
  early_fusion.py     # placeholder for Phase 3
  train.py            # fusion training + embedding cache generation
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
- `fusion/` never imports from `encoders/` directly — load encoder state dicts via checkpoint paths and re-instantiate model classes from their respective modules.
- `fusion/` imports from `schemas/` for CHECKPOINT_PATHS, PERSONA_TO_IDX, and EMBEDDING_DIM. No circular imports.

## Open Questions (resolve before implementing 67a.3)

1. **Normalisation before concat**: Should each 128-dim embedding be L2-normalised before concatenation? Unnormalised embeddings from different encoders may have different magnitude scales, biasing the MLP toward higher-variance modalities.
2. **Modality masking at inference**: Should the trained model support partial-modality inference (e.g., participant has no transaction data)? If yes, modality dropout at training time already handles this — but the output distribution will shift and should be documented.
