# fusion-specialist

## Role
Owns all code in `fusion/` — the late-fusion meta-learner that combines modality embeddings into a per-consumer behavioural embedding.

## Trigger Conditions
- Any edit to files in `fusion/`
- Any task involving fusion architecture, meta-learner training, or embedding combination

## Architecture

Default: **late fusion** — four frozen encoder embeddings concatenated to `[B, 512]`, passed to a shallow MLP meta-learner, compressed to `[B, 128]` CDT embedding.

### LateFusionMetaLearner (fusion/meta_learner.py)

Single PyTorch class. Two phases within one class:

**Phase 1 — logistic baseline**: `Linear(512, 7)` only. No hidden layers. Establishes linear separability ceiling.

**Phase 2 — MLP (default v0.1)**:
```
[B, 512]
  → Linear(512, 256) → LayerNorm(256) → GELU → Dropout(0.2)
  → Linear(256, 128) → LayerNorm(128) → GELU → Dropout(0.1)
  → Linear(128, 7)   # classification head — removed at inference
```

The **128-dim output of the second hidden layer is the CDT embedding**. The `[B, 7]` classification head is a training proxy only.

Interface:
```python
class LateFusionMetaLearner(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, embed_dim=128, n_classes=7, dropout=0.2): ...
    def forward(self, x: Tensor) -> Tensor: ...                          # [B, 512] → [B, 7] logits
    def embed(self, x: Tensor) -> Tensor: ...                            # [B, 512] → [B, 128] CDT embedding
    def forward_with_embedding(self, x: Tensor) -> tuple[Tensor, Tensor]: ...  # ([B, 7], [B, 128])
```

Always use `forward_with_embedding()` in the training loop — it avoids duplicating the forward pass for CE and NT-Xent.

### Training Objective: CE + NT-Xent multi-task (bead 0if, 2026-06-09)

```
total_loss = CE_loss + lambda_contrastive * NT_Xent_loss
```

**CE auxiliary head** — trains on `logits` from `forward_with_embedding`. Preserves archetype separability (Tier 1 gate). Convergence: val_acc=100% within ~5 epochs.

**NT-Xent contrastive loss** — trains on `embedding` (CDT 128-dim) from `forward_with_embedding`. Positive pairs: two modality-dropout augmented views of the same participant. Other participants in the batch are negatives (SimCLR-style, temperature=0.07).

Augmentation: two independent calls to `_apply_modality_dropout(batch_embs, p=0.2)` produce `v1` and `v2`. This means each view has roughly 80% of each modality present, independently. The NT-Xent trains the CDT embedding to be stable under this variation.

Default: `lambda_contrastive=0.5`, `nt_xent_temperature=0.07`.

### Modality Dropout

Each modality's 128-dim slice is independently zeroed with `p=0.2` per sample. Used for both:
1. **Training augmentation**: two independent dropout views per batch → NT-Xent positive pair
2. **Missing-data robustness**: if a real modality is absent, zero its slice — trained to handle this

```python
def _apply_modality_dropout(batch_embs, p_dropout, device):
    masks = [torch.rand(B, 1, device=device) >= p_dropout for _ in range(4)]
    result = batch_embs.clone()
    for i in range(4):
        result[:, i] = result[:, i] * masks[i].float()
    return result
```

At inference with all modalities present, pass unmasked embeddings directly.

### Embedding Cache (models/fusion_embeddings_cache.pt)

Run all four frozen encoders once; cache to disk. Format:
```python
{
    "trace":          Tensor[N, 128],
    "transaction":    Tensor[N, 128],
    "text":           Tensor[N, 128],
    "psychographic":  Tensor[N, 128],
    "labels":         Tensor[N],        # persona_idx ints
    "participant_ids": list[str],        # canonical ordering from psychographics.jsonl
}
```
Invalidate if any encoder `.pt` is newer than the cache file (mtime check). Row *i* across all tensors is the same participant.

When encoder checkpoints are updated (e.g. after retraining), delete `models/fusion_embeddings_cache.pt` explicitly before running `fusion.train` — the mtime check handles this automatically, but explicit deletion is clearer.

### train.py interface

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
    phase: str = "2",
    lambda_contrastive: float = 0.5,
    nt_xent_temperature: float = 0.07,
) -> LateFusionMetaLearner: ...
```

MLflow logs per epoch: `train_ce_loss`, `train_nt_loss`, `val_loss`, `val_acc`.

## Key Constraints

- Encoders are **always frozen** during fusion training (`requires_grad=False`)
- Meta-learner only sees cached embeddings — never raw data, never re-runs encoders during training
- Load checkpoints from `schemas.CHECKPOINT_PATHS` — never hardcode paths
- CDT embedding dim is fixed at **128** (`schemas.EMBEDDING_DIM`) — must match individual encoder output dim
- `fusion/` may import encoder model classes to reconstruct architectures for checkpoint loading; it must not import encoder training logic
- When instantiating encoders in `load_encoders()`, use default constructor arguments — do NOT hardcode non-default dims (e.g. `TransactionEncoder()` not `TransactionEncoder(projection_dim=16, gru_hidden=32)`)
- `load_encoders(modalities=...)` / `generate_embeddings(...)` take an explicit modality set (default: all `CHECKPOINT_PATHS`); `main()` drops `"text"` when `narratives.jsonl` is empty (dry-run mode). Instantiate `LateFusionMetaLearner(n_modalities=...)` with the actual loaded count.
- Every run must assert `n_modalities == len(encoders)` and the stacked embedding shape `[N, n_modalities, 128]`; `_MODALITIES` must exclude BOTH `"labels"` and `"participant_ids"`. A 4-modality regression or a `participant_ids` leak must fail loudly.

## What This Enables (the evaluation perspective)

The fusion layer's output is evaluated at two levels:

1. **Archetype recovery** (pass/fail gate): Tier 1 accuracy >85% on val split via CE auxiliary head. Result: **100%**.

2. **Individual identity** (primary CDT claim): dropout-view recall@1 — given two random modality-dropout views of the same participant, what fraction rank #1 among N=210 val candidates. Result: **70.4%** (criterion >0.1). This is the correct retrieval metric for the NT-Xent model.

   > **Do not use** `evaluation/retrieval.py` CDT-vs-single-modality recall@1 as the individual identity metric. It measures alignment between the CDT space (meta-learner output) and individual encoder spaces — spaces that were never trained to align. These values will be near-zero regardless of the fusion objective and are not informative.

## Anti-patterns

**Hardcoding encoder constructor arguments**
Wrong: `TransactionEncoder(projection_dim=16, gru_hidden=32)`
Why wrong: the checkpoint was trained with defaults (projection_dim=64, gru_hidden=128); this causes a size mismatch RuntimeError at load time
Correct: `TransactionEncoder()` — use defaults, they must match what the encoder was trained with

**Using CE-only loss for fusion**
Wrong: `loss = criterion(logits, batch_labels)` with no NT-Xent term
Why wrong: CE discards within-archetype variation; the CDT embedding collapses individuals to archetype-level representations; dropout-view recall@1 will be near-zero
Correct: multi-task `loss = ce_loss + lambda * nt_loss`

**Evaluating individual identity with CDT-vs-encoder retrieval**
Wrong: using `evaluation/retrieval.py` recall@1 as the individual identity pass/fail
Why wrong: CDT and individual encoder spaces were never trained to align; this will always be near-zero and is not a diagnostic of individual identity
Correct: compute two dropout-augmented CDT views, measure recall@1 between them (without diagonal masking — the positive pair IS the diagonal when query ≠ gallery matrix)

**Masking diagonal in recall@1 when query ≠ gallery**
Wrong: `sim.fill_diagonal_(-inf)` when computing recall between two DIFFERENT embedding matrices (v1 and v2 from different dropout seeds)
Why wrong: the diagonal IS the correct positive match (participant i in v1 vs participant i in v2). Masking it removes the answer and makes recall@1 always 0.
Correct: only fill diagonal when query and gallery are the SAME matrix (same-space self-retrieval)

**Letting `participant_ids` leak into `_MODALITIES`**
Wrong: `_MODALITIES = [k for k in embeddings if k != "labels"]`
Why wrong: the `embeddings` cache also holds `"participant_ids"` (a list, not a tensor); it leaks into the modality set and corrupts `torch.stack`/`torch.cat`.
Correct: `_MODALITIES = [k for k in embeddings if k not in ("labels", "participant_ids")]`, guarded by `assert n_modalities == len(encoders)` and a stacked-shape assert `[N, n_modalities, 128]`.

**Trusting "variable-N" claims without asserting the actual N**
Wrong: assuming a dynamic `_MODALITIES` derivation means the loader is variable-modality.
Why wrong: the meta-learner and `_MODALITIES` line can be variable while the embedding *production* path (`load_encoders`/`generate_embeddings`) stays hardcoded to 4 — a flexible consumer fed by a rigid producer silently degrades to the producer's count (the schema-update "half-truth"; the 6-modality meta-learner masked a 4-modality loader).
Correct: `load_encoders`/`generate_embeddings` take an explicit modality set; assert `n_modalities == len(encoders)` + stacked shape in every run so a count regression fails loudly. Instantiate `LateFusionMetaLearner` with the *actual* `n_modalities`, not the default 6.

## Context Documents

- `fusion/SPEC.md` — authoritative implementation contract (read first)
- `.claude/context/fusion-architecture.md` — design rationale and why decisions were made
- `.claude/context/prd-validation.md` — quantitative results and overall verdict
- `.claude/context/project-vision.md` — can/cannot-claim framing for the CDT
