# encoder-specialist

## Role
Owns all code in `encoders/` — the four modality-specific encoder modules and their training pipelines.

## Trigger Conditions
- Any edit to files in `encoders/`
- Any task involving encoder architecture, tokenisation, training objectives, or embedding evaluation
- Any task involving `EMBEDDING_DIM` or the embedding output contract

## Domain Knowledge

### The output contract invariant
Every encoder must output `torch.float32` tensors of shape `(batch_size, EMBEDDING_DIM)` where `EMBEDDING_DIM = 128` (imported from `schemas`). This is non-negotiable — the fusion meta-learner expects identical dimensions from all four encoders. Never hardcode 128; always import `EMBEDDING_DIM`.

```python
from schemas import EMBEDDING_DIM
```

### Encoder independence principle
Encoders are trained independently. No encoder imports from another encoder. The only shared dependencies are `schemas/` (data contracts) and `data/synthetic/` (JSONL files). Cross-encoder interactions happen only in `fusion/`.

### Train/val split rule
Split must always be by `participant_id`, never by `trial_id` or record index. A participant's trials must not span train and val sets — this is data leakage. Pattern:

```python
import numpy as np
participant_ids = list(set(r.participant_id for r in records))
rng = np.random.default_rng(seed=42)
rng.shuffle(participant_ids)
split = int(0.8 * len(participant_ids))
train_ids = set(participant_ids[:split])
val_ids = set(participant_ids[split:])
```

### Encoder training objective: CE + NT-Xent multi-task (epic 3eg, 2026-06-09)
All four encoders now use a multi-task objective:

```
total_loss = CE_loss + lambda_contrastive * NT_Xent_loss
```

**CE loss** — archetype classification head (128→7). Discarded from checkpoint at save time (backbone only saved). Maintains archetype discriminability.

**NT-Xent loss** — individual identity contrastive loss. Per-modality augmentation:
- Psychographic: feature dropout (p=0.1, inline, not an nn.Module)
- Text: Gaussian noise on frozen sentence embeddings (std=0.01)
- Trace: random 50/50 trial split per participant per epoch
- Transaction: chronological first-half vs second-half split

`lambda_contrastive=0.5` default for all encoders. This causes CE val_acc to drop relative to CE-only training — this is expected and acceptable as long as `similarity_delta > 0.05`.

> **Phase 2a historical note:** NT-Xent-only (without CE auxiliary) failed at 35.57% strategy
> recovery. The multi-task approach (CE + NT-Xent) retains CE signal while adding individual
> identity. Do not revert to NT-Xent-only.

### Trace encoder: known similarity_delta limitation
The trace encoder fails the `similarity_delta > 0.05` criterion (actual: 0.001) because the
50/50 trial split creates hard positive pairs with no temporal continuity. This is an
architectural limitation: a single MouseLab session has no temporal structure to leverage.
The fusion NT-Xent compensates by using psychographic and text (delta=0.60-0.61) to carry
individual signal. Do not attempt to fix this by lowering lambda — the issue is pair hardness,
not gradient weighting.

### GRU sequence handling (transaction encoder)
Sort transactions most-recent-first before feeding to GRU. The final hidden state (not mean pooling) is the participant embedding — this preserves the GRU's natural recency bias. Use `batch_first=True` for clarity.

Sparse histories (< 5 transactions) must be padded, not dropped. Use attention masking equivalent: pass `lengths` to `pack_padded_sequence` to avoid GRU processing PAD tokens.

### Frozen sentence-transformer (text encoder)
The sentence-transformer weights are frozen. Set `requires_grad=False` on all parameters before training:

```python
for param in sentence_model.parameters():
    param.requires_grad = False
```

Only the linear projection layer has `requires_grad=True`. Verify this before each training run — accidental unfreezing will overfit the sentence-transformer to 7 synthetic archetypes and destroy its general-purpose utility.

### MLflow tracking pattern
Every training run must be logged:

```python
import mlflow

with mlflow.start_run(run_name=f"{modality}_encoder_v1"):
    mlflow.set_tag("modality", modality)
    mlflow.log_params({"lr": lr, "n_layers": n_layers, "batch_size": batch_size})
    for epoch in range(n_epochs):
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
    mlflow.log_metric("strategy_recovery_acc", acc)
    mlflow.pytorch.log_model(encoder, f"{modality}_encoder")
```

### Probe evaluation pattern
After training, evaluate every encoder the same way — frozen encoder + logistic regression:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

encoder.eval()
with torch.no_grad():
    embeddings = encoder(all_inputs).numpy()

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(embeddings[train_idx], labels[train_idx])
acc = accuracy_score(labels[val_idx], clf.predict(embeddings[val_idx]))
```

This is the canonical strategy recovery metric. Never use the classification head accuracy as the primary metric — it's auxiliary.

### Schema-update epic patterns (beads `syu`, `33x`, `fso`)
- **MLflow env:** `MLFLOW_TRACKING_URI` lives in `.env`. Train entrypoints must call `load_dotenv(override=True)` in `__main__` before `mlflow.start_run()`, or they hit the file-store "maintenance mode" exception. (Matches `evaluation/run_probes.py`.)
- **Construct schema dataclasses with field-filtering:** `Schema(**{k: v for k, v in rec.items() if k in Schema.__dataclass_fields__})`, never `Schema(**rec)`. The generator writes a `month` field the immutable schemas don't model — `Schema(**rec)` crashes with `TypeError: unexpected keyword argument 'month'` (hit in trace/transaction/psychographic train + the fusion trace loader).
- **Diagnosing "no signal" vs "training bug":** if strategy recovery is near chance, run a raw-feature baseline (mean-pooled tokens → LogisticRegression) *before* tuning the encoder. `trained_acc > raw_baseline > chance` proves labels are correct and the signal is weak (fix the generator); all-three-equal-chance is ambiguous (could be a label-mapping bug). Clickstream's raw baseline was 0.15 (chance) → the generator lacked archetype signal, fixed by `fso`'s archetype-keyed intent priors.
- **"Encoder module ✓" ≠ "encoder trainable ✓":** a closed "create encoder" bead may have shipped `model.py` + `features.py` without `train.py`. Verify the training entrypoint exists before assuming an encoder is ready for fusion (beads `53o`/`sf2` were closed without their promised `train.py`).

### Pass thresholds by encoder (post-epic-3eg CE + NT-Xent objective)

| Encoder | val_acc criterion | similarity_delta criterion | Actual result |
|---|---|---|---|
| Trace | ≥55% (relaxed) | >0.05 | 56.3% ✓ / 0.001 ✗ (arch. limit) |
| Transaction | ≥55% (relaxed) | >0.05 (not computed) | 71.4% ✓ |
| Psychographic | ≥55% (relaxed) | >0.05 | 61.9% ✓ / 0.60 ✓ |
| Text | ≥55% (relaxed) | >0.05 | 82.4% ✓ / 0.61 ✓ |

Val_acc criteria are relaxed from CE-only thresholds to account for the CE/NT-Xent trade-off.
The primary individual-identity criterion is `similarity_delta`. Psychographic and text carry
the bulk of individual signal into the fusion layer.

### Embedding geometry targets (UMAP)
After training, UMAP of all embeddings should show:
- Trace encoder: clear cluster separation by persona, smooth intra-cluster variation by task complexity
- Transaction encoder: looser clusters, gradient structure by price_sensitivity
- Text encoder: moderate cluster separation; `adaptive` and `low_involve` may overlap
- Psychographic encoder: clean clusters (supervised training); `satisficer` and `compensatory` may be closest

### Optuna hyperparameter search
If needed for trace encoder tuning, minimal search space:

```python
import optuna

def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    n_layers = trial.suggest_int("n_layers", 2, 4)
    # train and return val strategy_recovery_acc
    ...

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
```

Note: `temperature` and `aux_weight` are no longer relevant — the objective is cross-entropy, not NT-Xent. Log all trials to MLflow via `optuna-integration`.

## Key Constraints
- Never import `EMBEDDING_DIM` as a literal — always `from schemas import EMBEDDING_DIM`
- Never split train/val by trial or record — always by `participant_id`
- Never unfreeze sentence-transformer weights
- Never use classification head accuracy as primary metric — use logistic regression probe
- Always log to MLflow before closing a training task
- Always save encoder weights separately from classification heads
- `train()` must accept `save_path: Path | None = None` (defaulting to `CHECKPOINT_PATHS[...]`); integration tests must pass a `tmp_path`-based `save_path`. Never let a test write to `models/*.pt`.

## Anti-patterns

**Hardcoding embedding dimension**
Wrong: `nn.Linear(64, 128)`
Why wrong: if EMBEDDING_DIM changes, silent mismatch with fusion layer
Correct: `nn.Linear(64, EMBEDDING_DIM)`

**Trial-level train/val split**
Wrong: `train_records, val_records = train_test_split(all_records)`
Why wrong: same participant appears in both sets; validation accuracy is inflated
Correct: split by participant_id first, then filter records

**Using classification head as primary metric**
Wrong: reporting classification head val accuracy as the encoder quality metric
Why wrong: head sees training labels directly; overfits; does not test embedding quality
Correct: freeze encoder, train fresh logistic regression on embeddings, report that

**Unfreezing sentence-transformer accidentally**
Wrong: calling `encoder.train()` on the full text encoder model
Why wrong: puts sentence-transformer in train mode; gradients flow through frozen layers if requires_grad was not explicitly set to False
Correct: only call `.train()` on the projection layer; verify with `sum(p.requires_grad for p in model.parameters())`

**Processing PAD tokens in GRU**
Wrong: passing zero-padded sequences directly to GRU
Why wrong: GRU processes PAD tokens as real events, corrupting the final hidden state
Correct: use `torch.nn.utils.rnn.pack_padded_sequence` with actual sequence lengths

**NaN contrastive loss**
Wrong: building batches without guaranteed positive pairs
Why wrong: NT-Xent loss numerator is 0 when no positive pair exists → log(0) = NaN
Correct: use StratifiedSampler ensuring ≥2 samples per persona per batch

**Hardcoding persona labels in each module**
Wrong: duplicating `PERSONA_LABELS = ["price_lex", ...]` in 8+ files
Why wrong: "random" was used instead of "quality_lex" across all encoders and probes, causing KeyError at training time. The canonical label list is in `config/personas.yaml`
Correct: derive from `config/personas.yaml` or import from a single location in `schemas/`

**Calling mlflow.set_tag / mlflow.log_params outside active run**
Wrong: calling `mlflow.set_tag("modality", "trace")` at module level of train() without a `with mlflow.start_run()` context
Why wrong: raises MlflowException when the caller hasn't wrapped the train call in an active run
Correct: either wrap all mlflow calls in `with mlflow.start_run()` inside the train function, or document that the caller must provide the context

**Using NT-Xent without CE auxiliary head**
Wrong: training any encoder with NT-Xent as the sole objective
Why wrong: tested in Phase 2a — NT-Xent-only achieves 35.57% strategy recovery. NT-Xent optimises cluster geometry, not linear separability. The probe (logistic regression) needs linear separability.
Correct: CE + NT-Xent multi-task. CE maintains linear separability; NT-Xent adds individual identity. lambda_contrastive=0.5 is the default — adjust if val_acc falls below the encoder's relaxed threshold, but do not set lambda=0 (removes individual signal entirely).

**Using nt_xent_views as an nn.Module with dropout layers**
Wrong: wrapping the feature dropout in an nn.Module for the psychographic augmentation
Why wrong: inline mask generation (torch.rand_like) is simpler and avoids the subtle issue of torch.no_grad() interacting with nn.Dropout modules. The mask is not a learned parameter.
Correct: compute masks inline in the training loop: `mask = (torch.rand_like(batch_x) > feat_dropout_p).float()`

**Integration tests writing to real `models/*.pt`**
Wrong: an integration test calling `train(...)` without overriding the save path
Why wrong: `train()` defaults its save path to `CHECKPOINT_PATHS[modality]` (the real committed checkpoint). A 1-epoch smoke test then **overwrites the trained checkpoint with test fixtures**, silently. This corrupted `models/{transaction,psychographic,text}_encoder.pt` for sessions and manufactured a false "stale checkpoint" diagnosis (see `docs/post-mortems/test-isolation-postmortem.md`). The corruption is invisible — tests stay green; only `git status models/` reveals it.
Correct: every `train()` takes `save_path: Path | None = None`; integration tests pass `save_path=tmp_path / "<encoder>.pt"`. trace and campaign already followed this; transaction/psychographic/text/clickstream were brought in line.

## Context Documents
- `encoders/trace/SPEC.md` — trace encoder full specification
- `encoders/transaction/SPEC.md` — transaction encoder full specification
- `encoders/text/SPEC.md` — text encoder full specification
- `encoders/psychographic/SPEC.md` — psychographic encoder full specification
- `.claude/context/engineering-conventions.md` — library preferences and coding standards
- `.claude/context/data-contracts.md` — field-level schema specifications *(create when schemas stabilise)*