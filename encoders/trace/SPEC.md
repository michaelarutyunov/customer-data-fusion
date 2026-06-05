# Trace Encoder Specification

## Current Version: 0.1

## Purpose

Encode variable-length MouseLab acquisition sequences into fixed-dimension embeddings that capture decision strategy. The embedding must be discriminative between persona archetypes and transferable across task formats (information board → conjoint).

## Inputs

| Source | File | Schema |
|---|---|---|
| Acquisition events | `data/synthetic/traces.jsonl` | `AcquisitionEvent` |
| Trial metadata | `data/synthetic/trials.jsonl` | `TrialRecord` |

Loading pattern:
```python
from schemas import AcquisitionEvent, TrialRecord
import json
from pathlib import Path

DATA = Path("data/synthetic")
events = [AcquisitionEvent(**json.loads(l)) for l in (DATA / "traces.jsonl").open()]
trials = {r.trial_id: r for l in (DATA / "trials.jsonl").open()
          for r in [TrialRecord(**json.loads(l))]}
```

Group events by `trial_id` before tokenisation. Sort by `event_index` within each trial.

## Tokenisation

Each `AcquisitionEvent` maps to one token. Token feature vector (5-dim per event):

| Feature | Derivation | Notes |
|---|---|---|
| `attribute_embed` | Learned embedding, vocab = all unique `attribute_id` values | dim=16 |
| `alternative_embed` | Learned embedding, vocab = all unique `alternative_id` values | dim=8 |
| `timestamp_norm` | `timestamp_s / trial_duration_s` | 0–1 normalised |
| `dwell_zscore` | Z-score of `dwell_ms` within trial | Removes absolute scale |
| `is_reinspection` | Float cast of bool (0.0 or 1.0) | |

Final token dim: 16 + 8 + 1 + 1 + 1 = **27-dim per token**.

Sequence length varies per trial. Use padding + attention mask for batching. Max sequence length: 200 tokens (covers 99th percentile of synthetic data).

## Architecture

```
Input sequence: [T × 27]
        ↓
Linear projection: [T × 64]
        ↓
+ Positional encoding (learned, max_len=200)
        ↓
Transformer encoder: 4 heads, 3 layers, d_model=64, d_ff=256, dropout=0.1
        ↓
CLS token output: [64]
        ↓
Linear projection: [128]
        ↓
e_trace: [EMBEDDING_DIM=128]
```

CLS token is prepended to the sequence (index 0), not appended.

## Training Objectives

### Primary: Contrastive loss (NT-Xent / SimCLR variant)

Positive pairs: two trials from the same `persona_id`.
Negative pairs: trials from different `persona_id` within the same batch.
Temperature: τ = 0.07 (tunable via Optuna).

$$\mathcal{L}_{contrastive} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k)/\tau)}$$

### Auxiliary: Strategy classification head (weight = 0.3)

Linear head on top of frozen CLS embedding → 7-class softmax (one per persona archetype).
Cross-entropy loss.

$$\mathcal{L}_{total} = \mathcal{L}_{contrastive} + 0.3 \cdot \mathcal{L}_{classification}$$

Classification head is auxiliary only — it is discarded after training. It provides supervision signal to prevent embedding collapse during contrastive training.

## Training Configuration

| Parameter | Value | Notes |
|---|---|---|
| Batch size | 256 | Must contain multiple trials per persona for contrastive pairs |
| Learning rate | 1e-3 | Tune with Optuna |
| Epochs | 50 | Early stopping on validation contrastive loss |
| Optimiser | AdamW, weight_decay=1e-4 | |
| Train/val split | 80/20 by participant | Never split by trial — leakage risk |
| Device | CPU | Sufficient at prototype scale |

## Evaluation

| Metric | Method | Pass threshold |
|---|---|---|
| Strategy recovery accuracy | Freeze encoder; train logistic regression on `e_trace`; predict `persona_id` | >85% |
| Intra/inter cluster ratio | Mean intra-persona distance / mean inter-persona distance in embedding space | <0.5 |
| Contrastive loss | Final validation loss | Decreasing trend; no collapse |
| Format transfer | Embed conjoint-format trials; verify they land in correct persona cluster | Visual inspection |

## Output Contract

```python
# Shape: (n_trials, EMBEDDING_DIM)
# dtype: torch.float32
# One embedding per trial (not per participant)
# Aggregation to participant level: mean pooling over trials
e_trace = encoder(batch)  # [batch_size, 128]
```

Participant-level embedding: mean of all trial embeddings for that participant.

## Known Constraints

- Train/val split must be by `participant_id`, never by `trial_id` — same participant's trials must not span train and val sets (data leakage)
- Batch construction must ensure each batch contains at least 2 trials per persona for contrastive pairs to exist
- Sequence padding must use attention mask — do not include PAD tokens in CLS attention
- `adaptive` persona has highest intra-class variance by design — expect lower recovery accuracy for this archetype specifically