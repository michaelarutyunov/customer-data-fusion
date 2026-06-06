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

### Primary: Supervised cross-entropy classification

Direct 7-class classification on `persona_id` archetype label. Loss:

$$\mathcal{L} = \text{CrossEntropy}(\text{logits}, \text{persona\_label})$$

A linear classification head is added on top of the CLS embedding for training. The head is discarded after training; only the encoder backbone (up to and including the 128-dim projection) is saved to checkpoint.

**Why not contrastive (NT-Xent)?** NT-Xent was evaluated first (Phase 2a) and produced 35.57% strategy recovery regardless of participant count. The root cause is that NT-Xent optimises cluster *geometry* (same-class embeddings close, different-class far), while the downstream logistic regression probe requires *linear separability*. With only 7 classes and stochastic strategy simulation, NT-Xent cannot learn the required structure. Supervised cross-entropy directly optimises linear separability and achieves 95%+ strategy recovery on the same data. See `.claude/context/phase2a-fix-postmortem.md` for the full diagnosis.

## Training Configuration

| Parameter | Value | Notes |
|---|---|---|
| Batch size | 256 | |
| Learning rate | 1e-3 | |
| Epochs | 50 | Early stopping on validation classification loss (patience=10) |
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
- Sequence padding must use attention mask — do not include PAD tokens in CLS attention
- `adaptive` persona has highest intra-class variance by design — expect lower recovery accuracy for this archetype specifically
- The classification head (linear, 128→7) is auxiliary to training — discard it after training, save only the encoder backbone up to `e_trace`