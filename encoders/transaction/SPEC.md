# Transaction Encoder Specification

## Current Version: 0.1

## Purpose

Encode 12-month purchase histories into fixed-dimension embeddings that capture preference magnitude, price sensitivity, and behavioural drift. Provides the revealed preference signal that process traces alone cannot supply.

## Inputs

| Source | File | Schema |
|---|---|---|
| Transaction records | `data/synthetic/transactions.jsonl` | `TransactionRecord` |

Loading pattern:
```python
from schemas import TransactionRecord
import json
from pathlib import Path
from collections import defaultdict

DATA = Path("data/synthetic")
by_participant: dict[str, list[TransactionRecord]] = defaultdict(list)
for line in (DATA / "transactions.jsonl").open():
    r = TransactionRecord(**json.loads(line))
    by_participant[r.participant_id].append(r)

# Sort each participant's history: most recent first
for pid in by_participant:
    by_participant[pid].sort(key=lambda r: r.days_before_session)
```

## Tokenisation

Each `TransactionRecord` maps to one token. Token feature vector (7-dim per event):

| Feature | Derivation | Notes |
|---|---|---|
| `brand_tier_embed` | Learned embedding, vocab = {premium, mid, value, own_label} | dim=8 |
| `channel_embed` | Learned embedding, vocab = Channel enum values | dim=4 |
| `purchase_type_embed` | Learned embedding, vocab = PurchaseType enum values | dim=4 |
| `price_paid_norm` | `TransactionRecord.price_paid_normalised` | 0–1, direct |
| `on_promotion` | Float cast of bool | 0.0 or 1.0 |
| `quantity_norm` | `min(quantity / 5.0, 1.0)` | Capped normalisation |
| `recency_norm` | `1.0 - (days_before_session / 365.0)` | 1.0 = today, 0.0 = 365 days ago |

Final token dim: 8 + 4 + 4 + 1 + 1 + 1 + 1 = **20-dim per token**.

Sequence length varies by participant (mean ~30, range 10–60). Max sequence length: 80 tokens. Pad shorter sequences.

## Architecture

```
Input sequence: [T × 20]
        ↓
Linear projection: [T × 64]
        ↓
GRU: hidden_size=128, num_layers=2, dropout=0.1, batch_first=True
        ↓
Final hidden state (last layer): [128]
        ↓
Linear projection + LayerNorm: [128]
        ↓
e_transaction: [EMBEDDING_DIM=128]
```

Use final hidden state of last GRU layer, not mean pooling — the GRU's recency bias (final state reflects recent history more strongly) is a feature, not a bug, for preference modelling.

## Training Objective

### Self-supervised: Next brand_tier prediction

Given transaction history t₁...tₙ, predict `brand_tier` of tₙ₊₁.

```
GRU output at step n → linear head → 4-class softmax over brand_tier vocab
```

Cross-entropy loss. Sequence is shifted by 1: input = t₁...tₙ, target = t₂...tₙ₊₁.

This is self-supervised — uses only the transaction sequence itself, no persona labels required.

### Why next brand_tier (not price)

Price is already in the token as `price_paid_normalised`. Predicting brand_tier forces the GRU to learn switching behaviour and loyalty patterns — the signals most relevant to the twin's downstream tasks (response to new entrants, brand substitution).

## Training Configuration

| Parameter | Value | Notes |
|---|---|---|
| Batch size | 128 | Participant-level batching |
| Learning rate | 5e-4 | Tune with Optuna |
| Epochs | 30 | Early stopping on val loss |
| Optimiser | AdamW, weight_decay=1e-4 | |
| Train/val split | 80/20 by participant | |
| Device | CPU | |

## Evaluation

| Metric | Method | Pass threshold |
|---|---|---|
| Next brand_tier accuracy | Top-1 accuracy on held-out val set | >45% (baseline: 25% uniform) |
| Price sensitivity recovery | Pearson r between mean(price_paid_norm) and `price_sensitivity` from psychographic | >0.7 |
| Strategy recovery (auxiliary) | Freeze encoder; logistic regression on `e_transaction`; predict `persona_id` | >60% (lower than trace encoder — expected) |

Transaction encoder is expected to carry less strategy signal than the trace encoder — it carries magnitude and loyalty signal instead. The 60% strategy recovery threshold reflects this.

## Output Contract

```python
# Shape: (batch_size, EMBEDDING_DIM)
# dtype: torch.float32
# One embedding per participant (GRU processes full history)
e_transaction = encoder(batch)  # [batch_size, 128]
```

No trial-level aggregation needed — output is already participant-level.

## Known Constraints

- Participants with fewer than 5 transactions are unreliable — flag but do not exclude; the encoder must handle sparse histories gracefully via padding
- Sort order matters: most recent first, so the GRU final hidden state reflects recent behaviour
- `days_before_session` normalisation assumes 365-day lookback — do not use raw days as a feature
- Brand tier embedding must be learned, not one-hot — vocabulary is small but ordinal relationships (premium > mid > value) should be learnable