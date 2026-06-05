# Psychographic Encoder Specification

## Current Version: 0.1

## Purpose

Encode fixed-width psychographic and demographic survey vectors into embeddings that capture trait-level priors. Simplest encoder — MLP projector, fully supervised. Serves as the baseline modality in ablation experiments.

## Inputs

| Source | File | Schema |
|---|---|---|
| Psychographic vectors | `data/synthetic/psychographics.jsonl` | `PsychographicVector` |

Loading pattern:
```python
from schemas import PsychographicVector
import json
from pathlib import Path

DATA = Path("data/synthetic")
psychographics = [PsychographicVector(**json.loads(l))
                  for l in (DATA / "psychographics.jsonl").open()]
```

## Feature Engineering

`PsychographicVector` contains mixed types — continuous floats and categorical strings. Pre-processing before MLP:

### Continuous fields (pass directly, already 0–1 normalised)
- `involvement_score`
- `maximiser_score`
- `risk_tolerance`
- `price_consciousness`
- `brand_sensitivity`
- `openness_to_new`

### Categorical fields (ordinal or nominal encoding)
| Field | Encoding | Output dim |
|---|---|---|
| `decision_style_dominant` | One-hot (5 classes) | 5 |
| `age_band` | Ordinal integer / 5.0 | 1 |
| `household_type` | One-hot (4 classes) | 4 |
| `employment_status` | One-hot (5 classes) | 5 |
| `purchase_frequency_band` | Ordinal integer / 3.0 | 1 |

Total input dim: 6 continuous + 5 + 1 + 4 + 5 + 1 = **22-dim**.

Note: demographics (`age_band`, `household_type`, `employment_status`) are population-level calibrators. Include them in the input but do not interpret their individual coefficients as preference predictors.

## Architecture

```
Input: [22]
        ↓
Linear(22 → 64) + ReLU + Dropout(0.2)
        ↓
Linear(64 → 128) + ReLU
        ↓
LayerNorm([128])
        ↓
e_psychographic: [EMBEDDING_DIM=128]
```

## Training Objective

Supervised strategy classification:

```
e_psychographic → Linear(128 → 7) → softmax → cross-entropy
```

Uses `persona_id` as ground truth label (7 classes). This is the only encoder that is fully supervised from the start — psychographic vectors are designed to correlate with strategy, so supervision is appropriate.

## Training Configuration

| Parameter | Value | Notes |
|---|---|---|
| Batch size | 128 | |
| Learning rate | 1e-3 | |
| Epochs | 40 | |
| Optimiser | AdamW, weight_decay=1e-4 | |
| Train/val split | 80/20 by participant | |
| Device | CPU | |

## Evaluation

| Metric | Method | Pass threshold |
|---|---|---|
| Strategy recovery accuracy | Freeze encoder; logistic regression on `e_psychographic`; predict `persona_id` | >75% |
| Ablation baseline | Compare against raw features (no MLP) | MLP must outperform raw features |

75% target reflects that psychographic signals correlate with strategy but with more noise than traces. If accuracy exceeds 90%, the synthetic psychographic generation is too deterministic — add more within-archetype variance in `generator/psychographic_generator.py`.

## Output Contract

```python
# Shape: (batch_size, EMBEDDING_DIM)
# dtype: torch.float32
# One embedding per participant
e_psychographic = encoder(features)  # [batch_size, 128]
```

## Feature Engineering Module

Pre-processing lives in `encoders/psychographic/features.py`, not in the model. Keep the model clean:

```python
# encoders/psychographic/features.py
def to_feature_vector(psych: PsychographicVector) -> torch.Tensor:
    """Convert PsychographicVector to 22-dim float tensor."""
    ...
```

## Known Constraints

- Demographics are included as features but must not dominate — if ablation shows psychographic encoder outperforms trace encoder, suspect over-fitting to demographic noise in synthetic data
- `years_buying_category` is `Optional[int]` — impute with median (5 years) when None; do not drop
- Do not normalise the already-normalised continuous fields — they are already 0–1
- One-hot encoding vocabulary must be fixed at training time and persisted for inference — do not infer vocabulary dynamically from each batch