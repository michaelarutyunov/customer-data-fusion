# Text Encoder Specification

## Current Version: 0.1

## Purpose

Encode persona narratives into fixed-dimension embeddings that capture motivational structure, category relationship, and decision style language. Uses a frozen pretrained sentence-transformer — no training loop required.

## Inputs

| Source | File | Schema |
|---|---|---|
| Persona narratives | `data/synthetic/narratives.jsonl` | `PersonaNarrative` |

Loading pattern:
```python
from schemas import PersonaNarrative
import json
from pathlib import Path

DATA = Path("data/synthetic")
narratives = [PersonaNarrative(**json.loads(l))
              for l in (DATA / "narratives.jsonl").open()]
```

Check `PersonaNarrative.embedding is None` before running inference — do not re-embed if already populated.

## Architecture

```
Input: PersonaNarrative.text (string, 250–350 words)
        ↓
sentence-transformers: all-MiniLM-L6-v2 (frozen, no gradient)
        ↓
Sentence embedding: [384]
        ↓
Linear projection (trained): [384 → 128]
        ↓
LayerNorm
        ↓
e_text: [EMBEDDING_DIM=128]
```

The sentence-transformer is fully frozen. Only the linear projection layer is trained.

## Training Objective

The projection layer is trained via strategy classification:

Linear projection output → linear head → 7-class softmax → cross-entropy loss.

This is the only trained component. It learns to project the 384-dim semantic space into a 128-dim subspace that is discriminative for decision strategy.

## Training Configuration

| Parameter | Value | Notes |
|---|---|---|
| Batch size | 64 | Participant-level; one narrative per participant |
| Learning rate | 1e-3 | Higher than other encoders — only projection is trained |
| Epochs | 20 | Small; projection layer converges fast |
| Optimiser | AdamW | |
| Train/val split | 80/20 by participant | |
| Device | CPU | sentence-transformer inference is CPU-viable at 1k samples |

## Embedding Persistence

After inference, write embeddings back to `narratives.jsonl`:

```python
from dataclasses import asdict, replace
import json

def save_embeddings(narratives: list[PersonaNarrative],
                    embeddings: list[list[float]],
                    path: Path) -> None:
    with path.open("w") as f:
        for narrative, emb in zip(narratives, embeddings):
            updated = replace(narrative,
                              embedding=emb,
                              embedding_model_id="all-MiniLM-L6-v2")
            f.write(json.dumps(asdict(updated)) + "\n")
```

Use `dataclasses.replace()` — do not mutate the frozen dataclass directly.

## Evaluation

| Metric | Method | Pass threshold |
|---|---|---|
| Strategy recovery accuracy | Freeze projection; logistic regression on `e_text`; predict `persona_id` | >70% |
| Intra-persona cosine similarity | Mean cosine sim between same-persona narrative embeddings | >0.6 |
| Cross-persona separation | Mean cosine sim between different-persona narrative embeddings | <0.4 |

70% strategy recovery is the target — lower than trace encoder (85%) because narrative text is a noisier signal than search behaviour. If recovery exceeds 80%, the LLM generation prompt is too deterministic (all price_lex narratives sound identical). Add narrative generation variance if this occurs.

## Output Contract

```python
# Shape: (n_narratives, EMBEDDING_DIM)
# dtype: torch.float32
# One embedding per participant
# Also persisted in PersonaNarrative.embedding field
e_text = encoder(texts)  # [batch_size, 128]
```

## Known Constraints

- sentence-transformer model must be downloaded once via `sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")` — requires internet on first run; cached thereafter
- Do not fine-tune the sentence-transformer weights — frozen inference only
- If narrative generation was skipped (`--skip-narratives` in pipeline), `narratives.jsonl` will be empty — text encoder cannot run; handle gracefully with a clear error message
- Embedding quality depends on narrative diversity — if all same-persona narratives are nearly identical (cosine sim > 0.95), the text modality adds no information beyond the psychographic vector