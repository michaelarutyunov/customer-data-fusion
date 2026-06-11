# customer-data-fusion

A research prototype for a **Consumer Digital Twin (CDT)** — a per-consumer behavioural model that learns *how* a consumer makes decisions, not just *what* they prefer.

The system synthesises decision process traces, transaction histories, psychographic surveys, and persona narratives, then trains a modular late-fusion encoder architecture to produce a per-consumer 128-dimensional behavioural embedding. Given two independent partial views of the same consumer (with random modalities missing), the system identifies the correct individual among 210 candidates **70% of the time** — 140× above random chance.

## What It Demonstrates

| Capability | Result |
|---|---|
| Archetype recovery (fused) | **100%** accuracy across 7 decision-making archetypes |
| Individual identity (dropout-view recall@1) | **70.4%** (140× over chance) |
| PersonaConfig regression (fused R²) | **0.79–0.96** on all 7 continuous personality parameters |
| Counterfactual simulation | Two methods: archetype redistribution + individual generator re-run |

### The 7 Behavioural Archetypes

| Archetype | Decision Style |
|---|---|
| `price_lex` | Price-first scanner, rejects above-threshold options |
| `quality_lex` | Quality-first, less price-sensitive |
| `compensatory` | Trades off price vs. quality across all attributes |
| `satisficer` | Stops at "good enough" option |
| `brand_affect` | Brand-loyal, minimal comparison |
| `adaptive` | Strategy shifts by context |
| `low_involve` | Minimal information search, habitual |

## Architecture

```
config/personas.yaml (7 archetypes)
        ↓ + LatentDeviation (5-axis individual variation)
generator/pipeline.py
        ↓ produces 4 modalities per participant
┌───────────────────────────────────────────────┐
│  Independent Encoders (CE + NT-Xent)          │
│                                               │
│  trace (Transformer)  ─→ 128-dim             │
│  transaction (GRU)    ─→ 128-dim             │
│  text (sentence-TF)   ─→ 128-dim             │
│  psychographic (MLP)  ─→ 128-dim             │
└───────────────────────────────────────────────┘
        ↓ concatenate + L2-normalise
fusion/meta_learner.py (3-layer MLP)
        ↓
  128-dim CDT embedding  +  7-class archetype prediction
```

Each encoder is independently trainable. The fusion meta-learner uses a multi-task objective: **CE classification** for archetype recovery + **NT-Xent contrastive** for individual identity. The 128-dim second hidden layer is the CDT embedding.

## Quick Start

### Requirements

- Python ≥3.14
- [uv](https://docs.astral.sh/uv/) package manager
- ~4GB disk (models + synthetic data)

### Install

```bash
git clone https://github.com/michaelarutyunov/customer-data-fusion.git
cd customer-data-fusion
uv sync
```

### Run the Pipeline

```bash
# Generate synthetic dataset (1000 participants, ~1 min)
uv run python -m generator.pipeline 2>&1 | tail -20

# Train all four encoders
uv run python -m encoders.trace.train 2>&1 | tail -10
uv run python -m encoders.transaction.train 2>&1 | tail -10
uv run python -m encoders.text.train 2>&1 | tail -10
uv run python -m encoders.psychographic.train 2>&1 | tail -10

# Train fusion meta-learner
uv run python -m fusion.train 2>&1 | tail -10
```

### Run Demos

```bash
# Strategy recovery: how well does fusion predict archetype?
uv run python -m evaluation.strategy_recovery

# Counterfactual: what if a consumer became more price-sensitive?
uv run python -c "
from evaluation.counterfactual_option_b import simulate_counterfactual
result = simulate_counterfactual('price_lex_0042', {'price_sensitivity': 0.99})
print(f'Cosine distance shift: {result[\"cosine_distance_shift\"]:.3f}')
print(f'Meaningful (≥0.27): {result[\"cosine_distance_shift\"] >= 0.27}')
"

# All encoder probes (validate each modality)
uv run python -m evaluation.run_probes

# Launch experiment tracker
uv run mlflow ui
```

### Pre-trained Models

Trained model checkpoints are included in `models/` and synthetic data in `data/synthetic/`, so you can run evaluations without retraining.

## Project Structure

```
schemas/              # Data contracts — all modules import from here
config/personas.yaml  # 7 persona archetype definitions (generative root)
generator/            # Synthetic data pipeline (all modalities)
  pipeline.py         # Orchestrates generation; supports counterfactual_overrides
  trace_simulator.py  # MouseLab-style decision process traces
  transaction_simulator.py  # Purchase history
  psychographic_generator.py  # Survey vectors
  text_generator.py   # Persona narratives (LLM-generated)
encoders/
  trace/              # Transformer encoder for process trace sequences
  transaction/        # GRU encoder for purchase history
  text/               # Frozen sentence-transformer for narratives
  psychographic/      # MLP projector for survey vectors
fusion/
  meta_learner.py     # 3-layer MLP late fusion (CDT embedding)
  train.py            # CE + NT-Xent multi-task training
evaluation/
  strategy_recovery.py    # Fusion archetype recovery
  retrieval.py            # Individual identity (dropout-view recall@1)
  config_probe.py         # PersonaConfig regression (R² per parameter)
  counterfactual.py       # Option A: archetype redistribution
  counterfactual_option_b.py  # Option B: generator re-run
  geometry.py             # UMAP embedding geometry
  ablation.py             # Per-modality contribution
notebooks/
  03_fusion_validation.ipynb
  04_counterfactual_tests.ipynb
tests/                # 421 tests (schemas, generators, encoders, evaluation)
```

## Key Concepts

**Personas are the generative root.** Every modality for a participant is generated from the same `PersonaConfig + LatentDeviation` — guaranteeing cross-modal consistency at the individual level.

**Schemas are the contract.** All modules import from `schemas/`. Generator and encoders never import each other. Modifying a dataclass requires updating all downstream generators and encoders.

**Late fusion by design.** Each encoder trains in isolation. Fusion combines their outputs. Early fusion is an explicit upgrade, not the default.

## Counterfactual Simulation

Two complementary approaches:

**Option A — Archetype redistribution** (`evaluation/counterfactual.py`): Applies rules from persona definitions to predict archetype-level shifts under market changes (price increase, new entrant, brand removal). Fast — operates on existing embeddings.

**Option B — Individual simulation** (`evaluation/counterfactual_option_b.py`): Re-runs the generator with modified personality parameters for a specific consumer, re-encodes through frozen models, and measures the CDT embedding shift. Answers: "If *this specific consumer* became more price-sensitive, how would their behaviour change?"

## Documentation

| Topic | Path |
|---|---|
| Project vision and results | [`.claude/context/project-vision.md`](.claude/context/project-vision.md) |
| PRD criteria validation | [`.claude/context/prd-validation.md`](.claude/context/prd-validation.md) |
| Stakeholder summary | [`.claude/context/prototype-summary.md`](.claude/context/prototype-summary.md) |
| Fusion architecture | [`.claude/context/fusion-architecture.md`](.claude/context/fusion-architecture.md) |
| Persona archetypes | [`.claude/context/persona-archetypes.md`](.claude/context/persona-archetypes.md) |
| Generator spec | [`generator/SPEC.md`](generator/SPEC.md) |
| Encoder specs | [`encoders/*/SPEC.md`](encoders/trace/SPEC.md) |
| Fusion spec | [`fusion/SPEC.md`](fusion/SPEC.md) |

## License

Research prototype — not yet licensed for production use.
