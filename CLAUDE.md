# CLAUDE.md — customer-data-fusion

> Context governance: see `.claude/context/codified-context-principles.md`

---

## Project Identity

`customer-data-fusion` is a research prototype for a **Consumer Digital Twin (CDT)** with multimodal behavioural data. It synthesises decision process traces (MouseLab-style), transaction histories, psychographic surveys, and persona narratives, then trains a modular late-fusion encoder architecture to produce a per-consumer behavioural embedding. The stack is Python 3.14 / PyTorch / uv. All modality encoders are independently trainable; fusion is late by default with a designed upgrade path to early fusion.

---

## Repo Structure

```
schemas/          # Data contracts — stable interfaces between all modules
config/           # personas.yaml — generative root for all synthetic data
generator/        # Synthetic data pipeline (all modalities)
encoders/
  trace/          # Transformer encoder for process trace sequences
  transaction/    # Temporal encoder for purchase history
  text/           # Frozen sentence-transformer for persona narratives
  psychographic/  # MLP projector for survey vectors
fusion/           # Late fusion meta-learner; early fusion placeholder
evaluation/       # Strategy recovery, geometry, ablation, counterfactual
notebooks/        # EDA and validation notebooks
data/
  synthetic/      # Generated data (gitignored)
  calibration/    # Published benchmark data for simulator calibration
.claude/
  agents/         # Tier 2 specialist agents
  context/        # Tier 3 knowledge base
  skills/         # Project-specific Claude Code skills
```

---

## Key Files

| File | Purpose |
|---|---|
| `schemas/persona.py` | `PersonaConfig` dataclass — generative root |
| `schemas/trace.py` | `AcquisitionEvent`, `TrialRecord` dataclasses |
| `schemas/transaction.py` | `TransactionRecord` dataclass |
| `schemas/text.py` | `PersonaNarrative` dataclass |
| `schemas/psychographic.py` | `PsychographicVector` dataclass |
| `config/personas.yaml` | 7 persona archetype definitions |
| `generator/pipeline.py` | Orchestrates all modalities per participant |
| `generator/validate.py` | Cross-modal consistency checks |
| `encoders/trace/model.py` | Transformer sequence encoder |
| `fusion/meta_learner.py` | Late fusion: logistic regression / shallow MLP |
| `PRD.md` | Top-level product requirements document |

---

## Architecture Principles

- **Schemas are the contract.** All modules import from `schemas/`. Generator and encoders never import each other.
- **Personas are the generative root.** Every modality is a downstream sample from the same `PersonaConfig`. Cross-modal consistency is guaranteed at generation time.
- **Modality independence.** Each encoder trains in isolation with its own objective before fusion. Never couple encoder training.
- **Late fusion first.** Default architecture is independent encoders + meta-learner. Early fusion is a deliberate upgrade, not a default.
- **uv only.** Never use `pip` directly. All package management via `uv add` / `uv run`.
- **PYTHONPATH=.** All imports are absolute from repo root. No relative imports across module boundaries.

---

## Non-Negotiable Conventions

- Dataclasses in `schemas/` are **immutable** — no field additions without updating all downstream generators and encoders
- New modality = new schema first, then generator, then encoder. Never in reverse order.
- `data/synthetic/` and `data/calibration/` are gitignored — never commit generated data
- `.env` is gitignored — API keys (DeepSeek, Anthropic) live there only
- Experiment runs logged to MLflow (`mlruns/` — also gitignored)
- All tests use `BEADS_DB=/tmp/test.db` — never pollute production Beads DB

---

## Build / Run / Test

```bash
uv run python -m generator.pipeline     # Generate synthetic dataset
uv run python -m encoders.trace.train   # Train trace encoder
uv run pytest                           # Run test suite
uv run mlflow ui                        # Launch experiment tracker
bd ready                                # Check available tasks (Beads)
```

---

## Shell Safety

**Always use non-interactive flags** — `cp`, `mv`, `rm` may be aliased to `-i` mode, which hangs agents.

```bash
cp -f src dst      # NOT: cp src dst
mv -f src dst      # NOT: mv src dst
rm -f file         # NOT: rm file
rm -rf dir         # NOT: rm -r dir
```

---

## Issue Tracking (Beads)

This project uses `bd` (Beads) for all task tracking.

> Issues live in a local Dolt DB (`.beads/dolt/`); sync via `bd dolt push/pull`.
> `.beads/issues.jsonl` is a passive export, not the source of truth.
> See [SYNC_CONCEPTS.md](https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md) for anti-patterns.

- `bd prime` — run at session start for workflow context
- `bd ready` — check before starting any work
- `bd update <id> --claim` — claim a task before working on it
- `bd close <id>` — close when complete
- `bd remember "insight"` — persist cross-session notes
- Never use `TodoWrite`, `TaskCreate`, or markdown TODO lists — use `bd` exclusively
- Never create `MEMORY.md` files — use `bd remember` instead

---

## Agent Trigger Table

| File pattern | Specialist agent |
|---|---|
| `schemas/**` | `.claude/agents/schema-guardian/AGENT.md` |
| `config/personas.yaml`, `generator/**` | `.claude/agents/generator-specialist/AGENT.md` |
| `encoders/**` | `.claude/agents/encoder-specialist/AGENT.md` |
| `fusion/**` | `.claude/agents/fusion-specialist/AGENT.md` |
| `evaluation/**` | `.claude/agents/evaluation-specialist/AGENT.md` |

> Agents not yet created are placeholders — create on first observed failure in that domain.

---

## Context Documents

| Topic | Path |
|---|---|
| Project vision | `.claude/context/project-vision.md` |
| Engineering conventions | `.claude/context/engineering-conventions.md` |
| Governance principles | `.claude/context/codified-context-principles.md` |
| Persona archetypes | `.claude/context/persona-archetypes.md` *(create when personas.yaml is populated)* |
| Data contracts | `.claude/context/data-contracts.md` *(create when schemas are stable)* |
| Fusion architecture | `.claude/context/fusion-architecture.md` *(create before encoder training phase)* |


---

## Current Phase

**Phase 1 — Skeleton.** Repo structure initialised. Next: populate `schemas/` and `config/personas.yaml`, then build `generator/`.