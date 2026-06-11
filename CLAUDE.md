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
| `.claude/context/project-vision.md` | Project purpose, architecture, success criteria |
| **Schemas (data contracts)** | |
| `schemas/__init__.py` | All exports including `EMBEDDING_DIM = 128` |
| `schemas/persona.py` | `PersonaConfig` — generative root for all modalities |
| **Configuration** | |
| `config/personas.yaml` | 7 persona archetype definitions — source of all synthetic data |
| **Generator** | |
| `generator/SPEC.md` | Generator module spec, output schemas, calibration targets |
| `generator/pipeline.py` | Orchestrates all modalities per participant; supports `counterfactual_overrides` |
| `generator/validate.py` | Cross-modal consistency checks |
| **Encoders** | |
| `encoders/trace/SPEC.md` | Trace encoder spec — primary behavioural signal |
| `encoders/transaction/SPEC.md` | Transaction encoder spec — GRU, next brand_tier |
| `encoders/text/SPEC.md` | Text encoder spec — frozen sentence-transformer |
| `encoders/psychographic/SPEC.md` | Psychographic encoder spec — MLP, supervised |
| **Fusion** | |
| `fusion/SPEC.md` | Fusion layer spec — multi-task CE + NT-Xent |
| **Evaluation** | |
| `evaluation/counterfactual.py` | Option A counterfactual (archetype redistribution) |
| `evaluation/counterfactual_option_b.py` | Option B counterfactual (generator re-run) |

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
- All tests set env var `BEADS_DB` to a temp path — never pollute production Beads DB

---

## Build / Run / Test

```bash
uv run python -m generator.pipeline     # Generate synthetic dataset
uv run python -m encoders.trace.train   # Train trace encoder
uv run pytest                           # Run test suite
uv run python -m evaluation.run_probes  # Validate all 4 encoder probes (run after any encoder change)
uv run mlflow ui                        # Launch experiment tracker
bd ready                                # Check available tasks (Beads)
```

> **Counterfactual simulation (Option B):** See `evaluation/counterfactual_option_b.py`.
> Run: `uv run python -c "from evaluation.counterfactual_option_b import simulate_counterfactual; print(simulate_counterfactual('price_lex_0042', {'price_sensitivity': 0.99}))"`

> **Warning — long-running output:** `generator.pipeline` and encoder train scripts emit many log lines.
> Pipe through `| tail -50` when running interactively to avoid flooding terminal/context.
> Example: `uv run python -m generator.pipeline 2>&1 | tail -50`

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
- Never create MEMORY.md files — use `bd remember` instead

### Dependency type constraint (beads)

`bd dep add` has two legal dependency types — using the wrong one causes `bd close` to fail:

| Relationship | Command | Type flag |
|---|---|---|
| Task B must complete after task A | `bd dep add B A` | *(default `blocks`)* |
| Task belongs to an epic | `bd dep add <task> <epic> -t parent-child` | `-t parent-child` |

**`blocks` between a task and an epic is rejected** — epics are containers, not sequenced work items. Always use `-t parent-child` to link tasks to their containing epic. Using the wrong type requires `bd close --force` as a workaround.

### Parallel sub-agent worktree merge

Parallel sub-agents (`TaskCreate` with `isolation: "worktree"`) write their output to an isolated git worktree, not to the main working directory. After a sub-agent completes, copy result files back manually using `cp -f` from the worktree path printed in the sub-agent result into the corresponding module directory in the main repo.

Every bead that dispatches parallel sub-agents must include an explicit **Merge step** in its acceptance criteria listing which files to copy back.

---

## Agent Trigger Table

| File pattern | Specialist agent |
|---|---|
| `schemas/**` | `.claude/agents/schema-guardian/AGENT.md` |
| `config/personas.yaml`, `generator/**` | `.claude/agents/generator-specialist/AGENT.md` |
| `encoders/**` | `.claude/agents/encoder-specialist/AGENT.md` |
| `fusion/**` | `.claude/agents/fusion-specialist/AGENT.md` |
| `evaluation/**` | `.claude/agents/evaluation-specialist/AGENT.md` |
| `applications/**` | `.claude/agents/applications-specialist/AGENT.md` |

> All agents are fully implemented with domain knowledge and anti-patterns.

---

## Context Documents

| Topic | Path |
|---|---|
| Project vision | `.claude/context/project-vision.md` |
| Engineering conventions | `.claude/context/engineering-conventions.md` |
| Governance principles | `.claude/context/codified-context-principles.md` |
| Persona archetypes | `.claude/context/persona-archetypes.md` |
| Data contracts | `.claude/context/data-contracts.md` |
| Phase 1 post-mortem | `docs/post-mortems/phase1-postmortem.md` |
| Phase 2a post-mortem | `docs/post-mortems/phase2a-postmortem.md` |
| Phase 2a fix post-mortem | `docs/post-mortems/phase2a-fix-postmortem.md` |
| Fusion architecture | `.claude/context/fusion-architecture.md` |
| Generator diagnostics | `.claude/context/generator-diagnostics.md` |
| Phase 2b post-mortem | `docs/post-mortems/phase2b-postmortem.md` |
| PRD validation | `.claude/context/prd-validation.md` |
| Prototype summary | `.claude/context/prototype-summary.md` |
| Post-prototype capabilities | `.claude/context/new-capabilities.md` |

---

## Current Phase

**Prototype Complete.** All phases closed. Four modality encoders trained with CE + NT-Xent multi-task objective (epic 3eg). Late-fusion meta-learner achieves 100% strategy recovery (Tier 1) and 70.4% dropout-view recall@1 (140× over chance — individual-level CDT, not just archetype classification). PersonaConfig regression R² 0.79–0.96 on all 7 parameters. Both counterfactual options implemented: Option A (archetype redistribution) and Option B (generator re-run via `counterfactual_overrides`, epic sei). PRD validation: 2 PASS, 2 PARTIAL. See `.claude/context/prd-validation.md` for formal criterion assessment and `.claude/context/prototype-summary.md` for stakeholder summary.