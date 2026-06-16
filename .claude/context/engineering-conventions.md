# Engineering Conventions

## Current Version: 0.2

## Package Management

| Rule | Detail |
|---|---|
| Always use `uv` | Never `pip install` directly |
| Adding a dependency | `uv add <package>` |
| Adding a dev dependency | `uv add --dev <package>` |
| Running scripts | `uv run python -m <module>` (e.g. uv run python -m generator.pipeline --n 100) |
| Running tests | `uv run pytest` |

## Language and Runtime

- Python 3.14 (pinned via `.python-version`)
- Type hints required on all function signatures and dataclass fields
- `from __future__ import annotations` at top of every file (deferred evaluation)
- No `Any` type without an explaining comment
- Frozen dataclasses for all data structures (`frozen=True`)

## Library Preferences

### Core choices — do not substitute

| Purpose | Library | Notes |
|---|---|---|
| Deep learning | `torch` (PyTorch) | Not JAX, not TensorFlow |
| Text embedding | `sentence-transformers` | Frozen inference only; no fine-tuning at prototype stage |
| Data manipulation | `polars` | For tabular/feature data; use `.to_pandas()` only at skrub boundary |
| Numerical | `numpy` | Standard array ops |
| Config parsing | `pyyaml` | For `config/personas.yaml` |
| Data validation | `pydantic` | For config loading and API payloads |
| Environment variables | `python-dotenv` | `load_dotenv()` at entry points; never hardcode keys |
| Logging | `structlog` | Never `print()`, never `logging.basicConfig()` |
| Experiment tracking | `mlflow` | Local tracking server; `MLFLOW_TRACKING_URI=mlruns` |
| Testing | `pytest` | With `pytest-cov` for coverage |
| Linting / formatting | `ruff` | Single tool for both; replaces black + flake8 |
| Type checking | `pyright` | Runs via LSP in VS Code continuously; no manual invocation needed |
| Dimensionality reduction | `umap-learn` | For embedding visualisation |
| Visualisation | `matplotlib`, `seaborn` | Not plotly at prototype stage |

### Conditional / phase-specific

| Purpose | Library | When |
|---|---|---|
| Tabular feature engineering | `skrub` | Meta-learner input preparation phase; requires `.to_pandas()` conversion from polars |
| Hyperparameter optimisation | `optuna` | Encoder tuning phase; MLflow integration via `optuna-integration` |
| LLM API (text generation) | `anthropic` client | DeepSeek via OpenAI-compat endpoint; fallback to Anthropic |
| Notebook environment | `jupyter` + `ipykernel` | EDA and validation notebooks only |

## Environment Variables

Load at every entry point (pipeline scripts, training scripts, notebooks):

```python
from dotenv import load_dotenv
load_dotenv()
```

Keys in use: `DEEPSEEK_API_KEY`, `ANTHROPIC_API_KEY`. Never hardcode, never pass as CLI arguments, never log.

Use structlog throughout. Standard pattern:

```python
import structlog
log = structlog.get_logger()

log.info("trial_generated", participant_id=pid, trial_id=tid, n_acquisitions=n)
log.warning("validation_failed", participant_id=pid, check="price_consistency", delta=0.42)
log.error("generation_error", participant_id=pid, error=str(e))
```

- Always include `participant_id` as a bound key when inside a generation loop
- Use structured key=value pairs, not f-strings in the message
- INFO every 100 participants in batch generation loops
- WARNING on cross-modal consistency check failures (do not raise)
- ERROR on unrecoverable failures (do raise after logging)

## MLflow Convention

```python
import mlflow

with mlflow.start_run(run_name="trace_encoder_v1"):
    mlflow.log_params({"n_layers": 2, "n_heads": 4, "lr": 1e-3})
    mlflow.log_metric("strategy_recovery_acc", acc, step=epoch)
    mlflow.log_artifact("data/synthetic/traces.jsonl")
```

- One run per encoder training session
- Always log hyperparameters at run start
- Log metrics per epoch where applicable
- Tag runs with modality: `mlflow.set_tag("modality", "trace")`

## Data Files

- Format: **JSONL** for all synthetic data (one JSON object per line)
- Location: `data/synthetic/` (gitignored)
- One file per modality: `data/synthetic/traces.jsonl`, `data/synthetic/trials.jsonl`, `data/synthetic/transactions.jsonl`, `data/synthetic/psychographics.jsonl`, `data/synthetic/narratives.jsonl`
- Serialisation: use `dataclasses.asdict()` + `json.dumps()` — do not use pickle
- Enum values: serialise as `.value` (string), not the Enum object

```python
import json
from dataclasses import asdict

with open("data/synthetic/traces.jsonl", "a") as f:
    f.write(json.dumps(asdict(acquisition_event)) + "\n")
```

## Project Structure Rules

- All imports are absolute from repo root (`from schemas import PersonaConfig`)
- No relative imports across module boundaries
- `schemas/` imports nothing from the project — zero internal dependencies
- Generator and encoders never import each other — only from `schemas/`
- Test files mirror source structure: `tests/generator/test_trace_simulator.py`

### Correct cross-modal generation pattern

```python
from dotenv import load_dotenv
load_dotenv()

from schemas import PersonaConfig
from generator.persona_sampler import sample_persona
from generator.trace_simulator import simulate_session
from generator.transaction_simulator import simulate_transactions
from generator.psychographic_generator import generate_psychographic
from generator.text_generator import generate_narrative

# One PersonaConfig instance — shared across all modalities
config: PersonaConfig = sample_persona(archetype_id="price_lex", random_seed=42)

# All generators receive the same config — consistency guaranteed at source
traces = simulate_session(config)
transactions = simulate_transactions(config)
psychographic = generate_psychographic(config)
narrative = generate_narrative(config)
```

All four calls read from the same PersonaConfig instance: its strategy, transactions, psychographic,
and narrative fields respectively. A price-lexicographic persona produces
price-first traces, price-skewed transactions, HIGH price_consciousness,
and price-conscious narrative language

## Testing Conventions

- Unit tests for all simulator functions (dwell sampling, Payne index, strategy logic)
- Use pytest parametrize for archetype-level tests
- Synthetic data tests use `random_seed=42` for reproducibility
- Beads DB isolation: set env var `BEADS_DB` to a temp path in test environment
- Coverage target: 80% for `generator/` and `schemas/`; encoders tested via training smoke tests
- Encoder integration tests must isolate checkpoint writes: pass `save_path=tmp_path/"<encoder>.pt"` to `train()`. A test that calls `train()` without it overwrites the real `models/*.pt` with a 1-epoch fixture — silently (tests stay green; only `git status models/` reveals it). See `docs/post-mortems/test-isolation-postmortem.md`.

## Git Conventions

- Branch: `feature/<short-description>`, `fix/<short-description>`
- Commits: conventional commits format — `feat:`, `fix:`, `chore:`, `docs:`
- Never commit: `data/synthetic/`, `mlruns/`, `.env`, `__pycache__/`
- Co-authored-by tag suppressed (`includeCoAuthoredBy: false` in settings.json)

## What Not To Do

- Never use `print()` — use structlog
- Never use `pip` — use uv
- Never use `pickle` — use JSONL
- Never use `pandas` directly — use polars; convert to pandas only at skrub boundary
- Never invoke `pyright` manually — it runs via LSP; fix errors as they appear in VS Code
- Never use relative imports across modules
- Never store raw price values — always normalise to 0–1 percentile within category
- Never add methods to schema dataclasses — schemas are contracts, not behaviour
- Never generate modalities from independent parameter draws — always use the same `PersonaConfig`
- Never hardcode API keys — always via `.env` + `python-dotenv`
- Never let an integration test write to `models/*.pt` — `train()` takes `save_path`; pass a `tmp_path`-based path
- Never diagnose an artifact as "stale/wrong" without first running `git status` on it — a modified tracked file was changed *this session*, not always