# Generator Module Specification

## Purpose

Produces all synthetic modalities for the CDT prototype. All modalities for a given participant originate from a single `PersonaConfig` instance — this is the cross-modal consistency guarantee.

## Inputs

| Input | Location | Description |
|---|---|---|
| Archetype definitions | `config/personas.yaml` | 7 persona archetypes; generative root parameters |
| Persona schema | `schemas/persona.py` | `PersonaConfig` and nested param dataclasses |
| All modality schemas | `schemas/` | Output contracts for each generator |

## Outputs

Four JSONL files written to `data/synthetic/`:

| File | Schema | One record per |
|---|---|---|
| `data/synthetic/traces.jsonl` | `AcquisitionEvent` | Cell inspection event |
| `data/synthetic/trials.jsonl` | `TrialRecord` | Completed trial |
| `data/synthetic/transactions.jsonl` | `TransactionRecord` | Purchase event |
| `data/synthetic/psychographics.jsonl` | `PsychographicVector` | Participant |
| `data/synthetic/narratives.jsonl` | `PersonaNarrative` | Participant |

Serialisation: `dataclasses.asdict()` + `json.dumps()`. Enum values as `.value`. No pickle.

## Module Map

| File | Responsibility |
|---|---|
| `generator/persona_sampler.py` | Load `config/personas.yaml`; sample `PersonaConfig` instances with noise around archetype params |
| `generator/trace_simulator.py` | Simulate MouseLab-style acquisition sequences per trial per participant |
| `generator/transaction_simulator.py` | Generate 12-month purchase history from `TransactionParams` |
| `generator/psychographic_generator.py` | Generate fixed-width psychographic vector from `PsychographicParams` |
| `generator/text_generator.py` | Call DeepSeek API (Anthropic fallback) to generate persona narrative |
| `generator/validate.py` | Cross-modal consistency checks post-generation |
| `generator/pipeline.py` | Orchestrate all generators for a full synthetic dataset |

## Cross-Modal Consistency Invariant

**One `PersonaConfig` instance per participant, shared across all generators.** No modality samples from independent parameters.

```python
config = sample_persona(archetype_id="price_lex", random_seed=i)
traces      = simulate_session(config)       # reads config.strategy
transactions = simulate_transactions(config) # reads config.transactions
psychographic = generate_psychographic(config) # reads config.psychographic
narrative   = generate_narrative(config)     # reads config.narrative
```

Consistency relationships enforced by `generator/validate.py`:

| Persona type | Cross-modal signal |
|---|---|
| `price_lex` | price-first traces + low `price_paid_normalised` + `price_consciousness=HIGH` + price-focused narrative |
| `compensatory` | deep traces + high `maximiser_score` + high `involvement_score` |
| `brand_affect` | brand-first traces + concentrated `brand_tier` in transactions + high `brand_sensitivity` |
| `satisficer` | mid-depth traces + `maximiser_score < 0.4` |
| `low_involve` | random traces + `involvement_score < 0.3` + habitual purchase pattern |

## Trace Simulation Rules

### Dwell times
Sampled from log-normal, not uniform:
```python
dwell_ms = np.random.lognormal(mean=7.0, sigma=0.5)  # ~1100ms mean
```

### Payne Index
Computed per trial before constructing `TrialRecord`:

$$PI = \frac{A - W}{A + W}$$

where $A$ = alternative-wise transitions, $W$ = attribute-wise transitions in the acquisition sequence.

### Calibration targets

| Archetype | Payne Index | prop_cells_inspected | mean dwell_ms |
|---|---|---|---|
| `price_lex` | -0.6 to -0.8 | 0.15–0.30 | 800–1200 |
| `compensatory` | -0.2 to +0.2 | 0.60–0.85 | 1000–1800 |
| `satisficer` | -0.3 to -0.5 | 0.30–0.55 | 900–1400 |
| `brand_affect` | -0.7 to -0.9 | 0.10–0.20 | 600–1000 |
| `low_involve` | -0.1 to +0.1 | 0.20–0.45 | 400–800 |

### Fatigue and time pressure
- Trials 15+: reduce `inspection_depth` by one level
- `time_pressure=True`: multiply target cells inspected by `time_pressure_multiplier`
- `p_strategy_lapse` per trial: sample a random strategy deviation

## Text Generation

- Primary: DeepSeek via OpenAI-compatible endpoint (`DEEPSEEK_API_KEY`)
- Fallback: Anthropic API (`ANTHROPIC_API_KEY`)
- Target: 250–350 words; validate `word_count` post-generation
- PersonaNarrative embedding field is `None` at generation — populated by `encoders/text/embed.py`
- Batch all LLM calls; never call inside a trial simulation loop

## Validation Rules (`generator/validate.py`)

Failures logged at WARNING via structlog; do not raise exceptions.

1. `price_consciousness` consistent with `first_attribute` and transaction price distribution
2. `brand_sensitivity` consistent with `brand_tier` concentration in transactions
3. Narrative word count within 200–400 range
4. Transaction `price_paid_normalised` mean consistent with `price_sensitivity`
5. Payne Index per trial within expected range for archetype (±0.2 tolerance)

## Pipeline Scale

Prototype target: 1,000 participants × 20 trials × 3 categories.

Expected output sizes:
- `data/synthetic/traces.jsonl`: ~600k–1.2M records
- `data/synthetic/transactions.jsonl`: ~30k records (mean 30 per participant)
- `data/synthetic/psychographics.jsonl`: 1,000 records
- `data/synthetic/narratives.jsonl`: 1,000 records

Log progress via structlog at INFO every 100 participants.