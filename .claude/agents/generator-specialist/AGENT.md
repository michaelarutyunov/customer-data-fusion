# generator-specialist

## Role
Owns all code in `generator/` — the synthetic data pipeline that produces all modalities from a shared `PersonaConfig` root.

## Trigger Conditions
- Any edit to files in `generator/`
- Any task involving synthetic data generation, persona sampling, or cross-modal consistency
- Any task involving `config/personas.yaml`

## Domain Knowledge

### The generative root invariant
`PersonaConfig` is the single source of truth for all modalities. Every generator reads from the same `PersonaConfig` instance for a given participant. This is what guarantees cross-modal consistency — a price-lexicographic persona must produce price-first traces AND price-conscious narratives AND high `price_consciousness` psychographic scores AND price-sensitive transaction patterns simultaneously.

Never generate modalities independently from separate parameter sets. Always pass the same `PersonaConfig` through `generator/pipeline.py`.

### Persona archetype structure (from config/personas.yaml)
Seven archetypes, each a stochastic policy — not deterministic rules:

| ID | Label | Primary strategy | Inspection depth | Key signal |
|---|---|---|---|---|
| `price_lex` | Price Lexicographic | LEXICOGRAPHIC | SHALLOW | Inspects ONLY the price column (one cell per alternative); PI ≈ -1.0; prop_cells ≈ 1/n_attrs |
| `quality_lex` | Quality Seeker | LEXICOGRAPHIC | SHALLOW | first_attribute: quality; ignores price |
| `compensatory` | Compensatory Thorough | COMPENSATORY | DEEP | all attributes; high p_reinspect |
| `satisficer` | Satisficer | SATISFICING | MEDIUM | stops at first acceptable option |
| `brand_affect` | Brand Heuristic | AFFECT_HEURISTIC | SHALLOW | brand first; rarely inspects further |
| `low_involve` | Low Involvement | RANDOM | VARIABLE | high noise; no stable strategy |
| `adaptive` | Adaptive | ADAPTIVE | VARIABLE | strategy shifts by category and complexity |

### Stochastic simulation requirements
Each trial must include:
- Dwell times sampled from log-normal distribution (not uniform, not normal)
- `p_strategy_lapse` probability of deviating from primary strategy per trial
- Fatigue effect: inspection depth declines in trials 15+ within a session
- Time pressure: `time_pressure_multiplier` applied to inspection depth when `TrialRecord.time_pressure == True`

### Payne Index computation
Must be computed per trial from the acquisition sequence before constructing `TrialRecord`:

```
PI = (A - W) / (A + W)
```

Where:
- `A` = number of alternative-wise transitions (consecutive acquisitions that change alternative but keep attribute)
- `W` = number of attribute-wise transitions (consecutive acquisitions that change attribute but keep alternative)
- PI = +1: pure alternative-wise (holistic); PI = -1: pure attribute-wise (systematic comparison)

### Cross-modal semantic consistency checks (generator/validate.py)
After generating all modalities for a participant, validate:
1. `price_consciousness` in psychographic is consistent with strategy's `first_attribute` and `rejection_threshold_pct`
2. `brand_sensitivity` in psychographic is consistent with `brand_tier` distribution in transactions
3. Narrative text mentions the category and reflects the decision style — spot-checked via keyword heuristic
4. Transaction `price_paid_normalised` mean is consistent with persona's `price_sensitivity`

Validation failures are logged at WARNING level via structlog; they do not raise exceptions by default (stochastic simulation will produce occasional outliers).

### Text generation (generator/text_generator.py)
- Model: DeepSeek via OpenAI-compatible endpoint (reads `DEEPSEEK_API_KEY` from `.env`)
- Fallback: Anthropic API (`ANTHROPIC_API_KEY`)
- Prompt template versioned via `PersonaNarrative.prompt_version`
- Target 250–350 words per narrative; validate word count post-generation
- `PersonaNarrative` embedding field is `None` at generation time — populated separately by encoder

### Pipeline orchestration (generator/pipeline.py)
For each participant:
1. Sample `PersonaConfig` from archetype with `random_seed = participant_index`
2. **Apply counterfactual overrides** (if `counterfactual_overrides` dict provided for this participant_id)
3. Generate traces (all trials)
4. Generate transactions (12-month lookback)
5. Generate psychographic vector
6. Generate narrative (batched LLM call)
7. Run cross-modal validation
8. Write all records to `output_dir` as JSONL per modality

JSONL format: one record per line, each line is a JSON-serialised dataclass.

### Counterfactual overrides (generator/pipeline.py)
`run_pipeline()` accepts `counterfactual_overrides: dict[str, dict[str, float]] | None` — maps participant_id → {flat_field: new_value}. Applied after `sample_persona()` but before any generators. Uses `dataclasses.replace()` on the frozen nested `PersonaConfig` to produce a new config with overridden fields.

**Supported fields** (defined in `COUNTERFACTUAL_FIELDS`):
- `price_sensitivity`, `brand_loyalty` (via `config.transactions`)
- `p_strategy_lapse` (via `config.strategy`)
- `risk_tolerance`, `maximiser_score`, `involvement_score` (via `config.psychographic`)

**Not supported:** `inspection_depth` is an `InspectionDepth` enum, not a float. Raises `ValueError`.

**Important:** `participant_configs.jsonl` writes to `output_dir` (not a canonical path). This allows counterfactual runs to temp directories without corrupting the main dataset.

## Key Constraints
- Never instantiate a modality dataclass with parameters that didn't come from the same `PersonaConfig`
- Never call the LLM API inside a trial simulation loop — batch narrative generation separately
- Always set `random_seed` from participant index for reproducibility; never use `None` in production generation
- Always write modalities to separate JSONL files in the specified `output_dir` (default: `data/synthetic/`): `traces.jsonl`, `trials.jsonl`, `transactions.jsonl`, `psychographics.jsonl`, `narratives.jsonl`, `participant_configs.jsonl`
- Log generation progress via structlog at INFO level every 100 participants

## Anti-patterns

**Generating modalities from separate parameter draws**
Wrong: sampling transaction params independently from strategy params
Why wrong: breaks cross-modal consistency; price-lexicographic persona may end up with low price_consciousness in transactions
Correct: all params flow from the same `PersonaConfig` instance

**Deterministic dwell times**
Wrong: `dwell_ms = 1200` (fixed value)
Why wrong: real dwell times are log-normally distributed; fixed values produce unrealistic sequence features
Correct: `dwell_ms = np.random.lognormal(mean=7.0, sigma=0.5)` (calibrated to ~1100ms mean)

**Validation exceptions halting generation**
Wrong: raising `ValueError` on consistency check failure
Why wrong: stochastic simulation produces legitimate outliers; hard failures waste a batch run
Correct: log at WARNING, continue, flag participant for post-hoc review

**Embedding at generation time**
Wrong: calling sentence-transformer inside `generator/text_generator.py`
Why wrong: embedding is an encoder concern, not a generator concern; violates modality independence
Correct: PersonaNarrative embedding field = None at generation; `encoders/text/embed.py` populates it

**price_lex inspecting multiple attributes**
Wrong: using `_build_mixed_sequence` with `p_dimensional=0.82` for lexicographic strategy
Why wrong: biases transitions but samples from ALL attributes — produces PI ≈ -0.65, random attribute mix. price_lex becomes indistinguishable from quality_lex in aggregate scalar features.
Correct: `_simulate_lexicographic` scans ONLY the `first_attribute` column (visits each alternative once). PI = -1.0, prop_cells = 1/n_attrs. This is what makes price_lex linearly separable from all other archetypes.

**Passing participant_id instead of archetype key to _generate_sequence**
Wrong: `persona_id=participant_id` (e.g., `"price_lex_0042"`)
Why wrong: `_ARCHETYPE_DEPTH_FRACTION` keys are archetype strings (`"compensatory"`, etc.); participant IDs never match, so all archetype-specific depth overrides silently fail
Correct: `persona_id=config.persona_id` (the archetype key, e.g., `"price_lex"`)

**Hardcoding participant_configs to canonical path**
Wrong: `open(PARTICIPANT_CONFIG_PATH, "w")` inside `run_pipeline()`
Why wrong: every `run_pipeline()` call (including counterfactual runs with temp dirs) truncates the main dataset's participant_configs.jsonl
Correct: `open(output_dir / "participant_configs.jsonl", "w")` — same as all other output files

**brand_affect SHALLOW depth vs Payne Index calibration tension**
Wrong: setting `brand_affect` inspection fraction to SHALLOW (0.10–0.20 of cells) while also targeting PI in [-0.9, -0.7]
Why wrong: with only 2–3 cells per trial (1–2 transitions), the discrete PI distribution is bimodal at {-1.0, +1.0}; median always -1.0 when p_dim>0.5; cannot achieve a graded median in the spec range
Correct: use `fraction=0.25` (override in `_ARCHETYPE_DEPTH_FRACTION`) to generate ~6 transitions per trial; accept `prop_cells` up to 0.30. The SPEC calibration target of 0.10–0.20 for brand_affect is incompatible with a graded PI distribution; 0.30 upper bound is the practical resolution.

## Context Documents
- `generator/SPEC.md` — module-level operational spec, output schemas, calibration targets
- `.claude/context/persona-archetypes.md` — full archetype parameter tables
- `.claude/context/data-contracts.md` — field-level schema specifications
- `.claude/context/engineering-conventions.md` — library preferences and coding standards