# Phase 2a Fix Post-Mortem: Data, Encoders, and Infrastructure Clean-up

> Written: 2026-06-06
> Phase: Phase 2a Clean-up (epic customer-data-fusion-kpn)
> Scope: schemas/, generator/, encoders/, evaluation/, tests/, CLAUDE.md, .env

---

## 1. What Was Built

### Tasks Completed

| Task | ID | Type | Outcome |
|------|----|------|---------|
| Centralize PERSONA_LABELS | `oop` | task (P2) | Single source of truth in `schemas/__init__.py`; 8 local definitions replaced |
| Fix participant_id threading | `lvn` | task (P1) | Per-archetype unique IDs `{archetype}_{0000-0142}`; `--n-per-archetype` flag |
| Fix transaction encoder bug | `kvh` | bug (P1) | `.detach()` on dataset tensors; `best_state` unbound fixed |
| Migrate MLflow to SQLite | `30w` | task (P3) | `MLFLOW_TRACKING_URI=sqlite:///mlruns.db` |
| Document probe CI step | `3i8` | task (P3) | `run_probes` in CLAUDE.md; argparse + `--help`; dotenv loading |
| Regenerate dataset | `quh` | task (P1) | 1001 participants (143/archetype); 4 modalities verified; narratives via batch script |
| Retrain encoders + probes | `ho4` | task (P1) | Trace 35.57%, Transaction 62.59%, Text 100%, Psychographic 100% |

### Files Modified

| File | Change | Task |
|------|--------|------|
| `schemas/__init__.py` | Added `PERSONA_LABELS`, `PERSONA_TO_IDX`, embeddings flush | `oop` |
| `generator/pipeline.py` | Per-archetype participant IDs, `--n-per-archetype`, file flushing, `--n` made optional | `lvn`, `quh` |
| `generator/trace_simulator.py` | `participant_id: str \| None = None` param | `lvn` |
| `generator/transaction_simulator.py` | `participant_id: str \| None = None` param | `lvn` |
| `generator/psychographic_generator.py` | `participant_id: str \| None = None` param | `lvn` |
| `generator/text_generator.py` | `participant_id` params on `generate_narrative()` and `generate_narratives_batch()` | `lvn` |
| `generator/validate.py` | `participant_id: str \| None = None` param | `lvn` |
| `generator/persona_sampler.py` | `_NOISE_SCALE` 0.05 → 0.15 | `lvn` |
| `encoders/transaction/train.py` | `.detach()` in `TransactionSequenceDataset`, `best_state` init, `_` loop var | `kvh` |
| `encoders/psychographic/train.py` | Import `PERSONA_LABELS`/`PERSONA_TO_IDX` from schemas | `oop` |
| `encoders/text/embed.py` | Import `PERSONA_LABELS`/`PERSONA_TO_IDX` from schemas (re-exports) | `oop` |
| `evaluation/run_probes.py` | Import from schemas, dotenv loading, argparse, docstring with thresholds | `oop`, `3i8` |
| `evaluation/trace_probe.py` | Import from schemas | `oop` |
| `evaluation/transaction_probe.py` | Import from schemas | `oop` |
| `evaluation/text_probe.py` | Import from schemas | `oop` |
| `evaluation/psychographic_probe.py` | Import from schemas | `oop` |
| `evaluation/generate_probe_plots.py` | Import from schemas | `oop` |
| `tests/generator/test_pipeline.py` | Updated mock to accept `participant_id` kwarg | `lvn` |
| `CLAUDE.md` | Added `run_probes` to Build/Run/Test section | `3i8` |
| `.env` | Added `MLFLOW_TRACKING_URI=sqlite:///mlruns.db` | `30w` |
| `scripts/generate_missing_narratives.py` | New — batch narrative generation from psychographics data | `quh` |

### Dataset Generated

| File | Records | Unique participant_ids |
|------|---------|------------------------|
| `traces.jsonl` | 205,221 | 1,001 |
| `trials.jsonl` | 20,020 | 1,001 |
| `transactions.jsonl` | 39,362 | 1,001 |
| `psychographics.jsonl` | 1,001 | 1,001 |
| `narratives.jsonl` | 1,001 | 1,001 |

All 7 archetypes have exactly 143 participants each. Participant IDs follow the pattern `{archetype}_{0000-0142}` with per-archetype indexing. Validation failures: 0.

### Probe Results (1001-participant dataset, after wva fix)

| Encoder | Strategy Recovery | Threshold | Pass? |
|---------|------------------|-----------|-------|
| Psychographic | 100.00% ± 0.00% | >75% | ✅ |
| Text | 100.00% ± 0.00% | >70% | ✅ |
| Transaction | 62.59% ± 2.43% | >60% | ✅ |
| Trace | **95.02%** | >85% | ✅ |

Pre-fix trace result was 35.57% (NT-Xent) and 37.25% (cross-entropy). Root cause was a generator bug, not the objective — see G3.

Transaction encoder additional metrics: Pearson r(price × consciousness) = -0.8876. Text encoder additional metrics: intra-persona cosine sim 0.71 (>0.6 ✅), inter-persona cosine sim -0.10 (<0.4 ✅).

---

## 2. Errors and Resolutions

### E1 — Psychographics file not written during piped pipeline runs — Buffering
**What failed**: When `generator/pipeline.py` was run with `2>&1 | tail -N`, the `psychographics.jsonl` file remained at 0 bytes while all other modality files were written correctly. The file handle was open (confirmed via `/proc/PID/fdinfo/`) but the write position was 0 even after 64+ participants had been processed.

**Root cause**: Python's default block buffering for non-TTY file handles. Traces, trials, and transactions have many writes per participant (100+ events, 20 trials, 30+ transactions), so their 8KB buffers fill and flush frequently. Psychographics has exactly 1 write per participant (~500 bytes), requiring ~16 participants to fill the buffer. Narratives has 1 write per participant too but ~2,000 bytes per write, filling the buffer faster. When the pipeline was killed mid-run, unflushed psychographics data was lost.

The `2>&1 | tail -N` pipe was a contributing factor: when `tail` exits after receiving N lines, it closes the pipe. Subsequent writes to stderr (structlog output) trigger `SIGPIPE`, which can terminate the Python process before file buffers are flushed.

**Resolution**: Added explicit `fh.flush()` for all file handles after each participant's writes in `pipeline.py`. This ensures all modality files are written to disk atomically per participant, regardless of buffer sizes or pipe state.

**Prevention**: Any pipeline that writes multiple output files with significantly different write frequencies should flush all handles together after each logical unit of work. The `| tail -N` pattern should be avoided for long-running pipelines; redirect to a file instead.

### E2 — `load_dotenv()` picked up stale MLflow tracking URI — Environment
**What failed**: `run_probes.py` calls `load_dotenv()` which loaded `MLFLOW_TRACKING_URI=mlruns` instead of the `sqlite:///mlruns.db` value in `.env`. Probe runs failed with "filesystem tracking backend is in maintenance mode."

**Root cause**: Not fully diagnosed. The `.env` file contained the correct value (`sqlite:///mlruns.db`), but `load_dotenv()` returned the old directory path. Possible causes: (a) a parent-directory `.env` overriding the project one, (b) `uv run` environment variable isolation, or (c) an environment variable already set in the shell session taking precedence over `.env` (dotenv does not override existing env vars by default).

**Resolution**: Worked around by passing `MLFLOW_TRACKING_URI=sqlite:///mlruns.db` explicitly on the command line. The `.env` file was verified to contain the correct value. Long-term: investigate whether `load_dotenv(override=True)` is needed, or whether `uv run` is injecting environment variables from `pyproject.toml`.

**Prevention**: Use `load_dotenv(override=True)` in scripts that must pick up `.env` values regardless of the shell environment. Document the expected `MLFLOW_TRACKING_URI` value in CLAUDE.md.

### E3 — `--n` was required even when `--n-per-archetype` was supplied — CLI Design
**What failed**: Running `python -m generator.pipeline --n-per-archetype 143` failed with "the following arguments are required: --n". The `--n` argument was `required=True` in argparse.

**Root cause**: The `--n-per-archetype` flag was added to `run_pipeline()` but the CLI parser still required `--n`. The two arguments are mutually exclusive in intent (you specify either total N or per-archetype N), but argparse doesn't know that.

**Resolution**: Made `--n` optional (`default=None`) and added validation: if neither `--n` nor `--n-per-archetype` is provided, print an error and exit. When `--n-per-archetype` is given, `n` is derived as `n_per_archetype × len(active_archetypes)`.

### E4 — Participant ID indexing scheme mismatch with bead acceptance criteria — Spec Drift
**What failed**: Initial implementation used global indexing (`{archetype}_{i:04d}` where `i` is the global participant index 0–1000). The bead's acceptance criterion "participant_id 'price_lex_0000' and 'price_lex_0142' both present" implied per-archetype indexing (0–142 within each archetype).

**Root cause**: The bead example `price_lex_0142` is only reachable with per-archetype indexing (143 participants per archetype, indices 0000–0142). With global indexing, the 143rd price_lex participant would be at global index 994 → `price_lex_0994`.

**Resolution**: Switched to per-archetype indexing using a `per_archetype_counter` dict that tracks how many participants have been generated for each archetype. This produces IDs like `price_lex_0000` through `price_lex_0142`.

**Prevention**: Bead acceptance criteria should explicitly state the indexing scheme, not just imply it through examples. Write: "participant_id format is `{archetype}_{per_archetype_index:04d}` where per_archetype_index is 0-based within each archetype."

---

## 3. Deviations from SPEC

### D1 — `--n` is now optional when `--n-per-archetype` is provided
**SPEC said** (pipeline.py docstring): `--n` is the sole way to specify participant count.

**What was implemented**: `--n` is optional; `--n-per-archetype` derives `n` as `n_per_archetype × len(archetypes)`. If neither is provided, the pipeline exits with an error.

**Why**: The `--n-per-archetype` flag produces balanced datasets without manual arithmetic. The bead required it for the 143-per-archetype × 7 = 1001 dataset.

### D2 — `PERSONA_LABELS` order changed from probe-file convention to schemas convention
**SPEC said** (implicitly via probe files): `PERSONA_LABELS = ["price_lex", "compensatory", "satisficer", "brand_affect", "quality_lex", "adaptive", "low_involve"]`.

**What was implemented**: Centralized in `schemas/__init__.py` as `["price_lex", "quality_lex", "compensatory", "satisficer", "brand_affect", "adaptive", "low_involve"]`.

**Why**: The order in `schemas/` groups archetypes logically (price_lex and quality_lex together as lexicographic variants). The probe files had a different ordering. Since `PERSONA_TO_IDX` is derived from `PERSONA_LABELS` and the integer labels are only used as classification targets (where numeric values don't matter), the order change has no functional impact. All tests use symbolic lookups (`PERSONA_TO_IDX["price_lex"]`) rather than hardcoded integers.

**Risk**: If any external code (notebooks, analysis scripts) hardcodes integer labels, they will break. A grep confirmed no such hardcoding exists in the repo.

### D3 — Narratives generated via separate batch script, not pipeline
**SPEC said** (quh bead): `PYTHONPATH=. uv run python -m generator.pipeline --n-per-archetype 143` generates all 5 modalities including narratives.

**What was implemented**: Pipeline run with `--skip-narratives` for the 4 fast modalities; narratives generated separately via `scripts/generate_missing_narratives.py` which reads `psychographics.jsonl` and calls the LLM API for each participant.

**Why**: Generating 1001 narratives via the pipeline would take ~2 hours (8 narratives/min, bottlenecked by LLM API latency). Running with `--skip-narratives` takes ~30 seconds for the other 4 modalities, allowing immediate verification of participant counts, archetype distribution, and ID formats. The batch script can be run independently, resumed if interrupted, and doesn't block encoder retraining (which only needs non-narrative modalities for trace, transaction, and psychographic encoders).

---

## 4. Context Infrastructure Gaps

### G1 — No canonical encoder checkpoint path registry
**Gap**: Each encoder training script saves its checkpoint independently (`models/trace_encoder.pt`, `models/transaction_encoder.pt`). The probe scripts sometimes load from these paths, sometimes train from scratch. There's no single place that maps encoder name → checkpoint path.

**Status**: Unchanged from Phase 2a. The `kvh` task added a checkpoint save to `encoders/transaction/train.py` (matching the trace encoder pattern), but the registry gap remains. The text and psychographic encoders don't save checkpoints at all — they train from scratch in every probe run.

**Fix** (deferred to Phase 2b): Add a `CHECKPOINT_PATHS` dict to `schemas/__init__.py` or `evaluation/probe.py` mapping modality names to expected checkpoint paths. All encoder training scripts should save to these paths; all probe scripts should prefer loading over retraining.

### G2 — `load_dotenv()` behaviour is unreliable across invocation patterns
**Gap**: The `.env` file contains `MLFLOW_TRACKING_URI=sqlite:///mlruns.db`, but `load_dotenv()` does not consistently pick it up. When running via `uv run python -m evaluation.run_probes`, the environment variable is sometimes the old `mlruns` value.

**Status**: Worked around with explicit `MLFLOW_TRACKING_URI=...` prefix on commands, but the root cause is not fixed. This affects all scripts that call `load_dotenv()`.

**Fix** (deferred): Investigate whether `uv run` injects environment variables from `pyproject.toml` or whether a parent-directory `.env` file takes precedence. Consider using `load_dotenv(override=True)` or a centralized config loader that reads `.env` once at project import time.

### G3 — Trace encoder probe blocked by generator flaw, not objective — RESOLVED
**Gap (original)**: The Phase 2a post-mortem (R3) hypothesized that the 35.57% trace recovery was due to only 7 participants. The kpn fix increased to 1001 participants, but trace recovery was unchanged at 35.57%.

**Investigation chain (2026-06-06/07)**:
1. Bead `6yl`: replaced NT-Xent with supervised cross-entropy → 37.25% (no improvement). Objective was not the bottleneck.
2. Scalar probe diagnostic: mean-pooled `prop_cells_inspected`, `total_acquisitions`, `payne_index` per participant → **63% ceiling**. `price_lex` achieved 0% recall even from gold scalars.
3. Root cause: `_simulate_lexicographic` used `_build_mixed_sequence(p_dimensional=0.82)` — biased transitions but sampled all attributes. `price_lex` was indistinguishable from other archetypes in trace scalar space.
4. Secondary bug: `simulate_session` passed `participant_id` (e.g., `"price_lex_0042"`) as `persona_id` to `_generate_sequence`. `_ARCHETYPE_DEPTH_FRACTION` uses archetype keys — overrides never fired.
5. Bead `wva`: fixed both bugs. `_simulate_lexicographic` now scans only the `first_attribute` column (PI = -1.0, prop = 1/n_attrs). Extended scalar probe with attribute-frequency features → **91.5% ceiling**.
6. Retrained trace encoder → **95.02% strategy recovery**. All 7 archetypes pass ≥80%.

**Status**: Resolved. Trace encoder probe result: 95.02% (was 35.57%).

---

## 5. Transaction Encoder Bug Root Cause Analysis

The double-backward bug (E4 from Phase 2a post-mortem, fixed in `kvh`) was definitively root-caused:

**Symptom**: `RuntimeError: Trying to backward through the graph a second time` on the first call to `loss.backward()` in `train_epoch()`.

**Root cause**: `TransactionSequenceDataset.__init__()` calls `vocab.encode_sequence()` which calls `TxVocabulary.to_token_vector()`. This method does `nn.Embedding` lookups on embedding weights that have `requires_grad=True`. The resulting token tensors inherit `requires_grad=True` and carry a live `grad_fn=<StackBackward0>`. These non-leaf tensors are stored in `self.sequences` and returned by `__getitem__`.

On the first training epoch, the computation graph from the embedding lookup is traversed during `loss.backward()`. After backward completes, the graph's saved intermediate values are freed. On the second epoch, the DataLoader returns the same stored tensors — but their `grad_fn` now references freed memory, triggering the "double backward" error.

**Fix**: One line — `.detach()` the token sequence in `TransactionSequenceDataset.__init__()`:
```python
self.sequences.append((token_seq.detach(), target_tensor, len(input_txs)))
```

**Why this wasn't caught earlier**: The bug only manifests when the dataset is iterated more than once (epoch ≥ 2). The Phase 2a implementation was tested with a single-epoch dry run. The `TxVocabulary` embedding weights being part of `encoder.parameters()` but never used in the forward pass (tokens are pre-computed) was an architectural smell that should have been caught in review.

**Lesson**: Dataset tensors must never carry gradient history. Any tensor stored in a `Dataset` and returned by `__getitem__` should be a leaf tensor (created by `torch.tensor()`, `torch.zeros()`, or explicitly `.detach()`ed).

---

## 6. Assumptions Validated / Invalidated

### A1 — 1001 diverse participants would fix trace encoder recovery — ❌ Invalidated
**Evidence**: Trace strategy recovery is 35.57% on 1001 participants — identical to the 37.75% on 7 participants. The NT-Xent contrastive objective does not benefit from more samples per class when the number of classes remains 7.

**Implication**: The contrastive objective is the bottleneck, not data diversity. Switching to supervised classification (bead `6yl`) is the correct path forward.

### A2 — Transaction encoder would train without errors after fix ✓ Confirmed
**Evidence**: Transaction encoder completed 30/30 epochs (no early stopping) on the 1001-participant dataset. Strategy recovery 62.59% exceeds the 60% threshold. Pearson r(price, consciousness) = -0.8876 confirms the encoder captures the expected inverse relationship.

### A3 — Text and psychographic encoders would not regress on new data ✓ Confirmed
**Evidence**: Both achieve 100% strategy recovery on the 1001-participant dataset, matching their Phase 2a results. The text encoder's intra-persona cosine sim of 0.71 and inter-persona sim of -0.10 are nearly identical to Phase 2a values (0.71 and -0.03), confirming embedding quality is stable.

### A4 — Per-archetype noise increase (0.05 → 0.15) would not break validation ✓ Confirmed
**Evidence**: Zero validation failures across 1001 participants despite the 3× increase in noise scale. The validation checks (price consciousness range, brand sensitivity, Payne Index range) have sufficient tolerance to accommodate realistic individual variation.

### A5 — `PERSONA_LABELS` centralization would be a transparent refactor ✓ Confirmed
**Evidence**: 369 tests pass without modification (except the pipeline mock update for the new `participant_id` kwarg). The `encoders.text.embed` re-export keeps existing imports working. No test checks hardcoded integer label values.

---

## 7. Recommendations for Phase 2b

### R1 — ✅ COMPLETE: Supervised trace encoder objective + generator fix
NT-Xent replaced with cross-entropy (bead `6yl`). Generator price_lex bug fixed (bead `wva`). All 4 encoders now pass their thresholds. Trace encoder: 95.02%.

### R2 — Fix `load_dotenv()` reliability before Phase 2b training runs
The MLflow tracking URI intermittently fails to load from `.env`. All Phase 2b encoder and fusion training scripts will need reliable MLflow logging. Either switch to `load_dotenv(override=True)` or add a centralized config loader that guarantees `.env` values are picked up.

### R3 — Add a checkpoint registry before fusion training
The fusion meta-learner needs to load embeddings from all 4 encoders. Without a canonical checkpoint path registry, the fusion training script will need to hardcode paths or retrain encoders from scratch each time. Add `CHECKPOINT_PATHS: dict[str, Path]` to `schemas/__init__.py` or a new `evaluation/checkpoint_registry.py`.

### R4 — Remove the `| tail -N` pattern from long-running pipeline commands
The `2>&1 | tail -N` pattern caused the psychographics buffering issue (E1) and makes debugging background tasks harder (only last N lines visible). Redirect to a file instead:
```bash
# Good
PYTHONPATH=. uv run python -m generator.pipeline --n-per-archetype 143 > /tmp/pipeline.log 2>&1 &

# Avoid
PYTHONPATH=. uv run python -m generator.pipeline --n-per-archetype 143 2>&1 | tail -10
```

### R5 — Write the fusion architecture context document
The CLAUDE.md references `.claude/context/fusion-architecture.md` as "create before Phase 2b." Phase 2b is the next epic (67a). This document should be the first task in that epic.

### R6 — Complete narrative quality validation
All 1001 narratives were generated successfully (0 failures), but no quality check beyond word count was performed. Before the text encoder is used in fusion, spot-check 10–20 narratives for coherence, persona consistency, and category relevance.

### R7 — Consider whether 100% strategy recovery is too high
Both text and psychographic encoders achieve 100% strategy recovery. While this passes the threshold, it raises a question: are the encoders learning behavioural patterns, or just memorizing persona identity? If the embeddings are perfectly separable by persona, they may not contribute unique variance in the fusion layer (each modality would encode the same 7-class signal). The fusion ablation evaluation (67a.6) should include a "single-modality vs all-modality" comparison to check for redundancy.
