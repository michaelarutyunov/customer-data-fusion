# Schema-Update Epic — Post-Mortem

> Epic `customer-data-fusion-1y0` (schema-update): extend the CDT to 6 modalities (clickstream + campaign) with variable-modality late fusion.
> Branch: `feature/1it-thread-participant-id`. Date: 2026-06-14.
> Beads: `1it`, `33x`, `syu`, `fso`, `hcx` (closed); `2io` (open — full 6-modality run).

## Outcome

5 of 6 immediate beads closed. Generators now produce per-participant-attributable clickstream/campaign data; both new encoders train; `fusion/train.py` is a variable-modality loader with an `n_modalities` assert. **5-modality dry run (text dropped): 95.0% archetype recovery.** The full 6-modality run (`2io`) is deferred pending narrative regeneration.

The epic's dominant failure mode was **schema/generator drift**: the data layer and generators evolved across phases, but the encoder train scripts, tests, and checkpoints were not kept in sync. Almost every issue below is a symptom of that one root cause.

## Error catalog (what surfaced this session)

| # | Issue | Where | Root cause |
|---|---|---|---|
| 1 | clickstream/campaign events keyed by archetype-level `customer_id` (`config.persona_id`), not individual | `clickstream_generator.py:393`, `campaign_generator.py:216` | generators never threaded `participant_id` (pipeline passes it to transactions but not these two). All participants in an archetype collapsed to one id → per-participant training impossible. (`1it`) |
| 2 | `--skip-narratives` truncated `narratives.jsonl` | `pipeline.py:295` | the file handle was opened `"w"` before the skip check (`:468`); skipping still wiped costly LLM output. Fixed: only open the handle when generating. |
| 3 | `Schema(**record)` crashed on `month` kwarg | `encoders/{trace,transaction,psychographic}/train.py` + `fusion/train.py` trace loader (4 sites) | generator writes a `month` field for temporal partitioning; the immutable dataclasses don't model it. Train scripts did `Schema(**json.loads(line))` passing every field. Fixed with `k in __dataclass_fields__` filtering. |
| 4 | fusion loader hardcoded to 4 modalities while claiming variable-N | `fusion/train.py` `load_encoders` (36–97) + `generate_embeddings` (193–330) | the meta-learner + `_MODALITIES` line were made variable (rq2), but the embedding *production* path stayed rigid. A flexible consumer fed by a rigid producer silently degrades to the producer's count. (`hcx`) |
| 5 | `_MODALITIES` leaked `participant_ids` | `fusion/train.py:527` | filter excluded only `"labels"`; `participant_ids` is a list (not a tensor) and would corrupt `torch.stack`/`torch.cat`. Fixed by excluding both + assert. |
| 6 | clickstream carries near-zero archetype signal | `clickstream_generator.py` design | transitions perturbed by within-archetype `config.latent`, not archetype-keyed. Raw baseline 0.15 (= chance). `fso` added archetype-keyed intent priors → raw 0.40, encoder 0.23→0.52. |
| 7 | MLflow file-store "maintenance mode" exception in train `__main__` | all encoder train scripts | `MLFLOW_TRACKING_URI` lives in `.env`; scripts that don't `load_dotenv()` hit the exception. Fixed by adding `load_dotenv(override=True)` in `__main__`. |
| 8 | `counterfactual_option_b` test failures (6) | `tests/evaluation/test_counterfactual_option_b.py` | `_load_fusion_model()` load_state_dict fails: saved fusion checkpoint is 4-modality/512-dim, meta-learner defaults to 6-modality/768 (rq2). Will resolve when `2io` retrains fusion at 6 modalities. |
| 9 | transaction test imports a renamed class | `tests/encoders/transaction/test_transaction_encoder.py` | imports `TransactionSequenceDataset`; `train.py` defines `SplitTransactionDataset` (renamed during the split-history refactor). Pre-existing, out of epic scope. |

## Lessons (reusable)

1. **When a system claims variable-N support, assert the actual N in the run.** Dynamic counting (`[k for k in embeddings ...]`) masks a static producer. The `assert n_modalities == len(encoders)` + shape assert is the only thing that forces producer and consumer to agree. Generalize: any "configurable count" feature needs a runtime assertion of the configured value.

2. **Cross-modal `customer_id`/`participant_id` join mismatches surface only at fusion.** Every per-module test passed (generator emits valid records; encoder instantiates; schema validates), yet the archetype-vs-individual key split was fatal at the fusion layer, where individual identity is the whole point. Add a cross-modal key-consistency test (already exists for trace↔psychographics; extend to clickstream/campaign — done in `1it`).

3. **Construct schema dataclasses with field-filtering, never `Schema(**record)`.** The generator legitimately writes metadata (`month`) the schemas don't model. `Schema(**{k: v for k, v in rec.items() if k in Schema.__dataclass_fields__})` is the project pattern — fusion's loader and all train scripts must use it.

4. **"Encoder module ✓" ≠ "encoder trainable ✓".** Beads `53o`/`sf2` were closed as encoder modules with `train.py` listed in their own acceptance, yet `train.py` was never written. Scope boundaries leak between phases. A "module complete" bead must not count a file it didn't deliver.

5. **A modality's archetype-recovery ceiling is set by generator archetype-determinism, not encoder capacity.** Campaign (archetype-keyed `_DISPATCH_WEIGHTS`) reached 0.71; clickstream (within-archetype latent) was at chance until archetype-keyed intent priors were added. If a new modality's individual-archetype probe is at chance, fix the generator before tuning the encoder — and verify with a raw-feature baseline (mean-pooled tokens → LogisticRegression) to separate "no signal" from "training bug". The chain `trained > raw > chance` proves labels are correct; all-equal-to-chance is ambiguous.

6. **`--skip-X` flags must be no-ops on X's output, never mutations.** "Skip narrative generation" truncated the file. A skip flag that touches state it's supposed to skip is a footgun.

7. **Train entrypoints must self-load their env.** `MLFLOW_TRACKING_URI` is in `.env`; `load_dotenv()` in `__main__` is now the project pattern (matches `evaluation/run_probes.py`).

## Follow-ups

- `2io` — full 6-modality fusion run (regenerate 1001 narratives, retrain fusion at `n_modalities=6`, evaluate recall@1 / R²). Also closes issue #8 (counterfactual_option_b).
- Fix `tests/encoders/transaction/test_transaction_encoder.py` import (`TransactionSequenceDataset` → `SplitTransactionDataset`) — issue #9, pre-existing, out of epic scope.
- `scripts/check_doc_drift.py` is referenced in `CLAUDE.md` but does not exist — either implement it or remove the reference.
- Optional: push clickstream recovery past 0.60 via archetype-keyed *transition* matrices (stronger than intent priors) if `2io`'s recall@1 shows clickstream underperforms.
