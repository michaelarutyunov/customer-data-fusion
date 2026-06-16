# Test-Isolation / Checkpoint-Corruption Post-Mortem

> Date: 2026-06-16. Commits: `a1a502f`, `9ffc25e` (+ `043852f`, `f815103` from the same session).
> Related: `docs/post-mortems/schema-update-postmortem.md` issues #8 and #9 (both resolved/re-explained here).

## Outcome

Integration tests for the transaction, psychographic, and text encoders called the real
`train()`, which saved its checkpoint to `models/*.pt` — **silently overwriting the
committed, trained checkpoints with 1-epoch test fixtures on every full-suite run.** The
corruption was invisible (tests stayed green); the only tell was `git status models/`.

This silent corruption **manufactured a false "stale checkpoint" diagnosis**: the
transaction encoder's on-disk dims read as `16/32/1` (the test fixture's), which was
misread as a checkpoint trained with non-default dims and never retrained. That led to a
`load_encoders` dim-inference shim and a planned retrain→rebuild→revalidate cascade —
**both unnecessary**, because the *committed* checkpoint was `128/64/2` (the current
defaults) all along. The shim was reverted once the corruption was discovered.

**Fix:** all six encoder `train()` functions now take `save_path: Path | None = None`
(defaulting to `CHECKPOINT_PATHS[...]`); integration tests pass a `tmp_path`-based
`save_path`. The trace and campaign encoders already had this param — transaction,
psychographic, text, and clickstream did not.

**Verification:** `md5sum models/*.pt` is byte-identical before and after a full
`uv run pytest` run, and `git status models/` is clean post-suite.

## Error catalog

| # | Issue | Where | Root cause |
|---|---|---|---|
| 1 | Integration tests overwrite committed `models/*.pt` | `tests/encoders/{transaction,psychographic,text}/*` → `train()` | `train()` resolved its save path via `CHECKPOINT_PATHS[...]` (hardcoded). The tests called `train(...)` for a 1-epoch smoke test; `train()` saved the barely-trained weights to the real checkpoint path. trace/campaign already accepted a `save_path` param; the other four did not. |
| 2 | Corruption was silent | — | tests passed (they only asserted shape/runnable, not checkpoint integrity); nothing asserted `models/*.pt` was unchanged after the suite. `git status` was the only signal. |
| 3 | "Stale checkpoint" misdiagnosis | `models/transaction_encoder.pt` read as `16/32/1` | the on-disk file was the test fixture (`gru_hidden=32, projection_dim=16, n_epochs=1`), not the committed `128/64/2` checkpoint. Diagnosed as staleness; was actually corruption. Led to a `load_encoders` shim (reverted) and a planned retrain (cancelled). |
| 4 | `load_encoders` appeared broken | `fusion/train.py` | downstream symptom of #3: `load_encoders` used model defaults (`128/64/2`) and could not load the corrupted (`16/32/1`) file. Not a real bug — resolved by restoring the real checkpoint. |

## Lessons (reusable)

1. **A test that calls a production training pipeline inherits that pipeline's writes.**
   Test isolation means isolating *all* side effects — checkpoint writes, MLflow runs,
   vocab files, cache files — not just inputs. Any `train()`-like entry point that
   persists artifacts must accept an override path, and tests must pass a tmp path. This
   is now the convention for all six encoders (`save_path` param); the
   `encoder-specialist` agent and `engineering-conventions.md` are updated to enforce it.

2. **Before diagnosing "the artifact is stale/wrong," run `git status` on it.** A
   *modified* tracked file means something changed it *this session* — not that it has
   always been that way. The 775 KB→44 KB size delta on `transaction_encoder.pt` was the
   real signal; the dims were a red herring planted by the corruption. When an artifact
   looks wrong, first establish whether it matches HEAD.

3. **Silent corruption is the worst failure mode: green tests, wrong reality.** Prefer a
   guard that detects drift. A cheap one for this codebase: after a full suite run,
   `git status models/` must be empty. (Could be a CI check.)

4. **Cross-reference — the schema-update post-mortem's #8 and #9 were the same drift
   family.** #8 ("counterfactual test failures — will resolve when fusion retrains") was
   actually the 4→6 modality mismatch (fixed) **plus** this checkpoint corruption; the
   "fusion retrain" remedy was a misdiagnosis. #9 (transaction test importing a renamed
   class) was fixed this session (`TransactionSequenceDataset` → `SplitTransactionDataset`
   + the stale 20-dim token-vector assertions). Both are closed.
