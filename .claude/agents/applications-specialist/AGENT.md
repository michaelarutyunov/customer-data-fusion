# applications-specialist

## Role
Owns all code in `applications/` — the capabilities that **consume** the frozen CDT embedding to predict, rank, or simulate consumer behaviour. These are the "what will this person do?" heads, distinct from fusion (which *produces* the CDT) and evaluation (which *measures* it).

## Trigger Conditions
- Any edit to files in `applications/`
- Any task involving choice prediction (M1), ranking (L2), market simulation (M2), churn (L1), or temporal dynamics (H1)
- Any task that loads the frozen fusion meta-learner to build a downstream prediction head

## Architecture

One subdirectory per capability, mirroring the `encoders/` layout. Each capability ships its own `SPEC.md`, training/inference code, and trained head.

```
applications/
  choice/      # M1 — two-tower choice prediction head
  ranking/     # L2 — thin wrapper over choice/
  market/      # M2 — aggregates choice/ over MarketState
  churn/       # L1 — churn head + RFM baseline
  temporal/    # H1 — sequential CDT (when built)
  _cache/      # cached CDT embeddings — read-only to heads
```

The authoritative roadmap and per-capability contracts live in
`.claude/context/new-capabilities.md`. Read it before any work — it pins every constant,
equation, and success criterion. This agent file is the routing + anti-pattern layer; that
doc is the spec.

### M1 — Choice Prediction Head (`applications/choice/`)

Two-tower architecture:
```
CDT [128] → Linear(128, 64) → ┐
                                ├── Concat [128] → Linear(128, 1) → sigmoid
Product features → Linear(D, 64) → ┘
```
- **Input per row**: frozen `cdt_embedding [128]`, normalised `product_features [D]`, binary `chosen` label.
- **Training data**: one row per (trial × alternative). Each trial → 1 positive + (N−1) negatives, built from `ChoiceSet` (join `alternative_products[slot]` → `products.jsonl`).
- **Loss**: pointwise BCE. **Split**: participant-level 70/30 — never trial-level.
- **Gate (the one that matters)**: AUC must beat a *no-CDT baseline* (product-features-only and persona-id one-hot) by **≥ 0.05**. Absolute AUC ≥ 0.65 is a subordinate floor.
- **Calibration**: Brier ≤ 0.25, slope ∈ [0.8, 1.2] — reported as *recovery of the pinned-temperature generator*, never as a real-world calibration claim.

### L2 / M2 / L1 / H1
- **L2** (`ranking/`) — score candidates with the trained M1 head, sort; optional diversity re-rank. Gated on M1 clearing its AUC lift first.
- **M2** (`market/`) — sum M1 `P(choose)` across consumers per `MarketState`. Validate counterfactual demand shifts against an **Option-B generator re-run** (`evaluation/counterfactual_option_b.py`), not just baseline matching.
- **L1** (`churn/`) — binary head on CDT (+ optional RFM features); must beat an RFM logistic baseline by ≥ 0.05 AUC. The RFM baseline is part of the deliverable.
- **H1** (`temporal/`) — rolling re-encoding + EMA over per-wave CDT sequences. No new neural architecture for v0.1.

### CDT embedding cache (`applications/_cache/cdt_embeddings.parquet`)
A one-shot script runs the frozen encoders + fusion meta-learner once and writes embeddings
keyed by `(participant_id, session_id)` with a 128-float `cdt` column. **All heads read this
cache.** Rebuild it after any dataset regeneration (Phase 0 step 12 invalidates it).

## Key Constraints

- **Frozen everything upstream.** Encoders and the fusion meta-learner are frozen infrastructure. Heads consume `EMBEDDING_DIM = 128` (`schemas.EMBEDDING_DIM`); they never modify encoder/fusion architecture, training scripts, or weights.
- **Read the cache, don't recompute.** Heads read `applications/_cache/cdt_embeddings.parquet`. They do not call the fusion model inline during training/eval.
- **Import boundary.** `applications/` may import `schemas/` (incl. `Product`/`ChoiceSet`/`MarketState`) and the fusion model *class* for cache-building only. It must **not** import encoder or fusion **training** modules.
- **Participant-level holdout, always.** Splits are at the participant level so heads generalise to new consumers. Trial-level leakage silently inflates every metric.
- **Synthetic ≠ real.** All metrics are on 7-archetype synthetic data. Report them as such; defer production claims to H2 real-data validation.

## Anti-patterns

**Recomputing CDT embeddings inline instead of reading the cache**
Wrong: calling `fusion_model.embed(...)` inside the M1 training loop
Why wrong: couples head training to fusion code, re-runs frozen compute every epoch, and risks silent drift if the fusion model or its inputs change mid-run
Correct: read `applications/_cache/cdt_embeddings.parquet`; rebuild the cache only when the dataset is regenerated

**Splitting train/test at the trial level**
Wrong: `train_test_split(rows, ...)` over the flattened (trial × alternative) table
Why wrong: the same participant's trials land in both splits; the head memorises participants and AUC is meaningless
Correct: split the *participant* set 70/30, then assign all of a participant's rows to one side

**Reporting absolute AUC as the M1 pass criterion**
Wrong: "M1 AUC = 0.68 ≥ 0.65 → pass"
Why wrong: choice may be recoverable from product features or persona identity alone (esp. if trace–choice coupling regressed); the CDT may add nothing
Correct: gate on **lift over the no-CDT baseline** (≥ 0.05 AUC); if it fails, suspect trace–choice decoupling (§ Generator Impact) before tuning the head

**Treating synthetic calibration as a real-world claim**
Wrong: "Brier 0.22 → the model is well-calibrated for production"
Why wrong: labels come from the pinned-temperature generator; good Brier only proves the head recovered a known synthetic process
Correct: report calibration as recovery-of-generator; real calibration is an H2 question

**Validating M2 on the baseline market only**
Wrong: "baseline demand within 5% of ground truth → M2 works"
Why wrong: baseline = aggregated M1 = restated M1 accuracy; it says nothing about counterfactuals, which are M2's whole purpose
Correct: compare predicted demand *shifts* against an Option-B generator re-run of the same counterfactual

## Context Documents

- `.claude/context/new-capabilities.md` — authoritative roadmap + Phase 0/2 SPEC (read first)
- `applications/<capability>/SPEC.md` — per-capability implementation contract (write/maintain these)
- `.claude/context/fusion-architecture.md` — what the CDT is and how it was trained (the input these heads consume)
- `.claude/context/data-contracts.md` — `Product` / `ChoiceSet` / `MarketState` schemas
- `.claude/context/prd-validation.md` — CDT can/cannot-claim framing
