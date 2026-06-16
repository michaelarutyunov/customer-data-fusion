# New Capabilities — Post-Prototype Enhancement Roadmap

> Version: 0.6
> Status: Proposals. None committed.
> Audience: Coding agents creating implementation epics; technical reviewers.
>
> **v0.6 changes (module governance):** Pinned the home for CDT-consumer capabilities —
> a new top-level `applications/` package (mirroring `encoders/`), with a boundary
> invariant, the CDT-embedding cache contract (`applications/_cache/cdt_embeddings.parquet`),
> and the Phase-2 prerequisite to create an `applications-specialist` agent + CLAUDE.md
> trigger row. Resolves the "where do M1/L2/M2/L1 heads live, and which agent owns them"
> gap (§ Module placement).
>
> **v0.5 changes (bead-readiness):** Added the **Phase 0 Generator SPEC** (§ pinning the
> attribute encoding, per-strategy utility equations, softmax/gain/lapse constants,
> catalogue-generation parameters, trial composition, and transaction selection) so the
> choice-model and M1 beads are creation-ready. Resolved the `ChoiceSet` slot-letter vs.
> product-ID contract contradiction (canonical key is the slot letter; product identity
> lives in `alternative_products`).
>
> **v0.4 changes (evaluation-validity hardening):** Coupled the choice model to the
> acquisition trace (§ Generator Impact, M1); pinned softmax temperature as a fixed
> generator constant to remove M1 calibration circularity (§ PersonaConfig, M1); made
> trace/transaction deltas the primary L3 metric; tied M2 counterfactual validation to
> the Option-B generator re-run; added a no-CDT lift gate to M1; costed Phase 0 in the
> complexity table with an explicit retraining trigger.

---

## Rationale

The prototype demonstrated that multi-modal encoders + late fusion can learn a per-consumer behavioural embedding (CDT) that preserves individual identity (70.4% recall@1, 140× over chance) and recovers latent traits (R² 0.79–0.96 on all 7 PersonaConfig parameters). The CDT answers *"who is this person?"*

The capabilities below extend the system toward *"what will this person do?"* — predicting choices, ranking products, simulating market responses. However, a critical prerequisite is missing: **the system has no product model.** In the current generator:

- **Choices are random.** In `trace_simulator.py`, the trial-loop computes `final_choice = str(rng.choice(alts))` — a uniform random draw among abstract alternatives, **independent of the acquisition sequence**. No preference model. No product attributes influence the decision. Critically, `StrategyParams` already defines the choice-rule fields (`first_attribute`, `attribute_weights`, `aspiration_levels`, `rejection_threshold_pct`) — they are *generated today but never consumed*. Phase 0's choice model wires up fields the schema already carries; it does not invent new ones.
- **Attribute values are ephemeral.** The information board has attributes (`price`, `brand`, `quality`, etc.) but only their *names* appear in `AcquisitionEvent.attribute_id`. The actual values shown to the consumer (price = £299, quality = 4.2 stars) are never generated and never stored.
- **Products don't exist as stable objects.** The transaction simulator's per-transaction loop (`transaction_simulator.py`, the `product_id = f"prod_{category}_..."` line) generates ephemeral IDs like `prod_electronics_mid_4823` — no product catalogue, no shared identity across consumers.

> **Reference-rot note.** Code locations here are given by file + symbol, not line number, because line numbers drift. Resolve with `rg` (e.g. `rg -n "rng.choice\(alts\)" generator/trace_simulator.py`) before editing.

Any capability that predicts, ranks, or simulates choices among products requires a new `Product` schema and a preference-driven choice model in the generator. Capabilities that operate purely on the CDT embedding (churn prediction, cross-session stability, temporal dynamics) are viable without these changes.

### Invariant

**Encoders and fusion training logic remain frozen.** The four modality encoders and the fusion meta-learner are trained infrastructure. No capability modifies their architecture, training scripts, or saved weights. All additions consume the frozen `EMBEDDING_DIM=128` CDT embedding.

**Schema and generator changes are permitted** when required by a capability. These are additive and backward-compatible where possible. When schema changes invalidate existing synthetic data, the dataset must be regenerated and encoders revalidated (but not retrained unless the input distribution changes materially).

### Module placement (CDT-consumer capabilities)

Capabilities that **consume** the frozen CDT but are not schemas/generator/encoders/fusion
live in a new top-level **`applications/`** package, mirroring the `encoders/` layout — one
subdirectory per capability, each with its own `SPEC.md`:

```
applications/
  choice/      # M1 — choice prediction head (two-tower)
  ranking/     # L2 — thin wrapper over choice/
  market/      # M2 — aggregates choice/ over MarketState
  churn/       # L1 — churn head + RFM baseline
  temporal/    # H1 — sequential CDT (when built)
  _cache/      # cached CDT embeddings (see M1 data requirements)
```

**Boundary invariant.** `applications/` modules import the frozen fusion meta-learner and
`EMBEDDING_DIM` from `schemas/`, and may import `Product`/`ChoiceSet`/`MarketState`. They
**never** import or modify encoder/fusion *training* code, and they read CDT embeddings from
the cache rather than recomputing them inline. Each ships its own trained head under its
subdir.

- **Why not `fusion/`** — fusion *produces* the CDT (it is the meta-learner); heads *consume*
  it. Co-locating would blur the "late fusion produces one embedding" boundary and risks
  coupling head training into fusion training, which the Invariant forbids.
- **Why not `evaluation/`** — evaluation *measures* the CDT (L3 stability, probes, geometry)
  and stays there. Applications are trained models with their own weights and inference paths.
  Dividing line: **L1 churn is an application** (ships a head); **L3 stability is evaluation**
  (ships a metric).

**CDT embedding cache contract** (resolves the open M1 dependency). A one-shot script writes
frozen embeddings to `applications/_cache/cdt_embeddings.parquet`, keyed by
`(participant_id, session_id)` with a 128-float `cdt` column. M1/L2/M2 read this cache; they
do not call the fusion model inline. Regeneration of the dataset invalidates the cache —
rebuild it after Phase 0 step 12.

**Phase 2 governance (in place).** The Tier-2 agent
`.claude/agents/applications-specialist/AGENT.md` exists and the CLAUDE.md Agent Trigger Table
routes `applications/**` to it. The agent encodes the boundary invariant and the anti-patterns
(recomputing embeddings instead of reading `_cache/`; importing encoder/fusion train modules;
trial-level instead of participant-level splits; reporting absolute AUC instead of no-CDT lift;
validating M2 on baseline only). The `applications/` directory itself is created lazily by the
first Phase-2 bead (M1).

> **Retraining trigger (made concrete).** "Input distribution changes materially" is
> not left to judgement. Coupling the choice model to the trace (see § Generator Impact)
> can shift the *acquisition-token* distribution the trace encoder consumes. After
> regeneration, compute the per-token-type frequency distribution over `AcquisitionEvent`
> streams and compare to the prototype baseline. If symmetric KL divergence exceeds
> **0.05 nats**, OR any encoder probe falls below its floor (§ Data Regeneration),
> retrain the affected encoder(s) before building dependent capabilities. The trace
> encoder is the one at risk; psych/text/transaction inputs are unaffected by the choice
> coupling.

---

## Capability Summary

| ID | Capability | Needs products? | Needs attribute values? | Needs preference model? | Complexity |
|----|-----------|:---:|:---:|:---:|:---:|
| **P0** | **Schema Foundation (product catalogue + coupled choice model + regen)** | — | — | — | **High** |
| L1 | Churn / Defection Prediction | No | No | No | Low |
| L2 | Personalised Ranking / Recommendation | Yes | Yes | Yes | Low\* |
| L3 | Cross-Session Stability | No | No | No | Low |
| M1 | Choice Prediction Head | Yes | Yes | Yes | Medium |
| M2 | Market-Level Demand Simulation | Yes | Yes | Yes | Medium |
| H1 | Temporal Dynamics (Sequential CDT) | No | No | No | High |
| H2 | Real Data Integration Pipeline | Partial | Partial | No | High |

\* L2 depends on M1. Standalone complexity is low but M1 is a prerequisite.

**P0 is the real critical path.** It gates M1, L2, and M2, and is the largest single
chunk in the program (full generator rewrite + regeneration + revalidation). M1's
"Medium / 1–2 weeks" estimate is *additional* to P0, not inclusive of it.

**Complexity definitions:**

| Level | Definition | Typical effort |
|-------|-----------|---------------|
| Low | Single-file module or evaluation script; reuses frozen embeddings; no new model architecture | 1–3 days |
| Medium | New model head with training loop; new dataset construction; foundational for other capabilities | 1–2 weeks |
| High | Multi-file subsystem; requires generator changes, repeated data collection, or external integration | 2–6 weeks |

---

## Schema Requirements

### Existing Schemas

| Schema | Status | Capabilities | Rationale |
|--------|--------|-------------|-----------|
| `AcquisitionEvent` | **No changes** | All (via trace encoder) | Token input for trace encoder. Captures which cells were inspected. Complete. |
| `TrialRecord` | **Needs revision** | M1, L2, M2 | Must link to `ChoiceSet` for product-level choice data. Records `final_choice` but not the products shown or their attributes. |
| `TransactionRecord` | **Needs revision** | L1, M2, H2 | `product_id` must reference a stable catalogue (currently ephemeral). Required for market simulation and real-data adapters. |
| `PsychographicVector` | **No changes** | All (via psych encoder) | Complete. |
| `PersonaNarrative` | **No changes** | All (via text encoder) | Complete. |
| `PersonaConfig` | **No structural changes** | Generator | Existing `StrategyParams`, `TransactionParams`, `PsychographicParams` contain enough information to derive a preference model. Choice rule is a generator implementation detail, not a schema field. |

### New Schemas Required

| Schema | Capabilities | Rationale |
|--------|-------------|-----------|
| `Product` | M1, L2, M2, H2 | Stable product identity with concrete attributes. Shared across trials and transactions. Enables product-level choice prediction and market simulation. |
| `ChoiceSet` | M1, L2, M2 | Links a trial to the products presented, their displayed attribute values, and the chosen alternative. Primary training data for the choice model. |
| `MarketState` | M2 | Snapshot of available products and conditions for counterfactual demand simulation. |

---

## Schema Revisions

### Changes to Existing Schemas

#### TrialRecord

Add one field:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `choice_set_id` | `str` | `""` | Foreign key to `ChoiceSet.trial_id`. Links the trial to the products shown and their attribute values. |

The existing `final_choice` field remains. The authoritative choice data lives in `ChoiceSet.chosen_alternative`. Backward-compatible: old `TrialRecord` data parses without error (empty `choice_set_id`).

#### TransactionRecord

**Semantic change, no structural change.** The `product_id` field currently holds ephemeral labels like `prod_electronics_mid_4823`. After the schema revision, it must reference stable product IDs from the new `Product` catalogue. The field name and type are unchanged.

This enables M2 to aggregate demand per product across consumers and H2 adapters to map real product IDs into the catalogue.

#### PersonaConfig

No structural changes. The existing parameter fields are sufficient to derive a preference model in the generator:

| Persona strategy | Choice rule (derived from existing params) |
|-----------------|-------------------------------------------|
| Lexicographic | Sort alternatives by `StrategyParams.first_attribute` value; pick best |
| Affect heuristic | Prefer alternatives matching the preferred `brand_tier` (from `brand_loyalty`) |
| Compensatory | Weighted utility sum; weights derived from `price_sensitivity`, `brand_loyalty`, `involvement_score` |
| Satisficing | Threshold-based acceptance; thresholds from existing params |
| Adaptive | Compensatory for simple boards, satisficing for complex |
| Random | Uniform random (applied on `p_strategy_lapse` probability) |

**Noise sources that modify the choice:**
- `p_strategy_lapse`: probability per trial of falling through to random choice, regardless of strategy. Already implemented in the trace simulator.
- `LatentDeviation`: continuous noise on each persona parameter (e.g., `z.thoroughness`, `z.impulsivity`, `z.brand_lean`). For the choice model, `z.brand_lean` shifts brand-sensitive weights, `z.impulsivity` increases the probability of choosing a non-optimal alternative. These are additive perturbations, not strategy changes.
- Stochastic choice via softmax temperature: the preference model outputs a utility score per alternative, converted to choice probability via softmax with a temperature parameter. Lower temperature → more deterministic (archetype policy dominates); higher temperature → more random. This prevents the model from outputting probability 1.0 for every dominant alternative.

> **Temperature must be a fixed, declared constant — not tuned to hit M1's calibration target.**
> Because M1 is both trained and evaluated on labels the generator produced at temperature
> `τ`, tuning `τ` to make M1's Brier score look good is circular — it measures only whether
> M1 recovered the generator's RNG, not real-world calibration. Pin `τ = 1.0` (or a single
> value justified on behavioural grounds) in the Phase 0 generator SPEC and treat it as a
> generative constant for the life of the dataset. M1's calibration check then verifies
> *recovery of a known process*, and must be reported as such — synthetic calibration is
> **not** evidence of calibration on real choices.

The noise-source magnitudes (the `LatentDeviation` scaling, `p_strategy_lapse`) are a generator implementation detail specified in the Phase 0 generator SPEC. The softmax temperature is **not** among them — it is pinned per the box above.

### New Schemas

#### Product

Stable product with concrete attributes. Multiple consumers can buy the same product. Multiple trials can feature the same product.

| Field | Type | Description |
|-------|------|-------------|
| `product_id` | `str` | Stable identifier, unique within a category. Shared across transactions and trials. |
| `category` | `str` | Product category (e.g., "electronics", "groceries"). |
| `brand_tier` | `str` | One of "premium", "mid", "value", "own_label". |
| `price_normalised` | `float` | Price as 0–1 percentile within category range. Enables cross-category comparison. |
| `quality_score` | `float` | 0–1 quality rating. |
| `warranty_score` | `float` | 0–1 warranty coverage. |
| `rating` | `float` | 0–5 average consumer rating. |
| `features_score` | `float` | 0–1 feature richness. |
| `availability` | `bool` | Whether currently in stock. |
| `design_score` | `float` | 0–1 aesthetic appeal. |
| `on_promotion` | `bool` | Whether currently on promotional offer. |

The attribute names (`price`, `quality`, `warranty`, `rating`, `features`, `availability`, `design`) match the trace simulator's `_ATTRIBUTES` list, ensuring that what the consumer inspects on the information board corresponds to actual product attributes.

**Consumers:** M1 uses Product fields as input features. M2 aggregates demand per product. H2 adapters map real catalogue data to this schema.

**Producers:** New product catalogue generator module. One output file: `data/synthetic/products.jsonl`.

#### ChoiceSet

Links a trial to the products presented, their displayed attribute values, and the consumer's choice.

| Field | Type | Description |
|-------|------|-------------|
| `trial_id` | `str` | Foreign key to `TrialRecord.trial_id`. |
| `session_id` | `str` | Session identifier. Matches `TrialRecord.session_id`. |
| `participant_id` | `str` | Participant identifier. |
| `alternatives` | `tuple[str, ...]` | **Slot letters** present this trial, e.g. `("A","B","C")`. These are the canonical alternative keys — identical to `AcquisitionEvent.alternative_id` and consistent with `TrialRecord.n_alternatives`. |
| `alternative_products` | `dict[str, str]` | `{slot_letter: product_id}` — maps each slot to its `Product.product_id`. This is where product identity lives; `alternatives` stays letters so trace tokens, `final_choice`, and `ChoiceSet` all share one key space. |
| `displayed_attributes` | `dict[str, dict[str, float]]` | `{slot_letter: {attr_name: float}}` — the values shown on the board. **All values are floats in [0, 1]** (encoding rules below); categorical/bool attributes are projected to floats so the dict type is honest. |
| `chosen_alternative` | `str` | The **slot letter** selected. Equals `TrialRecord.final_choice` exactly (no product-ID/letter mismatch). |
| `choice_mechanism` | `str` | One of `"preference"` or `"random_lapse"`. Driven by the persona's preference model vs. a strategy lapse (`trial_strategy == RANDOM`). |

Each row provides `(participant, slot→product map, displayed float attributes, chosen slot)` — everything needed to train a model that predicts choice from CDT embedding + product features. Resolve a product's full feature vector by joining `alternative_products[slot]` against `products.jsonl`.

> **Contract note (resolves the v0.3 ambiguity):** earlier drafts said `alternatives` held product IDs while `chosen_alternative` matched the letter `final_choice` — contradictory, because `TrialRecord.final_choice` is a slot letter (`_ALTERNATIVES = ["A".."G"]`). The canonical key is now the **slot letter** throughout; product identity is carried in the parallel `alternative_products` map.

**Consumers:** M1 trains on ChoiceSet records. L2 uses them for evaluation. M2 uses them to reconstruct market conditions.

**Producers:** Revised trace simulator. One output file: `data/synthetic/choice_sets.jsonl`.

#### MarketState

A snapshot of available products and market conditions for counterfactual simulation.

| Field | Type | Description |
|-------|------|-------------|
| `state_id` | `str` | Unique identifier for this market state. |
| `label` | `str` | Human-readable label (e.g., "baseline", "price_increase_10pct"). |
| `products` | `tuple[str, ...]` | Product IDs of all products available in this market state. |
| `product_modifiers` | `dict[str, dict[str, float]]` | Optional overrides: `{product_id: {"price_normalised": 0.75}}`. Products not listed use their catalogue values. |

Enables M2 to define a baseline market and counterfactual markets (price changes, new entrants, product withdrawals) without modifying the Product catalogue.

**Consumers:** M2.

**Producers:** Configuration files written by the analyst (not generated). Stored as `config/market_states/` JSON files.

---

## Generator Impact

### Trace Simulator

`generator/trace_simulator.py` requires two additions:

1. **Generate product attribute values per trial.** For each trial, select products from the catalogue (or generate novel products for "new entrant" scenarios). Assign concrete attribute values to each cell on the information board. Persist as a `ChoiceSet` record.

2. **Implement preference-driven choice — coupled to the trace.** Replace `final_choice = str(rng.choice(alts))` with a choice rule derived from `StrategyParams`, applied **over the attribute values the consumer actually inspected in this trial's acquisition sequence**, not over the full attribute matrix. The same per-trial `LatentDeviation` realisation that drives the acquisition sequence also drives the choice utility, so trace and choice share their stochastic source.

> **Why coupling is mandatory (not a refinement).** Today, trace and choice are two
> independent draws conditioned on the persona. If choice is computed from the full
> attribute matrix while the trace is generated separately, then **trace ⊥ choice | persona**:
> the trace carries no information about the choice beyond persona identity. M1 (whose CDT
> signal is dominated by the trace encoder) could then hit its AUC target purely through
> persona leakage, and we would wrongly conclude the *decision process* predicts the
> *decision*. A genuine MouseLab process has the opposite property — *what you inspect is
> the mechanism of what you choose* (lexicographic-on-price ⇒ you looked at price, ignored
> the rest, **and** chose on price). The choice rule must therefore read from the inspected
> cells. Concretely: a lexicographic persona chooses the best alternative on `first_attribute`
> *among the cells it actually opened*; a compensatory persona forms a weighted utility over
> *inspected* attributes; unobserved attributes do not enter the utility. This makes the
> trace genuinely predictive of the choice and is the foundation M1's premise rests on.

This is the one place the acquisition sequence generation is **not** left fully untouched: the *sequence* of which cells are inspected, in what order, with what dwell times, is unchanged (the prototype validated it), but the **revealed values at those cells now feed the choice**. The choice is a deterministic function of `(inspected attribute values, StrategyParams)` plus the pinned-temperature softmax and `p_strategy_lapse`. Because the inspection-token *stream* is unchanged, the trace encoder's input distribution should be stable — but this must be verified against the retraining trigger (§ Invariant) after regeneration.

### Transaction Simulator

`generator/transaction_simulator.py` requires one change:

- Select products from the product catalogue instead of generating ephemeral `product_id` labels. Sample products matching the consumer's `brand_tier` preferences and `price_sensitivity`.

### New Component: Product Catalogue Generator

A new module generates the synthetic product catalogue. For each category, produce a set of products with stable IDs and attribute values. The catalogue is shared across all participants.

Output: `data/synthetic/products.jsonl` — one `Product` record per line.

### Data Regeneration

Schema changes to `TrialRecord` and `TransactionRecord` invalidate existing synthetic data. After changes:

1. Generate product catalogue → `data/synthetic/products.jsonl`
2. Regenerate full synthetic dataset using updated generator
3. Revalidate with `generator/validate.py`
4. Revalidate encoder probes with `uv run python -m evaluation.run_probes`
5. Do **not** retrain encoders unless probe results drop below these floors:
   - Fused strategy recovery ≥ 95%
   - Individual recall@1 ≥ 60%
   - PersonaConfig R² ≥ 0.70 for ≥ 5/7 parameters
   If any metric falls below its floor, investigate root cause before retraining.

---

## Capability Descriptions

### L1 — Churn / Defection Prediction

**Complexity: Low**

**Description.** A binary classification head on the CDT embedding that predicts whether a consumer will churn (no transaction in next N days). Churn is about transaction frequency, not product choice — no product model required.

**User story.** A CRM analyst runs a weekly batch job that loads the frozen fusion model, computes a CDT embedding for each customer, passes it through the churn head, and outputs a CSV with `participant_id, p_churn_30d`. The analyst sorts by probability and hands the top 5% to the retention team for proactive outreach.

**Business questions.**

| Question | Stakeholder | Decision enabled |
|----------|-------------|-----------------|
| "Which customers are at highest risk of switching to a competitor next quarter?" | Retention Marketing | Proactive outreach list |
| "Does the CDT embedding predict churn better than an RFM logistic model?" | Analytics Manager | Model replacement justification |
| "What behavioural signals in the embedding predict churn 30 days before it happens?" | Data Science | Early warning feature analysis |

**Technical description.** Binary classification: `CDT [128] + optional temporal features (months_since_last_purchase, transaction_frequency, avg_spend) → linear/sigmoid → P(churn_30d)`. Churn labels derived from `TransactionRecord.days_before_session` by thresholding the gap since last transaction. The churn window is a configurable parameter.

**Data requirements.**

- No new schemas required. Churn labels derived from existing `transactions.jsonl`.
- No product model required.
- No encoder retraining. Uses frozen CDT embeddings.
- Class imbalance expected (real churn rates 5–10%). Strategy: class-weighted BCE or oversampling.

**Success criterion:** AUC improvement ≥ 0.05 over a transaction-only baseline (logistic regression on RFM features: recency, frequency, monetary value). The RFM baseline must be constructed as part of this capability.

**Risk:** Churn labels derived from transaction gaps may be fully explained by RFM features alone — the CDT embedding may not add discriminative signal beyond what the transaction history already contains. If so, report this honestly; the negative result is still informative.

---

### L2 — Personalised Ranking / Recommendation

**Complexity: Low (depends on M1)**

**Description.** Given a consumer's CDT embedding and a set of candidate products, score and rank each product using M1's choice model. Unlike collaborative filtering, this works for cold-start users — the ranking generalises from *how* the person decides, not *what* they have clicked.

**User story.** An e-commerce API receives a homepage request for a customer. It loads the customer's CDT embedding and the currently eligible products, scores each with the choice model, and returns the top 5.

**Business questions.**

| Question | Stakeholder | Decision enabled |
|----------|-------------|-----------------|
| "What is the top-5 product ranking for this specific customer?" | E-commerce Manager | Homepage personalisation |
| "Does the CDT embedding produce better cold-start ranking than collaborative filtering?" | Recommendation Engineer | Algorithm selection |
| "Which product attributes should we emphasise in messaging to Customer X?" | CRM Manager | Content personalisation |

**Technical description.** Thin wrapper around M1. For each candidate product, compute `P(choose | cdt_embedding, product_features)` and sort by score. Optional diversity regulariser or constrained re-ranking step to prevent homogeneous top-K lists.

**Data requirements.**

- Requires M1 (Choice Prediction Head) to be trained.
- Requires Product schema (inherited from M1).
- No additional data beyond M1.

**Prerequisite gate:** L2 can only be meaningfully evaluated after M1 achieves its discrimination criterion (AUC ≥ 0.65). If M1 barely clears this threshold, ranking quality will be poor — consider the M1 gate a minimum, not a guarantee of ranking quality.

**Success criterion:** NDCG@5 exceeds random ranking by ≥ 0.15 on held-out participants. Intra-list diversity ≥ 0.4.

---

### L3 — Cross-Session Stability

**Complexity: Low**

**Description.** A validation protocol that tests whether the same person produces the same CDT embedding across independent sessions separated in time. Not a model — a measurement tool.

**User story.** A research team re-invites 200 panel participants for a second wave, 6 months after the first. They compute CDT embeddings for each participant in each wave, then report the similarity delta. If the delta drops below threshold, the team concludes that panel refresh must happen more frequently.

**Business questions.**

| Question | Stakeholder | Decision enabled |
|----------|-------------|-----------------|
| "If we re-survey a customer 6 months later, will the CDT identify them as the same person?" | Research Director | Panel recruitment investment |
| "Does a customer's decision strategy drift after a major life event?" | Behavioural Scientist | Longitudinal study design |
| "What is the half-life of a CDT embedding?" | Product Manager | Data collection budget |

**Technical description.**
1. Generate two sessions per participant (same `PersonaConfig`, different random seed)
2. Compute CDT embeddings for both sessions independently
3. Measure cosine similarity: same-person pairs (positive) vs. different-person pairs (negative)
4. Primary metric: `similarity_delta = mean(same_person) - mean(diff_person)`

Synthetic implementation: call the generator pipeline twice per participant with the same `PersonaConfig` but different seed. The latent parameters stay fixed; only realisation noise changes.

**Data requirements.**

- New synthetic data: generate a second session per participant. No new schemas — use existing schemas with a `session_id` discriminator.
- No product model required.
- No encoder retraining.

**Success criterion (revised — behavioural modalities are the real test).** The fused delta is **not** a meaningful gate on its own: the psychographic vector is `params + small noise` and the narrative is generated from the persona, so both are near-deterministic in `PersonaConfig` (prototype deltas 0.60 / 0.61). A high *fused* delta is therefore guaranteed by two frozen modalities and tells us nothing about whether the genuinely stochastic behavioural signal is stable across sessions.

- **Primary metric:** trace-only and transaction-only `similarity_delta ≥ 0.20` each. These are the only modalities whose realisation actually varies across seeds, so they are what "cross-session stability" means here.
- **Secondary (headline) metric:** fused `similarity_delta ≥ 0.30`, reported for completeness.
- Report all four per-modality deltas. If either behavioural delta falls below 0.20, the embedding is *not* cross-session stable regardless of the fused number — investigate before proceeding to H1 (temporal dynamics), since H1's drift detection depends on behavioural stability.

---

### M1 — Choice Prediction Head

**Complexity: Medium — Critical path for L2 and M2**

**Description.** A learned module that maps `[CDT_embedding, product_features] → P(choose_this_product)`. This transforms the CDT from "who is this person?" into "what will this person do?"

**User story.** A product manager evaluates a new mid-tier SKU before launch. She defines the product's features in a configuration file, then runs the choice model against the consumer panel. For each consumer, the model outputs `P(choose_new_sku)`. She selects the top 10,000 customers for the launch campaign.

**Business questions.**

| Question | Stakeholder | Decision enabled |
|----------|-------------|-----------------|
| "If we launch a new product at a given price point, which customers are most likely to buy?" | Product Manager | Launch targeting |
| "What choice probability does Consumer X assign to each of 5 competing products?" | Marketing Analyst | Personalised messaging |
| "Does the CDT embedding beat a demographics-only logistic model?" | Data Science Lead | Model adoption go/no-go |

**Technical description.**

Two-tower architecture:
```
CDT [128] → Linear(128, 64) → ┐
                                ├── Concat [128] → Linear(128, 1) → sigmoid
Product features → Linear(D, 64) → ┘
```
Where `D` is the number of concrete product attributes from the Product schema.

**Input per training example:**
- `cdt_embedding`: frozen [128] from fusion meta-learner
- `product_features`: normalised vector of Product schema attributes
- `label`: binary — did the consumer choose this product in this trial?

**Training data:** Each conjoint trial produces 1 chosen + N−1 rejected alternatives. Emit N rows per trial (1 positive, N−1 negative). Source: `ChoiceSet` records linked to `TrialRecord`.

**Loss:** Binary cross-entropy (pointwise). Pairwise rank loss (BPR) is a future upgrade.

**Train/test split:** Participant-level holdout (70/30). Never split at trial level — the model must generalise to new consumers.

**Data requirements.**

- Requires `Product` schema (new).
- Requires `ChoiceSet` schema (new).
- Requires generator rewrite to produce preference-driven choices and persist attribute values.
- Requires data regeneration — existing synthetic data has random choices and no attribute values.
- Derived dataset: flat table of `(participant_id, cdt_embedding, product_features_vector, chosen_bool)` rows built from `ChoiceSet` (join `alternative_products[slot]` → `products.jsonl` for features) + CDT embeddings read from `applications/_cache/cdt_embeddings.parquet` (§ Module placement). Lives under `applications/choice/`.

**Success criteria:**
- **CDT contribution (the gating criterion).** AUC must beat a *no-CDT baseline* — the same two-tower head with the CDT tower replaced by (a) product-features-only and (b) a persona-id one-hot — by **≥ 0.05 AUC**. This mirrors L1's RFM-baseline discipline and is the real test: it proves the *decision process* (trace-dominated CDT) predicts the choice, rather than the choice being recoverable from product features or persona identity alone. An absolute AUC that does not clear this lift is a **fail**, because it means the CDT added nothing — a likely symptom of trace–choice decoupling (§ Generator Impact). Without the coupling fix, this gate is unpassable, which is exactly the point.
- **Discrimination (floor):** AUC ≥ 0.65 on held-out participants — a minimum, subordinate to the lift gate above.
- **Calibration (recovery check, not real-world claim):** Brier ≤ 0.25 and calibration-curve slope ∈ [0.8, 1.2] on a held-out diverse choice set. Because labels come from the pinned-temperature generator (§ PersonaConfig), this verifies *recovery of the known generative process*, not calibration on real choices. Report it as such.
- **Probability range:** Not a pass/fail criterion — a well-specified preference model may legitimately output near-0 or near-1 for dominant alternatives. Calibration is checked on an evaluation set designed to include both easy and hard choices (diverse attribute spreads within choice sets).

---

### M2 — Market-Level Demand Simulation

**Complexity: Medium — Depends on M1**

**Description.** Aggregates individual choice predictions (M1) to market-level outcomes. Computes expected demand per product under baseline and counterfactual conditions. Replaces the hand-coded counterfactual rules of `evaluation/counterfactual.py` with learned choice probabilities.

**User story.** A pricing analyst models a 10% price increase. She defines baseline and counterfactual market states, runs the simulator, and learns that aggregate demand falls 8% — but the price-sensitive segment falls 22% while the brand-loyal segment falls only 2%. She recommends segment-specific pricing.

**Business questions.**

| Question | Stakeholder | Decision enabled |
|----------|-------------|-----------------|
| "If we raise prices 10%, what is the demand shift across segments?" | Revenue Management | Pricing optimisation |
| "How many customers will a new entrant steal, and from which segments?" | Strategy | Competitive response |
| "What is the revenue impact of withdrawing Brand X?" | Portfolio Manager | SKU rationalisation |

**Technical description.**
1. For each consumer, compute `P(choose_product_i | market_state)` for all products using M1
2. Sum probabilities across consumers: `expected_demand_i = Σ_j P_j(choose_i)`
3. Compare expected demand under baseline vs. counterfactual `MarketState`

**Data requirements.**

- Requires M1 (choice prediction head) trained.
- Requires `Product` schema (inherited from M1).
- Requires `MarketState` schema (new).
- Market state definitions: configuration files in `config/market_states/`.
- No new training data.

**Success criteria:**
- **Baseline sanity check:** aggregate predicted demand matches the generator's known ground truth within 5% for the baseline market state. Note this is *weak* on its own — M2 is aggregated M1, and M1 is trained to mimic the generator, so baseline matching largely restates M1's accuracy. It catches aggregation bugs, nothing more.
- **Counterfactual validity (the criterion that matters):** for at least one price-change and one product-withdrawal `MarketState`, M2's predicted *demand shift* must match a ground-truth shift obtained by re-running the generator under the same counterfactual via the generator's `persona_overrides` utility (`generator/pipeline.py` `run_pipeline(persona_overrides=...)`, which re-runs with overridden `PersonaConfig` params). Target: predicted vs. ground-truth demand-shift direction agrees for every segment, and magnitude within 10%. This tests the thing the pricing analyst actually relies on; baseline matching does not.

---

### H1 — Temporal Dynamics (Sequential CDT)

**Complexity: High** *(validation attempted 2026-06-16 — FAILED — see below)*

> **❌ Validation FAILED (2026-06-16):** H1 cannot work with frozen fusion embeddings. All 1002 participants produced identical embeddings across 12 monthly observations (variance = 0.0). Root cause: fusion meta-learner trained with NT-Xent loss that collapses within-participant variance for identity stability. See `docs/post-mortems/h1-temporal-postmortem.md` for full analysis.

> **Readiness note (pre-validation):** the data prerequisites below are **already built** — `PersonaConfig.month` exists; `persona_sampler.sample_temporal_trajectory()` emits 12 monthly snapshots with AR(1) drift + injected regime shifts; `participant_configs.jsonl` carries `drift_label`/`drift_month` (the recall@1≥0.80 ground truth); every modality is fielded monthly (trace/transaction/psychographic/clickstream/campaign all carry `month`); `evaluation/temporal_split.py` does months 1–8 train / 9–12 eval. Data uses `month`, not `session_id`/`wave_id` (naming only). The implementation scripts exist (`generate_monthly_embeddings.py`, `extract_features.py`, `train_drift_detector.py`) but cannot succeed with frozen fusion.

**Status:** Validated — **negative result**. Frozen CDT embeddings encode identity, not temporal dynamics.

**Description.** Treat the CDT embedding as a time series. Produce a sequence of embeddings per consumer across multiple data collection waves, enabling drift detection and trajectory prediction.

**User story.** A brand manager runs a 3-month loyalty campaign. Each week, she refreshes CDT embeddings for the campaign cohort. A drift detector flags customers whose embedding has shifted toward "low brand loyalty" in the last 4 weeks. The CRM system sends escalating offers to drifting customers.

**Business questions.**

| Question | Stakeholder | Decision enabled |
|----------|-------------|-----------------|
| "Is this customer's price sensitivity increasing month over month?" | Pricing Strategist | Dynamic pricing eligibility |
| "Did the campaign shift this person's brand loyalty?" | Campaign Analyst | Attribution, ROI |
| "When should we re-engage a customer drifting toward churn?" | CRM Manager | Trigger-based outreach |

**Technical description.** Three approaches, ordered by complexity:

| Approach | Description | Use case |
|----------|------------|----------|
| A. Rolling window re-encoding | Re-encode each wave's data independently | Baseline; no new architecture |
| B. Temporal smoothing | Exponential moving average of CDT embeddings | Noise reduction |
| C. Sequential encoder (GRU/Transformer) | Learn a temporal model over CDT sequence | Predict next-month embedding |

Recommended for v0.1: Approach A + B. Rolling monthly encodings with exponential smoothing. No new neural architecture.

**Data requirements.**

- New synthetic data: generate K=6–12 monthly waves per participant. Each wave is a partial re-generation:
  - `transactions.jsonl`: append new monthly transactions
  - `traces.jsonl` + `choice_sets.jsonl`: new conjoint session per wave
  - `psychographics.jsonl`: same or slightly drifted
  - `narratives.jsonl`: new narrative per wave
- Schema additions: add `wave_id: int` and `session_date: str` to all output records. Additive — no breaking changes.
- Generator modification: support multiple waves per participant with controlled parameter drift (e.g., `price_sensitivity += 0.02 per wave`).
- No product model required beyond what already exists.

**Success criterion:** Drift detection identifies known synthetic drift (e.g., `price_sensitivity += 0.02 per wave`) with recall ≥ 0.80.

---

### H2 — Real Data Integration Pipeline

**Complexity: High**

**Description.** A data pipeline that replaces generator outputs with real data sources, producing schema-compliant JSONL files so existing encoders and fusion require zero code changes.

**User story.** A data engineering team writes adapters that map real data sources (transaction DB, MouseLab exports, survey platforms, CRM notes) to the project's schemas. After validation, they run encoder training scripts on real data without modification.

**Business questions.**

| Question | Stakeholder | Decision enabled |
|----------|-------------|-----------------|
| "Can the prototype ingest our actual transaction logs without modification?" | Engineering Lead | Production integration roadmap |
| "What is the performance drop from synthetic to real MouseLab data?" | Research Director | Synthetic-to-real validity |
| "Which modality contributes most with noisy, incomplete real data?" | Data Scientist | Data collection budget |

**Technical description.**

Adapter pattern: each real data source gets an adapter that normalises to the corresponding schema.

| Real source | Target schema | Adapter responsibility |
|-------------|--------------|----------------------|
| MouseLab export (CSV/JSON) | AcquisitionEvent, TrialRecord, ChoiceSet | Convert acquisition events, map trial metadata, extract product attributes |
| Transaction DB export | TransactionRecord | Map SKU/price/timestamp to schema fields; resolve `product_id` to catalogue |
| Survey platform (Qualtrics etc.) | PsychographicVector | Map Likert scales to psychometric fields |
| Interview transcripts / CRM notes | PersonaNarrative | Embed via sentence-transformer |

Each adapter implements a common interface: `load() → Iterator[SchemaRecord]`. The existing encoder training scripts read the same JSONL files regardless of whether the data is synthetic or real.

**Data requirements.**

- Requires `Product` schema for real product catalogue mapping. Without it, adapters for transaction and MouseLab data are partially blocked (can produce AcquisitionEvent and TransactionRecord but not ChoiceSet).
- No new synthetic data. This is a pipeline capability.
- Real data: requires institutional access to MouseLab deployment, transaction logs, and survey panels. Not a code deliverable — an operational dependency.

**Success criteria:**
- **Integration test (plumbing only):** each adapter produces JSONL that passes schema validation, and encoder training on real data completes without code changes. This verifies the adapter interface — it says nothing about whether real data yields a useful CDT.
- **Validity gate (the real criterion):** measure the synthetic-to-real performance drop on the encoder probes (strategy recovery, individual recall@1, PersonaConfig R²) using a labelled real subset. Report the drop honestly; a large drop is an informative result, not a failure to hide. This — not "training completes" — is what licenses any production claim.

---

## Phase 0 Generator SPEC

> This section pins the decisions that the roadmap above leaves at the prose level, so
> that Phase 0 beads are creation-ready (two implementers write byte-identical acceptance
> assertions). All constants below are **declared, fixed values** — not tunables. They
> belong in `generator/SPEC.md` and a `generator/choice_model.py` module on implementation.

### 0.1 Attribute → float encoding (board display)

The board shows the first `n_attrs` names of `_ATTRIBUTES` (`price, brand, quality,
warranty, rating, features, availability, design`). Each maps to a float in `[0, 1]` for
`ChoiceSet.displayed_attributes`:

| Board attribute | Source `Product` field | Encoding → float in [0,1] |
|---|---|---|
| `price` | `price_normalised` | as-is |
| `brand` | `brand_tier` | ordinal: `premium=1.0, mid=0.66, value=0.33, own_label=0.0` |
| `quality` | `quality_score` | as-is |
| `warranty` | `warranty_score` | as-is |
| `rating` | `rating` | `rating / 5.0` |
| `features` | `features_score` | as-is |
| `availability` | `availability` | `1.0` if in stock else `0.0` |
| `design` | `design_score` | as-is |

`on_promotion` is catalogue metadata, **not** a board attribute (it is not in `_ATTRIBUTES`);
it influences `price_normalised` at catalogue-generation time, not at display time.

### 0.2 Per-attribute "goodness" (higher = more preferred)

The choice model operates on goodness `g_a ∈ [0,1]`, derived from the displayed float `v_a`:

- `price`: `g = 1 − v` (lower price is better)
- every other attribute: `g = v` (higher is better)

### 0.3 Choice utility per strategy (over inspected cells only)

For a trial, let `inspected(slot)` = the set of attributes whose `(slot, attr)` cell appears
in this trial's `AcquisitionEvent` stream. Utility is computed **only over inspected
attributes** — this is what couples choice to the trace (§ Generator Impact). `ū(slot) ∈ [0,1]`:

> **Uninspected slots are not choosable.** A slot with `inspected(slot) = ∅` (the consumer
> never opened any of its cells) is **excluded from the choice set** before utilities are
> computed — you cannot choose what you did not look at, and it avoids the `Σ w_a = 0`
> division-by-zero. After exclusion, let `C` = the inspected slots. If `|C| ≥ 2`, choose
> among `C` by the rules below. If `|C| < 2` (degenerate trial — at most one alternative
> inspected), set `choice = rng.choice(all_slots)` uniform and `choice_mechanism =
> "random_lapse"` (treat as an effective lapse: no real comparison occurred).

| Strategy | Utility rule |
|---|---|
| **Compensatory** | `ū(slot) = Σ_{a∈inspected} w_a·g_a  /  Σ_{a∈inspected} w_a` (weights renormalised over inspected attrs). Weights `w_a` from `StrategyParams.attribute_weights` when present; otherwise the default-weight rule (0.4). |
| **Lexicographic** | `ū(slot) = g_{first_attribute}(slot)` if `first_attribute ∈ inspected(slot)`, else fall through to the Compensatory rule for that slot. (Single-attribute utility reproduces "sort by key attribute, pick best".) |
| **Satisficing** | `ū(slot) = min_{a∈inspected} (g_a − aspiration_a)` using `StrategyParams.aspiration_levels` (missing attr aspiration defaults to `0.0`). Highest `ū` = best-satisfied alternative; positive `ū` means all inspected aspirations met. |
| **Affect heuristic** | `ū(slot) = g_brand(slot)` (brand-tier preference dominates). If `brand` not inspected for a slot, `ū = 0.5` (neutral). |
| **Adaptive** | Compensatory rule when `n_attrs ≤ 6`; Satisficing rule when `n_attrs = 8`. Mirrors the existing trace rule ("compensatory for simple boards, satisficing for complex"). |
| **Random** | No utility; see § 0.5. |

**Default-weight rule (0.4)** — when `attribute_weights` is absent: start from uniform
weight `1/n_attrs` over the board attributes, then set `w_price ← w_price·(1+price_sensitivity)`,
`w_brand ← w_brand·(1+brand_loyalty)`, leave others unchanged, and renormalise to sum 1.

**The `other` catch-all weight** — `personas.yaml` `attribute_weights` dicts name only a
subset of attributes (`price, quality, brand`) plus an `other` key (e.g. `other: 0.15`).
Expand `other` **equally across every board attribute not explicitly named in the dict and
present on this board**: if `U` = `{board attrs in inspected(slot)} − {explicitly-named
attrs}`, then each `a ∈ U` gets `w_a = other / |U|`. Explicitly-named attributes use their
stated weight. After expansion, the renormalisation `Σ_{a∈inspected} w_a` proceeds as in the
Compensatory rule. If `U = ∅` (every inspected attribute is explicitly named), the `other`
mass is dropped and the named weights renormalise among themselves. This makes the weight
vector total over the board well-defined for any `n_attrs ∈ {4,6,8}`.

### 0.4 LatentDeviation coupling

The **same per-trial `z = config.latent` realisation** that drives the acquisition sequence
also perturbs the choice (shared stochastic source — the coupling guarantee):

- `z.brand_lean`: additive shift to `w_brand` *before* renormalisation in the Compensatory
  and default-weight rules (`w_brand ← max(0, w_brand + 0.25·z.brand_lean)`).
- `z.impulsivity`: already raises the per-trial lapse probability in the trace simulator
  (`impulsivity_boost`); no separate choice-side effect — lapses are handled in § 0.5.

### 0.5 Softmax, decisiveness gain, and lapse

- **Preference choice** (`trial_strategy != RANDOM`): logits `ℓ(slot) = GAIN·ū(slot)`,
  probabilities `P = softmax(ℓ / τ)`, choice `= rng.choice(slots, p=P)`.
  - `GAIN = 8.0` (fixed). `τ = 1.0` (fixed — see the temperature box in § PersonaConfig;
    **never tuned to hit M1 calibration**). With `GAIN=8`, a 0.3 goodness gap → ≈11:1 odds —
    decisive but not degenerate, so M1 labels carry variance.
  - `choice_mechanism = "preference"`.
- **Random lapse** (`trial_strategy == RANDOM`, already decided upstream by
  `p_strategy_lapse + impulsivity_boost`): `choice = rng.choice(slots)` uniform;
  `choice_mechanism = "random_lapse"`.

`TrialRecord.final_choice` and `ChoiceSet.chosen_alternative` are both set to the chosen slot
letter. Determinism: given `(inspected values, StrategyParams, z, rng state)`, the choice
distribution is fully specified.

### 0.6 Product catalogue generation

New module `generator/product_catalogue.py`. One run produces one category's catalogue
(matching the single-`--category` pipeline contract). Output `data/synthetic/products.jsonl`,
one record per line, accumulated across categories.

- **Reproducibility & idempotency (required).** The catalogue is generated from a **fixed
  dedicated seed** `CATALOGUE_SEED = 20260101`, independent of the pipeline `--seed`, so a
  category's products are byte-identical across regenerations and across H1 waves. Generation
  is **generate-once per category**: if `products.jsonl` already contains rows for the target
  category, the run is a **no-op** (do not append duplicates, do not re-randomise). A
  `--force` flag overwrites that category's rows deterministically (same seed → same products).
  The product RNG is seeded as `CATALOGUE_SEED + hash(category)` so categories are independent
  but each is stable.
- **Count:** `N_PRODUCTS_PER_CATEGORY = 20`. (Floor is set by the 15% tiers, not the total: at 20 the `premium`/`own_label` sub-catalogues hold 3 products each — enough variety for tier-conditioned transaction sampling without exhaustion — while keeping per-product demand dense for M2. 10 would leave those tiers with 1–2 products; 60 only dilutes M2.)
- **IDs:** `product_id = f"{category}_{i:02d}"` for `i` in `0..19` — stable, category-scoped, unique.
- **Brand-tier mix:** `premium 0.15, mid 0.35, value 0.35, own_label 0.15` → exactly `3 / 7 / 7 / 3` at N=20 (assigned by quota, not sampled, so counts are exact).
- **Attribute generation** (per product, seeded): draw a latent quality factor `q ~ U(0,1)`.
  - `price_normalised = clip(0.15 + 0.7·tier_level + 0.15·q + N(0,0.05), 0, 1)` where `tier_level ∈ {premium:1.0, mid:0.66, value:0.33, own_label:0.0}`.
  - `quality_score, features_score, design_score = clip(0.4·q + 0.4·tier_level + N(0,0.1), 0, 1)` (positively correlated with price/tier → real tradeoffs).
  - `warranty_score = clip(0.5·tier_level + N(0,0.15), 0, 1)`; `rating = clip(2.5 + 2.0·q + N(0,0.3), 0, 5)`.
  - `availability = rng.random() < 0.95`; `on_promotion = rng.random() < 0.15` (and if on promotion, `price_normalised ← price_normalised·0.85`).

### 0.7 Trial choice-set composition

In `simulate_session`, after `n_alts` is drawn (`{3,5,7}`):

1. Sample `n_alts` products **uniformly without replacement** from the category catalogue.
   (Uniform, not persona-weighted — this guarantees attribute spread *within* the choice set
   so M1 sees easy and hard choices; persona preference enters the *choice*, not the *set*.)
2. Map product `k` to slot letter `chr(65+k)` → `alternative_products`.
3. Build `displayed_attributes[slot]` from the product's first `n_attrs` board attributes via § 0.1.
4. Run the existing acquisition-sequence simulator (unchanged) to get inspected cells.
5. Compute the choice via §§ 0.3–0.5; emit one `ChoiceSet` row per trial.

### 0.8 Transaction product selection

In `transaction_simulator.py`, replace the ephemeral `product_id = f"prod_{category}_{tier}_{rand}"`
line: after `brand_tier = _sample_brand_tier(...)`, select a product by sampling **uniformly
with replacement among catalogue products of that `brand_tier`** in the category, and use its
`product_id`. **With replacement is required** — a consumer repurchases over a transaction
history, so without-replacement would exhaust a small (3-product) tier after three purchases.
The existing `brand_tier`/`on_promo` sampling logic is unchanged; only the ID becomes a
catalogue reference. `TransactionRecord.product_id` now resolves against `products.jsonl`
(Phase 0 step 8 validates this).

> Contrast with trial choice sets (§ 0.7), which sample **without** replacement — a board
> cannot show the same product in two slots, but a purchase history can repeat a product.

---

## Implementation Sequence

### Phase 0: Schema Foundation

Prerequisite for M1, L2, M2. Must complete before choice-dependent capabilities.

1. Add `Product` schema to `schemas/`
2. Add `ChoiceSet` schema to `schemas/`
3. Add `choice_set_id` field to `TrialRecord`
4. Implement product catalogue generator
5. Rewrite trace simulator: preference-driven choice model **coupled to the inspected cells** (§ Generator Impact), pinned-temperature softmax, + `ChoiceSet` output
6. Update transaction simulator: reference product catalogue
7. Regenerate full synthetic dataset
8. Validate `TransactionRecord.product_id` values resolve to catalogue products
9. **Extend `generator/validate.py`** with three named cross-modal checks, then run it: (a) `ChoiceSet.chosen_alternative == TrialRecord.final_choice` for every trial; (b) every `ChoiceSet.alternative_products` value resolves to a `products.jsonl` `product_id`; (c) every `displayed_attributes` value is a float in `[0, 1]`. ("Revalidate" here means *extend then run* — `validate.py` has no `ChoiceSet`/`Product` awareness today.)
10. **Verify trace–choice coupling (measurable):** fit a logistic regression on the frozen trace-encoder embedding to predict `chosen_alternative`; require top-1 accuracy **≥ 1.5×** the `1 / mean(n_alternatives)` chance baseline on held-out trials. Below that, trace and choice are still decoupled — fix before M1.
11. **Check the retraining trigger:** compute the acquisition-token distribution (frequency over `attribute_id` token types across all `AcquisitionEvent` streams) and its symmetric KL vs. the prototype baseline (§ Invariant); retrain trace encoder if > 0.05 nats.
12. Revalidate encoder probes with `uv run python -m evaluation.run_probes`

### Phase 1: Independent Capabilities

No product dependency. Can proceed in parallel with Phase 0.

- **L3** (Cross-Session Stability) — validation protocol, new session generation needed but no product model
- **L1** (Churn Prediction) — works on existing data, no schema changes needed

### Phase 2: Choice-Dependent Capabilities

Requires Phase 0 complete.

- **M1** (Choice Prediction Head) — requires `Product` + `ChoiceSet` + regenerated data
- **M2** (Market Simulation) — requires M1 trained + `MarketState` schema
- **L2** (Ranking) — requires M1 trained

### Phase 3: Infrastructure Capabilities

Partially independent of product model. Can start in parallel with Phase 2.

- **H1** (Temporal Dynamics) — requires generator wave support (`wave_id`, `session_date`), no product model
- **H2** (Real Data Integration) — full coverage requires `Product` schema; partial adapters work without it

Within each phase, capabilities can be built in parallel.

---

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Preference-driven choices too deterministic (archetype policy → near-certain choice) | M1 training data has no variance | Use softmax temperature in the generator's choice model to inject stochasticity. Verify calibration on a diverse evaluation set. |
| **Trace and choice decoupled** (choice computed from full attribute matrix, not inspected cells) | M1 passes AUC floor via persona leakage; "decision process predicts decision" claim is false | Couple choice to inspected cells (§ Generator Impact). Gate M1 on **lift over a no-CDT baseline** (§ M1), not absolute AUC. Verify with the trace-only classifier check in Phase 0 step 10. |
| **Calibration circularity** (softmax temperature tuned to hit M1's Brier target) | M1 "calibration" measures only RNG recovery, not real-world calibration | Pin temperature as a fixed generative constant (§ PersonaConfig). Report synthetic calibration as a recovery check, never as a real-world claim. |
| **L3 bar met by frozen modalities** (psych/text near-deterministic in `PersonaConfig`) | Cross-session stability "passes" without testing behavioural stability | Make trace/transaction deltas the primary L3 metric (§ L3); fused delta is headline only. |
| **M2 validated only on baseline** | Counterfactual demand (M2's actual use) is unvalidated | Validate counterfactual demand shifts against an Option-B generator re-run (§ M2). |
| Synthetic data ceiling — all capabilities trained on 7 known archetypes | All capabilities overfit to archetype structure | Evaluate on held-out participants. Real-data validation before production claims. |
| Choice set size effects (3 vs. 7 alternatives) distort probabilities | M1 miscalibrated | Include `n_alternatives` as context feature. |
| Cross-session stability is poor (`similarity_delta < 0.10`) | L3, H1 invalidated | Report honestly. Temporal capabilities may need different architecture. |
| Product feature mapping differs between synthetic and real data | M1, L2, M2 fail on real data | Define Product schema with fields that map to both synthetic attributes and real catalogue features. |
| Schema changes break existing encoder probe results | Phase 0 cascades into Phase 1 rework | Run probes immediately after data regeneration. Investigate before proceeding if results change materially. |
| Missing modalities in real data (no survey, no MouseLab) | H2 partial coverage | Modality dropout (p=0.2) in fusion already handles missing modalities. Validate with held-out-modality experiments. |
| Real data access delayed 12+ months | H2 blocked | Continue on synthetic. All code is adapter-agnostic. |
| Stakeholder expects causal inference, not prediction | M2 misused for policy | Document: CDT predicts correlation, not causation. Randomised experiments required for causal claims. |
