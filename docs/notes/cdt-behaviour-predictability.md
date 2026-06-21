# CDTs and Behaviour Predictability — Two Tasks, One Ceiling

> Position note. Written 2026-06-21, after the `3fx` epic (`c9v`, `b8b`, and the
> fusion-level choice-loss experiment) plus a literature review of the
> "digital-twins-can't-predict-behaviour" debate. Captures the distinction the
> field keeps blurring, the evidence, and how this project's experiments
> localise the mechanism.

## Thesis

Two claims circulate simultaneously and look contradictory:

- *"Consumer Digital Twins (CDTs) are bad at predicting behaviour"* (insights / consumer-research community).
- *"Recommender systems predict behaviour very well, and they run de facto CDTs on behavioural traces."*

They are **not contradictory** — they refer to **two different tasks**. Resolving the confusion tells us what a CDT is actually for, and why our M1 (choice prediction from a static CDT) failed in exactly the way the literature predicts.

## The two tasks

**Task A — in-stream behavioural prediction (what recommenders do).**
Predict behaviour at *t+1* from behaviour at *<t* **in the same modality/stream** (past clicks → next click; past views → next view; past purchases → next purchase). This is temporal auto-correlation within one behaviour modality. A recsys "CDT" is really a *compressed history in a single stream*. It works brilliantly — but it is close to tautological: predicting behaviour from prior behaviour *of the same kind*. It does **not** test "does a deep *who-you-are* profile predict behaviour."

**Task B — trait-to-instance / cross-modal behaviour prediction (the community's actual critique).**
Predict a specific behaviour — often in a *new* modality or a single *instance* — from a fused profile of *who the person is* (traits, psychographics, demographics, other modalities). "Given WHO you are, WHAT will you do here?" This has a hard ceiling, because a specific action is driven by the **immediate situation/state**, not by stable traits.

> **The community's "CDTs are bad at behaviour" means Task B. Recommender success is Task A. There is no contradiction; conflating the two is the error.** The behaviour CDTs miss is **specific, instance-level, context-dependent behaviour** — the action taken *in this situation* — not the habitual in-stream behaviour that recsys captures by auto-correlation.

## Evidence from the literature

**Stanford — *Generative Agent Simulations of 1,000 People*** (Park et al. 2024, [arXiv 2411.10109](https://arxiv.org/abs/2411.10109); [Stanford HAI](https://hai.stanford.edu/policy/simulating-human-behavior-with-ai-agents)). LLM agents of 1,052 real people built from 2-hour interviews. Split cleanly along the fault line:

- **Right (stable / trait-level):** Big Five personality and social-survey **attitudes** — the celebrated ~85%. Near-circular: an attitude/personality is a stable self-report, and the interview *contains* it. It simulates **tendencies**, not **actions**.
- **Wrong (instance / contextual behaviour):** the **economic games** — dictator, trust, public goods, prisoner's dilemma. Numerical/strategic decisions in a *specific* situation. Agents systematically diverged (e.g. more cooperative/altruistic than humans). Documented limits: tests survey/experiment responses, not real behaviour; no social dynamics; no longitudinal change; self-report bias.

**Columbia mega-study — *A Mega-Study of Digital Twins…*** (2025, [arXiv 2509.19088](https://arxiv.org/html/2509.19088v3)). Large public dataset; the decisive verdict:

- Digital-twin answers correlate only **~0.2** with the actual human's answers.
- Twins show **reduced variability** — they flatten individual differences back toward the archetype mean. Companion critique: ["funhouse mirrors"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5518418) — insufficient individuation, stereotyping, representation bias.
- They "capture patterns of judgment, not the lived interiority"; per one reviewer, digital twins "[can't simulate human ignorance](https://theboxinnovation.ai/digital-twins-vs-synthetic-users-vs-synthetic-data-the-complete-guide-to-ai-powered-customer-research-2025/)."

**The personality paradox** ([Youyou/Kosinski/Stillwell, PNAS 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4313801/); psychology canon). Digital traces predict **traits** better than humans can — but traits account for only a modest fraction of variance in any **single behaviour**. Situation dominates.

## Localisation — this project's three experiments

The `3fx` epic reproduced this ceiling **and explained its mechanism**. Same lift gate (lift over a product-only baseline, ≥0.05 to pass), regenerated data with oracle ceiling **0.876** (the choices are deeply learnable from the right information):

| experiment | choice objective applied at | M1_AUC | lift_over_product |
|---|---|---|---|
| `c9v` | nowhere | 0.5681 | +0.0035 |
| `b8b` | trace encoder (per-trial embedding) | 0.5597 | −0.0048 |
| fusion-level | the CDT itself (participant summary) | 0.5718 | +0.0072 |

All ≈ 0, regardless of where the choice objective is applied. **Smoking gun:** the fusion choice head's BCE plateaued at **≈0.50 — the base rate**. From `(participant CDT, product features)` the model cannot beat "guess the base rate." The participant-level CDT carries **~zero per-trial choice information**, and no choice objective (trace, fusion, or both) can put it there — because *which cells were inspected* is a property of the **trial**, not the **participant**. You cannot train information into a representation that has already averaged it away.

> **This is the personality paradox, stated mechanically.** Traits capture the *tendency*; the specific action is set by the *situation/state*; a participant-level profile is the wrong granularity for state. The Stanford agents nail attitudes but miss economic games for the same reason, and the mega-study's ~0.2 with flattened variability is the same phenomenon at scale.

## Implications for the CDT project

1. **The CDT is not broken.** It does what CDTs do well (predict stable consumer structure: identity, archetype, traits — the project's proven 70% identity recall, 100% archetype recovery, R² 0.79–0.96) and fails at what *all* CDTs fail at (instance behaviour). We reproduced the field's ceiling and localised the cause to representation granularity — not data, not objective.

2. **Trait-CDT and recommender systems are complementary, not competing.**
   - *Predict the next action* → a **stateful, in-stream sequence model** (recsys). Condition on recent behaviour.
   - *Predict / simulate stable consumer structure* → the **trait-CDT**.
   - Making the trait-CDT do the recsys job (next-behaviour from a static profile) is the **category error** — and it is the error the field, and this project's M1, both made.

3. **Defensible vs indefensible CDT claims.**
   - *Defensible:* identity / archetype / trait recovery; segmentation; personalization keyed to type; cross-channel identity resolution; cold-start; **type-level** counterfactual simulation ("how would the price-sensitive segment respond to a 10% rise?").
   - *Indefensible:* next-behaviour / instance-action prediction; anything requiring real-time state.

4. **Fusion must be earned per task.** The holistic (multi-modal) CDT is justified only where cross-modal fusion adds signal beyond simpler baselines (RFM, demographics, single-modality embeddings) — an empirical question per use case. The project's own ablation (trace −10.5%, psychographic −4.5% when dropped) shows modalities *are* non-redundant for the identity task; for choice they added nothing. Validate fusion-lift per trait-level task; do not assume the holistic embedding always wins.

## Decision frame

- **Goal = "predict this person's next action"** → build a recsys-style stateful model. The CDT is a complementary asset, not the predictor.
- **Goal = "understand / simulate customer types, personalise, resolve identity, cold-start"** → the CDT is the right tool. Re-pivot the flagship there; stop spending on instance-behaviour gates.

## Is there a purpose for a *holistic* CDT? (Business scenarios)

Yes — but the honest version is narrower than the hype, and the right question is not "can it predict behaviour" (settled above: no, not the instance kind). It is: **does fusing many modalities beat simpler baselines on stable-structure tasks, enough to amortise its cost?** For any *single* task a specialised model usually wins; a holistic CDT only earns its keep from three things — **cross-modal non-redundancy, cold-start/identity value, and amortisation across a family of tasks.**

### Where a holistic CDT genuinely wins

| Scenario | Why the holistic CDT (not recsys, not demographics) |
|---|---|
| **Cold-start personalisation** | A new user has almost no behaviour history — recsys is crippled. But you can fuse the *sparse* signals you do have (one survey, one browsing trace, declared preferences) into a CDT that generalises to their type. Inherently multi-modal *because* no single modality has enough signal. |
| **Cross-channel identity resolution / Customer 360** | "Is this the same person across web/app/store/email?" is a *cross-modal matching* problem. The project's proven 70% dropout-view identity recall is exactly this. Demographics/RFM can't; a fused behavioural signature can. |
| **Deep segmentation / audience architecture** | Clustering by *behavioural type* (decision-style, not just last-purchase or age). Powers media targeting, creative personalisation, portfolio management — the psychographic-targeting lineage (traits → targeted messaging). |
| **Personalisation keyed to *type* (not next-item)** | "Which tone / offer / framing fits this customer's style?" Matches the *experience* to the *person*; the CDT gives the type, a head maps type → creative. |
| **Type-level counterfactual simulation** | "If we raise prices 10%, how does the price-sensitive *segment* respond?" Aggregate/type-level — tractable even though *instance-level* simulation fails (per the three experiments). This is the vision's original "simulate under novel conditions" framing, at the right granularity. |
| **Synthetic customer panels for research** | Large-N synthetic consumers with realistic cross-modal structure to test surveys/products/messaging cheaply. Real market (Stanford-1000-style). **Caveat:** ~0.2 instance fidelity — good for directional/type-level read, not precise forecasting. |
| **Lookalike / audience expansion** | Seed high-value customers → find others with similar CDT profiles. Cross-modal similarity beats single-modality for "behaviourally similar" audiences. |

### When it is *not* worth it (the honest caveat)

- **One task → build a specialist.** A holistic CDT is a *platform*, not a predictor. Need only churn scores? A GBDT on RFM + support features beats it at a fraction of the cost.
- **Next-behaviour → recsys wins.** In-stream sequence models dominate; the CDT adds ~nothing (the lift table).
- **The 80/20 trap.** For many trait tasks, simpler baselines (demographics + RFM + a purchase-history embedding) capture most of the value. The holistic CDT must justify itself by *marginal* cross-modal signal *plus* amortisation *plus* cold-start/identity/unification value. If those don't materialise, it is over-engineering.

### Unifying framing

A holistic CDT is best understood as a **foundation model of the consumer, not an application.** Like a language backbone, its value is amortised across many downstream readers (segmentation head, identity head, cold-start head, type-simulation head), and it is justified only when (a) the task family is broad enough to amortise the cost, (b) modalities are non-redundant across that family (the project's ablation hints yes — trace −10.5%, psychographic −4.5% when dropped), and (c) cold-start / identity / unification genuinely matter. That *is* the Gartner "digital twin of the customer" positioning — the field's error is selling it as a behaviour *predictor*, the one job it provably cannot do.

> **One-line test for "should I use a CDT here?":** the CDT wins where signals are sparse, cross-modal, or about *who* someone stably is — and loses where the question is *what they do next*, which abundant in-stream data already answers.

### What this means for this project, concretely

The purpose is real and is what the market buys (CDP enrichment, personalisation, identity resolution, synthetic research). To claim it credibly rather than assert it, **validate fusion-lift over simpler baselines on one or two trait-level tasks** — e.g.:

1. **Cold-start personalisation** — fuse sparse signals for held-out new users; does the CDT beat a popularity/demographic baseline at predicting their *type* (not next item)?
2. **Cross-channel identity** — already half-proven (70% recall); sharpen into a business claim (deduplication / matching accuracy vs a demographic baseline).

If the holistic fusion beats single-modality baselines on those, the project has a *defensible* value prop: *"a reusable, cross-modal representation of stable consumer structure that works where single-stream models can't — cold-start, identity, sparse-signal personalisation."* That is a real product. It is just a different product than "predicts this person's next choice."

**Build the former; partner with (don't compete with) recsys for the latter.**

## Sources

- [Stanford HAI — Simulating Human Behavior with AI Agents](https://hai.stanford.edu/policy/simulating-human-behavior-with-ai-agents)
- [Park et al., Generative Agent Simulations of 1,000 People (arXiv 2411.10109)](https://arxiv.org/abs/2411.10109)
- [Columbia mega-study (arXiv 2509.19088)](https://arxiv.org/html/2509.19088v3)
- [Digital Twins are Funhouse Mirrors (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5518418)
- [Youyou, Kosinski, Stillwell — Computer-based personality judgments (PNAS 2015)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4313801/)
- Internal: `docs/post-mortems/m1-postmortem.md` §13 (the three-experiment results and convergent conclusion).
