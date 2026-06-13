It's a genuinely interesting addition, and the honest answer is: yes, it opens distinct business questions — but only if you extend the latent model so that process traces carry information that preferences don't, and only if at least one demo scenario is *unsolvable without it*. Otherwise it's decorative, and a technical audience will notice that ablating it costs nothing.

## What it uniquely encodes

Conjoint and transactions recover *preferences* — the $\beta_i$ in $U_{ij} = x_j^\top \beta_i + \varepsilon_{ij}$. Process traces recover the *decision procedure*: how the customer searches before choosing. Two customers with identical part-worths can differ completely here — one does exhaustive alternative-wise comparison, the other satisfices on the first acceptable option after checking price only. Utility models treat them identically; their behaviour under interventions diverges sharply.

That's the conceptual pitch: the rest of your modalities triangulate *what the customer wants*; this one captures *how they decide*. In latent-variable terms, you'd extend $z_i$ with a block of decision-style traits, deliberately weakly correlated with the preference block:

- search depth (how much of the information matrix gets inspected before choice),
- search orientation — Payne's search index, $SI = \frac{n_{alt} - n_{attr}}{n_{alt} + n_{attr}}$, where $n_{alt}$ is same-alternative transitions and $n_{attr}$ same-attribute transitions (alternative-wise vs attribute-wise processing),
- attentional weights over attributes (dwell share on price vs brand vs features), which can *diverge from* choice-revealed weights — that gap itself is informative,
- a strategy regime (weighted-additive vs lexicographic vs elimination-by-aspects), which you can implement as a mixture governing the trace generator.

The independence point is critical for the demo's integrity: if decision style is just a deterministic function of preferences, the modality is informationally redundant and fusion gains will be ~zero by construction.

## Which business questions it opens

**Personalised ranking — your existing scenario, but now actually deep.** For a satisficer with search depth 3, position is everything: an item ranked 6th effectively doesn't exist. For an exhaustive searcher, ranking barely matters. A CDT carrying search policy lets the ranking API optimise *position-weighted* expected utility rather than raw utility — and you can demonstrate the difference with simulated CTR under reranking. This is the cleanest "unsolvable without process data" scenario: a preferences-only model assigns these two customers identical rankings; ground truth (your generator) shows different optimal rankings.

**Choice prediction under changed choice architecture.** Your mid-tier SKU launch scenario gets a second layer: inserting a SKU doesn't just add a utility term, it changes the search problem. Context effects — compromise and decoy effects — emerge from process, not from stable utilities. A standard logit predicts IIA-style substitution; a process-aware model predicts the mid-tier option steals disproportionately from the extremes for attribute-wise searchers. You can bake a compromise effect into the generator and show that only the process-informed model anticipates it.

**Consideration-set realism in demand simulation.** Your pricing simulation currently assumes every customer evaluates the full assortment. With search policies, demand response to a price change depends on whether the price is *attended to* — low-attention-to-price customers under-react even if their elasticity parameter says otherwise. This separates "price-sensitive but not price-attentive" from "price-attentive" segments, which is a more refined story than the 22%/2% split, and is genuinely how attention-based discrete choice (e.g. consideration set models, rational inattention) frames it.

**Early-warning drift.** Attention shifts can precede behavioural shifts: a loyal customer who starts dwelling on price cells and opening competitor comparison rows is drifting *before* their purchase pattern changes. In your generator you can make attention drift lead behavioural drift by 2–4 weeks for the drift cohort, then show the process-aware drift detector flags customers earlier than the transaction-only detector. Lead time on churn signals is a metric any business audience understands, and the technical audience gets a clean ablation.

## The realism problem, and the honest framing

You're right that classic MouseLab is rare in production. The "truthful to real schema" goal needs a defensible vehicle, and there is one: **instrumented hover/exposure logging on a product comparison page or configurator**. Hover events over attribute cells, accordion expansions ("see full specs"), tab switches, comparison-table column adds — these are collectable with standard web analytics and some e-commerce sites do log them. Your schema then looks like a plausible front-end event log rather than a lab artefact:

```
customer_id | session_id | task_id | event_ts | event_type | alternative_id | attribute_id | dwell_ms
```

with `event_type ∈ {cell_hover, cell_open, column_add, sort_apply, choice}`. Frame it in the demo as exactly that — "this is what you'd get if you instrumented your comparison UI" — and acknowledge openly that it's the most speculative modality. Technical peers respect a labelled assumption far more than a hand-waved one. The second realistic vehicle: field it *jointly with the conjoint* on the same 250-customer subsample, as a MouseLab-style information board wrapped around the choice tasks — combining conjoint with process tracing is an established research design, so the provenance story is clean and you get preference and process data from the same instrument, same missingness pattern.

## Modelling and cost

The trace is a sequence of (alternative, attribute, duration) fixations — encode with a small GRU over embedded cell-IDs with dwell as a feature, or aggregate to engineered process metrics (SI, depth, dwell shares, re-inspection rate) for the baseline. At 1000 customers with traces on 250, the engineered-metrics version may well match the sequence encoder; show both.

Cost side, stated plainly: this adds the most generator complexity per modality of anything discussed — you need a search-policy simulator (the strategy mixture above), not just an outcome sampler, and you need to keep the choices it produces consistent with the conjoint utilities (the strategies should be implemented as different *evaluation procedures over the same underlying part-worths*, e.g. lexicographic = choose by the highest-attention attribute with tie-breaking, so process and preference data stay mutually coherent). Budget it as roughly equal effort to the clickstream generator. If scope pressure hits, the ranking scenario alone justifies it; the context-effects scenario is the most intellectually impressive but the fiddliest to calibrate.

One related concept worth a pointer in your demo notes: this whole framing — process data as evidence about the *policy* rather than the *reward function* — is structurally the same problem as inverse reinforcement learning vs behavioural cloning, and rational-inattention models in economics give you the normative scaffolding if anyone asks why attention should be a stable individual trait at all.