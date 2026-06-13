Good constraints. 1000 customers is workable for technical peers if you keep encoders small and lean on the multi-task + contrastive signal — but it pushes you toward modest architectures (a 2-layer GRU, not a Transformer with 8 heads) and makes the GBM baseline even more dangerous to your neural model. Worth accepting; honest comparisons land better with a technical audience anyway.

## The modality set

I'd generate five, chosen so each contributes *distinct, partially non-redundant* information about the latent traits. That's the design principle: if two modalities are informationally equivalent, fusion is theatre.

**1. CRM / transaction history** (temporal, sparse, event-based). This is the one missing from your original list and it's the most important — no business has conjoint data for every customer, but everyone has transactions. Realistic schema: order-line table.

```
customer_id | order_id | timestamp | sku | qty | unit_price | discount_applied | channel | payment_method
```

What it encodes from $z$: price sensitivity (response to discounts, price-tier of purchased SKUs), brand loyalty (share-of-wallet concentration), category involvement (breadth, frequency), and churn propensity (inter-purchase time distribution). Generate via a per-customer purchase process — a non-homogeneous Poisson or simply a hazard model on inter-purchase gaps where the rate depends on $z_i$ and seasonality, then a choice model picks the SKU given a purchase occasion.

**2. Web clickstream** (temporal, dense, session-based). Realistic schema is event-log, not pre-aggregated:

```
customer_id | session_id | event_ts | event_type | page_type | sku_viewed | referrer | device | dwell_ms
```

Event types: `page_view, product_view, add_to_cart, remove_from_cart, search, filter_apply, checkout_start, purchase`. Real clickstreams are messy: ~30–60% of sessions are short bounces, identity resolution is imperfect (drop 10–15% of sessions to "anonymous" to be faithful), and dwell times are log-normal with heavy tails. Generate as a Markov session model whose transition matrix is a function of $z_i$, session intent (browse/research/buy — sample intent per session), and device.

**3. Conjoint study output** (static snapshot, *sampled subpopulation*). Here's the realism point most demos miss: a business never has conjoint for the full base. Run the synthetic study on ~250 of your 1000, with the realistic deliverable shape — both layers:

```
# Raw choice tasks (what the platform exports)
respondent_id | task_id | concept_id | attr_price | attr_brand | attr_size | attr_feature | chosen (0/1)

# HB utilities (what the insights team actually circulates)
respondent_id | beta_price | beta_brandA | beta_brandB | ... | RLH
```

Generate raw choices from the true mixed logit, then actually *estimate* HB part-worths from those choices (PyMC or a cheap hierarchical MAP) rather than exposing $f(z_i)$ directly. The estimation noise is the realism — and it gives you a nice technical-audience subplot: the encoder sees noisy HB estimates, not truths, and you can show how recovery degrades with respondent task count. Link to the full base via customer_id for the 250; the other 750 have this modality missing. That missingness is your masked-fusion stress test, for free.

**4. Survey / psychographic profile** (static or slowly refreshed). Two realistic formats — pick one or do both:
- Structured: Likert batteries (brand attitude, price consciousness scale, innovativeness), NPS, satisfaction. Schema: wide table, one row per customer, with item-level responses including realistic patterns (straight-lining for ~5% of respondents, acquiescence bias, item non-response).
- Unstructured: open-ended verbatims ("Why did you choose us?"), 1–3 sentences. This replaces your "persona description" — real businesses don't have persona paragraphs per customer, they have verbatims and call-centre notes. Generate by conditioning an LLM on trait vector + recent behaviour, with deliberate noise (some verbatims uninformative: "it was on sale lol").

**5. Campaign interaction log** (temporal, sparse). Email/push: sent, opened, clicked, unsubscribed, with campaign metadata (discount depth, category). Schema:

```
customer_id | campaign_id | sent_ts | campaign_type | discount_pct | opened | clicked | converted | unsub
```

This one earns its place because it's the *intervention* modality — it's what makes your temporal-dynamics scenario (escalating offers to drifting customers) closeable as a loop, and it gives the demand simulator a second counterfactual lever beyond price.

I'd stop at five. Each additional modality past this point adds generator code linearly but demo insight sublinearly. Candidates I'd consciously exclude and say so: reviews (redundant with verbatims), customer service transcripts (high LLM-generation cost, low marginal signal), loyalty-app telemetry (clickstream clone).

## Temporal structure: which modalities, and how

Think of it as three temporal classes, and make the distinction explicit in your architecture — this is the part technical peers will probe.

**Class A — event streams (transactions, clickstream, campaigns).** These carry time natively. The key generator decision: put *dynamics in the latent state*, not just in the events. Concretely, let $z_{i,t}$ evolve as a slow stochastic process — e.g. hold most dimensions fixed but give loyalty and churn-propensity an AR(1) or piecewise drift for a designated cohort:

$$z_{i,t} = \alpha z_{i,t-1} + (1-\alpha)\mu_i + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2 I_{\text{dyn}})$$

with regime shifts injected for ~10–15% of customers (loyalty decay after a bad delivery experience, say — you can even log the triggering event). This is what makes drift detection a real task: the detector must distinguish latent drift from behavioural noise, and you have ground-truth drift labels to score it against.

**Class B — snapshot modalities (conjoint, survey).** Static, but *timestamped*. Stamp each with a fielding date. This creates a subtle, realistic problem worth surfacing in the demo: a conjoint from month 1 describes $z_{i,1}$, but you're scoring the customer at month 6 after drift. Attitudinal data goes stale. Your fusion module should receive modality age as an input (or you show empirically that prediction error grows with snapshot staleness — a great chart for this audience).

**Class C — the embedding itself.** Decide whether the CDT is a point-in-time encoding of a trailing window (encode last 90 days of events + latest snapshots → $e_{i,t}$, recomputed weekly) or a recurrent state updated online. For the demo, do the trailing-window version — it's simpler, matches how batch CRM systems actually work, and drift detection becomes displacement in embedding space: flag when $\|e_{i,t} - e_{i,t-4w}\|$ (or cosine distance) exceeds a calibrated threshold, scored against your ground-truth drift cohort with precision/recall.

Practical timeline: simulate 12 months of event data at daily resolution, with conjoint fielded month 1–2, surveys at months 1 and 7, campaigns throughout, and drift injected from month 6. That gives you a clean train (months 1–8) / evaluate (9–12) temporal split — use a temporal split, not random; your audience will check.

One sequencing suggestion: build the generator as a standalone, seeded, parameterised package with the trait→behaviour mappings documented, and validate the *synthetic data itself* before any encoder work — marginals (inter-purchase time distributions, session length distributions vs published e-commerce benchmarks), and a sanity check that an oracle model given true $z$ predicts the generated outcomes well. For a technical audience, a credible generator is half the demo's persuasive weight; if the data looks toy, nothing downstream rescues it.