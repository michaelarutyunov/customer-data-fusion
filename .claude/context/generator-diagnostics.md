  ---
  ★ Insight ─────────────────────────────────────
  The single most important architectural finding: there is a partial shared-individual latent already (brand_loyalty is noised once into PersonaConfig and read by both transaction and psychographic generators), but it's (a) too weak, (b) bypassed entirely by the narrative generator, and (c) never wired into the trace generator. The fix is to strengthen and complete an existing mechanism, not invent one.
  ─────────────────────────────────────────────────

---

## v0.2 Generator Updates (schema-update epic, 2026-06-13)

The original redesign (above) established the 5-axis `z` and the cross-modal consistency mechanism. The schema-update epic extended the generators:

| Generator | Change | Calibration target |
|---|---|---|
| `persona_sampler` | 5→7 axis `z`; AR(1) temporal drift over 12 months; regime shifts (~6–12% of cohort) | Drift only on loyalty/churn + attentional_bias; stable traits fixed |
| `trace_simulator` | Per-individual strategy mixture (softmax over z logits); attentional-weight divergence via `attentional_bias`; `EventType` enum; EBA strategy | Payne Index per archetype ±0.1; dwell-share diverges from choice weights when attentional_bias ≠ 0 |
| `transaction_simulator` | Order-line fields (sku, unit_price, discount, payment_method); hazard-model Poisson timing; 27-SKU catalog | discount ↔ price_sensitivity r > 0.3; months 11–12 at 1.8× volume |
| `clickstream_generator` (new) | Markov session model; 3 intent-specific transition matrices; realistic noise | 30–60% bounce sessions; 10–15% anonymous; log-normal dwell |
| `campaign_generator` (new) | Sigmoid response model; archetype-weighted dispatch; unsub feedback | 20–40% open; 5–15% click; <5% cumulative unsub |
| `pipeline` | 12-month temporal loop; partial trace coverage (250/1000); month-partitioned output; drift labeling | Ground-truth drift labels via `get_drift_metadata()` (not threshold detection) |

**Key design note:** the drift label is now ground-truth (from `sample_temporal_trajectory`'s regime-shift metadata), not a post-hoc threshold on `brand_lean` deviation. Threshold detection conflated regime shifts with ordinary AR(1) stochastic drift (36% false-positive rate); ground-truth metadata gives the targeted ~6–12%.

---

  Generator Redesign: Achieving Individual-Level Variation in the CDT

  1. Diagnosis per modality

  The root cause is shared across all four, then specialised. The shared cause: between-archetype parameter gaps are large (e.g. involvement_score 0.20 → 0.80) while per-participant noise is tiny (_NOISE_SCALE = 0.15 at the sampler, std_factor = 0.05 at the psychographic generator). With a signal-to-noise ratio this high, every modality is a near-direct readout of the archetype label.

  Traces — least broken (95%, not 100%)

  - trace_simulator.py:83 — mean dwell_ms is a hard dictionary lookup keyed on config.persona_id (_DWELL_MU.get(config.persona_id)). This is a literal label leak: dwell distribution is indexed by the archetype name, not by any continuous parameter.
  - _simulate_lexicographic (SPEC.md:88) inspects only the first_attribute column, producing PI = -1.0 deterministically except on a p_strategy_lapse trial (price_lex: 0.08). So 92% of price_lex trials are byte-identical in structure.
  - It survives at 95% (not 100%) only because the acquisition sequence is genuinely stochastic (random alternative order, log-normal dwells, reinspection draws). Traces are the one modality with real within-archetype variance — and it's the only one with a non-zero ablation delta (−10.45%).
  - What's missing: the trace generator reads config.strategy and nothing else. brand_loyalty, price_sensitivity, involvement — none reach it. So a "brand-loyal price_lex" produces identical traces to a "brand-indifferent price_lex."

  Transactions — weakest classifier (62%) but for the wrong reason

  - The mappings are reasonable: price_paid_normalised via Beta(4−3s, 1+3s) on price_sensitivity; brand_tier via a loyalty-weighted mixture. These do carry continuous individual signal.
  - It's weak as a classifier (62%) because several archetypes have overlapping price_sensitivity/brand_loyalty bases (satisficer 0.55/0.40 vs adaptive 0.55/0.45 vs low_involve 0.50/0.30) — they're already confusable. This is the one modality already near the 65–80% target, by accident.
  - But it does not contribute to cross-modal identity because its individual draws are decorrelated from the other modalities (see §4).

  Psychographic — saturated (100%)

  - Double-noise stack, both off the same seed: persona_sampler._noisy applies std = 0.15·base, then psychographic_generator._noisy applies another std = 0.05·base. Net CV ≈ 0.16 — still tiny against the between-archetype gaps.
  - Two pure label leaks:
    - decision_style_dominant (psychographic_generator.py:104) is a deterministic dictionary lookup on strategy.primary_strategy. The survey literally contains a transformed copy of the archetype's strategy enum.
    - price_consciousness is a 3-level categorical (LOW/MED/HIGH → 0.2/0.5/0.85) plus 5% noise — effectively a discrete class label.
  - A linear probe trivially separates 7 well-spaced Gaussians in 6-D. Hence 100%.

  Narratives — saturated (100%) — the smoking gun

  - _build_narrative (persona_sampler.py:182) takes no rng and applies zero noise. It copies decision_style_description verbatim from the YAML. Every one of the ~143 participants in an archetype receives a byte-identical prompt, including the same one-sentence strategy description ("Scans prices across all options first and immediately rejects…").
  - The sentence-transformer is frozen. Identical prompts → near-identical narratives → near-identical embeddings → 7 tight points in embedding space. The text encoder isn't learning decision style; it's clustering 7 pre-written paragraphs. This is the clearest case of "too faithful."

  ---
  2. Within-archetype variation design

  The mechanism: replace scattered per-modality jitter with one per-participant latent deviation vector z sampled once, stored on PersonaConfig, and read by every generator. This is the only way to make within-archetype variation consistent across modalities (§4) rather than independent noise.

  Latent definition. For participant i in archetype a:
  z_i ~ N(0, I_k),  k ≈ 5   # e.g. [price_lean, brand_lean, thoroughness, impulsivity, openness]

  Each archetype parameter θ is then a base + loaded deviation, not base + isotropic noise:
  θ_i = clip( θ_a + σ_θ · (L_θ · z_i) , lo, hi )

  where L_θ is a fixed loading row mapping the latent axes onto that parameter (so price_lean moves price_sensitivity, price_consciousness, price-trace
  behaviour, and price language together), and σ_θ is a per-parameter spread.

  Distributions / ranges:
  - Bounded [0,1] params (involvement, maximiser, etc.): use a logit-normal rather than clipped-normal, so the clip at boundaries doesn't pile up mass and create a spurious spike. θ = sigmoid(logit(θ_a) + s·u), u ~ N(0,1).
  - Set spread so within-archetype SD ≈ 35–45% of the nearest between-archetype gap. Concretely, raise the effective _NOISE_SCALE from 0.15 toward 0.30–0.40
  — but calibrate empirically (§5), don't hard-code.
  - price_consciousness: stop using a 3-level categorical. Make it a continuous [0,1] field driven by z.price_lean; derive the LOW/MED/HIGH enum from the continuous value only where a categorical is required downstream.

  Preventing collapse of between-archetype signal. The risk is that large z spread erases the class. Two guards:
  1. Cap the per-parameter spread at a fraction of the gap to the nearest neighbouring archetype on that parameter (not a fixed fraction of the base).
  Compute these gaps once from the YAML; this guarantees archetypes stay distinguishable in aggregate while individuals roam within their basin.
  2. Keep the strategy enum discrete and noise-free. Within-archetype variation lives in continuous parameters and acquisition stochasticity; the categorical strategy stays fixed. This anchors the class so recovery floors at a meaningful level rather than degenerating to chance.

  ---
  3. Between-archetype overlap design

  Overlap is induced by bringing base parameters closer on the modality where two archetypes should be confusable, combined with the §2 spread so their distributions interpenetrate. Express target overlap as the Bhattacharyya coefficient (BC) between the two archetypes' per-parameter (or per-modality-feature) distributions; BC = 1 is identical, 0 is disjoint.

  ┌────────────────────────────┬──────────────────────┬─────────────────────────────────────────────────────────────────────────────┬──────────────────┐
  │            Pair            │ Modality where they  │                                  Mechanism                                  │    Target BC     │
  │                            │       overlap        │                                                                             │                  │
  ├────────────────────────────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────────┼──────────────────┤
  │ satisficer ↔ compensatory  │ psychographic        │ maximiser_score bases (0.25 vs 0.75) are too far — move to 0.45 vs 0.60 and │ 0.5–0.65         │
  │                            │                      │  widen spread                                                               │                  │
  ├────────────────────────────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────────┼──────────────────┤
  │ price_lex ↔ satisficer     │ transaction          │ price_sensitivity 0.85 vs 0.55 → already close; widen spread so             │ 0.6–0.7          │
  │                            │                      │ price-percentile Betas overlap                                              │                  │
  ├────────────────────────────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────────┼──────────────────┤
  │ brand_affect ↔ quality_lex │ transaction          │ both low price-sens, high loyalty → already near; let brand-tier mixtures   │ 0.55–0.7         │
  │                            │                      │ overlap                                                                     │                  │
  ├────────────────────────────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────────┼──────────────────┤
  │ adaptive ↔ {satisficer,    │ traces +             │ adaptive is defined as a mixture; make its per-trial strategy               │ 0.5–0.6 (each)   │
  │ compensatory}              │ psychographic        │ stochastically switch between satisficing/compensatory                      │                  │
  ├────────────────────────────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────────┼──────────────────┤
  │ low_involve ↔ everything   │ traces               │ random strategy already overlaps; keep                                      │ high by          │
  │                            │                      │                                                                             │ construction     │
  └────────────────────────────┴──────────────────────┴─────────────────────────────────────────────────────────────────────────────┴──────────────────┘

  Critical constraint: overlap must be modality-specific, not global. satisficer and compensatory should be confusable from psychographics but separable from traces (compensatory inspects deeply, PI≈0; satisficer is mid-depth, PI≈−0.4). This is exactly what forces fusion to combine complementary views — the property H2 wanted and didn't get. If every pair overlaps in every modality, you've just recreated the trivial problem at lower accuracy.

  Target: no single modality should separate all 7 archetypes; each modality cleanly separates only a subset, and only the union of views recovers all 7.

  ---
  4. Individual consistency constraint (the hard one)

  What exists today: a partial mechanism. persona_sampler noises brand_loyalty once into the shared PersonaConfig; that same noised value drives both transaction brand-tier concentration and psychographic brand_sensitivity (= _noisy(txn.brand_loyalty)). So transaction↔psychographic individual consistency is real but too weak to survive, and it's diluted by the second independent noise layer.

  What's broken:
  - (a) The narrative generator reads none of it (no rng, verbatim text).
  - (b) The trace generator reads only config.strategy — brand_loyalty never reaches it, so the task's own example ("brand-loyal price_lex → more brand inspections") is literally unwired.
  - (c) Per-modality generators each call np.random.default_rng(config.random_seed) and consume the stream independently — so even shared-seed draws decorrelate across modalities.

  The mechanism — the shared latent z from §2 is the consistency device. Sample z_i once at sample_persona; store it as a new (immutable) field on PersonaConfig. Every generator reads the same z_i and projects it into its own modality. For an unusually brand-loyal price_lex (z.brand_lean high):

  ┌───────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │   Modality    │                                                      How z.brand_lean enters                                                      │
  ├───────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Traces        │ add a brand-inspection bias: probability of inspecting the brand column scales with z.brand_lean; reinspection of chosen brand ↑  │
  ├───────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Transactions  │ already wired via brand_loyalty — keep, strengthen                                                                                │
  ├───────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Psychographic │ brand_sensitivity driven by the same z.brand_lean, not a re-noised copy                                                           │
  ├───────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Narrative     │ inject z into the prompt as graded descriptors ("fiercely loyal to one or two trusted brands" vs "open to switching") — see below │
  └───────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  Why this doesn't reduce to reading the label: z is sampled within the archetype and is orthogonal to the archetype mean. Two price_lex participants share the strategy enum but differ in z; the cross-modal signal that ties P037's traces to P037's transactions is z_037, which is not recoverable from the archetype label (every price_lex has the same label but a different z). A model that only reads the label cannot do cross-modal retrieval; a model that recovers z can. That is precisely the individual-vs-archetype distinction the prototype is trying to test.

  Narrative implementation (replaces the verbatim copy): give _build_narrative the rng/z and construct the decision_style_description from a small templated lexicon graded by z — bin each relevant latent axis into 3 levels and select phrasing accordingly, optionally with paraphrase variety from the LLM. This breaks the byte-identical-prompt pathology and carries individual signal in language, satisfying the task's "language about trusted brands" requirement.

  ▎ Honest limit (carried into §7): even a perfect z shared across modalities will not by itself produce recall@1 > 0.1 or visible UMAP spread, because the fusion objective is 7-class cross-entropy, which provably discards within-archetype variation (`docs/post-mortems/phase2b-postmortem.md` §4 says exactly this and names contrastive learning as the fix). The generator makes individual identity present in the data; only a metric/contrastive training objective makes it survive into the embedding. These two targets are generator-necessary but not generator-sufficient.

  ---
  5. Calibration targets

  I will not fabricate the exact CV / BC that yields 70% recovery — it depends on the classifier, feature dimensionality, and probe regulariser, none of which can be derived from inspection. What's reliable is the monotonic relationship and an empirical calibration loop.

  Directional targets (starting hypotheses, to be tuned):

  ┌───────────────┬───────────────────────────────────┬───────────────────────────────────────────────────────────────────┐
  │   Modality    │ Within-archetype CV (target band) │                               Notes                               │
  ├───────────────┼───────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ Traces        │ 0.30–0.45                         │ already highest variance; mostly needs z-wiring, not more noise   │
  ├───────────────┼───────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ Transactions  │ 0.30–0.40                         │ near target already; add overlap, not noise                       │
  ├───────────────┼───────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ Psychographic │ 0.30–0.40                         │ up from ~0.16; remove label leaks first                           │
  ├───────────────┼───────────────────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ Narrative     │ n/a (CV ill-defined on text)      │ measure via embedding-space within/between variance ratio instead │
  └───────────────┴───────────────────────────────────┴───────────────────────────────────────────────────────────────────┘

  Between-archetype overlap: target BC ≈ 0.4–0.6 for the designed-overlap pairs (§3), BC < 0.2 for pairs that should stay separable in that modality.

  Calibration loop (the actual deliverable):
  1. Set a global spread parameter s (start s = 0.30).
  2. Regenerate dataset.
  3. Run `uv run python -m evaluation.run_probes` → per-modality single-modality recovery.
  4. If any modality > 80%: increase its spread / overlap. If < 65%: decrease.
  5. Check cross-modal: measure within-vs-between embedding variance ratio (proxy for recall potential).
  6. Repeat until all 4 modalities land in 65–80% single-modality recovery.
  The target band 65–80% is the stopping condition; CV/BC are the dials, not the goals.

  ---
  6. Implementation plan (highest → lowest impact)

  ┌─────┬────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────┬─────────────────────────────┐
  │  #  │              File              │                                     Change                                     │       Est. effect on        │
  │     │                                │                                                                                │  single-modality recovery   │
  ├─────┼────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────┤
  │ 1   │ persona_sampler.py             │ Fix narrative determinism. Pass rng/z into _build_narrative; build             │ text: 100% → ~70% (largest  │
  │     │                                │ decision_style_description from a z-graded templated lexicon.                  │ single drop)                │
  ├─────┼────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────┤
  │     │                                │ Remove label leaks. Drop decision_style_dominant from the leak-set (or derive  │                             │
  │ 2   │ psychographic_generator.py     │ it noisily/probabilistically); make price_consciousness continuous from        │ psych: 100% → ~80%          │
  │     │                                │ z.price_lean, not a 3-level map.                                               │                             │
  ├─────┼────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────┤
  │     │ schemas/persona.py +           │ Add shared latent z field to PersonaConfig; sample once; replace double-noise  │ enables cross-modal         │
  │ 3   │ persona_sampler.py             │ with θ = base + σ·(L·z) (logit-normal for bounded). Single source of           │ consistency; psych/txn →    │
  │     │                                │ per-participant deviation.                                                     │ ~75%                        │
  ├─────┼────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────┤
  │     │                                │ Wire z into traces (brand-inspection bias from z.brand_lean, depth jitter from │ trace: 95% → ~75%; removes  │
  │ 4   │ trace_simulator.py             │  z.thoroughness); replace the persona_id-keyed dwell lookup (line 83) with a   │ label leak                  │
  │     │                                │ continuous function of involvement/z.                                          │                             │
  ├─────┼────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────┤
  │ 5   │ config/personas.yaml           │ Tune base params for designed overlap (§3): move satisficer/compensatory       │ drives between-archetype BC │
  │     │                                │ maximiser closer; make adaptive a per-trial strategy mixture.                  │  into target band           │
  ├─────┼────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────┤
  │ 6   │ persona_sampler.py             │ Promote _NOISE_SCALE to a calibratable parameter (env/config); run the §5      │ tunes all modalities into   │
  │     │                                │ loop.                                                                          │ 65–80%                      │
  ├─────┼────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────┤
  │ 7   │ generator/validate.py          │ Relax validation tolerances — current checks (e.g. Payne ±0.2) assume          │ prevents false-positive     │
  │     │                                │ near-determinism and will now warn constantly; widen to distributional checks. │ warnings                    │
  └─────┴────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────┴─────────────────────────────┘

  ▎ Out of generator scope but required for targets 2 and 3 of the brief (recall@1, UMAP spread): swap fusion/train.py objective from cross-entropy to a contrastive/metric loss (same-participant-across-modalities = positive pairs). Flagging per §4; this is a fusion change, not a generator change.

  ---
  7. Risk assessment

  Per-change risks:
  - #1 (narrative): LLM paraphrase could re-introduce stereotyping if the template still names the strategy. Mitigate: grade by z, never emit the archetype label or strategy enum into the prompt. Risk of cost blow-up (1000 LLM calls) — already batched per SPEC.
  - #2 (psych leaks): removing decision_style_dominant outright may over-weaken psych below 65%. Mitigate: make it probabilistic (sample style from a z-conditioned distribution) rather than deleting.
  - #3/#4 (shared z): the immutability convention on PersonaConfig (CLAUDE.md: "no field additions without updating all downstream") means this touches every generator + schema-guardian. This is a real coordination cost, not a code risk.
  - #5 (YAML overlap): the dominant failure mode — over-overlapping. If you push BC too high on too many pairs, every modality drops below 65% and fusion can't recover 7 classes at all. Overlap must stay modality-specific (§3).

  Ways the problem becomes "trivially easy in a different way":
  - z becomes a new label leak. If z is low-dimensional and each generator reads it through a near-invertible linear map, a probe recovers z perfectly and cross-modal retrieval hits ~1.0 — trivially solved, but for the wrong reason (you've just relabelled each participant with a recoverable ID). Mitigate: pass z through modality-specific nonlinear, lossy, noisy projections so no single modality fully determines z; identity is recoverable only by combining views. This is the whole point.
  - Calibrating to exactly 70% by tuning one global knob can produce a dataset that's 70% by uniform confusion (every pair equally muddy) rather than structured confusion (designed pairs overlap, others separate). The first is uninformative; the second is the research target. Check the confusion matrix shape, not just the scalar accuracy.

  Honest structural limits of synthetic data:
  1. Real individual variation is not low-dimensional. A 5-D z is still a generative fiction; a model that recovers it isn't modelling a human, it's inverting your generator. Cross-modal retrieval working here proves the architecture can bind identity across views when identity exists in the data — it does not prove the architecture would find real consumers' latent structure. This is the same epistemic ceiling project-vision.md already states ("synthetic data validates architecture only").
  2. You cannot get recall@1 > 0.1 from generator changes alone. It is co-gated by the fusion objective (§4). Promising it as a generator deliverable would contradict `docs/post-mortems/phase2b-postmortem.md` §4.
  3. Trace remains the only modality with endogenous (non-z) individual variation — from acquisition stochasticity. The other three are z + noise by construction. So even after the fix, "individual signal" in psych/text/txn is injected, not emergent. With real data, traces would carry genuine idiosyncratic process variation the others can't manufacture — which is exactly why the upgrade path prioritises real MouseLab collection.

  Bottom line: generator changes can reliably deliver target #1 (65–80% single-modality recovery) and target #4 (within-archetype variation present and adjacent-archetype overlap visible in the data). Targets #2 (recall@1 > 0.1) and #3 (within-archetype UMAP spread in the embedding) require the generator fix and a contrastive fusion objective; the generator is necessary but not sufficient, and any plan that claims otherwise is overpromising.