# Modality Expansion Plan

> **Status:** Planned. Epic `customer-data-fusion-1y0` (schema-update), 14 beads across 5 phases.
> **Source recommendations:** `docs/modalities.md` (modality set + temporal structure), `docs/modalities/mouselab.md` (process trace specifics).

---

## Why

The prototype proved the CDT architecture works (100% archetype recovery, 70.4% dropout-view recall@1, R² ≥ 0.79 on PersonaConfig regression). But two gaps limit its persuasive weight with a technical audience:

1. **No temporal dynamics.** The latent vector `z` is sampled once and never evolves. Drift detection, early-warning signals, and temporal evaluation are impossible. The recommendation is explicit: *"use a temporal split, not random; your audience will check."*

2. **Process traces are informationally redundant.** Dwell attention and choice preferences use the same weights — fusion gains from the trace modality are decorative, not structural. The recommendation's central challenge: *"at least one demo scenario unsolvable without it. Otherwise it's decorative, and a technical audience will notice that ablating it costs nothing."*

This expansion addresses both by restructuring `z` to separate decision-style traits from preference traits, adding temporal evolution, and expanding the modality set from 4 to 6.

---

## What Changes

### Latent model restructuring

The core change is in `LatentDeviation` (schemas/persona.py). Two new fields create the independence property:

- **`search_orientation: float`** — makes Payne Index tendency a sampled latent trait, not just an emergent output of strategy simulation. Individuals within the same archetype now vary systematically in search orientation.
- **`attentional_bias: float`** — controls divergence between dwell attention allocation and choice preference weights. When `attentional_bias ≠ 0`, the trace modality carries information that preferences alone cannot recover. This is the critical non-redundancy property.

The preference block (price_lean, brand_lean) and the process block (thoroughness, impulsivity, search_orientation, attentional_bias) are now deliberately weakly correlated. Openness remains hybrid.

### Temporal dynamics

AR(1) evolution over 12 simulated months:

```
z[t] = α · z[t-1] + (1-α) · μ + ε,  ε ~ N(0, σ² · I_dyn)
```

Applied only to loyalty/churn dimensions (price_lean, brand_lean) and attentional_bias. Regime shifts injected for ~12% of cohort (sudden loyalty decay or attention shift at a random month 6–10). This unlocks:
- Ground-truth drift labels for evaluation
- Early-warning scenarios (attention shifts preceding behavioural shifts)
- Temporal train/test split (months 1–8 train, 9–12 eval)

### New modalities

| Modality | Schema | Generator | Encoder | Source |
|---|---|---|---|---|
| **Clickstream** (web session events) | `schemas/clickstream.py` | Markov session model conditioned on z | GRU (2-layer, hidden=128) | `docs/modalities.md` rec #2 |
| **Campaign** (email/push interaction log) | `schemas/campaign.py` | Dispatch + sigmoid response model | Self-attention (2 heads, 2 layers) | `docs/modalities.md` rec #5 |

Clickstream provides the dense temporal signal that transactions lack. Campaigns provide the intervention modality that closes the counterfactual loop.

### Existing modality changes

- **Traces:** Per-individual strategy mixture (softmax over logits from z), attentional weight divergence via `attentional_bias`, `EventType` enum for realistic hover/exposure framing, EBA strategy added.
- **Transactions:** Enriched to order-line schema (sku, qty, unit_price, discount_applied, payment_method). Hazard-model inter-purchase timing replaces fixed-count generation. Product catalog: 3 categories × 3 tiers = 27 SKUs.
- **Psychographics:** Unchanged (input distribution may shift if Likert noise is added later, but dimensionality stays at 19).
- **Text:** Unchanged (architecture handles any text length; narrative format may shift to shorter verbatims in future work).

---

## Modality count: 4 → 6

| Modality | Encoder output | Signal type |
|---|---|---|
| Process traces | Transformer → 128-dim | Decision process (how) |
| Transactions | GRU → 128-dim | Preferences (what, temporal) |
| Text narratives | Frozen sentence-transformer → 128-dim | Qualitative (who) |
| Psychographics | MLP → 128-dim | Traits (stable) |
| **Clickstream** (new) | GRU → 128-dim | Browsing process (how, dense) |
| **Campaigns** (new) | Self-attention → 128-dim | Intervention response (what-if) |

Fusion: concat 6 × 128 = 768 → MLP → 128-dim CDT embedding. Learned MISSING embedding replaces zero-fill for absent modalities (partial coverage: traces on 250/1000).

---

## Implementation Phases

```
Phase 1 (P0): Revise schemas
  └── jms — LatentDeviation +2 fields, EventType enum, PersonaConfig month, TransactionRecord enrichment
      ↓
Phase 2 (P1): Update generators (parallel)
  ├── 0w8 — persona_sampler: temporal z + regime shifts
  ├── b75 — trace_simulator: strategy mixture + attentional divergence + event_type
  ├── lpb — transaction_simulator: order-line + hazard timing
  ├── job — clickstream generator + schema (new)
  └── ghs — campaign generator + schema (new)
      ↓
  est — pipeline: 12-month temporal structure + partial coverage (bottleneck — depends on all above)
      ↓
Phase 3 (P2): Encoders
  ├── kut — minor tweaks: trace (event_type embedding) + transaction (input dim)
  ├── 53o — clickstream encoder (new, GRU)
  ├── sf2 — campaign encoder (new, attention)
  └── iz3 — engineered-metrics baseline (ablation: logreg vs Transformer on traces)
      ↓
Phase 4 (P3): Fusion
  ├── rq2 — concat dimensionality → 768, configurable n_modalities, learned MISSING embedding
  └── hcx — retrain + evaluate (targets: >95% archetype, ≥70.4% recall@1, R² ≥ 0.79)
      ↓
Evaluation (P2)
  └── lp4 — temporal train/test split (months 1–8 / 9–12)
```

**Critical path:** `jms` → `0w8` → `b75` → `est` → `kut` → `rq2` → `hcx` (7 beads deep).

---

## Key Design Decisions (pinned during bead audit)

| Decision | Choice | Rationale |
|---|---|---|
| Temporal PersonaConfig | `month:int` field directly on PersonaConfig | Simpler than wrapper class; sampler creates one per month |
| TransactionRecord enrichment | Committed (not optional) | Phase 2c depends on these fields |
| Strategy mixture formula | Softmax over 4 logits from z, temperature=1.0 | Produces archetypal mixtures at archetype-mean z; individual variation within |
| Attentional divergence | `attentional_bias * 0.3` rotation on dwell shares | Controllable gap between attention and preference; zero = no divergence |
| AR(1) drift scope | price_lean, brand_lean, attentional_bias only | Thoroughness/impulsivity/openness/search_orientation are stable traits |
| Inter-purchase seasonality | Months 11–12 at 1.8×, rest at 1.0× | Holiday peak; simple step function |
| Product catalog | 3 categories × 3 tiers = 27 SKUs | Enough variety for brand-tier signal, small enough to generate quickly |
| Clickstream Markov graph | 3 intent-specific transition matrices (BROWSE/RESEARCH/BUY) | Legal transitions by intent; z scales exit and repeat probabilities |
| Campaign base rates | Pinned per campaign type (5 types, 3 stages each) | Ensures funnel distribution is realistic and reproducible |
| Temporal split semantics | Per-event (same customer in both sets, different time windows) | Correct temporal evaluation; no future leakage |

---

## What This Does NOT Cover

These are future work, not part of the schema-update epic:

- **Demo scenarios** (personalised ranking, context effects, drift detection) — require the infrastructure this epic builds
- **Text modality rework** (verbatim-style narratives instead of polished prose) — independent of this chain
- **Likert noise** (straight-lining, acquiescence bias, item non-response) — affects psychographic generator only
- **Early fusion upgrade path** — the existing placeholder in `fusion/` remains; this expansion keeps late fusion as default
