# Data Contracts

> Schemas defined in `schemas/`. All dataclasses are `frozen=True`.  
> Enums use `(str, Enum)` pattern for JSON serialisability.  
> Created: Phase 1 implementation complete (2026-06-05)
> Updated: schema-update epic (2026-06-13) — LatentDeviation 5→7 axes, EventType/PaymentMethod enums, temporal month field, clickstream + campaign modalities

---

## Serialisation Convention

All schemas serialise to JSONL via `dataclasses.asdict()` + `json.dumps()`.  
Enum values serialise as `.value` (string), not the enum name.  
`None` fields serialise as JSON `null`.  
No pickle. No custom `__json__` methods.

---

## PersonaConfig (`schemas/persona.py`)

The generative root. One instance per participant, shared across all generators.

| Field | Type | Description |
|---|---|---|
| `persona_id` | `str` | Archetype ID, e.g. `"price_lex"` |
| `label` | `str` | Human-readable label |
| `random_seed` | `Optional[int]` | RNG seed; `None` = non-deterministic |
| `strategy` | `StrategyParams` | Decision strategy parameters |
| `transactions` | `TransactionParams` | Purchase behaviour parameters |
| `psychographic` | `PsychographicParams` | Survey score parameters |
| `narrative` | `NarrativeParams` | Narrative generation parameters |

| `narrative` | `NarrativeParams` | Narrative generation parameters |
| `latent` | `Optional[LatentDeviation]` | 7-axis deviation vector `z` — cross-modal individual consistency |
| `month` | `int` | Temporal month index (0=baseline, 1–12 for AR(1) drift). Default 0. |

**Not written to JSONL** — used only as generator input. Never commit a `PersonaConfig` to disk.

### LatentDeviation (`z`) — 7 axes

The cross-modal individual consistency device. Sampled once per participant (`z ~ N(0, I)`), read by every generator. Split into a preference block and a process block, **deliberately weakly correlated** so process traces carry non-redundant information (the key independence property per `docs/modalities/mouselab.md`).

| Axis | Block | Drives |
|---|---|---|
| `price_lean` | preference | price_sensitivity, price-column inspection, price language |
| `brand_lean` | preference | brand_loyalty, brand-column inspection, trusted-brand language |
| `thoroughness` | process | inspection depth, involvement_score, "deliberate" language |
| `impulsivity` | process | p_strategy_lapse, impulse purchases, shorter dwell |
| `search_orientation` | process | Payne Index tendency as a latent trait (± = alternative/attribute-wise) |
| `attentional_bias` | process | divergence of dwell attention from choice weights (non-redundancy) |
| `openness` | hybrid | openness_to_new, risk_tolerance, brand-tier spread |

### Key nested types

**StrategyParams**: `primary_strategy` (Strategy enum — 7 values including `ELIMINATION_BY_ASPECTS`), `inspection_depth` (InspectionDepth enum), `first_attribute`, `p_reinspect`, `p_strategy_lapse`, `time_pressure_multiplier`, plus optional `attribute_weights` and `aspiration_levels`.

**TransactionParams**: `price_sensitivity`, `brand_loyalty`, `purchase_frequency_per_month`, `basket_size_mean`, `channel_mix` (dict summing to 1.0), `price_variance_tolerance`.

---

## AcquisitionEvent (`schemas/trace.py`)

One row per cell inspection event. Many rows per trial, many trials per participant.

| Field | Type | Description |
|---|---|---|
| `participant_id` | `str` | Matches `PersonaConfig.persona_id` |
| `trial_id` | `str` | `{session_uuid}_t{idx:03d}` |
| `event_index` | `int` | Position within trial sequence |
| `alternative_id` | `str` | Which alternative was inspected (e.g. `"A"`) |
| `attribute_id` | `str` | Which attribute was inspected (e.g. `"price"`) |
| `timestamp_s` | `float` | Cumulative time in trial (seconds) |
| `dwell_ms` | `float` | Time spent on this cell (log-normal, not uniform) |
| `is_reinspection` | `bool` | True if cell was already inspected in this trial |
| `event_type` | `EventType` | Interaction type: `CELL_HOVER`, `CELL_OPEN`, `COLUMN_ADD`, `SORT_APPLY`, `CHOICE`. Default `CELL_HOVER`. Realistic instrumented-hover framing per `docs/modalities/mouselab.md`. |

Output file: `data/synthetic/traces.jsonl` (fielded months 1–2 on coverage subset)

---

## TrialRecord (`schemas/trace.py`)

One row per completed trial. Aggregates across all AcquisitionEvents for that trial.

| Field | Type | Description |
|---|---|---|
| `participant_id` | `str` | Matches `PersonaConfig.persona_id` |
| `trial_id` | `str` | Same as AcquisitionEvent.trial_id |
| `session_id` | `str` | UUID4 — unique per `simulate_session()` call (not seeded) |
| `trial_index` | `int` | 0-based position within session |
| `category` | `str` | Product category label |
| `n_alternatives` | `int` | Board width (3, 5, or 7) |
| `n_attributes` | `int` | Board height (4, 6, or 8) |
| `time_pressure` | `bool` | Whether trial had time pressure (~30% of trials) |
| `final_choice` | `Optional[str]` | Chosen alternative ID |
| `confidence_rating` | `Optional[int]` | 1–5 Likert scale |
| `total_acquisitions` | `int` | Total cell inspections (including reinspections) |
| `prop_cells_inspected` | `float` | Unique cells / total cells |
| `payne_index` | `float` | -1.0 (attribute-wise) to +1.0 (alternative-wise) |
| `persona_id` | `str` | Archetype ID (redundant with participant_id for convenience) |

Output file: `data/synthetic/trials.jsonl`

**Note**: `session_id` is `uuid4()` — not deterministic across runs. `trial_id` inherits from `session_id` and also differs between runs. Downstream encoders should group by `participant_id`, not `session_id`.

---

## TransactionRecord (`schemas/transaction.py`)

One row per purchase event. Multiple rows per participant.

| Field | Type | Description |
|---|---|---|
| `participant_id` | `str` | Matches `PersonaConfig.persona_id` |
| `transaction_id` | `str` | `"tx_{participant_id}_{index:04d}"` |
| `days_before_session` | `int` | Days before the trace session (1–365) |
| `category` | `str` | Product category |
| `product_id` | `str` | Synthetic product ID |
| `sku` | `str` | Realistic order-line ID: `SKU-{category}-{tier}-{seq}` (default `""`, populated by Phase 2c) |
| `brand_tier` | `str` | `"premium"`, `"mid"`, `"value"`, `"own_label"` |
| `price_paid_normalised` | `float` | 0–1 normalised price (Beta distributed) |
| `unit_price` | `float` | Absolute price in currency units (default 0.0, populated by Phase 2c) |
| `quantity` | `int` | Units purchased |
| `channel` | `Channel` | `ONLINE` or `IN_STORE` |
| `purchase_type` | `PurchaseType` | `PLANNED`, `IMPULSE`, `ROUTINE`, `PROMOTIONAL` |
| `on_promotion` | `bool` | Whether item was purchased on promotion |
| `persona_id` | `str` | Archetype ID |
| `discount_applied` | `Optional[float]` | 0.0–0.3 discount fraction; `None` if no discount |
| `payment_method` | `PaymentMethod` | `CREDIT_CARD`, `DEBIT_CARD`, `PAYPAL`, `CASH`, `BNPL`. Default `CREDIT_CARD`. |
| `loyalty_card` | `Optional[bool]` | Retailer loyalty programme membership |

Output file: `data/synthetic/transactions.jsonl` (+ month-partitioned `transactions_month_{MM}.jsonl`)

---

## PsychographicVector (`schemas/psychographic.py`)

One row per participant. Fixed-width numeric vector.

| Field | Type | Range | Description |
|---|---|---|---|
| `participant_id` | `str` | — | Matches `PersonaConfig.persona_id` |
| `persona_id` | `str` | — | Archetype ID |
| `involvement_score` | `float` | 0–1 | Category involvement |
| `maximiser_score` | `float` | 0–1 | Maximising vs satisficing tendency |
| `risk_tolerance` | `float` | 0–1 | Willingness to choose unknown options |
| `price_consciousness` | `float` | 0–1 | Derived from `PriceConsciousness` enum |
| `brand_sensitivity` | `float` | 0–1 | Derived from `brand_loyalty` |
| `openness_to_new` | `float` | 0–1 | Willingness to try new products |
| `decision_style_dominant` | `str` | — | `"analytical"`, `"intuitive"`, `"heuristic"`, `"random"` |
| `age_band` | `str` | — | `"18-24"`, `"25-34"`, `"35-44"`, `"45-54"`, `"55+"` |
| `household_type` | `str` | — | From `NarrativeParams.household_type` |
| `employment_status` | `str` | — | `"full_time"`, `"part_time"`, `"self_employed"`, `"student"`, `"retired"` |
| `category` | `str` | — | Product category |
| `purchase_frequency_band` | `str` | — | `"weekly"`, `"monthly"`, `"quarterly"`, `"rarely"` |
| `years_buying_category` | `Optional[int]` | 0–30 | Years the participant has bought in this category; `None` for ~15% of records — impute with median (5) in feature engineering |

Output file: `data/synthetic/psychographics.jsonl`

---

## PersonaNarrative (`schemas/text.py`)

One row per participant. LLM-generated consumer narrative.

| Field | Type | Description |
|---|---|---|
| `participant_id` | `str` | Matches `PersonaConfig.persona_id` |
| `persona_id` | `str` | Archetype ID |
| `category` | `str` | Product category |
| `text` | `str` | Full narrative text (200–400 words) |
| `word_count` | `int` | Actual word count of `text` |
| `model_id` | `str` | LLM model used (`"deepseek-chat"` or `"claude-*"`) |
| `prompt_version` | `str` | Prompt template version (e.g. `"v1"`) |
| `embedding` | `Optional[list[float]]` | `None` at generation time; populated by `encoders/text/` |

Output file: `data/synthetic/narratives.jsonl`

**Important**: `embedding` is always `None` in the generator output. The text encoder populates it separately. Never call the sentence-transformer inside the generator.

---

## ClickstreamEvent (`schemas/clickstream.py`) — new modality

Web session event log. Many events per session, many sessions per customer per month. Anonymous sessions (10–15%) have `customer_id='anonymous'` and are excluded from encoder training.

| Field | Type | Description |
|---|---|---|
| `customer_id` | `str` | Matches `PersonaConfig.persona_id` (`'anonymous'` for unresolved sessions) |
| `session_id` | `str` | Unique session identifier |
| `event_ts` | `str` | ISO datetime string |
| `event_type` | `ClickstreamEventType` | `PAGE_VIEW`, `PRODUCT_VIEW`, `ADD_TO_CART`, `REMOVE_FROM_CART`, `SEARCH`, `FILTER_APPLY`, `CHECKOUT_START`, `PURCHASE` |
| `page_type` | `PageType` | `HOME`, `CATEGORY`, `PRODUCT`, `SEARCH_RESULTS`, `CART`, `CHECKOUT` |
| `sku_viewed` | `Optional[str]` | SKU viewed (if `PRODUCT_VIEW`) |
| `referrer` | `ReferrerType` | `DIRECT`, `ORGANIC`, `PAID_SEARCH`, `EMAIL`, `SOCIAL` |
| `device` | `DeviceType` | `DESKTOP`, `MOBILE`, `TABLET` |
| `dwell_ms` | `float` | Log-normal dwell (sigma=0.8) |
| `month` | `int` | Temporal month (1–12) |

Output files: `data/synthetic/clickstream/session_events.jsonl`, `sessions.jsonl` (+ month-partitioned)

---

## CampaignEvent (`schemas/campaign.py`) — new modality

Campaign interaction log. Email/push dispatch + response funnel.

| Field | Type | Description |
|---|---|---|
| `customer_id` | `str` | Matches `PersonaConfig.persona_id` |
| `campaign_id` | `str` | `CAMP-{type}-{seq:04d}` |
| `sent_ts` | `str` | ISO datetime string |
| `campaign_type` | `CampaignType` | `PROMOTION`, `NEWSLETTER`, `RE_ENGAGEMENT`, `LOYALTY`, `NEW_PRODUCT` |
| `discount_pct` | `float` | 0.0–0.5 |
| `category` | `str` | Product category |
| `opened` | `bool` | Email opened (20–40% base rate, z-conditioned) |
| `clicked` | `bool` | Link clicked (5–15% of opened) |
| `converted` | `bool` | Purchase attributed (1–5% of clicked) |
| `unsub` | `bool` | Unsubscribed — terminates future campaigns |
| `month` | `int` | Temporal month (1–12) |

Output files: `data/synthetic/campaigns.jsonl` (+ month-partitioned `campaigns_month_{MM}.jsonl`)

---

## Cross-Modal Key

All modalities share `participant_id` / `customer_id = PersonaConfig.persona_id`. The pipeline cycles archetypes, so the same `persona_id` string will appear for multiple participants when `n > 7`. For uniqueness tracking across a large dataset, use `(persona_id, participant_index)` or add a participant counter field in a future schema version.

All event-stream modalities (transactions, clickstream, campaigns) include a `month` field (1–12) for temporal partitioning. Snapshot modalities (traces, psychographics) are fielded at specific months (traces: 1–2 on coverage subset; psychographics: 1 and 7).
