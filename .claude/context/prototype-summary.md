# Prototype Summary — Consumer Digital Twin

> Audience: non-technical stakeholder
> Written: 2026-06-07 — Updated: 2026-06-09 (bead 0if: individual-level identity via contrastive learning)

## What we built

A software prototype that learns a compact "behavioural fingerprint" for each consumer by combining four types of data: how they browse a product choice screen (process traces), what they have purchased (transaction history), how they answered a brief psychological questionnaire (psychographics), and a short narrative describing their shopping style (text).

The system first trains a separate specialist model on each data type, then combines all four into a single 128-number summary — the Consumer Digital Twin (CDT) embedding — that represents the consumer's decision-making style.

## What we demonstrated

**Archetype recovery:** The prototype can identify which of seven decision-making archetypes a consumer belongs to from any single data source. Examples: "Price Lexicographic" (always picks cheapest option), "Brand Heuristic" (loyal to a specific brand), "Compensatory Thorough" (weighs all attributes before deciding). Single-modality accuracy sits at 62–79% — above chance (14%), leaving genuine room for individual variation.

**Individual identity (new):** After replacing the pure classification objective with a contrastive metric-learning objective (bead 0if), the system can now identify *which specific individual* a consumer is within their archetype. Given two independent partial views of the same consumer — where each view may be missing one or more data modalities due to random dropout — the system finds the correct consumer among 210 candidates **70% of the time**. Random chance is 0.5%.

This is the core CDT claim: the embedding captures not just "what type of shopper this person is" but "which specific person this is" — robust to incomplete data.

**Individual-level data generation:** The data generator produces consumers with genuinely different trait profiles within each archetype, driven by a 5-axis latent vector (price sensitivity, brand loyalty, inspection depth, impulsivity, risk tolerance). This was validated in bead 92v.

## How individual identity works

Each consumer's four data sources (trace, transaction, text, psychographic) are separately compressed into 128-number embeddings by specialist models. The fusion model combines these four into a single CDT embedding.

During training, the fusion model is shown two randomly degraded versions of the same consumer — one might be missing transaction data, another might be missing the narrative — and trained to make both map to the same location in the embedding space. Other consumers are pushed apart. This "contrastive" training (NT-Xent) teaches the model that individual identity is stable even when data is incomplete.

The classification head (archetype identification) is kept as a secondary objective to ensure the embedding still preserves archetype-level structure.

## What comes next

**Real data:** The synthetic generator approximates realistic individual variation, but real MouseLab sessions with human participants would provide ground-truth individual process traces. The architecture is designed to accept real data without modification.

**Cross-session stability:** The current dataset has one session per person. With multiple sessions, we could test whether the CDT embedding is stable across time — does the fingerprint still identify the same person six months later? This is the highest-value validation for real-world CDT applications.
