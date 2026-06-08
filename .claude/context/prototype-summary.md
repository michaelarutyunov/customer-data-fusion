# Prototype Summary — Consumer Digital Twin

> Audience: non-technical stakeholder
> Written: 2026-06-07 — Updated: 2026-06-08 (bead 92v: individual-level variation)

## What we built

A software prototype that learns a compact "behavioural fingerprint" for each consumer by combining four types of data: how they browse a product choice screen (process traces), what they have purchased (transaction history), how they answered a brief psychological questionnaire (psychographics), and a short narrative describing their shopping style (text).

The system first trains a separate specialist model on each data type, then combines all four into a single 128-number summary — the Consumer Digital Twin (CDT) embedding — that represents the consumer's decision-making style.

## What we demonstrated

The prototype can identify which of seven decision-making archetypes a consumer belongs to from any single data source. Examples of archetypes: "Price Lexicographic" (always picks cheapest option), "Brand Heuristic" (loyal to a specific brand regardless of price), "Compensatory Thorough" (weighs all attributes before deciding).

After fixing structural label leaks in the data generator (bead 92v), single-modality archetype recovery sits at 62–79% — above chance (14%), below trivial. This range confirms that the archetype signal is present but not overwhelming, leaving room for genuine individual variation. The psychographic modality (79%) falls within the designed 65–80% target range.

We also demonstrated that the generator produces individual-level variation: two consumers in the same archetype now have meaningfully different feature profiles, driven by a shared 5-axis latent trait vector (price sensitivity, brand loyalty, thoroughness, impulsivity, openness to novelty).

## What we could not yet show

The fingerprint currently identifies *which archetype* a consumer belongs to, but not *which specific individual* within that archetype. The training objective (7-class classification) discards within-archetype variation. Replacing it with a contrastive metric-learning objective (next step: bead 0if) would teach the model to distinguish individuals within the same archetype — the step toward a genuine digital twin.

## What comes next

Two parallel tracks:

1. **Contrastive learning (bead 0if):** Replace cross-entropy classification with NT-Xent contrastive loss so the embedding preserves individual geometry within archetypes, not just archetype separation.

2. **Real data:** The synthetic generator approximates realistic individual variation, but real MouseLab sessions with human participants would provide ground-truth individual process traces. The architecture is designed to accept real data without modification.
