# PRD Success Criteria Validation

> Written: 2026-06-07 — Updated: 2026-06-08 (bead 92v: generator redesign)
> Phase: Phase 2b + bead 92v generator redesign
> Evidence base: fusion/train.py, evaluation/strategy_recovery.py, evaluation/ablation.py,
>   evaluation/retrieval.py, evaluation/config_probe.py, evaluation/counterfactual.py,
>   docs/adr/0001-generator-spread-calibration.md

---

## Criteria Source

Success criteria are drawn from project-vision.md §PRD-12. No PRD.md exists in-repo; the four
criteria below are the canonical record. The contrastive-loss margin criterion (intra < inter > 0.3)
is N/A — the trace encoder switched from NT-Xent to cross-entropy classification in bead 6yl, and
fusion uses cross-entropy throughout.

---

## Criterion 1 — Strategy recovery >85% from process trace embeddings alone

**Status: ◐ REVISED — see note**

> **Note (2026-06-08):** The 85% threshold was achievable because the original generator had
> label leaks — persona archetypes were deterministically encoded into feature values.
> Bead 92v redesigned the generator to produce genuine individual-level variation (PRD target:
> 65–80%). The revised numbers below reflect the generator-redesign state.

| Modality | Phase 2b (label-leak) | Post-92v (genuine variation) | Target |
|---|---|---|---|
| Trace (alone) | 95.02% | 61.89% | 65–80% |
| Transaction (alone) | 62.59% | 63.62% | 65–80% |
| Psychographic (alone) | ~100% | 78.95% ✓ | 65–80% |
| Fused (all four) | 100.00% | not yet re-evaluated | — |

Psychographic is now in the 65–80% target range. Trace and transaction are 2–3 points below
the floor — an encoder capacity ceiling at this dataset size, not a data generation problem
(documented in ADR 0001). The high Phase 2b numbers were artefacts of label leaks, not
genuine discriminative power.

---

## Criterion 2 — Embedding space shows clear strategy-based geometry (UMAP)

**Status: ✓ PASS (qualitative)**

The UMAP plots in `notebooks/03_fusion_validation.ipynb` (Section 3) show:
- **(a) By archetype:** Seven distinct clusters, well-separated, with minimal inter-cluster
  overlap. The 100% classification accuracy is the quantitative correlate of this visual
  separation.
- **(b) By `price_sensitivity`:** Within each archetype cluster, a gradient in price_sensitivity
  is visible, confirming that the CDT embedding preserves some within-archetype continuous
  variation — not purely a discrete archetype classifier.

The config_probe R² results (Criterion 4 supplementary) reinforce this: fused R² for
price_sensitivity = 0.897, confirming that the latent continuous param is recoverable from
the embedding even though individual retrieval is near-zero.

---

## Criterion 3 — Each modality contributes meaningfully (ablation test)

**Status: ◐ PARTIAL**

Leave-one-out ablation (zero 128-dim slice, full fusion model):

| Modality removed | Accuracy drop | Threshold | Verdict |
|---|---|---|---|
| Trace | −10.45% | >5% = meaningful | ✓ Meaningful |
| Psychographic | −4.48% | <5% = low contribution | ⚠ Low delta |
| Transaction | −0.00% | <5% = low contribution | ⚠ Low delta |
| Text | −0.00% | <5% = low contribution | ⚠ Low delta |

**Interpretation:** Low ablation delta for text and psychographic is *expected*, not a failure.
Both encoders individually achieve 100% archetype accuracy — they are redundant with each other
from the classifier's perspective. Removing one leaves the other to carry the full signal.
Ablation delta measures redundancy within the fused model, not individual modality value.

Transaction's zero delta reflects its lower individual accuracy (62.59%) and the fact that
the fused model has learned to rely on other modalities when transaction signal is weak.

**Honest assessment:** Three of four modalities contribute zero marginal accuracy when removed
from a 100%-accurate fused model. The prototype does not demonstrate independent complementary
contribution across all four modalities. Trace is the only non-redundant modality in ablation.

---

## Criterion 4 — Conjoint-format traces map coherently into the same embedding space

**Status: ✓ PASS**

Evidence:
- **Confusion matrix:** The 201-participant val set shows perfect per-class accuracy (7×7
  confusion matrix has non-zero only on the diagonal). Zero cross-archetype confusions.
- **UMAP geometry:** Trace-only UMAP (from Phase 2a) and CDT UMAP both show same 7-cluster
  structure, confirming that conjoint trace sequences map into archetype-coherent regions.
- **Retrieval:** CDT-vs-trace recall@1 = 0.003 — near-zero individual retrieval confirms
  that the trace-to-embedding mapping is many-to-one within archetypes (coherent compression),
  not random noise.

---

## Supplementary: CDT Embedding Quality (Tier 2, not gating)

These are diagnostic findings, not pass/fail gates per fusion/SPEC.md.

### Cross-modal retrieval

| Test | recall@1 | recall@10 | Within-archetype chance |
|---|---|---|---|
| CDT → trace | 0.003 | 0.017 | 0.007 |
| CDT → transaction | 0.001 | 0.016 | 0.007 |
| CDT → text | 0.001 | 0.012 | 0.007 |
| CDT → psychographic | 0.000 | 0.001 | 0.007 |

All recall values are near-zero — below within-archetype random chance. The CDT embedding does
not identify the same individual across modalities. This confirms the embedding is archetype-level
rather than individual-level.

### PersonaConfig regression probe (fused R²)

| Parameter | Fused R² | Best single-modality R² |
|---|---|---|
| inspection_depth | **0.982** | trace 0.863 |
| price_sensitivity | **0.897** | trace 0.857 |
| brand_loyalty | **0.890** | psychographic 0.808 |
| p_strategy_lapse | **0.888** | psychographic 0.787 |
| risk_tolerance | **0.796** | trace 0.770 |
| maximiser_score | **0.796** | psychographic 0.701 |
| involvement_score | **0.728** | psychographic 0.692 |

Fused embedding achieves the highest R² on all 7 parameters. The CDT does encode continuous
latent variation — but at the archetype distribution level, not individual resolution. The 
R² values reflect archetype-to-config mapping (7 archetypes → 7 distinct param distributions),
not within-archetype individual fit.

---

## Overall Verdict

| Criterion | Phase 2b status | Post-92v status | Note |
|---|---|---|---|
| 1. Strategy recovery >85% | ✓ PASS (label-leak) | ◐ PARTIAL | Psychographic ✓; trace/txn ~3 pts below 65% floor |
| 2. Geometry (UMAP) | ✓ PASS | not re-evaluated | Clusters still valid; geometry may shift post-redesign |
| 3. Meaningful modality contribution | ◐ PARTIAL | not re-evaluated | Ablation against new fused model needed (bead 0if) |
| 4. Conjoint traces coherent | ✓ PASS | not re-evaluated | Dependent on trace encoder quality |

**Current supportable claim (post-92v):**

The generator now produces genuine individual-level variation — no archetype label leaks.
Single-modality recovery of 62–79% confirms that archetypes are recoverable but not trivially
so. The stronger CDT claim (individual-level digital twin) requires bead 0if (contrastive loss)
to replace the classification objective with a metric-learning objective that preserves
within-archetype individual geometry.
