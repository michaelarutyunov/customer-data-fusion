# Phase 2b Post-Mortem: Fusion and Evaluation

> Written: 2026-06-07
> Phase: Phase 2b — Fusion and Evaluation (epic customer-data-fusion-67a)
> Scope: fusion/, evaluation/ (ablation, geometry, retrieval, config_probe, counterfactual,
>   strategy_recovery), notebooks/03, notebooks/04, .claude/context/prd-validation.md

---

## 1. What Was Built

### Fusion Module

| File | Purpose |
|---|---|
| `fusion/SPEC.md` | Full fusion specification with resolved design decisions |
| `fusion/meta_learner.py` | LateFusionMetaLearner: MLP 512→256→128→7 with embed() method |
| `fusion/train.py` | Training script: embedding cache, modality dropout (p=0.2), early stopping |
| `models/fusion_meta_learner.pt` | Trained checkpoint (100% val_acc by epoch 3) |
| `models/fusion_embeddings_cache.pt` | Per-participant embeddings for all 4 modalities |

### Evaluation Infrastructure (Tier 2)

| File | Purpose |
|---|---|
| `evaluation/strategy_recovery.py` | Tier 1: archetype recovery accuracy + comparison table |
| `evaluation/ablation.py` | Leave-one-out modality importance |
| `evaluation/geometry.py` | UMAP of CDT embeddings (2D projection) |
| `evaluation/retrieval.py` | Cross-modal nearest-neighbour recall@1/10 |
| `evaluation/config_probe.py` | Ridge regression probe for 7 PersonaConfig float params |
| `evaluation/counterfactual.py` | Archetype-level redistribution (Option A) |

### Supporting Data

| File | Purpose |
|---|---|
| `data/synthetic/participant_configs.jsonl` | 1001 PersonaConfig records for config_probe |
| `notebooks/03_fusion_validation.ipynb` | Full Tier 1+2 evaluation notebook |
| `notebooks/04_counterfactual_tests.ipynb` | Counterfactual scenario notebook |
| `.claude/context/prd-validation.md` | Formal PRD success criteria validation |

### Architecture Decisions (Resolved from SPEC open questions)

| Decision | Resolution |
|---|---|
| L2-normalise before concat? | Yes — `F.normalize(e, dim=-1)` per modality before cat to [B,512] |
| Partial-modality inference? | Yes — zero the absent 128-dim slice |
| Training objective? | Cross-entropy (7-class archetype classification) |
| Ablation mechanism? | Zero the 128-dim modality slice (not re-run without encoder) |
| Evaluation Tier 1 gate? | >85% overall strategy recovery accuracy |

---

## 2. Results Summary

### Tier 1 (hard gate)

| Metric | Value | Gate | Result |
|---|---|---|---|
| Strategy recovery (fused) | 100.00% | >85% | ✓ PASS |
| Per-class accuracy | 100% all 7 classes | — | Perfect |

### Tier 2 (diagnostic)

| Evaluation | Key finding |
|---|---|
| Ablation | Trace: −10.45%; Text/Psych: −0.00%; Transaction: −0.00% |
| Cross-modal retrieval | recall@1 ≈ 0.001–0.003; below within-archetype chance (0.007) |
| Config probe (fused R²) | 0.728–0.982 across all 7 params; fused best on every param |
| Counterfactual (Option A) | All 3 sanity checks pass; price_lex most price-sensitive, brand_affect most brand-sensitive |

---

## 3. Errors and Resolutions

### E1 — TransactionEncoder constructor arg mismatch

**Symptom:** `RuntimeError: Error(s) in loading state_dict for TransactionEncoder`

**Root cause:** Checkpoint was saved with `projection_dim=16, gru_hidden=32` (small variant
used in Phase 2a training), but `train.py` initialised with default args (`projection_dim=64,
gru_hidden=128`). State dict size mismatch on load.

**Fix:** `TransactionEncoder(projection_dim=16, gru_hidden=32)` in `load_encoders()`.

### E2 — generate_embeddings() used made-up method names

**Symptom:** `AttributeError: 'TransactionEncoder' object has no attribute 'embed_transactions'`

**Root cause:** Initial implementation of `generate_embeddings()` was written with invented
method names (`embed_trace`, `embed_transactions`, `embed_psychographic`) that don't exist on
any encoder. Discovered on first training run.

**Fix:** Full rewrite using actual encoder interfaces traced from existing probe scripts
(evaluation/trace_probe.py, transaction_probe.py, etc.):
- Trace: load events + trials → build_vocab → tokenise_trial → mean-pool across trials
- Transaction: load records → sort → vocab.encode_sequence → encoder(token_seq, lengths)
- Text: encode_texts([text]) → encoder(sent_emb)
- Psychographic: PsychographicVector → to_feature_vector → encoder(vec.unsqueeze(0))

**Lesson:** Always trace encoder interfaces from existing working code (probes), never invent them.

### E3 — pyright errors (18) blocking clean run

**Symptom:** 18 pyright errors in fusion/train.py including `torch.zeros` not exported,
`nn.Module` has no `.vocab` attribute, `ReduceLROnPlateau(verbose=True)` invalid.

**Fix pattern:**
- `torch.zeros/long/tensor/cat`: `# type: ignore[reportPrivateImportUsage]` (project convention)
- `.vocab` access: assert/isinstance cast to `TransactionEncoder`
- `verbose=True`: removed (deprecated in PyTorch 2.x)
- Return types: relaxed to `dict` for heterogeneous dicts containing both Tensor and list[str]

### E4 — participant_configs.jsonl only 7 lines

**Symptom:** config_probe needed 1001 participant configs but file had only 7 (one per archetype).

**Root cause:** Pipeline change (bead c33) was implemented but the full dataset was not re-run.
Only the 7-participant test run had generated participant_configs.jsonl.

**Fix:** Reconstructed 1001 records by reading psychographics.jsonl for participant ordering
and re-sampling PersonaConfig with deterministic seeds (seed=i, round-robin archetype ordering).

**Lesson:** When the pipeline schema changes, regenerate the full dataset before running
downstream evaluations.

### E5 — MLflow file store deprecation

**Symptom:** `MlflowException: The filesystem tracking backend is in maintenance mode`

**Fix:** `MLFLOW_ALLOW_FILE_STORE=true` env var. Non-blocking — training completed despite
the error appearing before encoder load.

---

## 4. Prototype Retrospective

### Which hypotheses were supported?

The project-vision.md describes this as a latent variable recovery problem. Four implicit
hypotheses underlie the prototype:

**H1 — Multimodal late fusion of 4 modalities can recover behavioural archetypes with >85%
accuracy.**

*Result: Confirmed.* Fused accuracy = 100%. Even single modalities clear the bar (trace: 95.02%,
text: 100%, psychographic: 100%). The hypothesis was too easy given the data generating process
— all modalities are functions of the same 7-class PersonaConfig, so the problem is structurally
a 7-way classification with four different feature views.

**H2 — Each modality provides independent, non-redundant behavioural signal.**

*Result: Partially supported.* Trace is non-redundant (ablation delta: −10.45%). Text and
psychographic are mutually redundant in the fused model (both at 100% alone, both have 0%
ablation delta). Transaction is the weakest modality (62.59% alone, 0% ablation delta).
The architecture integrates all four, but three are substitutable for each other at the
archetype-classification level.

**H3 — The fused CDT embedding captures participant-level behavioural variation (individual
digital twin).**

*Result: Not supported at individual level.* Cross-modal retrieval recall@1 ≈ 0.001–0.003,
below within-archetype random chance (0.007). The CDT cannot identify the same participant
across modalities. The config_probe R² (0.73–0.98) suggests the embedding encodes
*archetype-level* continuous variation (different archetypes have different param distributions),
not individual-level variation within an archetype. The 7-class classification objective
structurally discards within-archetype variation.

**H4 — Process trace (MouseLab) data is the most discriminative single modality.**

*Result: Partially supported.* Trace achieves 95.02% alone, the highest accuracy of any
non-saturating modality. However, text and psychographic are at 100%, making trace the
"most useful" only because it avoids saturation. The discriminative ranking is:
text/psychographic (trivially saturated) > trace (genuinely informative) > transaction (weakest).

### What was the most wrong assumption?

**The assumption that four modalities would provide genuinely independent signal.**

All four modalities are generated from the same 7-class PersonaConfig object. Two of them
(text narratives, psychographic surveys) are near-direct readouts of the archetype label —
there is no noise or mixing that would require fusion to combine complementary information.
The generator was too faithful: each modality too cleanly encodes the latent class.

In a real-data setting, modalities would be noisy, measured at different times, and only
partially correlated with behaviour. The fusion problem would be genuinely hard. Here, it
was easy: any single modality (except transaction) was already sufficient.

### What would the next version look like with real data?

1. **Real process traces:** MouseLab sessions with human participants produce genuine individual
   variation. Two participants in the same "price-sensitive" archetype would still differ in
   strategy switching, reinspection rate, and time allocation. The trace encoder would be the
   primary source of individual-level signal.

2. **Real transactions:** Actual purchase histories have noise from channel availability,
   promotion exposure, and external shocks. Transaction patterns would be weakly correlated
   with archetype, not almost-deterministic functions of it. The transaction encoder would
   need to learn what is stable vs. context-dependent.

3. **Real narratives:** Interview transcripts or diary entries are not stereotyped persona
   descriptions. The sentence-transformer would need fine-tuning to extract decision-style
   signals from natural language, rather than clustering pre-written archetype descriptions.

4. **Replace classification objective with contrastive learning:** Once individual variation
   matters, the 7-class CE objective should be replaced with a metric learning objective
   (NT-Xent or triplet loss with same-session pairs as positives). This would preserve
   individual geometry rather than collapsing it.

5. **Longitudinal data:** A digital twin's value is in predicting change over time. The
   current architecture has no temporal structure above the modality level. A real CDT would
   model drift in PersonaConfig params as a consumer ages, changes life circumstances, or
   is exposed to market interventions.

---

## 5. Open Items

| Item | Bead | Status |
|---|---|---|
| Option B counterfactual (generator re-run) | sei | Deferred, P3 |
| Trace encoder at participant level (mean-pool before classification) | crz | Open, P2 |
| Final context update | 67a.13 | Next |

---

## 6. Addendum — Generator Redesign (bead 92v, 2026-06-08)

The Phase 2b results above were produced with a generator that had structural label leaks —
persona archetypes were deterministically encoded into feature values via dicts keyed by
`persona_id`. This made single-modality strategy recovery trivially high (95–100%) and
invalidated H3 ("individual-level digital twin") since all participants in an archetype
were statistically identical within each modality.

### What was fixed

| Leak | Fix |
|---|---|
| `_STRATEGY_TO_DECISION_STYLE` deterministic dict | z-conditioned softmax sampling |
| `_DWELL_MU` keyed by persona_id | continuous: `5.8 + 1.7*involvement + 0.4*z.thoroughness` |
| `_ARCHETYPE_DEPTH_FRACTION` keyed by persona_id | continuous: `base_frac + 0.06*z.thoroughness` |
| `household_type` one-hot in psychographic features | removed from feature vector (22→19 dims) |
| `price_consciousness` 3-level enum hardcoded | logit-normal projection via `project(z.price_lean, base, sigma)` |

All modalities now draw from a shared 5-axis latent `LatentDeviation` z vector per participant,
ensuring cross-modal individual consistency without archetype leakage.

### Calibration outcome

Two env vars control spread: `GENERATOR_SPREAD` (trace/transaction) and `PSYCHOGRAPHIC_SPREAD`
(psychographic only). Final calibrated values: `GENERATOR_SPREAD=0.2`, `PSYCHOGRAPHIC_SPREAD=4.0`
with 150 participants per archetype. See `docs/adr/0001-generator-spread-calibration.md`.

| Modality | Phase 2b | Post-redesign | PRD target |
|---|---|---|---|
| Trace | 95.02% | 61.89% | 65–80% |
| Transaction | 62.59% | 63.62% | 65–80% |
| Psychographic | ~100% | **78.95% ✓** | 65–80% |

Trace and transaction are ~2–3 points below the 65% floor. Encoder val_loss plateaus at ~1.04
regardless of data volume (100→150/arch), indicating an encoder capacity ceiling rather than
a data generation problem.

### Implication for previous findings

The Phase 2b Tier 1 result (100% fused accuracy) was an artefact of label leaks — the fused
model was doing near-trivial lookup. The fusion model and Tier 2 results need re-evaluation
against the redesigned generator once bead 0if (contrastive loss) is complete.
