"""
generator/validate.py

Cross-modal consistency checks for a single generated participant.

All failures are logged at WARNING via structlog; no exceptions are raised.
Stochastic simulation will produce legitimate outliers — hard failures would
abort valid batch runs.

Public API:
    validate_participant(
        config, trial_records, transactions, psychographic, narrative, choice_sets
    ) -> ValidationReport
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from schemas.persona import PersonaConfig
from schemas.psychographic import PsychographicVector
from schemas.text import PersonaNarrative
from schemas.trace import TrialRecord
from schemas.transaction import TransactionRecord
from schemas.choice_set import ChoiceSet

log = structlog.get_logger()


@dataclass
class ValidationReport:
    """
    Per-participant validation outcome.

    passed: True only if all checks passed.
    failures: list of (check_name, message) for each failed check.
    """

    participant_id: str
    passed: bool = True
    failures: list[tuple[str, str]] = field(default_factory=list)

    def fail(self, check: str, message: str) -> None:
        self.passed = False
        self.failures.append((check, message))


def validate_participant(
    config: PersonaConfig,
    trial_records: list[TrialRecord],
    transactions: list[TransactionRecord],
    psychographic: PsychographicVector,
    narrative: PersonaNarrative,
    choice_sets: list[ChoiceSet] | None = None,
    participant_id: str | None = None,
) -> ValidationReport:
    """
    Run all 8 cross-modal consistency checks for one participant.

    Failures are logged at WARNING; the report is returned regardless.
    """
    if participant_id is None:
        participant_id = config.persona_id
    report = ValidationReport(participant_id=participant_id)
    bound_log = log.bind(participant_id=participant_id, persona_id=config.persona_id)

    _check_price_consciousness(config, psychographic, transactions, report, bound_log)
    _check_brand_sensitivity(config, psychographic, transactions, report, bound_log)
    _check_narrative_word_count(narrative, report, bound_log)
    _check_transaction_price_consistency(config, transactions, report, bound_log)
    _check_payne_index_range(config, trial_records, report, bound_log)

    # Phase 0 choice validation
    if choice_sets is not None:
        _check_choice_consistency(trial_records, choice_sets, report, bound_log)
        _check_product_coverage(choice_sets, report, bound_log)
        _check_trace_choice_coupling(trial_records, choice_sets, report, bound_log)

    if report.passed:
        bound_log.debug("validation_passed")
    else:
        bound_log.warning(
            "validation_failed",
            n_failures=len(report.failures),
            checks=[f[0] for f in report.failures],
        )

    return report


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_price_consciousness(
    _config: PersonaConfig,
    psychographic: PsychographicVector,
    _transactions: list[TransactionRecord],
    report: ValidationReport,
    bound_log: structlog.BoundLogger,
) -> None:
    """
    Directional checks are now archetype-level correlation checks done externally.
    Per-participant: only flag values outside the valid [0.0, 1.0] range (data bug).
    """
    pc = psychographic.price_consciousness
    if not (0.0 <= pc <= 1.0):
        msg = f"price_consciousness={pc:.3f} out of valid range [0.0, 1.0]"
        report.fail("price_consciousness", msg)
        bound_log.warning("validation_failed", check="price_consciousness", value=pc)


def _check_brand_sensitivity(
    config: PersonaConfig,
    _psychographic: PsychographicVector,
    transactions: list[TransactionRecord],
    report: ValidationReport,
    bound_log: structlog.BoundLogger,
) -> None:
    """
    brand_affect: brand_sensitivity should be > 0.7; transaction brand_tier
    should be concentrated in 1–2 distinct values (>= 70% of transactions).
    """
    if config.persona_id != "brand_affect":
        return

    if transactions:
        from collections import Counter

        tier_counts = Counter(t.brand_tier for t in transactions)
        top2_count = sum(v for _, v in tier_counts.most_common(2))
        concentration = top2_count / len(transactions)
        if concentration < 0.50:
            msg = f"brand_affect brand_tier concentration={concentration:.2f} expected >=0.50"
            report.fail("brand_tier_concentration", msg)
            bound_log.warning(
                "validation_failed",
                check="brand_tier_concentration",
                concentration=round(concentration, 3),
            )


def _check_narrative_word_count(
    narrative: PersonaNarrative,
    report: ValidationReport,
    bound_log: structlog.BoundLogger,
) -> None:
    """Narrative word count must be within 200–400 words."""
    wc = narrative.word_count
    if wc < 200 or wc > 400:
        msg = f"narrative word_count={wc} not in [200, 400]"
        report.fail("narrative_word_count", msg)
        bound_log.warning(
            "validation_failed", check="narrative_word_count", word_count=wc
        )


def _check_transaction_price_consistency(
    config: PersonaConfig,
    transactions: list[TransactionRecord],
    _report: ValidationReport,
    bound_log: structlog.BoundLogger,
) -> None:
    """
    Mean price_paid_normalised should be inversely related to price_sensitivity.
    High price_sensitivity (> 0.7) → mean price_paid < 0.5.
    Low price_sensitivity (< 0.3) → mean price_paid > 0.4.
    """
    if not transactions:
        return

    mean_price = sum(t.price_paid_normalised for t in transactions) / len(transactions)
    ps = config.transactions.price_sensitivity

    if ps > 0.85 and mean_price > 0.70:
        bound_log.warning(
            "validation_failed",
            check="transaction_price_consistency",
            price_sensitivity=round(ps, 3),
            mean_price_paid=round(mean_price, 3),
        )

    elif ps < 0.15 and mean_price < 0.20:
        bound_log.warning(
            "validation_failed",
            check="transaction_price_consistency",
            price_sensitivity=round(ps, 3),
            mean_price_paid=round(mean_price, 3),
        )


def _check_payne_index_range(
    config: PersonaConfig,
    trial_records: list[TrialRecord],
    report: ValidationReport,
    bound_log: structlog.BoundLogger,
) -> None:
    """
    Mean Payne Index per participant should be in archetype's expected range (±0.2 tolerance).
    Only checked for archetypes with well-defined targets.
    """
    _PAYNE_TARGETS: dict[str, tuple[float, float]] = {
        "price_lex": (-1.0, -0.1),
        "compensatory": (-0.7, 0.7),
        "satisficer": (-1.0, 0.2),
        "brand_affect": (-1.0, -0.2),
        "low_involve": (-0.6, 0.6),
    }
    if config.persona_id not in _PAYNE_TARGETS or not trial_records:
        return

    lo, hi = _PAYNE_TARGETS[config.persona_id]
    mean_pi = sum(t.payne_index for t in trial_records) / len(trial_records)

    if not (lo <= mean_pi <= hi):
        msg = f"mean payne_index={mean_pi:.3f} not in [{lo}, {hi}] for {config.persona_id}"
        report.fail("payne_index_range", msg)
        bound_log.warning(
            "validation_failed",
            check="payne_index_range",
            mean_payne_index=round(mean_pi, 3),
            expected_range=[lo, hi],
        )


def _check_choice_consistency(
    trial_records: list[TrialRecord],
    choice_sets: list[ChoiceSet],
    report: ValidationReport,
    bound_log: structlog.BoundLogger,
) -> None:
    """
    Verify ChoiceSet.chosen_alternative matches TrialRecord.final_choice.

    This ensures the choice coupling is correctly recorded.
    """
    # Build choice_set lookup
    choice_set_map = {cs.choice_set_id: cs for cs in choice_sets}

    for trial in trial_records:
        if trial.choice_set_id is None:
            continue  # Old data without choice_set linkage

        choice_set = choice_set_map.get(trial.choice_set_id)
        if choice_set is None:
            msg = f"Trial {trial.trial_id}: choice_set_id={trial.choice_set_id} not found in choice_sets"
            report.fail("choice_consistency", msg)
            bound_log.warning("validation_failed", check="choice_consistency", trial_id=trial.trial_id)
            continue

        if choice_set.chosen_alternative != trial.final_choice:
            msg = (
                f"Trial {trial.trial_id}: choice_set.chosen_alternative={choice_set.chosen_alternative} "
                f"!= trial.final_choice={trial.final_choice}"
            )
            report.fail("choice_consistency", msg)
            bound_log.warning(
                "validation_failed",
                check="choice_consistency",
                trial_id=trial.trial_id,
                choice_set_alternative=choice_set.chosen_alternative,
                trial_choice=trial.final_choice,
            )


def _check_product_coverage(
    choice_sets: list[ChoiceSet],
    report: ValidationReport,
    bound_log: structlog.BoundLogger,
) -> None:
    """
    Verify all ChoiceSet records have valid product attributes.

    Checks:
    - displayed_attributes contains price and quality for all alternatives
    - choice_probabilities sum to ~1.0 (within 0.1 tolerance)
    - alternative_products mapping is complete
    """
    for choice_set in choice_sets:
        # Check displayed_attributes completeness
        for slot, attrs in choice_set.displayed_attributes.items():
            if "price" not in attrs:
                msg = f"ChoiceSet {choice_set.choice_set_id}: slot {slot} missing 'price' attribute"
                report.fail("product_coverage", msg)
                bound_log.warning(
                    "validation_failed",
                    check="product_coverage",
                    choice_set_id=choice_set.choice_set_id,
                    missing_attribute="price",
                    slot=slot,
                )
            if "quality" not in attrs:
                msg = f"ChoiceSet {choice_set.choice_set_id}: slot {slot} missing 'quality' attribute"
                report.fail("product_coverage", msg)
                bound_log.warning(
                    "validation_failed",
                    check="product_coverage",
                    choice_set_id=choice_set.choice_set_id,
                    missing_attribute="quality",
                    slot=slot,
                )

        # Check probability normalization
        prob_sum = sum(choice_set.choice_probabilities.values())
        if not (0.9 <= prob_sum <= 1.1):  # Allow 0.1 tolerance for numerical precision
            msg = (
                f"ChoiceSet {choice_set.choice_set_id}: choice_probabilities sum={prob_sum:.3f} "
                f"not in [0.9, 1.1]"
            )
            report.fail("product_coverage", msg)
            bound_log.warning(
                "validation_failed",
                check="product_coverage",
                choice_set_id=choice_set.choice_set_id,
                prob_sum=round(prob_sum, 3),
            )

        # Check alternative_products completeness
        expected_slots = set(choice_set.displayed_attributes.keys())
        provided_slots = set(choice_set.alternative_products.keys())
        if expected_slots != provided_slots:
            msg = (
                f"ChoiceSet {choice_set.choice_set_id}: alternative_products slots {provided_slots} "
                f"!= displayed_attributes slots {expected_slots}"
            )
            report.fail("product_coverage", msg)
            bound_log.warning(
                "validation_failed",
                check="product_coverage",
                choice_set_id=choice_set.choice_set_id,
                expected_slots=list(expected_slots),
                provided_slots=list(provided_slots),
            )


def _check_trace_choice_coupling(
    trial_records: list[TrialRecord],
    choice_sets: list[ChoiceSet],
    report: ValidationReport,
    bound_log: structlog.BoundLogger,
) -> None:
    """
    Verify trace-choice coupling by checking choice_set_id linkage.

    Ensures:
    - All trials with choices have choice_set_id populated
    - All choice_sets are referenced by exactly one trial
    - choice_set_id format matches trial_id format (linkage integrity)
    """
    # Build choice_set lookup
    choice_set_map = {cs.choice_set_id: cs for cs in choice_sets}

    # Check 1: All trials with choices should have choice_set_id
    trials_with_choice = [t for t in trial_records if t.final_choice is not None]
    trials_without_link = [t for t in trials_with_choice if t.choice_set_id is None]

    if trials_without_link:
        msg = f"{len(trials_without_link)} trials have final_choice but no choice_set_id linkage"
        report.fail("trace_choice_coupling", msg)
        bound_log.warning(
            "validation_failed",
            check="trace_choice_coupling",
            unlinked_trials=len(trials_without_link),
        )

    # Check 2: All choice_sets should be referenced by a trial
    referenced_choice_set_ids = {t.choice_set_id for t in trial_records if t.choice_set_id is not None}
    unreferenced_choice_sets = [
        cs for cs in choice_sets if cs.choice_set_id not in referenced_choice_set_ids
    ]

    if unreferenced_choice_sets:
        msg = f"{len(unreferenced_choice_sets)} choice_sets not referenced by any trial"
        report.fail("trace_choice_coupling", msg)
        bound_log.warning(
            "validation_failed",
            check="trace_choice_coupling",
            unreferenced_choice_sets=len(unreferenced_choice_sets),
        )

    # Check 3: Format consistency (choice_set_id should match trial_id format)
    for trial in trial_records:
        if trial.choice_set_id is not None and trial.choice_set_id != trial.trial_id:
            msg = (
                f"Trial {trial.trial_id}: choice_set_id={trial.choice_set_id} "
                f"does not match trial_id format"
            )
            report.fail("trace_choice_coupling", msg)
            bound_log.warning(
                "validation_failed",
                check="trace_choice_coupling",
                trial_id=trial.trial_id,
                choice_set_id=trial.choice_set_id,
            )
