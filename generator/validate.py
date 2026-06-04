"""
generator/validate.py

Cross-modal consistency checks for a single generated participant.

All failures are logged at WARNING via structlog; no exceptions are raised.
Stochastic simulation will produce legitimate outliers — hard failures would
abort valid batch runs.

Public API:
    validate_participant(
        config, trial_records, transactions, psychographic, narrative
    ) -> ValidationReport
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from schemas.persona import PersonaConfig, PriceConsciousness
from schemas.psychographic import PsychographicVector
from schemas.text import PersonaNarrative
from schemas.trace import TrialRecord
from schemas.transaction import TransactionRecord

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
) -> ValidationReport:
    """
    Run all 5 cross-modal consistency checks for one participant.

    Failures are logged at WARNING; the report is returned regardless.
    """
    report = ValidationReport(participant_id=config.persona_id)
    bound_log = log.bind(participant_id=config.persona_id, persona_id=config.persona_id)

    _check_price_consciousness(config, psychographic, transactions, report, bound_log)
    _check_brand_sensitivity(config, psychographic, transactions, report, bound_log)
    _check_narrative_word_count(narrative, report, bound_log)
    _check_transaction_price_consistency(config, transactions, report, bound_log)
    _check_payne_index_range(config, trial_records, report, bound_log)

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
    config: PersonaConfig,
    psychographic: PsychographicVector,
    transactions: list[TransactionRecord],
    report: ValidationReport,
    bound_log: structlog.BoundLogger,
) -> None:
    """
    price_lex: price_consciousness float should be > 0.7 (from HIGH enum).
    quality_lex: price_consciousness float should be < 0.4 (from LOW enum).
    """
    pc = psychographic.price_consciousness
    strategy_pc = config.psychographic.price_consciousness

    if strategy_pc == PriceConsciousness.HIGH and pc < 0.6:
        msg = f"HIGH price_consciousness expected >0.6, got {pc:.3f}"
        report.fail("price_consciousness", msg)
        bound_log.warning("validation_failed", check="price_consciousness", delta=0.6 - pc)

    elif strategy_pc == PriceConsciousness.LOW and pc > 0.5:
        msg = f"LOW price_consciousness expected <0.5, got {pc:.3f}"
        report.fail("price_consciousness", msg)
        bound_log.warning("validation_failed", check="price_consciousness", delta=pc - 0.5)


def _check_brand_sensitivity(
    config: PersonaConfig,
    psychographic: PsychographicVector,
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

    bs = psychographic.brand_sensitivity
    if bs < 0.6:
        msg = f"brand_affect brand_sensitivity expected >0.6, got {bs:.3f}"
        report.fail("brand_sensitivity", msg)
        bound_log.warning("validation_failed", check="brand_sensitivity", delta=0.6 - bs)

    if transactions:
        from collections import Counter
        tier_counts = Counter(t.brand_tier for t in transactions)
        top2_count = sum(v for _, v in tier_counts.most_common(2))
        concentration = top2_count / len(transactions)
        if concentration < 0.60:
            msg = f"brand_affect brand_tier concentration={concentration:.2f} expected >=0.60"
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
        bound_log.warning("validation_failed", check="narrative_word_count", word_count=wc)


def _check_transaction_price_consistency(
    config: PersonaConfig,
    transactions: list[TransactionRecord],
    report: ValidationReport,
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

    if ps > 0.7 and mean_price > 0.55:
        msg = f"high price_sensitivity={ps:.2f} but mean price_paid={mean_price:.3f} >0.55"
        report.fail("transaction_price_consistency", msg)
        bound_log.warning(
            "validation_failed",
            check="transaction_price_consistency",
            price_sensitivity=round(ps, 3),
            mean_price_paid=round(mean_price, 3),
        )

    elif ps < 0.3 and mean_price < 0.35:
        msg = f"low price_sensitivity={ps:.2f} but mean price_paid={mean_price:.3f} <0.35"
        report.fail("transaction_price_consistency", msg)
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
        "price_lex":    (-1.0, -0.4),
        "compensatory": (-0.4,  0.4),
        "satisficer":   (-0.7, -0.1),
        "brand_affect": (-1.0, -0.5),
        "low_involve":  (-0.3,  0.3),
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
