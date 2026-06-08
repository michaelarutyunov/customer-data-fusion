"""Unit tests for generator/validate.py."""

from __future__ import annotations


from schemas.persona import (
    InspectionDepth,
    NarrativeParams,
    PersonaConfig,
    PriceConsciousness,
    PsychographicParams,
    Strategy,
    StrategyParams,
    TransactionParams,
)
from schemas.psychographic import PsychographicVector
from schemas.text import PersonaNarrative
from schemas.trace import TrialRecord
from schemas.transaction import Channel, PurchaseType, TransactionRecord
from generator.validate import ValidationReport, validate_participant


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_config(
    persona_id: str,
    price_consciousness: PriceConsciousness = PriceConsciousness.HIGH,
    price_sensitivity: float = 0.85,
) -> PersonaConfig:
    return PersonaConfig(
        persona_id=persona_id,
        label=persona_id,
        strategy=StrategyParams(
            primary_strategy=Strategy.LEXICOGRAPHIC,
            inspection_depth=InspectionDepth.SHALLOW,
            first_attribute="price",
        ),
        transactions=TransactionParams(
            price_sensitivity=price_sensitivity,
            brand_loyalty=0.25,
            purchase_frequency_per_month=3.0,
            basket_size_mean=2,
            channel_mix={"online": 0.6, "in_store": 0.4},
            price_variance_tolerance=0.1,
        ),
        psychographic=PsychographicParams(
            involvement_score=0.55,
            maximiser_score=0.40,
            risk_tolerance=0.30,
            price_consciousness=price_consciousness,
            openness_to_new=0.35,
        ),
        narrative=NarrativeParams(
            age_range=(30, 45),
            household_type="family_with_children",
            category_relationship="habitual buyer",
            decision_style_description="Scans prices first.",
            price_attitude="price-first",
        ),
    )


def _make_psychographic(
    persona_id: str, price_consciousness: float = 0.85, brand_sensitivity: float = 0.25
) -> PsychographicVector:
    return PsychographicVector(
        participant_id=persona_id,
        persona_id=persona_id,
        involvement_score=0.55,
        maximiser_score=0.40,
        risk_tolerance=0.30,
        price_consciousness=price_consciousness,
        brand_sensitivity=brand_sensitivity,
        openness_to_new=0.35,
        decision_style_dominant="analytical",
        age_band="35-44",
        household_type="family_with_children",
        employment_status="full_time",
        category="electronics",
        purchase_frequency_band="monthly",
    )


def _make_narrative(word_count: int = 300) -> PersonaNarrative:
    return PersonaNarrative(
        participant_id="p001",
        persona_id="price_lex",
        category="electronics",
        text=" ".join(["word"] * word_count),
        word_count=word_count,
        model_id="deepseek-chat",
        prompt_version="v1",
    )


def _make_transaction(
    price_paid: float = 0.20, brand_tier: str = "value"
) -> TransactionRecord:
    return TransactionRecord(
        participant_id="p001",
        transaction_id="tx001",
        days_before_session=30,
        category="electronics",
        product_id="prod_abc",
        brand_tier=brand_tier,
        price_paid_normalised=price_paid,
        quantity=1,
        channel=Channel.ONLINE,
        purchase_type=PurchaseType.PLANNED,
        on_promotion=False,
        persona_id="price_lex",
    )


def _make_trial(
    persona_id: str = "price_lex", payne_index: float = -0.7
) -> TrialRecord:
    return TrialRecord(
        participant_id=persona_id,
        trial_id="t001",
        session_id="s001",
        trial_index=0,
        category="electronics",
        n_alternatives=3,
        n_attributes=4,
        time_pressure=False,
        final_choice="A",
        confidence_rating=4,
        total_acquisitions=6,
        prop_cells_inspected=0.5,
        payne_index=payne_index,
        persona_id=persona_id,
    )


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------


class TestValidationReport:
    def test_starts_passed(self):
        r = ValidationReport(participant_id="p001")
        assert r.passed is True
        assert r.failures == []

    def test_fail_marks_not_passed(self):
        r = ValidationReport(participant_id="p001")
        r.fail("check_name", "some message")
        assert r.passed is False
        assert len(r.failures) == 1
        assert r.failures[0] == ("check_name", "some message")

    def test_multiple_failures(self):
        r = ValidationReport(participant_id="p001")
        r.fail("check_a", "msg a")
        r.fail("check_b", "msg b")
        assert len(r.failures) == 2


# ---------------------------------------------------------------------------
# Price consciousness check
# ---------------------------------------------------------------------------


class TestPriceConsciousnessCheck:
    def test_price_lex_passes_with_high_pc(self):
        config = _make_config("price_lex", PriceConsciousness.HIGH)
        psycho = _make_psychographic("price_lex", price_consciousness=0.85)
        report = validate_participant(config, [], [], psycho, _make_narrative())
        checks = [f[0] for f in report.failures]
        assert "price_consciousness" not in checks

    def test_price_lex_no_fail_when_pc_in_range(self):
        config = _make_config("price_lex", PriceConsciousness.HIGH)
        # Directional checks removed — individual variation is expected. Only data bugs fail.
        psycho = _make_psychographic("price_lex", price_consciousness=0.20)
        report = validate_participant(config, [], [], psycho, _make_narrative())
        checks = [f[0] for f in report.failures]
        assert "price_consciousness" not in checks

    def test_quality_lex_passes_with_low_pc(self):
        config = _make_config(
            "quality_lex", PriceConsciousness.LOW, price_sensitivity=0.25
        )
        psycho = _make_psychographic("quality_lex", price_consciousness=0.20)
        report = validate_participant(config, [], [], psycho, _make_narrative())
        checks = [f[0] for f in report.failures]
        assert "price_consciousness" not in checks

    def test_quality_lex_no_fail_when_pc_in_range(self):
        config = _make_config(
            "quality_lex", PriceConsciousness.LOW, price_sensitivity=0.25
        )
        # Directional checks removed — individual variation is expected. Only data bugs fail.
        psycho = _make_psychographic("quality_lex", price_consciousness=0.80)
        report = validate_participant(config, [], [], psycho, _make_narrative())
        checks = [f[0] for f in report.failures]
        assert "price_consciousness" not in checks


# ---------------------------------------------------------------------------
# Brand sensitivity check
# ---------------------------------------------------------------------------


class TestBrandSensitivityCheck:
    def test_non_brand_affect_skips_check(self):
        config = _make_config("price_lex")
        psycho = _make_psychographic("price_lex", brand_sensitivity=0.10)
        report = validate_participant(config, [], [], psycho, _make_narrative())
        checks = [f[0] for f in report.failures]
        assert "brand_sensitivity" not in checks

    def test_brand_affect_passes_with_high_bs(self):
        config = _make_config("brand_affect")
        psycho = _make_psychographic("brand_affect", brand_sensitivity=0.80)
        transactions = [_make_transaction(brand_tier="premium")] * 8 + [
            _make_transaction(brand_tier="mid")
        ] * 2
        report = validate_participant(
            config, [], transactions, psycho, _make_narrative()
        )
        checks = [f[0] for f in report.failures]
        assert "brand_sensitivity" not in checks

    def test_brand_affect_no_fail_with_low_bs(self):
        config = _make_config("brand_affect")
        # brand_sensitivity directional check removed — individual variation is expected.
        psycho = _make_psychographic("brand_affect", brand_sensitivity=0.30)
        report = validate_participant(config, [], [], psycho, _make_narrative())
        checks = [f[0] for f in report.failures]
        assert "brand_sensitivity" not in checks

    def test_brand_tier_concentration_fails_when_spread(self):
        config = _make_config("brand_affect")
        psycho = _make_psychographic("brand_affect", brand_sensitivity=0.85)
        # Spread across 5 tiers equally → top-2 = 40% < 50% threshold
        transactions = [
            _make_transaction(brand_tier="premium"),
            _make_transaction(brand_tier="mid"),
            _make_transaction(brand_tier="value"),
            _make_transaction(brand_tier="own_label"),
            _make_transaction(brand_tier="luxury"),
        ]
        report = validate_participant(
            config, [], transactions, psycho, _make_narrative()
        )
        checks = [f[0] for f in report.failures]
        assert "brand_tier_concentration" in checks


# ---------------------------------------------------------------------------
# Narrative word count check
# ---------------------------------------------------------------------------


class TestNarrativeWordCount:
    def test_passes_within_range(self):
        report = validate_participant(
            _make_config("price_lex"),
            [],
            [],
            _make_psychographic("price_lex"),
            _make_narrative(word_count=300),
        )
        checks = [f[0] for f in report.failures]
        assert "narrative_word_count" not in checks

    def test_fails_too_short(self):
        report = validate_participant(
            _make_config("price_lex"),
            [],
            [],
            _make_psychographic("price_lex"),
            _make_narrative(word_count=150),
        )
        checks = [f[0] for f in report.failures]
        assert "narrative_word_count" in checks

    def test_fails_too_long(self):
        report = validate_participant(
            _make_config("price_lex"),
            [],
            [],
            _make_psychographic("price_lex"),
            _make_narrative(word_count=500),
        )
        checks = [f[0] for f in report.failures]
        assert "narrative_word_count" in checks


# ---------------------------------------------------------------------------
# Transaction price consistency check
# ---------------------------------------------------------------------------


class TestTransactionPriceConsistency:
    def test_high_sensitivity_low_price_passes(self):
        config = _make_config("price_lex", price_sensitivity=0.85)
        transactions = [_make_transaction(price_paid=0.20)] * 10
        report = validate_participant(
            config,
            [],
            transactions,
            _make_psychographic("price_lex"),
            _make_narrative(),
        )
        checks = [f[0] for f in report.failures]
        assert "transaction_price_consistency" not in checks

    def test_high_sensitivity_high_price_warns_not_fails(self):
        # Threshold widened to ps>0.85 (strict) and mean_price>0.70; this is warning-only now.
        config = _make_config("price_lex", price_sensitivity=0.90)
        transactions = [_make_transaction(price_paid=0.80)] * 10
        report = validate_participant(
            config,
            [],
            transactions,
            _make_psychographic("price_lex"),
            _make_narrative(),
        )
        # Not a hard failure — only a WARNING log; report should still pass
        checks = [f[0] for f in report.failures]
        assert "transaction_price_consistency" not in checks

    def test_empty_transactions_skips_check(self):
        config = _make_config("price_lex", price_sensitivity=0.85)
        report = validate_participant(
            config, [], [], _make_psychographic("price_lex"), _make_narrative()
        )
        checks = [f[0] for f in report.failures]
        assert "transaction_price_consistency" not in checks


# ---------------------------------------------------------------------------
# Payne index range check
# ---------------------------------------------------------------------------


class TestPayneIndexRangeCheck:
    def test_price_lex_in_range_passes(self):
        config = _make_config("price_lex")
        trials = [_make_trial("price_lex", payne_index=-0.70)] * 10
        report = validate_participant(
            config, trials, [], _make_psychographic("price_lex"), _make_narrative()
        )
        checks = [f[0] for f in report.failures]
        assert "payne_index_range" not in checks

    def test_price_lex_out_of_range_fails(self):
        config = _make_config("price_lex")
        # PI = +0.9 is entirely wrong for price_lex
        trials = [_make_trial("price_lex", payne_index=0.9)] * 10
        report = validate_participant(
            config, trials, [], _make_psychographic("price_lex"), _make_narrative()
        )
        checks = [f[0] for f in report.failures]
        assert "payne_index_range" in checks

    def test_unknown_archetype_skips_check(self):
        config = _make_config("adaptive")
        trials = [_make_trial("adaptive", payne_index=0.9)] * 5
        report = validate_participant(
            config, trials, [], _make_psychographic("adaptive"), _make_narrative()
        )
        checks = [f[0] for f in report.failures]
        assert "payne_index_range" not in checks

    def test_empty_trials_skips_check(self):
        config = _make_config("price_lex")
        report = validate_participant(
            config, [], [], _make_psychographic("price_lex"), _make_narrative()
        )
        checks = [f[0] for f in report.failures]
        assert "payne_index_range" not in checks


# ---------------------------------------------------------------------------
# Integration: all checks combined
# ---------------------------------------------------------------------------


class TestValidateParticipantIntegration:
    def test_well_configured_price_lex_passes_all(self):
        config = _make_config(
            "price_lex", PriceConsciousness.HIGH, price_sensitivity=0.85
        )
        psycho = _make_psychographic(
            "price_lex", price_consciousness=0.85, brand_sensitivity=0.25
        )
        transactions = [_make_transaction(price_paid=0.15)] * 10
        trials = [_make_trial("price_lex", payne_index=-0.70)] * 10
        narrative = _make_narrative(word_count=300)
        report = validate_participant(config, trials, transactions, psycho, narrative)
        assert report.passed
        assert report.failures == []

    def test_returns_validation_report_type(self):
        config = _make_config("price_lex")
        result = validate_participant(
            config, [], [], _make_psychographic("price_lex"), _make_narrative()
        )
        assert isinstance(result, ValidationReport)

    def test_participant_id_set(self):
        config = _make_config("price_lex")
        report = validate_participant(
            config, [], [], _make_psychographic("price_lex"), _make_narrative()
        )
        assert report.participant_id == "price_lex"
