"""
Tests for generator/psychographic_generator.py
"""

from __future__ import annotations

import pytest

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
from generator.psychographic_generator import generate_psychographic


def _make_config(
    persona_id: str = "test",
    strategy: Strategy = Strategy.COMPENSATORY,
    involvement_score: float = 0.5,
    maximiser_score: float = 0.7,
    risk_tolerance: float = 0.5,
    price_consciousness: PriceConsciousness = PriceConsciousness.MEDIUM,
    openness_to_new: float = 0.5,
    brand_loyalty: float = 0.5,
    purchase_frequency_per_month: float = 2.0,
    age_range: tuple[int, int] = (30, 45),
    household_type: str = "single",
    random_seed: int = 42,
) -> PersonaConfig:
    return PersonaConfig(
        persona_id=persona_id,
        label=persona_id,
        strategy=StrategyParams(
            primary_strategy=strategy,
            inspection_depth=InspectionDepth.MEDIUM,
        ),
        transactions=TransactionParams(
            price_sensitivity=0.5,
            brand_loyalty=brand_loyalty,
            purchase_frequency_per_month=purchase_frequency_per_month,
            basket_size_mean=1,
            channel_mix={"online": 1.0},
            price_variance_tolerance=0.1,
        ),
        psychographic=PsychographicParams(
            involvement_score=involvement_score,
            maximiser_score=maximiser_score,
            risk_tolerance=risk_tolerance,
            price_consciousness=price_consciousness,
            openness_to_new=openness_to_new,
        ),
        narrative=NarrativeParams(
            age_range=age_range,
            household_type=household_type,
            category_relationship="habitual buyer",
            decision_style_description="Weighs all attributes carefully.",
            price_attitude="value-seeker",
        ),
        random_seed=random_seed,
    )


class TestReturnType:
    def test_returns_psychographic_vector(self) -> None:
        config = _make_config()
        result = generate_psychographic(config)
        assert isinstance(result, PsychographicVector)


class TestContinuousFieldsInRange:
    def test_involvement_score_in_range(self) -> None:
        config = _make_config(involvement_score=0.5)
        result = generate_psychographic(config)
        assert 0.0 <= result.involvement_score <= 1.0

    def test_maximiser_score_in_range(self) -> None:
        config = _make_config(maximiser_score=0.7)
        result = generate_psychographic(config)
        assert 0.0 <= result.maximiser_score <= 1.0

    def test_risk_tolerance_in_range(self) -> None:
        config = _make_config(risk_tolerance=0.4)
        result = generate_psychographic(config)
        assert 0.0 <= result.risk_tolerance <= 1.0

    def test_price_consciousness_in_range(self) -> None:
        config = _make_config()
        result = generate_psychographic(config)
        assert 0.0 <= result.price_consciousness <= 1.0

    def test_brand_sensitivity_in_range(self) -> None:
        config = _make_config(brand_loyalty=0.8)
        result = generate_psychographic(config)
        assert 0.0 <= result.brand_sensitivity <= 1.0

    def test_openness_to_new_in_range(self) -> None:
        config = _make_config(openness_to_new=0.6)
        result = generate_psychographic(config)
        assert 0.0 <= result.openness_to_new <= 1.0


class TestCategoricalFieldValues:
    def test_decision_style_valid(self) -> None:
        valid = {"analytical", "intuitive", "dependent", "avoidant", "spontaneous"}
        for strategy in Strategy:
            config = _make_config(strategy=strategy)
            result = generate_psychographic(config)
            assert result.decision_style_dominant in valid

    def test_age_band_valid(self) -> None:
        valid = {"18-24", "25-34", "35-44", "45-54", "55-64", "65+"}
        config = _make_config()
        result = generate_psychographic(config)
        assert result.age_band in valid

    def test_employment_status_valid(self) -> None:
        valid = {"full_time", "part_time", "self_employed", "not_employed", "retired"}
        config = _make_config()
        result = generate_psychographic(config)
        assert result.employment_status in valid

    def test_purchase_frequency_band_valid(self) -> None:
        valid = {"weekly", "monthly", "quarterly", "annually_or_less"}
        config = _make_config()
        result = generate_psychographic(config)
        assert result.purchase_frequency_band in valid


class TestDecisionStyleMapping:
    @pytest.mark.parametrize(
        "strategy,expected",
        [
            (Strategy.LEXICOGRAPHIC, "analytical"),
            (Strategy.COMPENSATORY, "analytical"),
            (Strategy.SATISFICING, "dependent"),
            (Strategy.AFFECT_HEURISTIC, "intuitive"),
            (Strategy.RANDOM, "spontaneous"),
            (Strategy.ADAPTIVE, "avoidant"),
        ],
    )
    def test_strategy_maps_correctly(self, strategy: Strategy, expected: str) -> None:
        config = _make_config(strategy=strategy)
        result = generate_psychographic(config)
        assert result.decision_style_dominant == expected


class TestPurchaseFrequencyBandMapping:
    def test_weekly_when_freq_gte_4(self) -> None:
        config = _make_config(purchase_frequency_per_month=5.0)
        result = generate_psychographic(config)
        assert result.purchase_frequency_band == "weekly"

    def test_monthly_when_freq_gte_1(self) -> None:
        config = _make_config(purchase_frequency_per_month=2.0)
        result = generate_psychographic(config)
        assert result.purchase_frequency_band == "monthly"

    def test_quarterly_when_freq_gte_025(self) -> None:
        config = _make_config(purchase_frequency_per_month=0.5)
        result = generate_psychographic(config)
        assert result.purchase_frequency_band == "quarterly"

    def test_annually_or_less_when_freq_lt_025(self) -> None:
        config = _make_config(purchase_frequency_per_month=0.1)
        result = generate_psychographic(config)
        assert result.purchase_frequency_band == "annually_or_less"


class TestArchetypeAcceptanceCriteria:
    def test_price_lex_high_price_consciousness(self) -> None:
        """price_lex persona: price_consciousness > 0.7 (HIGH -> 0.85 + noise)"""
        config = _make_config(
            persona_id="price_lex",
            price_consciousness=PriceConsciousness.HIGH,
            random_seed=42,
        )
        result = generate_psychographic(config)
        assert result.price_consciousness > 0.7

    def test_compensatory_high_maximiser(self) -> None:
        """compensatory persona: maximiser_score > 0.6"""
        config = _make_config(
            persona_id="compensatory",
            strategy=Strategy.COMPENSATORY,
            maximiser_score=0.8,
            random_seed=42,
        )
        result = generate_psychographic(config)
        assert result.maximiser_score > 0.6

    def test_brand_affect_high_brand_sensitivity(self) -> None:
        """brand_affect persona: brand_sensitivity > 0.7"""
        config = _make_config(
            persona_id="brand_affect",
            brand_loyalty=0.85,
            random_seed=42,
        )
        result = generate_psychographic(config)
        assert result.brand_sensitivity > 0.7

    def test_low_involve_low_involvement_score(self) -> None:
        """low_involve persona: involvement_score < 0.3"""
        config = _make_config(
            persona_id="low_involve",
            involvement_score=0.2,
            random_seed=42,
        )
        result = generate_psychographic(config)
        assert result.involvement_score < 0.3


class TestReproducibility:
    def test_same_seed_same_result(self) -> None:
        config = _make_config(random_seed=42)
        r1 = generate_psychographic(config)
        r2 = generate_psychographic(config)
        assert r1 == r2

    def test_different_seeds_differ(self) -> None:
        c1 = _make_config(random_seed=1)
        c2 = _make_config(random_seed=999)
        r1 = generate_psychographic(c1)
        r2 = generate_psychographic(c2)
        # At least one continuous field should differ
        assert (
            r1.involvement_score != r2.involvement_score
            or r1.maximiser_score != r2.maximiser_score
        )


class TestIdentityFields:
    def test_participant_id_matches_persona_id(self) -> None:
        config = _make_config(persona_id="mytest")
        result = generate_psychographic(config)
        assert result.participant_id == "mytest"
        assert result.persona_id == "mytest"

    def test_category_passed_through(self) -> None:
        config = _make_config()
        result = generate_psychographic(config, category="appliances")
        assert result.category == "appliances"

    def test_years_buying_category_is_none(self) -> None:
        config = _make_config()
        result = generate_psychographic(config)
        assert result.years_buying_category is None

    def test_household_type_from_narrative(self) -> None:
        config = _make_config(household_type="family_with_children")
        result = generate_psychographic(config)
        assert result.household_type == "family_with_children"


class TestAgeBand:
    @pytest.mark.parametrize(
        "age_range,expected_band",
        [
            ((18, 18), "18-24"),
            ((25, 25), "25-34"),
            ((35, 35), "35-44"),
            ((45, 45), "45-54"),
            ((55, 55), "55-64"),
            ((65, 65), "65+"),
        ],
    )
    def test_age_band_boundary(
        self, age_range: tuple[int, int], expected_band: str
    ) -> None:
        config = _make_config(age_range=age_range, random_seed=42)
        result = generate_psychographic(config)
        assert result.age_band == expected_band
