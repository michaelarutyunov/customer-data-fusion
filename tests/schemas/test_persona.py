"""Unit tests for schemas/persona.py."""

from __future__ import annotations

import json
import dataclasses
import pytest

from schemas.persona import (
    PersonaConfig,
    StrategyParams,
    TransactionParams,
    PsychographicParams,
    NarrativeParams,
    Strategy,
    InspectionDepth,
    PriceConsciousness,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def strategy_lex() -> StrategyParams:
    return StrategyParams(
        primary_strategy=Strategy.LEXICOGRAPHIC,
        inspection_depth=InspectionDepth.SHALLOW,
        first_attribute="price",
        rejection_threshold_pct=0.35,
        p_reinspect=0.05,
        p_strategy_lapse=0.08,
        time_pressure_multiplier=0.50,
    )


@pytest.fixture
def strategy_compensatory() -> StrategyParams:
    return StrategyParams(
        primary_strategy=Strategy.COMPENSATORY,
        inspection_depth=InspectionDepth.DEEP,
        attribute_weights={
            "price": 0.35,
            "quality": 0.30,
            "brand": 0.20,
            "other": 0.15,
        },
        p_reinspect=0.30,
        p_strategy_lapse=0.05,
        time_pressure_multiplier=0.65,
    )


@pytest.fixture
def transaction_params() -> TransactionParams:
    return TransactionParams(
        price_sensitivity=0.85,
        brand_loyalty=0.25,
        purchase_frequency_per_month=3.5,
        basket_size_mean=2,
        channel_mix={"online": 0.60, "in_store": 0.40},
        price_variance_tolerance=0.10,
    )


@pytest.fixture
def psychographic_params() -> PsychographicParams:
    return PsychographicParams(
        involvement_score=0.55,
        maximiser_score=0.40,
        risk_tolerance=0.30,
        price_consciousness=PriceConsciousness.HIGH,
        openness_to_new=0.35,
    )


@pytest.fixture
def narrative_params() -> NarrativeParams:
    return NarrativeParams(
        age_range=(30, 45),
        household_type="family_with_children",
        category_relationship="habitual buyer",
        decision_style_description="Scans prices first.",
        price_attitude="price-first",
    )


@pytest.fixture
def persona_config(
    strategy_lex, transaction_params, psychographic_params, narrative_params
) -> PersonaConfig:
    return PersonaConfig(
        persona_id="price_lex",
        label="Price Lexicographic",
        strategy=strategy_lex,
        transactions=transaction_params,
        psychographic=psychographic_params,
        narrative=narrative_params,
    )


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_strategy_values_are_strings(self):
        for member in Strategy:
            assert isinstance(member.value, str)

    def test_strategy_serialises_as_value(self):
        assert json.dumps(Strategy.LEXICOGRAPHIC.value) == '"lexicographic"'
        assert json.dumps(Strategy.COMPENSATORY.value) == '"compensatory"'

    def test_inspection_depth_values(self):
        assert InspectionDepth.SHALLOW.value == "shallow"
        assert InspectionDepth.DEEP.value == "deep"
        assert InspectionDepth.VARIABLE.value == "variable"

    def test_price_consciousness_values(self):
        assert PriceConsciousness.LOW.value == "low"
        assert PriceConsciousness.MEDIUM.value == "medium"
        assert PriceConsciousness.HIGH.value == "high"

    @pytest.mark.parametrize(
        "enum_cls", [Strategy, InspectionDepth, PriceConsciousness]
    )
    def test_enum_is_str_subclass(self, enum_cls):
        for member in enum_cls:
            assert isinstance(member, str)


# ---------------------------------------------------------------------------
# StrategyParams tests
# ---------------------------------------------------------------------------


class TestStrategyParams:
    def test_frozen(self, strategy_lex):
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            strategy_lex.p_reinspect = 0.99  # type: ignore[misc]

    def test_lexicographic_fields(self, strategy_lex):
        assert strategy_lex.primary_strategy == Strategy.LEXICOGRAPHIC
        assert strategy_lex.first_attribute == "price"
        assert strategy_lex.rejection_threshold_pct == pytest.approx(0.35)

    def test_compensatory_attribute_weights_sum_to_one(self, strategy_compensatory):
        weights = strategy_compensatory.attribute_weights
        assert weights is not None
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_optional_fields_default_to_none(self):
        params = StrategyParams(
            primary_strategy=Strategy.RANDOM,
            inspection_depth=InspectionDepth.VARIABLE,
        )
        assert params.first_attribute is None
        assert params.rejection_threshold_pct is None
        assert params.attribute_weights is None
        assert params.aspiration_levels is None

    def test_default_noise_params(self):
        params = StrategyParams(
            primary_strategy=Strategy.RANDOM,
            inspection_depth=InspectionDepth.VARIABLE,
        )
        assert params.p_reinspect == pytest.approx(0.1)
        assert params.p_strategy_lapse == pytest.approx(0.05)
        assert params.time_pressure_multiplier == pytest.approx(0.6)

    def test_satisficing_aspiration_levels(self):
        params = StrategyParams(
            primary_strategy=Strategy.SATISFICING,
            inspection_depth=InspectionDepth.MEDIUM,
            aspiration_levels={"price": 0.60, "quality": 0.50},
        )
        assert params.aspiration_levels == {"price": 0.60, "quality": 0.50}


# ---------------------------------------------------------------------------
# TransactionParams tests
# ---------------------------------------------------------------------------


class TestTransactionParams:
    def test_frozen(self, transaction_params):
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            transaction_params.price_sensitivity = 0.0  # type: ignore[misc]

    def test_channel_mix_sums_to_one(self, transaction_params):
        assert sum(transaction_params.channel_mix.values()) == pytest.approx(1.0)

    def test_fields_present(self, transaction_params):
        assert 0.0 <= transaction_params.price_sensitivity <= 1.0
        assert 0.0 <= transaction_params.brand_loyalty <= 1.0
        assert transaction_params.purchase_frequency_per_month > 0
        assert transaction_params.basket_size_mean >= 1


# ---------------------------------------------------------------------------
# PsychographicParams tests
# ---------------------------------------------------------------------------


class TestPsychographicParams:
    def test_frozen(self, psychographic_params):
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            psychographic_params.maximiser_score = 0.0  # type: ignore[misc]

    def test_price_consciousness_is_enum(self, psychographic_params):
        assert isinstance(psychographic_params.price_consciousness, PriceConsciousness)

    def test_scores_in_range(self, psychographic_params):
        for attr in (
            "involvement_score",
            "maximiser_score",
            "risk_tolerance",
            "openness_to_new",
        ):
            val = getattr(psychographic_params, attr)
            assert 0.0 <= val <= 1.0, f"{attr}={val} out of [0,1]"


# ---------------------------------------------------------------------------
# NarrativeParams tests
# ---------------------------------------------------------------------------


class TestNarrativeParams:
    def test_frozen(self, narrative_params):
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            narrative_params.household_type = "other"  # type: ignore[misc]

    def test_age_range_tuple(self, narrative_params):
        lo, hi = narrative_params.age_range
        assert lo < hi

    def test_default_narrative_length(self, narrative_params):
        assert narrative_params.narrative_length_words == 300


# ---------------------------------------------------------------------------
# PersonaConfig tests
# ---------------------------------------------------------------------------


class TestPersonaConfig:
    def test_frozen(self, persona_config):
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            persona_config.persona_id = "other"  # type: ignore[misc]

    def test_fields_present(self, persona_config):
        assert persona_config.persona_id == "price_lex"
        assert persona_config.label == "Price Lexicographic"
        assert isinstance(persona_config.strategy, StrategyParams)
        assert isinstance(persona_config.transactions, TransactionParams)
        assert isinstance(persona_config.psychographic, PsychographicParams)
        assert isinstance(persona_config.narrative, NarrativeParams)

    def test_random_seed_defaults_none(self, persona_config):
        assert persona_config.random_seed is None

    def test_random_seed_can_be_set(
        self, strategy_lex, transaction_params, psychographic_params, narrative_params
    ):
        cfg = PersonaConfig(
            persona_id="price_lex",
            label="Price Lexicographic",
            strategy=strategy_lex,
            transactions=transaction_params,
            psychographic=psychographic_params,
            narrative=narrative_params,
            random_seed=42,
        )
        assert cfg.random_seed == 42

    def test_asdict_serialisable(self, persona_config):
        d = dataclasses.asdict(persona_config)
        # Enums serialise as their .value when manually extracted
        assert d["strategy"]["primary_strategy"] == "lexicographic"
        assert d["psychographic"]["price_consciousness"] == "high"

    def test_asdict_roundtrip_json(self, persona_config):
        d = dataclasses.asdict(persona_config)
        # Should be JSON-serialisable without a custom encoder
        json.dumps(d)
