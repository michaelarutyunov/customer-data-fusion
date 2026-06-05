"""Unit tests for generator/persona_sampler.py."""

from __future__ import annotations

import pytest

from schemas.persona import (
    PersonaConfig,
    Strategy,
    InspectionDepth,
    PriceConsciousness,
    StrategyParams,
    TransactionParams,
    PsychographicParams,
    NarrativeParams,
)
from generator.persona_sampler import sample_persona, list_archetype_ids


ALL_ARCHETYPES = [
    "price_lex",
    "quality_lex",
    "compensatory",
    "satisficer",
    "brand_affect",
    "low_involve",
    "adaptive",
]


class TestListArchetypeIds:
    def test_returns_all_seven(self):
        ids = list_archetype_ids()
        assert set(ids) == set(ALL_ARCHETYPES)

    def test_returns_list(self):
        assert isinstance(list_archetype_ids(), list)


class TestSamplePersonaReturnType:
    def test_returns_persona_config(self):
        cfg = sample_persona("price_lex", random_seed=42)
        assert isinstance(cfg, PersonaConfig)

    def test_nested_types(self):
        cfg = sample_persona("compensatory", random_seed=0)
        assert isinstance(cfg.strategy, StrategyParams)
        assert isinstance(cfg.transactions, TransactionParams)
        assert isinstance(cfg.psychographic, PsychographicParams)
        assert isinstance(cfg.narrative, NarrativeParams)

    def test_persona_id_matches_archetype(self):
        cfg = sample_persona("price_lex", random_seed=42)
        assert cfg.persona_id == "price_lex"

    def test_random_seed_stored(self):
        cfg = sample_persona("satisficer", random_seed=99)
        assert cfg.random_seed == 99

    def test_none_seed_accepted(self):
        cfg = sample_persona("low_involve", random_seed=None)
        assert cfg.random_seed is None


class TestStrategyFieldsPerArchetype:
    def test_price_lex_strategy(self):
        cfg = sample_persona("price_lex", random_seed=42)
        assert cfg.strategy.primary_strategy == Strategy.LEXICOGRAPHIC
        assert cfg.strategy.first_attribute == "price"
        assert cfg.strategy.inspection_depth == InspectionDepth.SHALLOW

    def test_compensatory_weights_sum_to_one(self):
        cfg = sample_persona("compensatory", random_seed=42)
        weights = cfg.strategy.attribute_weights
        assert weights is not None
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_satisficer_has_aspiration_levels(self):
        cfg = sample_persona("satisficer", random_seed=42)
        assert cfg.strategy.aspiration_levels is not None
        assert "price" in cfg.strategy.aspiration_levels
        assert "quality" in cfg.strategy.aspiration_levels

    def test_adaptive_has_both_weights_and_aspirations(self):
        cfg = sample_persona("adaptive", random_seed=42)
        assert cfg.strategy.attribute_weights is not None
        assert cfg.strategy.aspiration_levels is not None

    @pytest.mark.parametrize("archetype_id", ALL_ARCHETYPES)
    def test_p_reinspect_in_range(self, archetype_id):
        cfg = sample_persona(archetype_id, random_seed=1)
        assert 0.0 <= cfg.strategy.p_reinspect <= 1.0

    @pytest.mark.parametrize("archetype_id", ALL_ARCHETYPES)
    def test_p_strategy_lapse_in_range(self, archetype_id):
        cfg = sample_persona(archetype_id, random_seed=1)
        assert 0.0 <= cfg.strategy.p_strategy_lapse <= 1.0


class TestTransactionFields:
    def test_channel_mix_sums_to_one(self):
        cfg = sample_persona("price_lex", random_seed=42)
        assert sum(cfg.transactions.channel_mix.values()) == pytest.approx(
            1.0, abs=1e-9
        )

    def test_basket_size_at_least_one(self):
        for seed in range(20):
            cfg = sample_persona("low_involve", random_seed=seed)
            assert cfg.transactions.basket_size_mean >= 1

    @pytest.mark.parametrize("archetype_id", ALL_ARCHETYPES)
    def test_price_sensitivity_in_range(self, archetype_id):
        cfg = sample_persona(archetype_id, random_seed=7)
        assert 0.0 <= cfg.transactions.price_sensitivity <= 1.0

    @pytest.mark.parametrize("archetype_id", ALL_ARCHETYPES)
    def test_purchase_frequency_positive(self, archetype_id):
        cfg = sample_persona(archetype_id, random_seed=7)
        assert cfg.transactions.purchase_frequency_per_month > 0


class TestPsychographicFields:
    def test_price_lex_price_consciousness_high(self):
        cfg = sample_persona("price_lex", random_seed=42)
        assert cfg.psychographic.price_consciousness == PriceConsciousness.HIGH

    def test_quality_lex_price_consciousness_low(self):
        cfg = sample_persona("quality_lex", random_seed=42)
        assert cfg.psychographic.price_consciousness == PriceConsciousness.LOW

    @pytest.mark.parametrize("archetype_id", ALL_ARCHETYPES)
    def test_scores_in_unit_range(self, archetype_id):
        cfg = sample_persona(archetype_id, random_seed=3)
        for attr in (
            "involvement_score",
            "maximiser_score",
            "risk_tolerance",
            "openness_to_new",
        ):
            val = getattr(cfg.psychographic, attr)
            assert 0.0 <= val <= 1.0, f"{archetype_id}.{attr}={val} out of [0,1]"


class TestNarrativeFields:
    def test_age_range_valid(self):
        cfg = sample_persona("brand_affect", random_seed=42)
        lo, hi = cfg.narrative.age_range
        assert lo < hi

    def test_narrative_length_positive(self):
        cfg = sample_persona("compensatory", random_seed=42)
        assert cfg.narrative.narrative_length_words > 0

    def test_decision_style_nonempty(self):
        cfg = sample_persona("adaptive", random_seed=42)
        assert cfg.narrative.decision_style_description.strip() != ""


class TestReproducibility:
    def test_same_seed_same_result(self):
        cfg1 = sample_persona("price_lex", random_seed=42)
        cfg2 = sample_persona("price_lex", random_seed=42)
        assert cfg1.strategy.p_reinspect == pytest.approx(cfg2.strategy.p_reinspect)
        assert cfg1.transactions.price_sensitivity == pytest.approx(
            cfg2.transactions.price_sensitivity
        )

    def test_different_seeds_different_params(self):
        cfg1 = sample_persona("price_lex", random_seed=1)
        cfg2 = sample_persona("price_lex", random_seed=2)
        # With noise, continuous params should differ (extremely unlikely to collide)
        assert cfg1.strategy.p_reinspect != pytest.approx(cfg2.strategy.p_reinspect)

    @pytest.mark.parametrize("archetype_id", ALL_ARCHETYPES)
    def test_all_archetypes_reproducible(self, archetype_id):
        a = sample_persona(archetype_id, random_seed=42)
        b = sample_persona(archetype_id, random_seed=42)
        assert a.strategy.p_reinspect == pytest.approx(b.strategy.p_reinspect)


class TestInvalidArchetype:
    def test_raises_on_unknown_id(self):
        with pytest.raises(ValueError, match="Unknown archetype"):
            sample_persona("nonexistent_archetype", random_seed=0)
