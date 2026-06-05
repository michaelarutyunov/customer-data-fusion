"""Unit tests for schemas/text.py and schemas/psychographic.py."""

from __future__ import annotations

import dataclasses
import json
import pytest

from schemas.text import PersonaNarrative
from schemas.psychographic import PsychographicVector


# ---------------------------------------------------------------------------
# PersonaNarrative
# ---------------------------------------------------------------------------


@pytest.fixture
def persona_narrative() -> PersonaNarrative:
    return PersonaNarrative(
        participant_id="p001",
        persona_id="price_lex",
        category="electronics",
        text="This consumer prioritises price above all else...",
        word_count=47,
        model_id="deepseek-chat",
        prompt_version="v1",
    )


class TestPersonaNarrative:
    def test_frozen(self, persona_narrative):
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            persona_narrative.text = "overwrite"  # type: ignore[misc]

    def test_embedding_defaults_none(self, persona_narrative):
        assert persona_narrative.embedding is None
        assert persona_narrative.embedding_model_id is None

    def test_embedding_can_be_set(self):
        narrative = PersonaNarrative(
            participant_id="p001",
            persona_id="price_lex",
            category="electronics",
            text="Price-conscious shopper.",
            word_count=3,
            model_id="claude-sonnet-4-6",
            prompt_version="v1",
            embedding=[0.1, 0.2, 0.3],
            embedding_model_id="all-MiniLM-L6-v2",
        )
        assert narrative.embedding == [0.1, 0.2, 0.3]
        assert narrative.embedding_model_id == "all-MiniLM-L6-v2"

    def test_persona_id_present(self, persona_narrative):
        assert persona_narrative.persona_id == "price_lex"

    def test_asdict_json_serialisable(self, persona_narrative):
        json.dumps(dataclasses.asdict(persona_narrative))

    def test_asdict_with_embedding_json_serialisable(self):
        narrative = PersonaNarrative(
            participant_id="p001",
            persona_id="compensatory",
            category="food",
            text="Thorough shopper.",
            word_count=2,
            model_id="deepseek-chat",
            prompt_version="v1",
            embedding=[0.5] * 384,
            embedding_model_id="all-MiniLM-L6-v2",
        )
        json.dumps(dataclasses.asdict(narrative))


# ---------------------------------------------------------------------------
# PsychographicVector
# ---------------------------------------------------------------------------


@pytest.fixture
def psychographic_vector() -> PsychographicVector:
    return PsychographicVector(
        participant_id="p001",
        persona_id="price_lex",
        involvement_score=0.55,
        maximiser_score=0.40,
        risk_tolerance=0.30,
        price_consciousness=0.85,
        brand_sensitivity=0.25,
        openness_to_new=0.35,
        decision_style_dominant="analytical",
        age_band="35-44",
        household_type="family_with_children",
        employment_status="full_time",
        category="electronics",
        purchase_frequency_band="monthly",
    )


class TestPsychographicVector:
    def test_frozen(self, psychographic_vector):
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            psychographic_vector.maximiser_score = 0.0  # type: ignore[misc]

    def test_continuous_scores_in_range(self, psychographic_vector):
        for attr in (
            "involvement_score",
            "maximiser_score",
            "risk_tolerance",
            "price_consciousness",
            "brand_sensitivity",
            "openness_to_new",
        ):
            val = getattr(psychographic_vector, attr)
            assert 0.0 <= val <= 1.0, f"{attr}={val} out of [0,1]"

    def test_years_buying_defaults_none(self, psychographic_vector):
        assert psychographic_vector.years_buying_category is None

    def test_years_buying_can_be_set(self):
        vec = PsychographicVector(
            participant_id="p001",
            persona_id="compensatory",
            involvement_score=0.80,
            maximiser_score=0.75,
            risk_tolerance=0.55,
            price_consciousness=0.45,
            brand_sensitivity=0.50,
            openness_to_new=0.60,
            decision_style_dominant="analytical",
            age_band="45-54",
            household_type="couple",
            employment_status="full_time",
            category="food",
            purchase_frequency_band="weekly",
            years_buying_category=10,
        )
        assert vec.years_buying_category == 10

    def test_persona_id_present(self, psychographic_vector):
        assert psychographic_vector.persona_id == "price_lex"

    def test_asdict_json_serialisable(self, psychographic_vector):
        json.dumps(dataclasses.asdict(psychographic_vector))

    @pytest.mark.parametrize(
        "persona_id,expected_max_below",
        [
            ("satisficer", 0.40),
            ("low_involve", 0.40),
        ],
    )
    def test_satisficer_low_maximiser(self, persona_id, expected_max_below):
        vec = PsychographicVector(
            participant_id="p_test",
            persona_id=persona_id,
            involvement_score=0.30,
            maximiser_score=0.25,
            risk_tolerance=0.45,
            price_consciousness=0.55,
            brand_sensitivity=0.30,
            openness_to_new=0.45,
            decision_style_dominant="dependent",
            age_band="25-34",
            household_type="single",
            employment_status="full_time",
            category="food",
            purchase_frequency_band="monthly",
        )
        assert vec.maximiser_score < expected_max_below
