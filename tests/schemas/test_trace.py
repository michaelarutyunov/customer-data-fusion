"""Unit tests for schemas/trace.py."""
from __future__ import annotations

import dataclasses
import json
import pytest

from schemas.trace import AcquisitionEvent, TrialRecord


@pytest.fixture
def acquisition_event() -> AcquisitionEvent:
    return AcquisitionEvent(
        participant_id="p001",
        trial_id="t001",
        event_index=0,
        alternative_id="A",
        attribute_id="price",
        timestamp_s=0.5,
        dwell_ms=950.0,
        is_reinspection=False,
    )


@pytest.fixture
def trial_record() -> TrialRecord:
    return TrialRecord(
        participant_id="p001",
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
        payne_index=-0.7,
        persona_id="price_lex",
    )


class TestAcquisitionEvent:
    def test_frozen(self, acquisition_event):
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            acquisition_event.dwell_ms = 0.0  # type: ignore[misc]

    def test_fields(self, acquisition_event):
        assert acquisition_event.participant_id == "p001"
        assert acquisition_event.event_index == 0
        assert acquisition_event.is_reinspection is False
        assert acquisition_event.dwell_ms == pytest.approx(950.0)

    def test_boolean_field_type(self, acquisition_event):
        assert isinstance(acquisition_event.is_reinspection, bool)

    def test_reinspection_true(self):
        ev = AcquisitionEvent(
            participant_id="p001",
            trial_id="t001",
            event_index=5,
            alternative_id="A",
            attribute_id="price",
            timestamp_s=3.0,
            dwell_ms=800.0,
            is_reinspection=True,
        )
        assert ev.is_reinspection is True

    def test_asdict_json_serialisable(self, acquisition_event):
        json.dumps(dataclasses.asdict(acquisition_event))


class TestTrialRecord:
    def test_frozen(self, trial_record):
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            trial_record.payne_index = 0.0  # type: ignore[misc]

    def test_optional_fields_nullable(self):
        record = TrialRecord(
            participant_id="p001",
            trial_id="t002",
            session_id="s001",
            trial_index=1,
            category="food",
            n_alternatives=5,
            n_attributes=6,
            time_pressure=True,
            final_choice=None,
            confidence_rating=None,
            total_acquisitions=10,
            prop_cells_inspected=0.33,
            payne_index=0.1,
            persona_id="low_involve",
        )
        assert record.final_choice is None
        assert record.confidence_rating is None

    def test_prop_cells_range(self, trial_record):
        assert 0.0 <= trial_record.prop_cells_inspected <= 1.0

    def test_payne_index_range(self, trial_record):
        assert -1.0 <= trial_record.payne_index <= 1.0

    def test_persona_id_present(self, trial_record):
        assert trial_record.persona_id == "price_lex"

    def test_asdict_json_serialisable(self, trial_record):
        json.dumps(dataclasses.asdict(trial_record))
