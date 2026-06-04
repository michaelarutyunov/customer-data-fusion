"""Unit tests for schemas/transaction.py."""
from __future__ import annotations

import dataclasses
import json
import pytest

from schemas.transaction import TransactionRecord, Channel, PurchaseType


@pytest.fixture
def transaction_record() -> TransactionRecord:
    return TransactionRecord(
        participant_id="p001",
        transaction_id="tx001",
        days_before_session=30,
        category="electronics",
        product_id="prod_abc",
        brand_tier="mid",
        price_paid_normalised=0.45,
        quantity=1,
        channel=Channel.ONLINE,
        purchase_type=PurchaseType.PLANNED,
        on_promotion=False,
        persona_id="price_lex",
    )


class TestChannelEnum:
    def test_values_are_strings(self):
        for member in Channel:
            assert isinstance(member.value, str)

    def test_specific_values(self):
        assert Channel.ONLINE.value == "online"
        assert Channel.IN_STORE.value == "in_store"
        assert Channel.CLICK_AND_COLLECT.value == "click_and_collect"


class TestPurchaseTypeEnum:
    def test_values_are_strings(self):
        for member in PurchaseType:
            assert isinstance(member.value, str)

    def test_specific_values(self):
        assert PurchaseType.PLANNED.value == "planned"
        assert PurchaseType.HABITUAL.value == "habitual"


class TestTransactionRecord:
    def test_frozen(self, transaction_record):
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            transaction_record.price_paid_normalised = 0.0  # type: ignore[misc]

    def test_price_normalised_range(self, transaction_record):
        assert 0.0 <= transaction_record.price_paid_normalised <= 1.0

    def test_channel_is_enum(self, transaction_record):
        assert isinstance(transaction_record.channel, Channel)

    def test_purchase_type_is_enum(self, transaction_record):
        assert isinstance(transaction_record.purchase_type, PurchaseType)

    def test_loyalty_card_defaults_none(self, transaction_record):
        assert transaction_record.loyalty_card is None

    def test_loyalty_card_can_be_set(self):
        record = TransactionRecord(
            participant_id="p001",
            transaction_id="tx002",
            days_before_session=10,
            category="food",
            product_id="prod_xyz",
            brand_tier="value",
            price_paid_normalised=0.20,
            quantity=2,
            channel=Channel.IN_STORE,
            purchase_type=PurchaseType.HABITUAL,
            on_promotion=True,
            persona_id="brand_affect",
            loyalty_card=True,
        )
        assert record.loyalty_card is True

    def test_persona_id_present(self, transaction_record):
        assert transaction_record.persona_id == "price_lex"

    def test_asdict_enum_values(self, transaction_record):
        d = dataclasses.asdict(transaction_record)
        assert d["channel"] == "online"
        assert d["purchase_type"] == "planned"

    def test_asdict_json_serialisable(self, transaction_record):
        json.dumps(dataclasses.asdict(transaction_record))
