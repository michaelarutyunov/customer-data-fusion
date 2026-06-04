"""
Tests for generator/transaction_simulator.py

All fixtures use random_seed=42 for reproducibility.
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
from schemas.transaction import Channel, PurchaseType, TransactionRecord
from generator.transaction_simulator import simulate_transactions


# ---------------------------------------------------------------------------
# Shared fixture builders
# ---------------------------------------------------------------------------

def _make_config(
    persona_id: str,
    price_sensitivity: float,
    brand_loyalty: float,
    purchase_frequency_per_month: float,
    basket_size_mean: int = 1,
    channel_mix: dict[str, float] | None = None,
    random_seed: int = 42,
) -> PersonaConfig:
    if channel_mix is None:
        channel_mix = {"online": 0.60, "in_store": 0.40}
    return PersonaConfig(
        persona_id=persona_id,
        label=persona_id,
        strategy=StrategyParams(
            primary_strategy=Strategy.LEXICOGRAPHIC,
            inspection_depth=InspectionDepth.SHALLOW,
        ),
        transactions=TransactionParams(
            price_sensitivity=price_sensitivity,
            brand_loyalty=brand_loyalty,
            purchase_frequency_per_month=purchase_frequency_per_month,
            basket_size_mean=basket_size_mean,
            channel_mix=channel_mix,
            price_variance_tolerance=0.2,
        ),
        psychographic=PsychographicParams(
            involvement_score=0.5,
            maximiser_score=0.5,
            risk_tolerance=0.5,
            price_consciousness=PriceConsciousness.MEDIUM,
            openness_to_new=0.5,
        ),
        narrative=NarrativeParams(
            age_range=(30, 40),
            household_type="single",
            category_relationship="occasional shopper",
            decision_style_description="Decides quickly based on price.",
            price_attitude="price-first",
        ),
        random_seed=random_seed,
    )


# ---------------------------------------------------------------------------
# Persona fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def price_lex_config() -> PersonaConfig:
    """High price sensitivity (0.85) — buys cheap."""
    return _make_config("price_lex", price_sensitivity=0.85, brand_loyalty=0.20, purchase_frequency_per_month=3.0)


@pytest.fixture
def quality_lex_config() -> PersonaConfig:
    """Low price sensitivity (0.25) — buys premium."""
    return _make_config("quality_lex", price_sensitivity=0.25, brand_loyalty=0.60, purchase_frequency_per_month=2.0)


@pytest.fixture
def brand_affect_config() -> PersonaConfig:
    """High brand loyalty (0.85) — sticks to 1–2 tiers."""
    return _make_config("brand_affect", price_sensitivity=0.40, brand_loyalty=0.85, purchase_frequency_per_month=4.0)


@pytest.fixture
def low_involve_config() -> PersonaConfig:
    """High purchase frequency (5.0/month) → ~60 transactions."""
    return _make_config("low_involve", price_sensitivity=0.50, brand_loyalty=0.30, purchase_frequency_per_month=5.0)


# ---------------------------------------------------------------------------
# Basic structural tests
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_list_of_transaction_records(self, price_lex_config: PersonaConfig) -> None:
        result = simulate_transactions(price_lex_config)
        assert isinstance(result, list)
        assert all(isinstance(r, TransactionRecord) for r in result)

    def test_empty_possible_with_zero_frequency(self) -> None:
        config = _make_config("zero_freq", price_sensitivity=0.5, brand_loyalty=0.5, purchase_frequency_per_month=0.0)
        result = simulate_transactions(config)
        assert isinstance(result, list)


class TestFieldValues:
    def test_transaction_ids_formatted_correctly(self, price_lex_config: PersonaConfig) -> None:
        records = simulate_transactions(price_lex_config)
        for i, r in enumerate(records):
            assert r.transaction_id == f"tx_{price_lex_config.persona_id}_{i:04d}"

    def test_persona_id_matches_config(self, price_lex_config: PersonaConfig) -> None:
        records = simulate_transactions(price_lex_config)
        for r in records:
            assert r.persona_id == price_lex_config.persona_id
            assert r.participant_id == price_lex_config.persona_id

    def test_days_before_session_in_range(self, price_lex_config: PersonaConfig) -> None:
        records = simulate_transactions(price_lex_config)
        for r in records:
            assert 1 <= r.days_before_session <= 365

    def test_price_paid_normalised_in_unit_interval(self, price_lex_config: PersonaConfig) -> None:
        records = simulate_transactions(price_lex_config)
        for r in records:
            assert 0.0 <= r.price_paid_normalised <= 1.0

    def test_quantity_at_least_one(self, price_lex_config: PersonaConfig) -> None:
        records = simulate_transactions(price_lex_config)
        for r in records:
            assert r.quantity >= 1

    def test_loyalty_card_is_none(self, price_lex_config: PersonaConfig) -> None:
        records = simulate_transactions(price_lex_config)
        for r in records:
            assert r.loyalty_card is None

    def test_category_embedded_correctly(self, price_lex_config: PersonaConfig) -> None:
        records = simulate_transactions(price_lex_config, category="groceries")
        for r in records:
            assert r.category == "groceries"

    def test_brand_tier_valid_values(self, price_lex_config: PersonaConfig) -> None:
        valid = {"premium", "mid", "value", "own_label"}
        records = simulate_transactions(price_lex_config)
        for r in records:
            assert r.brand_tier in valid

    def test_channel_is_channel_enum(self, price_lex_config: PersonaConfig) -> None:
        records = simulate_transactions(price_lex_config)
        for r in records:
            assert isinstance(r.channel, Channel)

    def test_purchase_type_is_enum(self, price_lex_config: PersonaConfig) -> None:
        records = simulate_transactions(price_lex_config)
        for r in records:
            assert isinstance(r.purchase_type, PurchaseType)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_output(self, price_lex_config: PersonaConfig) -> None:
        r1 = simulate_transactions(price_lex_config)
        r2 = simulate_transactions(price_lex_config)
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a == b

    def test_different_seed_different_output(self) -> None:
        c1 = _make_config("p", price_sensitivity=0.5, brand_loyalty=0.5, purchase_frequency_per_month=3.0, random_seed=1)
        c2 = _make_config("p", price_sensitivity=0.5, brand_loyalty=0.5, purchase_frequency_per_month=3.0, random_seed=99)
        r1 = simulate_transactions(c1)
        r2 = simulate_transactions(c2)
        # Very unlikely to be identical
        prices1 = [r.price_paid_normalised for r in r1]
        prices2 = [r.price_paid_normalised for r in r2]
        assert prices1 != prices2


# ---------------------------------------------------------------------------
# Validation targets
# ---------------------------------------------------------------------------

class TestPriceDistribution:
    def test_price_lex_mean_below_0_4(self, price_lex_config: PersonaConfig) -> None:
        """price_sensitivity=0.85 → mean price_paid_normalised < 0.4"""
        records = simulate_transactions(price_lex_config, n_months=36)
        mean_price = sum(r.price_paid_normalised for r in records) / len(records)
        assert mean_price < 0.4, f"Expected mean < 0.4, got {mean_price:.3f}"

    def test_quality_lex_mean_above_0_5(self, quality_lex_config: PersonaConfig) -> None:
        """price_sensitivity=0.25 → mean price_paid_normalised > 0.5"""
        records = simulate_transactions(quality_lex_config, n_months=36)
        mean_price = sum(r.price_paid_normalised for r in records) / len(records)
        assert mean_price > 0.5, f"Expected mean > 0.5, got {mean_price:.3f}"


class TestBrandLoyalty:
    def test_brand_affect_concentrated_in_1_2_tiers(self, brand_affect_config: PersonaConfig) -> None:
        """brand_loyalty=0.85 → >= 70% transactions in 1–2 distinct tiers."""
        records = simulate_transactions(brand_affect_config, n_months=24)
        tier_counts: dict[str, int] = {}
        for r in records:
            tier_counts[r.brand_tier] = tier_counts.get(r.brand_tier, 0) + 1
        total = len(records)
        sorted_tiers = sorted(tier_counts.items(), key=lambda x: -x[1])
        top2_count = sum(c for _, c in sorted_tiers[:2])
        pct = top2_count / total
        assert pct >= 0.70, f"Expected >= 70% in top-2 tiers, got {pct:.2%}"


class TestPurchaseFrequency:
    def test_low_involve_expected_transaction_count(self, low_involve_config: PersonaConfig) -> None:
        """purchase_frequency_per_month=5.0 × 12 → ~60 transactions (Poisson)."""
        # Use multiple seeds to get a stable estimate
        counts = []
        for seed in range(20):
            config = _make_config(
                "low_involve",
                price_sensitivity=0.50,
                brand_loyalty=0.30,
                purchase_frequency_per_month=5.0,
                random_seed=seed,
            )
            counts.append(len(simulate_transactions(config, n_months=12)))
        mean_count = sum(counts) / len(counts)
        # Poisson(60) has std ~7.7; mean should be close to 60
        assert 45 <= mean_count <= 75, f"Expected ~60 transactions, got mean {mean_count:.1f}"


# ---------------------------------------------------------------------------
# Channel sampling
# ---------------------------------------------------------------------------

class TestChannelSampling:
    def test_channel_mix_respected(self) -> None:
        """online:0.60, in_store:0.40 → proportions within tolerance."""
        config = _make_config(
            "ch_test",
            price_sensitivity=0.5,
            brand_loyalty=0.5,
            purchase_frequency_per_month=5.0,
            channel_mix={"online": 0.60, "in_store": 0.40},
            random_seed=42,
        )
        records = simulate_transactions(config, n_months=60)
        online = sum(1 for r in records if r.channel == Channel.ONLINE)
        total = len(records)
        online_pct = online / total
        # Allow ±10% tolerance
        assert 0.50 <= online_pct <= 0.70, f"Expected ~60% online, got {online_pct:.2%}"

    def test_click_and_collect_channel_supported(self) -> None:
        config = _make_config(
            "cc_test",
            price_sensitivity=0.5,
            brand_loyalty=0.5,
            purchase_frequency_per_month=5.0,
            channel_mix={"online": 0.20, "in_store": 0.40, "click_and_collect": 0.40},
            random_seed=42,
        )
        records = simulate_transactions(config, n_months=24)
        channels = {r.channel for r in records}
        assert Channel.CLICK_AND_COLLECT in channels
