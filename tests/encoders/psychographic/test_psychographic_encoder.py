"""
Tests for the psychographic encoder module.

Covers:
- features.py: shape, vocab, one-hot encoding, ordinal encoding, None imputation
- model.py: forward pass shape, forward_with_logits shape, EMBEDDING_DIM contract
- train.py: integration test — 1 epoch on tiny fixtures, split by participant_id
"""

from __future__ import annotations

from typing import Optional

import pytest
import torch

from schemas import EMBEDDING_DIM
from schemas.psychographic import PsychographicVector

from encoders.psychographic.features import (
    AGE_BAND_ORDINAL,
    DECISION_STYLE_VOCAB,
    EMPLOYMENT_STATUS_VOCAB,
    FEATURE_DIM,
    PURCHASE_FREQUENCY_ORDINAL,
    YEARS_BUYING_MEDIAN,
    _one_hot,
    batch_to_feature_matrix,
    to_feature_vector,
)
from encoders.psychographic.model import PsychographicEncoder
from encoders.psychographic.train import (
    PERSONA_LABELS,
    PERSONA_TO_IDX,
    records_to_tensors,
    split_by_participant,
    train,
)


# ---------------------------------------------------------------------------
# Helpers — factory for PsychographicVector
# ---------------------------------------------------------------------------


def make_psych(
    participant_id: str = "p001",
    persona_id: str = "price_lex",
    involvement_score: float = 0.7,
    maximiser_score: float = 0.8,
    risk_tolerance: float = 0.3,
    price_consciousness: float = 0.9,
    brand_sensitivity: float = 0.2,
    openness_to_new: float = 0.4,
    decision_style_dominant: str = "analytical",
    age_band: str = "25-34",
    household_type: str = "single",
    employment_status: str = "full_time",
    category: str = "smartphones",
    purchase_frequency_band: str = "monthly",
    years_buying_category: Optional[int] = None,
) -> PsychographicVector:
    return PsychographicVector(
        participant_id=participant_id,
        persona_id=persona_id,
        involvement_score=involvement_score,
        maximiser_score=maximiser_score,
        risk_tolerance=risk_tolerance,
        price_consciousness=price_consciousness,
        brand_sensitivity=brand_sensitivity,
        openness_to_new=openness_to_new,
        decision_style_dominant=decision_style_dominant,
        age_band=age_band,
        household_type=household_type,
        employment_status=employment_status,
        category=category,
        purchase_frequency_band=purchase_frequency_band,
        years_buying_category=years_buying_category,
    )


# ---------------------------------------------------------------------------
# features.py tests
# ---------------------------------------------------------------------------


class TestFeatureVector:
    """Tests for to_feature_vector output shape and content."""

    def test_output_shape_is_19(self) -> None:
        vec = to_feature_vector(make_psych())
        assert vec.shape == (19,), f"Expected (19,), got {vec.shape}"

    def test_output_dtype_is_float32(self) -> None:
        vec = to_feature_vector(make_psych())
        assert vec.dtype == torch.float32

    def test_feature_dim_constant_matches(self) -> None:
        assert FEATURE_DIM == 19

    def test_continuous_fields_preserved(self) -> None:
        psych = make_psych(
            involvement_score=0.1,
            maximiser_score=0.2,
            risk_tolerance=0.3,
            price_consciousness=0.4,
            brand_sensitivity=0.5,
            openness_to_new=0.6,
        )
        vec = to_feature_vector(psych)
        # First 6 dims are continuous fields
        assert vec[0].item() == pytest.approx(0.1)
        assert vec[1].item() == pytest.approx(0.2)
        assert vec[2].item() == pytest.approx(0.3)
        assert vec[3].item() == pytest.approx(0.4)
        assert vec[4].item() == pytest.approx(0.5)
        assert vec[5].item() == pytest.approx(0.6)


class TestOneHotEncoding:
    """Tests for one-hot encoding of categorical fields."""

    def test_decision_style_onehot(self) -> None:
        psych = make_psych(decision_style_dominant="intuitive")
        vec = to_feature_vector(psych)
        # dims [6:12] = decision_style one-hot (6 styles)
        oh = vec[6:12]
        expected_idx = DECISION_STYLE_VOCAB.index("intuitive")
        assert oh.sum().item() == pytest.approx(1.0)
        assert oh[expected_idx].item() == pytest.approx(1.0)

    def test_employment_status_onehot(self) -> None:
        psych = make_psych(employment_status="retired")
        vec = to_feature_vector(psych)
        # dims [13:18] = employment_status one-hot (household_type removed)
        oh = vec[13:18]
        expected_idx = EMPLOYMENT_STATUS_VOCAB.index("retired")
        assert oh.sum().item() == pytest.approx(1.0)
        assert oh[expected_idx].item() == pytest.approx(1.0)

    def test_onehot_all_zeros_for_unknown_value(self) -> None:
        """If an unknown value is passed, one-hot is all zeros (no crash)."""
        result = _one_hot("nonexistent", DECISION_STYLE_VOCAB)
        assert all(v == 0.0 for v in result)

    def test_decision_style_vocab_has_6_classes(self) -> None:
        assert len(DECISION_STYLE_VOCAB) == 6

    def test_employment_status_vocab_has_5_classes(self) -> None:
        assert len(EMPLOYMENT_STATUS_VOCAB) == 5


class TestOrdinalEncoding:
    """Tests for ordinal encoding of age_band and purchase_frequency_band."""

    def test_age_band_ordinal_normalised(self) -> None:
        psych = make_psych(age_band="45-54")
        vec = to_feature_vector(psych)
        # dim [12] = age_band ordinal / 5.0 (after 6 continuous + 6 decision_style)
        expected = AGE_BAND_ORDINAL["45-54"] / 5.0
        assert vec[12].item() == pytest.approx(expected)

    def test_purchase_frequency_ordinal_normalised(self) -> None:
        psych = make_psych(purchase_frequency_band="weekly")
        vec = to_feature_vector(psych)
        # dim [18] = purchase_frequency ordinal / 3.0
        expected = PURCHASE_FREQUENCY_ORDINAL["weekly"] / 3.0
        assert vec[18].item() == pytest.approx(expected)

    def test_age_band_unknown_gets_default(self) -> None:
        """Unknown age_band uses fallback value 3.0 / 5.0."""
        psych = make_psych(age_band="unknown_band")
        vec = to_feature_vector(psych)
        assert vec[12].item() == pytest.approx(3.0 / 5.0)


class TestYearsBuyingImputation:
    """Tests for years_buying_category None imputation rule."""

    def test_median_is_5(self) -> None:
        assert YEARS_BUYING_MEDIAN == 5

    def test_years_buying_none_does_not_crash(self) -> None:
        """Records with years_buying_category=None should process without error."""
        psych = make_psych(years_buying_category=None)
        vec = to_feature_vector(psych)
        assert vec.shape == (19,)


class TestBatchToFeatureMatrix:
    """Tests for batch_to_feature_matrix."""

    def test_batch_shape(self) -> None:
        records = [make_psych(participant_id=f"p{i:03d}") for i in range(5)]
        matrix = batch_to_feature_matrix(records)
        assert matrix.shape == (5, 19)

    def test_batch_dtype(self) -> None:
        records = [make_psych()]
        matrix = batch_to_feature_matrix(records)
        assert matrix.dtype == torch.float32


# ---------------------------------------------------------------------------
# model.py tests
# ---------------------------------------------------------------------------


class TestPsychographicEncoderModel:
    """Tests for the PsychographicEncoder nn.Module."""

    def test_forward_output_shape(self) -> None:
        model = PsychographicEncoder()
        x = torch.randn(8, FEATURE_DIM)
        out = model(x)
        assert out.shape == (8, EMBEDDING_DIM), (
            f"Expected (8, {EMBEDDING_DIM}), got {out.shape}"
        )

    def test_forward_output_dtype(self) -> None:
        model = PsychographicEncoder()
        x = torch.randn(4, FEATURE_DIM)
        out = model(x)
        assert out.dtype == torch.float32

    def test_forward_with_logits_shapes(self) -> None:
        model = PsychographicEncoder(n_classes=7)
        x = torch.randn(4, FEATURE_DIM)
        embedding, logits = model.forward_with_logits(x)
        assert embedding.shape == (4, EMBEDDING_DIM)
        assert logits.shape == (4, 7)

    def test_embedding_dim_from_schemas(self) -> None:
        """EMBEDDING_DIM must be imported from schemas, never hardcoded."""
        assert EMBEDDING_DIM == 128

    def test_model_has_dropout(self) -> None:
        model = PsychographicEncoder()
        dropout_layers = [
            m for m in model.encoder.modules() if isinstance(m, torch.nn.Dropout)
        ]
        assert len(dropout_layers) == 1, "Expected exactly one Dropout layer"
        assert dropout_layers[0].p == 0.2

    def test_model_has_layernorm(self) -> None:
        model = PsychographicEncoder()
        ln_layers = [
            m for m in model.encoder.modules() if isinstance(m, torch.nn.LayerNorm)
        ]
        assert len(ln_layers) == 1, "Expected exactly one LayerNorm layer"

    def test_single_sample_forward(self) -> None:
        """Forward pass works with batch_size=1."""
        model = PsychographicEncoder()
        x = torch.randn(1, FEATURE_DIM)
        out = model(x)
        assert out.shape == (1, EMBEDDING_DIM)

    def test_gradient_flows(self) -> None:
        """Gradients flow through the encoder during backward pass."""
        model = PsychographicEncoder()
        x = torch.randn(4, FEATURE_DIM)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # Check that at least some parameters have non-None gradients
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad, "No gradients computed"


# ---------------------------------------------------------------------------
# train.py tests
# ---------------------------------------------------------------------------


class TestSplitByParticipant:
    """Tests for participant-level train/val splitting."""

    def test_no_participant_in_both_sets(self) -> None:
        records = [
            make_psych(participant_id=f"p{i:03d}", persona_id="price_lex")
            for i in range(20)
        ]
        train, val = split_by_participant(records, train_ratio=0.8, seed=42)
        train_ids = {r.participant_id for r in train}
        val_ids = {r.participant_id for r in val}
        assert train_ids.isdisjoint(val_ids), (
            "Same participant found in both train and val"
        )

    def test_all_records_accounted_for(self) -> None:
        records = [
            make_psych(participant_id=f"p{i:03d}", persona_id="price_lex")
            for i in range(20)
        ]
        train, val = split_by_participant(records, train_ratio=0.8, seed=42)
        assert len(train) + len(val) == len(records)

    def test_split_ratio_approximate(self) -> None:
        records = [
            make_psych(participant_id=f"p{i:03d}", persona_id="price_lex")
            for i in range(20)
        ]
        train, val = split_by_participant(records, train_ratio=0.8, seed=42)
        n_participants = len({r.participant_id for r in records})
        train_ids = {r.participant_id for r in train}
        assert len(train_ids) == pytest.approx(0.8 * n_participants, abs=1)


class TestRecordsToTensors:
    """Tests for converting records to feature/label tensors."""

    def test_feature_tensor_shape(self) -> None:
        records = [
            make_psych(participant_id="p001", persona_id="price_lex"),
            make_psych(participant_id="p002", persona_id="satisficer"),
        ]
        features, labels = records_to_tensors(records)
        assert features.shape == (2, FEATURE_DIM)

    def test_label_tensor_shape_and_dtype(self) -> None:
        records = [
            make_psych(participant_id="p001", persona_id="price_lex"),
            make_psych(participant_id="p002", persona_id="satisficer"),
        ]
        _, labels = records_to_tensors(records)
        assert labels.shape == (2,)
        assert labels.dtype == torch.long

    def test_labels_map_correctly(self) -> None:
        records = [
            make_psych(participant_id="p001", persona_id="price_lex"),
            make_psych(participant_id="p002", persona_id="satisficer"),
        ]
        _, labels = records_to_tensors(records)
        assert labels[0].item() == PERSONA_TO_IDX["price_lex"]
        assert labels[1].item() == PERSONA_TO_IDX["satisficer"]


class TestPersonaLabels:
    """Tests for the persona label mapping."""

    def test_seven_persona_classes(self) -> None:
        assert len(PERSONA_LABELS) == 7

    def test_all_personas_have_index(self) -> None:
        for p in PERSONA_LABELS:
            assert p in PERSONA_TO_IDX


class TestTrainingIntegration:
    """Integration test — train for 1 epoch on tiny synthetic fixtures."""

    @pytest.fixture()
    def tiny_dataset(self) -> list[PsychographicVector]:
        """Create a minimal dataset covering all 7 persona archetypes.

        3 participants per archetype, each with 2 records = 42 total records.
        This ensures every persona has train and val samples after the split.
        """
        records: list[PsychographicVector] = []
        for persona in PERSONA_LABELS:
            for i in range(3):
                pid = f"{persona}_p{i}"
                records.append(make_psych(participant_id=pid, persona_id=persona))
                records.append(
                    make_psych(
                        participant_id=pid,
                        persona_id=persona,
                        involvement_score=0.5,
                        maximiser_score=0.5,
                    )
                )
        return records

    def test_train_one_epoch(self, tiny_dataset: list[PsychographicVector]) -> None:
        """Training for 1 epoch should complete without error and return a model."""
        model = train(
            records=tiny_dataset,
            n_epochs=1,
            batch_size=16,
            log_mlflow=False,
        )
        assert isinstance(model, PsychographicEncoder)

    def test_model_output_after_training(
        self, tiny_dataset: list[PsychographicVector]
    ) -> None:
        """After training, the model should produce valid embeddings."""
        model = train(
            records=tiny_dataset,
            n_epochs=1,
            batch_size=16,
            log_mlflow=False,
        )
        model.eval()
        sample = to_feature_vector(make_psych()).unsqueeze(0)
        with torch.no_grad():
            embedding = model(sample)
        assert embedding.shape == (1, EMBEDDING_DIM)
        assert torch.isfinite(embedding).all(), "Embedding contains NaN or Inf"

    def test_loss_decreases(self, tiny_dataset: list[PsychographicVector]) -> None:
        """Training loss should decrease over a few epochs."""

        from torch.utils.data import DataLoader, TensorDataset

        # Manual training to capture loss values
        train_records, _ = split_by_participant(tiny_dataset, train_ratio=0.8)
        train_features, train_labels = records_to_tensors(train_records)
        ds = TensorDataset(train_features, train_labels)
        loader = DataLoader(ds, batch_size=16, shuffle=True)

        model = PsychographicEncoder(n_classes=7)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        losses: list[float] = []
        for epoch in range(5):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch_x, batch_y in loader:
                _, logits = model.forward_with_logits(batch_x)
                loss = criterion(logits, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            losses.append(epoch_loss / max(n_batches, 1))

        # Loss should generally decrease — check first > last
        assert losses[0] > losses[-1], (
            f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )
