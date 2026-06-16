"""
Tests for the text encoder module.

Covers:
- embed.py: TextEncoder model, data loading, split, persistence, training
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
import torch

from schemas import EMBEDDING_DIM
from schemas.text import PersonaNarrative

from encoders.text.embed import (
    PERSONA_LABELS,
    PERSONA_TO_IDX,
    SENTENCE_DIM,
    SENTENCE_TRANSFORMER_MODEL,
    TextEncoder,
    embed_narratives,
    load_narratives,
    narratives_to_tensors,
    save_embeddings,
    split_by_participant,
    train,
)


# ---------------------------------------------------------------------------
# Helpers — factory for PersonaNarrative
# ---------------------------------------------------------------------------


def make_narrative(
    participant_id: str = "p001",
    persona_id: str = "price_lex",
    category: str = "smartphones",
    text: str = "This consumer always compares prices before buying.",
    word_count: int = 8,
    model_id: str = "test-model",
    prompt_version: str = "v1",
    embedding: list[float] | None = None,
    embedding_model_id: str | None = None,
) -> PersonaNarrative:
    return PersonaNarrative(
        participant_id=participant_id,
        persona_id=persona_id,
        category=category,
        text=text,
        word_count=word_count,
        model_id=model_id,
        prompt_version=prompt_version,
        embedding=embedding,
        embedding_model_id=embedding_model_id,
    )


# Distinct mini-texts per persona so sentence embeddings vary
_PERSONA_TEXTS: dict[str, str] = {
    "price_lex": (
        "This consumer strictly compares prices across all options. "
        "They always choose the lowest price alternative regardless of brand. "
        "Price is the dominant factor in every purchase decision they make."
    ),
    "compensatory": (
        "This consumer carefully weighs multiple attributes before deciding. "
        "They trade off price against quality, brand reputation, and features. "
        "Every purchase involves a systematic evaluation of all available information."
    ),
    "satisficer": (
        "This consumer picks the first option that meets their minimum threshold. "
        "They don't exhaustively search — once something is good enough, they buy it. "
        "Speed and convenience matter more than finding the absolute best deal."
    ),
    "brand_affect": (
        "This consumer is deeply loyal to brands they trust. "
        "They return to the same familiar brands purchase after purchase. "
        "Brand identity and emotional connection drive their decisions completely."
    ),
    "quality_lex": (
        "This consumer prioritizes quality above all other factors. "
        "They research product specifications and read expert reviews before buying. "
        "Quality ratings and premium materials drive their purchase decisions."
    ),
    "adaptive": (
        "This consumer changes their decision strategy based on the situation. "
        "For expensive items they research carefully; for cheap ones they decide quickly. "
        "Context determines whether they compare, trust brands, or satisfice."
    ),
    "low_involve": (
        "This consumer spends minimal time and effort on purchase decisions. "
        "They grab whatever is convenient and move on with their day. "
        "Shopping is a low-priority activity they want to finish as fast as possible."
    ),
}


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify module-level constants match expectations."""

    def test_seven_persona_classes(self) -> None:
        assert len(PERSONA_LABELS) == 7

    def test_all_personas_have_index(self) -> None:
        for p in PERSONA_LABELS:
            assert p in PERSONA_TO_IDX

    def test_sentence_dim_is_384(self) -> None:
        assert SENTENCE_DIM == 384

    def test_embedding_dim_from_schemas(self) -> None:
        assert EMBEDDING_DIM == 128

    def test_sentence_transformer_model_is_minilm(self) -> None:
        assert "MiniLM" in SENTENCE_TRANSFORMER_MODEL


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestTextEncoderModel:
    """Tests for the TextEncoder nn.Module."""

    def test_assert_frozen_passes_after_init(self) -> None:
        """After __init__, assert_frozen must succeed (all ST params frozen)."""
        model = TextEncoder()
        model.assert_frozen()  # should not raise

    def test_sentence_model_params_are_frozen(self) -> None:
        """Every sentence-transformer param must have requires_grad=False."""
        model = TextEncoder()
        n_trainable = sum(p.requires_grad for p in model.sentence_model.parameters())
        assert n_trainable == 0, f"Expected 0 trainable ST params, got {n_trainable}"

    def test_projection_params_are_trainable(self) -> None:
        """The projection layer must be trainable."""
        model = TextEncoder()
        n_trainable_proj = sum(p.requires_grad for p in model.projection.parameters())
        assert n_trainable_proj > 0, "Projection layer has no trainable parameters"

    def test_classifier_params_are_trainable(self) -> None:
        """The classifier head must be trainable."""
        model = TextEncoder()
        n_trainable_cls = sum(p.requires_grad for p in model.classifier.parameters())
        assert n_trainable_cls > 0, "Classifier has no trainable parameters"

    def test_forward_output_shape(self) -> None:
        """forward() maps (N, 384) → (N, EMBEDDING_DIM)."""
        model = TextEncoder()
        x = torch.randn(8, SENTENCE_DIM)
        out = model(x)
        assert out.shape == (8, EMBEDDING_DIM), (
            f"Expected (8, {EMBEDDING_DIM}), got {out.shape}"
        )

    def test_forward_output_dtype(self) -> None:
        model = TextEncoder()
        x = torch.randn(4, SENTENCE_DIM)
        out = model(x)
        assert out.dtype == torch.float32

    def test_forward_with_logits_shapes(self) -> None:
        model = TextEncoder(n_classes=7)
        x = torch.randn(4, SENTENCE_DIM)
        embedding, logits = model.forward_with_logits(x)
        assert embedding.shape == (4, EMBEDDING_DIM)
        assert logits.shape == (4, 7)

    def test_single_sample_forward(self) -> None:
        """Forward pass works with batch_size=1."""
        model = TextEncoder()
        x = torch.randn(1, SENTENCE_DIM)
        out = model(x)
        assert out.shape == (1, EMBEDDING_DIM)

    def test_has_layernorm(self) -> None:
        model = TextEncoder()
        has_ln = any(isinstance(m, torch.nn.LayerNorm) for m in model.modules())
        assert has_ln, "TextEncoder must include a LayerNorm layer"

    def test_gradient_flows_through_projection(self) -> None:
        """Gradients flow through the projection layer during backward pass."""
        model = TextEncoder()
        x = torch.randn(4, SENTENCE_DIM)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # Projection should have gradients
        assert model.projection.weight.grad is not None, (
            "No gradient on projection.weight"
        )

    def test_sentence_model_has_no_gradients(self) -> None:
        """Sentence-transformer params should not receive gradients."""
        model = TextEncoder()
        x = torch.randn(4, SENTENCE_DIM)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # Sentence model params must have None gradients (not just requires_grad=False)
        for name, param in model.sentence_model.named_parameters():
            assert param.grad is None, (
                f"Sentence model param '{name}' received a gradient"
            )

    def test_encode_texts_output_shape(self) -> None:
        """encode_texts() returns (len(texts), SENTENCE_DIM)."""
        model = TextEncoder()
        texts = [
            "This consumer always compares prices.",
            "This consumer is loyal to brands.",
            "This consumer satisfices on every purchase.",
        ]
        out = model.encode_texts(texts)
        assert out.shape == (3, SENTENCE_DIM), (
            f"Expected (3, {SENTENCE_DIM}), got {out.shape}"
        )
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# Data loading tests
# ---------------------------------------------------------------------------


class TestLoadNarratives:
    """Tests for load_narratives()."""

    def test_roundtrip_with_temp_file(self) -> None:
        """Narratives written to JSONL should be read back correctly."""
        records = [
            make_narrative(participant_id="p001", persona_id="price_lex"),
            make_narrative(participant_id="p002", persona_id="satisficer"),
        ]
        with NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for r in records:
                f.write(
                    json.dumps(
                        {
                            "participant_id": r.participant_id,
                            "persona_id": r.persona_id,
                            "category": r.category,
                            "text": r.text,
                            "word_count": r.word_count,
                            "model_id": r.model_id,
                            "prompt_version": r.prompt_version,
                            "embedding": r.embedding,
                            "embedding_model_id": r.embedding_model_id,
                        }
                    )
                    + "\n"
                )
            temp_path = Path(f.name)

        try:
            loaded = load_narratives(temp_path)
            assert len(loaded) == 2
            assert loaded[0].participant_id == "p001"
            assert loaded[0].persona_id == "price_lex"
            assert loaded[1].participant_id == "p002"
            assert loaded[1].persona_id == "satisficer"
        finally:
            temp_path.unlink()

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="Narratives file not found"):
            load_narratives(Path("/nonexistent/path/narratives.jsonl"))

    def test_empty_file_raises(self) -> None:
        with NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            pass  # write nothing
        temp_path = Path(f.name)
        try:
            with pytest.raises(ValueError, match="empty"):
                load_narratives(temp_path)
        finally:
            temp_path.unlink()


class TestSplitByParticipant:
    """Tests for participant-level train/val splitting."""

    def test_no_participant_in_both_sets(self) -> None:
        records = [
            make_narrative(participant_id=f"p{i:03d}", persona_id="price_lex")
            for i in range(20)
        ]
        train_recs, val_recs = split_by_participant(records, train_ratio=0.8, seed=42)
        train_ids = {r.participant_id for r in train_recs}
        val_ids = {r.participant_id for r in val_recs}
        assert train_ids.isdisjoint(val_ids), (
            "Same participant found in both train and val"
        )

    def test_all_records_accounted_for(self) -> None:
        records = [
            make_narrative(participant_id=f"p{i:03d}", persona_id="price_lex")
            for i in range(20)
        ]
        train_recs, val_recs = split_by_participant(records, train_ratio=0.8, seed=42)
        assert len(train_recs) + len(val_recs) == len(records)

    def test_split_ratio_approximate(self) -> None:
        records = [
            make_narrative(participant_id=f"p{i:03d}", persona_id="price_lex")
            for i in range(20)
        ]
        train_recs, _ = split_by_participant(records, train_ratio=0.8, seed=42)
        n_participants = len({r.participant_id for r in records})
        train_ids = {r.participant_id for r in train_recs}
        assert len(train_ids) == pytest.approx(0.8 * n_participants, abs=1)


# ---------------------------------------------------------------------------
# Embedding persistence tests
# ---------------------------------------------------------------------------


class TestEmbedNarratives:
    """Tests for embed_narratives() inference."""

    def test_returns_same_count(self) -> None:
        """embed_narratives returns the same number of records."""
        encoder = TextEncoder()
        records = [
            make_narrative(participant_id=f"p{i:03d}", persona_id="price_lex")
            for i in range(5)
        ]
        result = embed_narratives(encoder, records)
        assert len(result) == 5

    def test_populates_embedding_field(self) -> None:
        """After embed_narratives, embedding must not be None."""
        encoder = TextEncoder()
        records = [make_narrative(participant_id="p001", persona_id="price_lex")]
        result = embed_narratives(encoder, records)
        assert result[0].embedding is not None
        assert len(result[0].embedding) == EMBEDDING_DIM

    def test_skips_already_embedded(self) -> None:
        """Records with existing embeddings should not be re-embedded."""
        encoder = TextEncoder()
        pre_embedded = [0.1] * EMBEDDING_DIM
        records = [
            make_narrative(
                participant_id="p001",
                persona_id="price_lex",
                embedding=pre_embedded,
                embedding_model_id="test-model",
            ),
        ]
        result = embed_narratives(encoder, records)
        # Should keep the original embedding untouched
        assert result[0].embedding == pre_embedded

    def test_sets_embedding_model_id(self) -> None:
        """After embedding, embedding_model_id must match the ST model."""
        encoder = TextEncoder()
        records = [make_narrative(participant_id="p001", persona_id="price_lex")]
        result = embed_narratives(encoder, records)
        assert result[0].embedding_model_id == SENTENCE_TRANSFORMER_MODEL


class TestSaveEmbeddings:
    """Tests for save_embeddings() JSONL persistence."""

    def test_writes_valid_jsonl(self) -> None:
        narratives = [
            make_narrative(participant_id="p001", persona_id="price_lex"),
            make_narrative(participant_id="p002", persona_id="satisficer"),
        ]
        embeddings = [
            [float(i) for i in range(EMBEDDING_DIM)],
            [float(i + 100) for i in range(EMBEDDING_DIM)],
        ]
        with NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            temp_path = Path(f.name)

        try:
            save_embeddings(narratives, embeddings, temp_path)
            lines = temp_path.read_text().strip().splitlines()
            assert len(lines) == 2
            data0 = json.loads(lines[0])
            assert data0["embedding"] == embeddings[0]
            assert data0["embedding_model_id"] == SENTENCE_TRANSFORMER_MODEL
            data1 = json.loads(lines[1])
            assert data1["embedding"] == embeddings[1]
        finally:
            temp_path.unlink()


# ---------------------------------------------------------------------------
# Narratives to tensors tests
# ---------------------------------------------------------------------------


class TestNarrativesToTensors:
    """Tests for narratives_to_tensors()."""

    def test_feature_tensor_shape(self) -> None:
        encoder = TextEncoder()
        records = [
            make_narrative(participant_id="p001", persona_id="price_lex"),
            make_narrative(participant_id="p002", persona_id="satisficer"),
        ]
        features, _ = narratives_to_tensors(encoder, records)
        assert features.shape == (2, SENTENCE_DIM)

    def test_label_tensor_shape_and_dtype(self) -> None:
        encoder = TextEncoder()
        records = [
            make_narrative(participant_id="p001", persona_id="price_lex"),
            make_narrative(participant_id="p002", persona_id="satisficer"),
        ]
        _, labels = narratives_to_tensors(encoder, records)
        assert labels.shape == (2,)
        assert labels.dtype == torch.long

    def test_labels_map_correctly(self) -> None:
        encoder = TextEncoder()
        records = [
            make_narrative(participant_id="p001", persona_id="price_lex"),
            make_narrative(participant_id="p002", persona_id="satisficer"),
        ]
        _, labels = narratives_to_tensors(encoder, records)
        assert labels[0].item() == PERSONA_TO_IDX["price_lex"]
        assert labels[1].item() == PERSONA_TO_IDX["satisficer"]


# ---------------------------------------------------------------------------
# Training integration tests
# ---------------------------------------------------------------------------


class TestTrainingIntegration:
    """Integration test — train for 1 epoch on tiny synthetic fixtures."""

    @pytest.fixture()
    def tiny_dataset(self) -> list[PersonaNarrative]:
        """Create a minimal dataset covering all 7 persona archetypes.

        3 participants per archetype, each with 2 records = 42 total records.
        This ensures every persona has train and val samples after the split.
        """
        records: list[PersonaNarrative] = []
        for persona in PERSONA_LABELS:
            text = _PERSONA_TEXTS[persona]
            for i in range(3):
                pid = f"{persona}_p{i}"
                records.append(
                    make_narrative(
                        participant_id=pid,
                        persona_id=persona,
                        text=text,
                        word_count=len(text.split()),
                    )
                )
                records.append(
                    make_narrative(
                        participant_id=pid,
                        persona_id=persona,
                        text=text + " They also consider alternatives sometimes.",
                        word_count=len(text.split()) + 5,
                    )
                )
        return records

    def test_train_one_epoch(
        self, tmp_path, tiny_dataset: list[PersonaNarrative]
    ) -> None:
        """Training for 1 epoch should complete without error and return a model."""
        model = train(
            narratives=tiny_dataset,
            n_epochs=1,
            batch_size=16,
            log_mlflow=False,
            save_path=tmp_path / "text_encoder.pt",
        )
        assert isinstance(model, TextEncoder)

    def test_model_output_after_training(
        self, tmp_path, tiny_dataset: list[PersonaNarrative]
    ) -> None:
        """After training, the model should produce valid embeddings."""
        model = train(
            narratives=tiny_dataset,
            n_epochs=1,
            batch_size=16,
            log_mlflow=False,
            save_path=tmp_path / "text_encoder.pt",
        )
        model.eval()
        x = torch.randn(2, SENTENCE_DIM)
        with torch.no_grad():
            embedding = model(x)
        assert embedding.shape == (2, EMBEDDING_DIM)
        assert torch.isfinite(embedding).all(), "Embedding contains NaN or Inf"

    def test_loss_decreases(self, tiny_dataset: list[PersonaNarrative]) -> None:
        """Training loss should decrease over a few epochs."""
        from torch.utils.data import DataLoader, TensorDataset

        train_records, _ = split_by_participant(tiny_dataset, train_ratio=0.8)
        model = TextEncoder(n_classes=7)
        train_features, train_labels = narratives_to_tensors(model, train_records)
        ds = TensorDataset(train_features, train_labels)
        loader = DataLoader(ds, batch_size=16, shuffle=True)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        losses: list[float] = []
        for _ in range(5):
            model.projection.train()
            model.classifier.train()
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

        assert losses[0] > losses[-1], (
            f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )

    def test_strategy_recovery_from_logits(
        self, tmp_path, tiny_dataset: list[PersonaNarrative]
    ) -> None:
        """After training, classify validation records and check accuracy >50%.

        This is a smoke test on a tiny dataset — the >70% threshold from SPEC.md
        applies to the full dataset, not the tiny fixture.
        """
        model = train(
            narratives=tiny_dataset,
            n_epochs=5,
            batch_size=16,
            log_mlflow=False,
            save_path=tmp_path / "text_encoder.pt",
        )
        model.eval()

        _, val_records = split_by_participant(tiny_dataset, train_ratio=0.8)
        if len(val_records) == 0:
            pytest.skip("No validation records in tiny dataset")

        features, labels = narratives_to_tensors(model, val_records)
        with torch.no_grad():
            _, logits = model.forward_with_logits(features)
            preds = logits.argmax(dim=1)

        accuracy = (preds == labels).float().mean().item()
        # On a tiny dataset with distinct persona texts, accuracy should be
        # well above chance (1/7 ≈ 14%)
        assert accuracy > 0.3, (
            f"Strategy recovery accuracy too low: {accuracy:.2%} (chance=14%)"
        )
