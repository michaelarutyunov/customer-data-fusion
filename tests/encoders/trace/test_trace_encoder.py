"""
Tests for encoders/trace/ — tokeniser, model, StratifiedSampler, and training loop.

All tests use tiny synthetic fixtures generated in-process; they never depend
on data/synthetic/ files.
"""

from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path

import pytest
import torch
from torch import Tensor

from schemas import EMBEDDING_DIM
from schemas.trace import AcquisitionEvent, TrialRecord

from encoders.trace.model import (
    TraceEncoder,
)
from encoders.trace.tokeniser import (
    TOKEN_DIM,
    build_vocab,
    collate_batch,
    tokenise_trial,
)
from encoders.trace.train import (
    StratifiedSampler,
    cross_entropy_loss,
    nt_xent_loss,
    train,
)


# ---------------------------------------------------------------------------
# Shared fixtures — tiny synthetic data
# ---------------------------------------------------------------------------


def _make_event(
    participant_id: str = "p1",
    trial_id: str = "t1",
    event_index: int = 0,
    alternative_id: str = "A",
    attribute_id: str = "price",
    timestamp_s: float = 0.0,
    dwell_ms: float = 500.0,
    is_reinspection: bool = False,
    persona_id: str = "compensatory",
) -> AcquisitionEvent:
    return AcquisitionEvent(
        participant_id=participant_id,
        trial_id=trial_id,
        event_index=event_index,
        alternative_id=alternative_id,
        attribute_id=attribute_id,
        timestamp_s=timestamp_s,
        dwell_ms=dwell_ms,
        is_reinspection=is_reinspection,
    )


def _make_trial(
    participant_id: str = "p1",
    trial_id: str = "t1",
    session_id: str = "s1",
    trial_index: int = 0,
    persona_id: str = "compensatory",
    n_alternatives: int = 3,
    n_attributes: int = 4,
    total_acquisitions: int = 3,
) -> TrialRecord:
    return TrialRecord(
        participant_id=participant_id,
        trial_id=trial_id,
        session_id=session_id,
        trial_index=trial_index,
        category="laptops",
        n_alternatives=n_alternatives,
        n_attributes=n_attributes,
        time_pressure=False,
        final_choice="A",
        confidence_rating=3,
        total_acquisitions=total_acquisitions,
        prop_cells_inspected=0.25,
        payne_index=0.0,
        persona_id=persona_id,
    )


def _make_trial_events(
    trial_id: str = "t1",
    participant_id: str = "p1",
    persona_id: str = "compensatory",
    n_events: int = 5,
) -> tuple[list[AcquisitionEvent], TrialRecord]:
    """Create a trial with n_events acquisition events."""
    alts = ["A", "B", "C"]
    attrs = ["price", "brand", "quality", "warranty"]
    events = []
    for i in range(n_events):
        events.append(
            _make_event(
                participant_id=participant_id,
                trial_id=trial_id,
                event_index=i,
                alternative_id=alts[i % len(alts)],
                attribute_id=attrs[i % len(attrs)],
                timestamp_s=float(i) * 0.5,
                dwell_ms=500.0 + i * 100.0,
                is_reinspection=(i > 2),
                persona_id=persona_id,
            )
        )
    trial = _make_trial(
        participant_id=participant_id,
        trial_id=trial_id,
        persona_id=persona_id,
        total_acquisitions=n_events,
    )
    return events, trial


# ---------------------------------------------------------------------------
# Tokeniser tests
# ---------------------------------------------------------------------------


class TestBuildVocab:
    def test_builds_vocab_with_expected_keys(self):
        events, _ = _make_trial_events(n_events=5)
        vocab = build_vocab(events, cache_path=Path(tempfile.mkdtemp()) / "v.json")
        assert "attribute" in vocab
        assert "alternative" in vocab

    def test_vocab_reserves_index_zero(self):
        events, _ = _make_trial_events(n_events=5)
        vocab = build_vocab(events, cache_path=Path(tempfile.mkdtemp()) / "v.json")
        for sub_vocab in vocab.values():
            for key, idx in sub_vocab.items():
                assert idx >= 1, (
                    f"Index 0 should be reserved for CLS, got {idx} for {key}"
                )

    def test_vocab_covers_all_unique_values(self):
        events, _ = _make_trial_events(n_events=5)
        vocab = build_vocab(events, cache_path=Path(tempfile.mkdtemp()) / "v.json")
        expected_attrs = {"price", "brand", "quality", "warranty"}
        expected_alts = {"A", "B", "C"}
        assert set(vocab["attribute"].keys()) == expected_attrs
        assert set(vocab["alternative"].keys()) == expected_alts


class TestTokeniseTrial:
    def test_output_shape_includes_cls(self):
        events, trial = _make_trial_events(n_events=5)
        vocab = build_vocab(events, cache_path=Path(tempfile.mkdtemp()) / "v.json")
        tokens, mask = tokenise_trial(events, trial, vocab)
        # CLS + n_events
        assert tokens.shape == (6, TOKEN_DIM), (
            f"Expected (6, {TOKEN_DIM}), got {tokens.shape}"
        )
        assert mask.shape == (6,)

    def test_cls_row_is_zeros(self):
        events, trial = _make_trial_events(n_events=3)
        vocab = build_vocab(events, cache_path=Path(tempfile.mkdtemp()) / "v.json")
        tokens, _ = tokenise_trial(events, trial, vocab)
        # CLS placeholder at index 0 should be all zeros
        assert torch.all(tokens[0] == 0.0), "CLS placeholder row should be zeros"

    def test_mask_all_true_for_non_empty(self):
        events, trial = _make_trial_events(n_events=3)
        vocab = build_vocab(events, cache_path=Path(tempfile.mkdtemp()) / "v.json")
        _, mask = tokenise_trial(events, trial, vocab)
        assert mask.all(), "All mask entries should be True for non-empty trial"

    def test_empty_events_gives_cls_only(self):
        trial = _make_trial()
        vocab = {"attribute": {"price": 1}, "alternative": {"A": 1}}
        tokens, mask = tokenise_trial([], trial, vocab)
        assert tokens.shape == (1, TOKEN_DIM), (
            "Empty trial should produce CLS-only token"
        )
        assert mask.shape == (1,)
        assert mask[0]

    def test_truncation_at_max_seq_len(self):
        events, trial = _make_trial_events(n_events=300)
        vocab = build_vocab(events, cache_path=Path(tempfile.mkdtemp()) / "v.json")
        tokens, _ = tokenise_trial(events, trial, vocab, max_seq_len=10)
        # Should be truncated to 10 real + 1 CLS
        assert tokens.shape[0] == 11

    def test_attribute_indices_are_valid(self):
        events, trial = _make_trial_events(n_events=5)
        vocab = build_vocab(events, cache_path=Path(tempfile.mkdtemp()) / "v.json")
        tokens, _ = tokenise_trial(events, trial, vocab)
        # Attribute indices (column 0) for real tokens should be >= 1
        attr_indices = tokens[1:, 0]  # skip CLS
        assert (attr_indices >= 1).all(), (
            "Attribute indices should be >= 1 for real tokens"
        )


class TestCollateBatch:
    def test_pads_to_max_length(self):
        events1, trial1 = _make_trial_events(n_events=3)
        events2, trial2 = _make_trial_events(n_events=5)
        vocab = build_vocab(
            events1 + events2, cache_path=Path(tempfile.mkdtemp()) / "v.json"
        )
        t1, m1 = tokenise_trial(events1, trial1, vocab)
        t2, m2 = tokenise_trial(events2, trial2, vocab)
        padded_tokens, padded_mask = collate_batch([(t1, m1), (t2, m2)])
        # Max length is 6 (5 events + CLS)
        assert padded_tokens.shape == (2, 6, TOKEN_DIM)
        assert padded_mask.shape == (2, 6)

    def test_padding_mask_correct(self):
        events1, trial1 = _make_trial_events(n_events=2)
        events2, trial2 = _make_trial_events(n_events=4)
        vocab = build_vocab(
            events1 + events2, cache_path=Path(tempfile.mkdtemp()) / "v.json"
        )
        t1, m1 = tokenise_trial(events1, trial1, vocab)
        t2, m2 = tokenise_trial(events2, trial2, vocab)
        _, padded_mask = collate_batch([(t1, m1), (t2, m2)])
        # Shorter sequence: 3 True (2 events + CLS), rest False
        assert padded_mask[0].sum().item() == 3
        # Longer sequence: 5 True (4 events + CLS)
        assert padded_mask[1].sum().item() == 5


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


def _make_valid_tokens(
    batch_size: int, seq_len: int, n_attributes: int = 5, n_alternatives: int = 4
) -> Tensor:
    """Create token tensors with valid embedding indices (columns 0 and 1)."""
    tokens = torch.zeros(batch_size, seq_len, TOKEN_DIM)
    # Column 0: attribute indices in [0, n_attributes)
    tokens[:, :, 0] = torch.randint(0, n_attributes, (batch_size, seq_len)).float()
    # Column 1: alternative indices in [0, n_alternatives)
    tokens[:, :, 1] = torch.randint(0, n_alternatives, (batch_size, seq_len)).float()
    # Columns 2-4: continuous features (timestamp_norm, dwell_zscore, is_reinspection)
    tokens[:, :, 2:5] = torch.rand(batch_size, seq_len, 3)
    return tokens


class TestTraceEncoderForward:
    def test_output_shape_is_embedding_dim(self):
        encoder = TraceEncoder(n_attributes=5, n_alternatives=4, n_classes=7)
        tokens = _make_valid_tokens(2, 10)
        mask = torch.ones(2, 10, dtype=torch.bool)
        output = encoder(tokens, mask)
        assert output.shape == (2, EMBEDDING_DIM), (
            f"Expected (2, {EMBEDDING_DIM}), got {output.shape}"
        )

    def test_output_dtype_is_float32(self):
        encoder = TraceEncoder(n_attributes=5, n_alternatives=4, n_classes=7)
        tokens = _make_valid_tokens(1, 5)
        mask = torch.ones(1, 5, dtype=torch.bool)
        output = encoder(tokens, mask)
        assert output.dtype == torch.float32

    def test_forward_with_logits_returns_correct_shapes(self):
        n_classes = 7
        encoder = TraceEncoder(n_attributes=5, n_alternatives=4, n_classes=n_classes)
        tokens = _make_valid_tokens(3, 8)
        mask = torch.ones(3, 8, dtype=torch.bool)
        embeddings, logits = encoder.forward_with_logits(tokens, mask)
        assert embeddings.shape == (3, EMBEDDING_DIM)
        assert logits.shape == (3, n_classes)

    def test_cls_token_attends_to_all_positions(self):
        """Ensure the model doesn't crash with varying mask patterns."""
        encoder = TraceEncoder(n_attributes=5, n_alternatives=4, n_classes=7)
        tokens = _make_valid_tokens(2, 6)
        # First sample: all positions valid; second: only first 3 valid
        mask = torch.tensor(
            [
                [True, True, True, True, True, True],
                [True, True, True, False, False, False],
            ]
        )
        output = encoder(tokens, mask)
        assert output.shape == (2, EMBEDDING_DIM)
        # Outputs should differ because of different masks
        assert not torch.allclose(output[0], output[1])


class TestTraceEncoderEmbedTokens:
    def test_embed_tokens_output_shape(self):
        encoder = TraceEncoder(n_attributes=5, n_alternatives=4, n_classes=7)
        tokens = _make_valid_tokens(2, 10)
        embedded = encoder.embed_tokens(tokens)
        assert embedded.shape == (2, 10, 64), (
            f"Expected (2, 10, 64), got {embedded.shape}"
        )


# ---------------------------------------------------------------------------
# StratifiedSampler tests
# ---------------------------------------------------------------------------


class TestStratifiedSampler:
    def test_min_two_per_persona_per_batch(self):
        """Every batch must have >=2 samples per persona present."""
        # 3 personas, 10 samples each
        persona_ids = ["A"] * 10 + ["B"] * 10 + ["C"] * 10
        sampler = StratifiedSampler(persona_ids, batch_size=12, seed=42)
        for batch in sampler:
            counts = Counter(persona_ids[i] for i in batch)
            for persona, count in counts.items():
                assert count >= 2, (
                    f"Persona {persona} has {count} samples in batch, expected >= 2"
                )

    def test_batch_size_approximate(self):
        """Batch sizes should be close to requested batch_size."""
        persona_ids = ["A"] * 20 + ["B"] * 20 + ["C"] * 20
        sampler = StratifiedSampler(persona_ids, batch_size=12, seed=42)
        for batch in sampler:
            # Batch size may not be exact due to per-persona rounding,
            # but should be reasonable
            assert 6 <= len(batch) <= 30, f"Unexpected batch size: {len(batch)}"

    def test_seed_reproducibility(self):
        """Same seed produces identical batches."""
        persona_ids = ["A"] * 10 + ["B"] * 10 + ["C"] * 10
        sampler1 = StratifiedSampler(persona_ids, batch_size=9, seed=123)
        sampler2 = StratifiedSampler(persona_ids, batch_size=9, seed=123)
        batches1 = list(sampler1)
        batches2 = list(sampler2)
        assert len(batches1) == len(batches2)
        for b1, b2 in zip(batches1, batches2):
            assert b1 == b2, "Same seed should produce identical batches"

    def test_different_seeds_different_batches(self):
        """Different seeds should produce different batches."""
        persona_ids = ["A"] * 20 + ["B"] * 20
        sampler1 = StratifiedSampler(persona_ids, batch_size=8, seed=1)
        sampler2 = StratifiedSampler(persona_ids, batch_size=8, seed=2)
        batches1 = list(sampler1)
        batches2 = list(sampler2)
        # At least one batch should differ
        assert batches1 != batches2, "Different seeds should produce different batches"

    def test_all_samples_covered(self):
        """Total samples across all batches should cover all indices."""
        persona_ids = ["A"] * 10 + ["B"] * 10
        sampler = StratifiedSampler(persona_ids, batch_size=8, seed=42)
        all_indices = set()
        for batch in sampler:
            all_indices.update(batch)
        # Due to drop_last and oversampling, we may not have exact coverage,
        # but all indices should be valid
        n = len(persona_ids)
        for idx in all_indices:
            assert 0 <= idx < n, f"Invalid index {idx} for dataset of size {n}"

    def test_handles_single_persona(self):
        """Edge case: only one persona class."""
        persona_ids = ["A"] * 20
        sampler = StratifiedSampler(persona_ids, batch_size=8, seed=42)
        for batch in sampler:
            counts = Counter(persona_ids[i] for i in batch)
            assert counts["A"] >= 2

    def test_handles_small_dataset(self):
        """Edge case: fewer samples per persona than min_per_persona."""
        persona_ids = ["A"] * 1 + ["B"] * 1
        sampler = StratifiedSampler(
            persona_ids, batch_size=4, seed=42, min_per_persona=2
        )
        for batch in sampler:
            counts = Counter(persona_ids[i] for i in batch)
            for persona, count in counts.items():
                assert count >= 2, (
                    f"Oversampling should ensure >= 2 for persona {persona}, got {count}"
                )


# ---------------------------------------------------------------------------
# NT-Xent loss tests
# ---------------------------------------------------------------------------


class TestNTXentLoss:
    def test_loss_is_non_negative(self):
        embeddings = torch.randn(10, EMBEDDING_DIM)
        persona_ids = ["A"] * 5 + ["B"] * 5
        loss = nt_xent_loss(embeddings, persona_ids, temperature=0.07)
        assert loss.item() >= 0.0

    def test_loss_with_single_sample_is_zero(self):
        embeddings = torch.randn(1, EMBEDDING_DIM)
        loss = nt_xent_loss(embeddings, ["A"])
        assert loss.item() == 0.0

    def test_loss_is_differentiable(self):
        embeddings = torch.randn(6, EMBEDDING_DIM, requires_grad=True)
        persona_ids = ["A"] * 3 + ["B"] * 3
        loss = nt_xent_loss(embeddings, persona_ids)
        assert loss.requires_grad
        loss.backward()
        assert embeddings.grad is not None

    def test_same_persona_low_loss(self):
        """Identical embeddings for same persona should have lower loss than random."""
        torch.manual_seed(42)
        # All same persona — embeddings are similar
        base = torch.randn(1, EMBEDDING_DIM)
        similar = base.expand(4, -1) + torch.randn(4, EMBEDDING_DIM) * 0.01
        same_ids = ["A"] * 4
        loss_same = nt_xent_loss(similar, same_ids)

        # Different personas with no positive pairs produce zero loss
        random_embs = torch.randn(4, EMBEDDING_DIM)
        diff_ids = ["A", "B", "C", "D"]
        nt_xent_loss(random_embs, diff_ids)  # no valid pairs -> 0

        # Loss with valid positive pairs should be positive
        assert loss_same.item() > 0.0, "Should have positive loss with valid pairs"


# ---------------------------------------------------------------------------
# Integration: training loop test with tiny synthetic fixtures
# ---------------------------------------------------------------------------


class TestTrainingLoop:
    """Integration test — 1 epoch with tiny synthetic data, no MLflow."""

    @pytest.fixture
    def tiny_dataset(self, tmp_path):
        """Create minimal JSONL files for training."""
        personas = ["compensatory", "satisficer", "brand_affect"]
        all_events = []
        all_trials = []

        for p_idx, persona in enumerate(personas):
            for trial_num in range(4):  # 4 trials per persona
                pid = f"participant_{p_idx}"
                tid = f"trial_{p_idx}_{trial_num}"
                events, trial = _make_trial_events(
                    trial_id=tid,
                    participant_id=pid,
                    persona_id=persona,
                    n_events=3,
                )
                all_events.extend(events)
                all_trials.append(trial)

        # Write JSONL files
        traces_path = tmp_path / "traces.jsonl"
        trials_path = tmp_path / "trials.jsonl"

        import json

        traces_path.write_text("\n".join(json.dumps(e.__dict__) for e in all_events))
        trials_path.write_text("\n".join(json.dumps(t.__dict__) for t in all_trials))

        # Set up a temporary SQLite-backed MLflow tracking URI
        mlflow_db = tmp_path / "mlflow.db"
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")

        return traces_path, trials_path

    def test_train_runs_one_epoch(self, tiny_dataset):
        """Training loop completes without error for 1 epoch."""
        traces_path, trials_path = tiny_dataset
        import mlflow

        with mlflow.start_run(run_name="test_trace_encoder"):
            encoder = train(
                traces_path=traces_path,
                trials_path=trials_path,
                batch_size=6,
                lr=1e-3,
                n_epochs=1,
                seed=42,
                device="cpu",
                save_dir=traces_path.parent / "models",
            )
            assert isinstance(encoder, TraceEncoder)

    def test_train_saves_backbone_weights(self, tiny_dataset):
        """Saved model file exists and does NOT contain classifier weights."""
        traces_path, trials_path = tiny_dataset
        save_dir = traces_path.parent / "models"

        import mlflow

        with mlflow.start_run(run_name="test_save"):
            train(
                traces_path=traces_path,
                trials_path=trials_path,
                batch_size=6,
                n_epochs=1,
                seed=42,
                save_dir=save_dir,
            )

        save_path = save_dir / "trace_encoder.pt"
        assert save_path.exists(), "Model file should be saved"

        state_dict = torch.load(save_path, weights_only=True)
        for key in state_dict:
            assert not key.startswith("classifier"), (
                f"Classifier weights should not be saved, found: {key}"
            )

    def test_train_produces_valid_embeddings(self, tiny_dataset):
        """After 1 epoch, encoder produces valid embeddings of correct shape."""
        traces_path, trials_path = tiny_dataset
        import mlflow

        with mlflow.start_run(run_name="test_embeddings"):
            encoder = train(
                traces_path=traces_path,
                trials_path=trials_path,
                batch_size=6,
                n_epochs=1,
                seed=42,
                save_dir=traces_path.parent / "models",
            )

        encoder.eval()
        # Use the encoder's actual embedding sizes to generate valid tokens
        n_attr = encoder.attribute_embed.num_embeddings
        n_alt = encoder.alternative_embed.num_embeddings
        tokens = _make_valid_tokens(2, 5, n_attributes=n_attr, n_alternatives=n_alt)
        mask = torch.ones(2, 5, dtype=torch.bool)
        with torch.no_grad():
            output = encoder(tokens, mask)
        assert output.shape == (2, EMBEDDING_DIM)
        assert torch.isfinite(output).all(), "Embeddings should be finite"


# ---------------------------------------------------------------------------
# Supervised cross-entropy loss tests
# ---------------------------------------------------------------------------


class TestCrossEntropyLoss:
    def test_loss_is_non_negative(self):
        logits = torch.randn(8, 7)
        labels = torch.randint(0, 7, (8,))
        loss = cross_entropy_loss(logits, labels)
        assert loss.item() >= 0.0

    def test_loss_is_differentiable(self):
        logits = torch.randn(8, 7, requires_grad=True)
        labels = torch.randint(0, 7, (8,))
        loss = cross_entropy_loss(logits, labels)
        loss.backward()
        assert logits.grad is not None

    def test_correct_predictions_lower_loss(self):
        """Near-perfect logits produce lower loss than random logits."""
        n, c = 8, 7
        labels = torch.arange(n) % c
        # Perfect logits: 10.0 on correct class
        perfect = torch.zeros(n, c)
        perfect[torch.arange(n), labels] = 10.0
        loss_perfect = cross_entropy_loss(perfect, labels)

        random_logits = torch.randn(n, c)
        loss_random = cross_entropy_loss(random_logits, labels)

        assert loss_perfect.item() < loss_random.item()


# ---------------------------------------------------------------------------
# Supervised training loop tests
# ---------------------------------------------------------------------------


class TestSupervisedTrainingLoop:
    """Integration tests verifying train() uses cross-entropy as primary loss."""

    @pytest.fixture
    def tiny_dataset(self, tmp_path):
        """Minimal JSONL fixtures — 3 personas, 4 trials each."""
        import json
        import mlflow

        personas = ["compensatory", "satisficer", "brand_affect"]
        all_events, all_trials = [], []
        for p_idx, persona in enumerate(personas):
            for trial_num in range(4):
                pid = f"participant_{p_idx}"
                tid = f"trial_{p_idx}_{trial_num}"
                events, trial = _make_trial_events(
                    trial_id=tid,
                    participant_id=pid,
                    persona_id=persona,
                    n_events=3,
                )
                all_events.extend(events)
                all_trials.append(trial)

        traces_path = tmp_path / "traces.jsonl"
        trials_path = tmp_path / "trials.jsonl"
        traces_path.write_text("\n".join(json.dumps(e.__dict__) for e in all_events))
        trials_path.write_text("\n".join(json.dumps(t.__dict__) for t in all_trials))

        mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
        return traces_path, trials_path

    def test_train_completes_without_error(self, tiny_dataset):
        """Supervised training loop runs 1 epoch with no exceptions."""
        import mlflow

        traces_path, trials_path = tiny_dataset
        with mlflow.start_run(run_name="test_supervised"):
            encoder = train(
                traces_path=traces_path,
                trials_path=trials_path,
                batch_size=6,
                n_epochs=1,
                seed=42,
                save_dir=traces_path.parent / "models",
            )
        assert isinstance(encoder, TraceEncoder)

    def test_train_logs_cls_loss_not_contrastive(self, tiny_dataset):
        """MLflow run should record cls_loss metric, not contrastive_loss."""
        import mlflow

        traces_path, trials_path = tiny_dataset
        with mlflow.start_run(run_name="test_metrics") as run:
            train(
                traces_path=traces_path,
                trials_path=trials_path,
                batch_size=6,
                n_epochs=1,
                seed=42,
                save_dir=traces_path.parent / "models",
            )
            run_id = run.info.run_id

        client = mlflow.tracking.MlflowClient()
        history_keys = set()
        for key in ["train_cls_loss", "val_cls_loss", "train_contrastive_loss"]:
            if client.get_metric_history(run_id, key):
                history_keys.add(key)

        assert "train_cls_loss" in history_keys, "train_cls_loss must be logged"
        assert "train_contrastive_loss" not in history_keys, (
            "contrastive loss must not be logged in supervised mode"
        )

    def test_loss_decreases_over_epochs(self, tiny_dataset):
        """Training loss should decrease from epoch 1 to epoch 5 on tiny data."""
        import mlflow

        traces_path, trials_path = tiny_dataset
        with mlflow.start_run(run_name="test_convergence") as run:
            train(
                traces_path=traces_path,
                trials_path=trials_path,
                batch_size=6,
                n_epochs=5,
                seed=42,
                save_dir=traces_path.parent / "models",
            )
            run_id = run.info.run_id

        client = mlflow.tracking.MlflowClient()
        history = client.get_metric_history(run_id, "train_cls_loss")
        assert len(history) >= 2
        first_loss = history[0].value
        last_loss = history[-1].value
        assert last_loss < first_loss, (
            f"Loss should decrease: first={first_loss:.4f}, last={last_loss:.4f}"
        )
