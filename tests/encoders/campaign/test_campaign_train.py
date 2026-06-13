"""
Tests for encoders/campaign/ — model forward pass and training utilities.

All tests use tiny synthetic fixtures generated in-process; they never
depend on data/synthetic/ files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from schemas import EMBEDDING_DIM
from schemas.campaign import CampaignEvent, CampaignType

from encoders.campaign.features import (
    MAX_EVENTS,
    TOKEN_DIM,
    CampaignVocabulary,
)
from encoders.campaign.model import CampaignEncoder
from encoders.campaign.train import (
    CampaignDataset,
    DEFAULT_LAMBDA_CONTRASTIVE,
    DEFAULT_TEMPERATURE,
    StratifiedSampler,
    collate_fn,
    cross_entropy_loss,
    nt_xent_views,
    train,
)


# ---------------------------------------------------------------------------
# Shared fixtures — tiny synthetic campaign data
# ---------------------------------------------------------------------------


def _make_event(
    participant_id: str = "p1",
    campaign_id: str = "c1",
    campaign_type: CampaignType = CampaignType.PROMOTION,
    discount_pct: float = 0.1,
    opened: bool = True,
    clicked: bool = False,
    converted: bool = False,
    unsub: bool = False,
    sent_ts: str = "2025-01-01T00:00:00Z",
    month: int = 1,
    customer_id: str = "price_lex",
    category: str = "electronics",
) -> CampaignEvent:
    return CampaignEvent(
        customer_id=customer_id,
        campaign_id=campaign_id,
        sent_ts=sent_ts,
        campaign_type=campaign_type,
        discount_pct=discount_pct,
        category=category,
        opened=opened,
        clicked=clicked,
        converted=converted,
        unsub=unsub,
        month=month,
        participant_id=participant_id,
    )


def _make_customer_events(
    participant_id: str = "p1",
    n_events: int = 5,
    start_month: int = 1,
) -> list[CampaignEvent]:
    """Build a chronologically-ordered campaign history for one customer."""
    types = list(CampaignType)
    events: list[CampaignEvent] = []
    for i in range(n_events):
        events.append(
            _make_event(
                participant_id=participant_id,
                campaign_id=f"c_{participant_id}_{i}",
                campaign_type=types[i % len(types)],
                discount_pct=round(0.05 * (i % 6), 2),
                opened=(i % 2 == 0),
                clicked=(i % 3 == 0),
                converted=(i % 4 == 0),
                sent_ts=f"2025-{start_month + i:02d}-01T00:00:00Z",
                month=start_month + i,
            )
        )
    return events


def _make_valid_tokens(batch_size: int, seq_len: int) -> torch.Tensor:
    """Random float tokens of shape (B, S, TOKEN_DIM) including CLS at 0."""
    return torch.rand(batch_size, seq_len, TOKEN_DIM)


# ---------------------------------------------------------------------------
# Model forward pass tests
# ---------------------------------------------------------------------------


class TestCampaignEncoderForward:
    def test_output_shape_is_embedding_dim(self):
        encoder = CampaignEncoder()
        tokens = _make_valid_tokens(2, 10)
        mask = torch.ones(2, 10, dtype=torch.bool)
        output = encoder(tokens, mask)
        assert output.shape == (2, EMBEDDING_DIM), (
            f"Expected (2, {EMBEDDING_DIM}), got {output.shape}"
        )

    def test_output_dtype_is_float32(self):
        encoder = CampaignEncoder()
        tokens = _make_valid_tokens(1, 5)
        mask = torch.ones(1, 5, dtype=torch.bool)
        output = encoder(tokens, mask)
        assert output.dtype == torch.float32

    def test_forward_with_logits_returns_correct_shapes(self):
        n_classes = 7
        encoder = CampaignEncoder()
        tokens = _make_valid_tokens(3, 8)
        mask = torch.ones(3, 8, dtype=torch.bool)
        embeddings, logits = encoder.forward_with_logits(tokens, mask)
        assert embeddings.shape == (3, EMBEDDING_DIM)
        assert logits.shape == (3, n_classes)

    def test_varying_masks_produce_different_outputs(self):
        encoder = CampaignEncoder()
        tokens = _make_valid_tokens(2, 6)
        mask = torch.tensor(
            [
                [True, True, True, True, True, True],
                [True, True, True, False, False, False],
            ]
        )
        output = encoder(tokens, mask)
        assert output.shape == (2, EMBEDDING_DIM)
        assert not torch.allclose(output[0], output[1])

    def test_embeddings_are_finite(self):
        encoder = CampaignEncoder()
        tokens = _make_valid_tokens(4, 12)
        mask = torch.ones(4, 12, dtype=torch.bool)
        output = encoder(tokens, mask)
        assert torch.isfinite(output).all()


# ---------------------------------------------------------------------------
# Tokenisation contract tests
# ---------------------------------------------------------------------------


class TestTokenisation:
    def test_encode_sequence_truncates_to_max_events(self):
        vocab = CampaignVocabulary()
        events = _make_customer_events(n_events=MAX_EVENTS + 20)
        seq = vocab.encode_sequence(events)
        assert seq.size(0) == MAX_EVENTS, (
            f"Expected truncation to {MAX_EVENTS}, got {seq.size(0)}"
        )

    def test_encode_sequence_most_recent_last(self):
        """The tail of the returned sequence is the most recent event."""
        vocab = CampaignVocabulary()
        events = _make_customer_events(n_events=3)
        seq = vocab.encode_sequence(events)
        # Last token corresponds to the last event; its discount column
        # (index 5) should match the last event's discount.
        last_event = events[-1]
        expected_discount = last_event.discount_pct
        assert torch.isclose(seq[-1, 5], torch.tensor(expected_discount))


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------


class TestLossFunctions:
    def test_cross_entropy_non_negative(self):
        logits = torch.randn(8, 7)
        labels = torch.randint(0, 7, (8,))
        loss = cross_entropy_loss(logits, labels)
        assert loss.item() >= 0.0

    def test_cross_entropy_differentiable(self):
        logits = torch.randn(8, 7, requires_grad=True)
        labels = torch.randint(0, 7, (8,))
        loss = cross_entropy_loss(logits, labels)
        loss.backward()
        assert logits.grad is not None

    def test_nt_xent_views_non_negative(self):
        v1 = torch.randn(6, EMBEDDING_DIM)
        v2 = torch.randn(6, EMBEDDING_DIM)
        loss = nt_xent_views(v1, v2, DEFAULT_TEMPERATURE)
        assert loss.item() >= 0.0

    def test_nt_xent_views_single_pair_is_zero(self):
        v1 = torch.randn(1, EMBEDDING_DIM)
        v2 = torch.randn(1, EMBEDDING_DIM)
        loss = nt_xent_views(v1, v2, DEFAULT_TEMPERATURE)
        assert loss.item() == 0.0

    def test_nt_xent_views_differentiable(self):
        v1 = torch.randn(6, EMBEDDING_DIM, requires_grad=True)
        v2 = torch.randn(6, EMBEDDING_DIM, requires_grad=True)
        loss = nt_xent_views(v1, v2)
        loss.backward()
        assert v1.grad is not None
        assert v2.grad is not None


# ---------------------------------------------------------------------------
# StratifiedSampler tests
# ---------------------------------------------------------------------------


class TestStratifiedSampler:
    def test_min_two_per_label_per_batch(self):
        from collections import Counter

        labels = [0] * 10 + [1] * 10 + [2] * 10
        sampler = StratifiedSampler(labels, batch_size=12, seed=42)
        for batch in sampler:
            counts = Counter(labels[i] for i in batch)
            for lbl, count in counts.items():
                assert count >= 2, f"Label {lbl} has {count} samples, expected >= 2"

    def test_seed_reproducibility(self):
        labels = [0] * 10 + [1] * 10
        s1 = StratifiedSampler(labels, batch_size=8, seed=42)
        s2 = StratifiedSampler(labels, batch_size=8, seed=42)
        assert list(s1) == list(s2)

    def test_all_indices_valid(self):
        labels = [0] * 10 + [1] * 10
        sampler = StratifiedSampler(labels, batch_size=8, seed=42)
        n = len(labels)
        for batch in sampler:
            for idx in batch:
                assert 0 <= idx < n


# ---------------------------------------------------------------------------
# Collate tests
# ---------------------------------------------------------------------------


class TestCollate:
    def test_pads_to_max_length(self):
        s1 = torch.rand(4, TOKEN_DIM)
        s2 = torch.rand(6, TOKEN_DIM)
        ds = CampaignDataset([s1, s2], [0, 1], ["p1", "p2"])
        tokens, mask, labels, pids = collate_fn([ds[0], ds[1]])
        assert tokens.shape == (2, 6, TOKEN_DIM)
        assert mask.shape == (2, 6)
        assert mask[0].sum().item() == 4
        assert mask[1].sum().item() == 6

    def test_padding_positions_are_zero(self):
        s1 = torch.rand(3, TOKEN_DIM)
        s2 = torch.rand(5, TOKEN_DIM)
        ds = CampaignDataset([s1, s2], [0, 1], ["p1", "p2"])
        tokens, mask, _, _ = collate_fn([ds[0], ds[1]])
        # Padded tail of the shorter sequence must be zero
        assert torch.all(tokens[0, 3:] == 0.0)


# ---------------------------------------------------------------------------
# Integration: training loop test with tiny synthetic data
# ---------------------------------------------------------------------------


def _write_tiny_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Write tiny campaigns.jsonl + psychographics.jsonl fixtures.

    Generates enough participants per persona (10 each) that the 80/20 split
    leaves >=2 per persona in val — required for strategy_recovery's
    cross-validation.
    """
    personas = ["price_lex", "quality_lex", "compensatory", "satisficer"]
    all_events: list[CampaignEvent] = []
    psych_rows: list[dict[str, str]] = []
    for persona_idx, persona in enumerate(personas):
        for copy in range(10):  # 10 participants per persona = 40 total
            pid = f"participant_{persona_idx}_{copy}"
            events = _make_customer_events(participant_id=pid, n_events=8)
            all_events.extend(events)
            psych_rows.append({"participant_id": pid, "persona_id": persona})

    campaigns_path = tmp_path / "campaigns.jsonl"
    psych_path = tmp_path / "psychographics.jsonl"
    campaigns_path.write_text("\n".join(json.dumps(ev.__dict__) for ev in all_events))
    psych_path.write_text("\n".join(json.dumps(r) for r in psych_rows))
    return campaigns_path, psych_path


class TestTrainingLoop:
    @pytest.fixture(autouse=True)
    def _temp_mlflow(self, tmp_path):
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
        yield

    def test_train_runs_one_epoch(self, tmp_path):
        import mlflow

        campaigns_path, psych_path = _write_tiny_dataset(tmp_path)
        save_path = tmp_path / "models" / "campaign_encoder.pt"
        with mlflow.start_run(run_name="test_campaign"):
            encoder = train(
                campaigns_path=campaigns_path,
                psychographics_path=psych_path,
                batch_size=8,
                n_epochs=1,
                seed=42,
                device="cpu",
                save_path=save_path,
            )
        assert isinstance(encoder, CampaignEncoder)

    def test_train_saves_backbone_without_classifier(self, tmp_path):
        import mlflow

        campaigns_path, psych_path = _write_tiny_dataset(tmp_path)
        save_path = tmp_path / "models" / "campaign_encoder.pt"
        with mlflow.start_run(run_name="test_save"):
            train(
                campaigns_path=campaigns_path,
                psychographics_path=psych_path,
                batch_size=8,
                n_epochs=1,
                seed=42,
                save_path=save_path,
            )
        assert save_path.exists()
        state_dict = torch.load(save_path, weights_only=True)
        for key in state_dict:
            assert not key.startswith("classifier"), (
                f"Classifier weights must not be saved, found: {key}"
            )

    def test_checkpoint_loads_strict_false(self, tmp_path):
        """Checkpoint must load via CampaignEncoder().load_state_dict(strict=False)."""
        import mlflow

        campaigns_path, psych_path = _write_tiny_dataset(tmp_path)
        save_path = tmp_path / "models" / "campaign_encoder.pt"
        with mlflow.start_run(run_name="test_load"):
            train(
                campaigns_path=campaigns_path,
                psychographics_path=psych_path,
                batch_size=8,
                n_epochs=1,
                seed=42,
                save_path=save_path,
            )
        encoder = CampaignEncoder()
        state_dict = torch.load(save_path, weights_only=True)
        missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
        # classifier.* are expected missing (backbone-only checkpoint)
        assert all("classifier" in m for m in missing), (
            f"Unexpected missing keys: {missing}"
        )
        assert unexpected == [], f"Unexpected keys: {unexpected}"

    def test_loss_decreases_over_epochs(self, tmp_path):
        """Train CE should strictly decrease from epoch 1 to epoch 6."""
        import mlflow

        campaigns_path, psych_path = _write_tiny_dataset(tmp_path)
        save_path = tmp_path / "models" / "campaign_encoder.pt"
        with mlflow.start_run(run_name="test_decrease") as run:
            train(
                campaigns_path=campaigns_path,
                psychographics_path=psych_path,
                batch_size=8,
                n_epochs=6,
                seed=42,
                save_path=save_path,
            )
            run_id = run.info.run_id

        client = mlflow.tracking.MlflowClient()
        history = client.get_metric_history(run_id, "train_cls_loss")
        assert len(history) >= 2
        first = history[0].value
        last = history[-1].value
        assert last < first, f"CE should decrease: first={first:.4f}, last={last:.4f}"

    def test_default_hyperparameters_match_spec(self):
        """Pinned hyperparameters must match the trace encoder contract."""
        assert DEFAULT_LAMBDA_CONTRASTIVE == 0.5
        assert DEFAULT_TEMPERATURE == 0.07
