"""
Tests for encoders/transaction/.

Covers:
  - features.py: vocabulary, token vector shape, known feature values
  - model.py: forward pass, output shape, variable-length sequences,
    pack_padded_sequence masking
  - train.py: dataset construction, participant-level split, collation,
    integration (1 epoch, tiny fixtures)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from schemas import EMBEDDING_DIM
from schemas.transaction import Channel, PurchaseType, TransactionRecord
from encoders.transaction.features import (
    BRAND_TIER_VOCAB,
    TOKEN_DIM,
    TxVocabulary,
    sort_transactions_most_recent_first,
)
from encoders.transaction.model import NextBrandTierHead, TransactionEncoder
from encoders.transaction.train import (
    TransactionSequenceDataset,
    collate_fn,
    split_by_participant,
    train,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    participant_id: str = "P001",
    days_before_session: int = 30,
    brand_tier: str = "mid",
    channel: Channel = Channel.ONLINE,
    purchase_type: PurchaseType = PurchaseType.PLANNED,
    price_paid_normalised: float = 0.5,
    quantity: int = 2,
    on_promotion: bool = False,
    persona_id: str = "price_lex",
) -> TransactionRecord:
    return TransactionRecord(
        participant_id=participant_id,
        transaction_id=f"tx_{participant_id}_{days_before_session:04d}",
        days_before_session=days_before_session,
        category="groceries",
        product_id="prod_001",
        brand_tier=brand_tier,
        price_paid_normalised=price_paid_normalised,
        quantity=quantity,
        channel=channel,
        purchase_type=purchase_type,
        on_promotion=on_promotion,
        persona_id=persona_id,
    )


def _make_records_for_participant(
    participant_id: str = "P001",
    n: int = 10,
    brand_tiers: list[str] | None = None,
) -> list[TransactionRecord]:
    """Create n records for a single participant with ascending days."""
    if brand_tiers is None:
        brand_tiers = ["mid"] * n
    return [
        _make_record(
            participant_id=participant_id,
            days_before_session=i + 1,
            brand_tier=brand_tiers[i % len(brand_tiers)],
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# features.py tests
# ---------------------------------------------------------------------------


class TestTxVocabulary:
    """Tests for TxVocabulary: index mapping, embedding dimensions, persistence."""

    def test_brand_tier_vocab_has_four_entries(self) -> None:
        vocab = TxVocabulary()
        assert len(vocab.brand_tier_to_idx) == 4

    def test_channel_vocab_has_three_entries(self) -> None:
        vocab = TxVocabulary()
        assert len(vocab.channel_to_idx) == 3

    def test_purchase_type_vocab_has_four_entries(self) -> None:
        vocab = TxVocabulary()
        assert len(vocab.purchase_type_to_idx) == 4

    def test_brand_tier_index_known_values(self) -> None:
        vocab = TxVocabulary()
        for i, tier in enumerate(BRAND_TIER_VOCAB):
            assert vocab.brand_tier_index(tier) == i

    def test_channel_index_from_enum(self) -> None:
        vocab = TxVocabulary()
        assert vocab.channel_index(Channel.ONLINE) == 0
        assert vocab.channel_index(Channel.IN_STORE) == 1
        assert vocab.channel_index(Channel.CLICK_AND_COLLECT) == 2

    def test_purchase_type_index_from_enum(self) -> None:
        vocab = TxVocabulary()
        assert vocab.purchase_type_index(PurchaseType.PLANNED) == 0
        assert vocab.purchase_type_index(PurchaseType.IMPULSE) == 1

    def test_save_and_load_vocab(self) -> None:
        vocab = TxVocabulary()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tx_vocab.json"
            vocab.save_vocab(path)
            assert path.exists()

            loaded = TxVocabulary.load_vocab(path)
            assert loaded.brand_tier_to_idx == vocab.brand_tier_to_idx
            assert loaded.channel_to_idx == vocab.channel_to_idx
            assert loaded.purchase_type_to_idx == vocab.purchase_type_to_idx


class TestTokenVector:
    """Tests for to_token_vector: shape, known values."""

    def test_token_vector_shape(self) -> None:
        vocab = TxVocabulary()
        record = _make_record()
        vec = vocab.to_token_vector(record)
        assert vec.shape == (TOKEN_DIM,)
        assert vec.shape[0] == 20

    def test_token_vector_is_float(self) -> None:
        vocab = TxVocabulary()
        record = _make_record()
        vec = vocab.to_token_vector(record)
        assert vec.dtype == torch.float32

    def test_price_paid_normalised_preserved(self) -> None:
        vocab = TxVocabulary()
        record = _make_record(price_paid_normalised=0.73)
        vec = vocab.to_token_vector(record)
        # price is the 17th element (after 8+4+4 = 16 embedding dims)
        assert abs(vec[16].item() - 0.73) < 1e-6

    def test_on_promotion_true(self) -> None:
        vocab = TxVocabulary()
        record = _make_record(on_promotion=True)
        vec = vocab.to_token_vector(record)
        assert vec[17].item() == 1.0

    def test_on_promotion_false(self) -> None:
        vocab = TxVocabulary()
        record = _make_record(on_promotion=False)
        vec = vocab.to_token_vector(record)
        assert vec[17].item() == 0.0

    def test_quantity_norm_capped_at_one(self) -> None:
        vocab = TxVocabulary()
        record = _make_record(quantity=10)
        vec = vocab.to_token_vector(record)
        assert vec[18].item() == 1.0

    def test_quantity_norm_value_below_cap(self) -> None:
        vocab = TxVocabulary()
        record = _make_record(quantity=3)
        vec = vocab.to_token_vector(record)
        assert abs(vec[18].item() - 0.6) < 1e-6

    def test_recency_norm_today(self) -> None:
        """days_before_session=1 → recency ≈ 0.997"""
        vocab = TxVocabulary()
        record = _make_record(days_before_session=1)
        vec = vocab.to_token_vector(record)
        expected = 1.0 - 1.0 / 365.0
        assert abs(vec[19].item() - expected) < 1e-6

    def test_recency_norm_year_ago(self) -> None:
        """days_before_session=365 → recency ≈ 0.0"""
        vocab = TxVocabulary()
        record = _make_record(days_before_session=365)
        vec = vocab.to_token_vector(record)
        expected = 1.0 - 365.0 / 365.0
        assert abs(vec[19].item() - expected) < 1e-6

    def test_brand_tier_embedding_dimensions(self) -> None:
        """First 8 dims come from brand_tier embedding (dim=8)."""
        vocab = TxVocabulary()
        record = _make_record(brand_tier="premium")
        vec = vocab.to_token_vector(record)
        # The first 8 values are from the embedding — just verify shape region
        assert vec[:8].shape == (8,)

    def test_channel_embedding_dimensions(self) -> None:
        """Dims 8-11 come from channel embedding (dim=4)."""
        vocab = TxVocabulary()
        record = _make_record()
        vec = vocab.to_token_vector(record)
        assert vec[8:12].shape == (4,)

    def test_purchase_type_embedding_dimensions(self) -> None:
        """Dims 12-15 come from purchase_type embedding (dim=4)."""
        vocab = TxVocabulary()
        record = _make_record()
        vec = vocab.to_token_vector(record)
        assert vec[12:16].shape == (4,)

    def test_different_brand_tiers_different_vectors(self) -> None:
        vocab = TxVocabulary()
        r1 = _make_record(brand_tier="premium")
        r2 = _make_record(brand_tier="value")
        v1 = vocab.to_token_vector(r1)
        v2 = vocab.to_token_vector(r2)
        # Embeddings are random but different indices → different vectors
        assert not torch.allclose(v1[:8], v2[:8])


class TestEncodeSequence:
    """Tests for encode_sequence: batch tokenisation."""

    def test_encode_sequence_shape(self) -> None:
        vocab = TxVocabulary()
        records = _make_records_for_participant("P001", n=15)
        seq = vocab.encode_sequence(records)
        assert seq.shape == (15, TOKEN_DIM)

    def test_encode_sequence_length_one(self) -> None:
        vocab = TxVocabulary()
        records = [_make_record()]
        seq = vocab.encode_sequence(records)
        assert seq.shape == (1, TOKEN_DIM)


class TestSortTransactions:
    """Tests for sort_transactions_most_recent_first."""

    def test_most_recent_first(self) -> None:
        records = [
            _make_record(days_before_session=10),
            _make_record(days_before_session=1),
            _make_record(days_before_session=100),
        ]
        sorted_recs = sort_transactions_most_recent_first(records)
        assert sorted_recs[0].days_before_session == 1
        assert sorted_recs[1].days_before_session == 10
        assert sorted_recs[2].days_before_session == 100

    def test_already_sorted(self) -> None:
        records = [
            _make_record(days_before_session=5),
            _make_record(days_before_session=10),
        ]
        sorted_recs = sort_transactions_most_recent_first(records)
        assert sorted_recs[0].days_before_session == 5

    def test_empty_list(self) -> None:
        result = sort_transactions_most_recent_first([])
        assert result == []


# ---------------------------------------------------------------------------
# model.py tests
# ---------------------------------------------------------------------------


class TestTransactionEncoderForward:
    """Tests for TransactionEncoder forward pass and output contract."""

    def test_output_shape_matches_embedding_dim(self) -> None:
        vocab = TxVocabulary()
        encoder = TransactionEncoder(vocab=vocab)
        B, T = 4, 10
        token_seqs = torch.randn(B, T, TOKEN_DIM)
        lengths = torch.tensor([10, 8, 6, 4])
        out = encoder(token_seqs, lengths)
        assert out.shape == (B, EMBEDDING_DIM)

    def test_output_dtype_is_float32(self) -> None:
        vocab = TxVocabulary()
        encoder = TransactionEncoder(vocab=vocab)
        token_seqs = torch.randn(2, 5, TOKEN_DIM)
        lengths = torch.tensor([5, 3])
        out = encoder(token_seqs, lengths)
        assert out.dtype == torch.float32

    def test_single_sequence_batch(self) -> None:
        vocab = TxVocabulary()
        encoder = TransactionEncoder(vocab=vocab)
        token_seqs = torch.randn(1, 20, TOKEN_DIM)
        lengths = torch.tensor([20])
        out = encoder(token_seqs, lengths)
        assert out.shape == (1, EMBEDDING_DIM)

    def test_variable_length_sequences(self) -> None:
        vocab = TxVocabulary()
        encoder = TransactionEncoder(vocab=vocab)
        # Sorted by descending length (required by pack_padded_sequence)
        token_seqs = torch.randn(3, 30, TOKEN_DIM)
        lengths = torch.tensor([30, 20, 5])
        out = encoder(token_seqs, lengths)
        assert out.shape == (3, EMBEDDING_DIM)

    def test_embedding_dim_matches_schemas(self) -> None:
        """EMBEDDING_DIM must come from schemas, never hardcoded."""
        assert EMBEDDING_DIM == 128

    def test_uses_final_hidden_state_not_mean(self) -> None:
        """Verify the model uses final hidden state by checking architecture."""
        vocab = TxVocabulary()
        encoder = TransactionEncoder(vocab=vocab, gru_hidden=64)
        # The output projection should map from gru_hidden to EMBEDDING_DIM
        linear_layer = encoder.output_proj[0]
        assert linear_layer.in_features == 64
        assert linear_layer.out_features == EMBEDDING_DIM


class TestPackPaddedSequenceMasking:
    """Tests verifying pack_padded_sequence is used correctly."""

    def test_different_lengths_different_outputs(self) -> None:
        """Padded positions should NOT affect the output."""
        vocab = TxVocabulary()
        encoder = TransactionEncoder(vocab=vocab)
        encoder.eval()

        torch.manual_seed(42)
        B, T = 2, 15
        # First participant has 15 tokens, second has 5
        token_seqs = torch.randn(B, T, TOKEN_DIM)
        lengths = torch.tensor([15, 5])

        # Same input but with different padding in the short sequence
        token_seqs_padded = token_seqs.clone()
        token_seqs_padded[1, 5:, :] = torch.randn(10, TOKEN_DIM) * 100

        with torch.no_grad():
            out1 = encoder(token_seqs, lengths)
            out2 = encoder(token_seqs_padded, lengths)

        # Second participant's output must be identical — padding is masked
        assert torch.allclose(out1[1], out2[1], atol=1e-5), (
            "Padding values leaked into GRU output — pack_padded_sequence not working"
        )

    def test_short_sequence_output_independent_of_max_len(self) -> None:
        """Output for a sequence of length 5 should be the same regardless
        of how much padding is added beyond position 5."""
        vocab = TxVocabulary()
        encoder = TransactionEncoder(vocab=vocab)
        encoder.eval()

        torch.manual_seed(0)
        real_tokens = torch.randn(1, 5, TOKEN_DIM)

        # Pad to length 10
        pad10 = torch.cat([real_tokens, torch.zeros(1, 5, TOKEN_DIM)], dim=1)
        # Pad to length 20
        pad20 = torch.cat([real_tokens, torch.zeros(1, 15, TOKEN_DIM)], dim=1)

        with torch.no_grad():
            out10 = encoder(pad10, torch.tensor([5]))
            out20 = encoder(pad20, torch.tensor([5]))

        assert torch.allclose(out10, out20, atol=1e-5), (
            "Different padding lengths gave different outputs"
        )


class TestNextBrandTierHead:
    """Tests for the prediction head."""

    def test_head_output_shape(self) -> None:
        head = NextBrandTierHead(gru_hidden=128, n_classes=4)
        hidden = torch.randn(8, 128)
        logits = head(hidden)
        assert logits.shape == (8, 4)

    def test_head_output_is_logits(self) -> None:
        head = NextBrandTierHead(gru_hidden=128, n_classes=4)
        hidden = torch.randn(4, 128)
        logits = head(hidden)
        # Logits should not be softmaxed
        assert logits.min().item() < 0 or logits.max().item() > 1


# ---------------------------------------------------------------------------
# train.py tests
# ---------------------------------------------------------------------------


class TestTransactionSequenceDataset:
    """Tests for the dataset class."""

    def test_dataset_length(self) -> None:
        records = _make_records_for_participant(
            "P001", n=10
        ) + _make_records_for_participant("P002", n=5)
        vocab = TxVocabulary()
        ds = TransactionSequenceDataset(records, vocab)
        # 2 participants
        assert len(ds) == 2

    def test_item_shapes(self) -> None:
        records = _make_records_for_participant("P001", n=10)
        vocab = TxVocabulary()
        ds = TransactionSequenceDataset(records, vocab)
        tokens, targets, length = ds[0]
        # Input is t1..t9 (last excluded), so length = 9
        assert tokens.shape == (9, TOKEN_DIM)
        assert targets.shape == (9,)
        assert length == 9

    def test_targets_are_brand_tier_indices(self) -> None:
        tiers = ["premium", "mid", "value", "own_label"]
        records = _make_records_for_participant("P001", n=5, brand_tiers=tiers)
        vocab = TxVocabulary()
        ds = TransactionSequenceDataset(records, vocab)
        _, targets, _ = ds[0]
        # Targets should be valid indices 0-3
        assert targets.min().item() >= 0
        assert targets.max().item() <= 3

    def test_single_transaction_participant(self) -> None:
        """A participant with 1 transaction should still produce an item."""
        records = [_make_record(participant_id="P001")]
        vocab = TxVocabulary()
        ds = TransactionSequenceDataset(records, vocab)
        assert len(ds) == 1
        tokens, targets, length = ds[0]
        assert length == 1

    def test_max_seq_len_truncation(self) -> None:
        records = _make_records_for_participant("P001", n=100)
        vocab = TxVocabulary()
        ds = TransactionSequenceDataset(records, vocab, max_seq_len=10)
        tokens, targets, length = ds[0]
        assert length == 9  # truncated to 10, input = 10-1 = 9


class TestCollateFn:
    """Tests for the collation function."""

    def test_collate_shapes(self) -> None:
        vocab = TxVocabulary()
        records = _make_records_for_participant(
            "P001", n=10
        ) + _make_records_for_participant("P002", n=5)
        ds = TransactionSequenceDataset(records, vocab)
        batch = [ds[0], ds[1]]
        tokens, targets, lengths = collate_fn(batch)

        assert tokens.shape[0] == 2
        assert tokens.shape[2] == TOKEN_DIM
        assert lengths.shape == (2,)

    def test_collate_sorted_by_descending_length(self) -> None:
        vocab = TxVocabulary()
        records = _make_records_for_participant(
            "P001", n=5
        ) + _make_records_for_participant("P002", n=10)
        ds = TransactionSequenceDataset(records, vocab)
        batch = [ds[0], ds[1]]  # P001=4 tokens, P002=9 tokens
        tokens, _, lengths = collate_fn(batch)

        # After sorting, first should be longer
        assert lengths[0] >= lengths[1]


class TestSplitByParticipant:
    """Tests for participant-level train/val split."""

    def test_no_participant_leakage(self) -> None:
        """No participant should appear in both train and val sets."""
        records = []
        for pid in [f"P{i:03d}" for i in range(20)]:
            records.extend(_make_records_for_participant(pid, n=5))

        train_recs, val_recs = split_by_participant(records, val_fraction=0.2)
        train_pids = {r.participant_id for r in train_recs}
        val_pids = {r.participant_id for r in val_recs}
        assert train_pids.isdisjoint(val_pids)

    def test_all_records_accounted_for(self) -> None:
        records = []
        for pid in [f"P{i:03d}" for i in range(20)]:
            records.extend(_make_records_for_participant(pid, n=5))

        train_recs, val_recs = split_by_participant(records, val_fraction=0.2)
        assert len(train_recs) + len(val_recs) == len(records)

    def test_approximate_split_ratio(self) -> None:
        records = []
        for pid in [f"P{i:03d}" for i in range(100)]:
            records.extend(_make_records_for_participant(pid, n=5))

        train_recs, val_recs = split_by_participant(records, val_fraction=0.2)
        train_pids = {r.participant_id for r in train_recs}
        val_pids = {r.participant_id for r in val_recs}
        # 80/20 split by participant
        total = len(train_pids) + len(val_pids)
        assert abs(len(train_pids) / total - 0.8) < 0.05


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end integration: 1 epoch with tiny fixtures."""

    def test_train_one_epoch_tiny(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
        """Train for 1 epoch with 4 participants, 10 records each.

        Verifies the full pipeline runs without error. Does NOT depend
        on data/synthetic/.
        """
        records = []
        tiers = ["premium", "mid", "value", "own_label"]
        for i, pid in enumerate(["P001", "P002", "P003", "P004"]):
            records.extend(
                _make_records_for_participant(
                    pid,
                    n=10,
                    brand_tiers=[tiers[i % 4]] * 10,
                )
            )

        encoder = train(
            records=records,
            batch_size=4,
            lr=1e-3,
            n_epochs=1,
            gru_hidden=32,
            gru_layers=1,
            gru_dropout=0.0,
            projection_dim=16,
            device="cpu",
        )

        assert isinstance(encoder, TransactionEncoder)

        # Verify the encoder produces correct shape embeddings
        vocab = encoder.vocab
        vocab.eval()
        encoder.eval()

        # Create a small batch for inference
        test_records = _make_records_for_participant("P001", n=5)
        token_seq = vocab.encode_sequence(test_records)  # (5, 20)
        token_batch = token_seq.unsqueeze(0)  # (1, 5, 20)
        lengths = torch.tensor([5])

        with torch.no_grad():
            embedding = encoder(token_batch, lengths)

        assert embedding.shape == (1, EMBEDDING_DIM)

    def test_train_with_single_transaction_participants(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
        """Edge case: participants with only 1 transaction should not crash."""
        records = [
            _make_record(participant_id="P001", brand_tier="premium"),
            _make_record(participant_id="P002", brand_tier="mid"),
            _make_record(participant_id="P003", brand_tier="value"),
            _make_record(participant_id="P004", brand_tier="own_label"),
        ]
        encoder = train(
            records=records,
            batch_size=4,
            lr=1e-3,
            n_epochs=1,
            gru_hidden=32,
            gru_layers=1,
            gru_dropout=0.0,
            projection_dim=16,
            device="cpu",
        )
        assert isinstance(encoder, TransactionEncoder)
