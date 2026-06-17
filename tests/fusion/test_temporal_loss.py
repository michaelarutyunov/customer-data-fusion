import torch
from fusion.temporal_loss import temporal_contrastive_loss, _get_positive_pairs


def test_temporal_contrastive_loss_shape():
    """Test that temporal loss returns scalar."""
    embeddings = torch.randn(4, 12, 128)
    loss = temporal_contrastive_loss(embeddings)
    assert loss.dim() == 0  # scalar
    assert not torch.isnan(loss)


def test_temporal_positive_pairs():
    """Test that positive pairs are correctly constructed."""
    embeddings = torch.randn(4, 12, 128)
    pairs = _get_positive_pairs(embeddings)
    assert len(pairs) == 4 * 11  # 11 adjacent pairs per participant
    assert pairs[0] == (0, 1)  # (participant_0, month_0) -> (participant_0, month_1)


def test_temporal_missing_padding():
    """Test that missing months are handled correctly."""
    embeddings = torch.randn(2, 12, 128)

    # Create mask: participant 0 has month 5 missing, participant 1 complete
    missing_mask = torch.ones(2, 12, dtype=torch.bool)
    missing_mask[0, 5] = False  # mark month 5 as missing

    loss = temporal_contrastive_loss(embeddings, missing_mask=missing_mask)
    assert not torch.isnan(loss)
    assert loss > 0  # should still have positive pairs


def test_temporal_all_missing():
    """Test edge case where all months are missing for one participant."""
    embeddings = torch.randn(2, 12, 128)
    missing_mask = torch.zeros(2, 12, dtype=torch.bool)  # all missing
    missing_mask[1, :] = True  # second participant complete

    loss = temporal_contrastive_loss(embeddings, missing_mask=missing_mask)
    # Should only use participant 1's pairs
    assert not torch.isnan(loss)
