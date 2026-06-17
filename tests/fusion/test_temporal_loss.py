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
