from __future__ import annotations
import torch
import torch.nn.functional as F


def _get_positive_pairs(monthly_embeddings: torch.Tensor) -> list[tuple[int, int]]:
    """Get positive pair indices for adjacent months.

    Parameters
    ----------
    monthly_embeddings : Tensor, shape [B, 12, 128]
        Monthly embeddings per participant.

    Returns
    -------
    list of (int, int) tuples
        Positive pair indices. Each tuple is (idx1, idx2) where idx1 and idx2
        are flattened indices into monthly_embeddings.
    """
    B, T, _ = monthly_embeddings.shape
    positive_pairs = []

    for i in range(B):
        for t in range(T - 1):
            idx1 = i * T + t  # (participant_i, month_t)
            idx2 = i * T + t + 1  # (participant_i, month_{t+1})
            positive_pairs.append((idx1, idx2))

    return positive_pairs


def temporal_contrastive_loss(
    monthly_embeddings: torch.Tensor,
    temperature: float = 0.07,
    missing_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Temporal contrastive loss for monthly embedding sequences.

    Positive pairs: (participant_i, month_t) with (participant_i, month_{t+1})
    Negative pairs: (participant_i, month_t) with (participant_j, any_month)

    Parameters
    ----------
    monthly_embeddings : Tensor, shape [B, 12, 128]
        B participants, 12 monthly observations, 128-dim embeddings
    temperature : float
        NT-Xent temperature parameter (default 0.07)
    missing_mask : Tensor | None, shape [B, 12]
        Boolean mask where True = valid, False = missing. If None, all assumed valid.

    Returns
    -------
    Tensor
        Scalar loss tensor
    """
    B, T, D = monthly_embeddings.shape

    # Handle missing data: if mask provided, replace missing entries with zeros
    if missing_mask is not None:
        embeddings_clean = monthly_embeddings * missing_mask.unsqueeze(-1)
    else:
        embeddings_clean = monthly_embeddings

    # Flatten to [B*T, D] for SimCLR-style contrastive
    embeddings_flat = embeddings_clean.reshape(-1, D)
    embeddings_norm = F.normalize(embeddings_flat, dim=1, p=2)

    # Compute similarity matrix: [B*T, B*T]
    sim_matrix = torch.mm(embeddings_norm, embeddings_norm.t()) / temperature

    # Get positive pairs
    positive_pairs = _get_positive_pairs(monthly_embeddings)

    if len(positive_pairs) == 0:
        return torch.tensor(0.0, device=monthly_embeddings.device)

    # Compute NT-Xent loss
    total_loss = torch.tensor(0.0, device=monthly_embeddings.device)
    for idx1, idx2 in positive_pairs:
        sim_positive = sim_matrix[idx1, idx2]

        # Sum over all negatives (all k except idx1 itself)
        sim_all = sim_matrix[idx1].clone()
        sim_all[idx1] = float("-inf")  # exclude self

        # Log-sum-exp trick for numerical stability
        max_sim = torch.max(sim_all)
        log_sum_exp = torch.log(torch.sum(torch.exp(sim_all - max_sim))) + max_sim

        loss_i = log_sum_exp - sim_positive
        total_loss += loss_i

    return total_loss / len(positive_pairs)
