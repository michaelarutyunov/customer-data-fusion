"""
Smoke tests for encoders/clickstream/.

Covers:
  - model.py: forward pass with a tiny synthetic batch, output is 128-dim
  - forward_with_logits returns (embedding, logits) with expected shapes
"""

from __future__ import annotations

import torch

from schemas import EMBEDDING_DIM
from encoders.clickstream.features import TOKEN_DIM
from encoders.clickstream.model import ClickstreamEncoder


def _tiny_batch(
    batch: int = 4,
    n_sessions: int = 3,
    max_events: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a tiny synthetic customer-session batch.

    Returns session_tokens (B, N, T, TOKEN_DIM), session_lens (B, N), and a
    boolean session_mask (B, N).
    """
    session_tokens = torch.rand(batch, n_sessions, max_events, TOKEN_DIM)
    session_lens = torch.randint(low=1, high=max_events + 1, size=(batch, n_sessions))
    session_mask = torch.ones(batch, n_sessions, dtype=torch.bool)
    return session_tokens, session_lens, session_mask


def test_forward_output_is_128_dim() -> None:
    """forward(session_embeddings, session_mask) yields a (B, 128) embedding."""
    encoder = ClickstreamEncoder()
    encoder.eval()
    session_tokens, session_lens, session_mask = _tiny_batch()

    with torch.no_grad():
        # Encode sessions to (B, N, gru_hidden), then aggregate.
        B, N, T = (
            session_tokens.shape[0],
            session_tokens.shape[1],
            session_tokens.shape[2],
        )
        flat = session_tokens.view(B * N, T, TOKEN_DIM)
        session_embs = encoder.encode_session(flat, session_lens.view(B * N))
        session_embs = session_embs.view(B, N, -1)
        emb = encoder(session_embs, session_mask)

    assert emb.shape == (4, EMBEDDING_DIM)
    assert emb.dtype == torch.float32


def test_forward_with_logits_shapes() -> None:
    """forward_with_logits returns (emb [B,128], logits [B,7])."""
    encoder = ClickstreamEncoder()
    encoder.eval()
    session_tokens, session_lens, session_mask = _tiny_batch()

    with torch.no_grad():
        B, N, T = (
            session_tokens.shape[0],
            session_tokens.shape[1],
            session_tokens.shape[2],
        )
        flat = session_tokens.view(B * N, T, TOKEN_DIM)
        session_embs = encoder.encode_session(flat, session_lens.view(B * N))
        session_embs = session_embs.view(B, N, -1)
        emb, logits = encoder.forward_with_logits(session_embs, session_mask)

    assert emb.shape == (4, EMBEDDING_DIM)
    assert logits.shape == (4, 7)


def test_forward_handles_single_session_customer() -> None:
    """A customer with one real session still produces a 128-dim embedding."""
    encoder = ClickstreamEncoder()
    encoder.eval()
    batch, n_sessions, max_events = 2, 1, 4
    session_tokens = torch.rand(batch, n_sessions, max_events, TOKEN_DIM)
    session_lens = torch.full((batch, n_sessions), max_events, dtype=torch.long)
    session_mask = torch.ones(batch, n_sessions, dtype=torch.bool)

    with torch.no_grad():
        B, N, T = (
            session_tokens.shape[0],
            session_tokens.shape[1],
            session_tokens.shape[2],
        )
        flat = session_tokens.view(B * N, T, TOKEN_DIM)
        session_embs = encoder.encode_session(flat, session_lens.view(B * N))
        session_embs = session_embs.view(B, N, -1)
        emb = encoder(session_embs, session_mask)

    assert emb.shape == (batch, EMBEDDING_DIM)
