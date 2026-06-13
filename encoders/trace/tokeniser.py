"""
Trace tokeniser — converts AcquisitionEvent sequences to padded tensors.

Builds attribute_id and alternative_id vocabularies from traces.jsonl on first
run and caches the vocab to ``data/synthetic/trace_vocab.json``.

Token dim = 31 per event:
  - attribute embedding lookup index (learned in model)  -> 16-dim
  - alternative embedding lookup index (learned in model) -> 8-dim
  - event_type embedding lookup index (learned in model) -> 4-dim
  - timestamp_norm  -> 1
  - dwell_zscore    -> 1
  - is_reinspection -> 1

CLS token is prepended at index 0 using a learned embedding (handled by the
model, not the tokeniser — tokeniser reserves index 0 in all vocab dicts).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from schemas.trace import AcquisitionEvent, EventType, TrialRecord

# Fixed token feature dimensionality
TOKEN_DIM: int = 31

# Fixed event_type vocabulary (5 EventType values; index 0 reserved for CLS).
# CELL_HOVER=1, CELL_OPEN=2, COLUMN_ADD=3, SORT_APPLY=4, CHOICE=5
EVENT_TYPE_VOCAB: dict[str, int] = {et.value: i + 1 for i, et in enumerate(EventType)}
N_EVENT_TYPES: int = len(EVENT_TYPE_VOCAB) + 1  # +1 for CLS (index 0)

# Maximum sequence length (covers 99th percentile of synthetic data)
MAX_SEQ_LEN: int = 200

# Special index for the CLS token placeholder in vocab lookups
CLS_VOCAB_IDX: int = 0

DATA_DIR = Path("data/synthetic")
VOCAB_PATH = DATA_DIR / "trace_vocab.json"


# ---------------------------------------------------------------------------
# Vocabulary building
# ---------------------------------------------------------------------------


def build_vocab(
    events: list[AcquisitionEvent],
    cache_path: Path = VOCAB_PATH,
) -> dict[str, dict[str, int]]:
    """
    Build attribute_id and alternative_id vocabularies from events.

    Index 0 is reserved for CLS.  Results are cached to *cache_path*.

    Returns
    -------
    dict with keys ``"attribute"`` and ``"alternative"``, each mapping a
    string value to an integer index >= 1.
    """
    attributes: set[str] = set()
    alternatives: set[str] = set()
    for ev in events:
        attributes.add(ev.attribute_id)
        alternatives.add(ev.alternative_id)

    attr_vocab = {a: i + 1 for i, a in enumerate(sorted(attributes))}
    alt_vocab = {a: i + 1 for i, a in enumerate(sorted(alternatives))}

    vocab = {"attribute": attr_vocab, "alternative": alt_vocab}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(vocab, indent=2))

    return vocab


def load_vocab(cache_path: Path = VOCAB_PATH) -> dict[str, dict[str, int]] | None:
    """Load cached vocab if it exists, otherwise return None."""
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    return None


def get_or_build_vocab(
    events: list[AcquisitionEvent],
    cache_path: Path = VOCAB_PATH,
) -> dict[str, dict[str, int]]:
    """Return cached vocab or build and cache a new one."""
    cached = load_vocab(cache_path)
    if cached is not None:
        return cached
    return build_vocab(events, cache_path)


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


def tokenise_trial(
    events: list[AcquisitionEvent],
    trial: TrialRecord,
    vocab: dict[str, dict[str, int]],
    max_seq_len: int = MAX_SEQ_LEN,
) -> tuple[Tensor, Tensor]:
    """
    Convert a single trial's events into a token tensor and attention mask.

    Parameters
    ----------
    events:
        AcquisitionEvents for one trial, sorted by event_index.
    trial:
        The corresponding TrialRecord (used for timestamp normalisation).
    vocab:
        Vocab dicts from :func:`get_or_build_vocab`.
    max_seq_len:
        Maximum number of real tokens (excluding CLS).  Excess tokens are
        truncated.

    Returns
    -------
    tokens:
        FloatTensor of shape ``(seq_len + 1, TOKEN_DIM)`` where seq_len is
        ``min(len(events), max_seq_len)``.  The first row (index 0) is a
        zero placeholder for the CLS position — the model will replace it
        with a learned embedding.
    mask:
        BoolTensor of shape ``(seq_len + 1,)`` — True for real tokens
        (including CLS), False for padding if we pad to a fixed length.
        Currently no padding is applied at the single-trial level; all
        entries are True.
    """
    attr_vocab = vocab["attribute"]
    alt_vocab = vocab["alternative"]

    # Truncate to max_seq_len
    truncated = events[:max_seq_len]
    seq_len = len(truncated)

    if seq_len == 0:
        # Edge case: trial with zero acquisitions — only CLS token
        tokens = torch.zeros(1, TOKEN_DIM, dtype=torch.float32)
        mask = torch.ones(1, dtype=torch.bool)
        return tokens, mask

    # Compute trial duration for timestamp normalisation
    timestamps = [e.timestamp_s for e in truncated]
    trial_duration = max(timestamps) if timestamps else 1.0
    if trial_duration == 0.0:
        trial_duration = 1.0

    # Compute dwell z-score within trial
    dwells = np.array([e.dwell_ms for e in truncated], dtype=np.float64)
    dwell_mean = dwells.mean()
    dwell_std = dwells.std()
    if dwell_std < 1e-8:
        dwell_std = 1.0
    dwell_zscore = (dwells - dwell_mean) / dwell_std

    # Build token matrix (seq_len, TOKEN_DIM)
    token_data = torch.zeros(seq_len, TOKEN_DIM, dtype=torch.float32)

    for i, ev in enumerate(truncated):
        offset = 0
        # Attribute embedding index (scalar — model's embedding layer maps to 16-dim)
        token_data[i, offset] = float(attr_vocab.get(ev.attribute_id, 1))
        offset += 1
        # Alternative embedding index
        token_data[i, offset] = float(alt_vocab.get(ev.alternative_id, 1))
        offset += 1
        # Event type embedding index (5 types + CLS placeholder).
        # Handle both EventType enum and raw string (for legacy/test data).
        event_type_val = (
            ev.event_type.value
            if hasattr(ev.event_type, "value")
            else str(ev.event_type)
        )
        token_data[i, offset] = float(EVENT_TYPE_VOCAB.get(event_type_val, 1))
        offset += 1
        # timestamp_norm
        token_data[i, offset] = ev.timestamp_s / trial_duration
        offset += 1
        # dwell_zscore
        token_data[i, offset] = float(dwell_zscore[i])
        offset += 1
        # is_reinspection
        token_data[i, offset] = float(ev.is_reinspection)

    # Prepend CLS placeholder row (all zeros — model will replace)
    cls_row = torch.zeros(1, TOKEN_DIM, dtype=torch.float32)
    tokens = torch.cat([cls_row, token_data], dim=0)  # (seq_len+1, 27)

    # Attention mask: all True (no padding at single-trial level)
    mask = torch.ones(seq_len + 1, dtype=torch.bool)

    return tokens, mask


def collate_batch(
    batch_tokens: list[tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor]:
    """
    Pad a list of ``(tokens, mask)`` tuples to a uniform batch.

    Returns
    -------
    padded_tokens:
        FloatTensor ``(B, max_S, TOKEN_DIM)``
    padded_mask:
        BoolTensor ``(B, max_S)`` — True for real positions.
    """
    max_len = max(t.size(0) for t, _ in batch_tokens)
    batch_size = len(batch_tokens)

    padded_tokens = torch.zeros(batch_size, max_len, TOKEN_DIM, dtype=torch.float32)
    padded_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, (tokens, mask) in enumerate(batch_tokens):
        s = tokens.size(0)
        padded_tokens[i, :s, :] = tokens
        padded_mask[i, :s] = mask

    return padded_tokens, padded_mask
