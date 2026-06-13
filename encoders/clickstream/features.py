"""Clickstream feature tokenisation.

Converts ClickstreamEvent sequences into fixed-dim token vectors using
learned embeddings for categorical features and normalised continuous features.

Token dimension breakdown (8 + 6 + 3 + 1 + 1 = 19):
  event_type_embed   8  (learned, vocab = 8 ClickstreamEventType values)
  page_type_embed    6  (learned, vocab = 6 PageType values)
  device_embed       3  (learned, vocab = 3 DeviceType values)
  dwell_log          1  (log1p(dwell_ms / 1000.0))
  is_purchase        1  (1.0 if event_type == PURCHASE else 0.0)
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

from schemas.clickstream import (
    ClickstreamEvent,
    ClickstreamEventType,
    DeviceType,
    PageType,
)

# Canonical vocabularies — stable across train/test
EVENT_TYPE_VOCAB: list[str] = [e.value for e in ClickstreamEventType]
PAGE_TYPE_VOCAB: list[str] = [p.value for p in PageType]
DEVICE_VOCAB: list[str] = [d.value for d in DeviceType]

TOKEN_DIM: int = 19  # 8 + 6 + 3 + 1 + 1
MAX_SESSIONS: int = 50  # max sessions per customer (truncate most recent)
MAX_EVENTS_PER_SESSION: int = 40  # max events per session


class ClickstreamVocabulary(nn.Module):
    """Learned embedding tables for clickstream categorical features."""

    def __init__(self) -> None:
        super().__init__()
        self.event_type_to_idx = {v: i for i, v in enumerate(EVENT_TYPE_VOCAB)}
        self.page_type_to_idx = {v: i for i, v in enumerate(PAGE_TYPE_VOCAB)}
        self.device_to_idx = {v: i for i, v in enumerate(DEVICE_VOCAB)}

        self.event_type_embed = nn.Embedding(len(EVENT_TYPE_VOCAB), 8)
        self.page_type_embed = nn.Embedding(len(PAGE_TYPE_VOCAB), 6)
        self.device_embed = nn.Embedding(len(DEVICE_VOCAB), 3)

    def _idx(self, mapping: dict[str, int], value: object, default: int = 0) -> int:
        key = value.value if hasattr(value, "value") else str(value)
        return mapping.get(key, default)

    def to_event_token(self, event: ClickstreamEvent) -> torch.Tensor:
        """Convert a single ClickstreamEvent into a 19-dim float tensor."""
        et_idx = torch.tensor(self._idx(self.event_type_to_idx, event.event_type))
        pt_idx = torch.tensor(self._idx(self.page_type_to_idx, event.page_type))
        dev_idx = torch.tensor(self._idx(self.device_to_idx, event.device))

        et_vec = self.event_type_embed(et_idx)  # (8,)
        pt_vec = self.page_type_embed(pt_idx)  # (6,)
        dev_vec = self.device_embed(dev_idx)  # (3,)

        dwell_log = torch.tensor([math.log1p(event.dwell_ms / 1000.0)])
        et_val = (
            event.event_type.value
            if hasattr(event.event_type, "value")
            else str(event.event_type)
        )
        is_purchase = torch.tensor([1.0 if et_val == "purchase" else 0.0])

        return torch.cat([et_vec, pt_vec, dev_vec, dwell_log, is_purchase])  # (19,)

    def encode_session(self, events: Sequence[ClickstreamEvent]) -> torch.Tensor:
        """Encode a single session's events into a (T, 19) tensor."""
        truncated = list(events)[:MAX_EVENTS_PER_SESSION]
        if not truncated:
            return torch.zeros(1, TOKEN_DIM)
        tokens = [self.to_event_token(e) for e in truncated]
        return torch.stack(tokens)  # (T, 19)
