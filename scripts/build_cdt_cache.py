#!/usr/bin/env python3
"""
Build the frozen CDT embedding cache consumed by application capabilities.

Materialises the per-participant Consumer Digital Twin (CDT) embedding — the
128-dim output of the frozen fusion meta-learner's ``embed`` head — to a Parquet
file that downstream capabilities (L1 churn, M1 choice, M2 market, L2 ranking)
read instead of recomputing embeddings inline.

Contract (new-capabilities.md § Module placement):
    applications/_cache/cdt_embeddings.parquet
    keyed by (participant_id, session_id) with a 128-float ``cdt`` column.

The frozen fusion cache (``models/fusion_embeddings_cache.pt``) stores
per-MODALITY embeddings; this script concatenates them in the meta-learner's
canonical slot order and runs ``embed`` once. It does NOT call ``load_encoders``
(the frozen per-modality embeddings are already in the cache), so it is
unaffected by encoder-checkpoint drift.

Run:
    uv run python scripts/build_cdt_cache.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

# Project modules live at the repo root, not in scripts/ — bootstrap sys.path so
# this runs without PYTHONPATH=. (mirrors scripts/generate_missing_narratives.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fusion.meta_learner import LateFusionMetaLearner  # noqa: E402
from schemas import EMBEDDING_DIM  # noqa: E402

CACHE_PATH = Path("models/fusion_embeddings_cache.pt")
FUSION_CHECKPOINT = Path("models/fusion_meta_learner.pt")
OUT_PATH = Path("applications/_cache/cdt_embeddings.parquet")

# Canonical modality slot order — MUST match LateFusionMetaLearner's concat order
# and the fusion cache key order. Stacking in any other order silently corrupts
# the CDT embedding.
_MODALITY_ORDER: tuple[str, ...] = (
    "trace",
    "transaction",
    "text",
    "psychographic",
    "clickstream",
    "campaign",
)


def build(out_path: Path = OUT_PATH) -> Path:
    cache = torch.load(CACHE_PATH, weights_only=False)  # noqa: S614 — cache holds participant_ids (list[str])
    missing = [m for m in _MODALITY_ORDER if m not in cache]
    if missing:
        raise KeyError(
            f"Fusion cache {CACHE_PATH} missing modalities {missing}; "
            f"rebuild it with fusion/train.py before running this script."
        )

    model = LateFusionMetaLearner()  # defaults: n_modalities=6, phase="2"
    model.load_state_dict(
        torch.load(FUSION_CHECKPOINT, map_location="cpu", weights_only=True)
    )
    model.eval()

    pids: list[str] = cache["participant_ids"]
    n = len(pids)

    # [N, n_modalities * EMBEDDING_DIM] — one row per participant
    stacked = torch.cat([cache[m] for m in _MODALITY_ORDER], dim=1)
    expected = len(_MODALITY_ORDER) * EMBEDDING_DIM
    if stacked.shape != (n, expected):
        raise ValueError(
            f"Stacked embeddings {tuple(stacked.shape)} != expected "
            f"({n}, {expected}); cache modality layout changed."
        )

    with torch.no_grad():
        cdt = model.embed(stacked)  # [N, 128]
    assert cdt.shape == (n, EMBEDDING_DIM), cdt.shape

    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "participant_id": pa.array(pids, type=pa.string()),
            # session_id reserved for the future multi-session key; the frozen
            # cache holds one baseline CDT per participant (month-1 dedup'd).
            "session_id": pa.array(["baseline"] * n, type=pa.string()),
            "cdt": pa.array(cdt.tolist(), type=pa.list_(pa.float32(), EMBEDDING_DIM)),
        }
    )
    pq.write_table(table, out_path)
    print(f"Wrote {n} CDT embeddings ({EMBEDDING_DIM}-d) to {out_path}")
    return out_path


if __name__ == "__main__":
    build()
