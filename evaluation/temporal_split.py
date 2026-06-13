"""
Temporal train/test split for CDT evaluation.

Per docs/modalities.md: 'use a temporal split, not random; your audience will check.'

The split is per-event, NOT per-participant: the same customer appears in both
train (their months 1-8 events) and eval (their months 9-12 events). This is the
correct temporal evaluation for CDT models — no future events leak into training.

For snapshot modalities (traces, psychographics): fielded at specific months,
so train/eval uses those fielding dates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import structlog

log = structlog.get_logger(__name__)

DATA_DIR = Path("data/synthetic")

# Default temporal split: months 1-8 train, 9-12 eval
DEFAULT_TRAIN_MONTHS: tuple[int, ...] = tuple(range(1, 9))  # 1..8
DEFAULT_EVAL_MONTHS: tuple[int, ...] = tuple(range(9, 13))  # 9..12


def temporal_train_test_split(
    records: list[dict],
    train_months: Sequence[int] = DEFAULT_TRAIN_MONTHS,
    eval_months: Sequence[int] = DEFAULT_EVAL_MONTHS,
) -> tuple[list[dict], list[dict]]:
    """Split records by month into train and eval sets.

    Parameters
    ----------
    records : list of dict
        Records with a 'month' field (1-indexed).
    train_months : sequence of int
        Months to include in the train set (default: 1-8).
    eval_months : sequence of int
        Months to include in the eval set (default: 9-12).

    Returns
    -------
    (train_records, eval_records) — disjoint by month, same customer can
    appear in both with different time windows.
    """
    train_set = set(train_months)
    eval_set = set(eval_months)

    train_records: list[dict] = []
    eval_records: list[dict] = []

    for rec in records:
        month = rec.get("month", 0)
        if month in train_set:
            train_records.append(rec)
        elif month in eval_set:
            eval_records.append(rec)

    log.info(
        "temporal_split.applied",
        n_total=len(records),
        n_train=len(train_records),
        n_eval=len(eval_records),
        train_months=sorted(train_set),
        eval_months=sorted(eval_set),
    )
    return train_records, eval_records


def load_monthly_modality(
    modality: str, months: Sequence[int], data_dir: Path = DATA_DIR
) -> list[dict]:
    """Load month-partitioned files for an event-stream modality.

    Reads ``{modality}_month_{MM}.jsonl`` files for the given months.
    """
    records: list[dict] = []
    for month in months:
        path = data_dir / f"{modality}_month_{month:02d}.jsonl"
        if not path.exists():
            log.warning("temporal_split.missing_month_file", path=str(path))
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def verify_no_temporal_leakage(
    train_records: list[dict], eval_records: list[dict]
) -> bool:
    """Verify no month appears in both train and eval sets.

    Returns True if clean (no leakage), False otherwise.
    """
    train_months = {r.get("month") for r in train_records}
    eval_months = {r.get("month") for r in eval_records}
    overlap = train_months & eval_months
    overlap_clean = {str(m) for m in overlap}
    if overlap:
        log.error(
            "temporal_split.leakage_detected", overlap_months=sorted(overlap_clean)
        )
        return False
    return True


def split_summary(train_records: list[dict], eval_records: list[dict]) -> dict:
    """Produce a summary of the temporal split for reporting."""
    train_parts = {
        r.get("participant_id") or r.get("customer_id") for r in train_records
    }
    eval_parts = {r.get("participant_id") or r.get("customer_id") for r in eval_records}
    overlap_parts = train_parts & eval_parts
    return {
        "n_train_records": len(train_records),
        "n_eval_records": len(eval_records),
        "n_train_participants": len(train_parts),
        "n_eval_participants": len(eval_parts),
        "n_overlapping_participants": len(overlap_parts),
        "note": (
            f"{len(overlap_parts)} participants appear in both train and eval "
            f"with different time windows (correct temporal evaluation)"
        ),
    }
