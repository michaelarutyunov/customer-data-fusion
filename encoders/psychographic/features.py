"""
Feature engineering for psychographic encoder.

Converts PsychographicVector dataclass into a fixed-width 22-dim tensor.
Vocabularies for categorical one-hot encodings are fixed at training time
and persisted to JSON for consistent inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch

from schemas.psychographic import PsychographicVector

# ---------------------------------------------------------------------------
# Fixed vocabularies — must match generator output exactly
# ---------------------------------------------------------------------------

DECISION_STYLE_VOCAB: list[str] = [
    "analytical",
    "intuitive",
    "dependent",
    "avoidant",
    "spontaneous",
    "deliberate",
]

AGE_BAND_ORDINAL: dict[str, float] = {
    "18-24": 1.0,
    "25-34": 2.0,
    "35-44": 3.0,
    "45-54": 4.0,
    "55-64": 5.0,
    "65+": 6.0,
}

HOUSEHOLD_TYPE_VOCAB: list[str] = [
    "single",
    "couple",
    "family_with_children",
    "other",
]

EMPLOYMENT_STATUS_VOCAB: list[str] = [
    "full_time",
    "part_time",
    "self_employed",
    "not_employed",
    "retired",
]

PURCHASE_FREQUENCY_ORDINAL: dict[str, float] = {
    "annually_or_less": 1.0,
    "quarterly": 2.0,
    "monthly": 3.0,
    "weekly": 4.0,
}

# Median imputation value for years_buying_category (when None)
YEARS_BUYING_MEDIAN: int = 5

# Total feature dimension: 6 continuous + 6 decision_style + 1 age_band
# + 5 employment_status + 1 purchase_frequency = 19
# household_type excluded: it is deterministic per archetype (label leak).
FEATURE_DIM: int = 19

# Path where vocabularies are persisted
VOCAB_PATH: Path = Path("data/synthetic/psych_vocab.json")


def build_vocab_dict() -> dict[str, object]:
    """Build the complete vocabulary dictionary for persistence."""
    return {
        "decision_style": DECISION_STYLE_VOCAB,
        "age_band_ordinal": AGE_BAND_ORDINAL,
        "household_type": HOUSEHOLD_TYPE_VOCAB,
        "employment_status": EMPLOYMENT_STATUS_VOCAB,
        "purchase_frequency_ordinal": PURCHASE_FREQUENCY_ORDINAL,
        "years_buying_median": YEARS_BUYING_MEDIAN,
    }


def save_vocab(path: Optional[Path] = None) -> Path:
    """Persist vocabularies to JSON for inference-time consistency."""
    target = path or VOCAB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as f:
        json.dump(build_vocab_dict(), f, indent=2)
    return target


def _one_hot(value: str, vocab: list[str]) -> list[float]:
    """Return one-hot encoding of *value* against *vocab*."""
    vec = [0.0] * len(vocab)
    if value in {v: i for i, v in enumerate(vocab)}:
        idx = vocab.index(value)
        vec[idx] = 1.0
    return vec


def to_feature_vector(
    psych: PsychographicVector,
    *,
    years_buying_median: int = YEARS_BUYING_MEDIAN,
) -> torch.Tensor:
    """Convert a PsychographicVector into a 19-dim float32 tensor.

    Feature layout (19 dims):
      [0:6]   -- continuous attitudinal scales (already 0-1)
      [6:12]  -- decision_style_dominant one-hot (6, incl. "deliberate")
      [12]    -- age_band ordinal / 5.0
      [13:18] -- employment_status one-hot (5)
      [18]    -- purchase_frequency_band ordinal / 3.0

    household_type is excluded: it is deterministic per archetype (label leak).
    """
    # Continuous fields (6 dims)
    continuous = [
        psych.involvement_score,
        psych.maximiser_score,
        psych.risk_tolerance,
        psych.price_consciousness,
        psych.brand_sensitivity,
        psych.openness_to_new,
    ]

    # decision_style_dominant one-hot (6 dims, incl. "deliberate")
    decision_style_oh = _one_hot(psych.decision_style_dominant, DECISION_STYLE_VOCAB)

    # age_band ordinal normalised (1 dim)
    age_ordinal = AGE_BAND_ORDINAL.get(psych.age_band, 3.0) / 5.0

    # employment_status one-hot (5 dims)
    employment_oh = _one_hot(psych.employment_status, EMPLOYMENT_STATUS_VOCAB)

    # purchase_frequency_band ordinal normalised (1 dim)
    freq_ordinal = (
        PURCHASE_FREQUENCY_ORDINAL.get(psych.purchase_frequency_band, 2.0) / 3.0
    )

    features = (
        continuous + decision_style_oh + [age_ordinal] + employment_oh + [freq_ordinal]
    )
    assert len(features) == FEATURE_DIM, (
        f"Feature vector has {len(features)} dims, expected {FEATURE_DIM}"
    )
    return torch.tensor(features, dtype=torch.float32)


def batch_to_feature_matrix(
    records: list[PsychographicVector],
) -> torch.Tensor:
    """Convert a list of PsychographicVector into a (N, FEATURE_DIM) tensor."""
    return torch.stack([to_feature_vector(r) for r in records])
