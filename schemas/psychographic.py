"""
Psychographic schema — fixed-width survey vector per participant.

Serves as trait-level prior in the fusion architecture.
Demographics are population-level calibrators, not individual predictors —
do not treat demographic fields as preference signals.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PsychographicVector:
    """
    Fixed-width psychographic and demographic profile.
    All continuous scales normalised to 0–1 unless noted.

    Scale sources:
    - involvement_score: adapted Personal Involvement Inventory (PII, Zaichkowsky 1985)
    - maximiser_score: Maximisation Scale (Schwartz et al. 2002)
    - decision_style: adapted from General Decision-Making Style inventory
    """

    participant_id: str
    persona_id: str  # ground truth archetype (synthetic data only)

    # --- Attitudinal scales (0–1 normalised) ---
    involvement_score: float  # category involvement
    maximiser_score: float  # maximiser (1.0) vs satisficer (0.0)
    risk_tolerance: float  # willingness to choose unknown options
    price_consciousness: float  # sensitivity to price information
    brand_sensitivity: float  # reliance on brand as decision cue
    openness_to_new: float  # willingness to try new entrants

    # --- Decision style (mutually exclusive categorical) ---
    decision_style_dominant: (
        str  # "analytical", "intuitive", "dependent", "avoidant", "spontaneous"
    )

    # --- Demographics (population-level calibrators only) ---
    age_band: str  # "18-24", "25-34", "35-44", "45-54", "55-64", "65+"
    household_type: str  # "single", "couple", "family_with_children", "other"
    employment_status: (
        str  # "full_time", "part_time", "self_employed", "not_employed", "retired"
    )

    # --- Category-specific ---
    category: str
    purchase_frequency_band: str  # "weekly", "monthly", "quarterly", "annually_or_less"
    years_buying_category: Optional[int] = None
