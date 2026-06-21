"""ChoiceSet schema — choice context for a single trial."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ChoiceSet:
    """Records the choice context for a single trial.

    Links to TrialRecord via choice_set_id. Contains product attributes
    actually displayed and the consumer's choice.
    """

    choice_set_id: str  # UUID linking to TrialRecord
    participant_id: str
    n_alternatives: int  # 3, 5, or 7
    alternative_products: Dict[str, str]  # {"A": "prod_xxx", "B": "prod_yyy"}
    displayed_attributes: Dict[
        str, Dict[str, float]
    ]  # {"A": {"price": 0.8, "quality": 0.6}}
    chosen_alternative: str  # "A", "B", "C", etc.
    choice_probabilities: Dict[str, float]  # {"A": 0.2, "B": 0.7, ...}
