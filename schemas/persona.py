"""
PersonaConfig — the generative root for all synthetic modalities.

Every modality generator reads from this dataclass. Cross-modal consistency
is guaranteed because all downstream data is sampled from the same config.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Strategy(str, Enum):
    LEXICOGRAPHIC = "lexicographic"
    COMPENSATORY = "compensatory"
    SATISFICING = "satisficing"
    AFFECT_HEURISTIC = "affect_heuristic"
    RANDOM = "random"
    ADAPTIVE = "adaptive"


class InspectionDepth(str, Enum):
    SHALLOW = "shallow"       # < 30% cells inspected
    MEDIUM = "medium"         # 30–60% cells inspected
    DEEP = "deep"             # > 60% cells inspected
    VARIABLE = "variable"     # shifts by task condition


class PriceConsciousness(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class StrategyParams:
    """
    Parameters governing search behaviour in process trace simulation.
    Values are distribution parameters, not deterministic rules.
    """
    primary_strategy: Strategy
    inspection_depth: InspectionDepth

    # Lexicographic params (used when strategy == LEXICOGRAPHIC)
    first_attribute: Optional[str] = None          # e.g. "price", "brand"
    rejection_threshold_pct: Optional[float] = None  # 0–1; reject if attr value > this percentile

    # Compensatory params (used when strategy == COMPENSATORY)
    attribute_weights: Optional[dict[str, float]] = None  # must sum to 1.0

    # Satisficing params (used when strategy == SATISFICING)
    aspiration_levels: Optional[dict[str, float]] = None  # attr -> minimum acceptable value (0-1 normalised)

    # Shared noise / variability params
    p_reinspect: float = 0.1           # probability of returning to a previously viewed cell
    p_strategy_lapse: float = 0.05     # probability of deviating from primary strategy on a given trial
    time_pressure_multiplier: float = 0.6  # inspection depth scaling factor under time pressure


@dataclass(frozen=True)
class TransactionParams:
    """
    Parameters governing synthetic purchase history generation.
    """
    price_sensitivity: float           # 0–1; higher = more elastic
    brand_loyalty: float               # 0–1; higher = stronger preference for known brands
    purchase_frequency_per_month: float  # mean purchases per month in this category
    basket_size_mean: int              # mean units per transaction
    channel_mix: dict[str, float]      # e.g. {"online": 0.7, "in_store": 0.3}; must sum to 1.0
    price_variance_tolerance: float    # std dev of acceptable price range (normalised 0–1)


@dataclass(frozen=True)
class PsychographicParams:
    """
    Parameters for psychographic vector generation.
    Mapped to published scale anchors where possible.
    """
    involvement_score: float           # 0–1; category involvement (adapted PII)
    maximiser_score: float             # 0–1; maximiser vs satisficer (Schwartz et al.)
    risk_tolerance: float              # 0–1; higher = more willing to try unknown options
    price_consciousness: PriceConsciousness
    openness_to_new: float             # 0–1; willingness to try new entrants


@dataclass(frozen=True)
class NarrativeParams:
    """
    Parameters for persona narrative (Option A text) generation.
    Passed to LLM prompt as structured context.
    """
    age_range: tuple[int, int]         # e.g. (28, 35)
    household_type: str                # e.g. "single", "couple", "family_with_children"
    category_relationship: str         # e.g. "habitual buyer", "occasional shopper", "reluctant purchaser"
    decision_style_description: str    # One sentence describing how they decide; derived from StrategyParams
    price_attitude: str                # e.g. "price-first", "quality-over-price", "value-seeker"
    narrative_length_words: int = 300  # target word count for generated narrative


@dataclass(frozen=True)
class PersonaConfig:
    """
    The generative root. All synthetic modalities for a participant are
    sampled from this config. Do not modify fields without updating all
    downstream generators and the schema-guardian agent.

    persona_id: unique archetype identifier (matches config/personas.yaml key)
    label: human-readable archetype name for logging and visualisation
    """
    persona_id: str
    label: str
    strategy: StrategyParams
    transactions: TransactionParams
    psychographic: PsychographicParams
    narrative: NarrativeParams

    # Noise seed for reproducible per-participant sampling
    # Set at participant instantiation, not at archetype definition
    random_seed: Optional[int] = None