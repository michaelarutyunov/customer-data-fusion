"""
schemas — data contracts for customer-data-fusion.

All modules import from here. Generator and encoders never import each other.
Modifying any dataclass requires updating all downstream generators and encoders.
"""

from schemas.persona import (
    PersonaConfig,
    StrategyParams,
    TransactionParams,
    PsychographicParams,
    NarrativeParams,
    Strategy,
    InspectionDepth,
    PriceConsciousness,
)
from schemas.trace import AcquisitionEvent, TrialRecord
from schemas.transaction import TransactionRecord, Channel, PurchaseType
from schemas.text import PersonaNarrative
from schemas.psychographic import PsychographicVector

# Output dimension for all modality encoders — must match fusion meta-learner input.
# Changing this requires retraining all encoders and the fusion layer.
EMBEDDING_DIM: int = 128

# Must match archetypes defined in config/personas.yaml.
# Adding an archetype is a deliberate schema change — update both this list
# and personas.yaml together.
PERSONA_LABELS: list[str] = [
    "price_lex",
    "quality_lex",
    "compensatory",
    "satisficer",
    "brand_affect",
    "adaptive",
    "low_involve",
]
PERSONA_TO_IDX: dict[str, int] = {p: i for i, p in enumerate(PERSONA_LABELS)}

__all__ = [
    # Encoder contract
    "EMBEDDING_DIM",
    # Archetype labels
    "PERSONA_LABELS",
    "PERSONA_TO_IDX",
    # Persona (generative root)
    "PersonaConfig",
    "StrategyParams",
    "TransactionParams",
    "PsychographicParams",
    "NarrativeParams",
    "Strategy",
    "InspectionDepth",
    "PriceConsciousness",
    # Trace
    "AcquisitionEvent",
    "TrialRecord",
    # Transaction
    "TransactionRecord",
    "Channel",
    "PurchaseType",
    # Text
    "PersonaNarrative",
    # Psychographic
    "PsychographicVector",
]
