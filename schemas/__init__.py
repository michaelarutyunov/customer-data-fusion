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

__all__ = [
    # Encoder contract
    "EMBEDDING_DIM",
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
