"""
schemas — data contracts for customer-data-fusion.

All modules import from here. Generator and encoders never import each other.
Modifying any dataclass requires updating all downstream generators and encoders.
"""

from pathlib import Path

from schemas.persona import (
    PersonaConfig,
    LatentDeviation,
    StrategyParams,
    TransactionParams,
    PsychographicParams,
    NarrativeParams,
    Strategy,
    InspectionDepth,
    PriceConsciousness,
)
from schemas.trace import AcquisitionEvent, TrialRecord, EventType
from schemas.transaction import (
    TransactionRecord,
    Channel,
    PurchaseType,
    PaymentMethod,
)
from schemas.text import PersonaNarrative
from schemas.psychographic import PsychographicVector
from schemas.clickstream import (
    ClickstreamEvent,
    ClickstreamEventType,
    SessionSummary,
    SessionIntent,
    PageType,
    ReferrerType,
    DeviceType,
)
from schemas.campaign import CampaignEvent, CampaignType

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

# Canonical checkpoint paths for all modality encoders.
# All encoder training scripts save to these paths; all probe/fusion scripts load from them.
# Paths are relative to the repo root.
CHECKPOINT_PATHS: dict[str, Path] = {
    "trace": Path("models/trace_encoder.pt"),
    "transaction": Path("models/transaction_encoder.pt"),
    "text": Path("models/text_encoder.pt"),
    "psychographic": Path("models/psychographic_encoder.pt"),
    "clickstream": Path("models/clickstream_encoder.pt"),
    "campaign": Path("models/campaign_encoder.pt"),
    "fusion": Path("models/fusion_meta_learner.pt"),
}

# Path for participant config continuous latent variables output.
# Written by generator/pipeline.py; read by evaluation/config_probe.py and geometry.py.
PARTICIPANT_CONFIG_PATH: Path = Path("data/synthetic/participant_configs.jsonl")

__all__ = [
    # Encoder contract
    "EMBEDDING_DIM",
    "CHECKPOINT_PATHS",
    "PARTICIPANT_CONFIG_PATH",
    # Archetype labels
    "PERSONA_LABELS",
    "PERSONA_TO_IDX",
    # Persona (generative root)
    "PersonaConfig",
    "LatentDeviation",
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
    "EventType",
    # Transaction
    "TransactionRecord",
    "Channel",
    "PurchaseType",
    "PaymentMethod",
    # Text
    "PersonaNarrative",
    # Psychographic
    "PsychographicVector",
    # Clickstream
    "ClickstreamEvent",
    "ClickstreamEventType",
    "SessionSummary",
    "SessionIntent",
    "PageType",
    "ReferrerType",
    "DeviceType",
    # Campaign
    "CampaignEvent",
    "CampaignType",
]
