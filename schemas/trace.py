"""
Process trace schema — AcquisitionEvent and TrialRecord.

Tokenisation target: each AcquisitionEvent is one token in the sequence encoder.
Variable-length sequences per trial handled by Transformer with positional encoding.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EventType(str, Enum):
    """
    Realistic event types for instrumented hover/exposure logging on a
    product comparison page or configurator. Framed as CRM-style front-end
    event log rather than lab artefact (see docs/modalities/mouselab.md).
    """

    CELL_HOVER = "cell_hover"  # Shallow inspection (dwell < 800ms)
    CELL_OPEN = "cell_open"  # Deep inspection (dwell >= 800ms)
    COLUMN_ADD = "column_add"  # First inspection in a new attribute column
    SORT_APPLY = "sort_apply"  # Strategy-driven attribute switch mid-sequence
    CHOICE = "choice"  # Final selection of an alternative


@dataclass(frozen=True)
class AcquisitionEvent:
    """
    A single cell inspection in a MouseLab-style information board.
    One token in the sequence encoder input.

    Coordinates identify the cell: (alternative_id, attribute_id).
    Timestamp is relative to trial start (seconds).
    Dwell time in milliseconds — log-normal distributed in real data.
    """

    participant_id: str
    trial_id: str
    event_index: int  # position in trial sequence (0-based)
    alternative_id: str  # e.g. "A", "B", "C"
    attribute_id: str  # e.g. "price", "brand", "quality"
    timestamp_s: float  # seconds from trial start
    dwell_ms: float  # inspection duration in milliseconds
    is_reinspection: bool  # True if this cell was previously inspected this trial
    event_type: EventType = (
        EventType.CELL_HOVER
    )  # interaction type for realistic framing


@dataclass(frozen=True)
class TrialRecord:
    """
    Metadata for a single information board trial.
    Links to its AcquisitionEvent sequence via trial_id.
    """

    participant_id: str
    trial_id: str
    session_id: str
    trial_index: int  # position in session (0-based)
    category: str  # product category label
    n_alternatives: int  # number of options shown (3, 5, or 7)
    n_attributes: int  # number of attributes shown (4, 6, or 8)
    time_pressure: bool  # whether soft time indicator was shown
    final_choice: Optional[str]  # alternative_id chosen; None if participant passed
    confidence_rating: Optional[
        int
    ]  # 1–5 self-report after trial; None if not collected
    total_acquisitions: int  # length of acquisition sequence for this trial
    prop_cells_inspected: float  # total_acquisitions / (n_alternatives * n_attributes)
    payne_index: float  # +1 = pure alternative-wise, -1 = pure attribute-wise
    persona_id: str  # ground truth archetype (synthetic data only)
