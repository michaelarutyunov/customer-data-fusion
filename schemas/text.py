"""
Text modality schema — persona narrative (Option A: structured prose).

Generated via LLM (DeepSeek or Anthropic) from NarrativeParams.
Embedded via frozen sentence-transformer at inference time.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PersonaNarrative:
    """
    A short third-person prose description of a consumer's relationship
    with a product category and their typical decision approach.

    Content is generated from PersonaConfig.narrative params and must be
    semantically consistent with the persona's StrategyParams —
    a price-lexicographic persona must sound price-conscious in their narrative.

    embedding: populated after sentence-transformer inference; None at generation time.
    model_id: the LLM used for generation (for reproducibility auditing).
    """
    participant_id: str
    persona_id: str
    category: str
    text: str                         # raw narrative text, 200–400 words
    word_count: int
    model_id: str                     # e.g. "deepseek-chat", "claude-sonnet-4-6"
    prompt_version: str               # version tag of generation prompt template
    embedding: Optional[list[float]] = None   # sentence-transformer output vector
    embedding_model_id: Optional[str] = None  # e.g. "all-MiniLM-L6-v2"