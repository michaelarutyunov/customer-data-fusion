"""
generator/text_generator.py

Generates persona narratives via LLM (DeepSeek primary, Anthropic fallback).

Public API:
    generate_narrative(config, category) -> PersonaNarrative
    generate_narratives_batch(configs, category) -> list[PersonaNarrative]

PersonaNarrative.embedding is always None at generation time.
Embedding is populated later by encoders/text/embed.py.
"""

from __future__ import annotations

import os
import structlog
from dotenv import load_dotenv

from schemas.persona import PersonaConfig
from schemas.text import PersonaNarrative

load_dotenv()

log = structlog.get_logger()

_PROMPT_VERSION = "v1"

_SYSTEM_PROMPT = (
    "You are a consumer research assistant. Write concise, vivid third-person "
    "prose descriptions of consumer shopping personas. Stay factual and specific."
)


def _build_prompt(config: PersonaConfig, category: str) -> str:
    n = config.narrative
    lo, hi = n.age_range
    return (
        f"Write a 280–320 word third-person description of a consumer who shops for {category}. "
        f"Details: age {lo}–{hi}, {n.household_type.replace('_', ' ')} household, "
        f"described as a '{n.category_relationship}'. "
        f"Decision style: {n.decision_style_description.strip()} "
        f"Price attitude: {n.price_attitude}. "
        "Focus on how they search for information, what they prioritise, and how they make choices. "
        "Do not use bullet points. Write flowing prose only."
    )


def _call_deepseek(prompt: str) -> tuple[str, str]:
    """Returns (text, model_id)."""
    import openai  # local import — optional dependency path

    client = openai.OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=600,
        temperature=0.8,
    )
    text = response.choices[0].message.content or ""
    return text.strip(), "deepseek-chat"


def _call_anthropic(prompt: str) -> tuple[str, str]:
    """Returns (text, model_id)."""
    import anthropic  # local import — optional dependency path

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model_id = "claude-sonnet-4-6"
    message = client.messages.create(
        model=model_id,
        max_tokens=600,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = message.content[0].text if message.content else ""
    return text.strip(), model_id


def _llm_generate(prompt: str) -> tuple[str, str]:
    """Call DeepSeek if key present, else Anthropic. Raises RuntimeError if neither configured."""
    if os.environ.get("DEEPSEEK_API_KEY"):
        return _call_deepseek(prompt)
    if os.environ.get("ANTHROPIC_API_KEY"):
        return _call_anthropic(prompt)
    raise RuntimeError(
        "No LLM API key configured. Set DEEPSEEK_API_KEY or ANTHROPIC_API_KEY in .env"
    )


def generate_narrative(
    config: PersonaConfig,
    category: str = "electronics",
) -> PersonaNarrative:
    """Generate one persona narrative from a PersonaConfig."""
    prompt = _build_prompt(config, category)
    text, model_id = _llm_generate(prompt)
    word_count = len(text.split())

    if word_count < 200 or word_count > 400:
        log.warning(
            "narrative_word_count_out_of_range",
            persona_id=config.persona_id,
            word_count=word_count,
        )

    log.info(
        "narrative_generated",
        persona_id=config.persona_id,
        category=category,
        word_count=word_count,
        model_id=model_id,
    )

    return PersonaNarrative(
        participant_id=config.persona_id,
        persona_id=config.persona_id,
        category=category,
        text=text,
        word_count=word_count,
        model_id=model_id,
        prompt_version=_PROMPT_VERSION,
        embedding=None,
        embedding_model_id=None,
    )


def generate_narratives_batch(
    configs: list[PersonaConfig],
    category: str = "electronics",
) -> list[PersonaNarrative]:
    """Generate narratives for a list of configs sequentially."""
    results: list[PersonaNarrative] = []
    for config in configs:
        results.append(generate_narrative(config, category=category))
    return results
