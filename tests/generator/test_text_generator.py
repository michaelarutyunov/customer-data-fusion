"""Unit tests for generator/text_generator.py — all API calls mocked."""
from __future__ import annotations

import pytest
from unittest.mock import patch

from schemas.text import PersonaNarrative
from generator.persona_sampler import sample_persona


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MOCK_TEXT = " ".join(["word"] * 290)  # 290 words — within 200–400 range


# ---------------------------------------------------------------------------
# Tests using DeepSeek path
# ---------------------------------------------------------------------------

class TestGenerateNarrativeDeepSeek:
    @patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key", "ANTHROPIC_API_KEY": ""}, clear=False)
    @patch("generator.text_generator._call_deepseek", return_value=(MOCK_TEXT, "deepseek-chat"))
    def test_returns_persona_narrative(self, mock_call):
        from generator.text_generator import generate_narrative
        config = sample_persona("price_lex", random_seed=42)
        result = generate_narrative(config, category="electronics")
        assert isinstance(result, PersonaNarrative)

    @patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key", "ANTHROPIC_API_KEY": ""}, clear=False)
    @patch("generator.text_generator._call_deepseek", return_value=(MOCK_TEXT, "deepseek-chat"))
    def test_embedding_is_none(self, mock_call):
        from generator.text_generator import generate_narrative
        config = sample_persona("price_lex", random_seed=42)
        result = generate_narrative(config)
        assert result.embedding is None
        assert result.embedding_model_id is None

    @patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key", "ANTHROPIC_API_KEY": ""}, clear=False)
    @patch("generator.text_generator._call_deepseek", return_value=(MOCK_TEXT, "deepseek-chat"))
    def test_word_count_set_correctly(self, mock_call):
        from generator.text_generator import generate_narrative
        config = sample_persona("compensatory", random_seed=42)
        result = generate_narrative(config)
        assert result.word_count == len(MOCK_TEXT.split())

    @patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key", "ANTHROPIC_API_KEY": ""}, clear=False)
    @patch("generator.text_generator._call_deepseek", return_value=(MOCK_TEXT, "deepseek-chat"))
    def test_persona_id_and_category(self, mock_call):
        from generator.text_generator import generate_narrative
        config = sample_persona("brand_affect", random_seed=42)
        result = generate_narrative(config, category="food")
        assert result.persona_id == "brand_affect"
        assert result.category == "food"

    @patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key", "ANTHROPIC_API_KEY": ""}, clear=False)
    @patch("generator.text_generator._call_deepseek", return_value=(MOCK_TEXT, "deepseek-chat"))
    def test_model_id_is_deepseek(self, mock_call):
        from generator.text_generator import generate_narrative
        config = sample_persona("satisficer", random_seed=42)
        result = generate_narrative(config)
        assert result.model_id == "deepseek-chat"

    @patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key", "ANTHROPIC_API_KEY": ""}, clear=False)
    @patch("generator.text_generator._call_deepseek", return_value=(MOCK_TEXT, "deepseek-chat"))
    def test_prompt_version(self, mock_call):
        from generator.text_generator import generate_narrative
        config = sample_persona("adaptive", random_seed=42)
        result = generate_narrative(config)
        assert result.prompt_version == "v1"


# ---------------------------------------------------------------------------
# Tests using Anthropic fallback
# ---------------------------------------------------------------------------

class TestGenerateNarrativeAnthropicFallback:
    @patch("generator.text_generator._llm_generate", return_value=(MOCK_TEXT, "claude-sonnet-4-6"))
    def test_falls_back_to_anthropic(self, mock_llm):
        from generator.text_generator import generate_narrative
        config = sample_persona("quality_lex", random_seed=42)
        result = generate_narrative(config)
        assert result.model_id == "claude-sonnet-4-6"
        mock_llm.assert_called_once()

    @patch.dict("os.environ", {"DEEPSEEK_API_KEY": "", "ANTHROPIC_API_KEY": ""}, clear=False)
    def test_raises_when_no_key_set(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from generator.text_generator import _llm_generate
        with pytest.raises(RuntimeError, match="No LLM API key"):
            _llm_generate("test prompt")


# ---------------------------------------------------------------------------
# Batch tests
# ---------------------------------------------------------------------------

class TestGenerateNarrativesBatch:
    @patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key", "ANTHROPIC_API_KEY": ""}, clear=False)
    @patch("generator.text_generator._call_deepseek", return_value=(MOCK_TEXT, "deepseek-chat"))
    def test_batch_returns_correct_length(self, mock_call):
        from generator.text_generator import generate_narratives_batch
        configs = [sample_persona(aid, random_seed=i) for i, aid in enumerate(["price_lex", "compensatory", "satisficer"])]
        results = generate_narratives_batch(configs, category="electronics")
        assert len(results) == 3

    @patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key", "ANTHROPIC_API_KEY": ""}, clear=False)
    @patch("generator.text_generator._call_deepseek", return_value=(MOCK_TEXT, "deepseek-chat"))
    def test_batch_all_embeddings_none(self, mock_call):
        from generator.text_generator import generate_narratives_batch
        configs = [sample_persona("price_lex", random_seed=i) for i in range(3)]
        results = generate_narratives_batch(configs)
        assert all(r.embedding is None for r in results)

    @patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key", "ANTHROPIC_API_KEY": ""}, clear=False)
    @patch("generator.text_generator._call_deepseek", return_value=(MOCK_TEXT, "deepseek-chat"))
    def test_batch_personas_have_correct_ids(self, mock_call):
        from generator.text_generator import generate_narratives_batch
        archetypes = ["price_lex", "brand_affect"]
        configs = [sample_persona(aid, random_seed=i) for i, aid in enumerate(archetypes)]
        results = generate_narratives_batch(configs)
        assert [r.persona_id for r in results] == archetypes
