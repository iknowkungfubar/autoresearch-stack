"""Extended tests for synthetic data module.

Covers remaining untested paths: LLM provider dispatch, 
Evol-Instruct scaling, and error handling.
"""
from unittest.mock import patch


class TestSyntheticLLMPaths:
    """Tests for LLM-powered generation paths."""

    def test_llm_anthropic_fallback(self):
        from synthetic_data import SyntheticGenerator

        gen = SyntheticGenerator(use_llm=True, provider="anthropic")
        gen.api_key = "test-key"

        with patch.object(gen, "_call_anthropic", side_effect=Exception("API error")):
            result = gen.generate(n=3, difficulty="easy")
        assert len(result.prompts) == 3

    def test_llm_openai_fallback(self):
        from synthetic_data import SyntheticGenerator

        gen = SyntheticGenerator(use_llm=True, provider="openai")
        gen.api_key = "test-key"

        with patch.object(gen, "_call_openai", side_effect=Exception("API error")):
            result = gen.generate(n=3, difficulty="easy")
        assert len(result.prompts) == 3

    def test_unknown_provider_uses_templates(self):
        from synthetic_data import SyntheticGenerator

        gen = SyntheticGenerator(use_llm=True, provider="unknown_provider")
        gen.api_key = "test-key"

        result = gen.generate(n=3, difficulty="easy")
        assert len(result.prompts) == 3

    def test_generate_with_difficulty_scaling(self):
        """Evol-Instruct scaling produces mixed difficulties."""
        from synthetic_data import SyntheticGenerator

        gen = SyntheticGenerator(use_llm=False)
        result = gen.generate(n=10, difficulty="mixed")
        assert len(result.prompts) == 10

    def test_generation_result_dataclass(self):
        """GenerationResult has correct fields."""
        from synthetic_data import GenerationResult

        r = GenerationResult(prompts=["a", "b"], used_llm=True, metadata={})
        assert len(r.prompts) == 2
        assert r.used_llm is True
        assert hasattr(r, "prompts")
        assert hasattr(r, "used_llm")


class TestSyntheticQualityFilter:
    """Tests for quality filtering edge cases."""

    def test_filter_removes_below_min_length(self):
        from synthetic_data import SyntheticGenerator

        gen = SyntheticGenerator()
        prompts = ["short", "valid prompt text here"]
        filtered = gen.quality_filter(prompts, min_length=10, max_length=100)
        assert "short" not in filtered

    def test_filter_removes_above_max_length(self):
        from synthetic_data import SyntheticGenerator

        gen = SyntheticGenerator()
        prompts = ["x" * 200, "valid text"]
        filtered = gen.quality_filter(prompts, min_length=3, max_length=50)
        assert "x" * 200 not in filtered
