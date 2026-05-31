"""Tests for hypothesis generation module.

Covers:
- HypothesisGenerator with LLM=False (template path)
- generate_from_analysis for underfitting/overfitting/baseline
- Template selection edge cases (unknown change_type, empty templates)
- Fallback paths when LLM generation fails
"""

from unittest.mock import patch


class TestHypothesisTemplates:
    """Tests for template-based hypothesis generation."""

    def test_generate_optimization(self):
        from hypothesis import HypothesisGenerator

        gen = HypothesisGenerator(use_llm=False)
        results = gen.generate(n=3, change_type="optimization")
        assert len(results) == 3
        for h in results:
            assert h.change_type == "optimization"
            assert h.description
            assert h.code_diff

    def test_generate_architecture(self):
        from hypothesis import HypothesisGenerator

        gen = HypothesisGenerator(use_llm=False)
        results = gen.generate(n=2, change_type="architecture")
        assert len(results) == 2
        for h in results:
            assert h.change_type == "architecture"

    def test_generate_curriculum(self):
        from hypothesis import HypothesisGenerator

        gen = HypothesisGenerator(use_llm=False)
        results = gen.generate(n=2, change_type="curriculum")
        assert len(results) == 2
        for h in results:
            assert h.change_type == "curriculum"

    def test_generate_synthetic(self):
        from hypothesis import HypothesisGenerator

        gen = HypothesisGenerator(use_llm=False)
        results = gen.generate(n=2, change_type="synthetic")
        assert len(results) == 2
        for h in results:
            assert h.change_type == "synthetic"

    def test_generate_none_type(self):
        """When change_type is None, a random type is chosen."""
        from hypothesis import HypothesisGenerator

        gen = HypothesisGenerator(use_llm=False)
        results = gen.generate(n=1, change_type=None)
        assert len(results) == 1

    def test_generate_invalid_type(self):
        """Invalid change_type falls back to optimization."""
        from hypothesis import HypothesisGenerator

        gen = HypothesisGenerator(use_llm=False)
        results = gen.generate(n=2, change_type="unknown_type")
        assert len(results) == 2

    def test_generate_more_than_available(self):
        """Requesting more than available templates returns all available."""
        from hypothesis import HypothesisGenerator

        gen = HypothesisGenerator(use_llm=False)
        # Architecture only has 4 templates, request 10
        results = gen.generate(n=10, change_type="architecture")
        assert len(results) <= 10
        assert len(results) >= 1

    def test_generate_zero(self):
        """Requesting 0 hypotheses returns empty list."""
        from hypothesis import HypothesisGenerator

        gen = HypothesisGenerator(use_llm=False)
        results = gen.generate(n=0, change_type="optimization")
        assert len(results) == 0


class TestHypothesisAnalysis:
    """Tests for analysis-based hypothesis generation."""

    def test_high_training_loss(self):
        """High training loss suggests underfitting → higher LR."""
        from hypothesis import HypothesisGenerator

        gen = HypothesisGenerator(use_llm=False)
        h = gen.generate_from_analysis(training_loss=6.0, val_bpb=2.0)
        assert h is not None
        assert "learning_rate" in h.change or "learning_rate" in h.hypothesis_type

    def test_possible_overfitting(self):
        """Low train loss, high val_bpb suggests overfitting → dropout."""
        from hypothesis import HypothesisGenerator

        gen = HypothesisGenerator(use_llm=False)
        h = gen.generate_from_analysis(training_loss=0.05, val_bpb=2.0)
        assert h is not None
        assert h.change_type == "architecture"

    def test_poor_baseline(self):
        """High val_bpb with moderate loss → increase capacity."""
        from hypothesis import HypothesisGenerator

        gen = HypothesisGenerator(use_llm=False)
        h = gen.generate_from_analysis(training_loss=0.5, val_bpb=1.5)
        assert h is not None
        assert "n_embd" in h.change

    def test_good_baseline(self):
        """Low everything → general optimization."""
        from hypothesis import HypothesisGenerator

        gen = HypothesisGenerator(use_llm=False)
        h = gen.generate_from_analysis(training_loss=0.5, val_bpb=0.8)
        assert h is not None
        assert h.change_type == "optimization"

    def test_analysis_with_memory_context(self):
        """Memory context passed but not used in non-LLM mode."""
        from hypothesis import HypothesisGenerator

        gen = HypothesisGenerator(use_llm=False)
        memory = [{"change": "LR test", "status": "kept"}]
        h = gen.generate_from_analysis(
            training_loss=6.0, val_bpb=2.0, memory_context=memory
        )
        assert h is not None


class TestHypothesisLLMFallback:
    """Tests for LLM generation fallback paths.

    When LLM is enabled but the API call fails, it should
    fall back to template generation.
    """

    def test_llm_fallback_on_api_error(self):
        """When LLM API errors, fall back to templates."""
        from hypothesis import HypothesisGenerator

        gen = HypothesisGenerator(use_llm=True, provider="anthropic")
        gen.api_key = "sk-test-fake"

        with patch.object(gen, "_call_anthropic", side_effect=Exception("API error")):
            results = gen.generate(n=2, change_type="optimization")

        # Should fall back to templates
        assert len(results) == 2

    def test_llm_disabled_without_api_key(self):
        """When use_llm=True but no API key, uses templates."""
        from hypothesis import HypothesisGenerator

        gen = HypothesisGenerator(use_llm=True, provider="anthropic")
        gen.api_key = None

        results = gen.generate(n=2, change_type="optimization")
        assert len(results) == 2

    def test_generate_with_memory_context(self):
        """Memory context is accepted but doesn't affect template output."""
        from hypothesis import HypothesisGenerator

        gen = HypothesisGenerator(use_llm=False)
        memory = [
            {"change": "LR test 1", "status": "kept"},
            {"change": "LR test 2", "status": "reverted"},
        ]
        results = gen.generate(n=2, change_type="optimization", memory_context=memory)
        assert len(results) == 2


class TestHypothesisDataclass:
    """Tests for the Hypothesis dataclass."""

    def test_to_dict(self):
        from hypothesis import Hypothesis

        h = Hypothesis(
            change="learning_rate",
            description="Test hypothesis",
            change_type="optimization",
            hypothesis_type="learning_rate",
            expected_impact="high",
            reasoning="Testing",
            code_diff="config.lr *= 1.1",
        )
        d = h.to_dict()
        assert d["change"] == "learning_rate"
        assert d["description"] == "Test hypothesis"
        assert d["change_type"] == "optimization"
        assert d["expected_impact"] == "high"

    def test_hypothesis_str_fields(self):
        from hypothesis import Hypothesis

        h = Hypothesis(
            change="batch_size",
            description="Halve batch size",
            change_type="optimization",
            hypothesis_type="batch_size",
            expected_impact="low",
            reasoning="Smaller batches add noise",
            code_diff="config.batch_size //= 2",
        )
        assert isinstance(h.change, str)
        assert isinstance(h.description, str)
        assert isinstance(h.expected_impact, str)


class TestChangeType:
    """Tests for ChangeType enum."""

    def test_values(self):
        from hypothesis import ChangeType

        assert ChangeType.OPTIMIZATION.value == "optimization"
        assert ChangeType.ARCHITECTURE.value == "architecture"
        assert ChangeType.CURRICULUM.value == "curriculum"
        assert ChangeType.SYNTHETIC.value == "synthetic"
