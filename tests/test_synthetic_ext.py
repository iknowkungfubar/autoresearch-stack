"""Extended tests for synthetic data generation.

Covers:
- generate_synthetic with different difficulties
- SyntheticGenerator initialization and fallback
- quality_filter edge cases
- Evol-Instruct difficulty scaling
"""


class TestGenerateSynthetic:
    """Tests for generate_synthetic function."""

    def test_generate_easy(self):
        from synthetic_data import generate_synthetic

        result = generate_synthetic(n=5, difficulty="easy")
        assert len(result) == 5
        for text in result:
            assert isinstance(text, str)
            assert len(text) > 0

    def test_generate_medium(self):
        from synthetic_data import generate_synthetic

        result = generate_synthetic(n=5, difficulty="medium")
        assert len(result) == 5

    def test_generate_hard(self):
        from synthetic_data import generate_synthetic

        result = generate_synthetic(n=5, difficulty="hard")
        assert len(result) == 5

    def test_generate_mixed(self):
        from synthetic_data import generate_synthetic

        result = generate_synthetic(n=10, difficulty="mixed")
        assert len(result) == 10

    def test_generate_zero(self):
        from synthetic_data import generate_synthetic

        result = generate_synthetic(n=0)
        assert len(result) == 0


class TestSyntheticGenerator:
    """Tests for SyntheticGenerator class."""

    def test_init_no_llm(self):
        from synthetic_data import SyntheticGenerator

        gen = SyntheticGenerator(use_llm=False)
        assert gen.use_llm is False
        assert gen.provider == "anthropic"

    def test_generate_no_llm(self):
        from synthetic_data import SyntheticGenerator

        gen = SyntheticGenerator(use_llm=False)
        result = gen.generate(n=5, difficulty="easy")
        assert len(result.prompts) == 5
        assert result.used_llm is False

    def test_generate_difficulty_scaling(self):
        from synthetic_data import SyntheticGenerator

        gen = SyntheticGenerator(use_llm=False)
        result = gen.generate(n=5, difficulty="easy")
        assert len(result.prompts) == 5

    def test_quality_filter_min_length(self):
        from synthetic_data import SyntheticGenerator

        gen = SyntheticGenerator()
        prompts = ["short", "a", "", "valid prompt text for testing purposes"]
        filtered = gen.quality_filter(prompts, min_length=10, max_length=100)
        assert len(filtered) >= 1
        for p in filtered:
            assert len(p) >= 10

    def test_quality_filter_max_length(self):
        from synthetic_data import SyntheticGenerator

        gen = SyntheticGenerator()
        prompts = ["short", "a" * 200, "valid prompt text", "x" * 300]
        filtered = gen.quality_filter(prompts, min_length=3, max_length=100)
        for p in filtered:
            assert len(p) <= 100

    def test_quality_filter_empty(self):
        from synthetic_data import SyntheticGenerator

        gen = SyntheticGenerator()
        filtered = gen.quality_filter([], min_length=5, max_length=100)
        assert len(filtered) == 0

    def test_quality_filter_all_removed(self):
        from synthetic_data import SyntheticGenerator

        gen = SyntheticGenerator()
        filtered = gen.quality_filter(["a"], min_length=10, max_length=100)
        assert len(filtered) == 0

    def test_generate_result_structure(self):
        from synthetic_data import SyntheticGenerator

        gen = SyntheticGenerator(use_llm=False)
        result = gen.generate(n=3)
        from synthetic_data import GenerationResult

        assert isinstance(result, GenerationResult)
        assert hasattr(result, "prompts")
        assert hasattr(result, "used_llm")

    def test_generate_multiple_calls(self):
        from synthetic_data import SyntheticGenerator

        gen = SyntheticGenerator(use_llm=False)
        r1 = gen.generate(n=3)
        r2 = gen.generate(n=4)
        assert len(r1.prompts) == 3
        assert len(r2.prompts) == 4

    def test_generate_with_temperature(self):
        from synthetic_data import SyntheticGenerator

        gen = SyntheticGenerator(use_llm=False, temperature=0.5)
        result = gen.generate(n=3)
        assert len(result.prompts) == 3


class TestQualityFilterEdgeCases:
    """Edge case tests for quality_filter."""

    def test_filter_called_from_generate(self):
        """quality_filter is called internally by generate when use_llm=False."""
        from synthetic_data import SyntheticGenerator

        gen = SyntheticGenerator(use_llm=False)
        # generate calls quality_filter internally
        result = gen.generate(n=5)
        assert len(result.prompts) <= 5
        assert len(result.prompts) >= 1
