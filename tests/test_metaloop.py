"""Tests for infrastructure/metaloop.py - MetaLoop self-improvement system."""

import json

import pytest


class TestPromptTemplate:
    """Tests for PromptTemplate dataclass."""

    def test_default_created_at(self):
        from autoresearch.infrastructure.metaloop import PromptTemplate

        pt = PromptTemplate(name="test", version=1, content="hello")
        assert pt.name == "test"
        assert pt.version == 1
        assert pt.content == "hello"
        assert pt.created_at is not None
        assert pt.performance is None
        assert pt.notes == ""


class TestModification:
    """Tests for Modification dataclass."""

    def test_default_status_and_impact(self):
        from autoresearch.infrastructure.metaloop import Modification, ModificationType

        mod = Modification(
            id="mod_1",
            type=ModificationType.PROMPT,
            description="test",
            old_value="old",
            new_value="new",
            expected_impact=0.05,
        )
        assert mod.status == "pending"
        assert mod.actual_impact is None
        assert mod.type == ModificationType.PROMPT


class TestModificationType:
    """Tests for ModificationType enum."""

    def test_values(self):
        from autoresearch.infrastructure.metaloop import ModificationType

        assert ModificationType.PROMPT.value == "prompt"
        assert ModificationType.HYPERPARAMETER.value == "hyperparameter"
        assert ModificationType.STRATEGY.value == "strategy"
        assert ModificationType.ARCHITECTURE.value == "architecture"


class TestMetaConfig:
    """Tests for MetaConfig dataclass."""

    def test_defaults(self):
        from autoresearch.infrastructure.metaloop import MetaConfig

        cfg = MetaConfig()
        assert cfg.max_iterations == 10
        assert cfg.min_improvement == 0.01
        assert cfg.prompt_template_file == "prompts.json"
        assert cfg.modifications_file == "modifications.json"


class TestCreateDefaultPrompts:
    """Tests for create_default_prompts function."""

    def test_returns_three_prompts(self):
        from autoresearch.infrastructure.metaloop import create_default_prompts

        prompts = create_default_prompts()
        assert len(prompts) == 3
        assert "hypothesis" in prompts
        assert "evaluation" in prompts
        assert "execution" in prompts
        assert all(prompts.values())  # no empty strings

    def test_prompts_are_immutable(self):
        from autoresearch.infrastructure.metaloop import create_default_prompts

        prompts = create_default_prompts()
        original = dict(prompts)
        prompts.clear()
        # Calling again should still return the full set
        prompts2 = create_default_prompts()
        assert prompts2 == original


class TestMetaLoopInit:
    """Tests for MetaLoop initialization and basic operations."""

    def test_init_with_default_config(self, tmp_path):
        from autoresearch.infrastructure.metaloop import MetaLoop

        meta = MetaLoop()
        assert meta.iteration == 0
        assert meta.prompts == {}
        assert meta.modifications == []
        assert meta.config.max_iterations == 10

    def test_init_with_custom_config(self, tmp_path):
        from autoresearch.infrastructure.metaloop import MetaConfig, MetaLoop

        prompts_file = tmp_path / "prompts.json"
        mods_file = tmp_path / "mods.json"
        cfg = MetaConfig(
            max_iterations=5,
            prompt_template_file=str(prompts_file),
            modifications_file=str(mods_file),
        )
        meta = MetaLoop(config=cfg)
        assert meta.config.max_iterations == 5
        # Should not crash when files don't exist yet
        assert meta.prompts == {}

    def test_loads_existing_prompts(self, tmp_path):
        from autoresearch.infrastructure.metaloop import MetaConfig, MetaLoop

        prompts_file = tmp_path / "prompts.json"
        prompts_file.write_text(json.dumps({
            "hypothesis": [{
                "name": "hypothesis",
                "version": 1,
                "content": "test content",
                "created_at": "2025-01-01T00:00:00",
                "performance": None,
                "notes": "",
            }],
        }))
        cfg = MetaConfig(prompt_template_file=str(prompts_file))
        meta = MetaLoop(config=cfg)
        assert len(meta.prompts) == 1
        assert "hypothesis_v1" in meta.prompts

    def test_loads_existing_modifications(self, tmp_path):
        from autoresearch.infrastructure.metaloop import MetaConfig, MetaLoop

        mods_file = tmp_path / "mods.json"
        mods_file.write_text(json.dumps([{
            "id": "mod_1",
            "type": "prompt",
            "description": "test",
            "old_value": "old",
            "new_value": "new",
            "expected_impact": 0.05,
            "actual_impact": None,
            "status": "pending",
            "timestamp": "2025-01-01T00:00:00",
        }]))
        cfg = MetaConfig(modifications_file=str(mods_file))
        meta = MetaLoop(config=cfg)
        assert len(meta.modifications) == 1
        assert meta.modifications[0].id == "mod_1"


class TestMetaLoopRegisterPrompt:
    """Tests for register_prompt method."""

    def test_register_first_version(self, tmp_path):
        from autoresearch.infrastructure.metaloop import MetaConfig, MetaLoop

        cfg = MetaConfig(
            prompt_template_file=str(tmp_path / "prompts.json"),
            modifications_file=str(tmp_path / "mods.json"),
        )
        meta = MetaLoop(config=cfg)
        pt = meta.register_prompt("test", "hello world")
        assert pt.name == "test"
        assert pt.version == 1
        assert pt.content == "hello world"
        assert "test_v1" in meta.prompts

    def test_register_version_increments(self, tmp_path):
        from autoresearch.infrastructure.metaloop import MetaConfig, MetaLoop

        cfg = MetaConfig(
            prompt_template_file=str(tmp_path / "prompts.json"),
            modifications_file=str(tmp_path / "mods.json"),
        )
        meta = MetaLoop(config=cfg)
        meta.register_prompt("test", "v1 content")
        pt2 = meta.register_prompt("test", "v2 content")
        assert pt2.version == 2
        assert pt2.content == "v2 content"
        assert len(meta.prompts) == 2

    def test_register_other_prompt_starts_version_one(self, tmp_path):
        from autoresearch.infrastructure.metaloop import MetaConfig, MetaLoop

        cfg = MetaConfig(
            prompt_template_file=str(tmp_path / "prompts.json"),
            modifications_file=str(tmp_path / "mods.json"),
        )
        meta = MetaLoop(config=cfg)
        meta.register_prompt("hypothesis", "first")
        pt = meta.register_prompt("evaluation", "other")
        assert pt.version == 1
        assert pt.name == "evaluation"


class TestMetaLoopEvolveHeuristic:
    """Tests for evolve_prompt with heuristic fallback (no LLM)."""

    EVOLVE_PATCH = "autoresearch.infrastructure.metaloop.ANTHROPIC_AVAILABLE"

    def test_evolve_requires_existing_prompt(self, tmp_path):
        from unittest.mock import patch

        from autoresearch.infrastructure.metaloop import MetaConfig, MetaLoop

        cfg = MetaConfig(
            prompt_template_file=str(tmp_path / "prompts.json"),
            modifications_file=str(tmp_path / "mods.json"),
        )
        meta = MetaLoop(config=cfg)
        with pytest.raises(ValueError, match="No prompt found"):
            meta.evolve_prompt("nonexistent", "too vague", 0.5)

    def test_evolve_with_vague_feedback_adds_specificity(self, tmp_path):
        from unittest.mock import patch

        from autoresearch.infrastructure.metaloop import MetaConfig, MetaLoop

        cfg = MetaConfig(
            prompt_template_file=str(tmp_path / "prompts.json"),
            modifications_file=str(tmp_path / "mods.json"),
        )
        meta = MetaLoop(config=cfg)
        meta.register_prompt("hypothesis", "Analyze the results.")
        with patch(self.EVOLVE_PATCH, False):
            evolved = meta.evolve_prompt("hypothesis", "Too vague, be specific", 0.3)
        assert evolved.version == 2
        assert "Be specific and concrete" in evolved.content
        assert "Evolved from v1" in evolved.notes

    def test_evolve_with_unconstrained_feedback(self, tmp_path):
        from unittest.mock import patch

        from autoresearch.infrastructure.metaloop import MetaConfig, MetaLoop

        cfg = MetaConfig(
            prompt_template_file=str(tmp_path / "prompts.json"),
            modifications_file=str(tmp_path / "mods.json"),
        )
        meta = MetaLoop(config=cfg)
        meta.register_prompt("hypothesis", "Explore all possibilities.")
        with patch(self.EVOLVE_PATCH, False):
            evolved = meta.evolve_prompt(
                "hypothesis", "Too unconstrained, needs limits", 0.4
            )
        assert "Consider resource constraints" in evolved.content

    def test_evolve_creates_modification_record(self, tmp_path):
        from unittest.mock import patch

        from autoresearch.infrastructure.metaloop import MetaConfig, MetaLoop

        cfg = MetaConfig(
            prompt_template_file=str(tmp_path / "prompts.json"),
            modifications_file=str(tmp_path / "mods.json"),
        )
        meta = MetaLoop(config=cfg)
        meta.register_prompt("hypothesis", "Analyze the results.")
        with patch(self.EVOLVE_PATCH, False):
            meta.evolve_prompt("hypothesis", "vague feedback", 0.3)
        assert len(meta.modifications) >= 1
        assert meta.modifications[0].type.value == "prompt"
        assert "modified" in meta.modifications[0].description.lower() or \
               "evolved" in meta.modifications[0].description.lower()


class TestMetaLoopHyperparameter:
    """Tests for hyperparameter change proposals."""

    def test_propose_increase_numeric(self, tmp_path):
        from autoresearch.infrastructure.metaloop import MetaConfig, MetaLoop

        cfg = MetaConfig(
            prompt_template_file=str(tmp_path / "prompts.json"),
            modifications_file=str(tmp_path / "mods.json"),
        )
        meta = MetaLoop(config=cfg)
        mod = meta.propose_hyperparameter_change("learning_rate", 0.001, "increase")
        assert mod.type.value == "hyperparameter"
        assert "learning_rate" in mod.description
        assert "0.001" in mod.description
        assert "0.0011" in mod.new_value or "0.0011" in str(mod.new_value)

    def test_propose_decrease_numeric(self, tmp_path):
        from autoresearch.infrastructure.metaloop import MetaConfig, MetaLoop

        cfg = MetaConfig(
            prompt_template_file=str(tmp_path / "prompts.json"),
            modifications_file=str(tmp_path / "mods.json"),
        )
        meta = MetaLoop(config=cfg)
        mod = meta.propose_hyperparameter_change("batch_size", 64, "decrease")
        assert "57.6" in mod.new_value or "57.6" in str(mod.new_value)

    def test_propose_with_string_value(self, tmp_path):
        from autoresearch.infrastructure.metaloop import MetaConfig, MetaLoop

        cfg = MetaConfig(
            prompt_template_file=str(tmp_path / "prompts.json"),
            modifications_file=str(tmp_path / "mods.json"),
        )
        meta = MetaLoop(config=cfg)
        mod = meta.propose_hyperparameter_change(
            "optimizer", "adam", "increase"
        )
        assert mod.new_value == "adam"


class TestMetaLoopModificationLifecycle:
    """Tests for modification lifecycle: apply, revert, impact."""

    def test_apply_modification(self, tmp_path):
        from autoresearch.infrastructure.metaloop import MetaConfig, MetaLoop

        cfg = MetaConfig(
            prompt_template_file=str(tmp_path / "prompts.json"),
            modifications_file=str(tmp_path / "mods.json"),
        )
        meta = MetaLoop(config=cfg)
        mod = meta.propose_hyperparameter_change("lr", 0.01, "increase")
        assert mod.status == "pending"

        result = meta.apply_modification(mod.id)
        assert result is True
        assert meta.modifications[0].status == "applied"

    def test_apply_nonexistent_modification(self, tmp_path):
        from autoresearch.infrastructure.metaloop import MetaConfig, MetaLoop

        cfg = MetaConfig(
            prompt_template_file=str(tmp_path / "prompts.json"),
            modifications_file=str(tmp_path / "mods.json"),
        )
        meta = MetaLoop(config=cfg)
        assert meta.apply_modification("nonexistent") is False

    def test_revert_modification(self, tmp_path):
        from autoresearch.infrastructure.metaloop import MetaConfig, MetaLoop

        cfg = MetaConfig(
            prompt_template_file=str(tmp_path / "prompts.json"),
            modifications_file=str(tmp_path / "mods.json"),
        )
        meta = MetaLoop(config=cfg)
        mod = meta.propose_hyperparameter_change("lr", 0.01, "increase")
        meta.apply_modification(mod.id)
        result = meta.revert_modification(mod.id)
        assert result is True
        assert meta.modifications[0].status == "reverted"

    def test_record_impact(self, tmp_path):
        from autoresearch.infrastructure.metaloop import MetaConfig, MetaLoop

        cfg = MetaConfig(
            prompt_template_file=str(tmp_path / "prompts.json"),
            modifications_file=str(tmp_path / "mods.json"),
        )
        meta = MetaLoop(config=cfg)
        mod = meta.propose_hyperparameter_change("lr", 0.01, "increase")
        meta.apply_modification(mod.id)
        meta.record_impact(mod.id, 0.15)
        assert meta.modifications[0].actual_impact == 0.15

    def test_record_impact_nonexistent(self, tmp_path):
        from autoresearch.infrastructure.metaloop import MetaConfig, MetaLoop

        cfg = MetaConfig(
            prompt_template_file=str(tmp_path / "prompts.json"),
            modifications_file=str(tmp_path / "mods.json"),
        )
        meta = MetaLoop(config=cfg)
        # Should not raise
        meta.record_impact("nonexistent", 1.0)


class TestMetaLoopAnalysis:
    """Tests for analysis methods."""

    def test_get_successful_modifications_empty(self, tmp_path):
        from autoresearch.infrastructure.metaloop import MetaConfig, MetaLoop

        cfg = MetaConfig(
            prompt_template_file=str(tmp_path / "prompts.json"),
            modifications_file=str(tmp_path / "mods.json"),
        )
        meta = MetaLoop(config=cfg)
        assert meta.get_successful_modifications() == []

    def test_get_successful_modifications_filters(self, tmp_path):
        from autoresearch.infrastructure.metaloop import (
            MetaConfig,
            MetaLoop,
            Modification,
            ModificationType,
        )

        cfg = MetaConfig(
            prompt_template_file=str(tmp_path / "prompts.json"),
            modifications_file=str(tmp_path / "mods.json"),
        )
        meta = MetaLoop(config=cfg)
        # Positive impact
        meta.modifications.append(
            Modification(
                id="mod_1", type=ModificationType.PROMPT,
                description="good", old_value="a", new_value="b",
                expected_impact=0.1, actual_impact=0.2,
            )
        )
        # Negative impact
        meta.modifications.append(
            Modification(
                id="mod_2", type=ModificationType.HYPERPARAMETER,
                description="bad", old_value="c", new_value="d",
                expected_impact=0.1, actual_impact=-0.05,
            )
        )
        # No impact
        meta.modifications.append(
            Modification(
                id="mod_3", type=ModificationType.STRATEGY,
                description="unknown", old_value="e", new_value="f",
                expected_impact=0.1,
            )
        )
        successful = meta.get_successful_modifications()
        assert len(successful) == 1
        assert successful[0].id == "mod_1"

    def test_analyze_patterns_empty(self, tmp_path):
        from autoresearch.infrastructure.metaloop import MetaConfig, MetaLoop

        cfg = MetaConfig(
            prompt_template_file=str(tmp_path / "prompts.json"),
            modifications_file=str(tmp_path / "mods.json"),
        )
        meta = MetaLoop(config=cfg)
        result = meta.analyze_patterns()
        assert result == {"patterns": [], "insights": []}

    def test_analyze_patterns_with_data(self, tmp_path):
        from autoresearch.infrastructure.metaloop import (
            MetaConfig,
            MetaLoop,
            Modification,
            ModificationType,
        )

        cfg = MetaConfig(
            prompt_template_file=str(tmp_path / "prompts.json"),
            modifications_file=str(tmp_path / "mods.json"),
        )
        meta = MetaLoop(config=cfg)
        meta.modifications = [
            Modification(
                id="mod_1", type=ModificationType.PROMPT,
                description="good", old_value="a", new_value="b",
                expected_impact=0.1, actual_impact=0.2,
            ),
            Modification(
                id="mod_2", type=ModificationType.PROMPT,
                description="great", old_value="c", new_value="d",
                expected_impact=0.1, actual_impact=0.3,
            ),
        ]
        result = meta.analyze_patterns()
        assert result["total_successful"] == 2
        assert "prompt" in result["by_type"]
        assert result["by_type"]["prompt"] == 0.25
        assert len(result["insights"]) >= 1


class TestMetaLoopRunIteration:
    """Tests for run_iteration method."""

    def test_run_with_no_prompts(self, tmp_path):
        from autoresearch.infrastructure.metaloop import MetaConfig, MetaLoop

        cfg = MetaConfig(
            max_iterations=10,
            prompt_template_file=str(tmp_path / "prompts.json"),
            modifications_file=str(tmp_path / "mods.json"),
        )
        meta = MetaLoop(config=cfg)
        result = meta.run_iteration(feedback="test", performance=0.5)
        assert result["status"] == "no_prompts"
        assert result["iteration"] == 1

    def test_run_with_prompts_evolves(self, tmp_path):
        from unittest.mock import patch

        from autoresearch.infrastructure.metaloop import MetaConfig, MetaLoop

        cfg = MetaConfig(
            max_iterations=10,
            min_improvement=0.01,
            prompt_template_file=str(tmp_path / "prompts.json"),
            modifications_file=str(tmp_path / "mods.json"),
        )
        meta = MetaLoop(config=cfg)
        meta.register_prompt("hypothesis", "Analyze the data.")
        with patch("autoresearch.infrastructure.metaloop.ANTHROPIC_AVAILABLE", False):
            result = meta.run_iteration(feedback="too vague", performance=0.3)
        assert result["status"] == "evolved"
        assert result["iteration"] == 1
        assert result["prompt_version"] == 2

    def test_converges_when_below_threshold(self, tmp_path):
        from autoresearch.infrastructure.metaloop import (
            MetaConfig,
            MetaLoop,
            Modification,
            ModificationType,
        )

        cfg = MetaConfig(
            max_iterations=10,
            min_improvement=0.1,
            prompt_template_file=str(tmp_path / "prompts.json"),
            modifications_file=str(tmp_path / "mods.json"),
        )
        meta = MetaLoop(config=cfg)
        meta.iteration = 2
        # Add a modification with impact below threshold
        meta.modifications = [
            Modification(
                id="mod_1", type=ModificationType.PROMPT,
                description="test", old_value="a", new_value="b",
                expected_impact=0.05, actual_impact=0.02,
            ),
        ]
        result = meta.run_iteration(feedback="test", performance=0.5)
        assert result["status"] == "converged"
        assert "Improvement below threshold" in result["message"]
