"""
Test suite for v5.0+ modules.

Tests for figures, stats, paper, peer_review, metaloop, daemon,
providers, and orchestrators.
"""

import pytest


# Test figures module
class TestFigures:
    """Test figure generation module."""

    def test_figure_generator_init(self):
        """Test FigureGenerator initialization."""
        from figures import FigureGenerator

        gen = FigureGenerator()
        assert gen is not None
        assert gen.config.dpi == 100

    def test_figure_config(self):
        """Test FigureConfig."""
        from figures import FigureConfig

        config = FigureConfig(width=12, height=8, dpi=150)
        assert config.width == 12
        assert config.height == 8
        assert config.dpi == 150


# Test stats module
class TestStats:
    """Test statistics module."""

    def test_summary_statistics_init(self):
        """Test SummaryStatistics initialization."""
        from stats import SummaryStatistics

        stats = SummaryStatistics()
        assert stats is not None
        assert stats.experiments == []

    def test_calculate(self):
        """Test calculate method."""
        from stats import SummaryStatistics

        experiments = [
            {"id": 1, "status": "kept", "val_bpb_after": 0.95},
            {"id": 2, "status": "reverted", "val_bpb_after": 0.97},
            {"id": 3, "status": "kept", "val_bpb_after": 0.92},
        ]

        stats = SummaryStatistics(experiments, baseline=1.0)
        result = stats.calculate()

        assert result.total == 3
        assert result.kept == 2
        assert result.reverted == 1
        assert result.best_val_bpb == 0.92

    def test_to_dict(self):
        """Test to_dict serialization."""
        from stats import SummaryStatistics

        experiments = [{"id": 1, "status": "kept", "val_bpb_after": 0.95}]
        stats = SummaryStatistics(experiments)
        d = stats.to_dict()

        assert "total" in d
        assert "kept" in d

    def test_summary_text(self):
        """Test text summary generation."""
        from stats import SummaryStatistics

        experiments = [
            {"id": 1, "status": "kept", "val_bpb_after": 0.95},
        ]
        stats = SummaryStatistics(experiments)
        text = stats.summary_text()

        assert "Total" in text
        assert "Kept" in text


# Test paper module
class TestPaper:
    """Test paper generation module."""

    def test_paper_init(self):
        """Test Paper initialization."""
        from paper import Paper

        paper = Paper()
        assert paper is not None

    def test_paper_sections(self):
        """Test paper has sections."""
        from paper import Paper

        paper = Paper()
        assert "abstract" in paper.sections
        assert "introduction" in paper.sections
        assert "method" in paper.sections

    def test_paper_render(self):
        """Test paper rendering."""
        from paper import Paper

        paper = Paper()
        content = paper.render()
        assert "# Autonomous Research" in content
        assert "## Introduction" in content

    def test_generate_paper_from_experiments(self):
        """Test paper generation from experiments."""
        from paper import generate_paper_from_experiments

        experiments = [
            {"id": 1, "status": "kept", "val_bpb_before": 1.0, "val_bpb_after": 0.95},
        ]

        paper = generate_paper_from_experiments(experiments)
        assert paper is not None


# Test peer review module
class TestPeerReview:
    """Test peer review simulation module."""

    def test_simulator_init(self):
        """Test PeerReviewSimulator initialization."""
        from peer_review import PeerReviewSimulator, ReviewSimulationConfig

        config = ReviewSimulationConfig(num_reviewers=3, seed=42)
        simulator = PeerReviewSimulator(config)
        assert simulator is not None

    def test_simulate_review_round(self):
        """Test review round simulation."""
        from peer_review import PeerReviewSimulator, ReviewSimulationConfig

        config = ReviewSimulationConfig(num_reviewers=2, seed=42)
        simulator = PeerReviewSimulator(config)

        reviews = simulator.simulate_review_round(
            paper_title="Test Paper",
            paper_content="This is a test about ML and transformers.",
        )

        assert len(reviews) == 2
        for review in reviews:
            assert review.verdict is not None
            assert review.score >= 1

    def test_consensus(self):
        """Test consensus calculation."""
        from peer_review import PeerReviewSimulator, ReviewSimulationConfig

        config = ReviewSimulationConfig(num_reviewers=3, seed=42)
        simulator = PeerReviewSimulator(config)

        reviews = simulator.simulate_review_round(
            paper_title="Test",
            paper_content="ML test",
        )

        consensus = simulator.get_consensus(reviews)
        assert "verdicts" in consensus
        assert "average_score" in consensus

    def test_review_report(self):
        """Test review report generation."""
        from peer_review import PeerReviewSimulator, ReviewSimulationConfig

        config = ReviewSimulationConfig(num_reviewers=2, seed=42)
        simulator = PeerReviewSimulator(config)

        reviews = simulator.simulate_review_round(
            paper_title="Test Paper",
            paper_content="ML test",
        )

        report = simulator.generate_review_report("Test Paper", reviews)
        assert "Peer Review Report" in report


# Test metaloop module
class TestMetaLoop:
    """Test self-modification module."""

    def test_metaloop_init(self):
        """Test MetaLoop initialization."""
        from metaloop import MetaLoop

        meta = MetaLoop()
        assert meta is not None
        assert meta.iteration == 0

    def test_register_prompt(self):
        """Test prompt registration."""
        from metaloop import MetaLoop

        meta = MetaLoop()
        prompt = meta.register_prompt("my_test_prompt", "Test prompt content")

        assert prompt.name == "my_test_prompt"
        assert prompt.version >= 1

    def test_run_iteration(self):
        """Test iteration run."""
        from metaloop import MetaLoop

        meta = MetaLoop()
        meta.register_prompt("test", "Test prompt")

        result = meta.run_iteration(
            feedback="Too vague",
            performance=0.95,
        )

        assert result["status"] in ["evolved", "no_prompts"]


# Test daemon module
class TestDaemon:
    """Test daemon module."""

    def test_daemon_init(self):
        """Test Daemon initialization."""
        from daemon import Daemon, DaemonConfig, DaemonState

        config = DaemonConfig(pid_file="/tmp/test.pid")
        daemon = Daemon(config)
        assert daemon.state == DaemonState.STOPPED

    def test_daemon_config(self):
        """Test DaemonConfig."""
        from daemon import DaemonConfig

        config = DaemonConfig(
            log_file="test.log",
            pid_file="/tmp/test.pid",
            health_check_interval=30,
        )
        assert config.log_file == "test.log"
        assert config.health_check_interval == 30

    def test_health_status(self):
        """Test health status."""
        from daemon import HealthStatus, DaemonState

        status = HealthStatus(state=DaemonState.HEALTHY)
        assert status.state == DaemonState.HEALTHY


# Test providers module
class TestProviders:
    """Test LLM providers module."""

    def test_provider_factory_create(self):
        """Test ProviderFactory.create."""
        from providers import LLMProviderFactory, ProviderType

        # This should create provider without error
        provider = LLMProviderFactory.create(ProviderType.OPENAI)
        assert provider is not None

    def test_provider_types(self):
        """Test ProviderType enum."""
        from providers import ProviderType

        assert ProviderType.ANTHROPIC.value == "anthropic"
        assert ProviderType.OPENAI.value == "openai"
        assert ProviderType.OLLAMA.value == "ollama"

    def test_model_registry(self):
        """Test model registry."""
        from providers import MODEL_REGISTRY

        assert "claude-3-5-sonnet-20241022" in MODEL_REGISTRY
        assert "gpt-4o" in MODEL_REGISTRY
        assert "llama-3.1-70b" in MODEL_REGISTRY

    def test_llm_response(self):
        """Test LLMResponse dataclass."""
        from providers import LLMResponse

        response = LLMResponse(
            content="Hello",
            model="gpt-4o",
            provider="openai",
        )
        assert response.content == "Hello"
        assert response.model == "gpt-4o"


# Test orchestrators module
class TestOrchestrators:
    """Test orchestrator module."""

    def test_orchestrator_types(self):
        """Test OrchestratorType enum."""
        from orchestrators import OrchestratorType

        assert OrchestratorType.LANGCHAIN.value == "langchain"
        assert OrchestratorType.CREWAI.value == "crewai"

    def test_agent_task(self):
        """Test AgentTask dataclass."""
        from orchestrators import AgentTask

        task = AgentTask(description="Test task")
        assert task.description == "Test task"
        assert task.context is None

    def test_agent_result(self):
        """Test AgentResult dataclass."""
        from orchestrators import AgentResult

        result = AgentResult(content="Test result")
        assert result.content == "Test result"
        assert result.tool_calls == []

    def test_orchestrator_factory(self):
        """Test OrchestratorFactory.create."""
        from orchestrators import (
            OrchestratorFactory,
            OrchestratorType,
            LangChainIntegrator,
        )

        orch = OrchestratorFactory.create(OrchestratorType.LANGCHAIN)
        assert orch is not None
        assert isinstance(orch, LangChainIntegrator)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
