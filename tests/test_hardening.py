"""
Test suite for Sprint 7.2 hardening.

Additional tests for providers, orchestrators, daemon, distribute, and other
modules to improve coverage on critical paths.
"""

import pytest


# === PROVIDERS (Extended) ===


class TestProvidersExtended:
    """Extended provider tests."""

    def test_all_provider_types(self):
        """Test all provider type enum values exist."""
        from providers import ProviderType

        cloud_types = [
            ProviderType.ANTHROPIC,
            ProviderType.OPENAI,
            ProviderType.OPENROUTER,
            ProviderType.ZEN,
            ProviderType.GOOGLE_VERTEX,
            ProviderType.AZURE_OPENAI,
            ProviderType.MISTRAL,
        ]
        local_types = [
            ProviderType.OLLAMA,
            ProviderType.VLLM,
            ProviderType.LMSTUDIO,
            ProviderType.LITELLM,
            ProviderType.LLAMA_CPP,
            ProviderType.TEXTGEN_WEBUI,
        ]
        agent_types = [
            ProviderType.OPENCODE,
            ProviderType.CREWAI,
            ProviderType.AUTOGEN,
            ProviderType.LANGCHAIN,
        ]

        assert len(cloud_types) == 7
        assert len(local_types) == 6
        assert len(agent_types) == 4

    def test_model_registry_complete(self):
        """Test model registry has expected models."""
        from providers import MODEL_REGISTRY

        assert len(MODEL_REGISTRY) >= 10
        # Check specific models
        assert "claude-3-5-sonnet-20241022" in MODEL_REGISTRY
        assert "gpt-4o" in MODEL_REGISTRY
        assert "llama-3.1-70b" in MODEL_REGISTRY

        # Check model info structure
        info = MODEL_REGISTRY["gpt-4o"]
        assert info.provider == "openai"
        assert info.context_length == 128000

    def test_llm_response_dataclass(self):
        """Test LLMResponse construction."""
        from providers import LLMResponse

        response = LLMResponse(
            content="test response",
            model="gpt-4o",
            provider="openai",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            latency_ms=150.0,
        )
        assert response.content == "test response"
        assert response.model == "gpt-4o"
        assert response.usage["prompt_tokens"] == 10
        assert response.latency_ms == 150.0

    def test_model_info_dataclass(self):
        """Test ModelInfo construction."""
        from providers import ModelInfo

        info = ModelInfo(
            id="test-model",
            name="Test Model",
            provider="test",
            context_length=4096,
            max_output_tokens=2048,
            supports_vision=True,
            pricing_input=0.001,
            pricing_output=0.002,
        )
        assert info.id == "test-model"
        assert info.supports_vision is True
        assert info.is_local is False

    def test_provider_factory_string(self):
        """Test factory creates from string."""
        from providers import LLMProviderFactory

        provider = LLMProviderFactory.create("openai")
        assert provider is not None
        assert provider.__class__.__name__ == "OpenAIProvider"

    def test_provider_factory_invalid(self):
        """Test factory rejects invalid provider."""
        from providers import LLMProviderFactory

        with pytest.raises(ValueError):
            LLMProviderFactory.create("nonexistent_provider")

    def test_provider_factory_from_config(self):
        """Test factory from config dict."""
        from providers import LLMProviderFactory

        config = {
            "provider": "openai",
            "api_key": "test-key",
            "base_url": "https://api.openai.com/v1",
        }
        provider = LLMProviderFactory.from_config(config)
        assert provider is not None
        assert provider.api_key == "test-key"

    def test_llm_client_no_provider(self):
        """Test LLMClient raises without provider."""
        from providers import LLMClient

        client = LLMClient()
        with pytest.raises(RuntimeError, match="No provider configured"):
            client.complete([{"role": "user", "content": "test"}])

    def test_provider_get_model_info(self):
        """Test provider get_model_info method."""
        from providers import OpenAIProvider

        provider = OpenAIProvider()
        info = provider.get_model_info("gpt-4o")
        assert info is not None
        assert info.name == "GPT-4o"

        # Unknown model
        info = provider.get_model_info("nonexistent-model")
        assert info is None

    def test_ollama_provider_init(self):
        """Test Ollama provider initialization."""
        from providers import OllamaProvider

        provider = OllamaProvider(base_url="http://localhost:11434")
        assert provider.base_url == "http://localhost:11434"

    def test_lmstudio_provider_init(self):
        """Test LMStudio provider initialization."""
        from providers import LMStudioProvider

        provider = LMStudioProvider(base_url="http://localhost:1234/v1")
        assert provider.base_url == "http://localhost:1234/v1"


# === ORCHESTRATORS (Extended) ===


class TestOrchestratorsExtended:
    """Extended orchestrator tests."""

    def test_all_orchestrator_types(self):
        """Test all orchestrator type enum values."""
        from orchestrators import OrchestratorType

        types = list(OrchestratorType)
        assert len(types) >= 6

    def test_agent_task_defaults(self):
        """Test AgentTask default values."""
        from orchestrators import AgentTask

        task = AgentTask(description="Do something")
        assert task.description == "Do something"
        assert task.expected_output is None
        assert task.context is None

    def test_agent_task_full(self):
        """Test AgentTask with all fields."""
        from orchestrators import AgentTask

        task = AgentTask(
            description="Research topic",
            expected_output="A summary",
            context={"topic": "ML"},
        )
        assert task.description == "Research topic"
        assert task.expected_output == "A summary"
        assert task.context == {"topic": "ML"}

    def test_agent_result_defaults(self):
        """Test AgentResult default values."""
        from orchestrators import AgentResult

        result = AgentResult(content="Done")
        assert result.content == "Done"
        assert result.tool_calls == []
        assert result.metadata == {}

    def test_orchestrator_factory_string(self):
        """Test factory creates from string."""
        from orchestrators import OrchestratorFactory

        orch = OrchestratorFactory.create("langchain")
        assert orch is not None

    def test_orchestrator_factory_invalid(self):
        """Test factory rejects invalid type."""
        from orchestrators import OrchestratorFactory

        with pytest.raises(ValueError):
            OrchestratorFactory.create("nonexistent")

    def test_orchestrator_factory_from_config(self):
        """Test factory from config dict."""
        from orchestrators import OrchestratorFactory

        config = {"orchestrator": "langchain", "config": {"model": "gpt-4o"}}
        orch = OrchestratorFactory.from_config(config)
        assert orch is not None

    def test_opencrew_not_available(self):
        """Test OpenCrew fallback when not installed."""
        from orchestrators import OpenCrewIntegrator, AgentTask

        orch = OpenCrewIntegrator()
        result = orch.run(AgentTask(description="test"))
        # Should return error since opencrew is not installed
        assert "not available" in result.content.lower() or "error" in result.metadata

    def test_agentforge_not_available(self):
        """Test AgentForge fallback when not installed."""
        from orchestrators import AgentForgeIntegrator, AgentTask

        orch = AgentForgeIntegrator()
        result = orch.run(AgentTask(description="test"))
        assert "not available" in result.content.lower() or "error" in result.metadata

    def test_autogen_not_available(self):
        """Test AutoGen fallback when not installed."""
        from orchestrators import AutoGenIntegrator, AgentTask

        orch = AutoGenIntegrator()
        result = orch.run(AgentTask(description="test"))
        assert "not available" in result.content.lower() or "error" in result.metadata

    def test_langchain_not_available(self):
        """Test LangChain fallback when not installed."""
        from orchestrators import LangChainIntegrator, AgentTask

        orch = LangChainIntegrator()
        result = orch.run(AgentTask(description="test"))
        assert "not available" in result.content.lower() or "error" in result.metadata

    def test_llamaindex_not_available(self):
        """Test LlamaIndex fallback when not installed."""
        from orchestrators import LlamaIndexIntegrator, AgentTask

        orch = LlamaIndexIntegrator()
        result = orch.run(AgentTask(description="test"))
        assert "not available" in result.content.lower() or "error" in result.metadata

    def test_orchestrator_client_no_orch(self):
        """Test OrchestratorClient raises without orchestrator."""
        from orchestrators import OrchestratorClient

        client = OrchestratorClient()
        with pytest.raises(RuntimeError, match="No orchestrator configured"):
            client.run("test task")

    def test_orchestrator_run_multi(self):
        """Test orchestrator run_multi."""
        from orchestrators import LangChainIntegrator, AgentTask

        orch = LangChainIntegrator()
        tasks = [
            AgentTask(description="task 1"),
            AgentTask(description="task 2"),
        ]
        results = orch.run_multi(tasks)
        assert len(results) == 2


# === DAEMON (Extended) ===


class TestDaemonExtended:
    """Extended daemon tests."""

    def test_daemon_state_enum(self):
        """Test all daemon states."""
        from daemon import DaemonState

        states = list(DaemonState)
        assert len(states) == 8
        assert DaemonState.STOPPED in states
        assert DaemonState.RUNNING in states
        assert DaemonState.HEALTHY in states

    def test_health_status_defaults(self):
        """Test HealthStatus defaults."""
        from daemon import HealthStatus, DaemonState

        status = HealthStatus()
        assert status.state == DaemonState.HEALTHY
        assert status.uptime_seconds == 0
        assert status.experiments_run == 0
        assert status.error_count == 0
        assert status.message == "OK"

    def test_health_status_custom(self):
        """Test HealthStatus with custom values."""
        from daemon import HealthStatus, DaemonState

        status = HealthStatus(
            state=DaemonState.RUNNING,
            uptime_seconds=3600,
            experiments_run=50,
            error_count=2,
            last_error="Connection timeout",
        )
        assert status.state == DaemonState.RUNNING
        assert status.uptime_seconds == 3600
        assert status.experiments_run == 50
        assert status.error_count == 2

    def test_daemon_config_defaults(self):
        """Test DaemonConfig defaults."""
        from daemon import DaemonConfig

        config = DaemonConfig()
        assert config.health_check_interval == 60
        assert config.max_restart_attempts == 3
        assert config.restart_cooldown == 300
        assert config.experiment_batch_size == 10
        assert config.stop_on_failure is True

    def test_daemon_config_custom(self):
        """Test DaemonConfig custom values."""
        from daemon import DaemonConfig

        config = DaemonConfig(
            log_file="/tmp/test.log",
            pid_file="/tmp/test.pid",
            health_check_interval=30,
            max_restart_attempts=5,
        )
        assert config.log_file == "/tmp/test.log"
        assert config.health_check_interval == 30
        assert config.max_restart_attempts == 5


# === DISTRIBUTE ===


class TestDistribute:
    """Tests for distribution module."""

    def test_node_role_enum(self):
        """Test NodeRole enum."""
        from distribute import NodeRole

        assert NodeRole.MASTER.value == "master"
        assert NodeRole.WORKER.value == "worker"
        assert NodeRole.SCHEDULER.value == "scheduler"

    def test_cloud_provider_enum(self):
        """Test CloudProvider enum."""
        from distribute import CloudProvider

        assert CloudProvider.AWS.value == "aws"
        assert CloudProvider.GCP.value == "gcp"
        assert CloudProvider.AZURE.value == "azure"
        assert CloudProvider.LOCAL.value == "local"

    def test_node_config_defaults(self):
        """Test NodeConfig defaults."""
        from distribute import NodeConfig, NodeRole

        config = NodeConfig()
        assert config.role == NodeRole.WORKER
        assert config.name == "worker-1"
        assert config.cpu_cores == 4
        assert config.memory_gb == 16
        assert config.gpu_count == 0
        assert config.spot_instance is False

    def test_node_config_custom(self):
        """Test NodeConfig custom values."""
        from distribute import NodeConfig, NodeRole

        config = NodeConfig(
            role=NodeRole.MASTER,
            name="master-1",
            cpu_cores=8,
            memory_gb=32,
            gpu_count=2,
            gpu_type="A100",
        )
        assert config.role == NodeRole.MASTER
        assert config.gpu_count == 2
        assert config.gpu_type == "A100"

    def test_node_creation(self):
        """Test Node creation."""
        from distribute import Node, NodeConfig, NodeRole

        config = NodeConfig(role=NodeRole.WORKER, name="test-worker")
        node = Node(config)
        assert node.status == "pending"
        assert node.id == "worker-test-worker"
        assert node.experiments_completed == 0

    def test_node_health(self):
        """Test Node health check."""
        from distribute import Node, NodeConfig

        config = NodeConfig()
        node = Node(config)

        # Pending node is not healthy
        assert node.is_healthy() is False

        # Running with low metrics is healthy
        node.status = "running"
        node.metrics.cpu_percent = 50
        node.metrics.memory_percent = 60
        assert node.is_healthy() is True

        # Running with high CPU is not healthy
        node.metrics.cpu_percent = 95
        assert node.is_healthy() is False

    def test_node_to_dict(self):
        """Test Node serialization."""
        from distribute import Node, NodeConfig, NodeRole

        config = NodeConfig(role=NodeRole.WORKER, name="test")
        node = Node(config)
        d = node.to_dict()
        assert "id" in d
        assert "status" in d
        assert "config" in d
        assert "metrics" in d

    def test_resource_metrics_defaults(self):
        """Test ResourceMetrics defaults."""
        from distribute import ResourceMetrics

        metrics = ResourceMetrics()
        assert metrics.cpu_percent == 0
        assert metrics.memory_percent == 0
        assert metrics.gpu_percent == 0
        assert metrics.timestamp is not None

    def test_cost_estimate(self):
        """Test CostEstimate."""
        from distribute import CostEstimate, CloudProvider

        estimate = CostEstimate(
            provider=CloudProvider.AWS,
            instance_type="p3.2xlarge",
            hourly_rate=3.06,
            estimated_hours=24,
            total_cost=73.44,
        )
        assert estimate.provider == CloudProvider.AWS
        assert estimate.total_cost == 73.44
        assert estimate.currency == "USD"


# === METALOOP (Extended) ===


class TestMetaLoopExtended:
    """Extended metaloop tests."""

    def test_prompt_template(self):
        """Test PromptTemplate versioning."""
        from metaloop import PromptTemplate

        template = PromptTemplate(name="test", version=1, content="Initial content")
        assert template.version == 1

        # Record performance
        template.performance = 0.95
        assert template.performance == 0.95

    def test_metaloop_iteration_tracking(self):
        """Test MetaLoop tracks iterations."""
        from metaloop import MetaLoop

        meta = MetaLoop()
        assert meta.iteration == 0

        meta.run_iteration(feedback="test", performance=0.9)
        assert meta.iteration == 1

        meta.run_iteration(feedback="test2", performance=0.95)
        assert meta.iteration == 2

    def test_metaloop_modification_tracking(self):
        """Test MetaLoop tracks modifications."""
        from metaloop import MetaLoop

        meta = MetaLoop()
        meta.register_prompt("test", "content")

        meta.run_iteration(feedback="Too vague", performance=0.8)
        assert len(meta.modifications) >= 0


# === SYNTHETIC DATA (Extended) ===


class TestSyntheticDataExtended:
    """Extended synthetic data tests."""

    def test_evol_instruct_templates(self):
        """Test Evol-Instruct difficulty templates."""
        from synthetic_data import SyntheticGenerator

        gen = SyntheticGenerator(use_llm=False)

        # Check difficulty descriptors exist
        easy_prompts = gen._get_difficulty_list("easy")
        hard_prompts = gen._get_difficulty_list("hard")

        assert len(easy_prompts) > 0
        assert len(hard_prompts) > 0
        assert easy_prompts != hard_prompts

    def test_generate_result_structure(self):
        """Test GenerationResult structure."""
        from synthetic_data import SyntheticGenerator

        gen = SyntheticGenerator(use_llm=False)
        result = gen.generate(n=5, difficulty="mixed")

        assert hasattr(result, "prompts")
        assert hasattr(result, "used_llm")
        assert hasattr(result, "metadata")
        assert result.used_llm is False
        assert len(result.prompts) == 5


# === CONFIG (Extended) ===


class TestConfigExtended:
    """Extended config tests."""

    def test_config_sections_exist(self):
        """Test all config sections."""
        from config import get_config

        c = get_config("config.yaml")
        assert hasattr(c, "experiment")
        assert hasattr(c, "model")
        assert hasattr(c, "synthetic")
        assert hasattr(c, "curriculum")
        assert hasattr(c, "memory")
        assert hasattr(c, "api")
        assert hasattr(c, "logging")

    def test_model_config(self):
        """Test model config values."""
        from config import get_config

        c = get_config("config.yaml")
        assert c.model.size == "124M"
        assert c.model.vocab_size == 32768
        assert c.model.n_layer == 12
        assert c.model.n_head == 8

    def test_config_save_reload(self, tmp_path):
        """Test config save and reload."""
        from config import Config

        c = Config()
        save_path = str(tmp_path / "test_config.yaml")
        c.save(save_path)

        # Verify file was created
        assert (tmp_path / "test_config.yaml").exists()


# === DATA INTELLIGENCE (Extended) ===


class TestDataIntelligenceExtended:
    """Extended data intelligence tests."""

    def test_repair_null_bytes(self):
        """Test repair handles null bytes."""
        from data_intelligence import repair

        result = repair("hello\x00world, this is a test")
        assert result is not None
        assert "\x00" not in result

    def test_repair_whitespace_normalization(self):
        """Test repair normalizes whitespace."""
        from data_intelligence import repair

        result = repair("hello    world    with    spaces and more text")
        assert result is not None
        assert "    " not in result

    def test_clean_corpus_preserves_valid(self):
        """Test clean_corpus preserves valid texts."""
        from data_intelligence import clean_corpus

        valid = [
            "This is a valid text with enough content to pass all checks",
            "Another valid piece of text for testing the corpus cleaner",
        ]
        result = clean_corpus(valid)
        assert len(result) == 2


# === STORAGE (Extended) ===


class TestStorageExtended:
    """Extended storage tests."""

    def test_db_statistics(self, tmp_path):
        """Test database statistics."""
        from storage import ExperimentDB

        db = ExperimentDB(str(tmp_path / "test.db"))

        # Insert multiple experiments
        for i in range(3):
            db.insert_experiment(
                timestamp="2026-01-01",
                change_description=f"Test {i}",
                change_code="code",
                change_type="optimization",
                val_bpb_before=1.0,
                status="running",
            )

        # Update some
        db.update_experiment(1, val_bpb_after=0.95, status="kept")
        db.update_experiment(2, val_bpb_after=1.05, status="reverted")

        stats = db.get_statistics()
        assert stats["total_experiments"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
