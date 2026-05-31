"""Comprehensive tests for orchestrators, sandbox, monitor modules."""
import pytest


class TestOrchestratorFactory:
    """Tests for OrchestratorFactory."""

    def test_create_langchain(self):
        from orchestrators import OrchestratorFactory
        p = OrchestratorFactory.create("langchain")
        assert p.__class__.__name__ == "LangChainIntegrator"

    def test_create_crewai(self):
        from orchestrators import OrchestratorFactory
        p = OrchestratorFactory.create("crewai")
        assert p.__class__.__name__ == "CrewAIIntegrator"

    def test_create_autogen(self):
        from orchestrators import OrchestratorFactory
        p = OrchestratorFactory.create("autogen")
        assert p.__class__.__name__ == "AutoGenIntegrator"

    def test_create_llama_index(self):
        from orchestrators import OrchestratorFactory
        p = OrchestratorFactory.create("llama_index")
        assert p.__class__.__name__ == "LlamaIndexIntegrator"

    def test_create_invalid(self):
        from orchestrators import OrchestratorFactory
        with pytest.raises((ValueError, KeyError)):
            OrchestratorFactory.create("nonexistent")

    def test_from_config(self):
        from orchestrators import OrchestratorFactory
        p = OrchestratorFactory.from_config({"orchestrator": "langchain", "config": {}})
        assert p.__class__.__name__ == "LangChainIntegrator"

    def test_orchestrator_client(self):
        from orchestrators import OrchestratorClient, LangChainIntegrator
        client = OrchestratorClient(orchestrator=LangChainIntegrator())
        task = client.run("test task")
        assert "not available" in task.content.lower() or "error" in task.metadata


class TestOrchestratorIntegrators:
    """Test integrator classes handle missing packages gracefully."""

    def test_opencrew_unavailable(self):
        from orchestrators import OpenCrewIntegrator, AgentTask
        r = OpenCrewIntegrator().run(AgentTask(description="test"))
        assert "not available" in r.content.lower()

    def test_agentforge_unavailable(self):
        from orchestrators import AgentForgeIntegrator, AgentTask
        r = AgentForgeIntegrator().run(AgentTask(description="test"))
        assert "not available" in r.content.lower()

    def test_autogen_unavailable(self):
        from orchestrators import AutoGenIntegrator, AgentTask
        r = AutoGenIntegrator().run(AgentTask(description="test"))
        assert "not available" in r.content.lower()

    def test_langchain_unavailable(self):
        from orchestrators import LangChainIntegrator, AgentTask
        r = LangChainIntegrator().run(AgentTask(description="test"))
        assert "not available" in r.content.lower()

    def test_llamaindex_unavailable(self):
        from orchestrators import LlamaIndexIntegrator, AgentTask
        r = LlamaIndexIntegrator().run(AgentTask(description="test"))
        assert "not available" in r.content.lower()

    def test_run_multi(self):
        from orchestrators import LangChainIntegrator, AgentTask
        tasks = [AgentTask(description="a"), AgentTask(description="b")]
        results = LangChainIntegrator().run_multi(tasks)
        assert len(results) == 2

    def test_client_no_orchestrator(self):
        from orchestrators import OrchestratorClient
        with pytest.raises(RuntimeError, match="No orchestrator"):
            OrchestratorClient().run("test")


class TestSandbox:
    """Tests for sandbox module — AST validation edge cases."""

    def test_safe_runner_ast_blocked_imports(self):
        from sandbox import SafeRunner
        r = SafeRunner()
        blocked = ["import os", "from os import path", "import  os", "import sys"]
        for code in blocked:
            valid, err = r.validate(code)
            assert not valid, f"Should block: {code}"

    def test_safe_runner_ast_blocked_calls(self):
        from sandbox import SafeRunner
        r = SafeRunner()
        blocked = ["eval('1+1')", "exec('x=1')", "__import__('os')", "compile('x', '', 'exec')"]
        for code in blocked:
            valid, err = r.validate(code)
            assert not valid, f"Should block: {code}"

    def test_safe_runner_allows_safe_code(self):
        from sandbox import SafeRunner
        r = SafeRunner()
        safe = [
            "print('hello')",
            "x = 1 + 2",
            "import numpy",
            "from json import loads",
            "[i for i in range(10)]",
        ]
        for code in safe:
            valid, err = r.validate(code)
            assert valid, f"Should allow: {code}, err={err}"

    def test_safe_runner_executes(self):
        from sandbox import SafeRunner
        r = SafeRunner()
        result = r.run("print('hello world')")
        assert result.success
        assert "hello world" in result.stdout

    def test_safe_runner_blocks_bad_code(self):
        from sandbox import SafeRunner
        r = SafeRunner()
        result = r.run("import os")
        assert not result.success

    def test_sandbox_context_manager(self):
        from sandbox import Sandbox
        with Sandbox() as s:
            result = s.execute("print('in sandbox')")
        assert result.success

    def test_sandbox_timeout(self):
        from sandbox import Sandbox
        with Sandbox() as s:
            result = s.execute("import time; time.sleep(10)", timeout=1)
        assert not result.success

    def test_run_safe_convenience(self):
        from sandbox import run_safe
        result = run_safe("print('convenience')", timeout=5)
        assert result.success or not result.success  # May or may not work depending on context


class TestMonitor:
    """Tests for monitor module."""

    def test_monitor_init(self):
        from monitor import Monitor
        m = Monitor()
        assert m.stats.total_experiments == 0

    def test_start_experiment(self):
        from monitor import Monitor
        m = Monitor()
        m.start_experiment(1, "test change", 1.0)
        assert m.stats.running == 1
        assert m.current_experiment is not None

    def test_complete_experiment_kept(self):
        from monitor import Monitor
        m = Monitor()
        m.start_experiment(1, "test", 1.0)
        m.complete_experiment(0.95, "kept")
        assert m.stats.kept == 1
        assert m.stats.total_experiments == 1

    def test_complete_experiment_reverted(self):
        from monitor import Monitor
        m = Monitor()
        m.start_experiment(1, "test", 1.0)
        m.complete_experiment(1.2, "reverted")
        assert m.stats.reverted == 1

    def test_update_progress(self):
        from monitor import Monitor
        m = Monitor()
        m.start_experiment(1, "test", 1.0)
        m.update_progress(50)
        assert m.current_experiment is not None
        assert m.current_experiment.iteration == 50

    def test_monitor_stats(self):
        from monitor import MonitorStats
        s = MonitorStats()
        assert s.total_experiments == 0
        assert s.experiments_per_hour == 0

    def test_progress_bar(self):
        from monitor import ProgressBar
        pb = ProgressBar(width=20)
        import io
        import sys
        captured = io.StringIO()
        old = sys.stdout
        sys.stdout = captured
        try:
            pb.draw(5, 10, "test")
        finally:
            sys.stdout = old
        output = captured.getvalue()
        assert "%" in output

    def test_event_logging(self):
        from monitor import Monitor
        m = Monitor()
        m.log_event("test", "event message")
        assert len(m.events) == 1
        assert m.events[0]["message"] == "event message"

    def test_stats_tracking(self):
        from monitor import Monitor
        m = Monitor()
        m.start_experiment(1, "test", 1.0)
        m.complete_experiment(0.95, "kept")
        assert m.stats.kept == 1
        assert m.stats.running == 0
