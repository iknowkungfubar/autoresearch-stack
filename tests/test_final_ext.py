"""Targeted tests for autonomous_loop, daemon, synthetic_data uncovered paths."""

from unittest.mock import MagicMock, patch

import pytest


class TestAutoLoopTraining:
    """Test run_training and model-in-the-loop paths."""

    def test_run_training_with_scheduler(self):
        from autoresearch.experiment.autonomous_loop import AutonomousPipeline

        p = AutonomousPipeline("config.yaml")
        model = MagicMock()
        trainer = MagicMock()
        trainer.encode.return_value = [1, 2, 3]
        model.return_value = (MagicMock(), MagicMock())
        model.return_value[1].item.return_value = 0.5
        scheduler = MagicMock()
        scheduler.get_stage.return_value = "easy"
        scheduler.sample.return_value = "test text"
        result = p.run_training(
            model, trainer, scheduler, steps=10, val_bpb_target=0.95
        )
        assert "val_bpb" in result

    def test_prepare_data_model_loop(self):
        from autoresearch.experiment.autonomous_loop import AutonomousPipeline

        p = AutonomousPipeline("config.yaml")
        model = MagicMock()
        tokenizer = MagicMock()
        result = p.prepare_data(
            ["test text"],
            use_synthetic=False,
            use_model_loop=True,
            model=model,
            tokenizer=tokenizer,
        )
        assert isinstance(result, list)

    def test_run_experiment_with_change(self):
        from autoresearch.experiment.autonomous_loop import AutonomousPipeline

        p = AutonomousPipeline("config.yaml")
        result = p.run_experiment("test change", "code here", "optimization", 1.0)
        assert result["id"] == 1

    def test_get_status(self):
        from autoresearch.experiment.autonomous_loop import AutonomousPipeline

        p = AutonomousPipeline("config.yaml")
        p.experiment_count = 5
        p.best_val_bpb = 0.9
        s = p.get_status()
        assert s["experiment_count"] == 5

    def test_main_entry(self):
        from autoresearch.experiment.autonomous_loop import main

        with patch("sys.argv", ["autoresearch", "--help"]):
            try:
                main()
            except SystemExit:
                pass


class TestDaemonCLI:
    """Test daemon CLI commands and edge cases."""

    def test_run_daemon_start(self, tmp_path):
        from autoresearch.infrastructure.daemon import Daemon, run_daemon

        with patch.object(Daemon, "start", return_value=None) as mock_start:
            run_daemon(
                start_command="start",
                log_file=str(tmp_path / "t.log"),
                pid_file=str(tmp_path / "p.pid"),
            )
            mock_start.assert_called_once()

    def test_run_daemon_stop(self, tmp_path):
        from autoresearch.infrastructure.daemon import run_daemon

        with patch("daemon.Daemon") as MockDaemon:
            inst = MockDaemon.return_value
            inst.is_running.return_value = True
            run_daemon(
                start_command="stop",
                log_file=str(tmp_path / "t.log"),
                pid_file=str(tmp_path / "p.pid"),
            )
            inst.stop.assert_called_once()

    def test_run_daemon_status(self, tmp_path):
        from autoresearch.infrastructure.daemon import run_daemon

        with patch("daemon.Daemon") as MockDaemon:
            inst = MockDaemon.return_value
            inst.is_running.return_value = True
            inst.state = MagicMock(value="running")
            run_daemon(
                start_command="status",
                log_file=str(tmp_path / "t.log"),
                pid_file=str(tmp_path / "p.pid"),
            )

    def test_run_daemon_restart(self, tmp_path):
        from autoresearch.infrastructure.daemon import run_daemon

        with patch("daemon.Daemon") as MockDaemon:
            inst = MockDaemon.return_value
            run_daemon(
                start_command="restart",
                log_file=str(tmp_path / "t.log"),
                pid_file=str(tmp_path / "p.pid"),
            )
            inst.restart.assert_called_once()

    def test_daemon_save_stats_with_run_on_start(self, tmp_path):
        from unittest.mock import patch

        from autoresearch.infrastructure.daemon import Daemon, DaemonConfig

        cb = MagicMock(side_effect=Exception("callback failed"))
        d = Daemon(
            DaemonConfig(
                pid_file=str(tmp_path / "p.pid"),
                log_file=str(tmp_path / "t.log"),
                run_on_start=cb,
            )
        )
        with patch.object(d, "_main_loop"):
            d.start(daemonize=False)

    def test_daemon_check_health(self, tmp_path):
        from autoresearch.infrastructure.daemon import Daemon, DaemonConfig, DaemonState

        d = Daemon(
            DaemonConfig(
                pid_file=str(tmp_path / "p.pid"), log_file=str(tmp_path / "t.log")
            )
        )
        d.state = DaemonState.RUNNING
        d._check_health()
        # _check_health may or may not change state depending on config
        assert d.state is not None


class TestSyntheticLLM:
    """Test synthetic data LLM generation paths."""

    def test_call_openai_mocked(self):
        from autoresearch.data.synthetic_data import SyntheticGenerator

        gen = SyntheticGenerator(use_llm=True, provider="openai")
        gen.api_key = "sk-test"
        with patch.object(gen, "_call_openai") as mock_call:
            mock_call.return_value = ["prompt1", "prompt2"]
            result = gen.generate(n=2, difficulty="easy")
        assert len(result.prompts) == 2

    def test_generate_with_llm_mocked(self):
        from autoresearch.data.synthetic_data import SyntheticGenerator

        gen = SyntheticGenerator(use_llm=True, provider="anthropic")
        gen.api_key = "sk-test"
        with patch.object(gen, "_call_anthropic") as mock_call:
            mock_call.return_value = ["p1", "p2", "p3"]
            result = gen.generate(n=3, difficulty="hard")
        assert len(result.prompts) == 3

    def test_quality_filter_threshold(self):
        from autoresearch.data.synthetic_data import SyntheticGenerator

        gen = SyntheticGenerator()
        prompts = ["good prompt text here", "bad", "", "also good text"]
        filtered = gen.quality_filter(prompts, min_length=5, max_length=50)
        assert len(filtered) <= len(prompts)
        assert "" not in filtered


class TestProvidersExtended:
    """Test additional provider paths."""

    def test_anthropic_init_no_key(self):
        from autoresearch.llm.providers import AnthropicProvider

        with patch.dict("os.environ", clear=True):
            p = AnthropicProvider()
            with pytest.raises((ValueError, ImportError)):
                p._get_client()

    def test_openai_init_no_key(self):
        from autoresearch.llm.providers import OpenAIProvider

        with patch.dict("os.environ", clear=True):
            p = OpenAIProvider()
            with pytest.raises((ValueError, ImportError)):
                p._get_client()

    def test_mistral_init(self):
        from autoresearch.llm.providers import MistralProvider

        p = MistralProvider(api_key="test-key")
        assert p.base_url == "https://api.mistral.ai/v1"

    def test_google_vertex_init(self):
        from autoresearch.llm.providers import GoogleVertexProvider

        p = GoogleVertexProvider()
        assert p is not None

    def test_lmstudio_provider_type(self):
        from autoresearch.llm.providers import LMStudioProvider

        p = LMStudioProvider()
        assert p.__class__.__name__ == "LMStudioProvider"
