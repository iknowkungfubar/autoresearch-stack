"""Comprehensive tests for daemon module."""


class TestDaemonState:
    """Tests for DaemonState enum."""

    def test_all_states(self):
        from daemon import DaemonState

        assert DaemonState.STOPPED.value == "stopped"
        assert DaemonState.RUNNING.value == "running"
        assert DaemonState.HEALTHY.value == "healthy"
        assert DaemonState.FAILED.value == "failed"

    def test_state_count(self):
        from daemon import DaemonState

        assert len(list(DaemonState)) >= 5


class TestDaemonConfig:
    """Tests for DaemonConfig dataclass."""

    def test_defaults(self):
        from daemon import DaemonConfig

        config = DaemonConfig()
        assert config.health_check_interval == 60
        assert config.max_restart_attempts == 3
        assert config.restart_cooldown == 300
        assert config.experiment_batch_size == 10
        assert config.stop_on_failure is True

    def test_custom_values(self):
        from daemon import DaemonConfig

        config = DaemonConfig(
            log_file="/tmp/test_daemon.log",
            pid_file="/tmp/test_daemon.pid",
            health_check_interval=30,
            max_restart_attempts=5,
            restart_cooldown=100,
            experiment_batch_size=25,
        )
        assert config.log_file == "/tmp/test_daemon.log"
        assert config.pid_file == "/tmp/test_daemon.pid"
        assert config.health_check_interval == 30
        assert config.max_restart_attempts == 5
        assert config.restart_cooldown == 100
        assert config.experiment_batch_size == 25


class TestDaemon:
    """Tests for the Daemon class."""

    def test_init_defaults(self, tmp_path):
        from daemon import Daemon, DaemonConfig, DaemonState

        config = DaemonConfig(log_file=str(tmp_path / "d.log"), pid_file=str(tmp_path / "d.pid"))
        daemon = Daemon(config)
        assert daemon.state == DaemonState.STOPPED
        assert daemon.experiments_run == 0
        assert daemon.error_count == 0
        assert daemon.restart_count == 0

    def test_state_transitions(self, tmp_path):
        from daemon import Daemon, DaemonConfig, DaemonState

        config = DaemonConfig(log_file=str(tmp_path / "d.log"), pid_file=str(tmp_path / "d.pid"))
        daemon = Daemon(config)
        assert daemon.state == DaemonState.STOPPED
        daemon.state = DaemonState.RUNNING
        assert daemon.state == DaemonState.RUNNING
        daemon.state = DaemonState.HEALTHY
        assert daemon.state == DaemonState.HEALTHY
        daemon.state = DaemonState.FAILED
        assert daemon.state == DaemonState.FAILED

    def test_experiment_tracking(self, tmp_path):
        from daemon import Daemon, DaemonConfig

        config = DaemonConfig(log_file=str(tmp_path / "d.log"), pid_file=str(tmp_path / "d.pid"))
        daemon = Daemon(config)
        assert daemon.experiments_run == 0
        daemon.experiments_run = 5
        assert daemon.experiments_run == 5

    def test_error_tracking(self, tmp_path):
        from daemon import Daemon, DaemonConfig

        config = DaemonConfig(log_file=str(tmp_path / "d.log"), pid_file=str(tmp_path / "d.pid"))
        daemon = Daemon(config)
        assert daemon.error_count == 0
        daemon.error_count = 3
        assert daemon.error_count == 3
        assert daemon.last_error is None
        daemon.last_error = "Test error"
        assert daemon.last_error == "Test error"

    def test_restart_count(self, tmp_path):
        from daemon import Daemon, DaemonConfig

        config = DaemonConfig(log_file=str(tmp_path / "d.log"), pid_file=str(tmp_path / "d.pid"))
        daemon = Daemon(config)
        assert daemon.restart_count == 0
        daemon.restart_count = 2
        assert daemon.restart_count == 2

    def test_config_stop_on_failure(self, tmp_path):
        from daemon import DaemonConfig

        config = DaemonConfig(
            log_file=str(tmp_path / "d.log"),
            pid_file=str(tmp_path / "d.pid"),
            stop_on_failure=False,
        )
        assert config.stop_on_failure is False

    def test_config_run_on_start(self, tmp_path):
        from daemon import DaemonConfig

        config = DaemonConfig(
            log_file=str(tmp_path / "d.log"),
            pid_file=str(tmp_path / "d.pid"),
            run_on_start=None,
        )
        assert config.run_on_start is None
