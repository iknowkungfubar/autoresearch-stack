"""Daemon tests — PID, stats, HealthChecker, signals."""

import json
import os
from unittest.mock import MagicMock, patch


class TestDaemonPid:
    def test_write_pid(self, tmp_path):
        from daemon import Daemon, DaemonConfig

        c = DaemonConfig(
            pid_file=str(tmp_path / "p.pid"), log_file=str(tmp_path / "t.log")
        )
        Daemon(c)._write_pid()
        assert (tmp_path / "p.pid").read_text().strip() == str(os.getpid())

    def test_remove_pid(self, tmp_path):
        from daemon import Daemon, DaemonConfig

        (tmp_path / "p.pid").write_text("12345")
        c = DaemonConfig(
            pid_file=str(tmp_path / "p.pid"), log_file=str(tmp_path / "t.log")
        )
        Daemon(c)._remove_pid()
        assert not (tmp_path / "p.pid").exists()

    def test_get_pid(self, tmp_path):
        from daemon import Daemon, DaemonConfig

        (tmp_path / "p.pid").write_text("99999")
        assert (
            Daemon(
                DaemonConfig(
                    pid_file=str(tmp_path / "p.pid"), log_file=str(tmp_path / "t.log")
                )
            )._get_pid()
            == 99999
        )

    def test_get_pid_none(self, tmp_path):
        from daemon import Daemon, DaemonConfig

        assert (
            Daemon(
                DaemonConfig(
                    pid_file=str(tmp_path / "x.pid"), log_file=str(tmp_path / "t.log")
                )
            )._get_pid()
            is None
        )


class TestDaemonRunning:
    def test_not_running_no_pid(self, tmp_path):
        from daemon import Daemon, DaemonConfig

        assert (
            Daemon(
                DaemonConfig(
                    pid_file=str(tmp_path / "x.pid"), log_file=str(tmp_path / "t.log")
                )
            ).is_running()
            is False
        )

    def test_is_running(self, tmp_path):
        from daemon import Daemon, DaemonConfig

        (tmp_path / "p.pid").write_text(str(os.getpid()))
        assert (
            Daemon(
                DaemonConfig(
                    pid_file=str(tmp_path / "p.pid"), log_file=str(tmp_path / "t.log")
                )
            ).is_running()
            is True
        )

    def test_stale_pid(self, tmp_path):
        from daemon import Daemon, DaemonConfig

        (tmp_path / "p.pid").write_text("999999999")
        assert (
            Daemon(
                DaemonConfig(
                    pid_file=str(tmp_path / "p.pid"), log_file=str(tmp_path / "t.log")
                )
            ).is_running()
            is False
        )


class TestDaemonStats:
    def test_save_and_load(self, tmp_path):
        from daemon import Daemon, DaemonConfig

        sf = tmp_path / "stats.json"
        d = Daemon(
            DaemonConfig(
                stats_file=str(sf),
                log_file=str(tmp_path / "t.log"),
                pid_file=str(tmp_path / "p.pid"),
            )
        )
        d._save_stats({"experiments_run": 5})
        assert json.loads(sf.read_text())["experiments_run"] == 5

    def test_load_missing(self, tmp_path):
        from daemon import Daemon, DaemonConfig

        d = Daemon(
            DaemonConfig(
                stats_file=str(tmp_path / "m.json"),
                log_file=str(tmp_path / "t.log"),
                pid_file=str(tmp_path / "p.pid"),
            )
        )
        assert d._load_stats() == {}


class TestDaemonSignals:
    def test_handle_shutdown(self, tmp_path):
        from daemon import Daemon, DaemonConfig

        d = Daemon(
            DaemonConfig(
                pid_file=str(tmp_path / "p.pid"), log_file=str(tmp_path / "t.log")
            )
        )
        d._running = True
        d._handle_shutdown(15, None)
        assert d._running is False

    def test_handle_restart(self, tmp_path):
        from daemon import Daemon, DaemonConfig, DaemonState

        d = Daemon(
            DaemonConfig(
                pid_file=str(tmp_path / "p.pid"), log_file=str(tmp_path / "t.log")
            )
        )
        d.state = DaemonState.RUNNING
        d._handle_restart(1, None)
        assert d.state == DaemonState.RESTARTING


class TestHealthChecker:
    def test_init(self, tmp_path):
        from daemon import DaemonConfig, HealthChecker

        c = HealthChecker(
            DaemonConfig(
                log_file=str(tmp_path / "t.log"), pid_file=str(tmp_path / "t.pid")
            )
        )
        assert c.checks == []

    def test_register_check(self, tmp_path):
        from daemon import DaemonConfig, HealthChecker

        c = HealthChecker(
            DaemonConfig(
                log_file=str(tmp_path / "t.log"), pid_file=str(tmp_path / "t.pid")
            )
        )
        c.register_check("ok", lambda: True)
        assert len(c.checks) == 1

    def test_check_all_healthy(self, tmp_path):
        from daemon import DaemonConfig, DaemonState, HealthChecker

        c = HealthChecker(
            DaemonConfig(
                log_file=str(tmp_path / "t.log"), pid_file=str(tmp_path / "t.pid")
            )
        )
        c.register_check("ok", lambda: True)
        assert c.check_all().state == DaemonState.HEALTHY

    def test_check_all_unhealthy(self, tmp_path):
        from daemon import DaemonConfig, DaemonState, HealthChecker

        c = HealthChecker(
            DaemonConfig(
                log_file=str(tmp_path / "t.log"), pid_file=str(tmp_path / "t.pid")
            )
        )
        c.register_check("bad", lambda: False)
        assert c.check_all().state == DaemonState.UNHEALTHY

    def test_check_all_exception(self, tmp_path):
        from daemon import DaemonConfig, DaemonState, HealthChecker

        def _fail():
            raise RuntimeError("fail")

        c = HealthChecker(
            DaemonConfig(
                log_file=str(tmp_path / "t.log"), pid_file=str(tmp_path / "t.pid")
            )
        )
        c.register_check("err", _fail)
        assert c.check_all().state == DaemonState.UNHEALTHY


class TestDaemonStart:
    def test_start_when_running(self, tmp_path):
        import os

        from daemon import Daemon, DaemonConfig

        (tmp_path / "p.pid").write_text(str(os.getpid()))
        d = Daemon(
            DaemonConfig(
                pid_file=str(tmp_path / "p.pid"), log_file=str(tmp_path / "t.log")
            )
        )
        assert d.start(daemonize=False) is False

    def test_start_with_mocked_loop(self, tmp_path):
        from daemon import Daemon, DaemonConfig, DaemonState

        d = Daemon(
            DaemonConfig(
                pid_file=str(tmp_path / "p.pid"), log_file=str(tmp_path / "t.log")
            )
        )
        with patch.object(d, "_main_loop"), patch("os.chdir"), patch("os.setsid"):
            result = d.start(daemonize=False)
        assert result is True
        assert d.state == DaemonState.RUNNING

    def test_start_fork_fail(self, tmp_path):
        from daemon import Daemon, DaemonConfig

        d = Daemon(
            DaemonConfig(
                pid_file=str(tmp_path / "p.pid"), log_file=str(tmp_path / "t.log")
            )
        )
        with patch("os.fork", side_effect=OSError("fail")):
            assert d.start(daemonize=True) is False

    def test_signal_handlers(self, tmp_path):
        from daemon import Daemon, DaemonConfig

        d = Daemon(
            DaemonConfig(
                pid_file=str(tmp_path / "p.pid"), log_file=str(tmp_path / "t.log")
            )
        )
        with (
            patch.object(d, "_main_loop"),
            patch("os.chdir"),
            patch("os.setsid"),
            patch("signal.signal") as ms,
        ):
            d.start(daemonize=False)
            assert ms.called

    def test_run_on_start_called(self, tmp_path):
        from daemon import Daemon, DaemonConfig

        cb = MagicMock()
        d = Daemon(
            DaemonConfig(
                pid_file=str(tmp_path / "p.pid"),
                log_file=str(tmp_path / "t.log"),
                run_on_start=cb,
            )
        )
        with patch.object(d, "_main_loop"), patch("os.chdir"), patch("os.setsid"):
            d.start(daemonize=False)
        cb.assert_called_once()
