"""
Daemon mode - Continuous autonomous research operation.

Phase 6.3: Continuous Operation.
"""

import os
import sys
import time
import signal
import atexit
import logging
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

# PID file management
PID_FILE = ".autoresearch.pid"
LOG_FILE = "autoresearch.log"


class DaemonState(Enum):
    """Daemon states."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    RESTARTING = "restarting"
    FAILED = "failed"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthStatus:
    """Health check status."""

    state: DaemonState = DaemonState.HEALTHY
    uptime_seconds: float = 0
    experiments_run: int = 0
    last_experiment_time: Optional[str] = None
    error_count: int = 0
    last_error: Optional[str] = None
    cpu_percent: float = 0
    memory_mb: float = 0
    message: str = "OK"


@dataclass
class DaemonConfig:
    """Daemon configuration."""

    log_file: str = LOG_FILE
    pid_file: str = PID_FILE
    stats_file: str = "daemon_stats.json"
    health_check_interval: int = 60  # seconds
    health_check_timeout: int = 10
    max_restart_attempts: int = 3
    restart_cooldown: int = 300  # seconds between restarts
    experiment_batch_size: int = 10
    run_on_start: Callable = None  # Function to run on start
    stop_on_failure: bool = True


class Daemon:
    """
    Autonomous research daemon for continuous operation.

    Features:
    - Background execution
    - Health checks
    - Auto-restart on failure
    - Graceful shutdown
    - PID file management
    """

    def __init__(self, config: Optional[DaemonConfig] = None):
        self.config = config or DaemonConfig()
        self.state = DaemonState.STOPPED
        self.start_time: Optional[float] = None
        self.restart_count = 0
        self.last_restart_time: Optional[float] = None
        self.error_count = 0
        self.last_error: Optional[str] = None
        self.experiments_run = 0
        self._running = False

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Set up logging."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.config.log_file),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def _write_pid(self):
        """Write PID file."""
        with open(self.config.pid_file, "w") as f:
            f.write(str(os.getpid()))

    def _remove_pid(self):
        """Remove PID file."""
        pid_path = Path(self.config.pid_file)
        if pid_path.exists():
            pid_path.unlink()

    def _load_stats(self) -> Dict[str, Any]:
        """Load daemon stats."""
        stats_path = Path(self.config.stats_file)
        if stats_path.exists():
            return json.loads(stats_path.read_text())
        return {}

    def _save_stats(self, stats: Dict[str, Any]):
        """Save daemon stats."""
        with open(self.config.stats_file, "w") as f:
            json.dump(stats, f, indent=2)

    def _get_pid(self) -> Optional[int]:
        """Get PID from file."""
        pid_path = Path(self.config.pid_file)
        if pid_path.exists():
            return int(pid_path.read_text().strip())
        return None

    def is_running(self) -> bool:
        """Check if daemon is running."""
        pid = self._get_pid()
        if pid is None:
            return False
        # Check if process exists
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def start(
        self,
        daemonize: bool = True,
        working_dir: Optional[str] = None,
    ):
        """
        Start the daemon.

        Args:
            daemonize: If True, fork to background
            working_dir: Working directory
        """
        if self.is_running():
            self.logger.error("Daemon already running")
            return False

        if daemonize:
            # Fork to background
            try:
                pid = os.fork()
                if pid > 0:
                    # Parent exits
                    sys.exit(0)
            except OSError as e:
                self.logger.error(f"Fork failed: {e}")
                return False

        # Child process setup
        os.chdir(working_dir or os.getcwd())
        os.setsid()  # New session

        # Write PID
        self._write_pid()
        atexit.register(self._remove_pid)

        # Signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGHUP, self._handle_restart)

        # Update state
        self.state = DaemonState.RUNNING
        self.start_time = time.time()
        self._running = True
        self.restart_count = 0

        self.logger.info("Daemon started")
        self.logger.info(f"PID: {os.getpid()}")

        # Run on start if configured
        if self.config.run_on_start:
            try:
                self.config.run_on_start()
            except Exception as e:
                self.logger.error(f"start callback failed: {e}")

        # Main loop
        self._main_loop()

        return True

    def _main_loop(self):
        """Main daemon loop."""
        last_health_check = time.time()

        while self._running:
            try:
                # Health check
                if time.time() - last_health_check >= self.config.health_check_interval:
                    self._check_health()
                    last_health_check = time.time()

                # Sleep
                time.sleep(1)

            except KeyboardInterrupt:
                self.logger.info("Interrupted")
                break
            except Exception as e:
                self.logger.error(f"Loop error: {e}")
                self.error_count += 1
                self.last_error = str(e)

                if self.config.stop_on_failure:
                    break

        self.state = DaemonState.STOPPED
        self._remove_pid()
        self.logger.info("Daemon stopped")

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signal."""
        self.logger.info(f"Received signal {signum}")
        self._running = False

    def _handle_restart(self, signum, frame):
        """Handle restart signal."""
        self.logger.info("Received SIGHUP, restarting")
        self.state = DaemonState.RESTARTING
        self.restart_count += 1
        self._running = False

    def _check_health(self) -> HealthStatus:
        """Check daemon health."""
        uptime = time.time() - self.start_time if self.start_time else 0

        status = HealthStatus(
            state=DaemonState.HEALTHY,
            uptime_seconds=uptime,
            experiments_run=self.experiments_run,
            error_count=self.error_count,
            last_error=self.last_error,
        )

        # Check for issues
        if self.error_count > 10:
            status.state = DaemonState.UNHEALTHY
            status.message = "Too many errors"

        # Save health status
        stats = self._load_stats()
        stats.update(
            {
                "last_health_check": datetime.now().isoformat(),
                "uptime_seconds": uptime,
                "experiments_run": self.experiments_run,
                "error_count": self.error_count,
                "state": status.state.value,
            }
        )
        self._save_stats(stats)

        return status

    def get_status(self) -> HealthStatus:
        """Get daemon status."""
        if not self.is_running():
            return HealthStatus(state=DaemonState.STOPPED, message="Not running")

        return self._check_health()

    def stop(self):
        """Stop the daemon."""
        pid = self._get_pid()
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
                self.logger.info(f"Sent SIGTERM to {pid}")
            except OSError as e:
                self.logger.error(f"Failed to stop: {e}")

    def restart(self):
        """Restart the daemon."""
        self.stop()
        time.sleep(2)
        self.start()

    def status(self):
        """Print daemon status."""
        pid = self._get_pid()
        if not pid:
            print("Daemon not running")
            return

        print(f"Dameon running with PID: {pid}")
        status = self.get_status()
        print(f"Uptime: {status.uptime_seconds:.0f}s")
        print(f"Experiments: {status.experiments_run}")
        print(f"Errors: {status.error_count}")
        print(f"State: {status.state.value}")


class HealthChecker:
    """Standalone health checker."""

    def __init__(self, config: Optional[DaemonConfig] = None):
        self.config = config or DaemonConfig()
        self.checks: List[Callable[[], bool]] = []

    def register_check(self, name: str, check_fn: Callable[[], bool]):
        """Register a health check."""
        self.checks.append(check_fn)

    def check_all(self) -> HealthStatus:
        """Run all health checks."""
        status = HealthStatus()

        for check in self.checks:
            try:
                if not check():
                    status.state = DaemonState.UNHEALTHY
                    status.message = f"Check failed: {check.__name__}"
            except Exception as e:
                status.state = DaemonState.UNHEALTHY
                status.message = f"Check error: {e}"

        return status


def run_daemon(
    working_dir: Optional[str] = None,
    log_file: str = LOG_FILE,
    pid_file: str = PID_FILE,
    start_command: Optional[str] = None,
):
    """
    Run the autonomous research daemon.

    Usage:
        python daemon.py start
        python daemon.py stop
        python daemon.py restart
        python daemon.py status
    """
    config = DaemonConfig(
        log_file=log_file,
        pid_file=pid_file,
    )
    daemon = Daemon(config)

    if start_command == "start":
        daemon.start(working_dir=working_dir)
    elif start_command == "stop":
        daemon.stop()
    elif start_command == "restart":
        daemon.restart()
    elif start_command == "status":
        daemon.status()
    else:
        print(f"Usage: python daemon.py [start|stop|restart|status]")


if __name__ == "__main__":
    # Test daemon
    print("Testing daemon module...")

    # Default: status check
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
    else:
        cmd = "status"

    run_daemon(start_command=cmd)
