"""
Monitor - Real-time status display for experiments.

Phase 5: Production Hardening - Monitoring.
"""

import sys
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque


@dataclass
class ExperimentStatus:
    """Current experiment status."""

    experiment_id: int
    change_description: str
    start_time: float
    status: str = "pending"  # pending, running, complete, failed
    val_bpb_before: float = 0.0
    val_bpb_after: float = 0.0
    iteration: int = 0
    notes: str = ""


@dataclass
class MonitorStats:
    """Monitor statistics."""

    total_experiments: int = 0
    kept: int = 0
    reverted: int = 0
    running: int = 0
    best_val_bpb: float = float("inf")
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    @property
    def experiments_per_hour(self) -> float:
        if self.elapsed_time == 0:
            return 0
        return self.total_experiments / (self.elapsed_time / 3600)

    @property
    def success_rate(self) -> float:
        if self.total_experiments == 0:
            return 0
        return self.kept / self.total_experiments


class Monitor:
    """Real-time experiment monitor."""

    def __init__(self):
        self.stats = MonitorStats()
        self.current_experiment: Optional[ExperimentStatus] = None
        self.history: List[ExperimentStatus] = []
        self.events: deque = deque(maxlen=100)

        # For display
        self._last_update = 0
        self._update_interval = 1.0  # seconds

    def log_event(self, event_type: str, message: str):
        """Log an event."""
        self.events.append(
            {
                "timestamp": datetime.now().isoformat(),
                "type": event_type,
                "message": message,
            }
        )

    def start_experiment(
        self,
        experiment_id: int,
        change_description: str,
        val_bpb_before: float,
    ):
        """Start an experiment."""
        self.current_experiment = ExperimentStatus(
            experiment_id=experiment_id,
            change_description=change_description,
            val_bpb_before=val_bpb_before,
            start_time=time.time(),
            status="running",
        )
        self.stats.running = 1

        self.log_event("start", f"Experiment {experiment_id}: {change_description}")

    def update_progress(self, iteration: int):
        """Update experiment progress."""
        if self.current_experiment:
            self.current_experiment.iteration = iteration

    def complete_experiment(
        self,
        val_bpb_after: float,
        status: str,
    ):
        """Complete an experiment."""
        if self.current_experiment:
            self.current_experiment.val_bpb_after = val_bpb_after
            self.current_experiment.status = status
            self.current_experiment.notes = "complete"

            # Update stats
            self.stats.total_experiments += 1
            self.stats.running = 0

            if status == "kept":
                self.stats.kept += 1
                if val_bpb_after < self.stats.best_val_bpb:
                    self.stats.best_val_bpb = val_bpb_after
            else:
                self.stats.reverted += 1

            # Add to history
            self.history.append(self.current_experiment)

            self.log_event(
                "complete",
                f"Experiment {self.current_experiment.experiment_id}: "
                f"val_bpb {self.current_experiment.val_bpb_before:.4f} -> "
                f"{val_bpb_after:.4f} ({status})",
            )

            self.current_experiment = None

    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return {
            "stats": {
                "total": self.stats.total_experiments,
                "kept": self.stats.kept,
                "reverted": self.stats.reverted,
                "running": self.stats.running,
                "best_val_bpb": self.stats.best_val_bpb,
                "elapsed_time": self.stats.elapsed_time,
                "experiments_per_hour": self.stats.experiments_per_hour,
                "success_rate": self.stats.success_rate,
            },
            "current": None
            if not self.current_experiment
            else {
                "experiment_id": self.current_experiment.experiment_id,
                "change": self.current_experiment.change_description,
                "val_bpb_before": self.current_experiment.val_bpb_before,
                "iteration": self.current_experiment.iteration,
            },
            "recent_events": list(self.events)[-5:],
        }

    def print_status(self):
        """Print status to console."""
        status = self.get_status()
        stats = status["stats"]

        print(f"\r{'=' * 60}")
        print(
            f"EXPERIMENTS: {stats['total']} | KEPT: {stats['kept']} | "
            f"REVERTED: {stats['reverted']} | BEST: {stats['best_val_bpb']:.4f}"
        )
        print(
            f"RATE: {stats['experiments_per_hour']:.1f}/hr | "
            f"SUCCESS: {stats['success_rate'] * 100:.1f}% | "
            f"ELAPSED: {self._format_time(stats['elapsed_time'])}"
        )

        if status["current"]:
            curr = status["current"]
            print(f"CURRENT: exp{curr['experiment_id']} - {curr['change'][:40]}")

        print(f"{'=' * 60}", end="\r")
        sys.stdout.flush()

    def _format_time(self, seconds: float) -> str:
        """Format elapsed time."""
        td = timedelta(seconds=int(seconds))
        return str(td)

    def should_update(self) -> bool:
        """Check if should update display."""
        now = time.time()
        if now - self._last_update >= self._update_interval:
            self._last_update = now
            return True
        return False


class ProgressBar:
    """Simple progress bar."""

    def __init__(self, width: int = 40):
        self.width = width

    def draw(self, current: int, total: int, prefix: str = ""):
        """Draw progress bar."""
        if total == 0:
            percent = 0
        else:
            percent = current / total

        filled = int(self.width * percent)
        bar = "█" * filled + "░" * (self.width - filled)

        print(f"\r{prefix}[{bar}] {percent * 100:.1f}%", end="")


def get_monitor() -> Monitor:
    """Get monitor instance."""
    return Monitor()


if __name__ == "__main__":
    # Demo
    print("Testing monitor...")

    m = get_monitor()

    # Simulate some experiments
    m.start_experiment(1, "Increase learning rate", 1.0)
    time.sleep(0.1)
    m.complete_experiment(0.95, "kept")

    m.start_experiment(2, "Add dropout", 0.95)
    time.sleep(0.1)
    m.complete_experiment(0.97, "reverted")

    print("\nFinal status:")
    print(m.get_status())

    print("\nMonitor OK")
