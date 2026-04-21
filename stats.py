"""
Summary Statistics - Statistical analysis of experiment results.

Phase 5.1: Summary Statistics.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict

# Optional numpy/scipy imports
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    from scipy import stats as scipy_stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    scipy_stats = None


@dataclass
class ExperimentStats:
    """Statistics for a single experiment."""

    total: int = 0
    kept: int = 0
    reverted: int = 0
    running: int = 0
    failed: int = 0

    # Performance metrics
    best_val_bpb: Optional[float] = None
    worst_val_bpb: Optional[float] = None
    mean_val_bpb: Optional[float] = None
    std_val_bpb: Optional[float] = None
    median_val_bpb: Optional[float] = None

    # Improvement metrics
    baseline_val_bpb: Optional[float] = None
    improvement: Optional[float] = None
    improvement_pct: Optional[float] = None

    # Timing stats
    total_time_seconds: Optional[float] = None
    mean_time_seconds: Optional[float] = None

    # Change type breakdown
    change_type_counts: Dict[str, int] = field(default_factory=dict)
    change_type_success: Dict[str, int] = field(default_factory=dict)

    # Run metrics
    success_rate: float = 0.0
    kept_rate: float = 0.0


@dataclass
class SummaryStatistics:
    """Summary statistics calculator."""

    experiments: List[Dict[str, Any]] = field(default_factory=list)
    baseline: Optional[float] = None

    def __init__(
        self,
        experiments: Optional[List[Dict[str, Any]]] = None,
        baseline: Optional[float] = None,
    ):
        self.experiments = experiments or []
        self.baseline = baseline

    def calculate(self) -> ExperimentStats:
        """Calculate all statistics."""
        stats = ExperimentStats()

        if not self.experiments:
            return stats

        # Basic counts
        stats.total = len(self.experiments)
        stats.kept = sum(1 for e in self.experiments if e.get("status") == "kept")
        stats.reverted = sum(
            1 for e in self.experiments if e.get("status") == "reverted"
        )
        stats.running = sum(1 for e in self.experiments if e.get("status") == "running")
        stats.failed = sum(1 for e in self.experiments if e.get("status") == "failed")

        # Rates
        stats.kept_rate = stats.kept / stats.total if stats.total > 0 else 0
        stats.success_rate = stats.kept_rate  # Alias

        # Performance metrics (kept experiments only)
        kept_exps = [e for e in self.experiments if e.get("status") == "kept"]
        if kept_exps:
            val_bpbs = [e.get("val_bpb_after", 0) for e in kept_exps]

            if NUMPY_AVAILABLE:
                stats.best_val_bpb = float(np.min(val_bpbs))
                stats.worst_val_bpb = float(np.max(val_bpbs))
                stats.mean_val_bpb = float(np.mean(val_bpbs))
                stats.std_val_bpb = float(np.std(val_bpbs))
                stats.median_val_bpb = float(np.median(val_bpbs))
            else:
                stats.best_val_bpb = min(val_bpbs)
                stats.worst_val_bpb = max(val_bpbs)
                stats.mean_val_bpb = sum(val_bpbs) / len(val_bpbs)
                stats.median_val_bpb = sorted(val_bpbs)[len(val_bpbs) // 2]

        # Improvement over baseline
        if self.baseline is not None and stats.best_val_bpb is not None:
            stats.baseline_val_bpb = self.baseline
            stats.improvement = self.baseline - stats.best_val_bpb
            stats.improvement_pct = (
                (stats.improvement / self.baseline) * 100 if self.baseline > 0 else 0
            )

        # Timing
        times = [
            e.get("training_time", 0)
            for e in self.experiments
            if e.get("training_time")
        ]
        if times:
            if NUMPY_AVAILABLE:
                stats.total_time_seconds = float(np.sum(times))
                stats.mean_time_seconds = float(np.mean(times))
            else:
                stats.total_time_seconds = sum(times)
                stats.mean_time_seconds = sum(times) / len(times)

        # Change type breakdown
        type_counts = defaultdict(int)
        type_success = defaultdict(int)

        for e in self.experiments:
            ctype = e.get("change_type", "unknown")
            type_counts[ctype] += 1
            if e.get("status") == "kept":
                type_success[ctype] += 1

        stats.change_type_counts = dict(type_counts)
        stats.change_type_success = dict(type_success)

        return stats

    def get_convergence_metrics(self) -> Dict[str, Any]:
        """Get convergence metrics."""
        if not self.experiments:
            return {}

        # Sort by experiment ID (assuming chronological)
        sorted_exps = sorted(self.experiments, key=lambda e: e.get("id", 0))

        # Get kept experiments
        kept = [e for e in sorted_exps if e.get("status") == "kept"]
        if not kept:
            return {"converged": False, "reason": "no_kept_experiments"}

        val_bpbs = [e.get("val_bpb_after", float("inf")) for e in kept]

        # Check if we've converged (last 5 experiments within 1% of each other)
        if len(val_bpbs) >= 5:
            last_5 = val_bpbs[-5:]
            if NUMPY_AVAILABLE:
                std = np.std(last_5)
                mean = np.mean(last_5)
                relative_std = std / mean if mean > 0 else float("inf")
                converged = relative_std < 0.01
            else:
                std = (max(last_5) - min(last_5)) / 2
                mean = sum(last_5) / 5
                converged = std / mean < 0.01 if mean > 0 else False
        else:
            converged = False

        return {
            "converged": converged,
            "best_val_bpb": min(val_bpbs),
            "latest_val_bpb": val_bpbs[-1],
            "total_kept": len(kept),
        }

    def get_trend_metrics(self, window: int = 10) -> Dict[str, Any]:
        """Get trend metrics (rolling window)."""
        if not self.experiments or len(self.experiments) < window:
            return {}

        sorted_exps = sorted(self.experiments, key=lambda e: e.get("id", 0))
        recent = sorted_exps[-window:]

        # Recent success rate
        recent_kept = sum(1 for e in recent if e.get("status") == "kept")
        recent_rate = recent_kept / window

        # Recent performance
        recent_vals = [
            e.get("val_bpb_after", float("inf"))
            for e in recent
            if e.get("status") == "kept"
        ]

        trend = {}
        if recent_vals:
            if NUMPY_AVAILABLE:
                trend = {
                    "recent_success_rate": recent_rate,
                    "recent_mean_val_bpb": float(np.mean(recent_vals)),
                    "recent_best_val_bpb": float(np.min(recent_vals)),
                }
            else:
                trend = {
                    "recent_success_rate": recent_rate,
                    "recent_mean_val_bpb": sum(recent_vals) / len(recent_vals),
                    "recent_best_val_bpb": min(recent_vals),
                }

        return trend

    def compare_sessions(
        self,
        other_experiments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compare with another session."""
        other_stats = SummaryStatistics(other_experiments).calculate()
        our_stats = self.calculate()

        return {
            "our_best": our_stats.best_val_bpb,
            "other_best": other_stats.best_val_bpb,
            "our_keep_rate": our_stats.kept_rate,
            "other_keep_rate": other_stats.kept_rate,
            "our_total": our_stats.total,
            "other_total": other_stats.total,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        stats = self.calculate()
        return {
            "total": stats.total,
            "kept": stats.kept,
            "reverted": stats.reverted,
            "kept_rate": stats.kept_rate,
            "success_rate": stats.success_rate,
            "best_val_bpb": stats.best_val_bpb,
            "worst_val_bpb": stats.worst_val_bpb,
            "mean_val_bpb": stats.mean_val_bpb,
            "std_val_bpb": stats.std_val_bpb,
            "median_val_bpb": stats.median_val_bpb,
            "baseline_val_bpb": stats.baseline_val_bpb,
            "improvement": stats.improvement,
            "improvement_pct": stats.improvement_pct,
            "total_time_seconds": stats.total_time_seconds,
            "mean_time_seconds": stats.mean_time_seconds,
            "change_type_counts": stats.change_type_counts,
            "change_type_success": stats.change_type_success,
            "convergence": self.get_convergence_metrics(),
            "trend": self.get_trend_metrics(),
        }

    def to_json(self, path: str):
        """Save to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def summary_text(self) -> str:
        """Generate text summary."""
        stats = self.calculate()
        convergence = self.get_convergence_metrics()
        trend = self.get_trend_metrics()

        lines = [
            "=== Summary Statistics ===",
            "",
            f"Total Experiments: {stats.total}",
            f"Kept: {stats.kept} ({stats.kept_rate:.1%})",
            f"Reverted: {stats.reverted}",
            "",
        ]

        if stats.best_val_bpb is not None:
            lines.append(f"Best val_bpb: {stats.best_val_bpb:.4f}")
            lines.append(f"Mean val_bpb: {stats.mean_val_bpb:.4f}")
            if stats.std_val_bpb:
                lines.append(f"Std val_bpb: {stats.std_val_bpb:.4f}")

        if stats.improvement_pct is not None:
            lines.append("")
            lines.append(f"Improvement: {stats.improvement_pct:.2f}%")

        if convergence:
            lines.append("")
            lines.append(f"Converged: {convergence.get('converged', False)}")

        if trend:
            lines.append("")
            lines.append(
                f"Recent Success Rate: {trend.get('recent_success_rate', 0):.1%}"
            )

        if stats.change_type_counts:
            lines.append("")
            lines.append("Change Types:")
            for ctype, count in stats.change_type_counts.items():
                success = stats.change_type_success.get(ctype, 0)
                rate = success / count if count > 0 else 0
                lines.append(f"  {ctype}: {count} ({rate:.0%} success)")

        return "\n".join(lines)


def load_experiments(path: str) -> List[Dict[str, Any]]:
    """Load experiments from JSON file."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return data
    return data.get("experiments", [])


if __name__ == "__main__":
    print("Testing summary statistics...")

    # Sample experiments
    experiments = [
        {
            "id": 1,
            "change_description": "Increase LR",
            "change_type": "optimization",
            "status": "kept",
            "val_bpb_before": 1.0,
            "val_bpb_after": 0.95,
            "training_time": 100,
        },
        {
            "id": 2,
            "change_description": "Add dropout",
            "change_type": "architecture",
            "status": "reverted",
            "val_bpb_before": 0.95,
            "val_bpb_after": 0.97,
            "training_time": 120,
        },
        {
            "id": 3,
            "change_description": "Adjust warmup",
            "change_type": "curriculum",
            "status": "kept",
            "val_bpb_before": 0.95,
            "val_bpb_after": 0.92,
            "training_time": 90,
        },
        {
            "id": 4,
            "change_description": "Reduce batch",
            "change_type": "optimization",
            "status": "kept",
            "val_bpb_before": 0.92,
            "val_bpb_after": 0.90,
            "training_time": 95,
        },
        {
            "id": 5,
            "change_description": "Add weight decay",
            "change_type": "optimization",
            "status": "kept",
            "val_bpb_before": 0.90,
            "val_bpb_after": 0.88,
            "training_time": 85,
        },
    ]

    # Calculate stats
    stats = SummaryStatistics(experiments, baseline=1.0)
    result = stats.calculate()

    print(f"Total: {result.total}")
    print(f"Kept: {result.kept} ({result.kept_rate:.1%})")
    print(f"Best val_bpb: {result.best_val_bpb:.4f}")
    print(f"Improvement: {result.improvement_pct:.2f}%")

    print()
    print(stats.summary_text())

    # Save to JSON
    stats.to_json("experiment_stats.json")
    print("\nSaved to experiment_stats.json")
