"""
Figure generation - Visualization of experiment results.

Phase 5.1: Figure Generation.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Optional numpy import
try:
    import numpy as np
except ImportError:
    np = None

# Optional matplotlib import (graceful fallback)
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.figure as mplt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    mplt = None
    plt = None


# Optional seaborn for better styling
try:
    import seaborn as sns  # noqa: F401

    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


@dataclass
class FigureConfig:
    """Figure configuration."""

    width: float = 10
    height: float = 6
    dpi: int = 100
    style: str = "seaborn-v0_8-whitegrid" if SEABORN_AVAILABLE else "ggplot"
    title_fontsize: int = 14
    label_fontsize: int = 12
    legend_fontsize: int = 10


class FigureGenerator:
    """Generate figures from experiment results."""

    def __init__(self, config: Optional[FigureConfig] = None):
        self.config = config or FigureConfig()
        self._setup_style()

    def _setup_style(self):
        """Set up matplotlib style."""
        if not MATPLOTLIB_AVAILABLE:
            return

        try:
            plt.style.use(self.config.style)
        except Exception:
            pass  # Fallback to default

    def _check_matplotlib(self):
        """Check if matplotlib is available."""
        if not MATPLOTLIB_AVAILABLE:
            raise RuntimeError(
                "matplotlib is required for figure generation. "
                "Install with: pip install matplotlib seaborn"
            )

    def plot_learning_curve(
        self,
        experiments: List[Dict[str, Any]],
        metric: str = "val_bpb",
        title: str = "Learning Curve",
    ) -> Optional[object]:
        """Plot learning curve from experiments."""
        self._check_matplotlib()

        fig = mplt.Figure(figsize=(self.config.width, self.config.height))
        ax = fig.add_subplot(111)

        # Extract data
        ids = [e.get("id", i) for i, e in enumerate(experiments)]
        before_vals = [e.get(f"{metric}_before", 0) for e in experiments]
        after_vals = [e.get(f"{metric}_after", 0) for e in experiments]

        # Plot lines
        ax.plot(ids, before_vals, "o-", label=f"{metric} (before)", alpha=0.7)
        ax.plot(ids, after_vals, "s-", label=f"{metric} (after)", alpha=0.7)

        # Fill between for improvement
        ax.fill_between(ids, before_vals, after_vals, alpha=0.2, color="green")

        ax.set_xlabel("Experiment ID", fontsize=self.config.label_fontsize)
        ax.set_ylabel(metric, fontsize=self.config.label_fontsize)
        ax.set_title(title, fontsize=self.config.title_fontsize)
        ax.legend(fontsize=self.config.legend_fontsize)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def plot_success_rate(
        self,
        experiments: List[Dict[str, Any]],
        window_size: int = 10,
        title: str = "Success Rate Over Time",
    ) -> Optional[object]:
        """Plot rolling success rate."""
        self._check_matplotlib()

        fig = mplt.Figure(figsize=(self.config.width, self.config.height))
        ax = fig.add_subplot(111)

        # Calculate rolling success
        statuses = [e.get("status", "reverted") for e in experiments]
        successes = [1 if s == "kept" else 0 for s in statuses]

        # Rolling average
        rolling = []
        for i in range(len(successes)):
            start = max(0, i - window_size + 1)
            window = successes[start : i + 1]
            rolling.append(sum(window) / len(window))

        # Plot
        ax.plot(range(len(rolling)), rolling, "b-", linewidth=2)
        ax.axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="50% baseline")

        ax.set_xlabel("Experiment ID", fontsize=self.config.label_fontsize)
        ax.set_ylabel("Success Rate", fontsize=self.config.label_fontsize)
        ax.set_title(title, fontsize=self.config.title_fontsize)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=self.config.legend_fontsize)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def plot_metric_distribution(
        self,
        experiments: List[Dict[str, Any]],
        metric: str = "val_bpb_after",
        title: str = "Metric Distribution",
    ) -> Optional[object]:
        """Plot metric distribution histogram."""
        self._check_matplotlib()

        fig = mplt.Figure(figsize=(self.config.width, self.config.height))
        ax = fig.add_subplot(111)

        # Extract kept experiments
        kept_exps = [e for e in experiments if e.get("status") == "kept"]
        values = [e.get(metric, 0) for e in kept_exps]

        if not values:
            ax.text(
                0.5,
                0.5,
                "No kept experiments to plot",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

        # Plot histogram
        ax.hist(values, bins=min(20, len(values)), edgecolor="black", alpha=0.7)

        # Add mean line
        import numpy as np

        mean_val = np.mean(values)
        ax.axvline(mean_val, color="r", linestyle="--", label=f"Mean: {mean_val:.4f}")

        ax.set_xlabel(metric, fontsize=self.config.label_fontsize)
        ax.set_ylabel("Count", fontsize=self.config.label_fontsize)
        ax.set_title(title, fontsize=self.config.title_fontsize)
        ax.legend(fontsize=self.config.legend_fontsize)
        ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        return fig

    def plot_change_type_performance(
        self,
        experiments: List[Dict[str, Any]],
        title: str = "Performance by Change Type",
    ) -> Optional[object]:
        """Plot performance by change type."""
        self._check_matplotlib()

        fig = mplt.Figure(figsize=(self.config.width, self.config.height))
        ax = fig.add_subplot(111)

        # Group by change type
        from collections import defaultdict

        type_groups = defaultdict(list)

        for e in experiments:
            if e.get("status") == "kept":
                ctype = e.get("change_type", "unknown")
                val_bpb = e.get("val_bpb_after", float("inf"))
                type_groups[ctype].append(val_bpb)

        # Calculate means
        types = list(type_groups.keys())
        means = [
            float(sum(type_groups[t])) / len(type_groups[t])
            if not np
            else np.mean(type_groups[t])
            for t in types
        ]

        # Bar plot
        x_pos = range(len(types))
        ax.bar(x_pos, means, edgecolor="black", alpha=0.7)

        ax.set_xlabel("Change Type", fontsize=self.config.label_fontsize)
        ax.set_ylabel("Mean val_bpb", fontsize=self.config.label_fontsize)
        ax.set_title(title, fontsize=self.config.title_fontsize)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(types, rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        return fig

    def plot_improvement_timeline(
        self,
        experiments: List[Dict[str, Any]],
        baseline: float,
        metric: str = "val_bpb",
        title: str = "Improvement Over Baseline",
    ) -> Optional[object]:
        """Plot improvement over baseline over time."""
        self._check_matplotlib()

        fig = mplt.Figure(figsize=(self.config.width, self.config.height))
        ax = fig.add_subplot(111)

        # Calculate improvement
        improvements = []
        for e in experiments:
            if e.get("status") == "kept":
                val_after = e.get(f"{metric}_after", baseline)
                # Improvement as percentage (lower bpb is better)
                improvement = ((baseline - val_after) / baseline) * 100
                improvements.append((e.get("id"), improvement))

        if not improvements:
            ax.text(
                0.5,
                0.5,
                "No kept experiments to plot",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

        ids, imps = zip(*improvements)

        # Plot
        ax.plot(ids, imps, "g-", linewidth=2, marker="o")
        ax.axhline(0, color="r", linestyle="--", alpha=0.5, label="Baseline")

        ax.set_xlabel("Experiment ID", fontsize=self.config.label_fontsize)
        ax.set_ylabel("Improvement (%)", fontsize=self.config.label_fontsize)
        ax.set_title(title, fontsize=self.config.title_fontsize)
        ax.legend(fontsize=self.config.legend_fontsize)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def save_figure(self, fig: object, path: str) -> str:
        """Save figure to file."""
        fig.savefig(path, dpi=self.config.dpi)
        return path

    def generate_all_figures(
        self,
        experiments: List[Dict[str, Any]],
        output_dir: str = "figures",
        baseline: float = 1.0,
    ) -> Dict[str, str]:
        """Generate all standard figures."""
        self._check_matplotlib()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved = {}

        # Learning curve
        fig = self.plot_learning_curve(experiments)
        if fig:
            path = str(output_path / "learning_curve.png")
            self.save_figure(fig, path)
            saved["learning_curve"] = path

        # Success rate
        fig = self.plot_success_rate(experiments)
        if fig:
            path = str(output_path / "success_rate.png")
            self.save_figure(fig, path)
            saved["success_rate"] = path

        # Distribution
        fig = self.plot_metric_distribution(experiments)
        if fig:
            path = str(output_path / "distribution.png")
            self.save_figure(fig, path)
            saved["distribution"] = path

        # By type
        fig = self.plot_change_type_performance(experiments)
        if fig:
            path = str(output_path / "by_type.png")
            self.save_figure(fig, path)
            saved["by_type"] = path

        # Timeline
        fig = self.plot_improvement_timeline(experiments, baseline)
        if fig:
            path = str(output_path / "improvement.png")
            self.save_figure(fig, path)
            saved["improvement"] = path

        return saved


def load_experiments(path: str) -> List[Dict[str, Any]]:
    """Load experiments from JSON file."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return data
    return data.get("experiments", [])


if __name__ == "__main__":
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not installed - skipping figure generation demo")
    else:
        print("Testing figure generation...")

        # Sample experiments
        experiments = [
            {
                "id": 1,
                "change_description": "Increase LR",
                "change_type": "optimization",
                "status": "kept",
                "val_bpb_before": 1.0,
                "val_bpb_after": 0.95,
            },
            {
                "id": 2,
                "change_description": "Add dropout",
                "change_type": "architecture",
                "status": "reverted",
                "val_bpb_before": 0.95,
                "val_bpb_after": 0.97,
            },
            {
                "id": 3,
                "change_description": "Adjust warmup",
                "change_type": "curriculum",
                "status": "kept",
                "val_bpb_before": 0.95,
                "val_bpb_after": 0.92,
            },
            {
                "id": 4,
                "change_description": "Reduce batch",
                "change_type": "optimization",
                "status": "kept",
                "val_bpb_before": 0.92,
                "val_bpb_after": 0.90,
            },
            {
                "id": 5,
                "change_description": "Add weight decay",
                "change_type": "optimization",
                "status": "kept",
                "val_bpb_before": 0.90,
                "val_bpb_after": 0.88,
            },
        ]

        # Generate figures
        gen = FigureGenerator()
        saved = gen.generate_all_figures(experiments, "figures", baseline=1.0)

        print("Generated figures:")
        for name, path in saved.items():
            print(f"  {name}: {path}")
