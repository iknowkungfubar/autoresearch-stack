#!/usr/bin/env python3
"""Quick experiment example using autoresearch-stack.

Demonstrates setting up and running a minimal research experiment
with hyperparameter sweep and results reporting.

Usage:
    python examples/quick_experiment.py

Requirements:
    pip install autoresearch-stack
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Allow running from repo checkout
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autoresearch.cli import experiment

# Use a simple built-in experiment: grid-search learning rate and batch size
CONFIG = {
    "experiment": {
        "name": "quick-demo",
        "max_epochs": 2,
    },
    "sweep": {
        "learning_rate": [0.001, 0.01],
        "batch_size": [16, 32],
    },
    "reporting": {
        "enabled": True,
        "output_dir": tempfile.mkdtemp(prefix="autoresearch-demo-"),
    },
}


def main() -> None:
    print("=" * 60)
    print("autoresearch-stack — Quick Experiment Demo")
    print("=" * 60)
    print("This example runs a minimal 2-epoch hyperparameter sweep")
    print("with 2 x 2 = 4 trial configurations.")
    print()

    result = experiment.run(CONFIG)

    print()
    print("=" * 60)
    print("Experiment Complete")
    print("=" * 60)
    print(f"Trials completed: {result.get('trials_completed', 0)}")
    print(f"Best config: {result.get('best_config', {})}")
    print(f"Best score: {result.get('best_score', 'N/A')}")
    print(f"Report saved to: {CONFIG['reporting']['output_dir']}")


if __name__ == "__main__":
    main()
