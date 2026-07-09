"""
Autonomous Research Loop - CLI and convenience entry points.

Provides the ``autonomous_pipeline()`` convenience function for backward
compatibility, and the ``main()`` CLI entry point.
"""

import argparse
from typing import Any, Dict, List, Optional

from autoresearch.experiment.autonomous_loop.pipeline import (
    AutonomousPipeline,
)

# Version - mirrors the package-level version
__version__ = "0.7.3"


def autonomous_pipeline(
    raw,
    model=None,
    tokenizer=None,
    config_path: str = "config.yaml",
) -> List[str]:
    """Convenience function for backward compatibility.

    Args:
        raw: Raw texts or path to text file
        model: Model for model-in-the-loop (optional)
        tokenizer: Tokenizer (optional)
        config_path: Path to config

    Returns:
        Prepared dataset
    """
    # Handle raw as path or list
    if isinstance(raw, str):
        with open(raw) as f:
            raw_texts = [line.strip() for line in f if line.strip()]
    else:
        raw_texts = raw

    # Create pipeline
    pipeline = AutonomousPipeline(config_path)

    # Prepare data
    data = pipeline.prepare_data(
        raw_texts,
        use_synthetic=True,
        use_model_loop=model is not None,
        model=model,
        tokenizer=tokenizer,
    )

    return data


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Autonomous Research Loop")
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"autoresearch-stack v{__version__}",
        help="Show version and exit",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--input",
        "-i",
        help="Input text file (one text per line)",
    )
    parser.add_argument(
        "--experiments",
        "-n",
        type=int,
        default=None,
        help="Number of experiments to run",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare data, don't run experiments",
    )

    args = parser.parse_args()

    # Load texts
    if args.input:
        with open(args.input) as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        # Default sample texts
        texts = [
            "Machine learning is a method of data analysis.",
            "Neural networks are inspired by biological neurons.",
            "Transformers use attention mechanisms.",
            "Backpropagation trains neural networks.",
            "Gradient descent optimizes model parameters.",
        ]

    # Create pipeline
    pipeline = AutonomousPipeline(args.config)

    # Prepare data
    data = pipeline.prepare_data(texts)
    print(f"\nPrepared {len(data)} training examples")

    # Show config
    print("\nConfiguration:")
    print(f"  Experiments: {args.experiments or pipeline.config.experiment.budget}")
    print(f"  Time per exp: {pipeline.config.experiment.time_per_experiment}s")
    print(f"  Target val_bpb: {pipeline.config.experiment.val_target}")

    if args.prepare_only:
        print("\nData preparation complete (--prepare-only)")
        return

    # Prepare curriculum
    _scheduler = pipeline.prepare_curriculum(data)

    # Run autonomous loop with prepared data (not raw texts)
    pipeline.run_autonomous_loop(data, args.experiments)
