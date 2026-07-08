"""
Autonomous Research Loop - Package Scaffold

This package was extracted from autonomous_loop.py to improve module depth
and testability.  All public symbols are re-exported here for backward
compatibility so ``from autoresearch.experiment.autonomous_loop import X``
continues to work.
"""

from autoresearch.experiment.autonomous_loop.cli import (
    autonomous_pipeline,
    main,
)
from autoresearch.experiment.autonomous_loop.pipeline import (
    AutonomousPipeline,
)

__all__ = [
    "AutonomousPipeline",
    "autonomous_pipeline",
    "main",
]
