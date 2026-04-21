"""
Feedback and evaluation for autonomous research.

Features:
- val_bpb metric computation
- Experiment logging
- Failure classification
- Reward computation
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from enum import Enum
from pathlib import Path
from datetime import datetime


class FailureClassification(Enum):
    """Why an experiment failed."""

    # Optimization
    LR_TOO_HIGH = "lr_too_high"
    LR_TOO_LOW = "lr_too_low"
    OPTIMIZER_INSTABILITY = "optimizer_instability"

    # Model
    UNDERFITTING = "underfitting"
    OVERFITTING = "overfitting"
    CAPACITY_LIMIT = "capacity_limit"

    # Data
    DATA_NOISE = "data_noise"
    DATA_DISTRIBUTION = "data_distribution"
    CURRICULUM_MISMATCH = "curriculum_mismatch"

    # Training
    GRADIENT_EXPLOSION = "gradient_explosion"
    GRADIENT_VANISHING = "gradient_vanishing"
    LOSS_SPIKE = "loss_spike"

    # General
    TIMING = "timing"
    NOISE = "noise"
    UNKNOWN = "unknown"


class ExperimentStatus(Enum):
    """Experiment outcome."""

    KEPT = "kept"
    REVERTED = "reverted"
    RUNNING = "running"
    FAILED = "failed"


@dataclass
class Experiment:
    """Single experiment record."""

    id: int
    timestamp: str
    change_description: str
    change_code: str
    val_bpb_before: float
    val_bpb_after: float
    training_loss: Optional[float] = None
    eval_loss: Optional[float] = None
    training_time: Optional[float] = None
    memory_used: Optional[float] = None
    status: str = "running"
    failure_classification: Optional[str] = None
    failure_diagnosis: Optional[str] = None
    git_commit: Optional[str] = None
    notes: str = ""

    @property
    def delta(self) -> float:
        """Change in val_bpb."""
        return self.val_bpb_after - self.val_bpb_before

    @property
    def improved(self) -> bool:
        """Did this experiment improve?"""
        return self.val_bpb_after < self.val_bpb_before

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class Feedback:
    """Feedback system for autonomous experiments."""

    def __init__(
        self,
        experiment_log_path: str = "./experiments.jsonl",
        baseline_val_bpb: Optional[float] = None,
    ):
        self.experiment_log_path = Path(experiment_log_path)
        self.baseline_val_bpb = baseline_val_bpb
        self.experiments: List[Experiment] = []
        self._load_experiments()

    def _load_experiments(self):
        """Load existing experiments from log."""
        if self.experiment_log_path.exists():
            with open(self.experiment_log_path, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            self.experiments.append(Experiment(**data))
                        except:
                            pass

    def reward(self, val_bpb: float, score: float = 0.0) -> float:
        """Compute reward from val_bpb and optional score.

        Args:
            val_bpb: Validation bits per byte (lower is better)
            score: Optional secondary score

        Returns:
            Reward value (higher is better)
        """
        # Lower val_bpb = higher reward
        val_bpb_reward = 1.0 / (val_bpb + 1e-6)

        # Add secondary score if provided
        score_bonus = score / 100.0

        return val_bpb_reward + score_bonus

    def log(
        self,
        step: int,
        val_bpb: float,
        score: float = 0.0,
        training_loss: Optional[float] = None,
        eval_loss: Optional[float] = None,
    ) -> float:
        """Log training progress and compute reward.

        Args:
            step: Training step
            val_bpb: Validation bits per byte
            score: Optional score
            training_loss: Training loss
            eval_loss: Evaluation loss

        Returns:
            Reward value
        """
        r = self.reward(val_bpb, score)
        print(f"R: step={step} val_bpb={val_bpb:.6f} score={score} reward={r:.4f}")

        if training_loss is not None:
            print(f"    train_loss={training_loss:.6f}")
        if eval_loss is not None:
            print(f"    eval_loss={eval_loss:.6f}")

        return r

    def start_experiment(
        self,
        change_description: str,
        change_code: str,
        val_bpb_before: float,
    ) -> Experiment:
        """Start a new experiment.

        Args:
            change_description: What changed
            change_code: The actual code change
            val_bpb_before: val_bpb before change

        Returns:
            Experiment record
        """
        exp_id = len(self.experiments) + 1
        exp = Experiment(
            id=exp_id,
            timestamp=datetime.now().isoformat(),
            change_description=change_description,
            change_code=change_code,
            val_bpb_before=val_bpb_before,
            val_bpb_after=val_bpb_before,  # Will be updated
            status="running",
        )
        self.experiments.append(exp)
        return exp

    def complete_experiment(
        self,
        experiment: Experiment,
        val_bpb_after: float,
        status: ExperimentStatus,
        failure_classification: Optional[FailureClassification] = None,
        failure_diagnosis: Optional[str] = None,
        training_time: Optional[float] = None,
        memory_used: Optional[float] = None,
    ):
        """Complete an experiment with results.

        Args:
            experiment: The experiment to complete
            val_bpb_after: val_bpb after change
            status: Whether kept or reverted
            failure_classification: Why it failed (if reverted)
            failure_diagnosis: Short diagnosis
            training_time: Time taken
            memory_used: Memory used
        """
        experiment.val_bpb_after = val_bpb_after
        experiment.status = status.value
        experiment.training_time = training_time
        experiment.memory_used = memory_used

        if status == ExperimentStatus.REVERTED:
            experiment.failure_classification = (
                failure_classification.value if failure_classification else None
            )
            experiment.failure_diagnosis = failure_diagnosis

        # Save to file
        self._save_experiment(experiment)

    def _save_experiment(self, experiment: Experiment):
        """Append experiment to log file."""
        with open(self.experiment_log_path, "a") as f:
            f.write(json.dumps(experiment.to_dict()) + "\n")

    def classify_failure(
        self,
        val_bpb_before: float,
        val_bpb_after: float,
        training_loss: float,
        eval_loss: float,
        training_stable: bool = True,
    ) -> FailureClassification:
        """Classify why an experiment failed.

        Args:
            val_bpb_before: Baseline val_bpb
            val_bpb_after: New val_bpb
            training_loss: Final training loss
            eval_loss: Final eval loss
            training_stable: Whether training was stable

        Returns:
            Failure classification
        """
        delta = val_bpb_after - val_bpb_before
        ratio = eval_loss / training_loss if training_loss > 0 else 1.0

        if not training_stable:
            if training_loss > 10:
                return FailureClassification.GRADIENT_EXPLOSION
            elif training_loss < 0.001 and eval_loss > training_loss:
                return FailureClassification.GRADIENT_VANISHING
            else:
                return FailureClassification.LOSS_SPIKE

        # Check for overfitting (eval much worse than train)
        if ratio > 2.0:
            return FailureClassification.OVERFITTING

        # Check for underfitting (both high)
        if training_loss > 5.0 and eval_loss > 5.0:
            return FailureClassification.UNDERFITTING

        # Check if it just didn't help (small delta)
        if abs(delta) < 0.01:
            return FailureClassification.TIMING

        # Check if it made things worse
        if delta > 0:
            # Got worse - guess why based on magnitude
            if delta > 0.5:
                return FailureClassification.LR_TOO_HIGH
            else:
                return FailureClassification.LR_TOO_LOW

        return FailureClassification.UNKNOWN

    def get_baseline(self) -> float:
        """Get baseline val_bpb."""
        if self.baseline_val_bpb is not None:
            return self.baseline_val_bpb

        # Try to find from kept experiments
        kept = [e for e in self.experiments if e.status == "kept"]
        if kept:
            return min(e.val_bpb_after for e in kept)

        return float("inf")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments."""
        if not self.experiments:
            return {"total": 0, "kept": 0, "reverted": 0}

        kept = [e for e in self.experiments if e.status == "kept"]
        reverted = [e for e in self.experiments if e.status == "reverted"]

        return {
            "total": len(self.experiments),
            "kept": len(kept),
            "reverted": len(reverted),
            "baseline_val_bpb": self.get_baseline(),
            "best_val_bpb": min(e.val_bpb_after for e in kept) if kept else None,
            "total_improvement": (
                self.get_baseline() - min(e.val_bpb_after for e in kept)
            )
            if kept
            else 0,
        }

    def get_recent_failures(self, n: int = 10) -> List[Experiment]:
        """Get recent failed experiments."""
        reverted = [e for e in self.experiments if e.status == "reverted"]
        return sorted(reverted, key=lambda x: x.id, reverse=True)[:n]


def load_feedback(log_path: str = "./experiments.jsonl") -> Feedback:
    """Load existing feedback system."""
    return Feedback(experiment_log_path=log_path)


# Backward compatibility
class LegacyFeedback:
    """Legacy feedback for backward compatibility."""

    def reward(self, val_bpb: float, score: float) -> float:
        return (1.0 / (val_bpb + 1e-6)) + score / 100.0

    def log(self, step: int, val_bpb: float, score: float) -> float:
        r = self.reward(val_bpb, score)
        print(f"R: {step} {val_bpb} {score} {r}")
        return r
