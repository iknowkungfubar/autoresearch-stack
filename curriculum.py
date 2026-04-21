"""
Curriculum learning for LLM training.

Features:
- Difficulty-based staging (easy/medium/hard)
- Adaptive scheduling based on performance
- Integration with curriculus library (optional)
- Multiple difficulty metrics
"""

import random
import math
import bisect
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum


class DifficultyMetric(Enum):
    """How to measure example difficulty."""

    LENGTH = "length"  # Log of text length
    PERPLEXITY = "perplexity"  # Model perplexity (if available)
    COMPLEXITY = "complexity"  # Syntactic complexity
    HYBRID = "hybrid"  # Combination


class ScheduleStrategy(Enum):
    """Curriculum scheduling strategies."""

    LINEAR = "linear"  # Linear progression
    EXPONENTIAL = "exponential"  # Exponential
    STEP = "step"  # Step function
    ADAPTIVE = "adaptive"  # Performance-based


@dataclass
class DifficultyScore:
    """Difficulty score for a text."""

    text: str
    score: float
    source: str = "length"  # How it was computed


class AdaptiveScheduler:
    """Performance-aware curriculum scheduler."""

    def __init__(
        self,
        curriculum: Dict[str, List[str]],
        strategy: str = "linear",
        warmup_ratio: float = 0.1,
        window_size: int = 50,
    ):
        self.curriculum = curriculum
        self.strategy = ScheduleStrategy(strategy)
        self.warmup_ratio = warmup_ratio
        self.window_size = window_size

        # Track performance for adaptive scheduling
        self.loss_history: List[float] = []
        self.stage_improvements: Dict[str, List[float]] = {
            "easy": [],
            "medium": [],
            "hard": [],
        }

        # Current state
        self.current_stage = "easy"
        self.stage_transitions = 0

    def get_stage(self, step: int, total: int) -> str:
        """Get current curriculum stage based on progress."""
        if total == 0:
            return "easy"

        progress = step / total

        # Warmup phase
        if progress < self.warmup_ratio:
            return "easy"

        # Adjust progress for warmup
        adjusted_progress = (progress - self.warmup_ratio) / (1 - self.warmup_ratio)

        if self.strategy == ScheduleStrategy.LINEAR:
            return self._linear_stage(adjusted_progress)
        elif self.strategy == ScheduleStrategy.EXPONENTIAL:
            return self._exponential_stage(adjusted_progress)
        elif self.strategy == ScheduleStrategy.STEP:
            return self._step_stage(adjusted_progress)
        elif self.strategy == ScheduleStrategy.ADAPTIVE:
            return self._adaptive_stage(step)

        return "medium"

    def _linear_stage(self, progress: float) -> str:
        """Linear stage progression."""
        if progress < 0.33:
            return "easy"
        elif progress < 0.66:
            return "medium"
        return "hard"

    def _exponential_stage(self, progress: float) -> str:
        """Exponential stage progression - harder earlier."""
        exp_p = progress**0.5  # Square root makes it push harder earlier
        if exp_p < 0.33:
            return "easy"
        elif exp_p < 0.66:
            return "medium"
        return "hard"

    def _step_stage(self, progress: float) -> str:
        """Step function - stay in each stage longer."""
        if progress < 0.25:
            return "easy"
        elif progress < 0.5:
            return "medium"
        elif progress < 0.75:
            return "medium"  # Extended medium
        return "hard"

    def _adaptive_stage(self, step: int) -> str:
        """Adaptive based on recent loss improvement."""
        if len(self.loss_history) < self.window_size:
            return "easy"

        # Calculate recent improvement
        recent = self.loss_history[-self.window_size :]
        if len(recent) < 2:
            return self.current_stage

        # Improvement rate
        improvement = (recent[0] - recent[-1]) / recent[0] if recent[0] > 0 else 0

        # If improving fast, advance; if stagnating, stay or regress
        if improvement > 0.1:  # >10% improvement
            if self.current_stage == "easy":
                return "medium"
        elif improvement < 0.01:  # <1% improvement
            if self.current_stage == "hard":
                return "medium"

        return self.current_stage

    def update_performance(self, loss: float, stage: str):
        """Update scheduler with training performance."""
        self.loss_history.append(loss)

        # Trim history
        if len(self.loss_history) > 1000:
            self.loss_history = self.loss_history[-500:]

        # Track per-stage performance
        if stage in self.stage_improvements:
            self.stage_improvements[stage].append(loss)

    def sample(self, stage: str) -> str:
        """Sample a text from the current stage."""
        if stage not in self.curriculum:
            stage = "medium"  # Fallback

        texts = self.curriculum.get(stage, [])
        if not texts:
            # Fallback to any available
            for s in ["easy", "medium", "hard"]:
                texts = self.curriculum.get(s, [])
                if texts:
                    stage = s
                    break

        if not texts:
            return ""

        return random.choice(texts)

    def get_stage_ratio(self, step: int, total: int) -> Dict[str, float]:
        """Get probability distribution across stages."""
        stage = self.get_stage(step, total)
        if stage == "easy":
            return {"easy": 0.7, "medium": 0.2, "hard": 0.1}
        elif stage == "medium":
            return {"easy": 0.2, "medium": 0.6, "hard": 0.2}
        else:
            return {"easy": 0.1, "medium": 0.3, "hard": 0.6}


class Scheduler:
    """Simple scheduler for backward compatibility."""

    def __init__(self, curriculum: Dict[str, List[str]]):
        self.curriculum = curriculum
        self._random = random.Random()

    def get_stage(self, step: int, total: int) -> str:
        """Get stage based on progress."""
        if total == 0:
            return "medium"

        p = step / total
        if p < 0.33:
            return "easy"
        elif p < 0.66:
            return "medium"
        return "hard"

    def sample(self, stage: str) -> str:
        """Sample from a stage."""
        texts = self.curriculum.get(stage, self.curriculum.get("medium", []))
        if not texts:
            return ""
        return self._random.choice(texts)


def compute_difficulty(text: str, metric: str = "length") -> float:
    """Compute difficulty score for a text.

    Args:
        text: The text to score
        metric: Difficulty metric to use

    Returns:
        Difficulty score (higher = harder)
    """
    if metric == "length":
        return math.log1p(len(text))
    elif metric == "complexity":
        # Count sentence-like segments
        sentences = text.count(".") + text.count("!") + text.count("?")
        # Count distinct word types
        words = text.split()
        diversity = len(set(words)) / max(len(words), 1)
        # Combine
        return math.log1p(len(text)) * (1 + diversity)
    elif metric == "hybrid":
        length_score = math.log1p(len(text))
        words = text.split()
        avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
        return length_score * (1 + avg_word_len / 10)
    else:
        return math.log1p(len(text))


def build_curriculum(
    texts: List[str],
    stages: int = 3,
    metric: str = "length",
    custom_bins: Optional[List[float]] = None,
) -> Dict[str, List[str]]:
    """Build curriculum from texts.

    Args:
        texts: List of training texts
        stages: Number of difficulty stages
        metric: Difficulty metric to use
        custom_bins: Optional custom difficulty boundaries

    Returns:
        Dictionary mapping stage names to text lists
    """
    if not texts:
        return {"easy": [], "medium": [], "hard": []}

    # Compute difficulty for each text
    scored = [(t, compute_difficulty(t, metric)) for t in texts]
    scored.sort(key=lambda x: x[1])

    # Determine bins
    n = len(scored)

    if custom_bins:
        # Use custom bins
        stage_names = ["easy", "medium", "hard"]
        curriculum = {s: [] for s in stage_names[:stages]}

        for text, score in scored:
            # Find appropriate bin
            for i, bound in enumerate(custom_bins):
                if score < bound:
                    curriculum[stage_names[i]].append(text)
                    break
            else:
                curriculum[stage_names[-1]].append(text)
    else:
        # Equal split
        stage_size = n // stages

        if stages == 3:
            curriculum = {
                "easy": [t for t, _ in scored[: n // 3]],
                "medium": [t for t, _ in scored[n // 3 : 2 * n // 3]],
                "hard": [t for t, _ in scored[2 * n // 3 :]],
            }
        elif stages == 2:
            curriculum = {
                "easy": [t for t, _ in scored[: n // 2]],
                "hard": [t for t, _ in scored[n // 2 :]],
            }
        else:
            # Dynamic stages
            step = n // stages
            curriculum = {}
            for i in range(stages):
                start = i * step
                end = (i + 1) * step if i < stages - 1 else n
                stage_name = f"stage_{i}"
                curriculum[stage_name] = [t for t, _ in scored[start:end]]

    return curriculum


def create_scheduler(
    texts: List[str],
    strategy: str = "linear",
    adaptive: bool = False,
    warmup_ratio: float = 0.1,
) -> Any:
    """Create a curriculum scheduler.

    Args:
        texts: Training texts
        strategy: Scheduling strategy
        adaptive: Use adaptive scheduling
        warmup_ratio: Warmup proportion

    Returns:
        Scheduler instance
    """
    curriculum = build_curriculum(texts)

    if adaptive:
        return AdaptiveScheduler(
            curriculum=curriculum,
            strategy=strategy,
            warmup_ratio=warmup_ratio,
        )
    else:
        return Scheduler(curriculum)


# Backward compatibility
def difficulty(t: str) -> float:
    """Legacy difficulty function."""
    return compute_difficulty(t, "length")
