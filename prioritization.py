"""
Experiment prioritization using bandit-based selection.

Features:
- UCB1-based arm selection
- Epsilon-greedy exploration
- Thompson sampling (optional)
- Adaptive exploration rate
"""

import math
import random
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


class Strategy(Enum):
    """Selection strategy."""

    UCB1 = "ucb1"
    EPSILON_GREEDY = "epsilon_greedy"
    THOMPSON = "thompson"
    RANDOM = "random"


@dataclass
class Arm:
    """A single "arm" (experiment type) for multi-armed bandit."""

    name: str
    pull_count: int = 0
    total_reward: float = 0.0
    total_sq_reward: float = 0.0  # For variance

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.pull_count if self.pull_count > 0 else 0.0

    @property
    def variance(self) -> float:
        if self.pull_count < 2:
            return float("inf")
        mean = self.mean_reward
        var = (self.total_sq_reward / self.pull_count) - (mean**2)
        return max(0, var)


class BanditSelector:
    """Multi-armed bandit for experiment selection."""

    # Default arms (change types)
    DEFAULT_ARMS = {
        "optimization": [
            "learning_rate",
            "batch_size",
            "weight_decay",
            "warmup_steps",
            "optimizer",
        ],
        "architecture": [
            "dropout",
            "layer_size",
            "attention_heads",
            "hidden_size",
        ],
        "curriculum": [
            "stage_timing",
            "warmup_ratio",
            "difficulty_scaling",
        ],
        "synthetic": [
            "sample_count",
            "temperature",
            "difficulty_mix",
        ],
    }

    def __init__(
        self,
        strategy: str = "ucb1",
        epsilon: float = 0.1,
        exploration_constant: float = 1.0,
    ):
        self.strategy = Strategy(strategy)
        self.epsilon = epsilon
        self.c = exploration_constant

        # Initialize arms
        self.arms: Dict[str, Dict[str, Arm]] = defaultdict(dict)
        self._init_default_arms()

        # Statistics
        self.total_pulls = 0
        self.history: List[Tuple[str, float]] = []  # (arm_name, reward)

    def _init_default_arms(self):
        """Initialize default arms."""
        for category, arm_names in self.DEFAULT_ARMS.items():
            self.arms[category] = {name: Arm(name=name) for name in arm_names}

    def add_arm(self, category: str, arm_name: str):
        """Add a new arm."""
        if category not in self.arms:
            self.arms[category] = {}
        if arm_name not in self.arms[category]:
            self.arms[category][arm_name] = Arm(name=arm_name)

    def select(self, category: Optional[str] = None) -> str:
        """Select an arm to pull."""
        if category is None:
            # Random category
            category = random.choice(list(self.arms.keys()))

        arm_dict = self.arms.get(category, {})
        if not arm_dict:
            return "learning_rate"  # Default

        if self.strategy == Strategy.UCB1:
            return self._ucb1_select(arm_dict)
        elif self.strategy == Strategy.EPSILON_GREEDY:
            return self._epsilon_greedy_select(arm_dict)
        elif self.strategy == Strategy.THOMPSON:
            return self._thompson_select(arm_dict)
        else:
            return random.choice(list(arm_dict.keys()))

    def _ucb1_select(self, arms: Dict[str, Arm]) -> str:
        """UCB1 selection."""
        # First, try unexplored arms
        for name, arm in arms.items():
            if arm.pull_count == 0:
                return name

        # Calculate UCB1 for each
        total_pulls = sum(a.pull_count for a in arms.values())

        best_score = float("-inf")
        best_arm = list(arms.keys())[0]

        for name, arm in arms.items():
            if arm.pull_count == 0:
                return name

            # UCB1 formula
            exploitation = arm.mean_reward
            exploration = self.c * math.sqrt(math.log(total_pulls) / arm.pull_count)
            score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_arm = name

        return best_arm

    def _epsilon_greedy_select(self, arms: Dict[str, Arm]) -> str:
        """Epsilon-greedy selection."""
        if random.random() < self.epsilon:
            return random.choice(list(arms.keys()))

        # Exploit: best known
        return max(arms.keys(), key=lambda n: arms[n].mean_reward)

    def _thompson_select(self, arms: Dict[str, Arm]) -> str:
        """Thompson sampling with Gaussian posterior."""
        samples = []

        for name, arm in arms.items():
            if arm.pull_count < 2:
                samples.append((random.random(), name))
            else:
                # Sample from Gaussian posterior
                from random import gauss

                sample = gauss(arm.mean_reward, math.sqrt(arm.variance))
                samples.append((sample, name))

        return max(samples, key=lambda x: x[0])[1]

    def update(self, arm_name: str, reward: float, category: Optional[str] = None):
        """Update arm with reward."""
        if category is None:
            # Find the arm
            for cat, arm_dict in self.arms.items():
                if arm_name in arm_dict:
                    category = cat
                    break

        if category is None:
            return

        arm = self.arms.get(category, {}).get(arm_name)
        if arm is None:
            return

        arm.pull_count += 1
        arm.total_reward += reward
        arm.total_sq_reward += reward**2
        self.total_pulls += 1

        # Record history
        self.history.append((arm_name, reward))

    def get_statistics(self, category: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for arms."""
        stats = {}

        arms_to_check = (
            self.arms.get(category, {})
            if category
            else {k: v for cat in self.arms.values() for k, v in cat.items()}
        )

        for name, arm in arms_to_check.items():
            stats[name] = {
                "pulls": arm.pull_count,
                "mean_reward": arm.mean_reward,
                "variance": arm.variance,
            }

        return stats

    def get_best_arm(self, category: str) -> str:
        """Get best arm by mean reward."""
        arm_dict = self.arms.get(category, {})
        if not arm_dict:
            return "learning_rate"
        return max(arm_dict.keys(), key=lambda n: arm_dict[n].mean_reward)


class PrioritizationSystem:
    """System that prioritizes experiments using bandit selection."""

    def __init__(self, strategy: str = "ucb1"):
        self.selector = BanditSelector(strategy=strategy)
        self.experiment_history: List[Dict[str, Any]] = []

    def suggest_next(self, baseline_val_bpb: float) -> Dict[str, Any]:
        """Suggest next experiment based on history."""
        # Calculate recent improvements
        recent = self.experiment_history[-20:] if self.experiment_history else []

        if recent:
            # Track rewards by change type
            type_rewards: Dict[str, List[float]] = defaultdict(list)

            for exp in recent:
                delta = exp.get("val_bpb_before", 1.0) - exp.get("val_bpb_after", 1.0)
                reward = delta * 100  # Scale
                change_type = exp.get("change_type", "optimization")
                type_rewards[change_type].append(reward)

            # Update bandit
            for change_type, rewards in type_rewards.items():
                if rewards:
                    mean_reward = sum(rewards) / len(rewards)
                    # Update arms based on changes
                    for exp in recent:
                        arm = exp.get("change", "learning_rate")
                        self.selector.update(arm, mean_reward, change_type)

        # Select next category and arm
        category = self.selector.select()
        arm = self.selector.select(category)

        return {
            "category": category,
            "change": arm,
            "reasoning": f"Selected {arm} from {category} using {self.selector.strategy.value}",
        }

    def record_result(
        self,
        change: str,
        change_type: str,
        val_bpb_before: float,
        val_bpb_after: float,
    ):
        """Record experiment result."""
        delta = val_bpb_before - val_bpb_after
        reward = delta * 100  # Positive if improved

        self.selector.update(change, reward, change_type)

        self.experiment_history.append(
            {
                "change": change,
                "change_type": change_type,
                "val_bpb_before": val_bpb_before,
                "val_bpb_after": val_bpb_after,
                "reward": reward,
            }
        )


def get_prioritization(strategy: str = "ucb1") -> PrioritizationSystem:
    """Get prioritization system."""
    return PrioritizationSystem(strategy=strategy)


if __name__ == "__main__":
    # Demo
    ps = get_prioritization("ucb1")

    # Simulate some history
    for _ in range(10):
        suggestion = ps.suggest_next(1.0)
        print(f"Suggest: {suggestion}")

        # Record a result (simulated)
        ps.record_result(
            change=suggestion["change"],
            change_type=suggestion["category"],
            val_bpb_before=1.0,
            val_bpb_after=0.99,
        )

    print("\nArm statistics:")
    for arm_name, stats in ps.selector.get_statistics().items():
        print(f"  {arm_name}: {stats}")
