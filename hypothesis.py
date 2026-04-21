"""
Hypothesis generation for autonomous research.

Features:
- LLM-driven hypothesis proposer
- Rule-based fallback
- Integration with memory system
- Failure pattern analysis
"""

import random
import os
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ChangeType(Enum):
    """Type of change to propose."""

    OPTIMIZATION = "optimization"
    ARCHITECTURE = "architecture"
    CURRICULUM = "curriculum"
    SYNTHETIC = "synthetic"


class HypothesisCategory(Enum):
    """Category of hypothesis."""

    LEARNING_RATE = "learning_rate"
    BATCH_SIZE = "batch_size"
    WEIGHT_DECAY = "weight_decay"
    WARMUP = "warmup"
    DROPOUT = "dropout"
    LAYER_SIZE = "layer_size"
    ATTENTION = "attention"
    CURRICULUM_TIMING = "curriculum_timing"
    DIFFICULTY = "difficulty"
    SYNTHETIC_COUNT = "synthetic_count"


@dataclass
class Hypothesis:
    """A single hypothesis/proposal."""

    change: str
    description: str
    change_type: str
    hypothesis_type: str
    expected_impact: str  # high/medium/low
    reasoning: str
    code_diff: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "change": self.change,
            "description": self.description,
            "change_type": self.change_type,
            "hypothesis_type": self.hypothesis_type,
            "expected_impact": self.expected_impact,
            "reasoning": self.reasoning,
            "code_diff": self.code_diff,
        }


# Rule-based templates
HYPOTHESIS_TEMPLATES = {
    ChangeType.OPTIMIZATION: {
        HypothesisCategory.LEARNING_RATE: [
            Hypothesis(
                change="learning_rate",
                description="Increase learning rate by 10%",
                change_type="optimization",
                hypothesis_type="learning_rate",
                expected_impact="medium",
                reasoning="Higher lr can accelerate convergence on plateau",
                code_diff="config.model.learning_rate *= 1.1",
            ),
            Hypothesis(
                change="learning_rate",
                description="Decrease learning rate by 10%",
                change_type="optimization",
                hypothesis_type="learning_rate",
                expected_impact="medium",
                reasoning="Lower lr can improve stability and convergence",
                code_diff="config.model.learning_rate *= 0.9",
            ),
            Hypothesis(
                change="learning_rate",
                description="Add learning rate warmup",
                change_type="optimization",
                hypothesis_type="learning_rate",
                expected_impact="high",
                reasoning="Warmup improves early training stability",
                code_diff="config.model.warmup_steps = 200",
            ),
        ],
        HypothesisCategory.BATCH_SIZE: [
            Hypothesis(
                change="batch_size",
                description="Double batch size",
                change_type="optimization",
                hypothesis_type="batch_size",
                expected_impact="medium",
                reasoning="Larger batches provide smoother gradients",
                code_diff="config.model.batch_size *= 2",
            ),
            Hypothesis(
                change="batch_size",
                description="Halve batch size",
                change_type="optimization",
                hypothesis_type="batch_size",
                expected_impact="low",
                reasoning="Smaller batches add noise that can help generalization",
                code_diff="config.model.batch_size //= 2",
            ),
        ],
        HypothesisCategory.WEIGHT_DECAY: [
            Hypothesis(
                change="weight_decay",
                description="Increase weight decay by 50%",
                change_type="optimization",
                hypothesis_type="weight_decay",
                expected_impact="medium",
                reasoning="More regularization can prevent overfitting",
                code_diff="config.model.weight_decay *= 1.5",
            ),
            Hypothesis(
                change="weight_decay",
                description="Remove weight decay",
                change_type="optimization",
                hypothesis_type="weight_decay",
                expected_impact="low",
                reasoning="Less regularization for small datasets",
                code_diff="config.model.weight_decay = 0.0",
            ),
        ],
    },
    ChangeType.ARCHITECTURE: {
        HypothesisCategory.DROPOUT: [
            Hypothesis(
                change="dropout",
                description="Add dropout (0.1)",
                change_type="architecture",
                hypothesis_type="dropout",
                expected_impact="medium",
                reasoning="Dropout prevents overfitting",
                code_diff="config.model.dropout = 0.1",
            ),
            Hypothesis(
                change="dropout",
                description="Increase dropout to 0.2",
                change_type="architecture",
                hypothesis_type="dropout",
                expected_impact="medium",
                reasoning="More regularization needed",
                code_diff="config.model.dropout = 0.2",
            ),
        ],
        HypothesisCategory.LAYER_SIZE: [
            Hypothesis(
                change="n_embd",
                description="Increase embedding dimension by 20%",
                change_type="architecture",
                hypothesis_type="layer_size",
                expected_impact="high",
                reasoning="Larger capacity can fit more complex patterns",
                code_diff="config.model.n_embd = int(config.model.n_embd * 1.2)",
            ),
        ],
    },
    ChangeType.CURRICULUM: {
        HypothesisCategory.CURRICULUM_TIMING: [
            Hypothesis(
                change="warmup_ratio",
                description="Increase warmup ratio to 20%",
                change_type="curriculum",
                hypothesis_type="curriculum_timing",
                expected_impact="medium",
                reasoning="More warmup time improves stability",
                code_diff="config.curriculum.warmup_ratio = 0.2",
            ),
            Hypothesis(
                change="adaptive_curriculum",
                description="Enable adaptive curriculum",
                change_type="curriculum",
                hypothesis_type="curriculum_timing",
                expected_impact="high",
                reasoning="Adaptive scheduling responds to training",
                code_diff="config.curriculum.adaptive = True",
            ),
        ],
        HypothesisCategory.DIFFICULTY: [
            Hypothesis(
                change="difficulty_metric",
                description="Use perplexity-based difficulty",
                change_type="curriculum",
                hypothesis_type="difficulty",
                expected_impact="medium",
                reasoning="Perplexity measures actual difficulty",
                code_diff="config.curriculum.difficulty_metric = 'perplexity'",
            ),
        ],
    },
    ChangeType.SYNTHETIC: {
        HypothesisCategory.SYNTHETIC_COUNT: [
            Hypothesis(
                change="n_samples",
                description="Double synthetic samples",
                change_type="synthetic",
                hypothesis_type="synthetic_count",
                expected_impact="medium",
                reasoning="More training data improves generalization",
                code_diff="config.synthetic.n_samples *= 2",
            ),
            Hypothesis(
                change="temperature",
                description="Increase generation temperature",
                change_type="synthetic",
                hypothesis_type="synthetic_count",
                expected_impact="low",
                reasoning="More diverse synthetic data",
                code_diff="config.synthetic.temperature = 1.0",
            ),
        ],
    },
}


class HypothesisGenerator:
    """Generate hypotheses for experiments."""

    def __init__(
        self,
        use_llm: bool = False,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-20250514",
    ):
        self.use_llm = use_llm
        self.provider = provider
        self.model = model
        self.api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")

        # Track generation stats
        self.stats = {
            "llm_calls": 0,
            "template_calls": 0,
            "total_generated": 0,
        }

    def generate(
        self,
        n: int = 1,
        change_type: Optional[str] = None,
        memory_context: Optional[List[Dict]] = None,
    ) -> List[Hypothesis]:
        """Generate hypotheses.

        Args:
            n: Number to generate
            change_type: Preferred change type
            memory_context: Past experiments for context
        """
        if change_type is None:
            change_type = random.choice(list(ChangeType._value2member_map_.keys()))

        change_type_enum = ChangeType._value2member_map_.get(
            change_type, ChangeType.OPTIMIZATION
        )

        if self.use_llm and self.api_key:
            return self._generate_with_llm(n, change_type_enum, memory_context)
        else:
            return self._generate_templates(n, change_type_enum)

    def _generate_templates(
        self,
        n: int,
        change_type: ChangeType,
    ) -> List[Hypothesis]:
        """Generate from templates."""
        category_dict = HYPOTHESIS_TEMPLATES.get(change_type, {})

        if not category_dict:
            # Fallback to optimization
            category_dict = HYPOTHESIS_TEMPLATES.get(ChangeType.OPTIMIZATION, {})

        # Flatten all hypotheses from this type
        all_hypotheses = []
        for hypotheses in category_dict.values():
            all_hypotheses.extend(hypotheses)

        if not all_hypotheses:
            return []

        # Select n random unique hypotheses
        selected = random.sample(all_hypotheses, min(n, len(all_hypotheses)))

        self.stats["template_calls"] += n
        self.stats["total_generated"] += n

        return selected

    def _generate_with_llm(
        self,
        n: int,
        change_type: ChangeType,
        memory_context: Optional[List[Dict]],
    ) -> List[Hypothesis]:
        """Generate using LLM."""
        # Build prompt
        system_prompt = """You are a research hypothesis generator for LLM training.
Generate experimental hypotheses that could improve val_bpb.
Output ONLY a JSON array of objects with: change, description, change_type, expected_impact, reasoning"""

        user_prompt = f"""Generate {n} hypotheses for {change_type.value}.
Prioritize high-impact changes based on training dynamics.
"""

        # Add memory context if available
        if memory_context:
            recent = memory_context[-5:]
            user_prompt += "\nRecent experiments:\n"
            for exp in recent:
                user_prompt += f"- {exp.get('change', 'unknown')}: {exp.get('status', 'unknown')}\n"

        try:
            if self.provider == "anthropic":
                return self._call_anthropic(system_prompt, user_prompt, n)
            elif self.provider == "openai":
                return self._call_openai(system_prompt, user_prompt, n)
        except Exception as e:
            print(f"LLM generation failed: {e}")
            return self._generate_templates(n, change_type)

    def _call_anthropic(self, system: str, user: str, n: int) -> List[Hypothesis]:
        """Call Anthropic API."""
        import anthropic
        import json

        client = anthropic.Anthropic(api_key=self.api_key)

        response = client.messages.create(
            model=self.model,
            max_tokens=2048,
            temperature=0.7,
            system=system,
            messages=[{"role": "user", "content": user}],
        )

        content = response.content[0].text

        # Parse JSON
        try:
            data = json.loads(content)
            if isinstance(data, list):
                hypotheses = []
                for item in data:
                    # Validate required fields
                    if "change" in item and "description" in item:
                        hypotheses.append(
                            Hypothesis(
                                change=item.get("change", ""),
                                description=item.get("description", ""),
                                change_type=item.get("change_type", "optimization"),
                                hypothesis_type=item.get("change", "learning_rate"),
                                expected_impact=item.get("expected_impact", "medium"),
                                reasoning=item.get("reasoning", ""),
                                code_diff=item.get("code_diff", ""),
                            )
                        )
                return hypotheses
        except json.JSONDecodeError:
            pass

        # Fallback
        return self._generate_templates(n, ChangeType.OPTIMIZATION)

    def _call_openai(self, system: str, user: str, n: int) -> List[Hypothesis]:
        """Call OpenAI API."""
        import openai
        import json

        client = openai.OpenAI(api_key=self.api_key)

        response = client.chat.completions.create(
            model=self.model,
            max_tokens=2048,
            temperature=0.7,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )

        content = response.choices[0].message.content

        # Parse JSON
        try:
            data = json.loads(content)
            if isinstance(data, list):
                hypotheses = []
                for item in data:
                    if "change" in item and "description" in item:
                        hypotheses.append(
                            Hypothesis(
                                change=item.get("change", ""),
                                description=item.get("description", ""),
                                change_type=item.get("change_type", "optimization"),
                                hypothesis_type=item.get("change", "learning_rate"),
                                expected_impact=item.get("expected_impact", "medium"),
                                reasoning=item.get("reasoning", ""),
                                code_diff=item.get("code_diff", ""),
                            )
                        )
                return hypotheses
        except json.JSONDecodeError:
            pass

        return self._generate_templates(n, ChangeType.OPTIMIZATION)

    def generate_from_analysis(
        self,
        training_loss: float,
        val_bpb: float,
        memory_context: Optional[List[Dict]] = None,
    ) -> Hypothesis:
        """Generate hypothesis based on analysis of training state."""

        # Simple rule-based analysis
        if training_loss > 5.0:
            # High loss - underfitting
            return Hypothesis(
                change="learning_rate",
                description="Try higher learning rate (2x)",
                change_type="optimization",
                hypothesis_type="learning_rate",
                expected_impact="high",
                reasoning="High training loss suggests lr too low",
                code_diff="config.model.learning_rate *= 2",
            )
        elif training_loss < 0.1 and val_bpb > training_loss * 2:
            # Possible overfitting
            return Hypothesis(
                change="dropout",
                description="Add dropout to prevent overfitting",
                change_type="architecture",
                hypothesis_type="dropout",
                expected_impact="high",
                reasoning="Large gap between train and val suggests overfitting",
                code_diff="config.model.dropout = 0.1",
            )
        elif val_bpb > 1.0:
            # Poor baseline
            return Hypothesis(
                change="n_embd",
                description="Increase model capacity",
                change_type="architecture",
                hypothesis_type="layer_size",
                expected_impact="high",
                reasoning="Large val_bpb suggests model too small",
                code_diff="config.model.n_embd = int(config.model.n_embd * 1.5)",
            )
        else:
            # General optimization
            return random.choice(self._generate_templates(1, ChangeType.OPTIMIZATION))


def generate_hypothesis(
    use_llm: bool = False,
    change_type: Optional[str] = None,
) -> Hypothesis:
    """Convenience function."""
    gen = HypothesisGenerator(use_llm=use_llm)
    return gen.generate(n=1, change_type=change_type)[0]


if __name__ == "__main__":
    # Demo
    gen = HypothesisGenerator(use_llm=False)

    print("Generating hypotheses:")
    for h in gen.generate(n=5):
        print(f"  - {h.description} ({h.change_type}, {h.expected_impact})")
        print(f"    Reason: {h.reasoning}")
