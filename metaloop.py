"""
Meta-Loop - Self-modification and methodology evolution.

Phase 6.1: Self-Modification.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# Optional LLM import
try:
    from anthropic import Anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class ModificationType(Enum):
    """Types of self-modification."""

    PROMPT = "prompt"
    HYPERPARAMETER = "hyperparameter"
    STRATEGY = "strategy"
    ARCHITECTURE = "architecture"


@dataclass
class PromptTemplate:
    """Prompt template with versioning."""

    name: str
    version: int
    content: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    performance: Optional[float] = None
    notes: str = ""


@dataclass
class Modification:
    """A modification to the system."""

    id: str
    type: ModificationType
    description: str
    old_value: str
    new_value: str
    expected_impact: float
    actual_impact: Optional[float] = None
    status: str = "pending"  # pending, applied, reverted, failed
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class MetaConfig:
    """Meta-loop configuration."""

    max_iterations: int = 10
    min_improvement: float = 0.01
    prompt_template_file: str = "prompts.json"
    modifications_file: str = "modifications.json"
    llm_model: str = "claude-3-opus-20240229"


class MetaLoop:
    """
    Meta-learning loop for self-improvement.

    This system can:
    - Evolve prompts based on performance
    - Modify hyperparameters automatically
    - Adapt strategies based on results
    - Learn from past modifications
    """

    def __init__(self, config: Optional[MetaConfig] = None):
        self.config = config or MetaConfig()
        self.prompts: Dict[str, PromptTemplate] = {}
        self.modifications: List[Modification] = []
        self.iteration: int = 0

        self._load_prompts()
        self._load_modifications()

    def _load_prompts(self):
        """Load prompt templates from file."""
        path = Path(self.config.prompt_template_file)
        if path.exists():
            data = json.loads(path.read_text())
            for name, versions in data.items():
                for v in versions:
                    self.prompts[f"{name}_v{v['version']}"] = PromptTemplate(**v)

    def _save_prompts(self):
        """Save prompt templates to file."""
        data = {}
        for name, prompt in self.prompts.items():
            base_name = name.rsplit("_v", 1)[0]
            if base_name not in data:
                data[base_name] = []
            data[base_name].append(
                {
                    "name": prompt.name,
                    "version": prompt.version,
                    "content": prompt.content,
                    "created_at": prompt.created_at,
                    "performance": prompt.performance,
                    "notes": prompt.notes,
                }
            )
        Path(self.config.prompt_template_file).write_text(json.dumps(data, indent=2))

    def _load_modifications(self):
        """Load modification history."""
        path = Path(self.config.modifications_file)
        if path.exists():
            data = json.loads(path.read_text())
            for m in data:
                m["type"] = ModificationType(m["type"])
            self.modifications = [Modification(**m) for m in data]

    def _save_modifications(self):
        """Save modification history."""
        data = [
            {
                "id": m.id,
                "type": m.type.value,
                "description": m.description,
                "old_value": m.old_value,
                "new_value": m.new_value,
                "expected_impact": m.expected_impact,
                "actual_impact": m.actual_impact,
                "status": m.status,
                "timestamp": m.timestamp,
            }
            for m in self.modifications
        ]
        Path(self.config.modifications_file).write_text(json.dumps(data, indent=2))

    def register_prompt(self, name: str, content: str) -> PromptTemplate:
        """Register a new prompt template."""
        # Find latest version
        latest_version = 0
        for key, p in self.prompts.items():
            if p.name == name:
                latest_version = max(latest_version, p.version)

        prompt = PromptTemplate(
            name=name,
            version=latest_version + 1,
            content=content,
        )
        self.prompts[f"{name}_v{prompt.version}"] = prompt
        self._save_prompts()
        return prompt

    def evolve_prompt(
        self,
        name: str,
        feedback: str,
        current_performance: float,
    ) -> PromptTemplate:
        """
        Evolve a prompt based on feedback.

        Uses the LLM to improve the prompt.
        """
        # Get current prompt
        current = None
        for key, p in self.prompts.items():
            if p.name == name:
                if current is None or p.version > current.version:
                    current = p

        if current is None:
            raise ValueError(f"No prompt found with name: {name}")

        # Evolve using LLM if available
        if ANTHROPIC_AVAILABLE:
            new_content = self._evolve_with_llm(current.content, feedback)
        else:
            # Fallback: simple heuristics
            new_content = self._evolve_heuristic(current.content, feedback)

        # Create new version
        new_prompt = PromptTemplate(
            name=name,
            version=current.version + 1,
            content=new_content,
            notes=f"Evolved from v{current.version} based on feedback: {feedback[:100]}",
        )

        self.prompts[f"{name}_v{new_prompt.version}"] = new_prompt
        self._save_prompts()

        # Record modification
        mod = Modification(
            id=f"mod_{len(self.modifications) + 1}",
            type=ModificationType.PROMPT,
            description=f"Evolved prompt {name} from v{current.version} to v{new_prompt.version}",
            old_value=current.content[:200],
            new_value=new_content[:200],
            expected_impact=0.05,
        )
        self.modifications.append(mod)
        self._save_modifications()

        return new_prompt

    def _evolve_with_llm(self, prompt: str, feedback: str) -> str:
        """Evolve prompt using LLM."""
        client = Anthropic()

        system = """You are an expert at improving prompts for autonomous research agents.
Given the current prompt and feedback, propose an improved version.
Focus on clarity, specificity, and effectiveness."""

        user = f"""Current prompt:
{prompt}

Feedback:
{feedback}

Provide the improved prompt:"""

        response = client.messages.create(
            model=self.config.llm_model,
            max_tokens=2000,
            system=system,
            messages=[{"role": "user", "content": user}],
        )

        return response.content[0].text

    def _evolve_heuristic(self, prompt: str, feedback: str) -> str:
        """Evolve prompt using simple heuristics."""
        # Simple improvements
        improved = prompt

        # Add specificity if feedback mentions "vague"
        if "vague" in feedback.lower():
            improved += "\n\nBe specific and concrete in your analysis."

        # Add constraints if feedback mentions "unconstrained"
        if "unconstrained" in feedback.lower():
            improved += "\n\nConsider resource constraints and practical limitations."

        return improved

    def propose_hyperparameter_change(
        self,
        param: str,
        current_value: Any,
        direction: str = "increase",
    ) -> Modification:
        """Propose a hyperparameter change."""
        # Determine new value
        if isinstance(current_value, (int, float)):
            if direction == "increase":
                new_value = current_value * 1.1
            else:
                new_value = current_value * 0.9
        else:
            new_value = str(current_value)

        mod = Modification(
            id=f"mod_{len(self.modifications) + 1}",
            type=ModificationType.HYPERPARAMETER,
            description=f"Change {param} from {current_value} to {new_value}",
            old_value=str(current_value),
            new_value=str(new_value),
            expected_impact=0.02,
        )

        self.modifications.append(mod)
        self._save_modifications()
        return mod

    def apply_modification(self, mod_id: str) -> bool:
        """Mark a modification as applied."""
        for mod in self.modifications:
            if mod.id == mod_id:
                mod.status = "applied"
                self._save_modifications()
                return True
        return False

    def revert_modification(self, mod_id: str) -> bool:
        """Revert a modification."""
        for mod in self.modifications:
            if mod.id == mod_id:
                mod.status = "reverted"
                self._save_modifications()
                return True
        return False

    def record_impact(self, mod_id: str, impact: float):
        """Record actual impact of a modification."""
        for mod in self.modifications:
            if mod.id == mod_id:
                mod.actual_impact = impact
                self._save_modifications()
                break

    def get_successful_modifications(self) -> List[Modification]:
        """Get modifications with positive impact."""
        return [
            m for m in self.modifications if m.actual_impact and m.actual_impact > 0
        ]

    def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in successful modifications."""
        successful = self.get_successful_modifications()

        if not successful:
            return {"patterns": [], "insights": []}

        # Group by type
        by_type = {}
        for mod in successful:
            t = mod.type.value
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(mod.actual_impact)

        # Calculate average impact
        avg_impact = {}
        for t, impacts in by_type.items():
            avg_impact[t] = sum(impacts) / len(impacts)

        return {
            "total_successful": len(successful),
            "by_type": avg_impact,
            "insights": self._generate_insights(avg_impact),
        }

    def _generate_insights(self, avg_impact: Dict[str, float]) -> List[str]:
        """Generate insights from patterns."""
        insights = []

        best_type = max(avg_impact.items(), key=lambda x: x[1])
        insights.append(
            f"Most effective modification type: {best_type[0]} ({best_type[1]:.2%})"
        )

        return insights

    def run_iteration(
        self,
        feedback: str,
        performance: float,
    ) -> Dict[str, Any]:
        """
        Run one iteration of the meta-loop.

        Returns:
            Dictionary with results
        """
        self.iteration += 1

        # Check if improvement threshold met
        if self.iteration > 1:
            prev_best = self.get_successful_modifications()
            if prev_best:
                best_impact = max(m.actual_impact for m in prev_best)
                if best_impact < self.config.min_improvement:
                    return {
                        "status": "converged",
                        "iteration": self.iteration,
                        "message": "Improvement below threshold",
                    }

        # Evolve prompts
        if self.prompts:
            # Evolve the first prompt as example
            first_name = list(self.prompts.values())[0].name
            evolved = self.evolve_prompt(first_name, feedback, performance)
            return {
                "status": "evolved",
                "iteration": self.iteration,
                "prompt_version": evolved.version,
            }

        return {
            "status": "no_prompts",
            "iteration": self.iteration,
        }


def create_default_prompts() -> Dict[str, str]:
    """Create default prompt templates."""
    return {
        "hypothesis": """You are a research hypothesis generator.
Generate a hypothesis for improving LLM training.
Consider: learning rate, batch size, regularization, architecture.
Format: Clear hypothesis with expected outcome.""",
        "evaluation": """You are a research evaluation agent.
Analyze the experiment results and determine if the hypothesis was validated.
Consider: statistical significance, effect size, practical impact.""",
        "execution": """You are a research execution agent.
Implement the proposed modification and run the experiment.
Follow best practices for reproducibility.""",
    }


if __name__ == "__main__":
    print("Testing meta-loop...")

    # Create meta-loop
    meta = MetaLoop()

    # Register default prompts
    prompts = create_default_prompts()
    for name, content in prompts.items():
        meta.register_prompt(name, content)

    print(f"Registered {len(meta.prompts)} prompts")

    # Run an iteration
    result = meta.run_iteration(
        feedback="The hypothesis was too vague. Be more specific.",
        performance=0.95,
    )

    print(f"Iteration result: {result}")

    # Analyze patterns
    patterns = meta.analyze_patterns()
    print(f"Patterns: {patterns}")
