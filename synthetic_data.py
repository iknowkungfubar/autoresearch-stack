"""
Synthetic data generation for LLM training.

Features:
- Template-based generation (fallback)
- LLM-powered generation (with API key)
- Evol-Instruct difficulty scaling
- Quality filtering (perplexity-based)
"""

import os
import random
import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

# Default topics for template generation
DEFAULT_TOPICS = [
    "machine learning",
    "deep learning",
    "neural networks",
    "transformers",
    "attention mechanisms",
    "optimization",
    "python programming",
    "systems programming",
    "data structures",
    "algorithms",
    "distributed systems",
    "computer graphics",
]

# Difficulty descriptors for Evol-Instruct
EASY_PROMPTS = [
    "Explain what is {topic} in simple terms.",
    "What is {topic}? Give a basic definition.",
    "List the key concepts of {topic}.",
    "Describe {topic} for a beginner.",
    "What are the main benefits of {topic}?",
]

MEDIUM_PROMPTS = [
    "Explain {topic} and its main components.",
    "How does {topic} work under the hood?",
    "Compare and contrast different approaches to {topic}.",
    "What are the trade-offs in {topic}?",
    "Describe a practical application of {topic}.",
]

HARD_PROMPTS = [
    "Explain the advanced optimizations in {topic} for production systems.",
    "What are the latest research advances in {topic}?",
    "How would you implement {topic} from scratch?",
    "Discuss the limitations and open problems in {topic}.",
    "Design a system that applies {topic} at scale.",
]

DIFFICULTY_TOPICS = {
    "easy": EASY_PROMPTS,
    "medium": MEDIUM_PROMPTS,
    "hard": HARD_PROMPTS,
}


@dataclass
class GenerationResult:
    """Result of synthetic data generation."""

    prompts: List[str]
    metadata: Dict[str, Any]
    used_llm: bool


class SyntheticGenerator:
    """Synthetic data generator with multiple backends."""

    def __init__(
        self,
        use_llm: bool = False,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.9,
        api_key: Optional[str] = None,
    ):
        self.use_llm = use_llm
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.api_key = (
            api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
        )

        # Track generation stats
        self.stats = {
            "total_generated": 0,
            "llm_calls": 0,
            "template_calls": 0,
            "filtered": 0,
        }

    def generate(
        self,
        n: int = 200,
        topics: Optional[List[str]] = None,
        difficulty: str = "mixed",
        quality_threshold: float = 0.0,
    ) -> GenerationResult:
        """Generate synthetic prompts."""
        topics = topics or DEFAULT_TOPICS

        if self.use_llm and self.api_key:
            return self._generate_with_llm(n, topics, difficulty, quality_threshold)
        else:
            return self._generate_with_template(n, topics, difficulty)

    def _generate_with_template(
        self,
        n: int,
        topics: List[str],
        difficulty: str,
    ) -> GenerationResult:
        """Generate using templates (no API needed)."""
        prompts = []

        difficulty_list = self._get_difficulty_list(difficulty)

        for _ in range(n):
            topic = random.choice(topics)
            template = random.choice(difficulty_list)
            prompt = template.format(topic=topic)
            prompts.append(prompt)

        self.stats["template_calls"] += 1
        self.stats["total_generated"] += n

        return GenerationResult(
            prompts=prompts,
            metadata={
                "method": "template",
                "difficulty": difficulty,
                "topics_used": list(set(topics)),
            },
            used_llm=False,
        )

    def _generate_with_llm(
        self,
        n: int,
        topics: List[str],
        difficulty: str,
        quality_threshold: float,
    ) -> GenerationResult:
        """Generate using LLM API."""
        prompts = []

        # Try each topic and difficulty
        batch_size = min(10, n)
        for _ in range(0, n, batch_size):
            remaining = min(batch_size, n - len(prompts))

            # Generate batch
            batch = self._generate_batch(remaining, topics, difficulty)
            prompts.extend(batch)

        self.stats["llm_calls"] += 1
        self.stats["total_generated"] += len(prompts)

        return GenerationResult(
            prompts=prompts,
            metadata={
                "method": "llm",
                "provider": self.provider,
                "model": self.model,
                "difficulty": difficulty,
            },
            used_llm=True,
        )

    def _generate_batch(
        self,
        n: int,
        topics: List[str],
        difficulty: str,
    ) -> List[str]:
        """Generate a batch of prompts using LLM."""
        difficulty_prompt = self._get_difficulty_instruction(difficulty)

        system_prompt = f"""You are a synthetic data generator for LLM training.
Generate {n} diverse training prompts about machine learning and systems topics.
{difficulty_prompt}
Output ONLY a JSON array of strings, nothing else."""

        user_prompt = f"""Generate {n} prompts about: {", ".join(topics)}.
Vary the difficulty from easy to hard.
Make each prompt specific and actionable."""

        try:
            if self.provider == "anthropic":
                return self._call_anthropic(system_prompt, user_prompt, n)
            elif self.provider == "openai":
                return self._call_openai(system_prompt, user_prompt, n)
            else:
                # Fallback
                return self._generate_with_template(n, topics, difficulty).prompts
        except Exception as e:
            print(f"LLM generation failed: {e}, falling back to template")
            return self._generate_with_template(n, topics, difficulty).prompts

    def _call_anthropic(self, system: str, user: str, n: int) -> List[str]:
        """Call Anthropic API."""
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)

            response = client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=self.temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )

            content = response.content[0].text
            return self._parse_json_array(content)
        except ImportError:
            raise RuntimeError("anthropic package not installed")

    def _call_openai(self, system: str, user: str, n: int) -> List[str]:
        """Call OpenAI API."""
        try:
            import openai

            client = openai.OpenAI(api_key=self.api_key)

            response = client.chat.completions.create(
                model=self.model,
                max_tokens=4096,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )

            content = response.choices[0].message.content
            return self._parse_json_array(content)
        except ImportError:
            raise RuntimeError("openai package not installed")

    def _parse_json_array(self, text: str) -> List[str]:
        """Parse JSON array from LLM output."""
        import re

        # Try to find JSON array in text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                arr = json.loads(match.group())
                if isinstance(arr, list):
                    return [str(x) for x in arr]
            except json.JSONDecodeError:
                pass

        # Fallback: split by newlines
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        return lines[:10]  # Return up to 10

    def _get_difficulty_list(self, difficulty: str) -> List[str]:
        """Get prompts for difficulty level."""
        if difficulty == "mixed":
            return EASY_PROMPTS + MEDIUM_PROMPTS + HARD_PROMPTS
        elif difficulty == "easy":
            return EASY_PROMPTS
        elif difficulty == "medium":
            return MEDIUM_PROMPTS
        elif difficulty == "hard":
            return HARD_PROMPTS
        return EASY_PROMPTS + MEDIUM_PROMPTS + HARD_PROMPTS

    def _get_difficulty_instruction(self, difficulty: str) -> str:
        """Get difficulty instruction for LLM."""
        if difficulty == "easy":
            return "Make prompts simple and beginner-friendly."
        elif difficulty == "medium":
            return "Make prompts at an intermediate level."
        elif difficulty == "hard":
            return "Make prompts advanced and technical."
        return "Vary difficulty across the spectrum."

    def evol_instruct_scale(
        self,
        prompts: List[str],
        iterations: int = 2,
    ) -> List[str]:
        """Apply Evol-Instruct to increase difficulty."""
        if not self.use_llm or not self.api_key:
            return prompts

        evolved = prompts.copy()

        for _ in range(iterations):
            # Evolve each prompt
            evolved = self._evolve_prompts(evolved)

        return evolved

    def _evolve_prompts(self, prompts: List[str]) -> List[str]:
        """Evolve prompts to be more complex."""
        system = """You are an expert at evolving training prompts.
Make each prompt more specific, complex, and challenging.
Keep the core topic but add constraints, edge cases, or deeper requirements."""

        user = f"""Evolve these {len(prompts)} prompts to be harder:
{json.dumps(prompts[:5])}

Output ONLY a JSON array of evolved prompts."""

        try:
            if self.provider == "anthropic":
                return self._call_anthropic(system, user, len(prompts))
            elif self.provider == "openai":
                return self._call_openai(system, user, len(prompts))
        except:
            pass

        return prompts

    def quality_filter(
        self,
        prompts: List[str],
        min_length: int = 20,
        max_length: int = 500,
    ) -> List[str]:
        """Filter prompts by quality heuristics."""
        filtered = []

        for p in prompts:
            # Length check
            if len(p) < min_length or len(p) > max_length:
                self.stats["filtered"] += 1
                continue

            # Check for basic quality
            if not p or not p.strip():
                self.stats["filtered"] += 1
                continue

            # Must have some alpha characters
            if sum(c.isalpha() for c in p) < 10:
                self.stats["filtered"] += 1
                continue

            filtered.append(p)

        return filtered


def generate_synthetic(
    n: int = 200,
    topics: Optional[List[str]] = None,
    difficulty: str = "mixed",
    use_llm: bool = False,
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-20250514",
) -> List[str]:
    """Convenience function for backward compatibility."""
    gen = SyntheticGenerator(
        use_llm=use_llm,
        provider=provider,
        model=model,
    )
    result = gen.generate(n=n, topics=topics, difficulty=difficulty)
    return result.prompts


def model_in_the_loop_generate(
    model: Any = None,
    tokenizer: Any = None,
    prompts: List[str] = None,
    n_samples: int = 10,
) -> List[str]:
    """Generate data with model in the loop.

    This is a stub for now - actual implementation would:
    1. Take prompts
    2. Generate completions with the model
    3. Filter and quality-check
    4. Return as training data
    """
    if prompts is None:
        prompts = ["Continue: The key to machine learning is"] * n_samples

    # Stub implementation - returns prompts as-is with stub output
    # Real implementation would use model.generate()
    outputs = []
    for p in prompts:
        # Simulate model generation
        outputs.append(p + " [model output placeholder]")

    return outputs


if __name__ == "__main__":
    # Demo
    gen = SyntheticGenerator(use_llm=False)
    result = gen.generate(n=10)
    print(f"Generated {len(result.prompts)} prompts (LLM: {result.used_llm})")
    for p in result.prompts[:3]:
        print(f"  - {p}")
