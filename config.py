"""
Configuration management for Autonomous Research Stack.

Loads configuration from:
1. config.yaml (defaults)
2. Environment variables (overrides)
3. .env file (optional)
"""

import os
import json
import yaml
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    """Experiment settings."""

    budget: int = 500  # max experiments
    time_per_experiment: int = 300  # seconds (5 min)
    val_target: float = 0.95  # target val_bpb
    max_retries: int = 3


@dataclass
class ModelConfig:
    """Model settings."""

    size: str = "124M"
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 768
    sequence_len: int = 2048
    learning_rate: float = 1e-4
    batch_size: int = 32
    weight_decay: float = 0.01
    warmup_steps: int = 100


@dataclass
class SyntheticConfig:
    """Synthetic data generation settings."""

    n_samples: int = 200
    n_model_samples: int = 10
    use_llm: bool = False
    model_provider: str = "anthropic"  # anthropic or openai
    model_name: str = "claude-sonnet-4-20250514"
    temperature: float = 0.9
    quality_threshold: float = 0.8
    difficulty_scaling: bool = True


@dataclass
class CurriculumConfig:
    """Curriculum learning settings."""

    enabled: bool = True
    stages: int = 3  # easy, medium, hard
    difficulty_metric: str = "length"  # length or perplexity
    adaptive: bool = False
    warmup_ratio: float = 0.1


@dataclass
class MemoryConfig:
    """Memory system settings."""

    enabled: bool = False
    vector_store: str = "chroma"  # chroma or simple
    persist_dir: str = "./memory"
    semantic_search: bool = True


@dataclass
class APIConfig:
    """API configuration."""

    anthropic_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )
    openai_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    api_base: Optional[str] = None
    timeout: int = 60


@dataclass
class LoggingConfig:
    """Logging settings."""

    level: str = "INFO"
    format: str = "json"  # json or text
    log_dir: str = "./logs"
    experiment_log: str = "./experiments.jsonl"


@dataclass
class Config:
    """Main configuration container."""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def load(cls, path: str = "config.yaml") -> "Config":
        """Load configuration from YAML file with environment overrides."""
        config = cls()

        # Load from YAML if exists
        config_path = Path(path)
        if config_path.exists():
            with open(config_path, "r") as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    cls._apply_yaml(config, yaml_config)

        # Apply environment variable overrides
        cls._apply_env(config)

        return config

    @classmethod
    def _apply_yaml(cls, config: "Config", yaml_dict: dict):
        """Apply YAML configuration to config object."""
        for section, values in yaml_dict.items():
            if hasattr(config, section):
                section_obj = getattr(config, section)
                if isinstance(values, dict):
                    for key, value in values.items():
                        if hasattr(section_obj, key):
                            setattr(section_obj, key, value)

    @classmethod
    def _apply_env(cls, config: "Config"):
        """Apply environment variable overrides."""
        # Experiment settings
        if e := os.getenv("EXPERIMENT_BUDGET"):
            config.experiment.budget = int(e)
        if e := os.getenv("TIME_PER_EXPERIMENT"):
            config.experiment.time_per_experiment = int(e)
        if e := os.getenv("VAL_TARGET"):
            config.experiment.val_target = float(e)

        # Model settings
        if e := os.getenv("LEARNING_RATE"):
            config.model.learning_rate = float(e)
        if e := os.getenv("BATCH_SIZE"):
            config.model.batch_size = int(e)

        # Synthetic settings
        if e := os.getenv("SYNTHETIC_USE_LLM"):
            config.synthetic.use_llm = e.lower() in ("true", "1", "yes")
        if e := os.getenv("SYNTHETIC_PROVIDER"):
            config.synthetic.model_provider = e
        if e := os.getenv("SYNTHETIC_MODEL"):
            config.synthetic.model_name = e

        # Memory settings
        if e := os.getenv("MEMORY_ENABLED"):
            config.memory.enabled = e.lower() in ("true", "1", "yes")

    def to_dict(self) -> dict:
        """Convert config to dictionary (hides API keys)."""
        d = {}
        for section in [
            "experiment",
            "model",
            "synthetic",
            "curriculum",
            "memory",
            "logging",
        ]:
            section_obj = getattr(self, section)
            d[section] = {}
            for key in dir(section_obj):
                if not key.startswith("_"):
                    value = getattr(section_obj, key)
                    # Mask API keys
                    if "key" in key.lower() and value:
                        value = "***" + value[-4:] if len(value) > 4 else "****"
                    d[section][key] = value
        return d

    def save(self, path: str = "config.yaml"):
        """Save current config to YAML (with masked API keys)."""
        d = self.to_dict()
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False)


# Global config instance
_config: Optional[Config] = None


def get_config(path: str = "config.yaml") -> Config:
    """Get or create global config instance."""
    global _config
    if _config is None:
        _config = Config.load(path)
    return _config


def reset_config():
    """Reset global config (useful for testing)."""
    global _config
    _config = None


if __name__ == "__main__":
    # Demo: print current config
    c = get_config()
    print(json.dumps(c.to_dict(), indent=2))
