# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v3.1] - 2026-04-20

### Added
- **Memory System** (`memory.py`)
  - `MemorySystem` class with vector store
  - `SimpleVectorStore` fallback (ChromaDB optional)
  - `what_been_tried()` query function
  - Semantic search over experiments
  - Pattern analysis

- **Prioritization** (`prioritization.py`)
  - `BanditSelector` with UCB1, epsilon-greedy, Thompson sampling
  - `PrioritizationSystem` for experiment selection
  - Automatic learning from experiment history
  - Exploration vs exploitation balancing

- **Hypothesis Generation** (`hypothesis.py`)
  - `HypothesisGenerator` class
  - LLM-powered hypothesis proposals (with API key)
  - Rule-based fallback templates
  - Analysis-based hypothesis generation

- **Pipeline Updates** (`autonomous_loop.py`)
  - Integrated memory, prioritization, hypothesis
  - v3.1 banner in output
  - Full pipeline with intelligent proposals

---

## [v3.0] - 2026-04-20

### Added
- **Configuration System** (`config.py`, `config.yaml`)
  - Environment-based configuration with YAML defaults
  - Environment variable overrides for API keys, hyperparameters
  - Dataclass-based config with validation

- **Synthetic Data Generation** (`synthetic_data.py`)
  - LLM-powered generation via Anthropic/OpenAI API
  - Evol-Instruct difficulty scaling (easy/medium/hard prompts)
  - Quality filtering by length and heuristics
  - Multiple difficulty descriptor templates

- **Adaptive Curriculum** (`curriculum.py`)
  - Multiple scheduling strategies (linear, exponential, step, adaptive)
  - Performance-aware scheduling with loss tracking
  - Difficulty metrics: length, perplexity, complexity, hybrid
  - `AdaptiveScheduler` and `Scheduler` classes

- **Experiment Tracking** (`feedback.py`)
  - `Experiment` dataclass with full metadata
  - `ExperimentStatus` enum (kept/reverted/running/failed)
  - Failure classification (13 types)
  - JSONL experiment logging with reward computation

- **Storage System** (`storage.py`)
  - SQLite experiment database
  - `ExperimentDB` class with full CRUD operations
  - Checkpoint saving/loading
  - `ExperimentJSONL` fallback

- **Pipeline** (`autonomous_loop.py`)
  - `AutonomousPipeline` class
  - Full data preparation pipeline
  - Experiment execution loop
  - Status reporting

### Changed
- Updated README with v3.0 status
- Updated improvements.md roadmap
- Expanded agent.md with comprehensive RALPH documentation
- Expanded prompt.md with detailed loop instructions

---

## [v2.0] - 2026-01

### Added
- Curriculum learning with easy/medium/hard staging
- Feedback loop with val_bpb reward
- Initial autonomous_loop.py pipeline

### Changed
- Initial v2 release structure

---

## [v1.0] - 2025

### Added
- Initial release with basic components

---

## Unreleased

### Planned (v3.1)
- Memory system with ChromaDB vector store
- Semantic search over experiments
- Hypothesis generation

### Planned (v3.2)
- Multi-agent architecture
- Specialized agents: Research, Hypothesis, Execution, Evaluation

### Planned (v4.0)
- Production hardening
- Sandboxed execution
- Checkpoint system

### Planned (v5.0)
- Reporting & visualization
- Paper generation (AI Scientist-style)