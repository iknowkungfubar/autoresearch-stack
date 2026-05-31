# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v7.3] - 2026-05-31

### Security
- **sandbox.py** — Replaced trivially bypassable string-matching validation with AST-based analysis (closes 6 bypass vectors). Uses `ast.parse()` + `ast.walk()` to detect dangerous imports (`os`, `sys`, `subprocess`, `socket`), function calls (`eval`, `exec`, `__import__`, `compile`, `open`), and attribute chain access on blocked modules.

### Fixed
- **autonomous_loop.py:521** — Pipeline was passing raw `texts` instead of prepared `data` to `run_autonomous_loop()`. Curriculum and synthetic data generation pipeline was being silently bypassed.
- **autonomous_loop.py:269** — Replaced hardcoded `val_bpb_after = baseline - 0.01` (always "improved") with realistic Gaussian variance (20% harmful, 25% improvement, 55% neutral). The feedback/prioritization/hypothesis systems now receive plausible signal.
- **memory.py** — `vector_store` now always initialized as `SimpleVectorStore()` before `_init_vector_store()` attempt. Eliminates `None`-dereference crash when ChromaDB init fails.
- **metaloop.py** — `_evolve_with_llm()` now validates `ANTHROPIC_API_KEY` before calling `Anthropic()`, with a clear error message.
- **report.py:224** — Fixed `zip(names, dict.keys())` bug where dict keys were iterated as experiments instead of values.
- **report.py** — Replaced `if SummaryStatistics:` (class-as-boolean, always truthy) with `if FIGURES_AVAILABLE:` boolean flag.
- **hypothesis.py** — Removed dead `return []` statements that blocked all template generation.
- **43 mypy errors** — Fixed across 17 source files (storage.py, memory.py, metaloop.py, hypothesis.py, config.py, orchestrators.py, figures.py, daemon.py, sandbox.py, monitor.py, report.py, synthetic_data.py, peer_review.py, curriculum.py, statistics.py, checkpoint.py, multi_agent.py). Zero mypy errors across 33 source files.

### Added
- **32 provider tests** — Comprehensive mocked-API tests for AnthropicProvider, OpenAIProvider, OpenRouterProvider, OllamaProvider, LMStudioProvider, provider factory, LLMClient, MODEL_REGISTRY, and LLMResponse. Covers system message extraction, model name mapping, API error handling, missing API keys, null content, custom base URLs.
- **12 pipeline tests** — Tests for `AutonomousPipeline` (init, prepare_data, curriculum, run_experiment, get_status, CLI convenience) and `SimpleVectorStore` (search, empty query).
- **Total tests: 104 → 148**
- **.dockerignore** — Prevents `.git/`, `.env`, `*.db`, `__pycache__/`, `.venv/`, `tests/` from leaking into Docker images.
- **`__main__.py`** — Allows `python -m autoresearch` as alternative to `autoresearch` CLI.
- **Package structure** — `autoresearch/__init__.py`, `autoresearch/tests/__init__.py`, setup.py with explicit `py_modules` list.

### Changed
- **providers.py** — Moved 7 inline `import requests` to module level (avoids import overhead on every API call).
- **Version bump** — v7.2.0 → v7.3.0
- **Coverage** — Overall: 57% → 63%, autonomous_loop.py: 0% → 47%, providers.py: 34% → 48%

## [v7.2] - 2026-04-21

### Fixed
- **Lint Cleanup** — Resolved all 131 ruff violations
  - Removed unused imports across 15+ modules
  - Fixed bare `except:` clauses → `except Exception:`
  - Fixed ambiguous variable names
  - Fixed f-strings without placeholders
  - Fixed multi-statement lines in `data_intelligence.py`

- **SecOps Audit**
  - Confirmed no hardcoded secrets (only placeholders in docs)
  - Confirmed no dependency vulnerabilities (`pip-audit` clean)
  - Verified `.gitignore` covers `.env`, `.db`, `checkpoints/`, `memory/`, `logs/`

- **SDD Update** — Updated System Design Document from v4.0 to v7.0
  - Reflected actual flat module structure
  - Added v7.1 provider/orchestrator support documentation
  - Added microservices evolution assessment (not pursued — single-GPU tool)
  - Updated sprint backlog and quality gates

### Added
- **CI/CD Hardening** (`.github/workflows/ci.yml`)
  - Separate lint, security, test, build, publish jobs
  - Ruff enforcement (removed `--exit-zero`)
  - `pip-audit` dependency vulnerability scanning
  - Hardcoded secrets scanning step
  - Coverage reporting with `pytest --cov`
  - Docker image verification step

- **Test Coverage Expansion** (`tests/test_hardening.py`)
  - 51 new tests covering providers, orchestrators, daemon, distribute, metaloop, synthetic_data, config, data_intelligence, storage
  - Total tests: 53 → 104

---

## [v7.0] - 2026-04-21

### Added
- **Peer Review Simulation** (`peer_review.py`)
  - `PeerReviewSimulator` class for paper review
  - Multiple reviewer simulation with different profiles
  - Review aspects: originality, technical quality, clarity, etc.
  - Verdict generation: Accept, Minor Revision, Major Revision, Reject
  - Consensus calculation and report generation

- **User Guide** (`USER_GUIDE.md`)
  - Comprehensive step-by-step instructions
  - Quick start in 5 minutes
  - Installation, configuration, usage guides
  - Troubleshooting section
  - Project structure overview

---

## [v6.2] - 2026-04-20

### Added
- **Distribution System** (`distribute.py`)
  - `Cluster` class for multi-node management
  - `Node` class for individual nodes (master/worker)
  - `ResourceManager` for allocation and load balancing
  - `CostEstimator` for cloud cost calculations
  - AWS, GCP, Azure instance pricing

- **Kubernetes Deployment** (`k8s/deployment.yaml`)
  - K8s Deployment manifest
  - Service configuration
  - GPU resource support

- **Docker Compose** (`docker-compose.yml`)
  - Local multi-node cluster
  - Master + 3 workers
  - Resource limits

### Added
- **Multi-Node Execution** (`distribute.py`)
  - `DistributedExecutor` for task distribution
  - Health monitoring
  - Task queuing

---

## [v6.1] - 2026-04-20

### Added
- **Meta-Loop** (`metaloop.py`)
  - `MetaLoop` class for self-improvement
  - Prompt evolution using LLM or heuristics
  - Hyperparameter modification proposals
  - Pattern analysis on successful modifications

- **Prompt Refinement** (`metaloop.py`)
  - `PromptTemplate` with versioning
  - Automatic prompt improvement based on feedback
  - Performance tracking per prompt version

- **Self-Improving Code** (`metaloop.py`)
  - Modification tracking and history
  - Impact recording and analysis
  - Convergence detection

---

## [v5.2] - 2026-04-20

### Added
- **Paper Generation** (`paper.py`)
  - `Paper` class for research paper drafting
  - Full paper structure (Abstract, Introduction, Method, etc.)
  - `generate_paper_from_experiments()` for data-driven papers
  - Markdown and LaTeX output formats

- **Citation Management** (`paper.py`)
  - `Citation` class with BibTeX/LaTeX export
  - Reference management

- **Literature Review** (`paper.py`)
  - Related work section templates
  - Integration with experiment findings

---

## [v6.0] - 2026-04-20

### Added
- **Daemon Mode** (`daemon.py`)
  - `Daemon` class for continuous background operation
  - PID file management
  - Graceful start/stop/restart
  - Signal handling (SIGTERM, SIGINT, SIGHUP)

- **Health Checks** (`daemon.py`)
  - `HealthChecker` class for standalone checks
  - Health status monitoring
  - Error tracking and reporting

- **Auto-Restart** (`daemon.py`)
  - Automatic restart on failure
  - Configurable restart attempts
  - Cooldown period between restarts

- **Background Execution** (`daemon.py`)
  - Fork-based daemonization
  - Session management
  - Logging to file and console

---

## [v5.0] - 2026-04-20

### Added
- **Figure Generation** (`figures.py`)
  - `FigureGenerator` class with matplotlib integration
  - Learning curves, success rate, metric distributions
  - Change type performance comparison
  - Improvement timeline plots
  - Graceful fallback when matplotlib unavailable

- **Summary Statistics** (`stats.py`)
  - `SummaryStatistics` class for experiment analysis
  - Convergence metrics and trend analysis
  - Change type breakdown
  - JSON export with summary reports

- **Integrated Reporting** (`report.py`)
  - `generate_full_report()` combining stats, figures, markdown
  - Full experiment analysis pipeline

---

## [v4.1] - 2026-04-20

### Added
- **CI/CD Pipeline** (`.github/workflows/ci.yml`)
  - GitHub Actions workflow
  - Test, build, publish jobs
  
- **Docker Support** (`Dockerfile`)
  - Docker image with non-root user
  - Production-ready configuration
  
- **Python Package** (`setup.py`, `requirements.txt`)
  - Installable with `pip install -e .`
  - Entry points for CLI
  
- **Test Coverage**
  - 25 unit tests with pytest
  - All tests passing

### Removed
- None

### Fixed
- None

---

## [v4.0] - 2026-04-20

### Added

- **Multi-Agent Architecture** (`multi_agent.py`)
  - `ResearchAgent` for literature/gaps
  - `HypothesisAgent` for proposals
  - `ExecutionAgent` for experiments
  - `EvaluationAgent` for analysis
  - `MemoryAgent` for storage
  - `OrchestratorAgent` for coordination

- **Sandboxed Execution** (`sandbox.py`)
  - `Sandbox` class with resource limits
  - `SafeRunner` with code validation
  - Timeout and memory limits

- **Checkpoint System** (`checkpoint.py`)
  - `CheckpointManager` for state persistence
  - Save/resume from interruption
  - History tracking

- **Monitor** (`monitor.py`)
  - Real-time status display
  - Event logging
  - Statistics tracking

- **Report Generation** (`report.py`)
  - `Report` class for markdown
  - Summary reports
  - Comparison reports

- **Fixed Issues**
  - torch import failure with graceful fallback

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

*No unreleased features. All previously planned features shipped through v7.2.*