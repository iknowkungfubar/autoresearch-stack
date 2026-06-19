# AGENTS.md — Autonomous Research Stack

> Operating context for AI agents working on this repository. Load at session start.

## Project Identity

**Autonomous Research Stack** is a framework for building and shipping autonomous LLM training research systems. It provides abstractions for experiment orchestration, dataset management, and model training pipelines.

## Tech Stack

- **Language:** Python 3.11+
- **Build:** setuptools (pyproject.toml)
- **Linting:** ruff
- **Testing:** pytest
- **Type Checking:** mypy
- **Published:** PyPI as `autoresearch-stack`

## Repository Structure

```
├── src/autoresearch_stack/    # Main package
│   ├── core/                  # Orchestration, pipeline management
│   ├── experiments/           # Experiment definitions and runners
│   ├── datasets/              # Dataset loading, preprocessing, caching
│   ├── models/                # Model wrappers and training loops
│   ├── monitoring/            # Metrics logging, tracking, visualization
│   └── config.py              # Configuration management
├── tests/                     # Test suite
├── docs/                      # Documentation
├── AGENTS.md                  # This file
├── CHANGELOG.md               # Release history
├── CONTRIBUTING.md            # Contributor guide
├── LICENSE                    # MIT
└── README.md                  # Project documentation
```

## Conventions

- **Commits:** `feat:|fix:|refactor:|test:|docs:|chore: [scope] — [message]`
- **Type annotations** on all public functions
- **Docstrings** for all public modules, classes, and functions
- **Reproducibility** — all experiments must set explicit seeds and log hyperparameters

## Quality Gates

- `ruff check src/` — 0 errors
- `mypy src/` — strict mode
- `pytest tests/` — all tests pass

## Agent Workflow

1. **Read the docs** — Start with `docs/architecture.md` for the system design
2. **Understand the experiment pipeline** — Core orchestration lives in `src/core/`
3. **Tests first** — New experiment types need test coverage
4. **Reproducibility check** — Every experiment path must log its full configuration
5. **Verify** — Run full test suite before claiming done
