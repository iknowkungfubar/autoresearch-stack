# Autonomous Research Stack — Agent Context

## Overview

**Autonomous Research Stack** is a Python system for building and shipping autonomous LLM training research systems. Designed for single-GPU research labs, it provides automated experimentation pipelines, hyperparameter sweeps, and continuous improvement loops.

## Tech Stack

- **Language:** Python 3.11+
- **Build System:** setuptools / pyproject.toml
- **Package:** `autoresearch-stack` (published on PyPI v0.7.3)
- **Testing:** pytest
- **Linting:** ruff
- **Containerization:** Docker, docker-compose
- **Orchestration:** Kubernetes (k8s/) configs available

## Repository Structure

```
src/autoresearch/
├── reporting/              # Experiment reporting & stats
├── research/               # Research pipeline logic
├── training/               # Training loop orchestration
├── config/                 # Configuration parsing
└── cli/                    # CLI interface
tests/                      # Test suite
docs/                       # Documentation
```

## Key Commands

- `pip install autoresearch-stack` — Install from PyPI
- `pip install -e .` — Install from source (editable)
- `pytest tests/ -v` — Run tests
- `ruff check src/ tests/` — Lint check
- `python -m autoresearch --help` — CLI help

## Architecture

- **Research Pipeline**: Automated experiment lifecycle — hypothesis → execution → evaluation → logging
- **Configuration**: YAML-based config system (`config.yaml`)
- **Reporting**: Stats collection and experiment dashboards
- **Containerized**: Docker + K8s support for reproducible runs
- **API-Driven**: Environment-based API key config (Anthropic/OpenAI)

## Quality Gates

- `pytest tests/ -v --cov=src/`
- `ruff check src/ tests/`
- `mypy src/`
