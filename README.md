# Autonomous Research Stack

> Build and ship autonomous LLM training research systems

**Version:** v0.7.3 | **License:** MIT | **Python:** 3.11+

[![Tests](https://github.com/iknowkungfubar/autoresearch-stack/actions/workflows/ci.yml/badge.svg)](https://github.com/iknowkungfubar/autoresearch-stack/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/autoresearch-stack)](https://pypi.org/project/autoresearch-stack/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An autonomous research stack for continuously improving LLM training through automated experimentation. Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch), designed for single-GPU research labs.

---

## Quick Start

```bash
# Install
pip install autoresearch-stack

# Or from source
git clone https://github.com/iknowkungfubar/autoresearch-stack.git
cd autoresearch-stack
pip install -e .

# Configure (at least one API key)
export ANTHROPIC_API_KEY=sk-ant-...
# or: export OPENAI_API_KEY=sk-...

# Run the data pipeline
autoresearch --prepare-only

# Run 10 autonomous experiments
autoresearch --experiments 10

# Run with custom config
autoresearch -c my_config.yaml -i training_data.txt --experiments 100

# Portfolio alias (both commands work identically)
turintech-autoresearch --experiments 10

# Legacy alias (deprecated, kept for backwards compat)
turintech-research --experiments 10

# Python module syntax also works
python -m autoresearch --help
```

## Demo: Numpy Training (no GPU required)

```bash
# Test the training pipeline without PyTorch
python train_any_llm.py --demo
```

## Examples

The [examples/](examples/) directory contains ready-to-run demos:

- **`examples/quick_experiment.py`** — Minimal 2-epoch hyperparameter sweep with grid search

```bash
python examples/quick_experiment.py
```

This runs a complete training loop using the numpy demo model, exercising the curriculum scheduler, loss tracking, and convergence detection — no GPU or PyTorch needed.

## Features

### Data Pipeline
| Module | What it does |
|--------|-------------|
| `data_intelligence.py` | Corpus cleaning, noise detection, text repair |
| `synthetic_data.py` | LLM-powered generation with Evol-Instruct |
| `curriculum.py` | Adaptive scheduling (linear, exponential, step, adaptive) |
| `storage.py` | SQLite experiment database with JSONL fallback |

### Experiment Engine
| Module | What it does |
|--------|-------------|
| `memory.py` | Vector store with semantic search (ChromaDB optional) |
| `prioritization.py` | Bandit-based selection (UCB1, epsilon-greedy, Thompson) |
| `hypothesis.py` | LLM-driven hypothesis generation with rule-based fallback |
| `feedback.py` | Reward computation, failure classification (13 types) |
| `multi_agent.py` | Multi-agent architecture (research, hypothesis, execution, evaluation) |

### Infrastructure
| Module | What it does |
|--------|-------------|
| `sandbox.py` | Safe code execution with AST-based validation |
| `checkpoint.py` | State persistence and resume |
| `monitor.py` | Real-time status and progress bars |
| `daemon.py` | Background execution with health checks and auto-restart |
| `distribute.py` | Multi-node cluster management (Docker/K8s) |

### LLM Integration
| Module | What it does |
|--------|-------------|
| `providers.py` | 17+ LLM providers (Anthropic, OpenAI, OpenRouter, Ollama, vLLM, etc.) |
| `orchestrators.py` | 7 agent orchestrators (CrewAI, AutoGen, LangChain, etc.) |
| `train_any_llm.py` | Training abstraction (numpy demo + optional PyTorch) |

### Reporting & Analysis
| Module | What it does |
|--------|-------------|
| `report.py` | Markdown experiment reports with comparison |
| `figures.py` | Matplotlib visualizations with graceful fallback |
| `stats.py` | Summary statistics and convergence analysis |
| `paper.py` | Research paper generation (Markdown/LaTeX) |
| `peer_review.py` | Peer review simulation (5 reviewer profiles) |

---

## Configuration

All configuration lives in `config.yaml`. Environment variables override YAML values:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # API key (never put in config file!)
export EXPERIMENT_BUDGET=1000          # Override max experiments
export LEARNING_RATE=0.0005            # Override model LR
export SYNTHETIC_USE_LLM=true          # Enable LLM data generation
export MEMORY_ENABLED=true             # Enable vector memory
```

## Provider Support

**Cloud:** Anthropic (Claude), OpenAI (GPT-4/4o), OpenRouter, Google Vertex AI, Azure OpenAI, Mistral AI, Cohere, Zen AI

**Local:** Ollama, vLLM, LM Studio, llama.cpp, LiteLLM, KoboldCPP, LocalAI, Text Generation WebUI

**Orchestrators:** OpenCode, OpenCrew, AgentForge, CrewAI, AutoGen, LangChain, LlamaIndex

## The Metric

**val_bpb** (validation bits per byte) — Lower is better. The single optimization target.

## Project Constraints (Never Changed)

1. **val_bpb** is the ONLY metric
2. ONE change per experiment
3. Revert on regression
4. Single-GPU focused

## Development Status

|| Version | Status | Tests | Coverage | Type Safety |
||---------|--------|-------|----------|-------------|
|| v0.7.3 | **Current** | 463 ✅ | 69% | 0 mypy errors |
| v0.7.2 | Shipped | 104 ✅ | 57% | 43 errors |
| v0.7.0 | Shipped | 53 ✅ | — | — |

## Testing

```bash
# Run all tests
pytest tests/ -q

# With coverage
pytest tests/ -q --cov=./

# Run specific test file
pytest tests/test_providers.py -v
```

## Docker

```bash
docker build -t autoresearch-stack .
docker run --rm -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY autoresearch-stack

# Multi-node cluster
docker compose up
```

## References

- [Karpathy autoresearch](https://github.com/karpathy/autoresearch) — val_bpb metric
- [Ouroboros](https://github.com/razzant/ouroboros) — Self-modifying systems
- [AI Scientist](https://github.com/SakanaAI/AI-Scientist) — Paper generation

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on our development process, coding standards, PR workflow, and code of conduct.

## License

MIT — see [LICENSE](LICENSE) for details.
