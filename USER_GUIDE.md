# Autonomous Research Stack — User Guide

> Complete guide to installing, configuring, and running the autonomous LLM training research system

**Version:** 0.7.3 | **Status:** Production-Ready

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [CLI Usage](#cli-usage)
5. [Python API](#python-api)
6. [Demo Training](#demo-training)
7. [Provider Setup](#provider-setup)
8. [Testing](#testing)
9. [Deployment](#deployment)
10. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 30-Second Quick Start

```bash
# Install
pip install autoresearch-stack

# Set API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Run the data pipeline
autoresearch --prepare-only

# Run experiments
autoresearch --experiments 10
```

### 2-Minute Demo (no GPU, no API key)

```bash
# From source
git clone https://github.com/iknowkungfubar/autoresearch-stack.git
cd autoresearch-stack
pip install -e .

# Run the numpy demo training — exercises curriculum, loss tracking,
# convergence detection without PyTorch or an API key
python train_any_llm.py --demo
```

---

## Installation

### Requirements

- Python 3.11+
- Linux/macOS (Windows via WSL)
- 4GB RAM minimum
- GPU optional (numpy demo works on CPU)

### From PyPI

```bash
pip install autoresearch-stack
```

### From Source

```bash
git clone https://github.com/iknowkungfubar/autoresearch-stack.git
cd autoresearch-stack
python -m venv venv
source venv/bin/activate
pip install -e .
```

### Optional Dependencies

```bash
pip install anthropic        # Anthropic Claude provider
pip install openai           # OpenAI / OpenRouter provider
pip install chromadb         # Vector memory store
pip install matplotlib       # Figures and visualization
```

### Docker

```bash
docker build -t autoresearch-stack .
docker run --rm -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY autoresearch-stack
```

---

## Configuration

### Config File

Edit `config.yaml` (auto-created with defaults if missing):

```yaml
experiment:
  budget: 500              # Max experiments
  time_per_experiment: 300 # Seconds per experiment
  val_target: 0.95         # Target val_bpb to achieve

model:
  size: "124M"
  learning_rate: 0.0001
  batch_size: 32

synthetic:
  n_samples: 200
  use_llm: false           # Requires API key
  model_provider: "anthropic"
```

### Environment Variable Overrides

Any config value can be overridden via environment variables:

```bash
export EXPERIMENT_BUDGET=1000          # Override max experiments
export LEARNING_RATE=0.0005            # Override model LR
export SYNTHETIC_USE_LLM=true          # Enable LLM data generation
export MEMORY_ENABLED=true             # Enable vector memory
```

### API Keys (set via env vars, never in config file)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."     # Anthropic Claude
export OPENAI_API_KEY="sk-..."            # OpenAI
export OPENROUTER_API_KEY="sk-or-..."     # OpenRouter
export MISTRAL_API_KEY="..."              # Mistral AI
export ZEN_API_KEY="..."                  # Zen AI
```

---

## CLI Usage

```bash
# Show help
autoresearch --help

# Prepare data only (no experiments)
autoresearch --prepare-only

# Run N experiments
autoresearch --experiments 10

# Run with custom config and input
autoresearch -c my_config.yaml -i input_data.txt

# Run full autonomous loop (up to budget)
autoresearch --experiments 100

# Python module syntax (equivalent)
python -m autoresearch --help
```

### Daemon Mode

```bash
python daemon.py start          # Start background daemon
python daemon.py status         # Check health
python daemon.py stop           # Stop gracefully
python daemon.py restart        # Restart
```

---

## Python API

### Autonomous Pipeline

```python
from autonomous_loop import AutonomousPipeline
from config import get_config

config = get_config("config.yaml")
pipeline = AutonomousPipeline("config.yaml")

# Prepare data
data = pipeline.prepare_data(
    raw_texts=["machine learning is fun", "neural networks are powerful"],
    use_synthetic=True,
    use_model_loop=False,
)

# Run a single experiment
result = pipeline.run_experiment(
    change_description="Increase LR by 10%",
    change_code="config.model.learning_rate *= 1.1",
    change_type="optimization",
    baseline_val_bpb=1.0,
)
print(f"New val_bpb: {result['val_bpb_after']:.4f}")
print(f"Improved: {result['improved']}")
```

### Hypothesis Generation

```python
from hypothesis import HypothesisGenerator

gen = HypothesisGenerator(use_llm=False)
hypotheses = gen.generate(n=3, change_type="optimization")
for h in hypotheses:
    print(f"- {h.description} ({h.expected_impact})")
    print(f"  Code: {h.code_diff}")
```

### Provider Selection

```python
from providers import LLMProviderFactory

# Create a provider
provider = LLMProviderFactory.create("anthropic")

# Or from config
provider = LLMProviderFactory.from_config({
    "provider": "openai",
    "api_key": "sk-...",
})

# Generate completion
response = provider.complete(
    messages=[{"role": "user", "content": "Hello"}],
    model="gpt-4o",
)
print(response.content)
```

### Sandbox Execution

```python
from sandbox import SafeRunner

runner = SafeRunner()

# Safe code runs
result = runner.run("print('hello world')")
print(f"Output: {result.stdout}")

# Dangerous code is blocked
result = runner.run("import os; os.system('rm -rf /')")
print(f"Blocked: {not result.success}")
```

---

## Demo Training

### Numpy Demo (no GPU, no API key)

```bash
python train_any_llm.py --demo
```

Output:
```
NUMPY DEMO TRAINING
  Step 0: loss=0.035699, stage=easy
  Step 25: loss=0.043933, stage=easy
  Step 50: loss=0.073658, stage=medium
  Step 75: loss=0.443240, stage=hard

Training complete:
  Steps: 100
  Final training loss: 0.021189
  Val BPB: 0.7124
  Converged: True
```

This exercises: curriculum scheduler, numpy demo model, training loop,
loss tracking, and convergence detection — all without PyTorch.

### With PyTorch

When PyTorch is installed, `train_any_llm.py` uses it automatically:

```python
from train_any_llm import Trainer
# The Trainer class wraps torch models when available,
# falls back to numpy demo when not
```

---

## Provider Setup

### Supported Cloud Providers

| Provider | Env Variable | Package |
|----------|-------------|---------|
| Anthropic Claude | `ANTHROPIC_API_KEY` | `pip install anthropic` |
| OpenAI | `OPENAI_API_KEY` | `pip install openai` |
| OpenRouter | `OPENROUTER_API_KEY` | `pip install openai` |
| Google Vertex AI | `GOOGLE_CLOUD_PROJECT` | `pip install google-cloud-aiplatform` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` | `pip install openai` |
| Mistral AI | `MISTRAL_API_KEY` | `pip install mistralai` |

### Supported Local Providers

| Provider | Default URL | Notes |
|----------|------------|-------|
| Ollama | `http://localhost:11434` | `ollama pull llama3` |
| vLLM | `http://localhost:8000` | OpenAI-compatible API |
| LM Studio | `http://localhost:1234/v1` | Enable CORS in LM Studio |
| llama.cpp | `http://localhost:8080` | Server mode |

Set `provider: "ollama"` and `base_url: "http://localhost:11434"` in config.yaml.

---

## Testing

```bash
# Run all tests (148 total)
pytest tests/ -q

# With coverage report
pytest tests/ -q --cov=./

# Specific test files
pytest tests/test_providers.py -v     # 32 provider tests
pytest tests/test_core.py -v          # Core module tests
pytest tests/test_autonomous_loop.py -v  # Pipeline tests

# Run with mypy type checking
mypy . --ignore-missing-imports
```

---

## Deployment

### Docker

```bash
docker build -t autoresearch-stack .
docker run --rm -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY autoresearch-stack
```

### Docker Compose (multi-node)

```bash
docker compose up -d
docker compose logs -f
docker compose down
```

### Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml
kubectl get pods -l app=autoresearch
```

---

## Project Structure

```
autoresearch-stack/
├── autonomous_loop.py     # Main pipeline orchestration
├── autoresearch/          # Python package (__init__, __main__)
├── config.py              # YAML + env var configuration
├── config.yaml            # Default configuration
├── providers.py           # 17+ LLM provider integrations
├── orchestrators.py       # 7 agent orchestrator integrations
├── synthetic_data.py      # LLM-powered data generation
├── curriculum.py          # Adaptive scheduling
├── memory.py              # Vector store with search
├── prioritization.py      # Bandit-based selection
├── hypothesis.py          # Hypothesis generation
├── feedback.py            # Experiment logging & classification
├── sandbox.py             # AST-validated safe execution
├── checkpoint.py          # State persistence
├── monitor.py             # Real-time status display
├── daemon.py              # Background execution
├── distribute.py          # Multi-node cluster management
├── metaloop.py            # Self-modification system
├── report.py              # Markdown report generation
├── figures.py             # Visualization (matplotlib)
├── stats.py               # Summary statistics
├── paper.py               # Research paper generation
├── peer_review.py         # Peer review simulation
├── data_intelligence.py   # Corpus cleaning
├── train_any_llm.py       # Training abstraction (numpy demo + PyTorch)
├── storage.py             # SQLite database
├── multi_agent.py         # Multi-agent architecture
├── tests/                 # 148 passing tests
├── k8s/                   # Kubernetes deployment
├── docs/                  # Development documentation
├── Dockerfile             # Docker build
├── docker-compose.yml     # Multi-node cluster
└── setup.py               # pip installable package
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'anthropic'`

```bash
pip install anthropic
```

### `ValueError: ANTHROPIC_API_KEY required`

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# Or set OPENAI_API_KEY / OPENROUTER_API_KEY
```

### `No known vulnerabilities found` (pip-audit check)

This is a good thing — it means your dependencies are clean.

### Tests fail on first run

```bash
pip install -e .
```

### Out of memory

Reduce batch size or experiment count in config.yaml:
```yaml
model:
  batch_size: 8  # Default: 32
experiment:
  budget: 50     # Default: 500
```
