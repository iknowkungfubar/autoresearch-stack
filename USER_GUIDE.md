# Autonomous Research Stack - User Guide

> Complete guide to using the autonomous LLM training research system

**Version:** 7.2 | **Status:** Production-Ready

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running Experiments](#running-experiments)
5. [Components Overview](#components-overview)
6. [Advanced Features](#advanced-features)
7. [Deployment](#deployment)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 5-Minute Quick Start

```bash
# 1. Clone and install
git clone https://github.com/iknowkungfubar/autoresearch-stack.git
cd autoresearch-stack
pip install -e .

# 2. Set API key
export ANTHROPIC_API_KEY="sk-ant-..."

# 3. Run autonomous experiments
python autonomous_loop.py --experiments 10

# 4. View results
python -m pytest tests/  # Run tests
cat experiment_report.md  # View report
```

That's it! Your first autonomous experiment loop is running.

---

## Installation

### Requirements

- Python 3.11+
- Linux/macOS (Windows via WSL)
- 8GB RAM minimum (16GB recommended)
- GPU recommended for training

### Standard Installation

```bash
# Clone repository
git clone https://github.com/iknowkungfubar/autoresearch-stack.git
cd autoresearch-stack

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .
```

### Optional Dependencies

```bash
# For vector storage (optional)
pip install chromadb

# For visualization (optional)
pip install matplotlib seaborn

# For cloud deployment (optional)
pip install boto3 google-cloud-aiplatform azure-ml
```

### Docker Installation

```bash
# Build Docker image
docker build -t autoresearch .

# Run container
docker run --rm -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY autoresearch
```

---

## Configuration

### Environment Variables

Create a `.env` file or export:

```bash
# Required for LLM features
export ANTHROPIC_API_KEY="sk-ant-..."

# Optional: Alternative providers
export OPENAI_API_KEY="sk-..."
export SERPER_API_KEY="..."

# Configuration
export AUTORESEARCH_DB_PATH="./data/experiments.db"
export AUTORESEARCH_LOG_LEVEL="INFO"
```

### Configuration File

Edit `config.yaml`:

```yaml
# Training configuration
training:
  base_model: "gpt2"
  max_epochs: 10
  batch_size: 32
  learning_rate: 0.0001

# Experiment configuration
experiment:
  max_experiments: 100
  patience: 5
  target_metric: "val_bpb"

# Agent configuration
agents:
  use_llm: true
  model: "claude-3-sonnet-20240229"
  temperature: 0.7
```

### Programmatic Configuration

```python
from config import Config

config = Config(
    training=dict(
        max_epochs=20,
        batch_size=64,
    ),
    experiment=dict(
        max_experiments=50,
    )
)

# Use in pipeline
pipeline = AutonomousPipeline(config=config)
```

---

## Running Experiments

### Command Line Interface

```bash
# Run with defaults
python autonomous_loop.py

# Run specific number of experiments
python autonomous_loop.py --experiments 100

# Run with custom config
python autonomous_loop.py --config config.yaml

# Resume from checkpoint
python autonomous_loop.py --resume

# Daemon mode (continuous operation)
python daemon.py start
```

### Python API

```python
from autonomous_loop import AutonomousPipeline
from config import Config

# Create pipeline
config = Config()
pipeline = AutonomousPipeline(config=config)

# Run experiments
results = pipeline.run(num_experiments=50)

# Access results
print(f"Best val_bpb: {results['best_val_bpb']}")
print(f"Total experiments: {results['total']}")
```

### Using Multi-Agent System

```python
from multi_agent import OrchestratorAgent
from config import Config

# Initialize orchestrator
config = Config()
orchestrator = OrchestratorAgent(config)

# Run research cycle
result = orchestrator.run_cycle()

# Get hypothesis
hypothesis = result["hypothesis"]
print(f"Proposed: {hypothesis.description}")
```

---

## Components Overview

### Core Modules

| Module | Purpose | Quick Example |
|--------|---------|--------------|
| `config.py` | Configuration management | `Config().training.max_epochs` |
| `storage.py` | SQLite database | `ExperimentDB().save_experiment(exp)` |
| `autonomous_loop.py` | Main pipeline | `AutonomousPipeline().run()` |

### Intelligence Modules

| Module | Purpose |
|--------|---------|
| `memory.py` | Vector store for experiment history |
| `prioritization.py` | Bandit-based experiment selection |
| `hypothesis.py` | LLM-powered hypothesis generation |

### Production Modules

| Module | Purpose |
|--------|---------|
| `sandbox.py` | Safe code execution |
| `checkpoint.py` | State persistence |
| `monitor.py` | Real-time display |
| `daemon.py` | Continuous operation |

### Reporting Modules

| Module | Purpose |
|--------|---------|
| `report.py` | Markdown reports |
| `figures.py` | Visualizations |
| `stats.py` | Statistics |
| `paper.py` | Paper generation |
| `peer_review.py` | Review simulation |

---

## Advanced Features

### Using Memory System

```python
from memory import MemorySystem

# Initialize
memory = MemorySystem()

# Store experiment
memory.store({
    "change": "increased learning rate",
    "result": "improved val_bpb",
    "type": "optimization",
})

# Search similar experiments
similar = memory.query("learning rate adjustments", k=5)
print(similar)
```

### Bandit-Based Selection

```python
from prioritization import BanditSelector

selector = BanditSelector(method="ucb1")

# Select next experiment type
choice = selector.select()
selector.update(choice, reward=0.1)
```

### Generating Reports

```python
from report import generate_full_report

# Generate full report with figures
files = generate_full_report(
    experiments=experiments,
    output_dir="output",
    baseline=1.0,
)

print(f"Report: {files['report']}")
print(f"Figures: {files.get('figures')}")
```

### Generating Papers

```python
from paper import generate_paper_from_experiments

paper = generate_paper_from_experiments(experiments)
paper.save("research_paper.md")
```

### Peer Review Simulation

```python
from peer_review import PeerReviewSimulator

simulator = PeerReviewSimulator(num_reviewers=3)
reviews = simulator.simulate_review_round(
    paper_title="My Paper",
    paper_content=content,
)

# Get consensus
consensus = simulator.get_consensus(reviews)
print(f"Verdict: {consensus['agreement']}")
```

### Running as Daemon

```bash
# Start daemon (background)
python daemon.py start

# Check status
python daemon.py status

# Stop daemon
python daemon.py stop

# Restart
python daemon.py restart
```

---

## Deployment

### Local Cluster (Docker Compose)

```bash
# Start cluster
docker-compose up -d

# Scale workers
docker-compose up -d --scale worker=3

# View logs
docker-compose logs -f
```

### Kubernetes

```bash
# Deploy to K8s
kubectl apply -f k8s/deployment.yaml

# Check status
kubectl get pods -l app=autoresearch
```

### Cloud Deployment

```python
from distribute import CostEstimator, Cluster

# Estimate costs
cost = CostEstimator.estimate_experiment_run(
    provider=CloudProvider.AWS,
    instance_type="p3.2xlarge",
    num_experiments=100,
    minutes_per_experiment=30,
)
print(f"Estimated cost: ${cost.total_cost:.2f}")
```

---

## Testing

### Run All Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=. --cov-report=html

# Specific test file
pytest tests/test_core.py -v
```

### Run Specific Tests

```bash
# Test config
pytest tests/test_core.py::TestConfig -v

# Test agents
pytest tests/test_core.py::TestMultiAgent -v
```

---

## Troubleshooting

### Common Issues

**Issue:** `ModuleNotFoundError: No module named 'anthropic'`

```bash
pip install anthropic
```

**Issue:** `APIError: Invalid API key`

```bash
# Verify key is set
echo $ANTHROPIC_API_KEY

# Or set temporarily
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Issue:** Tests failing

```bash
# Check Python version
python --version  # Should be 3.10+

# Reinstall dependencies
pip install -e . --force-reinstall

# Clear cache
rm -rf __pycache__ .pytest_cache
```

**Issue:** Out of memory

```bash
# Reduce batch size in config.yaml
training:
  batch_size: 8  # Reduce from 32
```

### Getting Help

```bash
# View logs
cat autoresearch.log

# Debug mode
export AUTORESEARCH_LOG_LEVEL=DEBUG
python autonomous_loop.py
```

---

## Project Structure

```
autoresearch-stack/
├── autonomous_loop.py    # Main pipeline
├── config.py            # Configuration
├── storage.py           # Database
├── memory.py            # Vector store
├── prioritization.py    # Bandits
├── hypothesis.py       # Hypothesis generation
├── multi_agent.py      # Agent system
├── providers.py        # LLM provider integrations
├── orchestrators.py    # Agentic framework integrations
├── sandbox.py          # Safe execution
├── checkpoint.py       # Persistence
├── monitor.py          # Display
├── daemon.py           # Daemon mode
├── report.py           # Reports
├── figures.py          # Visualizations
├── stats.py            # Statistics
├── paper.py            # Paper generation
├── peer_review.py      # Review simulation
├── distribute.py       # Distribution
├── metaloop.py         # Self-modification
├── data_intelligence.py # Corpus cleaning
├── synthetic_data.py   # Synthetic generation
├── curriculum.py       # Adaptive scheduling
├── feedback.py         # Experiment logging
├── train_any_llm.py    # Training abstraction
│
├── k8s/                 # Kubernetes configs
├── docker-compose.yml   # Docker cluster
├── config.yaml          # Default config
├── setup.py             # Package setup
├── requirements.txt     # Dependencies
│
├── tests/               # Test suite (104 tests)
│   ├── test_core.py
│   ├── test_new_modules.py
│   └── test_hardening.py
├── README.md            # Overview
├── CHANGELOG.md         # Version history
├── SDD.md               # System design doc
├── USER_GUIDE.md        # This guide
└── .github/workflows/   # CI/CD pipeline
```

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for new features
4. Ensure all tests pass
5. Update documentation
6. Submit pull request

---

## License

MIT License - See LICENSE file.

---

## Support

- GitHub Issues: https://github.com/iknowkungfubar/autoresearch-stack/issues
- Documentation: https://github.com/iknowkungfubar/autoresearch-stack#readme
