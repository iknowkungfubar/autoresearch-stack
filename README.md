# Autonomous Research Stack

> Build and ship autonomous LLM training research systems

**Status:** v6.1 (Production) | **License:** MIT

An autonomous research stack for continuously improving LLM training through automated experimentation. Inspired by [Karpathy autorearch](https://github.com/karpathy/autoresearch), designed for single-GPU research labs.

---

## What's Included (v6.1)

| Component | File | Description |
|-----------|------|-------------|
| Data Intelligence | `data_intelligence.py` | Corpus cleaning, noise filtering |
| Synthetic Data | `synthetic_data.py` | LLM-powered generation with Evol-Instruct |
| Curriculum | `curriculum.py` | Adaptive scheduling |
| Feedback | `feedback.py` | Experiment logging & failure classification |
| Training | `train_any_llm.py` | Training abstraction (stub) |
| Pipeline | `autonomous_loop.py` | Full orchestration |
| Configuration | `config.py` + `config.yaml` | Environment-based config |
| Storage | `storage.py` | SQLite experiment database |
| Memory | `memory.py` | Vector store with search |
| Prioritization | `prioritization.py` | Bandit-based selection |
| Hypothesis | `hypothesis.py` | LLM-driven hypothesis generation |
| Multi-Agent | `multi_agent.py` | Specialized agent system |
| Sandbox | `sandbox.py` | Safe code execution |
| Checkpoint | `checkpoint.py` | State persistence |
| Monitor | `monitor.py` | Real-time status |
| Report | `report.py` | Markdown reports |
| Figures | `figures.py` | Visualization (matplotlib) |
| Stats | `stats.py` | Summary statistics |
| Paper | `paper.py` | Research paper generation |
| MetaLoop | `metaloop.py` | Self-modification |
| Daemon | `daemon.py` | Continuous operation |
| Agent | `agent.md` | RALPH agent guardrails |
| Instructions | `prompt.md` | Loop execution prompt |

---

## Development Roadmap

| Version | Features | Status |
|---------|----------|--------|
| v2 | Foundation, curriculum | Complete |
| v3.0 | LLM generation, config, DB | Complete |
| v3.1 | Memory, bandits, hypothesis | Complete |
| v4.0 | Multi-agent, sandbox, checkpoint | Shipped |
| v4.1 | CI/CD, Docker, tests | Shipped |
| v5.0 | Reporting, figures, stats | Shipped |
| v5.2 | Paper generation | Shipped |
| v6.0 | Advanced autonomy, daemon mode | Shipped |
| v6.1 | Self-modification, meta-loop | Shipped |
| v7.0 | Distribution (multi-node, K8s) | Planned |

---

## The Metric

**val_bpb** (validation bits per byte) - Lower is better

---

## Quick Start

```bash
# Install
pip install -e .

# Configure
export ANTHROPIC_API_KEY=sk-ant-...

# Run autonomous loop
python autonomous_loop.py --experiments 100

# Run tests
pytest tests/

# Or use Docker
docker build -t autoresearch-stack .
docker run --rm -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY autoresearch-stack
```

---

## Key Files

| Phase | Files |
|-------|-------|
| Foundation | config.py, synthetic_data.py, curriculum.py |
| Intelligence | memory.py, prioritization.py, hypothesis.py |
| Multi-Agent | multi_agent.py |
| Production | sandbox.py, checkpoint.py, monitor.py |
| Reporting | report.py, figures.py, stats.py |

---

## Research References

- [Karpathy autorearch](https://github.com/karpathy/autoresearch) - val_bpb metric
- [Ouroboros](https://github.com/razzant/ouroboros) - Self-modifying
- [AI Scientist](https://github.com/SakanaAI/AI-Scientist) - Paper generation

---

## Constraints (Never Changed)

1. **val_bpb** is the ONLY metric
2. ONE change per experiment
3. Revert on regression
4. Single-GPU focused

---

*Built for researchers who want autonomous experimentation.*