# Autonomous Research Stack

> Build and ship autonomous LLM training research systems

**Status:** v3.0 (Enhanced) | **License:** MIT

An autonomous research stack for continuously improving LLM training through automated experimentation. Inspired by [Karpathy autorearch](https://github.com/karpathy/autoresearch), designed for single-GPU research labs.

---

## The Vision

Give an AI agent a small LLM training setup and let it experiment autonomously. It modifies code, runs training, checks the metric, keeps improvements, and repeats—overnight you wake up to better models.

```
┌─────────────────────────────────────────────────────────────┐
│                    AUTONOMOUS LOOP                          │
│                                                             │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐           │
│   │ PROPOSE  │───▶│  TRAIN   │───▶│ EVALUATE  │           │
│   │ change   │    │  5 min   │    │ val_bpb  │           │
│   └──────────┘    └──────────┘    └──────────┘           │
│        ▲                                 │                 │
│        │                                 ▼                 │
│        │                            ┌──────────┐           │
│        └────────────────────────────│ DECIDE   │           │
│         revert if worse             │ keep/rej │           │
│                                     └──────────┘           │
└─────────────────────────────────────────────────────────────┘
```

---

## What's Included (v3.0)

| Component | File | Description |
|-----------|------|-------------|
| Data Intelligence | `data_intelligence.py` | Corpus cleaning, noise filtering |
| Synthetic Data | `synthetic_data.py` | LLM-powered generation with Evol-Instruct |
| Curriculum | `curriculum.py` | Adaptive scheduling |
| Feedback | `feedback.py` | Experiment logging & failure classification |
| Training | `train_any_llm.py` | Training abstraction |
| Pipeline | `autonomous_loop.py` | Full orchestration |
| Configuration | `config.py` + `config.yaml` | Environment-based config |
| Storage | `storage.py` | SQLite experiment database |
| Agent | `agent.md` | RALPH agent guardrails |
| Instructions | `prompt.md` | Loop execution prompt |

---

## The Metric

**val_bpb** (validation bits per byte)

- Lower is better
- Single scalar metric
- Fixed, never modified
- Used by Karpathy autorearch, proven effective

---

## The Agent: RALPH

**R**esearch **A**gent for **L**oop-scheduled **H**yperparameter **P**rimary optimization

RALPH follows these rules:
- Maximize improvement per experiment, not volume
- ONE change per cycle
- Revert on regression
- Classify every failure
- Preserve reproducibility

See `agent.md` for full constraints.

---

## Quick Start

```bash
# Clone and navigate
cd autoresearch-stack

# Install dependencies
pip install torch numpy

# Run autonomous loop
python autonomous_loop.py --input data/corpus.txt

# Or use the agent directly with an LLM
# (requires API key in config.yaml)
```

---

## Configuration

Edit `config.yaml`:

```yaml
# Experiment settings
experiment_budget: 500      # max experiments
time_per_experiment: 300    # seconds (5 min)
val_target: 0.95            # target val_bpb

# Model settings
model_size: 124M
learning_rate: 1e-4

# API keys (set via environment)
# ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...
```

---

## Development Roadmap

See `dev-plan.md` for full phases and `improvements.md` for detailed roadmap.

| Version | Features | Status |
|---------|----------|--------|
| v2 | Foundation, curriculum, feedback | Complete |
| v3.0 | LLM generation, adaptive curriculum, experiment DB | Shipped |
| v3.1 | Memory system, vector search, hypothesis | Next |
| v3.2 | Multi-agent architecture | Planned |
| v4 | Production hardening | Planned |
| v5 | Reporting, paper generation | Planned |
| v6 | Advanced autonomy | Future |

---

## Research Foundation

This project synthesizes insights from:

- **[Karpathy autorearch](https://github.com/karpathy/autoresearch)** (74K stars)
  - Three-file architecture, val_bpb metric
  
- **[Ouroboros](https://github.com/razzant/ouroboros)** (500+ stars)
  - Self-modifying methodology
  
- **[Sakana AI Scientist](https://github.com/SakanaAI/AI-Scientist)** (13K stars)
  - Full paper generation pipeline

- **Epsilon** (arXiv:2603.24402)
  - Epistemic integrity, p-hacking prevention

- **DUMP/actor-Curator**
  - UCB-based curriculum learning

---

## Key Constraints (Never Changed)

These are baked into the design:

1. **val_bpb** is the ONLY metric
2. ONE change per experiment
3. Revert on regression
4. Single-GPU focused
5. No dataset source modification
6. Reproducibility always preserved

---

## Project Structure

```
autoresearch-stack/
├── README.md              # This file
├── dev-plan.md           # Development plan
├── agent.md              # RALPH agent rules
├── prompt.md             # Loop instructions
├── improvements.md       # Roadmap
├── config.yaml           # Configuration
├── autonomous_loop.py    # Main pipeline
├── data_intelligence.py  # Data cleaning
├── synthetic_data.py     # Synthetic generation
├── curriculum.py         # Training curriculum
├── feedback.py           # Evaluation
└── train_any_llm.py     # Training abstraction
```

---

## Usage with LLM Agent

To run autonomous experiments with an LLM:

1. Set API key: `export ANTHROPIC_API_KEY=sk-ant-...`
2. Run with LLM prompting: See `prompt.md` for the full prompt
3. Agent will propose changes, run experiments, commit improvements

---

## Related Projects

| Project | Description |
|---------|-------------|
| [karpathy/autoresearch](https://github.com/karpathy/autoresearch) | Original inspiration |
| [joi-lab/ouroboros](https://github.com/joi-lab/ouroboros) | Self-modifying agent |
| [SakanaAI/AI-Scientist](https://github.com/SakanaAI/AI-Scientist) | Paper generation |
| [omarkamali/curriculus](https://github.com/omarkamali/curriculus) | Curriculum library |

---

## License

MIT - See LICENSE file

---

## Contributing

See `dev-plan.md` for implementation phases. Key constraints:

- Never change the metric
- One change per experiment
- Always log failures
- Preserve reproducibility

---

*Built for researchers who want autonomous experimentation without the overhead.*