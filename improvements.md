# Improvements Roadmap

> Current version: **v3.0** - Enhanced stack with LLM generation, adaptive curriculum, and experiment tracking

This document outlines the planned improvements to evolve the stack from v2 to production-ready.

---

## Current State (v3.0)

The v3.0 stack includes all v2 components plus:
- `data_intelligence.py` - Corpus cleaning and noise filtering
- `synthetic_data.py` - LLM-powered generation with Evol-Instruct
- `curriculum.py` - Adaptive scheduling with multiple strategies
- `feedback.py` - val_bpb reward and experiment logging
- `train_any_llm.py` - Training abstraction
- `autonomous_loop.py` - Full pipeline orchestration
- `config.py` - Environment-based configuration
- `config.yaml` - Configuration template
- `storage.py` - SQLite experiment database
- `agent.md` - RALPH agent rules
- `prompt.md` - Loop instructions

---

## v3.1: Enhanced Synthetic Data (Phase 1 - COMPLETE)

- [x] LLM-powered synthetic prompt generation (OpenAI/Anthropic API)
- [x] Evol-Instruct difficulty scaling
- [x] Perplexity-based quality filtering
- [ ] Self-instruct bootstrapping

### v3.2: Hypothesis Generation (Phase 2)

- [ ] LLM-driven hypothesis proposer
- [ ] SQLite experiment database
- [ ] Expanded failure classification
- [ ] "What has been tried" retrieval

### v3.3: Memory & Prioritization (Phase 3)

- [ ] ChromaDB vector store for semantic search
- [ ] UCB-based experiment prioritization
- [ ] Exploration vs exploitation balance
- [ ] Past experiment retrieval

### v3.4: Multi-Agent Architecture (Phase 4)

- [ ] Research Agent (literature, gaps)
- [ ] Hypothesis Agent (modifications)
- [ ] Execution Agent (run experiments)
- [ ] Evaluation Agent (analyze results)
- [ ] Consensus mechanism

---

## v4: Production Hardening

The v4 release makes the system production-ready.

### v4.1: Safe Execution

- [ ] Sandboxed subprocess execution
- [ ] Resource limits (memory, CPU, time)
- [ ] Timeout handling
- [ ] Error recovery and retry

### v4.2: Persistence

- [ ] Checkpoint system for resume
- [ ] State persistence to disk
- [ ] Graceful interruption
- [ ] Resume from last state

### v4.3: Monitoring

- [ ] Real-time status display
- [ ] Resource utilization tracking
- [ ] Progress bars
- [ ] Experiment history UI

---

## v5: Reporting & Research Automation

The v5 release adds research automation (AI Scientist-style).

### v5.1: Reporting

- [ ] Markdown report generation
- [ ] Figure generation from results
- [ ] Summary statistics
- [ ] Experiment comparisons

### v5.2: Paper Generation

- [ ] Automated manuscript drafting
- [ ] Literature review integration
- [ ] Citation management
- [ ] Peer-review simulation

---

## v6+: Advanced Autonomy

Future releases for advanced capabilities.

### v6.1: Self-Modification

- Meta-loop for methodology evolution (Ouroboros-style)
- Automatic prompt refinement
- Self-improving code

### v6.2: Distribution

- Multi-node execution
- Cloud deployment (Docker, Kubernetes)
- Resource management
- Cost estimation

### v6.3: Continuous Operation

- Daemon mode
- Auto-restart on failure
- Health checks
- Background execution

---

## Research References

The v3-v6 roadmap is inspired by:

| System | Key Innovation |
|--------|---------------|
| [Karpathy autorearch](https://github.com/karpathy/autoresearch) | val_bpb metric, 5-min loop |
| [Ouroboros](https://github.com/razzant/ouroboros) | Self-modifying methodology |
| [AI Scientist](https://github.com/SakanaAI/AI-Scientist) | Paper generation |
| [Epsilon](https://arxiv.org/abs/2603.24402v2) | Epistemic integrity |
| [DUMP](https://github.com/ZhentingWang/DUMP) | UCB curriculum |

---

## Not Planned

These are explicitly out of scope:

- ✗ Multi-GPU distributed training (single-GPU focused)
- ✗ Dataset source modification (fixed by design)
- ✗ Metric changes (val_bpb is universal)
- ✗ Full autonomous operation without bounds
- ✗ Commercial support

---

## Contributing

See `dev-plan.md` for implementation details and `agent.md` for constraints.

---

## Version History

| Version | Release Date | Description |
|---------|--------------|-------------|
| v1 | 2025 | Initial release |
| v2 | 2026 | Curriculum learning, feedback loop |
| v3.0 | 2026-04-20 | LLM generation, adaptive curriculum, experiment DB |
| v3.1 | TBD | Memory system, vector search, hypothesis |
| v3.2 | TBD | Multi-agent architecture |
| v4 | TBD | Production hardening |
| v5 | TBD | Reporting, paper generation |
| v6 | TBD | Advanced autonomy |

---

*This is the point where this stops being a framework and becomes a research agent system.*