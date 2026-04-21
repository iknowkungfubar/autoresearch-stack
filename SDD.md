# System Design Document (SDD): Autonomous Research Stack

## Version: v7.0 Production

---

## 1. Executive Summary

The Autonomous Research Stack is a production-ready system for continuously improving LLM training through automated experimentation. It implements the "Autonomously Improve Itself" pattern from Karpathy's autorearch.

**Current Status:** `STABLE` | `USABLE` | **104 tests passing**

---

## 2. Architecture Overview

### Monolith-First Design

The system is implemented as a **Modular Monolith** with clear internal boundaries:

```
┌─────────────────────────────────────────────────────────────────┐
│                    AUTONOMOUS ORCHESTRATION                       │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│  Data Layer  │ Logic Layer  │ Agent Layer  │ Infrastructure     │
├──────────────┼──────────────┼──────────────┼────────────────────┤
│ data_        │ synthetic_   │ multi_       │ config             │
│ intelligence │ data.py      │ agent.py     │ config.yaml        │
│ (corpus      │ (generation) │ (research)   │ storage            │
│  cleaning)   │ (curriculum) │ (hypothesis) │ (database)         │
│              │ (feedback)   │ (execution)  │ memory             │
│              │ (providers)  │ (orchestr.)  │ (checkpoint)       │
│              │              │              │ (monitor)          │
└──────────────┴──────────────┴──────────────┴────────────────────┘
```

### Module Boundaries

| Layer | Files | Responsibility |
|-------|-------|----------------|
| Data | `data_intelligence.py` | Corpus cleaning |
| Logic | `synthetic_data.py`, `curriculum.py`, `feedback.py`, `train_any_llm.py` | Training logic |
| Agents | `multi_agent.py`, `hypothesis.py`, `providers.py`, `orchestrators.py` | Decision making & LLM providers |
| Infrastructure | `config`, `storage`, `memory`, `sandbox`, `checkpoint`, `monitor`, `report.py` | Operations |
| Reporting | `report.py`, `figures.py`, `stats.py`, `paper.py`, `peer_review.py` | Research output |
| Autonomy | `metaloop.py`, `daemon.py`, `distribute.py` | Self-modification & distribution |

---

## 3. Tech Stack

| Component | Technology | Notes |
|----------|------------|-------|
| Language | Python 3.11+ | |
| Config | dataclasses + yaml | type-safe |
| Storage | SQLite | experimentDB |
| Vector Store | SimpleVectorStore (ChromaDB optional) | graceful fallback |
| API | Anthropic/OpenAI/13+ providers | optional, lazy-loaded |
| Testing | pytest | 104 tests |
| Linting | ruff | all checks passing |
| Container | Docker + Docker Compose | multi-node cluster |
| Orchestration | LangChain/CrewAI/AutoGen/LlamaIndex | optional integrations |

---

## 4. Folder Structure (Actual)

```
autoresearch-stack/
├── agent.md                  # RALPH agent guardrails
├── prompt.md                 # Loop instructions
├── README.md                 # Project overview
├── CHANGELOG.md              # Version history
├── SDD.md                    # THIS DOCUMENT
├── issues.md                 # Technical debt tracker
├── dev-plan.md               # Engineering roadmap
├── dev-agent-prompt.md        # Enterprise orchestrator directive
├── config.yaml               # Configuration
├── config.py                 # Config loader (dataclass-based)
├── autonomous_loop.py        # Main pipeline orchestrator
│
├── # Core Training
├── data_intelligence.py      # Corpus cleaning
├── synthetic_data.py         # LLM-powered generation (Evol-Instruct)
├── curriculum.py             # Adaptive scheduling
├── feedback.py               # Experiment logging & failure classification
├── train_any_llm.py          # Training abstraction (stub)
│
├── # Intelligence Layer
├── hypothesis.py             # LLM-driven hypothesis generation
├── memory.py                 # Vector store with search
├── prioritization.py         # Bandit-based experiment selection
│
├── # Multi-Agent & Providers
├── multi_agent.py            # Specialized agent system
├── providers.py              # 13+ LLM provider integrations
├── orchestrators.py          # 6 agentic framework integrations
├── sandbox.py                # Safe code execution
│
├── # Infrastructure
├── storage.py                # SQLite experiment database
├── checkpoint.py             # State persistence
├── monitor.py                # Real-time status
├── daemon.py                 # Continuous operation
├── distribute.py             # Multi-node cluster
│
├── # Reporting
├── report.py                 # Markdown report generation
├── figures.py                # Visualization (matplotlib)
├── stats.py                  # Summary statistics
├── paper.py                  # Research paper generation
├── peer_review.py            # Review simulation
│
├── # Meta
├── metaloop.py               # Self-modification loop
│
├── # Deployment
├── Dockerfile                # Docker image
├── docker-compose.yml        # Local multi-node cluster
├── k8s/deployment.yaml       # Kubernetes deployment
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
│
└── tests/
    ├── test_core.py          # Core module tests
    ├── test_new_modules.py   # v5.0+ module tests
    └── test_hardening.py     # v7.2 hardening tests (51 tests)
```

---

## 5. Data Flow

```
1. Config Load → 2. Data Prepare → 3. Loop → 4. Experiment → 5. Evaluate → 6. Report
     ↓              ↓           ↓          ↓            ↓            ↓
  config.py    data_intel  scheduler  multi_agent  feedback    report.py
                            ↓                    ↓            figures.py
                    hypothesis.py          prioritization    stats.py
                            ↓                         ↓      paper.py
                      memory.py                  storage
```

---

## 6. LLM Provider Support (v7.1)

| Provider | Type | Status |
|----------|------|--------|
| Anthropic | Cloud | ✅ |
| OpenAI | Cloud | ✅ |
| OpenRouter | Cloud | ✅ |
| Zen AI | Cloud | ✅ |
| Azure OpenAI | Cloud | ✅ |
| Google Vertex | Cloud | ✅ |
| Mistral | Cloud | ✅ |
| Ollama | Local | ✅ |
| vLLM | Local | ✅ |
| LM Studio | Local | ✅ |
| LiteLLM | Proxy | ✅ |
| llama.cpp | Local | ✅ |
| TextGen WebUI | Local | ✅ |

---

## 7. Quality Gates

### Security Gate (SecOps) ✅
- [x] No API keys in code (only placeholders in docs)
- [x] No known dependency vulnerabilities (pip-audit clean)
- [x] sandbox.py validates execution
- [x] Config masks API keys in output
- [x] `.gitignore` covers `.env`, `.db`, `checkpoints/`, `memory/`, `logs/`

### Quality Gate (QA) ✅
- [x] All 104 tests passing
- [x] All modules import without error
- [x] `autonomous_loop.py --prepare-only` runs
- [x] `ruff check .` passes with zero errors

### Documentation Gate ✅
- [x] README reflects v7.0
- [x] CHANGELOG updated through v7.2
- [x] SDD reflects current architecture
- [x] issues.md tracks active debt

---

## 8. Sprint Backlog

### Sprint 4.1→7.0: COMPLETE ✅

| Task | Status |
|-----|--------|
| Multi-agent system | DONE |
| Sandbox execution | DONE |
| Checkpoint system | DONE |
| Monitor | DONE |
| Report generation | DONE |
| Figure generation | DONE |
| Paper generation | DONE |
| Peer review simulation | DONE |
| Self-modification (metaloop) | DONE |
| Distribution system | DONE |
| Daemon mode | DONE |
| Multi-provider support | DONE |
| Multi-orchestrator support | DONE |

### Sprint 7.2: Hardening (COMPLETE ✅)

| Task | Priority | Status |
|-----|----------|--------|
| SDD update | HIGH | DONE |
| Lint cleanup (ruff) | HIGH | DONE |
| Security audit (SecOps) | HIGH | DONE |
| CI/CD hardening | HIGH | DONE |
| Test coverage expansion (51 new tests) | HIGH | DONE |
| Issues/changelog update | MEDIUM | DONE |

---

## 9. Definition of Done

For any feature to be merged:

- [x] **Code compiles**: `python -m py_compile *.py`
- [x] **Tests pass**: `pytest tests/ -v` (104/104)
- [x] **Lint passes**: `ruff check .` (0 errors)
- [x] **Docs updated**: README, CHANGELOG, SDD current
- [x] **No secrets**: No API keys in code
- [x] **No vulnerabilities**: `pip-audit` clean

---

## 10. Known Technical Debt

| Issue | Severity | Status |
|-------|----------|--------|
| torch fallback | MEDIUM | FIXED (graceful) |
| ChromaDB optional | LOW | OK (simple fallback) |
| train_any_llm stub | LOW | OK (by design) |
| CI/CD needs security scan | MEDIUM | FIXED (v7.2) |
| Test coverage for providers/orchestrators | MEDIUM | FIXED (v7.2 — 51 new tests) |
| `--exit-zero` in ruff CI | MEDIUM | FIXED (v7.2 — enforced) |

See `issues.md` for full list.

---

## 11. Microservices Evolution Assessment

Per the Enterprise Directive (Phase B trigger: STABLE + USABLE), a microservices decomposition was evaluated:

**Decision: NOT PURSUED**

Rationale:
- The system is a single-GPU research tool, not a web service
- No external API endpoints to split
- No high-load or high-change features suitable for extraction
- The modular monolith design provides sufficient decoupling
- Service discovery / API gateways add complexity without benefit

**Recommended path:** Continue Phase A hardening — CI/CD, test coverage, security scanning.

---

*Status: v7.0 Production Ready — Sprint 7.2 Hardening COMPLETE*
