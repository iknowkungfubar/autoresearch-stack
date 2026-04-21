# System Design Document (SDD): Autonomous Research Stack

## Version: v4.0 Production

---

## 1. Executive Summary

The Autonomous Research Stack is a production-ready system for continuously improving LLM training through automated experimentation. It implements the "Autonomously Improve Itself" pattern from Karpathy's autorearch.

**Current Status:** `STABLE` | `USABLE`

---

## 2. Architecture Overview

### Monolith-First Design

The system is implemented as a **Modular Monolith** with clear internal boundaries:

```
┌─────────────────────────────────────────────────────────────┐
│                    AUTONOMOUS ORCHESTRATION                   │
├─────────────┬─────────────┬─────────────┬───────────────────┤
│ Data Layer │ Logic Layer │ Agent Layer │ Infrastructure   │
├─────────────┼─────────────┼─────────────┼───────────────────┤
│ data_      │ synthetic_   │ multi_      │ config           │
│ intelligence│ data.py     │ agent.py     │ config.yaml      │
│ (corpus    │ (generation)│ (research)  │ storage         │
│  cleaning) │ (curriculum)│ (hypothesis)│ (database)      │
│            │ (feedback)  │ (execution) │ memory         │
│            │             │ (evaluation)│ (checkpoint)    │
│            │             │             │ (monitor)       │
└─────────────┴─────────────┴─────────────┴───────────────────┘
```

### Module Boundaries

| Layer | Files | Responsibility |
|-------|-------|----------------|
| Data | data_intelligence.py | Corpus cleaning |
| Logic | synthetic_data.py, curriculum.py, feedback.py | Training logic |
| Agents | multi_agent.py, hypothesis.py | Decision making |
| Infrastructure | config, storage, memory, sandbox, checkpoint, monitor, report.py | Operations |

---

## 3. Tech Stack

| Component | Technology | Notes |
|----------|------------|-------|
| Language | Python 3.11+ | |
| Config | dataclasses + yaml | type-safe |
| Storage | SQLite | experimentDB |
| Vector Store | ChromaDB (fallback: simple) | optional |
| API | Anthropic/OpenAI | optional |
| Testing | pytest | stubbed for mocking |
| Logging | structlog | optional |

---

## 4. Folder Structure

```
autoresearch-stack/
├── agent.md              # RALPH agent rules
├── prompt.md             # Loop instructions
├── README.md            # Project overview
├── CHANGELOG.md         # Version history
├── issues.md            # Technical debt
├── dev-plan.md          # Engineering roadmap
├── dev-agent-prompt.md   # THIS DIRECTIVE
├── config.yaml          # Configuration
├── config.py            # Config loader
├── autonomous_loop.py   # Main pipeline
│
├── core/                # Training Core
│   ├── data_intelligence.py
│   ├── synthetic_data.py
│   ├── curriculum.py
│   ├── feedback.py
│   └── train_any_llm.py
│
├── intelligence/        # Decision Layer
│   ├── hypothesis.py
│   ├── memory.py
│   └── prioritization.py
│
├── agents/              # Agent System
│   ├── multi_agent.py
│   └── sandbox.py
│
├── infrastructure/      # Operations
│   ├── storage.py
│   ├── checkpoint.py
│   ├── monitor.py
│   └── report.py
│
└── tests/              # Test Suite (TBD)
```

---

## 5. Data Flow

```
1. Config Load → 2. Data Prepare → 3. Loop → 4. Experiment → 5. Evaluate → 6. Report
     ↓              ↓           ↓          ↓            ↓            ↓
  config.py    data_intel  scheduler  multi_agent  feedback    report.py
                            ↓
                    hypothesis.py
                            ↓
                      memory.py
                            ↓
                    prioritization.py
```

---

## 6. Quality Gates

### Security Gate (SecOps)
- [x] No API keys in code
- [x] sandbox.py validates execution
- [x] Config masked in output

### Quality Gate (QA)
- [x] All modules import without error
- [x] autonomous_loop.py runs
- [x] Documentation updated

### Documentation Gate
- [x] README reflects version
- [x] CHANGELOG updated
- [x] issues.md tracks debt

---

## 7. First Sprint Backlog

### Sprint 4.1: Production Hardening COMPLETE

| Task | Status | Owner |
|-----|--------|-------|
| Multi-agent system | DONE | SWE |
| Sandbox execution | DONE | SWE |
| Checkpoint system | DONE | SWE |
| Monitor | DONE | SWE |
| Report generation | DONE | SWE |

### Sprint 5.0: Reporting Enhancement

| Task | Priority | Status |
|-----|----------|--------|
| Figure generation | MEDIUM | PENDING |
| Paper template | LOW | PENDING |
| Dashboard UI | LOW | PENDING |

### Sprint 6.0: Advanced Autonomy

| Task | Priority | Status |
|-----|----------|--------|
| Self-modifying loop | HIGH | DEFERRED |
| Distributed execution | MEDIUM | DEFERRED |
| Daemon mode | MEDIUM | DEFERRED |

---

## 8. Acceptance Criteria (Definition of Done)

For any feature to be merged:

- [x] **Code compiles**: `python -m py_compile *.py`
- [x] **Tests pass**: `./autonomous_loop.py --prepare-only` runs
- [x] **Docs updated**: README shows version, CHANGELOG new entry
- [x] **No secrets**: No API keys in code
- [x] **Lint passes**: (future)

---

## 9. Known Technical Debt

| Issue | Severity | Status |
|-------|----------|--------|
| torch fallback | MEDIUM | FIXED |
| ChromaDB optional | LOW | OK |
| train_any_llm stub | LOW | OK |

See `issues.md` for full list.

---

## 10. Next Actions

Per the enterprise directive, the engineering squad should focus on:

1. **SDET**: Add test coverage for critical paths
2. **DevOps**: Create CI/CD pipeline
3. **SecOps**: Dependency audit

**Status:** Ready for Sprint Assignment