# Contributing to Autonomous Research Stack

Thanks for your interest! This project is open-source under the MIT License.

## Getting Started

```bash
git clone https://github.com/iknowkungfubar/autoresearch-stack.git
cd autoresearch-stack
python -m venv venv
source venv/bin/activate
pip install -e .
```

## Development

### Code Style

This project uses [ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
ruff check .      # lint
ruff check --fix .  # auto-fix
```

### Type Checking

[mypy](https://mypy-lang.org/) is used for static type checking:

```bash
mypy . --ignore-missing-imports
```

Goal: **0 mypy errors** at all times.

### Testing

[pytest](https://docs.pytest.org/) with coverage:

```bash
pytest tests/ -q                    # all tests
pytest tests/ -q --cov=./           # with coverage
pytest tests/test_providers.py -v   # specific file
```

Goal: **All tests passing** before every commit.

### Project Structure

```
autoresearch-stack/
├── autonomous_loop.py     # Main pipeline orchestration
├── autoresearch/          # Python package
├── config.py              # Configuration
├── providers.py           # LLM provider integrations
├── orchestrators.py       # Agent orchestrator integrations
├── synthetic_data.py      # LLM data generation
├── curriculum.py          # Adaptive scheduling
├── memory.py              # Vector store
├── prioritization.py      # Bandit selection
├── hypothesis.py          # Hypothesis generation
├── feedback.py            # Experiment logging
├── sandbox.py             # Safe execution
├── checkpoint.py          # State persistence
├── monitor.py             # Status display
├── daemon.py              # Background execution
├── distribute.py          # Multi-node management
├── metaloop.py            # Self-modification
├── report.py              # Report generation
├── figures.py             # Visualization
├── stats.py               # Statistics
├── paper.py               # Paper generation
├── peer_review.py         # Review simulation
├── data_intelligence.py   # Corpus cleaning
├── train_any_llm.py       # Training abstraction
├── storage.py             # SQLite database
├── multi_agent.py         # Multi-agent architecture
├── tests/                 # 148+ tests
├── k8s/                   # Kubernetes deployment
├── docs/dev/              # Development documentation
└── setup.py               # Package setup
```

## Making Changes

1. Create a feature branch: `git checkout -b feat/my-change`
2. Make focused, atomic commits
3. Run tests: `pytest tests/ -q`
4. Run type checks: `mypy . --ignore-missing-imports`
5. Run lint: `ruff check .`
6. Open a PR against `main`

### Commit Message Format

```
type: description

- Bullet points with specific changes
- Reference issues where applicable
```

Types: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `chore`

## Release Process

1. Update version in `setup.py` and `autonomous_loop.py`
2. Update `CHANGELOG.md`
3. Tag: `git tag v0.7.x && git push --tags`
4. CI builds and publishes to PyPI automatically on tagged commits

## Code of Conduct

Be respectful, constructive, and inclusive. This is a research tool — help each other learn.
