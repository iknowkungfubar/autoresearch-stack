# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 7.3.x   | ✅ Active development |
| 7.2.x   | ✅ Supported |
| < 7.2   | ❌ Not supported |

## Reporting a Vulnerability

This project uses GitHub's **private vulnerability reporting** feature (preferred). To report a vulnerability:

1. Go to https://github.com/iknowkungfubar/autoresearch-stack/security/advisories
2. Click "Report a vulnerability"
3. Describe the issue, including steps to reproduce

Alternatively, email `turin@autoresearch.io` with:
- Subject: `[SECURITY] Vulnerability in autoresearch-stack`
- Description of the vulnerability
- Steps to reproduce
- Affected versions

### Response Times

| Severity | Initial response | Fix timeline |
|----------|-----------------|--------------|
| 🔴 Critical | Within 24 hours | 72 hours |
| 🟡 High | Within 48 hours | 1 week |
| 🟢 Medium | Within 1 week | 2 weeks |
| 🔵 Low | Within 2 weeks | Next release |

## Security Design Principles

### No Authentication or User Data
This is a local research tool with no API endpoints, no user accounts, and no stored PII. It runs entirely on the user's machine and connects only to external LLM APIs that the user explicitly configures.

### API Key Safety
- API keys are read from **environment variables only** — never from config files
- The `config.to_dict()` method automatically masks API keys in output
- The CI pipeline scans for hardcoded secrets on every commit

### Code Execution Safety
The sandbox module (`sandbox.py`) uses **AST-based validation** to prevent dangerous code execution:
- Blocks imports of `os`, `sys`, `subprocess`, `socket`, `ctypes`
- Blocks calls to `eval()`, `exec()`, `__import__()`, `compile()`, `open()`
- String-matching bypasses (e.g., `import  os`, `importos`) are caught by AST parsing

### Dependency Security
- All dependencies are audited with `pip-audit` in CI
- No known vulnerabilities in the dependency tree
- Dependencies are minimal (numpy, pyyaml, requests — optional packages for providers)

## Hall of Fame

We appreciate responsible disclosure. Contributors who report valid security issues will be credited here (with permission).

## Security-Relevant Files

| File | Purpose |
|------|---------|
| `sandbox.py` | Code execution sandbox with AST validation |
| `providers.py` | API key handling for 17+ LLM providers |
| `.github/workflows/ci.yml` | CI pipeline with secret scanning |
| `SECURITY.md` | This file |
