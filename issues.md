# Issues & Technical Debt

Documenting known issues, bugs, and technical debt in the autoresearch-stack.

---

## Critical Issues

*None*

---

## Resolved Issues

### 1. torch Import Failure → FIXED
**Status:** FIXED (graceful fallback with TORCH_AVAILABLE check)

### 2. Documentation Version Mismatch → FIXED
**Status:** FIXED in v7.2 — SDD updated to v7.0

### 3. Missing Sandbox Execution → FIXED
**Status:** FIXED — sandbox.py shipped in v4.0

### 4. Missing Checkpoint System → FIXED
**Status:** FIXED — checkpoint.py shipped in v4.0

### 5. Test Suite Missing → FIXED
**Status:** FIXED in v7.2 — 104 tests passing

### 6. CI/CD Pipeline Minimal → FIXED
**Status:** FIXED in v7.2 — Added lint enforcement, security scan, coverage

### 7. Lint Errors (131 ruff violations) → FIXED
**Status:** FIXED in v7.2 — All ruff checks pass

### 8. Bare except Clauses → FIXED
**Status:** FIXED in v7.2 — All changed to `except Exception`

---

## Minor Issues / Technical Debt

### 9. ChromaDB Not Available

**Severity:** LOW

**Description:** Vector search uses SimpleVectorStore fallback. Could benefit from ChromaDB with `pip install chromadb`.

**Status:** OPTIONAL ENHANCEMENT (by design — graceful fallback)

---

### 10. train_any_llm Stub

**Severity:** LOW

**Description:** Minimal placeholder for actual LLM training integration.

**Status:** OK (by design — placeholder for future integration)

---

## Version History

| Date | Issue | Status |
|------|------|--------|
| 2026-04-20 | torch import | FIXED (graceful) |
| 2026-04-20 | documentation mismatch | FIXED |
| 2026-04-20 | sandbox missing | FIXED |
| 2026-04-20 | checkpoint missing | FIXED |
| 2026-04-21 | test suite expansion | FIXED (104 tests) |
| 2026-04-21 | CI/CD hardening | FIXED |
| 2026-04-21 | lint cleanup | FIXED (131→0 ruff errors) |
| 2026-04-21 | bare except clauses | FIXED |
| 2026-04-21 | SDD outdated (v4.0) | FIXED (updated to v7.0) |

---

## Backlog

| Priority | Issue | Owner |
|----------|-------|-------|
| LOW | ChromaDB integration | SWE |
| LOW | Actual LLM training integration | SWE |
| MEDIUM | Increase test coverage to 80% | SDET |

---

*Updated: 2026-04-21 — Sprint 7.2 Hardening*
