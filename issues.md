# Issues & Technical Debt

Documenting known issues, bugs, and technical debt in the autoresearch-stack.

---

## Critical Issues

### 1. torch Import Failure (WORKAROUND PROVIDED)

**Issue:** `librocm_smi64.so.1: cannot open shared object file`

**Severity:** MEDIUM (Workaround Provided)

**Description:** Torch cannot be imported due to missing ROCm library (AMD GPU). The `train_any_llm.py` now handles this gracefully with fallback.

**Workaround:** 
- Already handled in v3.0 with `TORCH_AVAILABLE` check
- Use CPU-only PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

**Status:** FIXED (graceful fallback)

---

## Resolved Issues (v4.0)

### 2. Documentation Version Mismatch → FIXED

**Status:** RESOLVED in v4.0

### 3. Missing Sandbox Execution → FIXED

**Status:** RESOLVED - sandbox.py shipped in v4.0

### 4. Missing Checkpoint System → FIXED

**Status:** RESOLVED - checkpoint.py shipped in v4.0

---

## Minor Issues / Technical Debt

### 5. ChromaDB Not Available

**Severity:** LOW

**Description:** Vector search uses simple keyword matching. Could benefit from ChromaDB with `pip install chromadb`.

**Status:** OPTIONAL ENHANCEMENT

---

### 6. train_any_llm Stub

**Severity:** LOW

**Description:** Minimal placeholder for actual LLM training integration.

**Status:** OK (by design - placeholder for future integration)

---

### 7. Missing Test Suite

**Severity:** MEDIUM

**Description:** No comprehensive test coverage for critical paths.

**Status:** TECHNICAL DEBT

---

### 8. CI/CD Pipeline Missing

**Severity:** MEDIUM

**Description:** No automated pipeline for testing and deployment.

**Status:** TECHNICAL DEBT

---

## Version History

| Date | Issue | Status |
|------|------|--------|
| 2026-04-20 | torch import | FIXED (graceful) |
| 2026-04-20 | documentation mismatch | FIXED |
| 2026-04-20 | sandbox missing | FIXED |
| 2026-04-20 | checkpoint missing | FIXED |
| 2026-04-20 | test suite | DEBT |
| 2026-04-20 | CI/CD | DEBT |

---

## Backlog

| Priority | Issue | Owner |
|----------|-------|-------|
| LOW | Add test coverage | SDET |
| MEDIUM | Create CI/CD pipeline | DevOps |
| LOW | ChromaDB integration | SWE |