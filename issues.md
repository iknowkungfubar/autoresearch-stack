# Issues & Technical Debt

Documenting known issues, bugs, and technical debt in the autoresearch-stack.

---

## Critical Issues

### 1. torch Import Failure

**Issue:** `librocm_smi64.so.1: cannot open shared object file`

**Severity:** HIGH

**Description:** Torch cannot be imported due to missing ROCm library (AMD GPU support library). This affects `train_any_llm.py` and any module that imports torch.

**Workaround:** 
- Install ROCm libraries: `apt install rocm-libs`
- Or use CPU-only PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

**Status:** OPEN

---

## Medium Issues

### 2. Documentation Version Mismatch

**Issue:** `improvements.md` still shows v3.1 as incomplete

**Severity:** MEDIUM

**Description:** improvements.md marks v3.1 features as incomplete even though they're shipped.

**Status:** NEEDS UPDATE

---

### 3. Missing Sandbox Execution

**Issue:** No sandboxed code execution

**Severity:** MEDIUM  

**Description:** Phase 4 requires sandboxed subprocess execution but not yet implemented.

**Status:** PLANNED (Phase 5)

---

### 4. Missing Checkpoint System

**Issue:** No checkpoint/resume for interrupted experiments

**Severity:** MEDIUM

**Description:** Experiments cannot resume from interruption point.

**Status:** PLANNED (Phase 5)

---

## Minor Issues / Technical Debt

### 5. ChromaDB Not Available

**Issue:** Uses SimpleVectorStore fallback

**Severity:** LOW

**Description:** Vector search uses simple keyword matching. Could benefit from ChromaDB.

**Status:** DOCUMENTED

---

### 6. train_any_llm Stub

**Issue:** Minimal training abstraction

**Severity:** LOW

**Description:** train_any_llm.py is a minimal stub. Would benefit from proper LLM training integration.

**Status:** OK (by design - placeholder)

---

## Version History

| Date | Issue | Status |
|------|------|--------|
| 2026-04-20 | torch import | OPEN |
| 2026-04-20 | documentation mismatch | FIXED |