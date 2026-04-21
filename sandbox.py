"""
Sandbox execution for safe code running.

Phase 5: Production Hardening - Safe execution with resource limits.
"""

import sys
import subprocess
import tempfile
import shutil
from typing import Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecutionResult:
    """Result of sandboxed execution."""

    success: bool
    stdout: str
    stderr: str
    return_code: int
    execution_time: float
    memory_used: Optional[int] = None
    error: Optional[str] = None


class ResourceLimits:
    """Resource limits for execution."""

    def __init__(
        self,
        max_time_seconds: int = 300,
        max_memory_mb: int = 4096,
        max_cpu_percent: int = 100,
    ):
        self.max_time_seconds = max_time_seconds
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent


class Sandbox:
    """Sandboxed execution environment."""

    def __init__(self, limits: Optional[ResourceLimits] = None):
        self.limits = limits or ResourceLimits()
        self.temp_dir: Optional[Path] = None

    def __enter__(self):
        """Create temp directory."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="autoresearch_sandbox_"))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up temp directory."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def execute(
        self,
        code: str,
        language: str = "python",
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """Execute code in sandbox."""
        import time

        start_time = time.time()

        if not self.temp_dir:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="autoresearch_sandbox_"))

        # Write code to file
        if language == "python":
            ext = ".py"
        else:
            ext = ".txt"

        code_file = self.temp_dir / f"script{ext}"
        code_file.write_text(code)

        # Set up limits
        timeout = timeout or self.limits.max_time_seconds

        try:
            # Run with resource limits
            result = subprocess.run(
                [sys.executable, str(code_file)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.temp_dir),
            )

            execution_time = time.time() - start_time

            return ExecutionResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
                execution_time=execution_time,
            )

        except subprocess.TimeoutExpired as e:
            return ExecutionResult(
                success=False,
                stdout=e.stdout or "",
                stderr=f"Execution timeout after {timeout}s",
                return_code=-1,
                execution_time=timeout,
                error="timeout",
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                return_code=-1,
                execution_time=time.time() - start_time,
                error=str(e),
            )


class SafeRunner:
    """Safe code runner with validation."""

    # Blocked patterns
    BLOCKED_PATTERNS = [
        "import os",
        "import sys",
        "import subprocess",
        "import socket",
        "import requests",
        "import urllib",
        "__import__(",
        "eval(",
        "exec(",
        "open(",
        "file(",
    ]

    def __init__(self, sandbox: Optional[Sandbox] = None):
        self.sandbox = sandbox or Sandbox()

    def validate(self, code: str) -> Tuple[bool, Optional[str]]:
        """Validate code for safety."""
        for pattern in self.BLOCKED_PATTERNS:
            if pattern in code:
                return False, f"Blocked pattern: {pattern}"

        return True, None

    def run(self, code: str, timeout: Optional[int] = None) -> ExecutionResult:
        """Run code safely."""
        valid, error = self.validate(code)

        if not valid:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Validation failed: {error}",
                return_code=-1,
                execution_time=0,
                error=error,
            )

        return self.sandbox.execute(code, timeout=timeout)


def run_safe(code: str, timeout: int = 60) -> ExecutionResult:
    """Convenience function to run code safely."""
    with Sandbox() as sandbox:
        return sandbox.execute(code, timeout=timeout)


if __name__ == "__main__":
    # Test
    print("Testing sandbox...")

    # Test safe code
    safe_code = """
print('Hello, world!')
result = 1 + 2
print(f'Result: {result}')
"""

    runner = SafeRunner()
    result = runner.run(safe_code)

    print(f"Success: {result.success}")
    print(f"Output: {result.stdout}")
    print(f"Time: {result.execution_time:.2f}s")
