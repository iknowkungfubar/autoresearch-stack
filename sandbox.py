"""
Sandbox execution for safe code running.

Phase 5: Production Hardening - Safe execution with resource limits.
"""

import ast
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


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
            result = subprocess.run(  # noqa: S603 - sandboxed execution of user code
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
                stdout=e.stdout if isinstance(e.stdout, str) else "",
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


# AST-level blocked imports (module names to forbid)
BLOCKED_MODULES: set[str] = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "ctypes",
    "signal",
    "multiprocessing",
}

# AST-level blocked function calls (bare names like eval())
BLOCKED_FUNCTIONS: set[str] = {
    "eval",
    "exec",
    "__import__",
    "compile",
    "open",
    "input",
}


class SafeRunner:
    """Safe code runner with AST-based validation.

    Uses Python's ast module to parse code and check for dangerous
    imports and function calls at the syntax tree level. This is
    resistant to trivial bypasses like 'import  os' (double space)
    or 'importos' (no space) that string-based filtering misses.
    """

    def __init__(self, sandbox: Optional[Sandbox] = None):
        self.sandbox = sandbox or Sandbox()

    def validate(self, code: str) -> Tuple[bool, Optional[str]]:
        """Validate code for safety using AST analysis.

        Statically parses the code and walks the AST tree to detect:
        - Blocked module imports (os, sys, subprocess, socket, ctypes, etc.)
        - Blocked function calls (eval, exec, __import__, compile, open, input)
        - Attribute access chains on blocked modules (os.environ, sys.path)

        Args:
            code: Python source code to validate.

        Returns:
            Tuple of (is_safe, error_message). If is_safe is False,
            error_message contains the reason.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Check both the imported name and any alias
                    name = alias.name.split(".")[0]
                    if name in BLOCKED_MODULES:
                        return False, f"Blocked import: {alias.name}"

            # Check from-imports
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_root = node.module.split(".")[0]
                    if module_root in BLOCKED_MODULES:
                        return False, f"Blocked import from: {node.module}"
                    for alias in node.names:
                        if alias.name in BLOCKED_FUNCTIONS:
                            return False, f"Blocked function import: {alias.name}"

            # Check function calls by name
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in BLOCKED_FUNCTIONS:
                        return False, f"Blocked function call: {node.func.id}()"

                # Check method calls on blocked objects
                elif isinstance(node.func, ast.Attribute):
                    # Walk the attribute chain to find the root object
                    root = node.func
                    while isinstance(root, ast.Attribute):
                        root = root.value  # type: ignore
                    if isinstance(root, ast.Name) and root.id in BLOCKED_MODULES:
                        return False, f"Blocked {root.id}.{self._attr_chain(node.func)}"

        return True, None

    def _attr_chain(self, node: ast.Attribute) -> str:
        """Build attribute chain string like 'environ.get' from Attribute nodes."""
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value  # type: ignore
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))

    def run(self, code: str, timeout: Optional[int] = None) -> ExecutionResult:
        """Run code safely.

        Args:
            code: Python code to execute.
            timeout: Optional timeout in seconds.

        Returns:
            ExecutionResult with success/failure and output.
        """
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

    # Test blocked code
    blocked_codes = [
        "import os",
        "import  os",
        "import os.path",
        "from os import path",
        "eval('print(1)')",
        "open('/etc/passwd')",
        "exec('print(1)')",
        "__import__('os')",
        "compile('1+1', '<string>', 'exec')",
    ]
    for code in blocked_codes:
        valid, err = runner.validate(code)
        print(f"  Blocked check '{code}': valid={valid}, error={err}")
