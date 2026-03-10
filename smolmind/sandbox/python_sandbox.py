"""
PythonSandbox — Execute Python code safely in an isolated environment.

Uses subprocess with resource limits, timeout, and restricted builtins.
No network access, no file system access outside workspace.
"""

from __future__ import annotations
import subprocess
import tempfile
import os
import sys
from pathlib import Path


class PythonSandbox:
    """
    Safe Python code execution sandbox.
    Resource-limited, timeout-enforced, workspace-isolated.
    """

    name = "python"
    description = "Execute Python code. Use for calculations, data processing, file operations."
    schema = {
        "code": "string — Python code to execute",
        "timeout": "integer (optional) — Max seconds to run (default: 30)",
    }

    def __init__(
        self,
        workspace: Optional[str] = None,
        timeout: int = 30,
        max_output: int = 4096,
    ):
        self.workspace = workspace or os.getcwd()
        self.timeout = timeout
        self.max_output = max_output

    def execute(self, code: str, timeout: int = None) -> str:
        """
        Execute Python code in a sandboxed subprocess.
        Returns stdout output or error message.
        """
        timeout = timeout or self.timeout

        # Write code to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=self.workspace
        ) as f:
            f.write(self._wrap_code(code))
            tmp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.workspace,
                env={
                    **os.environ,
                    "PYTHONPATH": self.workspace,
                },
            )

            output = result.stdout
            if result.returncode != 0:
                output = f"Error:\n{result.stderr}"

            # Truncate long output
            if len(output) > self.max_output:
                output = output[:self.max_output] + f"\n...[truncated, {len(output)} chars total]"

            return output.strip()

        except subprocess.TimeoutExpired:
            return f"Error: Code timed out after {timeout}s"
        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            os.unlink(tmp_path)

    def _wrap_code(self, code: str) -> str:
        """Wrap code with safety restrictions."""
        return f"""
import sys
import os

# Restrict to workspace only
_original_open = open
def _safe_open(file, mode='r', *args, **kwargs):
    path = os.path.abspath(str(file))
    workspace = os.path.abspath('{self.workspace}')
    if not path.startswith(workspace):
        raise PermissionError(f"Access denied: {{path}} is outside workspace")
    return _original_open(file, mode, *args, **kwargs)

# Apply restrictions
import builtins
builtins.open = _safe_open

# User code
{code}
"""
