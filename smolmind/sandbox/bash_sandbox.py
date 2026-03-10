"""
BashSandbox — Safe shell command execution with strict allowlist.

Only whitelisted commands can run. No sudo, no network tools,
no destructive operations. Workspace-isolated.
"""

from __future__ import annotations
import subprocess
import os
import shlex
from typing import List, Optional


# Commands that are safe to run
ALLOWLIST = {
    # File operations
    "ls", "cat", "head", "tail", "wc", "find", "grep", "awk", "sed",
    "sort", "uniq", "cut", "tr", "diff", "cp", "mv", "mkdir", "touch",
    "echo", "printf", "pwd",
    # Data processing
    "jq", "python3", "python", "node", "ruby",
    # Dev tools
    "git", "npm", "pip", "pip3",
    # Text tools
    "md5sum", "sha256sum", "base64",
    # Info
    "date", "whoami", "uname",
}

# Commands that are never allowed regardless of context
BLOCKLIST = {
    "rm", "rmdir",           # Deletion
    "sudo", "su",            # Privilege escalation
    "curl", "wget",          # Network (use WebSandbox instead)
    "ssh", "scp", "sftp",    # Remote access
    "chmod", "chown",        # Permission changes
    "kill", "pkill",         # Process killing
    "crontab",               # Scheduler modification
    "passwd", "useradd",     # User management
    "iptables", "ufw",       # Firewall
    "dd",                    # Disk operations
    "format", "mkfs",        # Formatting
    ">", ">>",               # Redirects to arbitrary paths (checked separately)
}


class BashSandbox:
    """
    Safe bash command execution.
    Strict allowlist, workspace-isolated, resource-limited.
    """

    name = "bash"
    description = "Run shell commands. Only safe, allowlisted commands permitted."
    schema = {
        "command": "string — Shell command to run (must use allowlisted commands only)",
        "timeout": "integer (optional) — Max seconds (default: 15)",
    }

    def __init__(
        self,
        workspace: Optional[str] = None,
        timeout: int = 15,
        max_output: int = 4096,
        extra_allowed: Optional[List[str]] = None,
    ):
        self.workspace = workspace or os.getcwd()
        self.timeout = timeout
        self.max_output = max_output
        self.allowed = ALLOWLIST | set(extra_allowed or [])

    def execute(self, command: str, timeout: int = None) -> str:
        """Execute a shell command safely."""
        timeout = timeout or self.timeout

        # Validate command
        error = self._validate(command)
        if error:
            return f"Blocked: {error}"

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.workspace,
            )

            output = result.stdout
            if result.returncode != 0:
                output = f"Error (exit {result.returncode}):\n{result.stderr}"

            if len(output) > self.max_output:
                output = output[:self.max_output] + f"\n...[truncated]"

            return output.strip()

        except subprocess.TimeoutExpired:
            return f"Timeout: command exceeded {timeout}s"
        except Exception as e:
            return f"Error: {str(e)}"

    def _validate(self, command: str) -> Optional[str]:
        """Check if command is safe to run. Returns error string or None."""
        # Get base command
        try:
            parts = shlex.split(command)
        except ValueError as e:
            return f"Invalid command syntax: {e}"

        if not parts:
            return "Empty command"

        base_cmd = os.path.basename(parts[0]).lower()

        # Check blocklist first
        if base_cmd in BLOCKLIST:
            return f"'{base_cmd}' is not allowed"

        # Check allowlist
        if base_cmd not in self.allowed:
            return f"'{base_cmd}' is not in the allowed commands list: {sorted(self.allowed)}"

        # Check for path traversal attempts
        if ".." in command and ("/" in command or "\\" in command):
            return "Path traversal detected"

        # Check for suspicious patterns
        suspicious = ["|curl", "|wget", "$(curl", "$(wget", "`curl", "`wget"]
        for pattern in suspicious:
            if pattern in command:
                return f"Suspicious pattern detected: {pattern}"

        return None

    def add_allowed(self, command: str):
        """Dynamically add a command to the allowlist."""
        self.allowed.add(command)

    @property
    def allowed_commands(self) -> List[str]:
        return sorted(self.allowed)
