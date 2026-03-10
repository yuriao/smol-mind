"""
OpenClawSkillBridge — Reuse OpenClaw/NanoClaw skills inside SmolMind.

Reads installed OpenClaw skills from the workspace and wraps them
as SmolMind-compatible tools that agents can use natively.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Dict
import subprocess
import sys
import os


class OpenClawTool:
    """
    Wraps an OpenClaw skill as a SmolMind tool.
    The agent calls it like any other tool — the bridge handles execution.
    """

    def __init__(self, skill_name: str, skill_path: Path, description: str):
        self.name = f"openclaw_{skill_name}"
        self.skill_name = skill_name
        self.skill_path = skill_path
        self.description = description
        self.schema = {
            "instruction": f"string — What to do with the {skill_name} skill",
            "args": "dict (optional) — Additional arguments",
        }

    def execute(self, instruction: str, args: Optional[Dict] = None) -> str:
        """
        Execute an OpenClaw skill instruction.
        Runs the skill's main script if available, otherwise uses the SKILL.md
        as context for the model to guide execution.
        """
        # Check for executable scripts in the skill directory
        for script_name in ["main.py", "run.py", "index.js", "main.sh"]:
            script = self.skill_path / script_name
            if script.exists():
                return self._run_script(script, instruction, args)

        # Fallback: return skill context for model to use
        skill_md = self.skill_path / "SKILL.md"
        if skill_md.exists():
            content = skill_md.read_text()[:2000]
            return f"Skill '{self.skill_name}' context:\n{content}\n\nInstruction: {instruction}"

        return f"Skill '{self.skill_name}' found but no executable entry point."

    def _run_script(self, script: Path, instruction: str, args: Optional[Dict]) -> str:
        """Run a skill script with the instruction as input."""
        env = {**os.environ, "SMOLMIND_INSTRUCTION": instruction}
        if args:
            for k, v in args.items():
                env[f"SMOLMIND_ARG_{k.upper()}"] = str(v)

        try:
            if script.suffix == ".py":
                result = subprocess.run(
                    [sys.executable, str(script)],
                    capture_output=True, text=True, timeout=30,
                    env=env, cwd=str(self.skill_path),
                )
            elif script.suffix == ".js":
                result = subprocess.run(
                    ["node", str(script)],
                    capture_output=True, text=True, timeout=30,
                    env=env, cwd=str(self.skill_path),
                )
            elif script.suffix == ".sh":
                result = subprocess.run(
                    ["bash", str(script)],
                    capture_output=True, text=True, timeout=30,
                    env=env, cwd=str(self.skill_path),
                )
            else:
                return f"Unknown script type: {script.suffix}"

            return result.stdout.strip() or result.stderr.strip() or "Done."

        except subprocess.TimeoutExpired:
            return "Skill timed out after 30s"
        except Exception as e:
            return f"Skill error: {str(e)}"


class OpenClawSkillBridge:
    """
    Discovers and wraps installed OpenClaw skills as SmolMind tools.

    Usage:
        bridge = OpenClawSkillBridge()
        tools = bridge.get_tools(["meeting-note", "weather", "github"])
        agent = Agent(model=..., tools=tools)
    """

    DEFAULT_PATHS = [
        Path.home() / ".openclaw/workspace/skills",
        Path("/opt/homebrew/lib/node_modules/openclaw/skills"),
    ]

    def __init__(self, skills_dir: Optional[str] = None):
        self.skills_dir = Path(skills_dir) if skills_dir else self._find_skills_dir()
        self._discovered: Optional[Dict[str, OpenClawTool]] = None

    def discover(self) -> Dict[str, OpenClawTool]:
        """Scan skills directory and return available tools."""
        if self._discovered is not None:
            return self._discovered

        if not self.skills_dir or not self.skills_dir.exists():
            print(f"⚠️  No OpenClaw skills directory found. Checked: {self.DEFAULT_PATHS}")
            return {}

        tools = {}
        for skill_dir in self.skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue

            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue

            # Extract description from SKILL.md
            description = self._extract_description(skill_md)
            tool = OpenClawTool(
                skill_name=skill_dir.name,
                skill_path=skill_dir,
                description=description,
            )
            tools[skill_dir.name] = tool

        self._discovered = tools
        return tools

    def get_tools(self, skill_names: Optional[List[str]] = None) -> List[OpenClawTool]:
        """
        Get SmolMind-compatible tools for specified skills.
        If skill_names is None, returns all discovered skills.
        """
        all_tools = self.discover()

        if skill_names is None:
            return list(all_tools.values())

        result = []
        for name in skill_names:
            if name in all_tools:
                result.append(all_tools[name])
            else:
                print(f"⚠️  Skill '{name}' not found. Available: {list(all_tools.keys())[:10]}")

        return result

    def list_skills(self) -> List[str]:
        """List all discovered skill names."""
        return list(self.discover().keys())

    def _extract_description(self, skill_md: Path) -> str:
        """Extract description from SKILL.md frontmatter."""
        try:
            content = skill_md.read_text()
            for line in content.split("\n"):
                if line.startswith("description:"):
                    return line.replace("description:", "").strip().strip('"').strip("'")[:200]
        except Exception:
            pass
        return f"OpenClaw skill: {skill_md.parent.name}"

    def _find_skills_dir(self) -> Optional[Path]:
        """Auto-detect OpenClaw skills directory."""
        for path in self.DEFAULT_PATHS:
            if path.exists():
                return path
        return None
