"""
Worker — A specialized agent that handles a specific domain of tasks.
Works as part of a Swarm, receives subtasks from the Coordinator.
"""

from __future__ import annotations
from typing import List, Optional
from smolmind.core.executor import StepExecutor
from smolmind.core.memory import MemoryManager
from smolmind.core.verifier import Verifier


class Worker:
    """
    A specialized agent worker.
    Each worker has a role, set of tools, and its own memory.
    Receives subtasks from the Coordinator and returns results.
    """

    def __init__(
        self,
        role: str,
        model,
        tools: Optional[List] = None,
        max_retries: int = 3,
    ):
        self.role = role
        self.model = model
        self.tools = tools or []
        self.memory = MemoryManager(max_tokens=model.context_window // 3)
        self.executor = StepExecutor(
            model=model,
            tools=self.tools,
            memory=self.memory,
            max_retries=max_retries,
        )
        self.verifier = Verifier(model=model)
        self.task_count = 0
        self.success_count = 0

    def handle(self, subtask: str, context: Optional[str] = None) -> str:
        """
        Handle a subtask assigned by the Coordinator.
        Returns the result.
        """
        self.task_count += 1

        # Build step from subtask
        from smolmind.core.orchestrator import Step
        step = Step(
            index=self.task_count,
            description=subtask,
            tool=self._pick_tool(subtask),
        )

        context_list = [context] if context else []
        result = self.executor.execute(step, context=context_list)

        # Verify
        success, feedback = self.verifier.verify(step, result)
        if success:
            self.success_count += 1
        else:
            result = self.executor.execute(step, context=context_list, retry_feedback=feedback)

        self.memory.add(step=step, result=result)
        return result

    def _pick_tool(self, subtask: str) -> Optional[str]:
        """Heuristic: pick the right tool based on subtask description."""
        lower = subtask.lower()
        tool_hints = {
            "python": "python",
            "code": "python",
            "script": "python",
            "calculate": "python",
            "web": "web",
            "search": "web",
            "fetch": "web",
            "url": "web",
            "http": "web",
            "bash": "bash",
            "shell": "bash",
            "command": "bash",
            "git": "bash",
            "file": "bash",
        }
        tool_names = {t.name for t in self.tools}
        for hint, tool in tool_hints.items():
            if hint in lower and tool in tool_names:
                return tool
        return None

    @property
    def success_rate(self) -> float:
        if self.task_count == 0:
            return 0.0
        return self.success_count / self.task_count

    def __repr__(self):
        return f"Worker(role={self.role}, model={self.model.model}, tasks={self.task_count})"
