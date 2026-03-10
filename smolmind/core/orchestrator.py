"""
Orchestrator — Decomposes complex tasks into micro-steps a small model can handle.

Key insight: Small models fail at multi-step tasks not because they can't reason,
but because they're asked to do too much at once. The Orchestrator breaks tasks
into atomic units — each step is simple enough for a 7B model to handle reliably.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import json


@dataclass
class Step:
    """A single atomic step in the execution plan."""
    index: int
    description: str                # What to do
    tool: Optional[str] = None      # Which tool to use (if any)
    expected_output: str = ""       # What success looks like
    depends_on: List[int] = field(default_factory=list)  # Step dependencies


@dataclass
class Plan:
    """Execution plan: ordered list of micro-steps."""
    task: str
    steps: List[Step]
    total_steps: int

    def __repr__(self):
        lines = [f"Plan for: {self.task}", f"Steps: {self.total_steps}"]
        for s in self.steps:
            tool_str = f" [{s.tool}]" if s.tool else ""
            lines.append(f"  {s.index + 1}. {s.description}{tool_str}")
        return "\n".join(lines)


DECOMPOSE_PROMPT = """You are a task planner for a small AI model (7B parameters).
Your job is to break a complex task into simple, atomic steps the model can execute one at a time.

Rules:
- Each step must be ONE clear action (not two)
- If a step needs a tool, specify which one
- Steps should be ordered (no circular dependencies)
- Each step should be completable in isolation given previous results
- Prefer using tools over relying on model knowledge

Available tools: {tools}

Task: {task}

Output a JSON plan:
{{
  "steps": [
    {{
      "index": 0,
      "description": "Clear description of this step",
      "tool": "tool_name or null",
      "expected_output": "What success looks like",
      "depends_on": []
    }}
  ]
}}

JSON plan:"""


ASSEMBLE_PROMPT = """You are assembling a final answer from completed steps.

Original task: {task}

Completed steps and results:
{results}

Write a clear, complete final answer to the original task. Be concise.

Final answer:"""


class Orchestrator:
    """
    Decomposes tasks into executable plans for small models.
    Adapts decomposition granularity based on task complexity.
    """

    def __init__(self, model, memory=None):
        self.model = model
        self.memory = memory

    def decompose(self, task: str) -> Plan:
        """Break a task into atomic steps."""
        tools = self.model.available_tools if hasattr(self.model, "available_tools") else []
        tool_list = ", ".join(tools) if tools else "none"

        prompt = DECOMPOSE_PROMPT.format(task=task, tools=tool_list)

        response = self.model.complete(prompt, max_tokens=1024)

        try:
            # Extract JSON from response
            plan_data = self._extract_json(response)
            steps = [
                Step(
                    index=s["index"],
                    description=s["description"],
                    tool=s.get("tool"),
                    expected_output=s.get("expected_output", ""),
                    depends_on=s.get("depends_on", []),
                )
                for s in plan_data["steps"]
            ]
        except (json.JSONDecodeError, KeyError):
            # Fallback: treat entire task as single step
            steps = [Step(index=0, description=task, tool=None, expected_output="Task complete")]

        return Plan(task=task, steps=steps, total_steps=len(steps))

    def assemble(self, task: str, results: List[str]) -> str:
        """Combine step results into a final coherent answer."""
        results_str = "\n".join(
            f"Step {i + 1}: {r}" for i, r in enumerate(results)
        )
        prompt = ASSEMBLE_PROMPT.format(task=task, results=results_str)
        return self.model.complete(prompt, max_tokens=512)

    def _extract_json(self, text: str) -> dict:
        """Extract JSON from model response (handles markdown code blocks)."""
        # Strip markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        # Find JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])

        raise json.JSONDecodeError("No JSON found", text, 0)
