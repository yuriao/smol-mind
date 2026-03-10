"""
Coordinator — Orchestrates multiple Worker agents.

Receives a complex task, decomposes it into parallel subtasks,
assigns each to the best-suited Worker, and assembles results.
"""

from __future__ import annotations
from typing import List, Dict, Optional
import json


ASSIGN_PROMPT = """You are a task coordinator managing specialized AI workers.

Available workers:
{workers}

Task to distribute: {task}

Decompose this task into subtasks and assign each to the best worker.
Output JSON only:
{{
  "subtasks": [
    {{
      "worker_role": "worker role name",
      "subtask": "specific subtask description",
      "depends_on": []
    }}
  ]
}}

JSON:"""


ASSEMBLE_PROMPT = """Combine these worker results into a final coherent answer.

Original task: {task}

Worker results:
{results}

Final answer:"""


class Coordinator:
    """
    Multi-agent coordinator.
    Decomposes tasks and dispatches to specialized Workers.
    """

    def __init__(self, model, workers: List, verbose: bool = True):
        self.model = model
        self.workers: Dict[str, object] = {w.role: w for w in workers}
        self.verbose = verbose

    def run(self, task: str) -> str:
        """Run a task using multiple specialized workers."""
        if self.verbose:
            print(f"\n🎯 Coordinator: {task}")
            print(f"   Workers: {list(self.workers.keys())}\n")

        # Decompose into worker assignments
        assignments = self._assign(task)

        if self.verbose:
            print(f"📋 {len(assignments)} subtasks assigned")

        # Execute (respecting dependencies)
        results = {}
        for assignment in assignments:
            role = assignment["worker_role"]
            subtask = assignment["subtask"]
            depends = assignment.get("depends_on", [])

            # Build context from dependencies
            context = None
            if depends:
                dep_results = [results.get(str(d), "") for d in depends if str(d) in results]
                context = "\n".join(dep_results) if dep_results else None

            if role not in self.workers:
                role = list(self.workers.keys())[0]  # Fallback to first worker

            worker = self.workers[role]
            if self.verbose:
                print(f"⚡ [{role}] {subtask[:60]}...")

            result = worker.handle(subtask, context=context)
            results[subtask[:30]] = result

            if self.verbose:
                print(f"   ✅ Done\n")

        # Assemble final answer
        return self._assemble(task, results)

    def _assign(self, task: str) -> List[dict]:
        """Decompose task into worker assignments."""
        workers_desc = "\n".join(
            f"- {role}: {w.model.model} with tools [{', '.join(t.name for t in w.tools)}]"
            for role, w in self.workers.items()
        )

        prompt = ASSIGN_PROMPT.format(task=task, workers=workers_desc)
        response = self.model.complete(prompt, max_tokens=1024)

        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            data = json.loads(response[start:end])
            return data.get("subtasks", [])
        except Exception:
            # Fallback: assign everything to first worker
            first_role = list(self.workers.keys())[0]
            return [{"worker_role": first_role, "subtask": task, "depends_on": []}]

    def _assemble(self, task: str, results: Dict[str, str]) -> str:
        """Combine worker results into final answer."""
        results_str = "\n\n".join(
            f"[{role}]: {result}" for role, result in results.items()
        )
        prompt = ASSEMBLE_PROMPT.format(task=task, results=results_str)
        return self.model.complete(prompt, max_tokens=1024)

    def worker_stats(self) -> str:
        """Print worker performance stats."""
        lines = ["Worker Stats:"]
        for role, worker in self.workers.items():
            lines.append(f"  {role}: {worker.task_count} tasks, {worker.success_rate:.0%} success")
        return "\n".join(lines)
