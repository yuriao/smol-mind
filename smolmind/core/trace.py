"""TraceLogger — Full audit trail of agent execution."""

from __future__ import annotations
from datetime import datetime
from typing import List
import json


class TraceLogger:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.events: List[dict] = []
        self.start_time = None

    def start(self, task: str):
        self.start_time = datetime.now()
        self._log("task_start", {"task": task})
        if self.verbose:
            print(f"\n🧠 SmolMind starting: {task}\n")

    def log_plan(self, plan):
        self._log("plan", {"steps": len(plan.steps)})
        if self.verbose:
            print(f"📋 Plan: {plan.total_steps} steps")
            for s in plan.steps:
                tool = f" → [{s.tool}]" if s.tool else ""
                print(f"   {s.index + 1}. {s.description}{tool}")
            print()

    def log_step(self, index: int, step):
        self._log("step_start", {"index": index, "description": step.description})
        if self.verbose:
            print(f"⚡ Step {index + 1}: {step.description}")

    def log_result(self, index: int, result: str):
        self._log("step_done", {"index": index, "result": result[:200]})
        if self.verbose:
            print(f"   ✅ {result[:100]}{'...' if len(result) > 100 else ''}\n")

    def log_retry(self, index: int, feedback: str):
        self._log("retry", {"index": index, "feedback": feedback})
        if self.verbose:
            print(f"   🔄 Retry: {feedback}")

    def finish(self, result: str):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self._log("done", {"elapsed_s": elapsed, "result": result[:200]})
        if self.verbose:
            print(f"\n✨ Done in {elapsed:.1f}s\n")

    def _log(self, event: str, data: dict):
        self.events.append({
            "ts": datetime.now().isoformat(),
            "event": event,
            **data,
        })

    def export(self) -> str:
        return json.dumps(self.events, indent=2)
