"""
BenchmarkSuite — Compare SmolMind agents against each other and GPT-4o baseline.

Runs standard tasks, measures pass rate, latency, and step efficiency.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
import time
import json


@dataclass
class TaskResult:
    task_id: str
    category: str
    difficulty: str
    passed: bool
    result: str
    steps_used: int
    latency_s: float
    error: Optional[str] = None


@dataclass
class BenchmarkReport:
    agent_name: str
    total: int
    passed: int
    failed: int
    by_category: Dict[str, Dict] = field(default_factory=dict)
    by_difficulty: Dict[str, Dict] = field(default_factory=dict)
    avg_latency_s: float = 0.0
    results: List[TaskResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    def __str__(self):
        lines = [
            f"\n{'='*50}",
            f"BENCHMARK REPORT: {self.agent_name}",
            f"{'='*50}",
            f"Overall:  {self.passed}/{self.total} ({self.pass_rate:.0%})",
            f"Latency:  {self.avg_latency_s:.1f}s avg",
            "",
            "By Category:",
        ]
        for cat, stats in self.by_category.items():
            rate = stats['passed'] / stats['total'] if stats['total'] > 0 else 0
            lines.append(f"  {cat:<25} {stats['passed']}/{stats['total']} ({rate:.0%})")

        lines.append("\nBy Difficulty:")
        for diff, stats in self.by_difficulty.items():
            rate = stats['passed'] / stats['total'] if stats['total'] > 0 else 0
            lines.append(f"  {diff:<25} {stats['passed']}/{stats['total']} ({rate:.0%})")

        lines.append("\nFailed Tasks:")
        for r in self.results:
            if not r.passed:
                lines.append(f"  ❌ [{r.task_id}] {r.error or 'check failed'}")

        lines.append("=" * 50)
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps({
            "agent": self.agent_name,
            "pass_rate": self.pass_rate,
            "passed": self.passed,
            "total": self.total,
            "avg_latency_s": self.avg_latency_s,
            "by_category": self.by_category,
            "by_difficulty": self.by_difficulty,
        }, indent=2)


class BenchmarkSuite:
    """
    Runs benchmark tasks against a SmolMind agent and produces reports.

    Usage:
        suite = BenchmarkSuite(tasks=BENCHMARK_TASKS)
        report = suite.run(agent, name="qwen3:7b + SmolMind")
        print(report)
    """

    def __init__(self, tasks: Optional[List[Dict]] = None, verbose: bool = True):
        from smolmind.benchmark.tasks import BENCHMARK_TASKS
        self.tasks = tasks or BENCHMARK_TASKS
        self.verbose = verbose

    def run(
        self,
        agent,
        name: Optional[str] = None,
        categories: Optional[List[str]] = None,
        difficulties: Optional[List[str]] = None,
        timeout: int = 120,
    ) -> BenchmarkReport:
        """Run benchmark tasks against an agent."""
        agent_name = name or repr(agent)
        tasks = self._filter_tasks(categories, difficulties)

        if self.verbose:
            print(f"\n🏁 Benchmarking: {agent_name}")
            print(f"   Tasks: {len(tasks)}")
            print()

        results = []
        for task in tasks:
            result = self._run_task(agent, task, timeout)
            results.append(result)

            if self.verbose:
                icon = "✅" if result.passed else "❌"
                print(f"{icon} [{task['id']}] {task['task'][:50]}... ({result.latency_s:.1f}s)")

        return self._build_report(agent_name, results)

    def compare(self, agents: List[tuple], **kwargs) -> str:
        """
        Compare multiple agents side by side.
        agents: list of (agent, name) tuples
        """
        reports = []
        for agent, name in agents:
            report = self.run(agent, name=name, **kwargs)
            reports.append(report)

        lines = ["\n" + "=" * 60, "COMPARISON REPORT", "=" * 60]
        lines.append(f"{'Agent':<35} {'Pass Rate':>10} {'Latency':>10}")
        lines.append("-" * 60)
        for r in sorted(reports, key=lambda x: x.pass_rate, reverse=True):
            lines.append(f"{r.agent_name:<35} {r.pass_rate:>10.0%} {r.avg_latency_s:>9.1f}s")
        lines.append("=" * 60)
        return "\n".join(lines)

    def _run_task(self, agent, task: Dict, timeout: int) -> TaskResult:
        """Run a single benchmark task."""
        t0 = time.time()
        error = None
        result_str = ""
        passed = False

        try:
            result_str = agent.run(task["task"])
            passed = task["check"](result_str)
        except Exception as e:
            error = str(e)
            passed = False

        latency = time.time() - t0

        return TaskResult(
            task_id=task["id"],
            category=task["category"],
            difficulty=task["difficulty"],
            passed=passed,
            result=result_str[:500],
            steps_used=0,
            latency_s=latency,
            error=error,
        )

    def _filter_tasks(self, categories, difficulties) -> List[Dict]:
        tasks = self.tasks
        if categories:
            tasks = [t for t in tasks if t["category"] in categories]
        if difficulties:
            tasks = [t for t in tasks if t["difficulty"] in difficulties]
        return tasks

    def _build_report(self, agent_name: str, results: List[TaskResult]) -> BenchmarkReport:
        by_cat: Dict[str, Dict] = {}
        by_diff: Dict[str, Dict] = {}

        for r in results:
            # By category
            if r.category not in by_cat:
                by_cat[r.category] = {"passed": 0, "total": 0}
            by_cat[r.category]["total"] += 1
            if r.passed:
                by_cat[r.category]["passed"] += 1

            # By difficulty
            if r.difficulty not in by_diff:
                by_diff[r.difficulty] = {"passed": 0, "total": 0}
            by_diff[r.difficulty]["total"] += 1
            if r.passed:
                by_diff[r.difficulty]["passed"] += 1

        avg_latency = sum(r.latency_s for r in results) / len(results) if results else 0

        return BenchmarkReport(
            agent_name=agent_name,
            total=len(results),
            passed=sum(1 for r in results if r.passed),
            failed=sum(1 for r in results if not r.passed),
            by_category=by_cat,
            by_difficulty=by_diff,
            avg_latency_s=avg_latency,
            results=results,
        )
