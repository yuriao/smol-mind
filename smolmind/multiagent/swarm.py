"""
Swarm — High-level multi-agent system builder.

Simplifies creating a Coordinator + Workers setup with sensible defaults.
Each worker can use a different model — e.g. a fast small model for research,
a larger one for writing, a code-specialized one for implementation.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any
from smolmind.multiagent.coordinator import Coordinator
from smolmind.multiagent.worker import Worker


class Swarm:
    """
    High-level multi-agent system.

    Example:
        swarm = Swarm(coordinator_model=OllamaAdapter("qwen3:14b"))
        swarm.add_worker("researcher", OllamaAdapter("qwen3:7b"), tools=[WebSandbox()])
        swarm.add_worker("coder", OllamaAdapter("qwen3:7b"), tools=[PythonSandbox()])
        swarm.add_worker("writer", OllamaAdapter("qwen3:14b"), tools=[])
        result = swarm.run("Research async Python patterns and write a tutorial with examples")
    """

    def __init__(self, coordinator_model, verbose: bool = True):
        self.coordinator_model = coordinator_model
        self.verbose = verbose
        self._workers: List[Worker] = []
        self._coordinator: Optional[Coordinator] = None

    def add_worker(
        self,
        role: str,
        model,
        tools: Optional[List] = None,
        max_retries: int = 3,
    ) -> "Swarm":
        """Add a worker to the swarm. Returns self for chaining."""
        worker = Worker(role=role, model=model, tools=tools or [], max_retries=max_retries)
        self._workers.append(worker)
        self._coordinator = None  # Reset coordinator on change
        return self

    def run(self, task: str) -> str:
        """Execute a task using the full swarm."""
        if not self._workers:
            raise ValueError("No workers added. Use swarm.add_worker() first.")

        coordinator = self._get_coordinator()
        return coordinator.run(task)

    def stats(self) -> str:
        """Get worker performance statistics."""
        if self._coordinator:
            return self._coordinator.worker_stats()
        return "No tasks run yet."

    def _get_coordinator(self) -> Coordinator:
        if self._coordinator is None:
            self._coordinator = Coordinator(
                model=self.coordinator_model,
                workers=self._workers,
                verbose=self.verbose,
            )
        return self._coordinator

    def __repr__(self):
        return f"Swarm(workers={[w.role for w in self._workers]})"
