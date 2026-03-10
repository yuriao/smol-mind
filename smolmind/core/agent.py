"""
Agent — Top-level entry point for SmolMind.
Wires together orchestrator, executor, sandbox, memory, and verifier.
"""

from __future__ import annotations
from typing import Any, List, Optional
from smolmind.core.orchestrator import Orchestrator
from smolmind.core.executor import StepExecutor
from smolmind.core.memory import MemoryManager
from smolmind.core.verifier import Verifier
from smolmind.core.trace import TraceLogger


class Agent:
    """
    Main agent class. Takes a task, decomposes it, executes step-by-step,
    verifies outputs, recovers from failures.

    Example:
        agent = Agent(model=OllamaAdapter("qwen3:7b"), tools=[PythonSandbox()])
        result = agent.run("Summarize the latest AI news and save to news.md")
    """

    def __init__(
        self,
        model,
        tools: Optional[List] = None,
        max_steps: int = 20,
        max_retries: int = 3,
        verbose: bool = True,
    ):
        self.model = model
        self.tools = tools or []
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.verbose = verbose

        self.memory = MemoryManager(max_tokens=model.context_window // 2)
        self.orchestrator = Orchestrator(model=model, memory=self.memory)
        self.executor = StepExecutor(
            model=model,
            tools=self.tools,
            memory=self.memory,
            max_retries=max_retries,
        )
        self.verifier = Verifier(model=model)
        self.trace = TraceLogger(verbose=verbose)

    def run(self, task: str) -> str:
        """
        Execute a task end-to-end.
        Returns the final result as a string.
        """
        self.trace.start(task)

        # Step 1: Decompose task into micro-steps
        plan = self.orchestrator.decompose(task)
        self.trace.log_plan(plan)

        results = []
        for i, step in enumerate(plan.steps):
            self.trace.log_step(i, step)

            # Step 2: Execute step
            output = self.executor.execute(step, context=results)

            # Step 3: Verify output
            verified, feedback = self.verifier.verify(step, output)
            if not verified:
                self.trace.log_retry(i, feedback)
                # Retry with feedback
                output = self.executor.execute(
                    step, context=results, retry_feedback=feedback
                )

            results.append(output)
            self.trace.log_result(i, output)

            # Step 4: Update memory
            self.memory.add(step=step, result=output)

        # Assemble final answer
        final = self.orchestrator.assemble(task, results)
        self.trace.finish(final)
        return final

    def __repr__(self):
        return f"Agent(model={self.model}, tools={len(self.tools)}, max_steps={self.max_steps})"
