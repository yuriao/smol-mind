"""
Benchmark example — compare two agent configurations.
"""

from smolmind import Agent
from smolmind.adapters import OllamaAdapter
from smolmind.sandbox import PythonSandbox, WebSandbox
from smolmind.benchmark import BenchmarkSuite

suite = BenchmarkSuite()

# Agent 1: Small model, no SmolMind scaffolding (baseline)
class NaiveAgent:
    def __init__(self, model):
        self.model = model
    def run(self, task):
        return self.model.complete(task, max_tokens=1024)

# Agent 2: SmolMind-powered
smart_agent = Agent(
    model=OllamaAdapter("qwen3:7b"),
    tools=[PythonSandbox(), WebSandbox()],
    verbose=False,
)

naive_agent = NaiveAgent(OllamaAdapter("qwen3:7b"))

# Compare
print(suite.compare([
    (smart_agent, "qwen3:7b + SmolMind"),
    (naive_agent, "qwen3:7b (naive)"),
], categories=["reasoning", "coding"], difficulties=["easy", "medium"]))
