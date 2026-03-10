"""
Multi-agent SmolMind example.
Researcher + Coder + Writer working together.
"""

from smolmind.adapters import OllamaAdapter
from smolmind.multiagent import Swarm
from smolmind.sandbox import PythonSandbox, WebSandbox

swarm = (
    Swarm(coordinator_model=OllamaAdapter("qwen3:14b"))
    .add_worker("researcher", OllamaAdapter("qwen3:7b"), tools=[WebSandbox()])
    .add_worker("coder",      OllamaAdapter("qwen3:7b"), tools=[PythonSandbox()])
    .add_worker("writer",     OllamaAdapter("qwen3:14b"), tools=[])
)

result = swarm.run(
    "Research the top Python async libraries in 2026, "
    "write a benchmark script comparing asyncio vs trio, "
    "and produce a markdown report with findings."
)

print(result)
print(swarm.stats())
