"""
Basic SmolMind example — research task with web + python tools.
"""

from smolmind import Agent
from smolmind.adapters import OllamaAdapter
from smolmind.sandbox import PythonSandbox, WebSandbox

agent = Agent(
    model=OllamaAdapter("qwen3:7b"),
    tools=[
        PythonSandbox(workspace="./workspace"),
        WebSandbox(),
    ],
    verbose=True,
)

result = agent.run(
    "Search for the top 3 Python web frameworks in 2026, "
    "compare them in a table, and save to frameworks.md"
)

print("\n--- Final Result ---")
print(result)
