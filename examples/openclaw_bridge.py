"""
OpenClaw skill bridge example.
Use your installed OpenClaw skills inside SmolMind agents.
"""

from smolmind import Agent
from smolmind.adapters import OllamaAdapter
from smolmind.sandbox import PythonSandbox
from smolmind.openclaw import OpenClawSkillBridge

# Discover installed OpenClaw skills
bridge = OpenClawSkillBridge()
print("Available OpenClaw skills:", bridge.list_skills())

# Get specific skills as tools
openclaw_tools = bridge.get_tools(["meeting-note", "context-doctor"])

agent = Agent(
    model=OllamaAdapter("qwen3:7b"),
    tools=[PythonSandbox(), *openclaw_tools],
    verbose=True,
)

result = agent.run(
    "Summarize today's work session and create structured meeting notes."
)
print(result)
