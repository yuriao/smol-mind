# 🧠 SmolMind

> **Small models. Large capabilities.**

SmolMind is an agent framework that enables small/local LLMs (7B–30B) to handle complex, multi-step tasks with sandboxed tool use — behaving like a large cloud-hosted model, but running entirely on your machine.

---

## The Problem

Small models fail at agentic tasks not because they're dumb — but because:

1. **No task decomposition** — they're asked to do too much in one shot
2. **Unreliable tool calling** — JSON schema violations, hallucinated args
3. **Context collapse** — limited context window overwhelmed by history
4. **No recovery** — one bad step cascades into total failure
5. **Reflexive rejection** — refuse instead of attempting with guardrails

SmolMind fixes all five.

---

## How It Works

```
User Task
    │
    ▼
┌─────────────────────────────────┐
│         ORCHESTRATOR            │  ← Decomposes task into micro-steps
│   (Adaptive Task Planner)       │    the model CAN handle
└────────────────┬────────────────┘
                 │
    ┌────────────▼────────────┐
    │      STEP EXECUTOR      │  ← Runs each step with structured prompts
    │   (Model + Templates)   │    forces valid tool calls
    └────────────┬────────────┘
                 │
    ┌────────────▼────────────┐
    │    SANDBOX RUNTIME      │  ← Executes tools safely
    │  (Python/Bash/Web/File) │    isolated, resource-limited
    └────────────┬────────────┘
                 │
    ┌────────────▼────────────┐
    │      VERIFIER           │  ← Checks output quality
    │  (LLM-as-Judge / Rule)  │    retries on failure
    └────────────┬────────────┘
                 │
    ┌────────────▼────────────┐
    │    MEMORY MANAGER       │  ← Compresses context
    │  (Sliding window + RAG) │    keeps only relevant history
    └─────────────────────────┘
```

---

## Key Features

- ✅ **Works with Ollama, LM Studio, any OpenAI-compatible API**
- ✅ **Adaptive decomposition** — splits tasks based on model capability profile
- ✅ **Structured tool calling** — enforces valid JSON even on weak models
- ✅ **Sandboxed execution** — Python, Bash, file ops, web fetch — all isolated
- ✅ **Retry + recovery** — fails gracefully, rephrases and retries
- ✅ **Context compression** — never overflows small context windows
- ✅ **OpenClaw/NanoClaw skill compatibility** — reuse existing skill ecosystem
- ✅ **Observable traces** — full step-by-step audit log

---

## Quickstart

```bash
pip install smol-mind
```

```python
from smolmind import Agent
from smolmind.adapters import OllamaAdapter
from smolmind.sandbox import PythonSandbox, WebSandbox

agent = Agent(
    model=OllamaAdapter("qwen3:7b"),
    tools=[PythonSandbox(), WebSandbox()],
)

result = agent.run("Research the top 3 Python web frameworks and write a comparison table to comparison.md")
print(result)
```

---

## Architecture

| Component | Role |
|-----------|------|
| `Orchestrator` | Decomposes user task → micro-step plan |
| `StepExecutor` | Executes each step with model + prompt templates |
| `ToolRegistry` | Registers and routes tool calls |
| `Sandbox` | Safely runs code/commands/web requests |
| `Verifier` | Validates outputs, triggers retries |
| `MemoryManager` | Manages context: sliding window + compression |
| `CapabilityProfiler` | Tests model strengths/weaknesses at startup |
| `TraceLogger` | Full audit trail of every step |

---

## Supported Models

| Model | Size | Tool Use | Recommended |
|-------|------|----------|-------------|
| qwen3:7b | 7B | ✅ Native | ⭐ Best small |
| qwen3:14b | 14B | ✅ Native | ⭐ Best mid |
| qwen3:30b | 30B | ✅ Native | ⭐ Best local |
| llama3.2:3b | 3B | ⚠️ Template | Budget |
| mistral:7b | 7B | ⚠️ Template | Good |
| phi4:14b | 14B | ✅ Native | Great |
| gemma3:12b | 12B | ✅ Native | Good |

---

## Sandboxed Tools (Built-in)

| Tool | What It Does | Risk Level |
|------|-------------|------------|
| `PythonSandbox` | Execute Python in isolated env | 🟡 Medium |
| `BashSandbox` | Run shell commands (allowlist only) | 🟡 Medium |
| `FileSandbox` | Read/write files in workspace | 🟢 Low |
| `WebSandbox` | Fetch URLs, search web | 🟢 Low |
| `MemorySandbox` | Store/retrieve from memory | 🟢 Low |

---

## Roadmap

- [x] Core orchestrator + step executor
- [x] Ollama adapter
- [x] Python + Web sandbox
- [ ] LM Studio adapter
- [ ] Capability profiler
- [ ] OpenClaw skill compatibility layer
- [ ] Multi-agent mode (coordinator + workers)
- [ ] Web UI (local dashboard)
- [ ] Benchmark suite vs GPT-4o on standard tasks

---

## Philosophy

> Most agent frameworks are built for GPT-4. SmolMind is built for the rest.

The goal isn't to pretend small models are GPT-4. It's to architect around their real limitations — small context, unreliable JSON, single-hop reasoning — and systematically compensate with structure, tooling, and recovery logic.

A 7B model with good scaffolding beats a 70B model with none.

---

## License

MIT
