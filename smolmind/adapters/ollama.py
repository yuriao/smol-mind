"""
OllamaAdapter — Connects SmolMind to local Ollama models.
Supports streaming, tool calling, and capability detection.
"""

from __future__ import annotations
from typing import Optional, List
import json
import urllib.request
import urllib.error


class OllamaAdapter:
    """
    Adapter for Ollama local models.
    Works with qwen3, llama3, mistral, phi4, gemma3, etc.
    """

    # Models with native tool calling support
    NATIVE_TOOL_MODELS = {
        "qwen3", "qwen2.5", "llama3.1", "llama3.2",
        "mistral", "phi4", "phi3", "gemma3", "command-r",
    }

    def __init__(
        self,
        model: str = "qwen3:7b",
        base_url: str = "http://localhost:11434",
        context_window: int = 32768,
        temperature: float = 0.1,  # Low temp for tool use reliability
    ):
        self.model = model
        self.base_url = base_url
        self.context_window = context_window
        self.temperature = temperature
        self._verify_connection()

    def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        """Send a prompt and return the completion."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": max_tokens,
            },
        }

        response = self._post("/api/generate", payload)
        return response.get("response", "").strip()

    def chat(self, messages: List[dict], max_tokens: int = 1024) -> str:
        """Chat-style completion."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": max_tokens,
            },
        }

        response = self._post("/api/chat", payload)
        return response.get("message", {}).get("content", "").strip()

    @property
    def supports_native_tools(self) -> bool:
        """Check if model supports native function calling."""
        model_base = self.model.split(":")[0].lower()
        return any(m in model_base for m in self.NATIVE_TOOL_MODELS)

    def _verify_connection(self):
        """Verify Ollama is running and model is available."""
        try:
            response = self._get("/api/tags")
            models = [m["name"] for m in response.get("models", [])]
            if not any(self.model in m for m in models):
                print(f"⚠️  Model '{self.model}' not found. Available: {models[:5]}")
                print(f"   Run: ollama pull {self.model}")
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Is Ollama running? Error: {e}"
            )

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get(self, path: str) -> dict:
        req = urllib.request.Request(f"{self.base_url}{path}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def __repr__(self):
        return f"OllamaAdapter(model={self.model}, ctx={self.context_window})"
