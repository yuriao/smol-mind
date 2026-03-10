"""
LMStudioAdapter — Connects SmolMind to LM Studio local models.
LM Studio exposes an OpenAI-compatible API at localhost:1234.
"""

from __future__ import annotations
from typing import List, Optional
import json
import urllib.request


class LMStudioAdapter:
    """
    Adapter for LM Studio local models.
    LM Studio runs an OpenAI-compatible server at http://localhost:1234.

    Usage:
        model = LMStudioAdapter()  # uses whatever model is loaded in LM Studio
        agent = Agent(model=model, tools=[...])
    """

    def __init__(
        self,
        model: str = "local-model",
        base_url: str = "http://localhost:1234",
        context_window: int = 32768,
        temperature: float = 0.1,
    ):
        self.model = model
        self.base_url = base_url
        self.context_window = context_window
        self.temperature = temperature
        self._verify_connection()

    def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        """Send prompt via OpenAI-compatible /v1/completions."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }

        response = self._post("/v1/completions", payload)
        return response["choices"][0]["text"].strip()

    def chat(self, messages: List[dict], max_tokens: int = 1024) -> str:
        """Send messages via OpenAI-compatible /v1/chat/completions."""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }

        response = self._post("/v1/chat/completions", payload)
        return response["choices"][0]["message"]["content"].strip()

    def list_models(self) -> List[str]:
        """List models available in LM Studio."""
        try:
            response = self._get("/v1/models")
            return [m["id"] for m in response.get("data", [])]
        except Exception:
            return []

    def _verify_connection(self):
        """Check LM Studio is running."""
        try:
            models = self.list_models()
            if models:
                self.model = models[0]  # Use first available model
                print(f"✅ LM Studio connected. Using: {self.model}")
            else:
                print("⚠️  LM Studio running but no model loaded. Load a model in LM Studio first.")
        except Exception:
            raise ConnectionError(
                f"Cannot connect to LM Studio at {self.base_url}. "
                "Is LM Studio running with 'Start Server' enabled?"
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
        return f"LMStudioAdapter(model={self.model}, url={self.base_url})"
