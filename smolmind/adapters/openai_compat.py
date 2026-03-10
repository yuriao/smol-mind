"""
OpenAICompatAdapter — Works with ANY OpenAI-compatible API endpoint.
Supports: OpenAI, Anthropic (via openai SDK), Together, Groq, OpenRouter,
          vLLM, text-generation-webui, koboldcpp, llamacpp server.
"""

from __future__ import annotations
from typing import List
import json
import urllib.request


class OpenAICompatAdapter:
    """
    Generic OpenAI-compatible adapter.
    Works with any provider that implements /v1/chat/completions.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com",
        api_key: str = "",
        context_window: int = 128000,
        temperature: float = 0.1,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.context_window = context_window
        self.temperature = temperature

    def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        return self.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )

    def chat(self, messages: List[dict], max_tokens: int = 1024) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=data,
            headers=headers,
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        return result["choices"][0]["message"]["content"].strip()

    def __repr__(self):
        return f"OpenAICompatAdapter(model={self.model}, base={self.base_url})"


# Convenience factories
def GroqAdapter(model: str = "llama-3.3-70b-versatile", api_key: str = "") -> OpenAICompatAdapter:
    return OpenAICompatAdapter(model=model, base_url="https://api.groq.com/openai", api_key=api_key, context_window=128000)

def TogetherAdapter(model: str = "meta-llama/Llama-3-70b-chat-hf", api_key: str = "") -> OpenAICompatAdapter:
    return OpenAICompatAdapter(model=model, base_url="https://api.together.xyz", api_key=api_key, context_window=8192)

def OpenRouterAdapter(model: str = "openai/gpt-4o-mini", api_key: str = "") -> OpenAICompatAdapter:
    return OpenAICompatAdapter(model=model, base_url="https://openrouter.ai/api", api_key=api_key, context_window=128000)
