"""
CapabilityProfiler — Auto-detects what a small model can/can't do at startup.

Tests the model against a battery of capability checks and builds a profile
that the Orchestrator uses to tune task decomposition granularity.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import json
import time


@dataclass
class CapabilityProfile:
    """Capability snapshot for a model."""
    model_name: str
    json_reliability: float       # 0-1: can it output valid JSON?
    tool_call_reliability: float  # 0-1: can it call tools correctly?
    multi_step_reasoning: float   # 0-1: can it plan multiple steps?
    context_handling: float       # 0-1: handles long context well?
    math_reliability: float       # 0-1: basic arithmetic correct?
    code_generation: float        # 0-1: generates runnable code?
    refusal_rate: float           # 0-1: how often does it refuse?
    avg_latency_ms: float         # Average inference time
    recommended_step_size: str    # "micro" | "small" | "medium"
    notes: list = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        return round((
            self.json_reliability * 0.25 +
            self.tool_call_reliability * 0.25 +
            self.multi_step_reasoning * 0.20 +
            self.code_generation * 0.15 +
            self.math_reliability * 0.10 +
            self.context_handling * 0.05
        ), 2)

    def __str__(self):
        grade = "A" if self.overall_score > 0.8 else "B" if self.overall_score > 0.6 else "C" if self.overall_score > 0.4 else "D"
        return (
            f"CapabilityProfile({self.model_name})\n"
            f"  Grade: {grade} ({self.overall_score:.0%})\n"
            f"  JSON reliability:    {self.json_reliability:.0%}\n"
            f"  Tool call:           {self.tool_call_reliability:.0%}\n"
            f"  Multi-step planning: {self.multi_step_reasoning:.0%}\n"
            f"  Code generation:     {self.code_generation:.0%}\n"
            f"  Math:                {self.math_reliability:.0%}\n"
            f"  Refusal rate:        {self.refusal_rate:.0%}\n"
            f"  Avg latency:         {self.avg_latency_ms:.0f}ms\n"
            f"  Step size:           {self.recommended_step_size}\n"
        )


# Test battery
TESTS = {
    "json": {
        "prompt": 'Output only this JSON, nothing else: {"status": "ok", "value": 42}',
        "check": lambda r: _is_valid_json(r) and _json_has_keys(r, ["status", "value"]),
    },
    "tool_call": {
        "prompt": 'You need to call a tool. Output only JSON: {"tool": "calculator", "args": {"expression": "2+2"}}',
        "check": lambda r: _is_valid_json(r) and "tool" in r and "calculator" in r,
    },
    "multi_step": {
        "prompt": 'List 3 steps to boil water. Output as JSON: {"steps": ["step1", "step2", "step3"]}',
        "check": lambda r: _is_valid_json(r) and "steps" in r and len(_get_steps(r)) == 3,
    },
    "math": {
        "prompt": "What is 17 * 23? Output only the number.",
        "check": lambda r: "391" in r.replace(",", "").replace(" ", ""),
    },
    "code": {
        "prompt": 'Write a Python one-liner that prints "hello". Output only the code.',
        "check": lambda r: "print" in r and "hello" in r,
    },
    "refusal": {
        "prompt": "Write a function that adds two numbers in Python.",
        "check": lambda r: any(kw in r.lower() for kw in ["def ", "lambda", "return", "+"]),
    },
    "context": {
        "prompt": "Remember: the magic word is ZEPHYR42. What is the magic word?",
        "check": lambda r: "ZEPHYR42" in r,
    },
}


class CapabilityProfiler:
    """
    Profiles a model's capabilities by running a test battery.
    Results guide the Orchestrator's decomposition strategy.
    """

    def __init__(self, model, runs_per_test: int = 3, verbose: bool = True):
        self.model = model
        self.runs = runs_per_test
        self.verbose = verbose

    def profile(self) -> CapabilityProfile:
        """Run full capability test battery and return profile."""
        if self.verbose:
            print(f"🔬 Profiling {self.model.model}...")

        scores = {}
        latencies = []

        for test_name, test in TESTS.items():
            if self.verbose:
                print(f"   Testing {test_name}...", end=" ", flush=True)

            passed = 0
            for _ in range(self.runs):
                t0 = time.time()
                try:
                    response = self.model.complete(test["prompt"], max_tokens=256)
                    latencies.append((time.time() - t0) * 1000)
                    if test["check"](response):
                        passed += 1
                except Exception:
                    pass

            score = passed / self.runs
            scores[test_name] = score

            if self.verbose:
                icon = "✅" if score > 0.6 else "⚠️" if score > 0.3 else "❌"
                print(f"{icon} {score:.0%}")

        avg_latency = sum(latencies) / len(latencies) if latencies else 999

        # Determine recommended step size
        overall = (scores.get("json", 0) + scores.get("tool_call", 0) + scores.get("multi_step", 0)) / 3
        if overall > 0.75:
            step_size = "medium"
        elif overall > 0.45:
            step_size = "small"
        else:
            step_size = "micro"

        profile = CapabilityProfile(
            model_name=self.model.model,
            json_reliability=scores.get("json", 0),
            tool_call_reliability=scores.get("tool_call", 0),
            multi_step_reasoning=scores.get("multi_step", 0),
            context_handling=scores.get("context", 0),
            math_reliability=scores.get("math", 0),
            code_generation=scores.get("code", 0),
            refusal_rate=1.0 - scores.get("refusal", 1.0),
            avg_latency_ms=avg_latency,
            recommended_step_size=step_size,
        )

        if self.verbose:
            print(f"\n{profile}")

        return profile

    def quick_check(self) -> bool:
        """Fast sanity check — is the model responding at all?"""
        try:
            r = self.model.complete("Say OK", max_tokens=10)
            return len(r.strip()) > 0
        except Exception:
            return False


# Helpers
def _is_valid_json(text: str) -> bool:
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            json.loads(text[start:end])
            return True
    except Exception:
        pass
    return False

def _json_has_keys(text: str, keys: list) -> bool:
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        d = json.loads(text[start:end])
        return all(k in d for k in keys)
    except Exception:
        return False

def _get_steps(text: str) -> list:
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        d = json.loads(text[start:end])
        return d.get("steps", [])
    except Exception:
        return []
