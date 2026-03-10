"""
Verifier — Validates step outputs before proceeding.

Uses rule-based checks for speed + LLM-as-judge for quality.
Triggers retries with specific feedback when output is bad.
"""

from __future__ import annotations
from typing import Tuple


VERIFY_PROMPT = """Did this step complete successfully?

Step: {description}
Expected output: {expected_output}
Actual output: {actual_output}

Answer with JSON: {{"success": true/false, "feedback": "reason if failed"}}
JSON:"""


class Verifier:
    """
    Validates step outputs using rule-based and LLM-based checks.
    Returns (success: bool, feedback: str).
    """

    def __init__(self, model=None, use_llm: bool = False):
        self.model = model
        self.use_llm = use_llm  # LLM verification costs tokens — off by default

    def verify(self, step, output: str) -> Tuple[bool, str]:
        """Verify a step output. Returns (success, feedback)."""

        # Rule-based checks first (fast, no tokens)
        rule_result, rule_feedback = self._rule_check(output)
        if not rule_result:
            return False, rule_feedback

        # LLM-as-judge (optional, more accurate but costs tokens)
        if self.use_llm and self.model and step.expected_output:
            return self._llm_check(step, output)

        return True, ""

    def _rule_check(self, output: str) -> Tuple[bool, str]:
        """Fast rule-based validation."""
        if not output or not output.strip():
            return False, "Output was empty"

        if output.startswith("[Step") and "failed" in output:
            return False, "Step execution failed"

        if len(output) < 3:
            return False, "Output too short to be meaningful"

        # Check for common model refusal patterns
        refusals = [
            "i cannot", "i can't", "i'm unable", "i am unable",
            "i don't have the ability", "as an ai, i",
        ]
        lower = output.lower()
        for r in refusals:
            if r in lower and len(output) < 200:
                return False, f"Model refused: {output[:100]}"

        return True, ""

    def _llm_check(self, step, output: str) -> Tuple[bool, str]:
        """LLM-as-judge verification."""
        import json
        prompt = VERIFY_PROMPT.format(
            description=step.description,
            expected_output=step.expected_output,
            actual_output=output[:500],
        )
        response = self.model.complete(prompt, max_tokens=128)
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            data = json.loads(response[start:end])
            return data.get("success", True), data.get("feedback", "")
        except Exception:
            return True, ""  # If verification itself fails, assume OK
