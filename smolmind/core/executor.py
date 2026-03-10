"""
StepExecutor — Executes individual steps using the model + tools.

Handles the core challenge of small models: unreliable tool calling.
Uses structured prompt templates to force valid JSON tool calls,
with retry logic and fallback strategies.
"""

from __future__ import annotations
from typing import List, Optional, Any
import json
import re


TOOL_CALL_PROMPT = """You are executing step {index} of a task.

Step: {description}
Tool to use: {tool}
Tool schema: {tool_schema}

Previous results:
{context}

{retry_hint}

Call the tool by outputting ONLY valid JSON matching this exact schema:
{tool_schema}

Your tool call JSON:"""


NO_TOOL_PROMPT = """You are executing step {index} of a task.

Step: {description}

Previous results:
{context}

{retry_hint}

Complete this step. Be concise and direct. Output only the result, nothing else.

Result:"""


class StepExecutor:
    """
    Executes a single step with the model.
    Handles tool routing, prompt templating, retry logic.
    """

    def __init__(self, model, tools: List, memory=None, max_retries: int = 3):
        self.model = model
        self.tools = {t.name: t for t in tools}
        self.memory = memory
        self.max_retries = max_retries

    def execute(
        self,
        step,
        context: List[str],
        retry_feedback: Optional[str] = None,
    ) -> str:
        """Execute a single step, with retries on failure."""
        for attempt in range(self.max_retries):
            try:
                result = self._try_execute(step, context, retry_feedback, attempt)
                return result
            except Exception as e:
                retry_feedback = f"Previous attempt failed: {str(e)}. Try a different approach."
                if attempt == self.max_retries - 1:
                    return f"[Step {step.index} failed after {self.max_retries} attempts: {str(e)}]"

    def _try_execute(self, step, context, retry_feedback, attempt) -> str:
        """Single execution attempt."""
        context_str = self._format_context(context)
        retry_hint = f"Note: {retry_feedback}" if retry_feedback else ""

        if step.tool and step.tool in self.tools:
            return self._execute_with_tool(step, context_str, retry_hint)
        else:
            return self._execute_without_tool(step, context_str, retry_hint)

    def _execute_with_tool(self, step, context_str: str, retry_hint: str) -> str:
        """Execute step using a tool."""
        tool = self.tools[step.tool]

        prompt = TOOL_CALL_PROMPT.format(
            index=step.index + 1,
            description=step.description,
            tool=step.tool,
            tool_schema=json.dumps(tool.schema, indent=2),
            context=context_str,
            retry_hint=retry_hint,
        )

        response = self.model.complete(prompt, max_tokens=512)

        # Parse tool call from response
        tool_args = self._parse_tool_call(response)

        # Execute tool in sandbox
        result = tool.execute(**tool_args)
        return str(result)

    def _execute_without_tool(self, step, context_str: str, retry_hint: str) -> str:
        """Execute step using model knowledge only."""
        prompt = NO_TOOL_PROMPT.format(
            index=step.index + 1,
            description=step.description,
            context=context_str,
            retry_hint=retry_hint,
        )

        return self.model.complete(prompt, max_tokens=1024)

    def _parse_tool_call(self, response: str) -> dict:
        """Extract tool call JSON from model response. Robust to malformed output."""
        # Try direct JSON parse
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        # Strip markdown
        for pattern in ["```json", "```"]:
            if pattern in response:
                parts = response.split(pattern)
                if len(parts) > 1:
                    try:
                        return json.loads(parts[1].split("```")[0].strip())
                    except json.JSONDecodeError:
                        pass

        # Find JSON object in text
        match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse tool call from: {response[:200]}")

    def _format_context(self, context: List[str]) -> str:
        if not context:
            return "None yet."
        lines = []
        for i, r in enumerate(context[-3:]):  # Last 3 results only (context management)
            lines.append(f"Step {i + 1} result: {r[:500]}")  # Truncate long results
        return "\n".join(lines)
