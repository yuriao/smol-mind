"""
MemoryManager — Context window management for small models.

Small models have limited context (4k-32k tokens typically).
This manager ensures we never overflow by using a sliding window
and compressing old steps into summaries.
"""

from __future__ import annotations
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class MemoryEntry:
    step_index: int
    step_description: str
    result: str
    token_estimate: int = 0

    def __post_init__(self):
        self.token_estimate = len(self.result) // 4  # Rough estimate


class MemoryManager:
    """
    Manages context window for small models.
    Strategy: keep recent steps in full, summarize older ones.
    """

    def __init__(self, max_tokens: int = 4096, keep_recent: int = 5):
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent
        self.entries: List[MemoryEntry] = []
        self.summaries: List[str] = []
        self._total_tokens = 0

    def add(self, step, result: str):
        """Add a completed step to memory."""
        entry = MemoryEntry(
            step_index=step.index,
            step_description=step.description,
            result=result,
        )
        self.entries.append(entry)
        self._total_tokens += entry.token_estimate

        # Compress if over budget
        if self._total_tokens > self.max_tokens:
            self._compress()

    def get_context(self, max_entries: Optional[int] = None) -> str:
        """Get formatted context for the model."""
        n = max_entries or self.keep_recent
        recent = self.entries[-n:]

        parts = []
        if self.summaries:
            parts.append("Earlier steps (summarized): " + " | ".join(self.summaries))

        for entry in recent:
            parts.append(f"Step {entry.step_index + 1} ({entry.step_description}): {entry.result[:300]}")

        return "\n".join(parts) if parts else "No previous steps."

    def _compress(self):
        """Compress old entries into summaries to free context."""
        if len(self.entries) <= self.keep_recent:
            return

        # Take entries beyond keep_recent and summarize
        to_compress = self.entries[:-self.keep_recent]
        summary = f"Completed {len(to_compress)} steps: " + ", ".join(
            e.step_description for e in to_compress
        )
        self.summaries.append(summary)

        # Remove compressed entries
        freed = sum(e.token_estimate for e in to_compress)
        self.entries = self.entries[-self.keep_recent:]
        self._total_tokens -= freed

    def clear(self):
        """Reset memory."""
        self.entries = []
        self.summaries = []
        self._total_tokens = 0
