"""
WebSandbox — Safe web fetching and search for agents.
Rate-limited, timeout-enforced, content-truncated.
"""

from __future__ import annotations
import urllib.request
import urllib.parse
import json
import time


class WebSandbox:
    """
    Safe web fetch and search tool.
    Rate-limited to prevent abuse.
    """

    name = "web"
    description = "Fetch content from a URL or search the web."
    schema = {
        "action": "string — 'fetch' or 'search'",
        "url": "string (for fetch) — URL to fetch",
        "query": "string (for search) — Search query",
    }

    def __init__(
        self,
        timeout: int = 15,
        max_content: int = 8192,
        rate_limit: float = 1.0,  # Seconds between requests
    ):
        self.timeout = timeout
        self.max_content = max_content
        self.rate_limit = rate_limit
        self._last_request = 0.0

    def execute(self, action: str, url: str = None, query: str = None) -> str:
        """Fetch a URL or search the web."""
        self._rate_limit()

        if action == "fetch" and url:
            return self._fetch(url)
        elif action == "search" and query:
            return self._search(query)
        else:
            return "Error: provide action='fetch' with url, or action='search' with query"

    def _fetch(self, url: str) -> str:
        """Fetch and extract text from URL."""
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "SmolMind/0.1 (AI Agent Research)"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                content = resp.read().decode("utf-8", errors="ignore")

            # Strip HTML tags (basic)
            import re
            content = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL)
            content = re.sub(r"<style[^>]*>.*?</style>", "", content, flags=re.DOTALL)
            content = re.sub(r"<[^>]+>", " ", content)
            content = re.sub(r"\s+", " ", content).strip()

            if len(content) > self.max_content:
                content = content[:self.max_content] + "...[truncated]"

            return content

        except Exception as e:
            return f"Fetch error: {str(e)}"

    def _search(self, query: str) -> str:
        """Search using DuckDuckGo (no API key required)."""
        try:
            encoded = urllib.parse.quote(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded}"
            result = self._fetch(url)

            # Extract just the result snippets
            import re
            snippets = re.findall(r"result__snippet[^>]*>(.*?)</", result)
            if snippets:
                return "\n".join(snippets[:5])
            return result[:2000]

        except Exception as e:
            return f"Search error: {str(e)}"

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        now = time.time()
        wait = self.rate_limit - (now - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.time()
