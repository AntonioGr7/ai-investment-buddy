"""LLM-facing web-research tools: schemas + an executor bound to a WebResearch.

The counterpart to ``mem_tools`` (which looks inward at our own history) — these
look outward at the live web. Read-only, and currently offered only in the
post-run feedback dialogue, where the investor may raise something that happened
after the daily data pull.
"""

from __future__ import annotations

from ..config import SETTINGS
from ..data.web import WebResearch

WEB_TOOL_SPECS = [
    {
        "name": "web_search",
        "description": (
            "Search the live web and get titles, URLs and snippets. Use this whenever "
            "the investor references something that happened after your data snapshot "
            "(an earnings print, guidance, a price move, a news event) or when a claim "
            "hinges on a number you do not actually have. Prefer specific queries "
            "(e.g. 'META Q2 2026 earnings revenue capex guidance')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "max_results": {
                    "type": "integer",
                    "description": "How many results to return (1-10, default 6).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_url",
        "description": (
            "Fetch a web page and return its readable text (truncated). Use it to read "
            "a source you found via web_search — snippets alone are not enough to "
            "underwrite a valuation. Prefer primary sources (company IR/press release, "
            "SEC filing) over aggregators."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Absolute http(s) URL."},
                "max_chars": {
                    "type": "integer",
                    "description": "Character cap on the extracted text (default 6000).",
                },
            },
            "required": ["url"],
        },
    },
]


def web_tools_enabled() -> bool:
    return bool(SETTINGS.web_search_enabled)


def make_web_executor(research: WebResearch | None = None, on_call=None):
    """Return executor(name, args) -> str dispatching to WebResearch.

    ``on_call(name, args)`` is invoked for observability before each lookup. Tool
    errors come back as text: a dead link should cost the agent one turn, not the
    whole conversation."""
    research = research or WebResearch()

    def execute(name: str, args: dict) -> str:
        if on_call:
            on_call(name, args)
        try:
            if name == "web_search":
                return research.search(
                    str(args.get("query", "")), args.get("max_results")
                )
            if name == "fetch_url":
                return research.fetch(str(args.get("url", "")), args.get("max_chars"))
            return f"Unknown tool: {name}"
        except Exception as e:  # never let a tool error kill the loop
            return f"Tool {name} error: {e}"

    return execute
