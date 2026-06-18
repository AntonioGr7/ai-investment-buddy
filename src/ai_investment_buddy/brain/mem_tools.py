"""LLM-facing memory tools: schemas + an executor bound to a MemoryToolkit.

These are the *helper* tools an agentic node may call any number of times before
emitting its final structured output. They are read-only.
"""

from __future__ import annotations

from ..memory import MemoryToolkit

MEMORY_TOOL_SPECS = [
    {
        "name": "search_memory",
        "description": (
            "Regex/grep search across past journal entries and the trade ledger. "
            "Use to recall past reasoning, similar regimes, or prior decisions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Case-insensitive regex."},
                "scope": {
                    "type": "string",
                    "enum": ["all", "journal", "trades"],
                    "description": "Where to search (default all).",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "read_journal",
        "description": "Read the full journal entry for a given date (YYYY-MM-DD).",
        "input_schema": {
            "type": "object",
            "properties": {"date": {"type": "string", "description": "YYYY-MM-DD"}},
            "required": ["date"],
        },
    },
    {
        "name": "list_journal_days",
        "description": "List the dates for which journal entries exist.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "ticker_dossier",
        "description": (
            "Assemble everything known about one ticker: current position, all its "
            "trades, its standing thesis, and every journal mention."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
    },
]


def make_memory_executor(toolkit: MemoryToolkit, on_call=None):
    """Return executor(name, args) -> str dispatching to the toolkit.

    ``on_call(name, args)`` is invoked for observability before each lookup."""

    def execute(name: str, args: dict) -> str:
        if on_call:
            on_call(name, args)
        try:
            if name == "search_memory":
                return toolkit.search_memory(
                    args.get("pattern", ""), args.get("scope", "all")
                )
            if name == "read_journal":
                return toolkit.read_journal(args.get("date", ""))
            if name == "list_journal_days":
                return toolkit.list_journal_days()
            if name == "ticker_dossier":
                return toolkit.ticker_dossier(args.get("ticker", ""))
            return f"Unknown tool: {name}"
        except Exception as e:  # never let a tool error kill the loop
            return f"Tool {name} error: {e}"

    return execute
