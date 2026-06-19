"""The user's favorite stocks — a hand-curated watchlist.

Unlike the quant screener (which surfaces a different shortlist each day from
technicals), the watchlist is the user's explicit list of names they always
want the AI to look at. Every watchlist ticker is force-fed through the *entire*
daily process: always enriched with fundamentals + news, always made a
strategist finalist, and always valued by the analyst — regardless of whether
it would have made the quant cut.

The watchlist is part of the bot's state: it lives next to the portfolio at
``data/watchlist.jsonl`` and travels in export/import snapshots. It is optional
— no file (or an empty one) simply means "no favorites".

Storage format is JSONL, one ticker per line as ``{"ticker": "AAPL"}``. For
hand-editing we also accept a bare JSON string (``"AAPL"``); blank lines and
``#`` comments are ignored. Tickers are normalised (uppercased, ``.`` → ``-``)
so yfinance accepts them.
"""

from __future__ import annotations

import json

from .config import DATA_DIR, ensure_dirs

_WATCHLIST_FILE = DATA_DIR / "watchlist.jsonl"


def normalize(ticker: str) -> str:
    # yfinance uses '-' where some sources use '.' (e.g. BRK.B -> BRK-B).
    return str(ticker).strip().upper().replace(".", "-")


def _ticker_from_line(line: str) -> str | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        obj = line  # tolerate a bare unquoted token
    if isinstance(obj, str):
        raw = obj
    elif isinstance(obj, dict):
        raw = obj.get("ticker") or obj.get("symbol") or ""
    else:
        return None
    return normalize(raw) or None


def load_watchlist() -> list[str]:
    """Return the watchlist tickers (de-duplicated, order preserved).

    Missing file → empty list (the watchlist is optional)."""
    if not _WATCHLIST_FILE.exists():
        return []
    seen: list[str] = []
    for line in _WATCHLIST_FILE.read_text().splitlines():
        t = _ticker_from_line(line)
        if t and t not in seen:
            seen.append(t)
    return seen


def save_watchlist(tickers: list[str]) -> list[str]:
    """Overwrite the watchlist with ``tickers`` (normalised, de-duplicated)."""
    ensure_dirs()
    clean: list[str] = []
    for t in tickers:
        n = normalize(t)
        if n and n not in clean:
            clean.append(n)
    _WATCHLIST_FILE.write_text(
        "".join(json.dumps({"ticker": t}) + "\n" for t in clean)
    )
    return clean


def add(tickers: list[str]) -> list[str]:
    """Add tickers to the watchlist. Returns the ones newly added."""
    current = load_watchlist()
    added = [t for t in (normalize(x) for x in tickers) if t and t not in current]
    if added:
        save_watchlist(current + added)
    return added


def remove(tickers: list[str]) -> list[str]:
    """Remove tickers from the watchlist. Returns the ones actually removed."""
    current = load_watchlist()
    drop = {normalize(x) for x in tickers}
    removed = [t for t in current if t in drop]
    if removed:
        save_watchlist([t for t in current if t not in drop])
    return removed
