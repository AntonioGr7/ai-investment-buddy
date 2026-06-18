"""MemoryToolkit — the agent's read-only window into its own accumulated history.

As the experiment runs, `data/` fills with journal entries, a trade ledger, NAV
history and theses. Rather than blindly pushing a fixed recent slice into every
prompt, we expose these as navigable tools (grep-like search, read-by-date,
per-ticker dossiers) so the agent can *pull* exactly the history that's relevant
to today's decision. Everything here is read-only and sandboxed to DATA_DIR.
"""

from __future__ import annotations

import json
import re

from ..config import DATA_DIR, JOURNAL_DIR
from . import store

_NARRATIVE_FILE = JOURNAL_DIR / "narrative.md"
_TRADES_FILE = DATA_DIR / "trades.jsonl"
_THESES_FILE = JOURNAL_DIR / "theses.json"

_MAX_HITS = 40


class MemoryToolkit:
    """Bound to the on-disk memory; methods return plain strings for the LLM."""

    # --- Navigation -------------------------------------------------------
    def list_journal_days(self) -> str:
        days = sorted(p.stem for p in JOURNAL_DIR.glob("*.md") if p.stem != "narrative")
        if not days:
            return "No journal entries yet."
        return f"{len(days)} journal day(s): " + ", ".join(days)

    def read_journal(self, date: str) -> str:
        path = JOURNAL_DIR / f"{date}.md"
        if not path.exists():
            return (
                f"No journal entry for {date}. "
                f"Available: {self.list_journal_days()}"
            )
        return path.read_text()

    def search_memory(self, pattern: str, scope: str = "all") -> str:
        """Regex search across journal entries and the trade ledger (grep-like)."""
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"Invalid regex: {e}"

        hits: list[str] = []
        if scope in ("all", "journal"):
            for path in sorted(JOURNAL_DIR.glob("*.md")):
                if path.stem == "narrative":
                    continue
                for i, line in enumerate(path.read_text().splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{path.name}:{i}: {line.strip()}")
                        if len(hits) >= _MAX_HITS:
                            break
                if len(hits) >= _MAX_HITS:
                    break
        if scope in ("all", "trades") and _TRADES_FILE.exists() and len(hits) < _MAX_HITS:
            for i, line in enumerate(_TRADES_FILE.read_text().splitlines(), 1):
                if rx.search(line):
                    hits.append(f"trades.jsonl:{i}: {line.strip()}")
                    if len(hits) >= _MAX_HITS:
                        break

        if not hits:
            return f"No matches for /{pattern}/ in {scope}."
        capped = " (capped)" if len(hits) >= _MAX_HITS else ""
        return f"{len(hits)} match(es) for /{pattern}/{capped}:\n" + "\n".join(hits)

    def ticker_dossier(self, ticker: str) -> str:
        """Everything we know about one ticker: position, trades, thesis, mentions."""
        ticker = ticker.upper().strip()
        out: list[str] = [f"=== DOSSIER: {ticker} ==="]

        # Current position.
        try:
            pf = store.load_portfolio()
            pos = pf.positions.get(ticker)
            if pos:
                out.append(
                    f"POSITION: {pos.shares:.4f} shares @ avg cost ${pos.avg_cost:.2f}"
                )
            else:
                out.append("POSITION: none currently held.")
        except Exception:
            out.append("POSITION: (portfolio unavailable)")

        # Thesis.
        if _THESES_FILE.exists():
            theses = json.loads(_THESES_FILE.read_text())
            if ticker in theses:
                out.append("THESIS: " + json.dumps(theses[ticker]))

        # Trades.
        trades = [t for t in store.load_trades() if t.ticker == ticker]
        if trades:
            out.append(f"TRADES ({len(trades)}):")
            for t in trades:
                out.append(
                    f"  {t.timestamp.date()} {t.action.value} {t.shares:.4f} "
                    f"@ ${t.price:.2f} (${t.value:,.0f})"
                )
        else:
            out.append("TRADES: none.")

        # Journal mentions.
        mentions = self.search_memory(rf"\b{re.escape(ticker)}\b", scope="journal")
        out.append("JOURNAL MENTIONS:")
        out.append(mentions)
        return "\n".join(out)

    # --- Rolling narrative (long-horizon consolidated memory) -------------
    def read_narrative(self) -> str:
        if _NARRATIVE_FILE.exists():
            return _NARRATIVE_FILE.read_text()
        return ""

    def write_narrative(self, text: str) -> None:
        JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
        _NARRATIVE_FILE.write_text(text.strip() + "\n")
