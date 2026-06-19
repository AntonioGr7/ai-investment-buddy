"""The reasoning journal — the AI's evolving memory of *why* it did things.

Two parts:
  - Daily narrative entries (data/journal/YYYY-MM-DD.md): the market thesis and
    decisions for that day, so the AI can re-read its recent reasoning.
  - Per-ticker theses (data/journal/theses.json): a living view keyed by ticker
    that the AI maintains and revises over time.
"""

from __future__ import annotations

import json
from datetime import date

from ..config import JOURNAL_DIR, ensure_dirs
from ..models import Decision, StrategistView, ValuationAssessment

_THESES_FILE = JOURNAL_DIR / "theses.json"


class Journal:
    def __init__(self) -> None:
        ensure_dirs()

    # --- Daily narrative ---
    def _entry_path(self, as_of: date):
        return JOURNAL_DIR / f"{as_of.isoformat()}.md"

    def record_day(
        self,
        decision: Decision,
        strategy: StrategistView | None = None,
        assessments: list[ValuationAssessment] | None = None,
    ) -> None:
        lines = [f"# {decision.as_of.isoformat()}", ""]

        if strategy:
            lines += [
                f"**Regime:** {strategy.regime}",
                "",
                "## Strategist read",
                strategy.market_thesis.strip() or "_(none)_",
                "",
            ]
            if getattr(strategy, "sector_read", ""):
                lines += ["## Sector read (overreaction vs value trap)",
                          strategy.sector_read.strip(), ""]

        if assessments:
            lines += ["## Analyst valuations", ""]
            for a in assessments:
                lines.append(f"- {a.one_line()}")
            lines.append("")

        lines += [
            "## Market thesis (PM)",
            decision.market_thesis.strip() or "_(none)_",
            "",
            "## Orders",
        ]
        if decision.orders:
            for o in decision.orders:
                lines.append(
                    f"- **{o.action.value} {o.ticker}** → target {o.target_weight:.0%} "
                    f"(conviction {o.conviction}/5): {o.rationale.strip()}"
                )
        else:
            lines.append("- _No trades. Held existing allocation._")
        lines += [
            "",
            f"_Target cash weight: {decision.target_cash_weight:.0%}_",
            "",
            "## Notes to future self",
            decision.notes.strip() or "_(none)_",
            "",
        ]
        self._entry_path(decision.as_of).write_text("\n".join(lines))

    def recent_entries(self, n: int = 5) -> list[str]:
        """Return the text of the most recent ``n`` daily entries, oldest first."""
        files = sorted(JOURNAL_DIR.glob("*.md"))
        return [f.read_text() for f in files[-n:]]

    # --- Market-wide investor notes (from the feedback dialogue) ---
    def _investor_notes_path(self):
        return JOURNAL_DIR / "investor_notes.md"

    def append_investor_note(self, text: str, as_of: date) -> None:
        """Append a durable market-wide note from the investor dialogue. Always
        injected into future strategist + PM prompts."""
        text = text.strip()
        if not text:
            return
        JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
        p = self._investor_notes_path()
        prior = p.read_text() if p.exists() else "# Investor notes\n"
        p.write_text(prior.rstrip() + f"\n\n- [{as_of.isoformat()}] {text}\n")

    def read_investor_notes(self) -> str:
        p = self._investor_notes_path()
        return p.read_text() if p.exists() else ""

    def latest_strategy(self) -> tuple[str, str]:
        """Best-effort (regime, strategist thesis) from the most recent entry, so
        an on-demand valuation can borrow the last known market context. Empty
        strings if there is no journal yet."""
        files = sorted(JOURNAL_DIR.glob("*.md"))
        if not files:
            return "", ""
        text = files[-1].read_text()
        regime = ""
        for line in text.splitlines():
            if line.startswith("**Regime:**"):
                regime = line.split("**Regime:**", 1)[1].strip()
                break
        thesis = ""
        if "## Strategist read" in text:
            after = text.split("## Strategist read", 1)[1]
            thesis = after.split("\n## ", 1)[0].strip()
        return regime, thesis

    # --- Per-ticker theses ---
    def load_theses(self) -> dict[str, dict]:
        if not _THESES_FILE.exists():
            return {}
        return json.loads(_THESES_FILE.read_text())

    def update_theses(self, decision: Decision) -> None:
        """Fold the day's order rationales into the living per-ticker theses."""
        theses = self.load_theses()
        for o in decision.orders:
            theses[o.ticker] = {
                "thesis": o.rationale.strip(),
                "conviction": o.conviction,
                "last_action": o.action.value,
                "updated": decision.as_of.isoformat(),
            }
        _THESES_FILE.write_text(json.dumps(theses, indent=2, sort_keys=True))
