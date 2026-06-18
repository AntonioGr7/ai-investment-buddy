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
