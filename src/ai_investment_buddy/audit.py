"""Per-run audit trail: what the agent did, and what it read.

Two diagnostic artifacts, written under ``data/`` so they are gitignored and
excluded from state snapshots — you can inspect them to understand a decision,
then delete them whenever you like:

  - ``data/logs/<date>.log``      — the agent's full step-by-step run log
  - ``data/news/<date>/``         — the raw macro + company news it pulled

Both are best-effort: a failure to write never interrupts a run.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from .config import LOGS_DIR, NEWS_DIR, SETTINGS
from .models import (
    Decision,
    MacroSnapshot,
    StrategistView,
    TickerData,
    ValuationAssessment,
)


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def write_log(as_of: date, log: list[str]) -> None:
    """Write the run's step log to ``data/logs/<date>.log`` (overwrites per day)."""
    if not SETTINGS.write_audit or not log:
        return
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        header = f"=== AI Investment Buddy run log — {as_of.isoformat()} (written {_stamp()}) ===\n\n"
        (LOGS_DIR / f"{as_of.isoformat()}.log").write_text(header + "\n".join(log) + "\n")
    except Exception:
        pass


def _fmt_market_news(market_news: list[dict]) -> str:
    lines = [f"# Macro & market news — {len(market_news)} headlines\n"]
    last_cat = None
    for it in market_news:
        cat = it.get("category", "")
        if cat != last_cat:
            lines.append(f"\n## {cat}")
            last_cat = cat
        pub = f" ({it['published']})" if it.get("published") else ""
        lines.append(f"\n**{it.get('source', '?')}{pub}** — {it.get('title', '')}")
        if it.get("summary"):
            lines.append(f"\n> {it['summary']}")
    return "\n".join(lines) + "\n"


def _fmt_company_news(shortlist: list[TickerData]) -> str:
    lines = ["# Company news (shortlist the agent deep-dived)\n"]
    for td in shortlist:
        head = f"\n## {td.ticker} — {td.name or '?'} ({td.sector or '?'})"
        lines.append(head)
        if td.headlines:
            for h in td.headlines:
                lines.append(f"- {h}")
        else:
            lines.append("- _(no headlines retrieved)_")
    return "\n".join(lines) + "\n"


def _fmt_macro(macro: MacroSnapshot) -> str:
    lines = [f"# Macro snapshot — {macro.as_of.isoformat()}\n"]
    for k, v in macro.indicators.items():
        lines.append(f"- **{k}**: {v}")
    if macro.notes:
        lines.append("\n## Notes")
        for n in macro.notes:
            lines.append(f"- {n}")
    return "\n".join(lines) + "\n"


def _fmt_assessment_full(a: ValuationAssessment) -> str:
    fv = f"${a.fair_value:.2f}" if a.fair_value is not None else "?"
    px = f"${a.current_price:.2f}" if a.current_price is not None else "?"
    up = f"{a.upside_pct:+.0f}%" if a.upside_pct is not None else "?"
    cached = " _(reused from cache)_" if getattr(a, "from_cache", False) else ""
    down = f"{a.downside_pct:+.0f}%" if a.downside_pct is not None else "?"
    rr = f"{a.risk_reward}" if a.risk_reward is not None else "?"
    lines = [
        f"### {a.ticker} — {a.archetype or '?'} — **{a.recommendation}** "
        f"({a.market_view}){cached}",
        f"- Fair value {fv} vs price {px} → upside {up} · {a.valuation_verdict} · "
        f"quality {a.quality_score}/5 · MoS {'yes' if a.margin_of_safety else 'no'} · "
        f"confidence {a.confidence}/5",
        f"- **Risk/reward {rr}** · downside {down} · structural risk "
        f"**{a.structural_risk or '?'}**",
    ]
    if a.why_market_disagrees:
        lines.append(f"- **Why the market disagrees:** {a.why_market_disagrees}")
    if a.rerating_catalyst:
        lines.append(f"- **Catalyst / horizon:** {a.rerating_catalyst}")
    if a.valuation_method:
        lines.append(f"- **Method:** {a.valuation_method}")
    if a.market_implied:
        lines.append(f"- **Market implies:** {a.market_implied}")
    if a.mispricing_thesis:
        lines.append(f"- **Mispricing thesis:** {a.mispricing_thesis}")
    if a.news_assessment or a.news_sentiment:
        lines.append(f"- **News ({a.news_sentiment or 'n/a'}):** {a.news_assessment}")
    if a.bull_case:
        lines.append(f"- **Bull:** {a.bull_case}")
    if a.bear_case:
        lines.append(f"- **Bear:** {a.bear_case}")
    if a.key_risks:
        lines.append(f"- **Risks:** {a.key_risks}")
    return "\n".join(lines)


def write_reasoning(
    as_of: date,
    strategy: StrategistView | None,
    assessments: list[ValuationAssessment],
    decision: Decision,
) -> None:
    """Write the agent's FULL reasoning — strategist thesis, every per-name
    valuation, and the PM's rationale — to ``data/logs/<date>-reasoning.md``.

    This is the 'why', captured on every run (dry included), so a decision is
    auditable beyond the terse step log."""
    if not SETTINGS.write_audit:
        return
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        out = [f"# Decision reasoning — {as_of.isoformat()} (written {_stamp()})\n"]

        if strategy:
            out += [
                "## 1. Strategist (top-down)",
                f"\n**Regime:** {strategy.regime}",
                f"\n**Market thesis:** {strategy.market_thesis}",
            ]
            if strategy.sector_read:
                out.append(f"\n**Sector read (overreaction vs value trap):** {strategy.sector_read}")
            if strategy.reasoning:
                out.append(f"\n**Why these finalists:** {strategy.reasoning}")
            if strategy.finalists:
                out.append(f"\n**Finalists:** {', '.join(strategy.finalists)}")

        out.append("\n## 2. Analyst (per-name valuations)\n")
        if assessments:
            for a in assessments:
                out.append(_fmt_assessment_full(a) + "\n")
        else:
            out.append("_(no assessments)_\n")

        out.append("## 3. Portfolio manager (allocation)\n")
        out.append(f"**Thesis:** {decision.market_thesis}\n")
        if decision.orders:
            out.append("**Orders:**")
            for o in decision.orders:
                out.append(
                    f"- **{o.action.value} {o.ticker}** → target {o.target_weight:.0%} "
                    f"(conviction {o.conviction}/5): {o.rationale}"
                )
        else:
            out.append("**Orders:** none — held.")
        out.append(f"\n**Target cash weight:** {decision.target_cash_weight:.0%}")
        if decision.notes:
            out.append(f"\n**Notes to future self:** {decision.notes}")

        (LOGS_DIR / f"{as_of.isoformat()}-reasoning.md").write_text("\n".join(out) + "\n")
    except Exception:
        pass


def write_news(
    as_of: date,
    macro: MacroSnapshot,
    market_news: list[dict],
    shortlist: list[TickerData],
    sector_scan: str = "",
) -> None:
    """Dump the raw inputs the agent read into ``data/news/<date>/`` as markdown."""
    if not SETTINGS.write_audit:
        return
    try:
        out = NEWS_DIR / as_of.isoformat()
        out.mkdir(parents=True, exist_ok=True)
        (out / "macro_market_news.md").write_text(_fmt_market_news(market_news))
        (out / "company_news.md").write_text(_fmt_company_news(shortlist))
        (out / "macro_snapshot.md").write_text(_fmt_macro(macro))
        if sector_scan:
            (out / "sector_performance.md").write_text(
                "# Sector performance (the agent's sector map)\n\n" + sector_scan + "\n"
            )
    except Exception:
        pass
