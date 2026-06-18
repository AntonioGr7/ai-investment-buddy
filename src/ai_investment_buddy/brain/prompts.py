"""Prompts and tool schemas for the 3-stage brain.

Stage 1 Strategist  — read the macro/news regime, pick finalists to deep-dive.
Stage 2 Analyst      — per name, a disciplined fair-value assessment.
Stage 3 PM           — allocate, may only buy names with acceptable valuation.
"""

from __future__ import annotations

import json
from datetime import date

from ..config import SETTINGS
from ..models import MacroSnapshot, TickerData, ValuationAssessment


# --- Shared formatters -------------------------------------------------------
def _fmt_macro(macro: MacroSnapshot) -> str:
    lines = ["MACRO SNAPSHOT (levels & moves):"]
    for k, v in macro.indicators.items():
        lines.append(f"  {k}: {v}")
    if macro.notes:
        lines.append("  notes: " + "; ".join(macro.notes))
    return "\n".join(lines)


def fmt_news(news: list[dict]) -> str:
    if not news:
        return "MARKET NEWS & MACRO EVENTS:\n  (no headlines retrieved)"
    lines = ["MARKET NEWS & MACRO EVENTS (read before deciding):"]
    last_cat = None
    for it in news:
        if it["category"] != last_cat:
            lines.append(f"  [{it['category']}]")
            last_cat = it["category"]
        date_str = f" ({it['published']})" if it.get("published") else ""
        lines.append(f"    - {it['source']}{date_str}: {it['title']}")
        if it.get("summary"):
            lines.append(f"        {it['summary']}")
    return "\n".join(lines)


def fmt_ticker(td: TickerData) -> str:
    fields = []

    def add(label, val):
        if val is not None and val != "":
            fields.append(f"{label}={val}")

    add("price", f"${td.price:.2f}" if td.price else None)
    add("1d", f"{td.change_pct:+.1f}%" if td.change_pct is not None else None)
    add("1m", f"{td.ret_1m:+.0f}%" if td.ret_1m is not None else None)
    add("3m", f"{td.ret_3m:+.0f}%" if td.ret_3m is not None else None)
    add("6m", f"{td.ret_6m:+.0f}%" if td.ret_6m is not None else None)
    add(">200dma", td.above_200dma)
    add("volx", td.vol_ratio)
    add("mktcap", f"${td.market_cap/1e9:.0f}B" if td.market_cap else None)
    add("PE", f"{td.pe:.0f}" if td.pe else None)
    add("fwdPE", f"{td.forward_pe:.0f}" if td.forward_pe else None)
    add("PEG", f"{td.peg:.1f}" if td.peg else None)
    add("PS", f"{td.ps:.1f}" if td.ps else None)
    add("margin", f"{td.profit_margin*100:.0f}%" if td.profit_margin else None)
    add("rev_g", f"{td.revenue_growth*100:.0f}%" if td.revenue_growth else None)
    add("eps_g", f"{td.earnings_growth*100:.0f}%" if td.earnings_growth else None)
    add("tgt", f"${td.target_mean_price:.0f}" if td.target_mean_price else None)
    add("rec", td.recommendation)

    header = f"{td.ticker} ({td.name or '?'}; {td.sector or '?'})"
    line = header + "\n    " + ", ".join(fields)
    if td.headlines:
        line += "\n    news: " + " | ".join(td.headlines[:4])
    return line


# =============================================================================
# Stage 1 — Strategist
# =============================================================================
STRATEGIST_SYSTEM = f"""You are the Chief Strategist of AI Investment Buddy, a paper portfolio \
trying to beat the S&P 500 and Nasdaq 100 over time.

Your job today is TOP-DOWN and SELECTIVE — you do NOT place trades. You:
1. Read the market news and macro events and state the current regime crisply (rates/Fed stance, \
growth, inflation, risk appetite, key catalysts and risks).
2. From the screened candidate list, choose a focused set of FINALISTS (up to {SETTINGS.shortlist_size}) \
that genuinely deserve a deep fundamental valuation today — names where the regime + setup make a \
careful look worthwhile. Do not just pick the biggest movers; pick what fits your thesis.
3. ALWAYS include every current holding in finalists (they must be re-evaluated for hold/trim/sell).

You have MEMORY TOOLS to consult your own history before deciding: search_memory (grep past \
journals/trades), read_journal (a past day), list_journal_days, and ticker_dossier (a name's full \
record). Use them when useful — e.g. recall how you positioned in a similar regime, or why you \
hold what you hold. Don't over-research; a few targeted lookups, then decide.

Be disciplined: a strong regime view does not justify chasing expensive names — that is the \
analyst's test next. When done, call submit_strategy exactly once."""

STRATEGIST_TOOL = {
    "name": "submit_strategy",
    "description": "Submit the regime read and the finalists to deep-dive.",
    "input_schema": {
        "type": "object",
        "properties": {
            "regime": {
                "type": "string",
                "description": "Short label for the market regime, e.g. 'risk-on, disinflationary'.",
            },
            "market_thesis": {
                "type": "string",
                "description": "Top-down read: rates/Fed, growth, inflation, risk appetite, catalysts, risks.",
            },
            "finalists": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tickers to deep-dive today (holdings always included).",
            },
            "reasoning": {
                "type": "string",
                "description": "Why these finalists, given the regime.",
            },
        },
        "required": ["regime", "market_thesis", "finalists", "reasoning"],
    },
}


def _fmt_narrative(narrative: str) -> str:
    if not narrative.strip():
        return ""
    return (
        "\n--- PORTFOLIO NARRATIVE (your consolidated long-horizon memory) ---\n"
        + narrative.strip()
    )


def build_strategist_message(
    as_of: date,
    macro: MacroSnapshot,
    market_news: list[dict],
    shortlist: list[TickerData],
    holdings: list[str],
    performance: str,
    narrative: str = "",
) -> str:
    parts = [f"=== STRATEGY FOR {as_of.isoformat()} ==="]
    parts.append(_fmt_narrative(narrative))
    parts.append("\n--- " + fmt_news(market_news))
    parts.append("\n--- " + _fmt_macro(macro))
    parts.append("\n--- PERFORMANCE VS BENCHMARKS ---")
    parts.append(performance)
    parts.append(f"\n--- CURRENT HOLDINGS (always finalists): {holdings or 'none'}")
    parts.append(f"\n--- SCREENED CANDIDATES ({len(shortlist)}) ---")
    for td in shortlist:
        parts.append(fmt_ticker(td))
    parts.append(
        "\nState the regime and choose finalists. Call submit_strategy exactly once."
    )
    return "\n".join(parts)


# =============================================================================
# Stage 2 — Fundamental Analyst
# =============================================================================
ANALYST_SYSTEM = """You are a rigorous Fundamental Analyst at AI Investment Buddy. For the ONE \
company given, deliver a disciplined valuation — your job is to answer 'what is it worth vs what \
it costs', not to chase momentum.

Method:
- Estimate a fair value per share. Anchor it: use earnings power and a justified multiple, growth, \
margins, sector comparables, FCF, and the analyst target as a sanity check (not gospel). Be \
explicit and conservative; when data is thin, widen your uncertainty and lower confidence.
- Compare fair value to the current price → upside/downside and a verdict \
(UNDERVALUED / FAIRLY_VALUED / OVERVALUED).
- Judge business quality (1-5) and whether there is a genuine MARGIN OF SAFETY (price meaningfully \
below fair value, not just a good story).
- Weigh the macro regime: rate sensitivity, cyclicality, exposure to current catalysts/risks.
- Give a recommendation (BUY/ADD/HOLD/WATCH/TRIM/SELL/AVOID) and a suggested max portfolio weight.

A high-flying, expensive, beloved stock should usually be FAIRLY_VALUED/OVERVALUED with no margin \
of safety unless you can defend the price with numbers. Do not rubber-stamp momentum. Call \
submit_assessment exactly once."""

ANALYST_TOOL = {
    "name": "submit_assessment",
    "description": "Submit the fair-value assessment for this one company.",
    "input_schema": {
        "type": "object",
        "properties": {
            "fair_value": {"type": "number", "description": "Estimated intrinsic value per share (USD)."},
            "upside_pct": {"type": "number", "description": "(fair_value/price - 1) * 100."},
            "valuation_verdict": {
                "type": "string",
                "enum": ["UNDERVALUED", "FAIRLY_VALUED", "OVERVALUED"],
            },
            "quality_score": {"type": "integer", "minimum": 1, "maximum": 5},
            "margin_of_safety": {"type": "boolean"},
            "bull_case": {"type": "string"},
            "bear_case": {"type": "string"},
            "key_risks": {"type": "string"},
            "recommendation": {
                "type": "string",
                "enum": ["BUY", "ADD", "HOLD", "WATCH", "TRIM", "SELL", "AVOID"],
            },
            "suggested_max_weight": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Suggested cap on this name as a fraction of NAV.",
            },
            "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        "required": [
            "fair_value", "valuation_verdict", "quality_score", "margin_of_safety",
            "bull_case", "bear_case", "key_risks", "recommendation",
            "suggested_max_weight", "confidence",
        ],
    },
}


def build_analyst_message(
    td: TickerData,
    regime: str,
    market_thesis: str,
    current_position: dict | None,
    dossier: str = "",
) -> str:
    parts = [f"=== VALUATION: {td.ticker} ({td.name or '?'}) ==="]
    parts.append(f"\nMARKET REGIME: {regime}")
    parts.append(f"STRATEGIST THESIS: {market_thesis}")
    if current_position:
        parts.append(
            f"\nWE ALREADY HOLD THIS: {json.dumps(current_position)} "
            "(assess hold/add/trim/sell)."
        )
    if dossier and "TRADES: none" not in dossier:
        parts.append("\n--- OUR HISTORY WITH THIS NAME ---")
        parts.append(dossier)
    parts.append("\n--- COMPANY DATA ---")
    parts.append(fmt_ticker(td))
    parts.append(
        "\nEstimate fair value and assess. Call submit_assessment exactly once."
    )
    return "\n".join(parts)


# =============================================================================
# Stage 3 — Portfolio Manager
# =============================================================================
PM_SYSTEM = f"""You are the Portfolio Manager of AI Investment Buddy. You make the final \
allocation for a paper portfolio trying to beat the S&P 500 and Nasdaq 100 over time.

You are given: the strategist's regime/thesis, the analysts' fair-value ASSESSMENTS for each \
finalist, your current portfolio, performance, and your memory (journal + theses).

Rules of engagement:
- You may BUY, ADD, HOLD, TRIM, SELL, or DO NOTHING. Doing nothing is valid; don't trade to be busy.
- VALUATION DISCIPLINE: only BUY/ADD a name with a supportive assessment — generally a BUY/ADD \
recommendation with a genuine margin of safety or clearly acceptable valuation. Do NOT add to \
OVERVALUED names with no margin of safety, however strong the story. Trim/exit names the analyst \
flags TRIM/SELL/AVOID or whose thesis has broken.
- Respect each analyst's suggested_max_weight as a soft ceiling, and the hard guardrails below.
- Size by conviction × margin of safety, and by fit with the regime and the existing book. It is \
fine — often wise — to scale in gradually and hold cash when little is compelling.

Hard guardrails (execution enforces them too):
- No single position above {SETTINGS.max_position_weight:.0%} of NAV. No leverage, no shorting. \
Total invested ≤ 100%; remainder is cash.

You also have MEMORY TOOLS (search_memory, read_journal, list_journal_days, ticker_dossier) to \
consult your own past decisions before committing — use them when it helps (e.g. check why you \
hold a name, or how a past trade played out). A few targeted lookups, then decide.

For each name you act on, emit an order with TARGET WEIGHT = the fraction of NAV that position \
should be AFTER trading. To exit fully, SELL with target_weight 0. Holdings you don't mention are \
left untouched. Put your reasoning in the fields. When done, call submit_decision exactly once."""

# PM uses the same decision schema as before.
PM_TOOL = {
    "name": "submit_decision",
    "description": "Submit the final portfolio decision for today.",
    "input_schema": {
        "type": "object",
        "properties": {
            "market_thesis": {"type": "string", "description": "Final synthesis of the day's positioning."},
            "orders": {
                "type": "array",
                "description": "Trades to make. Empty array means do nothing.",
                "items": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
                        "target_weight": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "rationale": {"type": "string"},
                        "conviction": {"type": "integer", "minimum": 1, "maximum": 5},
                    },
                    "required": ["ticker", "action", "target_weight", "rationale"],
                },
            },
            "target_cash_weight": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "notes": {"type": "string", "description": "Notes to your future self."},
        },
        "required": ["market_thesis", "orders", "target_cash_weight", "notes"],
    },
}


def _fmt_assessment(a: ValuationAssessment) -> str:
    return (
        f"{a.one_line()}\n"
        f"    suggested_max_weight={a.suggested_max_weight:.0%}\n"
        f"    bull: {a.bull_case}\n"
        f"    bear: {a.bear_case}\n"
        f"    risks: {a.key_risks}"
    )


def build_pm_message(
    as_of: date,
    regime: str,
    market_thesis: str,
    assessments: list[ValuationAssessment],
    portfolio_state: dict,
    performance: str,
    recent_journal: list[str],
    theses: dict[str, dict],
    narrative: str = "",
) -> str:
    parts = [f"=== ALLOCATION FOR {as_of.isoformat()} ==="]
    parts.append(f"\nMARKET REGIME: {regime}")
    parts.append(f"STRATEGIST THESIS: {market_thesis}")
    parts.append(_fmt_narrative(narrative))

    parts.append("\n--- YOUR PORTFOLIO ---")
    parts.append(json.dumps(portfolio_state, indent=2))

    parts.append("\n--- PERFORMANCE VS BENCHMARKS ---")
    parts.append(performance)

    parts.append(f"\n--- ANALYST ASSESSMENTS ({len(assessments)}) ---")
    for a in assessments:
        parts.append(_fmt_assessment(a))

    if theses:
        parts.append("\n--- YOUR STANDING THESES ---")
        parts.append(json.dumps(theses, indent=2))
    if recent_journal:
        parts.append("\n--- YOUR RECENT JOURNAL (oldest first) ---")
        parts.append("\n\n".join(recent_journal))

    parts.append(
        "\nMake the final allocation with valuation discipline. "
        "Call submit_decision exactly once."
    )
    return "\n".join(parts)
