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
    add("offHigh", f"{td.drawdown_pct:+.0f}%" if td.drawdown_pct is not None else None)
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
2. STUDY THE SECTOR SCAN before picking names. It shows median trailing returns and breadth for \
each sector, WORST-PERFORMING FIRST. For each beaten-down sector, form a view: is the market \
OVERREACTING (a whole group sold off on a narrative/fear — fertile ground for mispriced names) or \
is the de-rating DESERVED (broken fundamentals, a structural/secular threat — a value trap to \
avoid)? Your edge is buying what the market wrongly hates and avoiding what it rightly hates. The \
best opportunities are often NOT the leaderboard — a strong stock that just keeps rising is rarely \
mispriced; a quality name dragged down with its sector on fear often is. Recent examples of the \
shape to look for: a software/SaaS sell-off on AI-disruption fear, a bank sell-off on rate panic. \
3. Choose a focused set of FINALISTS (up to {SETTINGS.shortlist_size}) that genuinely deserve a \
deep fundamental valuation today. Deliberately include names from the sectors you judge \
oversold/overreacted — that is where the asymmetry is. Do NOT just pick the biggest movers or the \
strongest momentum; pick where price has diverged from likely value.
4. ALWAYS include every current holding in finalists (they must be re-evaluated for hold/trim/sell).

You have MEMORY TOOLS to consult your own history before deciding: search_memory (grep past \
journals/trades), read_journal (a past day), list_journal_days, and ticker_dossier (a name's full \
record). Use them when useful — e.g. recall how you positioned in a similar regime, or why you \
hold what you hold. Don't over-research; a few targeted lookups, then decide.

Be disciplined: a strong regime view does not justify chasing expensive names, and a cheap price \
does not by itself make a bargain — both get tested by the analyst next. State your sector read \
explicitly. When done, call submit_strategy exactly once."""

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
            "sector_read": {
                "type": "string",
                "description": "Your verdict on the beaten-down sectors: which look like "
                "OVERREACTIONS (opportunity) vs DESERVED de-ratings (value traps), and where "
                "momentum looks crowded. This is what justifies where you go hunting.",
            },
            "finalists": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tickers to deep-dive today (holdings always included). Include "
                "names from the sectors you judge oversold/overreacted.",
            },
            "reasoning": {
                "type": "string",
                "description": "Why these finalists, given the regime and your sector read.",
            },
        },
        "required": ["regime", "market_thesis", "sector_read", "finalists", "reasoning"],
    },
}


def _fmt_narrative(narrative: str) -> str:
    if not narrative.strip():
        return ""
    return (
        "\n--- PORTFOLIO NARRATIVE (your consolidated long-horizon memory) ---\n"
        + narrative.strip()
    )


def _fmt_investor_notes(notes: str) -> str:
    if not notes or not notes.strip():
        return ""
    return (
        "\n--- INVESTOR NOTES (the human you work for; weigh these seriously, but "
        "stay intellectually honest — agree only if the evidence does) ---\n"
        + notes.strip()
    )


def build_strategist_message(
    as_of: date,
    macro: MacroSnapshot,
    market_news: list[dict],
    shortlist: list[TickerData],
    holdings: list[str],
    watchlist: list[str] | None = None,
    performance: str = "",
    narrative: str = "",
    sector_scan: str = "",
    investor_notes: str = "",
) -> str:
    parts = [f"=== STRATEGY FOR {as_of.isoformat()} ==="]
    parts.append(_fmt_narrative(narrative))
    parts.append(_fmt_investor_notes(investor_notes))
    parts.append("\n--- " + fmt_news(market_news))
    parts.append("\n--- " + _fmt_macro(macro))
    if sector_scan:
        parts.append("\n--- " + sector_scan)
    parts.append("\n--- PERFORMANCE VS BENCHMARKS ---")
    parts.append(performance)
    parts.append(f"\n--- CURRENT HOLDINGS (always finalists): {holdings or 'none'}")
    if watchlist:
        parts.append(
            f"\n--- USER WATCHLIST (favorites; always finalists, always valued): "
            f"{', '.join(watchlist)}"
        )
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
it costs, and is the market right about it', not to chase momentum or reflexively buy a dip.

You have VALUATION TOOLS — use them, don't eyeball the number:
- dcf_two_stage: a proper discounted cash flow. Feed it YOUR estimated free cash flow, growth, \
terminal growth, and a sensible discount rate. Run more than once — a base, a bull and a bear case \
— to see the range, not a false-precision point.
- reverse_dcf: solves for the growth rate the CURRENT price implies. Run this on almost every name: \
comparing the implied growth to what the business can plausibly deliver is the cleanest test of \
whether the market is over-reacting (implied growth too pessimistic) or pricing perfection.
- exit_multiple: grow EPS/FFO/FCF-per-share and apply a justified terminal multiple — the right \
tool for financials (EPS×P/E), REITs (FFO×P/FFO), and mid-cycle cyclical earnings.
Inputs you don't have you must ESTIMATE with judgement and stated reasoning (expected growth, \
margins, a discount rate appropriate to the risk). Cross-check at least two methods where you can, \
and let the tool outputs — not a gut feel — anchor your fair_value and market_implied fields.

Method:
1. CLASSIFY THE BUSINESS first, because the right valuation method depends on what it is. Pick an \
archetype and value it the way that fits (and choose the matching tool):
   - HYPERGROWTH (e.g. SaaS/high-growth tech): Rule of 40 (growth + FCF margin), EV/Sales vs \
growth and gross margin, net revenue retention, path to and durability of FCF; sanity-check with a \
reverse-DCF on revenue. A high EV/S is only justified by durable high growth + strong margins.
   - COMPOUNDER (high-quality, steady grower): owner earnings / FCF yield, ROIC, EV/EBIT, a \
justified premium for quality and reinvestment runway.
   - VALUE / MATURE: P/E and EV/EBITDA vs history and peers, FCF yield, balance sheet, dividend \
durability.
   - CYCLICAL (industrials, materials, semis-equipment, autos): value off MID-CYCLE NORMALIZED \
earnings, never trough or peak; watch where we are in the cycle. Cheap-looking peak earnings is a \
trap; expensive-looking trough earnings can be opportunity.
   - FINANCIAL / BANK: P/TBV and ROTE/ROE, normalized earnings, credit/reserve quality. Use \
P/E and P/B — NOT EV/EBITDA or EV/Sales.
   - REIT: P/FFO (or AFFO), implied cap rate, NAV. Use FFO, not GAAP EPS.
   - TURNAROUND / DISTRESSED: scenario-weight (base/bull/bear), focus on balance-sheet survival \
and normalized earnings power if the fix works.
2. MARKET-IMPLIED / REVERSE VALUATION: state what TODAY'S PRICE is implying (growth rate, margins, \
multiple). Then judge whether that implied expectation is reasonable. This is the core question: \
is the market OVERREACTING (pricing in too much pessimism — e.g. a quality name dumped with its \
whole sector on a narrative), UNDERREACTING (complacent, ignoring deterioration), or FAIR? If \
mispriced, say specifically what the market is missing (your mispricing_thesis).
3. The sell-side analyst price target is a LAGGING cross-check, NOT an anchor. Targets are usually \
revised AFTER news, so right after a sell-off the target is stale and too high (and after a run-up, \
too low). If your independent value diverges from a freshly-set target, trust your work and note the \
divergence as a signal — do not average toward the target.
4. Estimate a fair value per share, compare to price → upside/downside and a verdict \
(UNDERVALUED / FAIRLY_VALUED / OVERVALUED). Be explicit and conservative; when data is thin, widen \
uncertainty and lower confidence.
5. Judge business quality (1-5) and whether there is a genuine MARGIN OF SAFETY (price meaningfully \
below a conservatively-estimated value — not just 'it fell a lot'). A falling price is necessary but \
NOT sufficient: distinguish a mispriced quality business from a deserved de-rating / value trap.
6. Weigh the macro regime: rate sensitivity, cyclicality, exposure to current catalysts/risks.
7. Give a recommendation (BUY/ADD/HOLD/WATCH/TRIM/SELL/AVOID) and a suggested max portfolio weight.

A high-flying, expensive, beloved stock should usually be FAIRLY_VALUED/OVERVALUED unless you can \
defend the price with numbers; a beaten-down name is only a BUY if your own valuation — not the \
size of the drop — says so. Call submit_assessment exactly once."""

ANALYST_TOOL = {
    "name": "submit_assessment",
    "description": "Submit the fair-value assessment for this one company.",
    "input_schema": {
        "type": "object",
        "properties": {
            "archetype": {
                "type": "string",
                "enum": ["HYPERGROWTH", "COMPOUNDER", "VALUE", "CYCLICAL", "FINANCIAL", "REIT", "TURNAROUND"],
                "description": "What kind of business this is — dictates the valuation method.",
            },
            "valuation_method": {
                "type": "string",
                "description": "The primary method you used and why it fits this archetype "
                "(e.g. 'Rule of 40 + EV/S vs growth; reverse-DCF on revenue').",
            },
            "fair_value": {"type": "number", "description": "Estimated intrinsic value per share (USD)."},
            "upside_pct": {"type": "number", "description": "(fair_value/price - 1) * 100."},
            "valuation_verdict": {
                "type": "string",
                "enum": ["UNDERVALUED", "FAIRLY_VALUED", "OVERVALUED"],
            },
            "quality_score": {"type": "integer", "minimum": 1, "maximum": 5},
            "margin_of_safety": {"type": "boolean"},
            "market_implied": {
                "type": "string",
                "description": "What today's price implies (growth/margins/multiple) — the reverse "
                "valuation — and whether that expectation is reasonable.",
            },
            "market_view": {
                "type": "string",
                "enum": ["OVERREACTING", "UNDERREACTING", "FAIR"],
                "description": "Your call on the market's pricing of this name right now.",
            },
            "mispricing_thesis": {
                "type": "string",
                "description": "If mispriced: what the market is missing and why it corrects. "
                "Empty if you judge it FAIR.",
            },
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
            "archetype", "valuation_method", "fair_value", "valuation_verdict",
            "quality_score", "margin_of_safety", "market_implied", "market_view",
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
    investor_notes: str = "",
) -> str:
    parts = [f"=== VALUATION: {td.ticker} ({td.name or '?'}) ==="]
    parts.append(f"\nMARKET REGIME: {regime}")
    parts.append(f"STRATEGIST THESIS: {market_thesis}")
    if current_position:
        parts.append(
            f"\nWE ALREADY HOLD THIS: {json.dumps(current_position)} "
            "(assess hold/add/trim/sell)."
        )
    if investor_notes and investor_notes.strip():
        parts.append(
            "\n--- INVESTOR'S NOTES ON THIS NAME (the human's view; engage with it "
            "honestly — incorporate if right, reason against it if not) ---"
        )
        parts.append(investor_notes.strip())
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
    out = (
        f"{a.one_line()}\n"
        f"    suggested_max_weight={a.suggested_max_weight:.0%}"
    )
    if a.market_implied:
        out += f"\n    market implies: {a.market_implied}"
    if a.mispricing_thesis:
        out += f"\n    mispricing thesis: {a.mispricing_thesis}"
    out += (
        f"\n    bull: {a.bull_case}"
        f"\n    bear: {a.bear_case}"
        f"\n    risks: {a.key_risks}"
    )
    return out


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
    investor_notes: str = "",
) -> str:
    parts = [f"=== ALLOCATION FOR {as_of.isoformat()} ==="]
    parts.append(f"\nMARKET REGIME: {regime}")
    parts.append(f"STRATEGIST THESIS: {market_thesis}")
    parts.append(_fmt_narrative(narrative))
    parts.append(_fmt_investor_notes(investor_notes))

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


# =============================================================================
# Post-run feedback dialogue
# =============================================================================
FEEDBACK_SYSTEM = """You are the Portfolio Manager of AI Investment Buddy, talking with the human \
investor you work for right after presenting today's decision. This is a real discussion, not a \
formality.

Engage substantively and with intellectual honesty:
- If the investor raises something you missed or got wrong, acknowledge it plainly and say how it \
changes your view.
- If you think they are mistaken, push back — challenge them with specific reasoning, evidence, or \
valuation logic. Do not cave just because they are the boss; your job is to be right, not agreeable.
- If it is a genuine judgement call, lay out both sides and where you land.
- Ask a sharp follow-up question when their input is interesting but incomplete.
Be concise and direct — a few sentences, like a smart colleague, not an essay.

Then capture what is worth REMEMBERING for your future self. A note should be a durable view about \
a company or the market that should inform future analysis — not chit-chat. Mark changes_thesis=true \
ONLY when the input genuinely changes how you'd value a name (this forces a fresh valuation next \
run). Attach company-specific views to that ticker; keep broad market views as a market_note. If \
there is nothing durable to store, return empty arrays. Call submit_feedback exactly once."""

FEEDBACK_TOOL = {
    "name": "submit_feedback",
    "description": "Reply to the investor and capture any durable takeaways.",
    "input_schema": {
        "type": "object",
        "properties": {
            "response": {
                "type": "string",
                "description": "Your reply to the investor — engage, challenge, or agree, honestly.",
            },
            "stance": {
                "type": "string",
                "enum": ["AGREE", "PARTIALLY_AGREE", "DISAGREE", "NEED_MORE_INFO"],
                "description": "Your stance toward the investor's latest point.",
            },
            "ticker_notes": {
                "type": "array",
                "description": "Company-specific takeaways worth remembering. Empty if none.",
                "items": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "note": {
                            "type": "string",
                            "description": "The durable view to store (synthesis of the exchange).",
                        },
                        "changes_thesis": {
                            "type": "boolean",
                            "description": "True only if this changes how you'd value the name.",
                        },
                    },
                    "required": ["ticker", "note", "changes_thesis"],
                },
            },
            "market_note": {
                "type": "string",
                "description": "A durable market-wide takeaway to remember, if any. Empty otherwise.",
            },
        },
        "required": ["response", "stance", "ticker_notes", "market_note"],
    },
}


def build_feedback_message(context: str, transcript: list[dict]) -> str:
    """``transcript`` is a list of {role: 'investor'|'pm', text: ...} turns."""
    parts = ["=== TODAY'S DECISION (context for the discussion) ===", context, ""]
    parts.append("=== CONVERSATION SO FAR ===")
    for turn in transcript:
        who = "INVESTOR" if turn["role"] == "investor" else "YOU (PM)"
        parts.append(f"{who}: {turn['text']}")
    parts.append(
        "\nRespond to the investor's latest message and capture any durable notes. "
        "Call submit_feedback exactly once."
    )
    return "\n".join(parts)
