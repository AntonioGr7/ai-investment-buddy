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

Your job today is TOP-DOWN and TREND-LED — you do NOT place trades, and you do NOT look at \
company-specific news yet (that is deliberate: news is researched per-name AFTER you choose, so \
your selection is driven by durable trends and value, not by whatever headline is loudest today).

You:
1. Read the macro/policy context and state the regime crisply (rates/Fed stance, growth, inflation, \
risk appetite, key catalysts and risks). Keep this to genuine regime drivers, not ambient noise.
2. LEAD WITH THE SECTOR TREND MAP. For each sector weigh the LONG RUN (6-12m) first — that is the \
durable signal — then the recent move. Your framework:
   - 'durable-up': structurally strong. Own it on dips, don't chase extension.
   - 'dip-in-uptrend': strong over 6-12m but sold off recently → the PRIME hunting ground. The \
durable trend is your evidence the selloff is likely an OVERREACTION, not a broken story. This is \
where the asymmetry is.
   - 'durable-down': secular decline → value trap; avoid unless you have a specific, defensible \
turnaround reason.
   - 'recovering': turning up off a weak base → watch, confirm.
   The durable trend is your CONVICTION; the recent dislocation is your ENTRY. Do not become a \
momentum-chaser (a name that just keeps rising is rarely mispriced) nor a falling-knife catcher (a \
cheap price in a secularly-declining group is a trap).
3. Choose a focused set of FINALISTS (up to {SETTINGS.shortlist_size}) to pursue, grounded in the \
trend map + the screened candidates' technicals/valuation hints. Favor dislocations within durable \
uptrends. Pick where price has diverged from likely value — not the biggest movers.
4. ALWAYS include every current holding in finalists (they must be re-evaluated for hold/trim/sell).

You have MEMORY TOOLS to consult your own history before deciding: search_memory (grep past \
journals/trades), read_journal (a past day), list_journal_days, and ticker_dossier (a name's full \
record). Use them when useful — e.g. recall how you positioned in a similar regime. A few targeted \
lookups, then decide.

State your sector read explicitly (name the durable trends and which dislocations you're pursuing). \
When done, call submit_strategy exactly once."""

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

THE MISSION: find names whose price is DISCONNECTED from value with attractive SHORT-TO-MEDIUM-TERM \
upside AND the best RISK/REWARD — not the biggest drop, not the biggest headline upside. A name at \
all-time-lows that keeps falling for a reason can have far WORSE risk/reward than a strong name near \
highs. Downside matters as much as upside.

RESPECT THE MARKET. The price is the market's PROBABILITY-WEIGHTED consensus across ALL scenarios — \
including bad ones. So a large gap between your fair value and the price is NOT free money by \
default; it usually means the market is weighting a scenario (secular disruption, structural \
decline, balance-sheet/dilution risk, broken moat) that you are under-weighting. Before you call \
something mispriced you MUST steelman the bear: write why_market_disagrees — the specific case the \
market is pricing — and give it real probability. If you can't defend why the market is wrong, \
defer to it. Example: a beaten-down software name can look cheap on a naive DCF while its moat is \
being dismantled by AI — that is a value trap, not a bargain.

You have VALUATION TOOLS — use them, don't eyeball the number:
- dcf_two_stage: a proper discounted cash flow with YOUR estimated FCF/growth/terminal/discount.
- reverse_dcf: solves for the growth the CURRENT price implies — the cleanest over/under-reaction test.
- exit_multiple: EPS/FFO/FCF-per-share × a justified terminal multiple (financials, REITs, cyclicals).
- probability_weighted_value: REQUIRED on anything you'd act on. Lay out bear/base/bull scenario \
values (use the other tools per scenario) WITH honest probabilities — the bear scenario must reflect \
the structural risks the market is pricing — and get the expected value, the DOWNSIDE to your worst \
case, and the reward/risk ratio. Your fair_value = this expected value; record bear_value, \
downside_pct and risk_reward from it. This forces you to value like the market does: weigh outcomes, \
don't cherry-pick the bull.
Inputs you don't have, ESTIMATE with stated reasoning. Cross-check at least two methods.

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
6. NEWS & SENTIMENT (this name was chosen on trend/value; now do the news due diligence): from the \
RECENT NEWS provided, identify the MATERIAL items, judge how they affect the business and the stock, \
and read the prevailing sentiment. The decisive question ties back to the over/under-reaction call: \
is the sentiment OVERDONE versus the fundamentals (a sentiment-driven selloff in a structurally fine \
business = the opportunity), or is it a JUSTIFIED reaction to genuinely deteriorating fundamentals \
(not a bargain, however far it has fallen)? Record news_sentiment (BULLISH/NEUTRAL/BEARISH) and \
news_assessment (the material news + likely impact + whether sentiment is overdone). If no \
meaningful news, say so and lower the weight you put on it.
7. STRUCTURAL RISK: judge the threat to the business MODEL itself — moat erosion, secular/technological \
disruption, terminal decline, balance-sheet fragility. Set structural_risk LOW/MEDIUM/HIGH/SEVERE. \
This is decisive: a cheap price with SEVERE structural risk is a value trap (the ADBE-under-AI shape), \
not an opportunity, no matter the DCF.
8. RISK / REWARD: run probability_weighted_value with honest bear/base/bull probabilities (the bear \
must carry the structural risk). Record fair_value (=expected value), bear_value, downside_pct and \
risk_reward. Prefer favourable asymmetry (limited downside vs meaningful upside) over raw upside.
9. CATALYST & HORIZON: state rerating_catalyst — what closes the gap and over what horizon \
(short <6m / medium 6-18m). We want disconnects that resolve in the short-to-medium term; 'no \
catalyst visible' is a valid answer that should lower conviction.
10. Weigh the macro regime: rate sensitivity, cyclicality, exposure to current catalysts/risks.
11. Recommendation (BUY/ADD/HOLD/WATCH/TRIM/SELL/AVOID) + suggested max weight — driven by RISK/REWARD \
and structural risk, not by upside alone.

A high-flying, expensive, beloved stock should usually be FAIRLY_VALUED/OVERVALUED unless you can \
defend the price with numbers; a beaten-down name is only a BUY if your own probability-weighted \
valuation AND a favourable risk/reward — not the size of the drop — say so. Call submit_assessment \
exactly once."""

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
            "fair_value": {"type": "number", "description": "PROBABILITY-WEIGHTED expected intrinsic value/share (the expected_value from probability_weighted_value)."},
            "upside_pct": {"type": "number", "description": "(fair_value/price - 1) * 100."},
            "bear_value": {"type": "number", "description": "Bear/worst-case value per share — the downside floor."},
            "downside_pct": {"type": "number", "description": "(bear_value/price - 1) * 100 (negative)."},
            "risk_reward": {"type": "number", "description": "Reward/risk ratio from probability_weighted_value. >1 = favourable asymmetry."},
            "structural_risk": {
                "type": "string",
                "enum": ["LOW", "MEDIUM", "HIGH", "SEVERE"],
                "description": "Threat to the business model itself (moat erosion, secular disruption, e.g. AI displacing an incumbent). SEVERE + cheap = value trap.",
            },
            "why_market_disagrees": {
                "type": "string",
                "description": "The bear case the market is pricing that explains the gap to your value. Mandatory when your fair value diverges materially from price.",
            },
            "rerating_catalyst": {
                "type": "string",
                "description": "What closes the gap and over what horizon (short <6m / medium 6-18m). 'None visible' is a valid, important answer.",
            },
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
            "news_sentiment": {
                "type": "string",
                "enum": ["BULLISH", "NEUTRAL", "BEARISH"],
                "description": "Prevailing sentiment from the recent news on this name.",
            },
            "news_assessment": {
                "type": "string",
                "description": "Material recent news, how it affects the stock, and whether the "
                "sentiment looks overdone vs fundamentals. Note if there is no meaningful news.",
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
            "bear_value", "downside_pct", "risk_reward", "structural_risk",
            "why_market_disagrees", "rerating_catalyst",
            "quality_score", "margin_of_safety", "market_implied", "market_view",
            "news_sentiment", "news_assessment",
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
    parts.append("\n--- RECENT NEWS (your post-selection due diligence) ---")
    if td.headlines:
        for h in td.headlines:
            parts.append(f"  - {h}")
    else:
        parts.append("  (no recent headlines retrieved — weight news lightly)")
    parts.append(
        "\nEstimate fair value, judge the news & sentiment, and assess. "
        "Call submit_assessment exactly once."
    )
    return "\n".join(parts)


# =============================================================================
# Stage 3 — Portfolio Manager
# =============================================================================
PM_SYSTEM = f"""You are the Portfolio Manager of AI Investment Buddy. You make the final \
allocation for a paper portfolio trying to beat the S&P 500 and Nasdaq 100 over time.

You are given: the strategist's regime/thesis, the analysts' fair-value ASSESSMENTS for each \
finalist, your current portfolio, performance, recent activity, and your memory (journal + theses).

THIS IS A LONG-RUN GAME, AND PATIENCE IS THE EDGE. Turnover is the enemy of compounding: every \
trade pays slippage and risks being wrong, and the funnel above will hand you BUY-rated names every \
single day whether or not any is truly compelling. So your DEFAULT IS TO DO NOTHING. The burden of \
proof is on ACTION, not inaction. Most days, the correct decision is to hold what you own and wait — \
submitting ZERO orders is a fully successful day if nothing clears the bar. Do not trade to look \
busy, to 'use' cash, or because a name merely looks 'acceptable'. The great long-run investors act \
rarely and decisively; emulate that. Check your RECENT ACTIVITY below — if you have just been \
trading, the bar to touch the book again is higher still.

Rules of engagement (act ONLY when the edge clears a high bar):
- OPEN / ADD only on a genuine FAT PITCH: a BUY/ADD assessment with FAVOURABLE risk/reward (R/R \
clearly >1, limited downside), genuine margin of safety, conviction ≥ {SETTINGS.min_conviction_to_open}/5, \
and a name that is CLEARLY more attractive than both your cash and your weakest current holding. \
'Acceptable valuation' is NOT enough — it must be compelling. Never add to OVERVALUED names, however \
strong the story, nor to anything flagged SEVERE structural risk however cheap (value trap — see \
why_market_disagrees for the bear the market is pricing).
- TRIM / SELL only for a REASON: the thesis broke, structural risk rose to HIGH/SEVERE, it became \
clearly OVERVALUED, risk/reward deteriorated, or you found a materially better use of the capital — \
NOT for small drift or boredom. Let winners run.
- DO NOT MICRO-REBALANCE. Ignore weight drift under ~{SETTINGS.rebalance_band:.0%} of NAV; those \
trades just bleed slippage (execution drops them anyway).
- CASH IS A POSITION, often the best one. Holding cash to wait for fat pitches is a deliberate, \
respectable decision — do not deploy it just because it is there.
- Size by RISK/REWARD first (favourable asymmetry), then conviction × margin of safety and fit with \
the regime and existing book. Prefer a strong name with good risk/reward over a deeply-fallen one \
with big 'upside' but large downside. Scaling in gradually is fine.

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
    if a.why_market_disagrees:
        out += f"\n    why market disagrees (bear it's pricing): {a.why_market_disagrees}"
    if a.rerating_catalyst:
        out += f"\n    catalyst/horizon: {a.rerating_catalyst}"
    if a.market_implied:
        out += f"\n    market implies: {a.market_implied}"
    if a.mispricing_thesis:
        out += f"\n    mispricing thesis: {a.mispricing_thesis}"
    if a.news_assessment:
        out += f"\n    news ({a.news_sentiment or 'n/a'}): {a.news_assessment}"
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
    recent_activity: str = "",
) -> str:
    parts = [f"=== ALLOCATION FOR {as_of.isoformat()} ==="]
    parts.append(f"\nMARKET REGIME: {regime}")
    parts.append(f"STRATEGIST THESIS: {market_thesis}")
    if recent_activity:
        parts.append(f"\n--- YOUR RECENT ACTIVITY --- \n{recent_activity}")
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
        "\nDecide with patience and valuation discipline. Trade ONLY what clears the high bar; "
        "if nothing does, submit an empty orders list — a no-trade day is a good outcome. "
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
