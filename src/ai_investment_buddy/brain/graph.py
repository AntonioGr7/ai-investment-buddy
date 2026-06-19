"""The 3-stage decision brain, orchestrated with LangGraph.

    START → strategist → analyst → portfolio_manager → END

Each node is a plain function that calls our provider-agnostic LLM client
(``brain.llm``), so the graph works identically with Claude, OpenAI, or Gemini.
The linear shape is deliberate for v1; LangGraph lets us later add conditional
edges and loops (e.g. a research node that re-runs when analyst confidence is
low) without restructuring callers.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..config import SETTINGS
from ..memory import MemoryToolkit, valuations
from ..models import (
    Action,
    Decision,
    MacroSnapshot,
    StrategistView,
    TickerData,
    TradeOrder,
    ValuationAssessment,
)
from . import prompts, valuation_tools
from .llm import LLMClient, get_llm_client
from .mem_tools import MEMORY_TOOL_SPECS, make_memory_executor

# How many finalists we deep-dive at most (cost guardrail; holdings always kept).
_MAX_FINALISTS = 15
_ANALYST_CONCURRENCY = 6


class BrainState(TypedDict, total=False):
    # Inputs
    as_of: date
    macro: MacroSnapshot
    market_news: list[dict]
    shortlist: list[TickerData]
    sector_scan: str
    holdings: list[str]
    watchlist: list[str]
    portfolio_state: dict
    performance: str
    recent_journal: list[str]
    theses: dict[str, dict]
    narrative: str
    investor_notes: str
    force_revaluation: bool
    toolkit: MemoryToolkit
    # Intermediates / outputs
    strategy: StrategistView
    assessments: list[ValuationAssessment]
    decision: Decision
    progress: Any  # optional callback(msg)


def _emit(state: BrainState, msg: str) -> None:
    cb = state.get("progress")
    if cb:
        cb(msg)


def _make_node_runner(client: LLMClient):
    """Build the three node functions bound to one LLM client."""

    def strategist_node(state: BrainState) -> dict:
        _emit(state, "Strategist: reading the macro/news regime (+ consulting memory)…")
        user = prompts.build_strategist_message(
            as_of=state["as_of"],
            macro=state["macro"],
            market_news=state["market_news"],
            shortlist=state["shortlist"],
            holdings=state["holdings"],
            watchlist=state.get("watchlist", []),
            performance=state["performance"],
            narrative=state.get("narrative", ""),
            sector_scan=state.get("sector_scan", ""),
            investor_notes=state.get("investor_notes", ""),
        )
        executor = make_memory_executor(
            state["toolkit"],
            on_call=lambda n, a: _emit(state, f"  ↳ memory: {n}({a})"),
        )
        payload = client.agentic_call(
            prompts.STRATEGIST_SYSTEM, user,
            MEMORY_TOOL_SPECS, prompts.STRATEGIST_TOOL, executor,
        )
        model_finalists = [str(t).upper().strip() for t in payload.get("finalists", [])]
        known = {td.ticker for td in state["shortlist"]} | set(state["holdings"])
        # Holdings + watchlist are FORCED finalists — they always go through the
        # full valuation, never capped out. The model's own picks fill the rest.
        forced = [t for t in dict.fromkeys(state["holdings"] + state.get("watchlist", []))
                  if t in known]
        extra = [t for t in model_finalists if t in known and t not in forced]
        finalists = forced + extra[: max(0, _MAX_FINALISTS - len(forced))]

        view = StrategistView(
            regime=str(payload.get("regime", "")),
            market_thesis=str(payload.get("market_thesis", "")),
            finalists=finalists,
            reasoning=str(payload.get("reasoning", "")),
            sector_read=str(payload.get("sector_read", "")),
        )
        _emit(state, f"Strategist: regime = {view.regime}; {len(finalists)} finalists.")
        return {"strategy": view}

    def analyst_node(state: BrainState) -> dict:
        view: StrategistView = state["strategy"]
        by_ticker = {td.ticker: td for td in state["shortlist"]}
        positions = {p["ticker"]: p for p in state["portfolio_state"].get("positions", [])}
        toolkit: MemoryToolkit = state["toolkit"]

        force = state.get("force_revaluation", False)
        as_of = state["as_of"]

        def assess(ticker: str) -> ValuationAssessment | None:
            td = by_ticker.get(ticker) or TickerData(ticker=ticker)
            # Skip a fresh model call if a recent valuation still holds (unless
            # forced). Keeps cost down and avoids re-deciding settled views.
            if not force:
                reuse = valuations.find_reusable(ticker, td.price, td.headlines, as_of)
                if reuse is not None:
                    cached, last = reuse
                    a = cached.model_copy(update={"from_cache": True})
                    if td.price:  # refresh to today's price for an accurate upside
                        a.current_price = td.price
                        if a.fair_value:
                            a.upside_pct = round((a.fair_value / td.price - 1) * 100, 1)
                    _emit(state, f"  ↳ {ticker}: reused valuation from {last} (no material change).")
                    return a
            try:
                dossier = toolkit.ticker_dossier(ticker)
            except Exception:
                dossier = ""
            return assess_ticker(
                client, td, view.regime, view.market_thesis,
                positions.get(ticker), dossier, _investor_notes_for(ticker),
                on_tool=lambda name, summary, t=ticker: _emit(state, f"  ↳ {t}: {summary}"),
            )

        _emit(state, f"Analyst: valuing {len(view.finalists)} finalists…")
        results: list[ValuationAssessment] = []
        with ThreadPoolExecutor(max_workers=_ANALYST_CONCURRENCY) as ex:
            for a in ex.map(assess, view.finalists):
                if a is not None:
                    results.append(a)
        # Most interesting first: BUY/ADD with margin of safety, then by upside.
        rank = {"BUY": 0, "ADD": 1, "HOLD": 2, "TRIM": 3, "WATCH": 4, "SELL": 5, "AVOID": 6}
        results.sort(
            key=lambda a: (rank.get(a.recommendation, 9), -(a.upside_pct or -999))
        )
        _emit(state, f"Analyst: {len(results)} assessments complete.")
        return {"assessments": results}

    def pm_node(state: BrainState) -> dict:
        _emit(state, "Portfolio manager: allocating with valuation discipline…")
        view: StrategistView = state["strategy"]
        user = prompts.build_pm_message(
            as_of=state["as_of"],
            regime=view.regime,
            market_thesis=view.market_thesis,
            assessments=state["assessments"],
            portfolio_state=state["portfolio_state"],
            performance=state["performance"],
            recent_journal=state["recent_journal"],
            theses=state["theses"],
            narrative=state.get("narrative", ""),
            investor_notes=state.get("investor_notes", ""),
        )
        executor = make_memory_executor(
            state["toolkit"],
            on_call=lambda n, a: _emit(state, f"  ↳ memory: {n}({a})"),
        )
        payload = client.agentic_call(
            prompts.PM_SYSTEM, user, MEMORY_TOOL_SPECS, prompts.PM_TOOL, executor
        )
        decision = _to_decision(state["as_of"], payload)
        return {"decision": decision}

    return strategist_node, analyst_node, pm_node


def _investor_notes_for(ticker: str) -> str:
    """Format the human investor's stored notes on a name for the analyst prompt."""
    rec = valuations.load(ticker)
    if not rec or not rec.notes:
        return ""
    lines = []
    for n in rec.notes[-5:]:
        lines.append(f"[{n.date}] INVESTOR: {n.user_view}")
        if n.agent_response:
            lines.append(f"          you ({n.stance}): {n.agent_response}")
    return "\n".join(lines)


def assess_ticker(
    client: LLMClient,
    td: TickerData,
    regime: str,
    market_thesis: str,
    position: dict | None = None,
    dossier: str = "",
    investor_notes: str = "",
    on_tool=None,
) -> ValuationAssessment | None:
    """Run ONE disciplined fair-value valuation, with the model free to call the
    DCF / reverse-DCF / exit-multiple calculators before submitting. Shared by the
    analyst node and the on-demand ``aib valuate`` command. None if the call fails."""
    user = prompts.build_analyst_message(
        td, regime, market_thesis, position, dossier, investor_notes
    )
    executor = valuation_tools.make_valuation_executor(on_call=on_tool)
    try:
        p = client.agentic_call(
            prompts.ANALYST_SYSTEM, user,
            valuation_tools.VALUATION_TOOL_SPECS, prompts.ANALYST_TOOL, executor,
            max_iters=SETTINGS.analyst_max_iters,
        )
    except Exception:
        return None
    price = td.price
    fair = p.get("fair_value")
    upside = p.get("upside_pct")
    if upside is None and fair and price:
        upside = (fair / price - 1) * 100
    return ValuationAssessment(
        ticker=td.ticker,
        sector=td.sector or "",
        archetype=str(p.get("archetype", "")),
        valuation_method=str(p.get("valuation_method", "")),
        fair_value=fair,
        current_price=price,
        upside_pct=round(upside, 1) if upside is not None else None,
        valuation_verdict=p.get("valuation_verdict", "FAIRLY_VALUED"),
        quality_score=int(p.get("quality_score", 3)),
        margin_of_safety=bool(p.get("margin_of_safety", False)),
        market_implied=str(p.get("market_implied", "")),
        market_view=str(p.get("market_view", "FAIR")),
        mispricing_thesis=str(p.get("mispricing_thesis", "")),
        bull_case=str(p.get("bull_case", "")),
        bear_case=str(p.get("bear_case", "")),
        key_risks=str(p.get("key_risks", "")),
        recommendation=p.get("recommendation", "WATCH"),
        suggested_max_weight=float(p.get("suggested_max_weight", 0.0)),
        confidence=int(p.get("confidence", 3)),
    )


def _to_decision(as_of: date, payload: dict) -> Decision:
    orders = []
    for o in payload.get("orders", []):
        try:
            orders.append(
                TradeOrder(
                    ticker=str(o["ticker"]).upper().strip(),
                    action=Action(o["action"]),
                    target_weight=float(o.get("target_weight", 0.0)),
                    rationale=str(o.get("rationale", "")),
                    conviction=int(o.get("conviction", 3)),
                )
            )
        except Exception:
            continue
    return Decision(
        as_of=as_of,
        market_thesis=str(payload.get("market_thesis", "")),
        orders=orders,
        target_cash_weight=float(payload.get("target_cash_weight", 0.0)),
        notes=str(payload.get("notes", "")),
    )


def build_graph(client: LLMClient | None = None):
    client = client or get_llm_client()
    strategist_node, analyst_node, pm_node = _make_node_runner(client)

    g = StateGraph(BrainState)
    g.add_node("strategist", strategist_node)
    g.add_node("analyst", analyst_node)
    g.add_node("portfolio_manager", pm_node)
    g.add_edge(START, "strategist")
    g.add_edge("strategist", "analyst")
    g.add_edge("analyst", "portfolio_manager")
    g.add_edge("portfolio_manager", END)
    return g.compile()
