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
from ..memory import MemoryToolkit
from ..models import (
    Action,
    Decision,
    MacroSnapshot,
    StrategistView,
    TickerData,
    TradeOrder,
    ValuationAssessment,
)
from . import prompts
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
    holdings: list[str]
    portfolio_state: dict
    performance: str
    recent_journal: list[str]
    theses: dict[str, dict]
    narrative: str
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
            performance=state["performance"],
            narrative=state.get("narrative", ""),
        )
        executor = make_memory_executor(
            state["toolkit"],
            on_call=lambda n, a: _emit(state, f"  ↳ memory: {n}({a})"),
        )
        payload = client.agentic_call(
            prompts.STRATEGIST_SYSTEM, user,
            MEMORY_TOOL_SPECS, prompts.STRATEGIST_TOOL, executor,
        )
        finalists = [str(t).upper().strip() for t in payload.get("finalists", [])]
        # Always include holdings; de-dup; cap.
        for h in state["holdings"]:
            if h not in finalists:
                finalists.append(h)
        # Keep only names we actually have data for (shortlist or holdings).
        known = {td.ticker for td in state["shortlist"]} | set(state["holdings"])
        finalists = [t for t in finalists if t in known][:_MAX_FINALISTS]

        view = StrategistView(
            regime=str(payload.get("regime", "")),
            market_thesis=str(payload.get("market_thesis", "")),
            finalists=finalists,
            reasoning=str(payload.get("reasoning", "")),
        )
        _emit(state, f"Strategist: regime = {view.regime}; {len(finalists)} finalists.")
        return {"strategy": view}

    def analyst_node(state: BrainState) -> dict:
        view: StrategistView = state["strategy"]
        by_ticker = {td.ticker: td for td in state["shortlist"]}
        positions = {p["ticker"]: p for p in state["portfolio_state"].get("positions", [])}

        toolkit: MemoryToolkit = state["toolkit"]

        def assess(ticker: str) -> ValuationAssessment | None:
            td = by_ticker.get(ticker) or TickerData(ticker=ticker)
            # Inject our own history with this name (cheap per-name memory).
            try:
                dossier = toolkit.ticker_dossier(ticker)
            except Exception:
                dossier = ""
            user = prompts.build_analyst_message(
                td, view.regime, view.market_thesis, positions.get(ticker), dossier
            )
            try:
                p = client.structured_call(
                    prompts.ANALYST_SYSTEM, user, prompts.ANALYST_TOOL
                )
            except Exception:
                return None
            price = td.price
            fair = p.get("fair_value")
            upside = p.get("upside_pct")
            if upside is None and fair and price:
                upside = (fair / price - 1) * 100
            return ValuationAssessment(
                ticker=ticker,
                fair_value=fair,
                current_price=price,
                upside_pct=round(upside, 1) if upside is not None else None,
                valuation_verdict=p.get("valuation_verdict", "FAIRLY_VALUED"),
                quality_score=int(p.get("quality_score", 3)),
                margin_of_safety=bool(p.get("margin_of_safety", False)),
                bull_case=str(p.get("bull_case", "")),
                bear_case=str(p.get("bear_case", "")),
                key_risks=str(p.get("key_risks", "")),
                recommendation=p.get("recommendation", "WATCH"),
                suggested_max_weight=float(p.get("suggested_max_weight", 0.0)),
                confidence=int(p.get("confidence", 3)),
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
