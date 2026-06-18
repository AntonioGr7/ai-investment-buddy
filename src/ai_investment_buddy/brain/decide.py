"""The decision engine: runs the 3-stage LangGraph brain and returns the result.

Returns not just the Decision but the strategist's regime view and every
analyst valuation, so the reasoning is fully auditable downstream."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from ..memory import MemoryToolkit
from ..models import (
    Decision,
    MacroSnapshot,
    StrategistView,
    TickerData,
    ValuationAssessment,
)
from .consolidate import update_narrative
from .graph import build_graph
from .llm import LLMClient, get_llm_client


@dataclass
class BrainResult:
    decision: Decision
    strategy: StrategistView
    assessments: list[ValuationAssessment] = field(default_factory=list)


class DecisionEngine:
    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or get_llm_client()
        self.graph = build_graph(self.client)

    def decide(
        self,
        as_of: date,
        portfolio_state: dict,
        macro: MacroSnapshot,
        shortlist: list[TickerData],
        recent_journal: list[str],
        theses: dict[str, dict],
        performance: str,
        market_news: list[dict] | None = None,
        holdings: list[str] | None = None,
        narrative: str = "",
        toolkit: MemoryToolkit | None = None,
        on_progress=None,
    ) -> BrainResult:
        state = {
            "as_of": as_of,
            "macro": macro,
            "market_news": market_news or [],
            "shortlist": shortlist,
            "holdings": holdings or [p["ticker"] for p in portfolio_state.get("positions", [])],
            "portfolio_state": portfolio_state,
            "performance": performance,
            "recent_journal": recent_journal,
            "theses": theses,
            "narrative": narrative,
            "toolkit": toolkit or MemoryToolkit(),
            "progress": on_progress,
        }
        out = self.graph.invoke(state)
        return BrainResult(
            decision=out["decision"],
            strategy=out["strategy"],
            assessments=out.get("assessments", []),
        )

    def consolidate(
        self,
        prior_narrative: str,
        as_of: date,
        strategy: StrategistView | None,
        decision: Decision,
        performance: str,
    ) -> str:
        return update_narrative(
            self.client, prior_narrative, as_of, strategy, decision, performance
        )
