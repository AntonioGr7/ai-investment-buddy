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
from .graph import assess_ticker, build_graph
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
        sector_scan: str,
        recent_journal: list[str],
        theses: dict[str, dict],
        performance: str,
        sleeve: list[TickerData] | None = None,
        market_news: list[dict] | None = None,
        holdings: list[str] | None = None,
        watchlist: list[str] | None = None,
        narrative: str = "",
        investor_notes: str = "",
        recent_activity: str = "",
        risk_summary: str = "",
        calibration_summary: str = "",
        industry_scan: str = "",
        force_revaluation: bool = False,
        news_fetcher=None,
        toolkit: MemoryToolkit | None = None,
        on_progress=None,
    ) -> BrainResult:
        state = {
            "as_of": as_of,
            "macro": macro,
            "market_news": market_news or [],
            "shortlist": shortlist,
            "sleeve": sleeve or [],
            "sector_scan": sector_scan,
            "industry_scan": industry_scan,
            "holdings": holdings or [p["ticker"] for p in portfolio_state.get("positions", [])],
            "watchlist": watchlist or [],
            "portfolio_state": portfolio_state,
            "performance": performance,
            "recent_journal": recent_journal,
            "theses": theses,
            "narrative": narrative,
            "investor_notes": investor_notes,
            "recent_activity": recent_activity,
            "risk_summary": risk_summary,
            "calibration_summary": calibration_summary,
            "force_revaluation": force_revaluation,
            "news_fetcher": news_fetcher,
            "toolkit": toolkit or MemoryToolkit(),
            "progress": on_progress,
        }
        out = self.graph.invoke(state)
        return BrainResult(
            decision=out["decision"],
            strategy=out["strategy"],
            assessments=out.get("assessments", []),
        )

    def valuate(
        self,
        td: TickerData,
        regime: str = "",
        market_thesis: str = "",
        position: dict | None = None,
        dossier: str = "",
    ) -> ValuationAssessment | None:
        """Run one on-demand valuation for a single name (the `aib valuate` path)."""
        from ..macro_sleeve import is_hedge
        from .graph import assess_macro_hedge

        # Sleeve instruments (gold/commodities/duration/dollar) have no cash flows —
        # value them on the regime/role hedge path, not the DCF analyst.
        if is_hedge(td.ticker):
            td.asset_class = "macro_hedge"
            return assess_macro_hedge(self.client, td, regime, market_thesis, position)
        return assess_ticker(
            self.client, td, regime, market_thesis, position, dossier
        )

    def curiosity_verdict(
        self,
        assessment: ValuationAssessment,
        regime: str = "",
        market_thesis: str = "",
        sector_context: str = "",
        portfolio_state: dict | None = None,
        recent_activity: str = "",
    ) -> dict:
        """The PM's candid take on a user-picked name (full-agent treatment). NOT a
        trade — explicitly framed as an investor curiosity. Returns the verdict payload."""
        from . import prompts

        user = prompts.build_curiosity_message(
            assessment, regime, market_thesis, sector_context,
            portfolio_state or {}, recent_activity,
        )
        return self.client.structured_call(
            prompts.CURIOSITY_VERDICT_SYSTEM, user, prompts.CURIOSITY_VERDICT_TOOL
        )

    def discuss(self, context: str, transcript: list[dict]) -> dict:
        """One turn of the post-run feedback dialogue. ``transcript`` is the
        running list of {role, text} turns (latest is the investor's). Returns the
        feedback tool payload: response, stance, ticker_notes, market_note."""
        from . import prompts

        user = prompts.build_feedback_message(context, transcript)
        return self.client.structured_call(
            prompts.FEEDBACK_SYSTEM, user, prompts.FEEDBACK_TOOL
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
