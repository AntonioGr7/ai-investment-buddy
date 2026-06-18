"""Memory consolidation: maintain a rolling 'portfolio narrative'.

Each committed day we fold the latest decision into a compact, evolving story of
the portfolio — overall stance, why we hold what we hold, lessons, and what we're
watching. This narrative is always loaded into the strategist/PM prompts, giving
cheap long-horizon memory without dumping the entire journal into context.
"""

from __future__ import annotations

from datetime import date

from ..models import Decision, StrategistView
from .llm import LLMClient

CONSOLIDATE_SYSTEM = """You maintain the PORTFOLIO NARRATIVE for AI Investment Buddy — a compact, \
evolving memory the manager re-reads every day. Given the previous narrative and today's \
decision, produce an UPDATED narrative.

Keep it tight (roughly 200-350 words). It should capture, in prose or short sections:
- Current overall stance and market regime view.
- Core positions and the one-line reason each is held (and conviction).
- Recent changes and WHY (what was added/trimmed/sold and the trigger).
- Lessons learned / mistakes to avoid, and standing rules you are following.
- What you are watching for next (catalysts, levels, theses to confirm/kill).

Drop stale detail; preserve durable insight. This is your memory of the journey, not a log. \
Call submit_narrative exactly once."""

NARRATIVE_TOOL = {
    "name": "submit_narrative",
    "description": "Submit the updated portfolio narrative.",
    "input_schema": {
        "type": "object",
        "properties": {"narrative": {"type": "string"}},
        "required": ["narrative"],
    },
}


def _orders_summary(decision: Decision) -> str:
    if not decision.orders:
        return "No trades today."
    return "; ".join(
        f"{o.action.value} {o.ticker}→{o.target_weight:.0%} (conv {o.conviction})"
        for o in decision.orders
    )


def update_narrative(
    client: LLMClient,
    prior_narrative: str,
    as_of: date,
    strategy: StrategistView | None,
    decision: Decision,
    performance: str,
) -> str:
    user = "\n".join(
        [
            "=== PREVIOUS NARRATIVE ===",
            prior_narrative.strip() or "(none yet — this is the first entry)",
            f"\n=== TODAY ({as_of.isoformat()}) ===",
            f"Regime: {strategy.regime if strategy else '?'}",
            f"PM thesis: {decision.market_thesis}",
            f"Orders: {_orders_summary(decision)}",
            f"Target cash: {decision.target_cash_weight:.0%}",
            f"Notes to self: {decision.notes}",
            f"\nPerformance:\n{performance}",
            "\nProduce the updated narrative. Call submit_narrative exactly once.",
        ]
    )
    payload = client.structured_call(CONSOLIDATE_SYSTEM, user, NARRATIVE_TOOL)
    return str(payload.get("narrative", prior_narrative)).strip()
