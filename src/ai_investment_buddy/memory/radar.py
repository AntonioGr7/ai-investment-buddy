"""The agent's self-built watchlist — its "radar".

The opportunity board (``valuations.py``) ranks every name we've ever valued by
risk-adjusted attractiveness. The radar is the *actionable slice* of it: the names
whose price has fallen to — or to within striking distance of — the ATTENTION
PRICE the analyst set for them (``ValuationAssessment.entry_price``).

This is the monitoring view. Most days the agent sits in cash because nothing is
cheap enough; the radar answers "then what should I be watching, and how far does
each name have to fall before I care?" — and as names drop into range they show
up here on their own, so the agent (and the human) can watch the market narrow
toward the next fat pitch without re-reading the whole board.

Value traps are deliberately excluded: a name flagged SEVERE structural risk or
rated AVOID is cheap for a reason, so hitting its (mechanically low) entry price
is not a signal to act. Pure computation over the persisted valuations — no
network, no LLM — so it's cheap to refresh every run and on demand.
"""

from __future__ import annotations

from pathlib import Path

from ..config import DATA_DIR, ensure_dirs
from ..models import ValuationAssessment, ValuationRecord
from . import valuations as _v

# Status ordering for the radar (most actionable first).
_STATUS_RANK = {"TRIGGERED": 0, "APPROACHING": 1}


def _is_trap(a: ValuationAssessment) -> bool:
    """A cheap-for-a-reason name that hitting its entry price does NOT redeem."""
    return a.structural_risk == "SEVERE" or a.recommendation in ("AVOID", "SELL")


def radar_rows(
    records: list[ValuationRecord] | None = None,
    prices: dict[str, float] | None = None,
) -> list[dict]:
    """Names at or approaching their attention price, most actionable first.

    ``prices`` (ticker → fresh price) overrides each record's stored current price
    when available — pass today's prices inside the daily run; omit it and we fall
    back to the price as of the last valuation."""
    records = records if records is not None else _v.load_all()
    prices = {k.upper(): v for k, v in (prices or {}).items()}
    rows: list[dict] = []
    for rec in records:
        a = rec.latest.assessment
        # Backfill for names valued before entry_price existed (not persisted here).
        if a.entry_price is None:
            a.entry_price = _v.derive_entry_price(a)
        if not a.entry_price or _is_trap(a):
            continue
        px = prices.get(a.ticker.upper(), a.current_price)
        status = a.entry_status(px)
        if status not in _STATUS_RANK:  # FAR / UNKNOWN — not on the radar
            continue
        rows.append({
            "ticker": a.ticker,
            "sector": a.sector or "",
            "recommendation": a.recommendation,
            "status": status,
            "price": px,
            "entry_price": a.entry_price,
            "distance_to_entry": a.distance_to_entry_pct(px),
            "fair_value": a.fair_value,
            "upside_pct": a.upside_pct,
            "risk_reward": a.risk_reward,
            "structural_risk": a.structural_risk or "",
            "quality_score": a.quality_score,
            "score": _v.opportunity_score(a),
            "rerating_catalyst": a.rerating_catalyst or "",
            "last_assessed": rec.last_assessed.isoformat(),
            "price_is_fresh": a.ticker.upper() in prices,
        })
    # Triggered before approaching; within each, most attractive (score) first.
    rows.sort(key=lambda r: (_STATUS_RANK[r["status"]], -r["score"]))
    return rows


def triggered_tickers(
    records: list[ValuationRecord] | None = None,
    prices: dict[str, float] | None = None,
) -> list[str]:
    """Just the tickers currently AT or BELOW their attention price."""
    return [r["ticker"] for r in radar_rows(records, prices) if r["status"] == "TRIGGERED"]


def format_radar_markdown(
    records: list[ValuationRecord] | None = None,
    prices: dict[str, float] | None = None,
) -> str:
    rows = radar_rows(records, prices)
    if not rows:
        return (
            "# Radar — the agent's watchlist\n\n"
            "_No names are at or near their attention price right now._ The market "
            "has to come to us. See the full board at `data/opportunities.md`.\n"
        )

    def cell(r, k):
        v = r[k]
        if v is None:
            return "?"
        if k == "distance_to_entry":
            return f"{v:+.0f}%"
        if k in ("price", "entry_price", "fair_value"):
            return f"${v:.2f}"
        if k == "score":
            return f"{v:+.1f}"
        if k == "risk_reward":
            return f"{v:.1f}"
        if k == "rerating_catalyst":
            return (str(v)[:40] + "…") if len(str(v)) > 40 else str(v)
        return str(v)

    cols = ["status", "ticker", "sector", "recommendation", "price", "entry_price",
            "distance_to_entry", "fair_value", "upside_pct", "risk_reward",
            "structural_risk", "score", "rerating_catalyst", "last_assessed"]
    headers = ["Status", "Ticker", "Sector", "Rec", "Price", "Entry", "ΔEntry",
               "Fair", "Upside", "R/R", "StructRisk", "Score", "Catalyst", "As of"]
    n_trig = sum(1 for r in rows if r["status"] == "TRIGGERED")
    stale = any(not r["price_is_fresh"] for r in rows)
    lines = [
        f"# Radar — {len(rows)} names on watch ({n_trig} triggered)",
        "",
        "Names at (TRIGGERED) or within striking distance of (APPROACHING) the "
        "attention price the analyst set. Value traps (SEVERE structural risk / "
        "AVOID) are excluded. Sorted most-actionable first.",
    ]
    if stale:
        lines.append("")
        lines.append("_Prices are as of each name's last valuation; refresh with a daily run._")
    lines += [
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]

    def fmt_upside(r):
        v = r["upside_pct"]
        return f"{v:+.0f}%" if v is not None else "?"

    for r in rows:
        cells = []
        for k in cols:
            cells.append(fmt_upside(r) if k == "upside_pct" else cell(r, k))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_radar(
    records: list[ValuationRecord] | None = None,
    prices: dict[str, float] | None = None,
) -> Path | None:
    """Write the current radar to ``data/radar.md``. Best-effort."""
    try:
        ensure_dirs()
        path = DATA_DIR / "radar.md"
        path.write_text(format_radar_markdown(records, prices))
        return path
    except Exception:
        return None
