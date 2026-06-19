"""Per-ticker valuation memory — the accumulating map of what the market offers.

Every fair-value assessment the analyst produces (in a daily run or an on-demand
``aib valuate``) is upserted to ``data/valuations/<TICKER>.json``: a new file the
first time we look at a name, an update (with the prior read kept in a capped
history) thereafter. Over time this covers the whole market, and lets us rank the
single most compelling opportunities across *everything* we've ever valued — not
just today's shortlist.

It is part of state: the directory travels in export/import snapshots.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from ..config import DATA_DIR, SETTINGS, ensure_dirs
from ..models import (
    InvestorNote,
    StoredValuation,
    ValuationAssessment,
    ValuationRecord,
)

VALUATIONS_DIR = DATA_DIR / "valuations"
_MAX_HISTORY = 12  # keep the last N prior reads per name


def _path(ticker: str) -> Path:
    return VALUATIONS_DIR / f"{ticker.upper()}.json"


def load(ticker: str) -> ValuationRecord | None:
    p = _path(ticker)
    if not p.exists():
        return None
    try:
        return ValuationRecord.model_validate_json(p.read_text())
    except Exception:
        return None


def load_all() -> list[ValuationRecord]:
    if not VALUATIONS_DIR.exists():
        return []
    out: list[ValuationRecord] = []
    for p in sorted(VALUATIONS_DIR.glob("*.json")):
        try:
            out.append(ValuationRecord.model_validate_json(p.read_text()))
        except Exception:
            continue
    return out


def save_valuation(
    assessment: ValuationAssessment,
    as_of: date,
    regime: str = "",
    headlines: list[str] | None = None,
) -> ValuationRecord:
    """Create or update the file for ``assessment.ticker``.

    The new read becomes ``latest``; the prior ``latest`` (if any, and from a
    different day) is pushed onto the capped history. Investor notes carry over."""
    ensure_dirs()
    VALUATIONS_DIR.mkdir(parents=True, exist_ok=True)
    ticker = assessment.ticker.upper()
    entry = StoredValuation(
        as_of=as_of, regime=regime, assessment=assessment,
        news_seen=list(headlines or []),
    )

    existing = load(ticker)
    if existing is None:
        rec = ValuationRecord(
            ticker=ticker, first_assessed=as_of, last_assessed=as_of, latest=entry
        )
    else:
        history = list(existing.history)
        # Don't duplicate same-day re-runs in history; just supersede.
        if existing.latest.as_of != as_of:
            history.append(existing.latest)
        history = history[-_MAX_HISTORY:]
        rec = ValuationRecord(
            ticker=ticker,
            first_assessed=existing.first_assessed,
            last_assessed=as_of,
            latest=entry,
            history=history,
            notes=existing.notes,
        )
    _path(ticker).write_text(rec.model_dump_json(indent=2))
    return rec


def save_many(
    assessments: list[ValuationAssessment],
    as_of: date,
    regime: str = "",
    headlines_by_ticker: dict[str, list[str]] | None = None,
) -> int:
    headlines_by_ticker = headlines_by_ticker or {}
    n = 0
    for a in assessments:
        # Don't re-persist cache hits — they weren't freshly analyzed.
        if a and a.ticker and not getattr(a, "from_cache", False):
            save_valuation(a, as_of, regime, headlines_by_ticker.get(a.ticker.upper()))
            n += 1
    return n


def add_note(ticker: str, note: InvestorNote) -> bool:
    """Attach an investor note to a name's valuation record. Returns False if we
    have never valued the ticker (no file to attach to)."""
    rec = load(ticker)
    if rec is None:
        return False
    rec.notes.append(note)
    _path(ticker).write_text(rec.model_dump_json(indent=2))
    return True


def find_reusable(
    ticker: str,
    current_price: float | None,
    current_headlines: list[str] | None,
    today: date,
) -> tuple[ValuationAssessment, date] | None:
    """Return (assessment, last_assessed) if a recent valuation can be reused as-is.

    Reusable only when: assessed within the TTL, no NEW headlines since, price has
    not moved materially, and no investor note has challenged the thesis since the
    last analysis. Otherwise None → caller should run a fresh valuation."""
    rec = load(ticker)
    if rec is None:
        return None
    if (today - rec.last_assessed).days > SETTINGS.valuation_ttl_days:
        return None
    # Investor feedback that challenges the thesis (added since the last analysis)
    # forces a fresh look so the agent incorporates it.
    for n in rec.notes:
        if n.changes_thesis and n.date >= rec.last_assessed:
            return None
    stored = rec.latest
    seen = set(stored.news_seen)
    if any(h not in seen for h in (current_headlines or [])):
        return None
    prior_px = stored.assessment.current_price
    if prior_px and current_price:
        if abs(current_price / prior_px - 1) > SETTINGS.revaluation_price_move:
            return None
    return stored.assessment, rec.last_assessed


# --- Cross-market ranking ----------------------------------------------------
_REC_SCORE = {
    "BUY": 2.0, "ADD": 1.5, "HOLD": 0.0, "WATCH": 0.3,
    "TRIM": -1.5, "SELL": -2.5, "AVOID": -2.5,
}
_MKT_SCORE = {"OVERREACTING": 1.0, "FAIR": 0.0, "UNDERREACTING": -1.0}


def opportunity_score(a: ValuationAssessment) -> float:
    """A blended attractiveness score for cross-market ranking.

    Rewards a buy-side recommendation, real upside to a conservatively-estimated
    fair value, business quality, a genuine margin of safety, conviction, and the
    market over-reacting (mispriced cheap). Higher = more compelling."""
    s = 0.0
    s += _REC_SCORE.get(a.recommendation, 0.0)
    s += (a.upside_pct or 0.0) / 20.0  # +20% upside ≈ +1.0
    s += (a.quality_score - 3) * 0.3
    s += (a.confidence - 3) * 0.2
    s += 1.0 if a.margin_of_safety else 0.0
    s += _MKT_SCORE.get(a.market_view, 0.0) * 0.5
    return round(s, 2)


def rank_opportunities(
    records: list[ValuationRecord] | None = None,
) -> list[tuple[ValuationRecord, float]]:
    """Return (record, score) pairs, most compelling first, using each name's
    latest assessment."""
    records = records if records is not None else load_all()
    scored = [(r, opportunity_score(r.latest.assessment)) for r in records]
    scored.sort(key=lambda rs: rs[1], reverse=True)
    return scored


# --- The market-wide board ---------------------------------------------------
BOARD_COLUMNS = [
    "score", "ticker", "sector", "archetype", "recommendation", "market_view",
    "current_price", "fair_value", "upside_pct", "valuation_verdict",
    "quality_score", "margin_of_safety", "confidence", "last_assessed",
]


def board_rows(records: list[ValuationRecord] | None = None) -> list[dict]:
    """Flat, sorted rows (most compelling first) summarising every name we've ever
    valued — cost (price) vs opportunity (fair value / upside / score). Powers the
    `aib opportunities` table, the CSV export, and the data/opportunities.md file."""
    rows = []
    for rec, score in rank_opportunities(records):
        a = rec.latest.assessment
        rows.append({
            "score": score,
            "ticker": a.ticker,
            "sector": a.sector or "",
            "archetype": a.archetype or "",
            "recommendation": a.recommendation,
            "market_view": a.market_view,
            "current_price": a.current_price,
            "fair_value": a.fair_value,
            "upside_pct": a.upside_pct,
            "valuation_verdict": a.valuation_verdict,
            "quality_score": a.quality_score,
            "margin_of_safety": a.margin_of_safety,
            "confidence": a.confidence,
            "last_assessed": rec.last_assessed.isoformat(),
        })
    return rows


def format_board_markdown(records: list[ValuationRecord] | None = None) -> str:
    """Render the full board as a markdown table (used for data/opportunities.md)."""
    rows = board_rows(records)
    if not rows:
        return "# Opportunity board\n\n_No valuations yet._\n"

    def cell(r, k):
        v = r[k]
        if v is None:
            return "?"
        if k == "upside_pct":
            return f"{v:+.0f}%"
        if k in ("current_price", "fair_value"):
            return f"${v:.2f}"
        if k == "margin_of_safety":
            return "Y" if v else "—"
        if k == "score":
            return f"{v:+.1f}"
        return str(v)

    headers = ["Score", "Ticker", "Sector", "Type", "Rec", "Market", "Price",
               "Fair", "Upside", "Verdict", "Q", "MoS", "Conf", "As of"]
    lines = [
        f"# Opportunity board — {len(rows)} names valued",
        "",
        "Sorted most-compelling first. Cost = Price; opportunity = Fair / Upside / Score.",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(cell(r, k) for k in BOARD_COLUMNS) + " |")
    return "\n".join(lines) + "\n"


def write_board() -> Path | None:
    """Write the always-current board to ``data/opportunities.md``. Best-effort."""
    try:
        ensure_dirs()
        path = DATA_DIR / "opportunities.md"
        path.write_text(format_board_markdown())
        return path
    except Exception:
        return None
