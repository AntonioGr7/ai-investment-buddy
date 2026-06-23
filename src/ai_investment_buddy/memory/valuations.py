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


# --- Attention price ---------------------------------------------------------
# Fallback only: the analyst is asked to author entry_price directly (it can weigh
# catalysts and name-specific nuance). When it omits it, we derive a sane default
# from fair value minus the margin of safety the business demands — wider for low
# quality and high structural risk, the same logic the analyst is told to apply.
_STRUCT_MARGIN = {"LOW": 0.0, "MEDIUM": 0.05, "HIGH": 0.15, "SEVERE": 0.30}
_BASE_MARGIN = 0.15
_MIN_MARGIN, _MAX_MARGIN = 0.10, 0.45


def required_margin(a: ValuationAssessment) -> float:
    """The margin of safety (as a fraction of fair value) this name should demand
    before it's worth acting on — used to derive a fallback attention price."""
    m = _BASE_MARGIN
    m += _STRUCT_MARGIN.get(a.structural_risk, 0.05)
    m += (3 - a.quality_score) * 0.03  # lower quality → demand a deeper discount
    return max(_MIN_MARGIN, min(_MAX_MARGIN, m))


def derive_entry_price(a: ValuationAssessment) -> float | None:
    """Fallback attention price = fair_value × (1 − required margin). None when we
    have no fair value to anchor on."""
    if not a.fair_value or a.fair_value <= 0:
        return None
    return round(a.fair_value * (1 - required_margin(a)), 2)


def load(ticker: str) -> ValuationRecord | None:
    """Read one name's record from the DB (the synchronized index). The per-name
    JSON file remains the durable export written alongside on every save."""
    from . import db

    return db.get_valuation(ticker)


def load_all() -> list[ValuationRecord]:
    from . import db

    return db.all_valuations()


def search(query: str, limit: int = 50) -> list[ValuationRecord]:
    """Full-text search the thesis prose (bull/bear/mispricing/news/etc.). Returns
    matching records, best match first. Empty if FTS is unavailable. Powers the
    eventual frontend's 'read the thesis' / search-by-idea navigation."""
    from . import db

    return [r for r in (load(t) for t in db.search_valuations(query, limit)) if r]


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
    # Backfill the attention price deterministically if the analyst didn't author one.
    if assessment.entry_price is None:
        assessment.entry_price = derive_entry_price(assessment)
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
    _persist(rec, regime)
    return rec


def _persist(rec: ValuationRecord, regime: str) -> None:
    """Dual-write: the per-name JSON file (durable, git-diffable, snapshot-bundled
    export) AND the SQLite index (the queryable read source). The file is written
    first so a DB hiccup never loses data — the index is rebuildable from it."""
    from . import db

    VALUATIONS_DIR.mkdir(parents=True, exist_ok=True)
    _path(rec.ticker).write_text(rec.model_dump_json(indent=2))
    a = rec.latest.assessment
    db.upsert_valuation(rec, regime, opportunity_score(a), a.entry_price)


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
    _persist(rec, rec.latest.regime)
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
# Structural/existential risk dominates — a cheap value trap must NOT rank well.
_STRUCT_PENALTY = {"LOW": 0.5, "MEDIUM": 0.0, "HIGH": -1.5, "SEVERE": -3.5}


def opportunity_score(a: ValuationAssessment) -> float:
    """A RISK-ADJUSTED attractiveness score for cross-market ranking.

    The objective is best risk/reward on a short-to-medium horizon — not raw
    upside. So we reward favourable asymmetry (reward/risk), penalise downside and
    especially structural risk (a cheap name being structurally destroyed is a
    trap), and fold in recommendation, quality, conviction and the over-reaction
    read. Higher = more compelling."""
    s = 0.0
    s += _REC_SCORE.get(a.recommendation, 0.0)
    # Reward/risk asymmetry is the heart of it; fall back to raw upside if absent.
    if a.risk_reward is not None:
        s += min(a.risk_reward, 4.0)
    else:
        s += (a.upside_pct or 0.0) / 20.0
    # Penalise downside to the bear case (downside_pct is negative).
    if a.downside_pct is not None:
        s += max(a.downside_pct, -60.0) / 30.0  # -30% downside ≈ -1.0, capped
    s += (a.quality_score - 3) * 0.3
    s += (a.confidence - 3) * 0.2
    s += 1.0 if a.margin_of_safety else 0.0
    s += _MKT_SCORE.get(a.market_view, 0.0) * 0.5
    s += _STRUCT_PENALTY.get(a.structural_risk, 0.0)
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
    "current_price", "fair_value", "entry_price", "distance_to_entry", "entry_status",
    "upside_pct", "downside_pct", "risk_reward",
    "structural_risk", "valuation_verdict", "quality_score", "margin_of_safety",
    "confidence", "rerating_catalyst", "last_assessed",
]


def board_rows(records: list[ValuationRecord] | None = None) -> list[dict]:
    """Flat, sorted rows (most compelling first) summarising every name we've ever
    valued — cost (price) vs opportunity (fair value / upside / score). Powers the
    `aib opportunities` table, the CSV export, and the data/opportunities.md file."""
    rows = []
    for rec, score in rank_opportunities(records):
        a = rec.latest.assessment
        # Backfill a fallback attention price for names valued before the field
        # existed, so existing coverage gets it without waiting for re-valuation.
        if a.entry_price is None:
            a.entry_price = derive_entry_price(a)
        rows.append({
            "score": score,
            "ticker": a.ticker,
            "sector": a.sector or "",
            "archetype": a.archetype or "",
            "recommendation": a.recommendation,
            "market_view": a.market_view,
            "current_price": a.current_price,
            "fair_value": a.fair_value,
            "entry_price": a.entry_price,
            "distance_to_entry": a.distance_to_entry_pct(),
            "entry_status": a.entry_status(),
            "upside_pct": a.upside_pct,
            "downside_pct": a.downside_pct,
            "risk_reward": a.risk_reward,
            "structural_risk": a.structural_risk or "",
            "valuation_verdict": a.valuation_verdict,
            "quality_score": a.quality_score,
            "margin_of_safety": a.margin_of_safety,
            "confidence": a.confidence,
            "rerating_catalyst": a.rerating_catalyst or "",
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
        if k in ("upside_pct", "downside_pct", "distance_to_entry"):
            return f"{v:+.0f}%"
        if k in ("current_price", "fair_value", "entry_price"):
            return f"${v:.2f}"
        if k == "margin_of_safety":
            return "Y" if v else "—"
        if k == "score":
            return f"{v:+.1f}"
        if k == "risk_reward":
            return f"{v:.1f}"
        if k == "rerating_catalyst":
            return (str(v)[:40] + "…") if len(str(v)) > 40 else str(v)
        return str(v)

    headers = ["Score", "Ticker", "Sector", "Type", "Rec", "Market", "Price",
               "Fair", "Entry", "ΔEntry", "Status", "Upside", "Down", "R/R",
               "StructRisk", "Verdict", "Q", "MoS", "Conf", "Catalyst", "As of"]
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
