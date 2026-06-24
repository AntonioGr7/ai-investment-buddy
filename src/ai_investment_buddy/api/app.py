"""FastAPI app: read endpoints over the corpus + the on-demand valuation trigger.

All reads go through the existing module seams (valuations / radar / predictions),
which read from the SQLite index. Nothing here re-implements business logic — it
shapes the same data the CLI shows into JSON for a frontend.
"""

from __future__ import annotations

import os
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from ..config import DATA_DIR, SETTINGS
from ..engine.benchmark import compute_returns
from ..memory import db
from ..memory import predictions as P
from ..memory import radar as rad
from ..memory import store
from ..memory import valuations as v
from ..models import ValuationRecord
from .jobs import all_jobs, get_job, start_valuation

app = FastAPI(
    title="AI Investment Buddy API",
    version="0.1.0",
    description="Read access to the valuation corpus + an on-demand valuation trigger.",
)

# Dev-friendly CORS: a browser frontend will call from another origin. Lock this
# down (set AIB_API_CORS_ORIGINS to a comma-separated list) in production.
_origins = os.getenv("AIB_API_CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _origins.strip() == "*" else [o.strip() for o in _origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _summary(rec: ValuationRecord) -> dict:
    """Compact, list-friendly view of a name's latest assessment."""
    a = rec.latest.assessment
    return {
        "ticker": a.ticker,
        "sector": a.sector or None,
        "asset_class": a.asset_class or "equity",
        "archetype": a.archetype or None,
        "recommendation": a.recommendation,
        "market_view": a.market_view or None,
        "valuation_verdict": a.valuation_verdict,
        "fair_value": a.fair_value,
        "current_price": a.current_price,
        "entry_price": a.entry_price,
        "entry_status": a.entry_status(),
        "distance_to_entry_pct": a.distance_to_entry_pct(),
        "upside_pct": a.upside_pct,
        "downside_pct": a.downside_pct,
        "risk_reward": a.risk_reward,
        "structural_risk": a.structural_risk or None,
        "quality_score": a.quality_score,
        "margin_of_safety": a.margin_of_safety,
        "confidence": a.confidence,
        "score": v.opportunity_score(a),
        "last_assessed": rec.last_assessed.isoformat(),
    }


# --- Health ------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    conn = db.connect()
    nv = conn.execute("SELECT COUNT(*) FROM valuations").fetchone()[0]
    npred = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    return {
        "status": "ok",
        "valuations": nv,
        "predictions": npred,
        "full_text_search": db._has_fts.get(str(db._db_path()), False),
    }


# --- Portfolio (holdings, allocation, performance) ---------------------------
def _held_prices(pf) -> dict[str, float]:
    """Best-effort current prices for held tickers (mirrors the CLI status view)."""
    if not pf.positions:
        return {}
    from ..data import get_providers

    providers = get_providers()
    prices: dict[str, float] = {}
    try:
        hist = providers.prices.history(list(pf.positions.keys()), lookback_days=10)
        for t, df in hist.items():
            if df is not None and not df.empty and "Close" in df:
                prices[t] = float(df["Close"].dropna().iloc[-1])
    except Exception:
        pass
    for t in pf.positions:
        if t not in prices:
            try:
                px = providers.prices.latest_price(t)
                if px:
                    prices[t] = px
            except Exception:
                continue
    return prices


def _benchmark_levels() -> dict[str, float]:
    from ..data import get_providers

    try:
        macro = get_providers().macro.snapshot()
    except Exception:
        return {}
    return {
        label: macro.indicators[label]
        for label in SETTINGS.benchmarks
        if label in macro.indicators
    }


@app.get("/portfolio")
def portfolio() -> dict:
    """Current book: cash + marked-to-market positions, allocation weights,
    realized vs benchmark performance from inception. Fetches live prices, so
    this call can take a few seconds."""
    if not store.is_initialized():
        raise HTTPException(status_code=404, detail="No portfolio. Run `aib init` first.")
    pf = store.load_portfolio()
    prices = _held_prices(pf)
    nav = pf.nav(prices)
    invested = pf.invested_value(prices)

    positions = []
    for t, pos in pf.positions.items():
        px = prices.get(t)
        mv = pos.market_value(px) if px is not None else None
        pnl = pos.unrealized_pnl(px) if px is not None else None
        cost = pos.avg_cost * pos.shares
        positions.append({
            "ticker": t,
            "shares": pos.shares,
            "avg_cost": pos.avg_cost,
            "price": px,
            "value": mv,
            "weight": (mv / nav) if (mv is not None and nav > 0) else None,
            "unrealized_pnl": pnl,
            "unrealized_pnl_pct": ((px / pos.avg_cost - 1) * 100) if px and pos.avg_cost else None,
            "cost_basis": cost,
        })
    positions.sort(key=lambda p: (p["value"] is not None, p["value"] or 0), reverse=True)

    benchmarks = _benchmark_levels()
    nav_history = store.load_nav_history()
    returns = compute_returns(nav_history, nav, benchmarks)
    exp = store.load_experiment() or {}

    return {
        "cash": pf.cash,
        "invested": invested,
        "nav": nav,
        "n_positions": len(pf.positions),
        "cash_weight": (pf.cash / nav) if nav > 0 else 1.0,
        "starting_capital": SETTINGS.starting_capital,
        "positions": positions,
        "returns": returns,
        "benchmarks": list(SETTINGS.benchmarks.keys()),
        "experiment": exp,
        "data_dir": str(DATA_DIR),
        "runs_recorded": len(nav_history),
    }


@app.get("/trades")
def trades(limit: int = Query(0, ge=0, description="0 = all."), ticker: str | None = None) -> dict:
    """The append-only paper trade ledger, most recent first."""
    rows = store.load_trades()
    if ticker:
        rows = [t for t in rows if t.ticker.upper() == ticker.upper()]
    rows = list(reversed(rows))
    total = len(rows)
    if limit:
        rows = rows[:limit]
    return {"total": total, "trades": rows}


@app.get("/nav_history")
def nav_history() -> dict:
    """Raw NAV + benchmark levels per recorded run (for the equity curve)."""
    return {"rows": store.load_nav_history(), "benchmarks": list(SETTINGS.benchmarks.keys())}


# --- The board (every name we've valued, ranked) -----------------------------
@app.get("/board")
def board(
    limit: int = Query(50, ge=0, le=2000, description="0 = no limit."),
    offset: int = Query(0, ge=0),
    buys_only: bool = False,
    sector: str | None = None,
    min_upside: float | None = None,
) -> dict:
    rows = v.board_rows()
    if buys_only:
        rows = [r for r in rows if r["recommendation"] in ("BUY", "ADD")]
    if sector:
        rows = [r for r in rows if sector.lower() in (r["sector"] or "").lower()]
    if min_upside is not None:
        rows = [r for r in rows if (r["upside_pct"] if r["upside_pct"] is not None else -1e9) >= min_upside]
    total = len(rows)
    page = rows[offset:] if limit == 0 else rows[offset : offset + limit]
    return {"total": total, "limit": limit, "offset": offset, "rows": page}


# --- The agent's radar (at/near attention price) -----------------------------
@app.get("/radar")
def radar(triggered_only: bool = False) -> dict:
    rows = rad.radar_rows()
    if triggered_only:
        rows = [r for r in rows if r["status"] == "TRIGGERED"]
    return {"total": len(rows), "rows": rows}


# --- One company (full record + its predictions) -----------------------------
@app.get("/companies/{ticker}")
def company(ticker: str) -> dict:
    rec = v.load(ticker)
    if rec is None:
        raise HTTPException(
            status_code=404,
            detail=f"No valuation for {ticker.upper()}. Trigger one with POST /valuate/{ticker.upper()}.",
        )
    a = rec.latest.assessment
    preds = [p for p in P.load_all() if (p.ticker or "").upper() == ticker.upper()]
    return {
        "record": rec,
        "opportunity_score": v.opportunity_score(a),
        "entry_status": a.entry_status(),
        "distance_to_entry_pct": a.distance_to_entry_pct(),
        "predictions": preds,
    }


# --- Live key metrics for one company (report panel) -------------------------
@app.get("/companies/{ticker}/metrics")
def company_metrics(ticker: str) -> dict:
    """Rich, freshly-fetched fundamentals for the company report (P/E, EV/EBITDA,
    FCF, growth, … trailing & forward). Fetched live, so it works for any name and
    can take a second. Returns an empty dict (not an error) when unavailable — the
    page degrades gracefully."""
    from ..data import get_providers

    prov = get_providers().fundamentals
    getter = getattr(prov, "metrics", None)
    if getter is None:
        return {"ticker": ticker.upper(), "metrics": {}}
    try:
        m = getter(ticker.upper()) or {}
    except Exception:
        m = {}
    return {"ticker": ticker.upper(), "metrics": m}


# --- Thesis search (full-text over the prose) --------------------------------
@app.get("/search")
def search(
    q: str = Query(..., min_length=1, description="Full-text query over the thesis prose."),
    limit: int = Query(50, ge=1, le=500),
) -> dict:
    recs = v.search(q, limit)
    return {"query": q, "total": len(recs), "results": [_summary(r) for r in recs]}


# --- Predictions + calibration -----------------------------------------------
@app.get("/predictions")
def predictions(
    status: str = Query("all", pattern="^(all|open|resolved)$"),
    ticker: str | None = None,
) -> dict:
    preds = P.load_all()
    if status in ("open", "resolved"):
        preds = [p for p in preds if p.status == status]
    if ticker:
        preds = [p for p in preds if (p.ticker or "").upper() == ticker.upper()]
    return {"total": len(preds), "predictions": preds}


@app.get("/calibration")
def calibration() -> dict:
    cal = P.compute_calibration()
    return {"summary": asdict(cal), "text": P.format_calibration(cal)}


# --- On-demand valuation trigger ---------------------------------------------
@app.post("/valuate/{ticker}", status_code=202)
def trigger_valuation(ticker: str) -> dict:
    """Kick off a fresh valuation of one name (works for tickers not yet in the
    corpus). Returns a job to poll at GET /jobs/{job_id}."""
    if not ticker.strip():
        raise HTTPException(status_code=400, detail="ticker required")
    job = start_valuation(ticker)
    return {"job_id": job.id, "ticker": job.ticker, "status": job.status}


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    out = job.to_dict()
    if job.status == "done":
        rec = v.load(job.ticker)
        out["result"] = _summary(rec) if rec else None
    return out


@app.get("/jobs")
def jobs() -> dict:
    return {"jobs": [j.to_dict() for j in all_jobs()]}


# --- Static frontend (served at /) -------------------------------------------
# The single-page UI lives next to this module. Mounted last so it never shadows
# the JSON API routes above.
from pathlib import Path as _Path  # noqa: E402

from fastapi.staticfiles import StaticFiles  # noqa: E402

_STATIC_DIR = _Path(__file__).parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="ui")
