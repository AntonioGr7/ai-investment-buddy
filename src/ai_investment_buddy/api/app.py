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

from ..memory import db
from ..memory import predictions as P
from ..memory import radar as rad
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
