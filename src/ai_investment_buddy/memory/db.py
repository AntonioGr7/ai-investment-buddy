"""SQLite index for the company corpus — the queryable, concurrency-safe read
layer behind the valuation and prediction stores.

Why this exists: a frontend (and UI-triggered valuations) needs to filter, sort,
paginate and full-text-search across everything we've ever valued, while a daily
run may be writing at the same time. Globbing hundreds of JSON files can't do
that safely. So the per-name ``valuations/<TICKER>.json`` files and the
``predictions.jsonl`` ledger remain the durable, git-diffable, snapshot-friendly
record — and THIS database is an always-synchronized index over them that serves
reads. Every write dual-writes both (see valuations.py / predictions.py), so the
DB is fully REBUILDABLE from the files and is never a single point of data loss:
``rebuild_from_files()`` (and ``aib db rebuild``) regenerate it from scratch.

WAL mode lets a web process read while a run writes. The DB lives at
``<DATA_DIR>/aib.db`` so it travels per-experiment like the rest of the state.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ..config import DATA_DIR, ensure_dirs
from ..models import Prediction, ValuationRecord

# Core schema (always available). FTS is created separately so a sqlite build
# without FTS5 degrades to "search returns nothing" rather than failing outright.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS valuations (
    ticker            TEXT PRIMARY KEY,
    first_assessed    TEXT NOT NULL,
    last_assessed     TEXT NOT NULL,
    sector            TEXT,
    archetype         TEXT,
    recommendation    TEXT,
    market_view       TEXT,
    valuation_verdict TEXT,
    structural_risk   TEXT,
    fair_value        REAL,
    current_price     REAL,
    entry_price       REAL,
    upside_pct        REAL,
    downside_pct      REAL,
    risk_reward       REAL,
    quality_score     INTEGER,
    margin_of_safety  INTEGER,
    confidence        INTEGER,
    rerating_catalyst TEXT,
    regime            TEXT,
    score             REAL,
    record_json       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_val_score ON valuations(score DESC);
CREATE INDEX IF NOT EXISTS idx_val_rec ON valuations(recommendation);
CREATE INDEX IF NOT EXISTS idx_val_sector ON valuations(sector);

CREATE TABLE IF NOT EXISTS predictions (
    id              TEXT PRIMARY KEY,
    ticker          TEXT,
    created         TEXT,
    horizon_date    TEXT,
    status          TEXT,
    category        TEXT,
    probability     REAL,
    market_implied  REAL,
    outcome         INTEGER,
    brier           REAL,
    resolved_on     TEXT,
    pred_json       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pred_status ON predictions(status);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS valuations_fts USING fts5(
    ticker UNINDEXED, sector, archetype, recommendation,
    bull_case, bear_case, mispricing_thesis, news_assessment,
    why_market_disagrees, key_risks, rerating_catalyst, valuation_method
);
"""

# Columns mirrored from the latest assessment for querying/sorting.
_VAL_COLUMNS = (
    "ticker", "first_assessed", "last_assessed", "sector", "archetype",
    "recommendation", "market_view", "valuation_verdict", "structural_risk",
    "fair_value", "current_price", "entry_price", "upside_pct", "downside_pct",
    "risk_reward", "quality_score", "margin_of_safety", "confidence",
    "rerating_catalyst", "regime", "score", "record_json",
)

_conns: dict[str, sqlite3.Connection] = {}
_has_fts: dict[str, bool] = {}


def _db_path() -> Path:
    return DATA_DIR / "aib.db"


def connect() -> sqlite3.Connection:
    """Return the (cached) connection for the current DATA_DIR, initializing the
    schema and importing any pre-existing files the first time."""
    ensure_dirs()
    key = str(_db_path())
    conn = _conns.get(key)
    if conn is not None:
        return conn
    conn = sqlite3.connect(key, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_SCHEMA)
    try:
        conn.executescript(_FTS_SCHEMA)
        _has_fts[key] = True
    except sqlite3.OperationalError:
        _has_fts[key] = False  # FTS5 not compiled in — search degrades gracefully
    conn.commit()
    _conns[key] = conn
    _maybe_import_legacy(conn)
    return conn


def _reset_for_tests() -> None:
    """Drop cached connections (so a changed DATA_DIR is picked up)."""
    for c in _conns.values():
        try:
            c.close()
        except Exception:
            pass
    _conns.clear()
    _has_fts.clear()


# --- Valuations --------------------------------------------------------------
def _val_row(rec: ValuationRecord, regime: str, score: float,
             entry_price: float | None) -> tuple:
    a = rec.latest.assessment
    return (
        rec.ticker.upper(),
        rec.first_assessed.isoformat(),
        rec.last_assessed.isoformat(),
        a.sector or None, a.archetype or None, a.recommendation,
        a.market_view or None, a.valuation_verdict or None,
        a.structural_risk or None,
        a.fair_value, a.current_price, entry_price,
        a.upside_pct, a.downside_pct, a.risk_reward,
        a.quality_score, 1 if a.margin_of_safety else 0, a.confidence,
        a.rerating_catalyst or None, regime or None, score,
        rec.model_dump_json(),
    )


def upsert_valuation(rec: ValuationRecord, regime: str, score: float,
                     entry_price: float | None) -> None:
    """Insert/replace one name's row (and its FTS entry). ``score`` and
    ``entry_price`` are computed by the caller (valuations.py owns that logic)."""
    conn = connect()
    placeholders = ",".join("?" * len(_VAL_COLUMNS))
    conn.execute(
        f"INSERT OR REPLACE INTO valuations ({','.join(_VAL_COLUMNS)}) "
        f"VALUES ({placeholders})",
        _val_row(rec, regime, score, entry_price),
    )
    _index_fts(conn, rec)
    conn.commit()


def _index_fts(conn: sqlite3.Connection, rec: ValuationRecord) -> None:
    if not _has_fts.get(str(_db_path())):
        return
    a = rec.latest.assessment
    t = rec.ticker.upper()
    conn.execute("DELETE FROM valuations_fts WHERE ticker = ?", (t,))
    conn.execute(
        "INSERT INTO valuations_fts (ticker, sector, archetype, recommendation, "
        "bull_case, bear_case, mispricing_thesis, news_assessment, "
        "why_market_disagrees, key_risks, rerating_catalyst, valuation_method) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (t, a.sector or "", a.archetype or "", a.recommendation,
         a.bull_case, a.bear_case, a.mispricing_thesis, a.news_assessment,
         a.why_market_disagrees, a.key_risks, a.rerating_catalyst, a.valuation_method),
    )


def get_valuation(ticker: str) -> ValuationRecord | None:
    conn = connect()
    row = conn.execute(
        "SELECT record_json FROM valuations WHERE ticker = ?", (ticker.upper(),)
    ).fetchone()
    if row is None:
        return None
    try:
        return ValuationRecord.model_validate_json(row["record_json"])
    except Exception:
        return None


def all_valuations() -> list[ValuationRecord]:
    conn = connect()
    out: list[ValuationRecord] = []
    for row in conn.execute("SELECT record_json FROM valuations ORDER BY ticker"):
        try:
            out.append(ValuationRecord.model_validate_json(row["record_json"]))
        except Exception:
            continue
    return out


def search_valuations(query: str, limit: int = 50) -> list[str]:
    """Full-text search over the thesis prose. Returns matching tickers, best
    first. Empty if FTS is unavailable or the query is blank."""
    if not query or not query.strip():
        return []
    conn = connect()  # populates _has_fts for this process
    if not _has_fts.get(str(_db_path())):
        return []
    try:
        rows = conn.execute(
            "SELECT ticker FROM valuations_fts WHERE valuations_fts MATCH ? "
            "ORDER BY rank LIMIT ?",
            (query, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [r["ticker"] for r in rows]


# --- Predictions -------------------------------------------------------------
def _pred_row(p: Prediction) -> tuple:
    return (
        p.id, p.ticker or None, p.created.isoformat(), p.horizon_date.isoformat(),
        p.status, p.category, p.probability, p.market_implied,
        (None if p.outcome is None else (1 if p.outcome else 0)),
        p.brier, (p.resolved_on.isoformat() if p.resolved_on else None),
        p.model_dump_json(),
    )


_PRED_COLUMNS = (
    "id", "ticker", "created", "horizon_date", "status", "category",
    "probability", "market_implied", "outcome", "brier", "resolved_on", "pred_json",
)


def replace_predictions(preds: list[Prediction]) -> None:
    """Make the predictions table exactly mirror ``preds`` (the JSONL is the
    durable copy; this keeps the index in lockstep on every rewrite)."""
    conn = connect()
    placeholders = ",".join("?" * len(_PRED_COLUMNS))
    with conn:  # transaction
        conn.execute("DELETE FROM predictions")
        conn.executemany(
            f"INSERT INTO predictions ({','.join(_PRED_COLUMNS)}) "
            f"VALUES ({placeholders})",
            [_pred_row(p) for p in preds],
        )


def all_predictions() -> list[Prediction]:
    conn = connect()
    out: list[Prediction] = []
    for row in conn.execute("SELECT pred_json FROM predictions"):
        try:
            out.append(Prediction.model_validate_json(row["pred_json"]))
        except Exception:
            continue
    return out


# --- Migration / rebuild -----------------------------------------------------
def _maybe_import_legacy(conn: sqlite3.Connection) -> None:
    """One-time seed: if the DB is empty but file-based state already exists
    (an upgrade from the pre-SQLite layout), import it so existing coverage is
    immediately visible. Only ever fills an empty table — never overwrites."""
    n_val = conn.execute("SELECT COUNT(*) FROM valuations").fetchone()[0]
    n_pred = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    if n_val == 0 and (DATA_DIR / "valuations").exists():
        _import_valuation_files(conn)
    if n_pred == 0 and (DATA_DIR / "predictions.jsonl").exists():
        _import_prediction_file(conn)


def _import_valuation_files(conn: sqlite3.Connection) -> int:
    # Imported lazily to avoid an import cycle (valuations imports this module).
    from . import valuations as _v

    vdir = DATA_DIR / "valuations"
    n = 0
    for p in sorted(vdir.glob("*.json")):
        try:
            rec = ValuationRecord.model_validate_json(p.read_text())
        except Exception:
            continue
        a = rec.latest.assessment
        entry = a.entry_price if a.entry_price is not None else _v.derive_entry_price(a)
        conn.execute(
            f"INSERT OR REPLACE INTO valuations ({','.join(_VAL_COLUMNS)}) "
            f"VALUES ({','.join('?' * len(_VAL_COLUMNS))})",
            _val_row(rec, rec.latest.regime, _v.opportunity_score(a), entry),
        )
        _index_fts(conn, rec)
        n += 1
    conn.commit()
    return n


def _import_prediction_file(conn: sqlite3.Connection) -> int:
    preds: list[Prediction] = []
    for line in (DATA_DIR / "predictions.jsonl").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            preds.append(Prediction.model_validate_json(line))
        except Exception:
            continue
    if preds:
        replace_predictions(preds)
    return len(preds)


def rebuild_from_files() -> dict:
    """Drop and regenerate the whole index from the JSON/JSONL files. The files
    are authoritative; this is the recovery/migration path (``aib db rebuild``)."""
    conn = connect()
    with conn:
        conn.execute("DELETE FROM valuations")
        conn.execute("DELETE FROM predictions")
        if _has_fts.get(str(_db_path())):
            conn.execute("DELETE FROM valuations_fts")
    n_val = _import_valuation_files(conn) if (DATA_DIR / "valuations").exists() else 0
    n_pred = _import_prediction_file(conn) if (DATA_DIR / "predictions.jsonl").exists() else 0
    return {"valuations": n_val, "predictions": n_pred, "path": str(_db_path())}
