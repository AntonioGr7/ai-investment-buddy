"""Persistence for the portfolio, trade ledger, and NAV history.

Plain files on disk so the whole experiment state is human-readable and easy to
inspect, diff, and back up:
  data/portfolio.json     current cash + positions
  data/trades.jsonl       append-only trade ledger
  data/nav_history.csv    one row per run: NAV + benchmark levels
"""

from __future__ import annotations

import csv
import json
from datetime import date, datetime, timezone

from ..config import DATA_DIR, JOURNAL_DIR, SETTINGS, ensure_dirs
from ..models import Trade
from .portfolio import Portfolio

_PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
_TRADES_FILE = DATA_DIR / "trades.jsonl"
_NAV_FILE = DATA_DIR / "nav_history.csv"

_NAV_FIELDS = ["date", "nav", "cash", "invested", "n_positions"]


def is_initialized() -> bool:
    return _PORTFOLIO_FILE.exists()


def init_portfolio(capital: float | None = None, force: bool = False) -> Portfolio:
    ensure_dirs()
    if is_initialized() and not force:
        raise FileExistsError(
            f"Portfolio already initialized at {_PORTFOLIO_FILE}. Use force=True to reset."
        )
    cap = capital if capital is not None else SETTINGS.starting_capital
    pf = Portfolio(cash=cap, positions={})
    save_portfolio(pf)
    # Reset ledgers AND memory on (re)init, so state stays consistent.
    _TRADES_FILE.write_text("")
    if _NAV_FILE.exists():
        _NAV_FILE.unlink()
    if JOURNAL_DIR.exists():
        for f in JOURNAL_DIR.glob("*.md"):
            f.unlink()
        theses = JOURNAL_DIR / "theses.json"
        if theses.exists():
            theses.unlink()
    return pf


def load_portfolio() -> Portfolio:
    if not is_initialized():
        raise FileNotFoundError(
            "No portfolio found. Run `aib init` first to seed the experiment."
        )
    return Portfolio.model_validate_json(_PORTFOLIO_FILE.read_text())


def save_portfolio(pf: Portfolio) -> None:
    ensure_dirs()
    _PORTFOLIO_FILE.write_text(pf.model_dump_json(indent=2))


# --- Trade ledger ------------------------------------------------------------
def append_trades(trades: list[Trade]) -> None:
    if not trades:
        return
    ensure_dirs()
    with _TRADES_FILE.open("a") as f:
        for t in trades:
            f.write(t.model_dump_json() + "\n")


def load_trades() -> list[Trade]:
    if not _TRADES_FILE.exists():
        return []
    out = []
    for line in _TRADES_FILE.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(Trade.model_validate_json(line))
    return out


# --- NAV history -------------------------------------------------------------
def append_nav(
    as_of: date,
    nav: float,
    cash: float,
    invested: float,
    n_positions: int,
    benchmarks: dict[str, float] | None = None,
) -> None:
    """Append a NAV row. Benchmark index levels are added as extra columns so we
    can compute relative performance from inception later."""
    ensure_dirs()
    benchmarks = benchmarks or {}
    fields = _NAV_FIELDS + sorted(benchmarks.keys())
    row = {
        "date": as_of.isoformat(),
        "nav": round(nav, 2),
        "cash": round(cash, 2),
        "invested": round(invested, 2),
        "n_positions": n_positions,
        **{k: round(v, 4) for k, v in benchmarks.items()},
    }

    write_header = not _NAV_FILE.exists()
    # If schema widened (new benchmark column), rewrite with the union header.
    existing_rows = []
    if not write_header:
        with _NAV_FILE.open() as f:
            existing_rows = list(csv.DictReader(f))
        existing_fields = list(existing_rows[0].keys()) if existing_rows else []
        union = list(dict.fromkeys(existing_fields + fields))
        if union != existing_fields:
            fields = union
            write_header = True  # trigger full rewrite below

    if write_header and existing_rows:
        with _NAV_FILE.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in existing_rows:
                w.writerow(r)
            w.writerow(row)
    else:
        with _NAV_FILE.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                w.writeheader()
            w.writerow(row)


def load_nav_history() -> list[dict]:
    if not _NAV_FILE.exists():
        return []
    with _NAV_FILE.open() as f:
        return list(csv.DictReader(f))


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
