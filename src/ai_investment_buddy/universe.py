"""The investable universe: S&P 500 + Nasdaq-100 (large) + S&P SmallCap 600.

Fetched from Wikipedia and cached to disk for a day. If the network fails we
fall back to the last cached copy so a daily run never hard-stops on a flaky
fetch.

The S&P 600 adds the small-cap sleeve — where thin analyst coverage means
mispricings persist longer (the neglected-firm effect), the best hunting ground
for a differentiated, calibrated view. We use the S&P 600 specifically (not the
raw Russell 2000) because its profitability inclusion screen filters out the
junkiest, most blow-up-prone names. Each company carries a ``cap_tier``
('large' | 'small') so the rest of the system can demand more margin of safety,
model realistic small-cap slippage, and separate size-factor beta from alpha.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from io import StringIO

import pandas as pd
import requests

from .config import CACHE_DIR, ensure_dirs

_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_SP600_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
_NDX_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"
_CACHE_FILE = CACHE_DIR / "universe.json"
_CACHE_TTL_HOURS = 24
_UA = "Mozilla/5.0 (compatible; ai-investment-buddy/0.1; research)"


def _read_html(url: str) -> list[pd.DataFrame]:
    resp = requests.get(url, headers={"User-Agent": _UA}, timeout=30)
    resp.raise_for_status()
    return pd.read_html(StringIO(resp.text))


def _normalize(ticker: str) -> str:
    # yfinance uses '-' where Wikipedia uses '.' (e.g. BRK.B -> BRK-B).
    return ticker.strip().upper().replace(".", "-")


def _fetch_sp_table(url: str, index_label: str, cap_tier: str) -> list[dict]:
    """Fetch an S&P constituents table from Wikipedia (S&P 500 and S&P 600 share
    the same column layout: Symbol, Security, GICS Sector, GICS Sub-Industry)."""
    tables = _read_html(url)
    # The constituents table is the first one with a Symbol/Ticker column.
    df = None
    for t in tables:
        if "Symbol" in t.columns or "Ticker" in t.columns:
            df = t
            break
    if df is None:
        df = tables[0]
    sym_col = "Symbol" if "Symbol" in df.columns else "Ticker"
    out = []
    for _, row in df.iterrows():
        out.append(
            {
                "ticker": _normalize(str(row[sym_col])),
                "name": str(row.get("Security", "")),
                "sector": str(row.get("GICS Sector", "")),
                # GICS Sub-Industry is the granular grouping (e.g. 'Semiconductors'
                # vs 'Application Software') — the level where the dispersion that
                # matters lives. The 11-sector view averages it all away.
                "sub_industry": str(row.get("GICS Sub-Industry", "")),
                "cap_tier": cap_tier,
                "indices": [index_label],
            }
        )
    return out


def _fetch_sp500() -> list[dict]:
    return _fetch_sp_table(_SP500_URL, "S&P 500", "large")


def _fetch_sp600() -> list[dict]:
    return _fetch_sp_table(_SP600_URL, "S&P 600", "small")


def _fetch_ndx() -> list[str]:
    tables = _read_html(_NDX_URL)
    for df in tables:
        cols = {str(c).lower() for c in df.columns}
        if "ticker" in cols or "symbol" in cols:
            col = "Ticker" if "Ticker" in df.columns else "Symbol"
            return [_normalize(str(t)) for t in df[col].tolist()]
    return []


def _build() -> dict:
    sp = _fetch_sp500()
    by_ticker = {c["ticker"]: c for c in sp}

    # Merge Nasdaq-100 membership in; add any Nasdaq names not already present
    # (Nasdaq-100 is large-cap by construction).
    ndx = set(_fetch_ndx())
    for t in ndx:
        if t in by_ticker:
            by_ticker[t]["indices"].append("Nasdaq 100")
        else:
            by_ticker[t] = {
                "ticker": t, "name": "", "sector": "", "sub_industry": "",
                "cap_tier": "large", "indices": ["Nasdaq 100"],
            }

    # Add the S&P SmallCap 600 sleeve. Best-effort: a fetch failure leaves the
    # large-cap universe intact rather than crashing the run.
    try:
        for c in _fetch_sp600():
            if c["ticker"] in by_ticker:
                by_ticker[c["ticker"]]["indices"].append("S&P 600")
            else:
                by_ticker[c["ticker"]] = c
    except Exception:
        pass

    return {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "companies": sorted(by_ticker.values(), key=lambda c: c["ticker"]),
    }


def _cache_is_fresh(payload: dict) -> bool:
    try:
        ts = datetime.fromisoformat(payload["fetched_at"])
        age = datetime.now(timezone.utc) - ts
        return age.total_seconds() < _CACHE_TTL_HOURS * 3600
    except Exception:
        return False


def get_universe(force_refresh: bool = False) -> list[dict]:
    """Return the list of {ticker, name, sector, indices} dicts."""
    ensure_dirs()
    if not force_refresh and _CACHE_FILE.exists():
        payload = json.loads(_CACHE_FILE.read_text())
        if _cache_is_fresh(payload):
            return payload["companies"]

    try:
        payload = _build()
        _CACHE_FILE.write_text(json.dumps(payload, indent=2))
        return payload["companies"]
    except Exception:
        # Network/parse failure: serve stale cache rather than crashing the run.
        if _CACHE_FILE.exists():
            return json.loads(_CACHE_FILE.read_text())["companies"]
        raise


def get_tickers(force_refresh: bool = False) -> list[str]:
    return [c["ticker"] for c in get_universe(force_refresh=force_refresh)]
