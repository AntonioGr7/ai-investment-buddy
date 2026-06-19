"""Finnhub data layer (free tier: 60 req/min, ~300/day).

We use Finnhub for the high-value, per-name data where it clearly beats yfinance:
real **company news** (full headlines + summaries, the input to our news/sentiment
analysis) and richer **fundamentals**. Bulk price history stays on yfinance — one
500-ticker download would blow the daily request budget.

A single shared client enforces the rate limit across providers (news + fundamentals
count against the same 60/min), retries transient errors, and degrades to empty
results if the key is missing or a call fails — so a Finnhub hiccup never breaks a run.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import date, timedelta
from functools import lru_cache

import requests

from ..config import SETTINGS

_BASE = "https://finnhub.io/api/v1"


def _sym(ticker: str) -> str:
    # Our tickers are yfinance-style (BRK-B); Finnhub wants dots (BRK.B).
    return ticker.strip().upper().replace("-", ".")


class _RateLimiter:
    """Sliding-window limiter, thread-safe (the analyst hits this from a pool)."""

    def __init__(self, max_calls: int = 55, period: float = 60.0) -> None:
        self.max = max_calls
        self.period = period
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            while self._calls and now - self._calls[0] > self.period:
                self._calls.popleft()
            if len(self._calls) >= self.max:
                sleep_for = self.period - (now - self._calls[0]) + 0.05
                if sleep_for > 0:
                    time.sleep(sleep_for)
                now = time.monotonic()
                while self._calls and now - self._calls[0] > self.period:
                    self._calls.popleft()
            self._calls.append(time.monotonic())


# Once rate-limited/quota-exhausted, stop hitting Finnhub for this long and let
# callers fall back to yfinance — no point hammering a dead quota mid-run.
_EXHAUSTION_COOLDOWN = 300.0


class FinnhubClient:
    def __init__(self, api_key: str | None) -> None:
        self.api_key = api_key
        self._limiter = _RateLimiter()
        self._session = requests.Session()
        self._exhausted_until = 0.0
        self._exhausted_lock = threading.Lock()

    def is_exhausted(self) -> bool:
        return time.monotonic() < self._exhausted_until

    def _mark_exhausted(self) -> None:
        with self._exhausted_lock:
            self._exhausted_until = time.monotonic() + _EXHAUSTION_COOLDOWN

    def get(self, path: str, params: dict | None = None):
        """Return parsed JSON, or None on failure/exhaustion (caller falls back)."""
        if not self.api_key or self.is_exhausted():
            return None
        params = dict(params or {})
        params["token"] = self.api_key
        rate_limited = False
        for attempt in range(3):
            self._limiter.acquire()
            try:
                r = self._session.get(_BASE + path, params=params, timeout=20)
            except Exception:
                time.sleep(0.5 * (attempt + 1))
                continue
            if r.status_code == 429:  # rate limited — back off and retry
                rate_limited = True
                time.sleep(2.0 * (attempt + 1))
                continue
            if r.status_code >= 500:
                time.sleep(1.0)
                continue
            if r.status_code != 200:
                return None
            try:
                return r.json()
            except Exception:
                return None
        if rate_limited:
            # Out of calls — back off Finnhub for the rest of the run.
            self._mark_exhausted()
        return None


@lru_cache(maxsize=1)
def _client() -> FinnhubClient:
    """One shared client (and rate limiter) for all Finnhub providers."""
    return FinnhubClient(SETTINGS.finnhub_api_key)


def _pct(x):
    return x / 100.0 if isinstance(x, (int, float)) else None


def _rec_label(row: dict) -> str | None:
    """Collapse Finnhub's recommendation buckets into a yfinance-style key."""
    if not isinstance(row, dict):
        return None
    buckets = {
        "strong_buy": row.get("strongBuy", 0) or 0,
        "buy": row.get("buy", 0) or 0,
        "hold": row.get("hold", 0) or 0,
        "sell": row.get("sell", 0) or 0,
        "strong_sell": row.get("strongSell", 0) or 0,
    }
    if not any(buckets.values()):
        return None
    return max(buckets, key=buckets.get)


@lru_cache(maxsize=1)
def _yf_fundamentals():
    from .yfinance_provider import YFinanceFundamentals
    return YFinanceFundamentals()


@lru_cache(maxsize=1)
def _yf_news():
    from .yfinance_provider import YFinanceNews
    return YFinanceNews()


class FinnhubFundamentals:
    def fundamentals(self, ticker: str) -> dict:
        c = _client()
        sym = _sym(ticker)
        prof = c.get("/stock/profile2", {"symbol": sym})
        metric_raw = c.get("/stock/metric", {"symbol": sym, "metric": "all"})

        # The valuable block is `metric`; if it's unavailable (call failed or we're
        # rate-limited/out of quota), fall back to yfinance for a complete dict.
        metric = (metric_raw or {}).get("metric") if isinstance(metric_raw, dict) else None
        if not metric:  # None or empty → unavailable/exhausted → use yfinance
            return _yf_fundamentals().fundamentals(ticker)
        prof = prof or {}

        out: dict = {}
        if prof.get("name"):
            out["name"] = prof["name"]
        if prof.get("finnhubIndustry"):
            out["industry"] = prof["finnhubIndustry"]
        mc = prof.get("marketCapitalization")  # Finnhub reports in millions
        if isinstance(mc, (int, float)) and mc > 0:
            out["market_cap"] = mc * 1e6

        pe = metric.get("peTTM") or metric.get("peBasicExclExtraTTM") or metric.get("peNormalizedAnnual")
        if pe is not None:
            out["pe"] = pe
        fwd = metric.get("forwardPE") or metric.get("peNTM")
        if fwd is not None:
            out["forward_pe"] = fwd
        ps = metric.get("psTTM") or metric.get("psAnnual")
        if ps is not None:
            out["ps"] = ps
        pm = _pct(metric.get("netProfitMarginTTM"))
        if pm is not None:
            out["profit_margin"] = pm
        rg = _pct(metric.get("revenueGrowthTTMYoy"))
        if rg is not None:
            out["revenue_growth"] = rg
        eg = _pct(metric.get("epsGrowthTTMYoy"))
        if eg is not None:
            out["earnings_growth"] = eg
        de = metric.get("totalDebt/totalEquityAnnual") or metric.get("longTermDebt/equityAnnual")
        if de is not None:
            out["debt_to_equity"] = de
        fcf = metric.get("freeCashFlowTTM") or metric.get("freeCashFlowAnnual")
        if isinstance(fcf, (int, float)):
            out["free_cashflow"] = fcf * 1e6  # millions → absolute

        rec = c.get("/stock/recommendation", {"symbol": sym})
        if isinstance(rec, list) and rec:
            label = _rec_label(rec[0])
            if label:
                out["recommendation"] = label
        return out


class FinnhubNews:
    def headlines(self, ticker: str, limit: int = 5) -> list[str]:
        c = _client()
        today = date.today()
        frm = today - timedelta(days=14)
        items = c.get(
            "/company-news",
            {"symbol": _sym(ticker), "from": frm.isoformat(), "to": today.isoformat()},
        )
        out: list[str] = []
        if isinstance(items, list):
            for it in items:  # Finnhub returns newest first
                headline = (it.get("headline") or "").strip()
                if not headline:
                    continue
                summary = (it.get("summary") or "").strip()
                line = headline
                if summary and summary.lower() != headline.lower():
                    line += f" — {summary[:200]}"
                out.append(line)
                if len(out) >= limit:
                    break
        if out:
            return out
        # Finnhub failed, exhausted, or had nothing → fall back to yfinance.
        return _yf_news().headlines(ticker, limit)
