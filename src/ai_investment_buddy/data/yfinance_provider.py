"""Free-tier data layer backed by yfinance.

Notes on cost/scale: the *only* whole-universe call is ``history()`` (one bulk
download). Fundamentals and news are per-ticker and therefore only called for
the screener's shortlist plus current holdings.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from ..models import MacroSnapshot

# Macro instruments we sample each run. Mapping label -> yfinance symbol.
_MACRO_SYMBOLS = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Dow Jones": "^DJI",
    "Russell 2000": "^RUT",
    "VIX (volatility)": "^VIX",
    "US 10Y yield": "^TNX",
    "US 13W yield": "^IRX",
    "US Dollar Index": "DX-Y.NYB",
    "Gold": "GC=F",
    "Crude Oil (WTI)": "CL=F",
    "Bitcoin": "BTC-USD",
}


class YFinancePrices:
    def latest_price(self, ticker: str) -> float | None:
        """Freshest valid price for one ticker.

        The daily bar for the current/most-recent session often has a NaN close
        (volume present, close not yet settled), so we must NOT blindly take the
        last daily row — that returns NaN or silently falls back to a day-old
        close. We prefer a fresh intraday print, then the last *valid* daily
        close, and only return a real positive number."""
        t = yf.Ticker(ticker)
        # 1) Fresh intraday last trade (captures today's move).
        try:
            intra = t.history(period="1d", interval="1m", auto_adjust=True)
            if not intra.empty:
                c = intra["Close"].dropna()
                if len(c):
                    px = float(c.iloc[-1])
                    if px == px and px > 0:  # not NaN, positive
                        return px
        except Exception:
            pass
        # 2) Last *valid* daily close.
        try:
            daily = t.history(period="7d", auto_adjust=True)
            if not daily.empty:
                c = daily["Close"].dropna()
                if len(c):
                    px = float(c.iloc[-1])
                    if px == px and px > 0:
                        return px
        except Exception:
            pass
        return None

    def history(
        self, tickers: list[str], lookback_days: int = 260
    ) -> dict[str, pd.DataFrame]:
        if not tickers:
            return {}
        # Buffer trading-day count to calendar days.
        period_days = int(lookback_days * 1.6) + 10
        raw = yf.download(
            tickers,
            period=f"{period_days}d",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False,
        )
        out: dict[str, pd.DataFrame] = {}
        if raw is None or raw.empty:
            return out

        if isinstance(raw.columns, pd.MultiIndex):
            for t in tickers:
                if t in raw.columns.get_level_values(0):
                    sub = raw[t].dropna(how="all")
                    if not sub.empty:
                        out[t] = sub
        else:
            # Single ticker: flat columns.
            sub = raw.dropna(how="all")
            if not sub.empty:
                out[tickers[0]] = sub
        return out


class YFinanceFundamentals:
    def fundamentals(self, ticker: str) -> dict:
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception:
            return {}
        return {
            "name": info.get("shortName") or info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
            "pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "peg": info.get("pegRatio") or info.get("trailingPegRatio"),
            "ps": info.get("priceToSalesTrailing12Months"),
            "profit_margin": info.get("profitMargins"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "debt_to_equity": info.get("debtToEquity"),
            "free_cashflow": info.get("freeCashflow"),
            "target_mean_price": info.get("targetMeanPrice"),
            "recommendation": info.get("recommendationKey"),
            "prev_close": info.get("previousClose"),
        }

    def metrics(self, ticker: str) -> dict:
        """A RICH, display-oriented metric set for the company report (trailing +
        forward where available). Raw numbers; the frontend formats/labels them.

        Units: margins/growth are fractions (0.25 = 25%); debt_to_equity is yfinance's
        percentage figure converted to a ratio; large $ figures are absolute. Missing
        values are None. Best-effort — a bad/illiquid symbol just yields {}."""
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception:
            return {}

        def num(*keys):
            for k in keys:
                v = info.get(k)
                if isinstance(v, (int, float)) and v == v:  # not NaN
                    return float(v)
            return None

        mc = num("marketCap")
        ev = num("enterpriseValue")
        fcf = num("freeCashflow")
        rev = num("totalRevenue")
        de = num("debtToEquity")  # yfinance reports as a percentage (165.6 → 1.66x)

        def ratio(a, b):
            return round(a / b, 2) if a is not None and b not in (None, 0) else None

        return {
            "currency": info.get("currency") or "USD",
            # Size
            "market_cap": mc,
            "enterprise_value": ev,
            # Valuation multiples (trailing / forward)
            "pe": num("trailingPE"),
            "forward_pe": num("forwardPE"),
            "peg": num("pegRatio", "trailingPegRatio"),
            "ps": num("priceToSalesTrailing12Months"),
            "pb": num("priceToBook"),
            "p_fcf": ratio(mc, fcf),
            "ev_ebitda": num("enterpriseToEbitda"),
            "ev_sales": num("enterpriseToRevenue") or ratio(ev, rev),
            "ev_fcf": ratio(ev, fcf),
            # Per share
            "eps": num("trailingEps"),
            "forward_eps": num("forwardEps"),
            # Profitability ($ and margins)
            "net_income": num("netIncomeToCommon"),
            "free_cashflow": fcf,
            "ebitda": num("ebitda"),
            "revenue": rev,
            "gross_margin": num("grossMargins"),
            "operating_margin": num("operatingMargins"),
            "profit_margin": num("profitMargins"),
            "roe": num("returnOnEquity"),
            "roa": num("returnOnAssets"),
            # Growth (fractions, YoY)
            "revenue_growth": num("revenueGrowth"),
            "earnings_growth": num("earningsGrowth", "earningsQuarterlyGrowth"),
            # Balance sheet / risk
            "debt_to_equity": round(de / 100.0, 2) if de is not None else None,
            "current_ratio": num("currentRatio"),
            "beta": num("beta"),
            # Yield
            "dividend_yield": num("dividendYield"),
            "fcf_yield": (round(fcf / mc, 4) if fcf is not None and mc else None),
            # Range / target
            "fifty_two_week_high": num("fiftyTwoWeekHigh"),
            "fifty_two_week_low": num("fiftyTwoWeekLow"),
            "target_mean_price": num("targetMeanPrice"),
        }


class YFinanceNews:
    def headlines(self, ticker: str, limit: int = 5) -> list[str]:
        try:
            items = yf.Ticker(ticker).news or []
        except Exception:
            return []
        titles: list[str] = []
        for it in items:
            # yfinance has shifted the news schema over versions; handle both.
            content = it.get("content", it)
            title = content.get("title") or it.get("title")
            if title:
                titles.append(str(title).strip())
            if len(titles) >= limit:
                break
        return titles


class YFinanceMacro:
    def snapshot(self) -> MacroSnapshot:
        indicators: dict[str, float] = {}
        notes: list[str] = []
        symbols = list(_MACRO_SYMBOLS.values())
        try:
            raw = yf.download(
                symbols,
                period="10d",
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
        except Exception:
            raw = None

        for label, sym in _MACRO_SYMBOLS.items():
            try:
                if raw is not None and isinstance(raw.columns, pd.MultiIndex):
                    closes = raw[sym]["Close"].dropna()
                else:
                    closes = pd.Series(dtype=float)
                if len(closes) >= 1:
                    last = float(closes.iloc[-1])
                    indicators[label] = round(last, 2)
                    if len(closes) >= 2:
                        prev = float(closes.iloc[-2])
                        chg = (last / prev - 1) * 100 if prev else 0.0
                        indicators[f"{label} 1d %"] = round(chg, 2)
                    if len(closes) >= 6:
                        wk = float(closes.iloc[-6])
                        chg5 = (last / wk - 1) * 100 if wk else 0.0
                        indicators[f"{label} 5d %"] = round(chg5, 2)
            except Exception:
                continue

        return MacroSnapshot(
            as_of=datetime.now(timezone.utc),
            indicators=indicators,
            notes=notes,
        )
