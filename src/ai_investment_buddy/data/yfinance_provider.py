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
        try:
            df = yf.Ticker(ticker).history(period="5d", auto_adjust=True)
            if df.empty:
                return None
            return float(df["Close"].iloc[-1])
        except Exception:
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
