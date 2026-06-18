"""Provider interfaces.

Implementations live alongside (yfinance_provider.py); paid providers can be
added without touching any caller as long as they satisfy these Protocols.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from ..models import MacroSnapshot


@runtime_checkable
class PriceProvider(Protocol):
    def latest_price(self, ticker: str) -> float | None:
        """Most recent trade price for a single ticker (None if unavailable)."""
        ...

    def history(
        self, tickers: list[str], lookback_days: int = 260
    ) -> dict[str, pd.DataFrame]:
        """Bulk daily OHLCV history keyed by ticker. Used by the screener."""
        ...


@runtime_checkable
class FundamentalsProvider(Protocol):
    def fundamentals(self, ticker: str) -> dict:
        """Best-effort fundamentals for one company (keys map onto TickerData)."""
        ...


@runtime_checkable
class NewsProvider(Protocol):
    def headlines(self, ticker: str, limit: int = 5) -> list[str]:
        """Recent headlines for one company."""
        ...


@runtime_checkable
class MacroProvider(Protocol):
    def snapshot(self) -> MacroSnapshot:
        """Top-down market context: indices, rates, vol, commodities, FX."""
        ...


@runtime_checkable
class MarketNewsProvider(Protocol):
    def market_digest(self, days: int = 3, per_feed: int = 5) -> list[dict]:
        """Recent market-wide & macro/Fed headlines (newest first)."""
        ...
