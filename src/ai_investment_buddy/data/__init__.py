"""Swappable data layer.

The rest of the system depends only on the Protocols in ``base`` and obtains
concrete implementations through ``get_providers()``. To add a paid provider
(Polygon, FMP, NewsAPI, FRED, ...), implement the relevant Protocol and wire it
into the factory below — no caller changes required.
"""

from __future__ import annotations

from functools import lru_cache

from ..config import SETTINGS
from .base import (
    FundamentalsProvider,
    MacroProvider,
    MarketNewsProvider,
    NewsProvider,
    PriceProvider,
)


@lru_cache(maxsize=1)
def get_providers() -> "Providers":
    from .market_news import RSSMarketNews
    from .yfinance_provider import (
        YFinanceFundamentals,
        YFinanceMacro,
        YFinanceNews,
        YFinancePrices,
    )

    registry = {
        "price": {"yfinance": YFinancePrices},
        "fundamentals": {"yfinance": YFinanceFundamentals},
        "news": {"yfinance": YFinanceNews},
        "macro": {"yfinance": YFinanceMacro},
        "market_news": {"rss": RSSMarketNews},
    }

    def pick(kind: str, name: str):
        try:
            return registry[kind][name]()
        except KeyError as exc:
            raise ValueError(
                f"No {kind} provider registered under '{name}'."
            ) from exc

    return Providers(
        prices=pick("price", SETTINGS.price_provider),
        fundamentals=pick("fundamentals", SETTINGS.fundamentals_provider),
        news=pick("news", SETTINGS.news_provider),
        macro=pick("macro", SETTINGS.macro_provider),
        market_news=pick("market_news", SETTINGS.market_news_provider),
    )


class Providers:
    def __init__(
        self,
        prices: PriceProvider,
        fundamentals: FundamentalsProvider,
        news: NewsProvider,
        macro: MacroProvider,
        market_news: MarketNewsProvider,
    ):
        self.prices = prices
        self.fundamentals = fundamentals
        self.news = news
        self.macro = macro
        self.market_news = market_news
