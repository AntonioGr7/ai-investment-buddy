"""The Portfolio: cash + positions, with valuation and trade application.

Pure in-memory logic; persistence lives in ``store``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..models import Action, Position


class Portfolio(BaseModel):
    cash: float
    positions: dict[str, Position] = Field(default_factory=dict)

    # --- Valuation ---
    def invested_value(self, prices: dict[str, float]) -> float:
        total = 0.0
        for t, pos in self.positions.items():
            px = prices.get(t)
            if px is not None:
                total += pos.market_value(px)
        return total

    def nav(self, prices: dict[str, float]) -> float:
        """Net asset value = cash + marked-to-market positions."""
        return self.cash + self.invested_value(prices)

    def weights(self, prices: dict[str, float]) -> dict[str, float]:
        nav = self.nav(prices)
        if nav <= 0:
            return {}
        w = {}
        for t, pos in self.positions.items():
            px = prices.get(t)
            if px is not None:
                w[t] = pos.market_value(px) / nav
        return w

    def cash_weight(self, prices: dict[str, float]) -> float:
        nav = self.nav(prices)
        return self.cash / nav if nav > 0 else 1.0

    # --- Mutation ---
    def apply_buy(self, ticker: str, shares: float, price: float) -> float:
        """Add shares at ``price``; returns cash spent. Updates avg cost basis."""
        cost = shares * price
        pos = self.positions.get(ticker)
        if pos is None:
            self.positions[ticker] = Position(
                ticker=ticker, shares=shares, avg_cost=price
            )
        else:
            new_shares = pos.shares + shares
            pos.avg_cost = (pos.avg_cost * pos.shares + cost) / new_shares
            pos.shares = new_shares
        self.cash -= cost
        return cost

    def apply_sell(self, ticker: str, shares: float, price: float) -> float:
        """Reduce/close a position; returns cash received."""
        pos = self.positions.get(ticker)
        if pos is None:
            return 0.0
        shares = min(shares, pos.shares)
        proceeds = shares * price
        pos.shares -= shares
        self.cash += proceeds
        if pos.shares <= 1e-9:
            del self.positions[ticker]
        return proceeds
