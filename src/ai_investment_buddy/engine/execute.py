"""Paper execution: turn target-weight orders into trades against live prices.

Enforces the risk guardrails regardless of what the model asked for:
  - clamp each target weight to max_position_weight,
  - no leverage (buys limited by available cash; sells settle first),
  - ignore sub-min_trade_value dust,
  - apply slippage against us on every fill.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..config import SETTINGS
from ..memory.portfolio import Portfolio
from ..models import Action, Decision, Trade


def _slip(price: float, side: Action) -> float:
    adj = SETTINGS.slippage_bps / 10_000.0
    return price * (1 + adj) if side == Action.BUY else price * (1 - adj)


def execute(
    portfolio: Portfolio, decision: Decision, prices: dict[str, float]
) -> list[Trade]:
    """Mutate ``portfolio`` to enact ``decision``; return the executed trades."""
    nav = portfolio.nav(prices)
    if nav <= 0:
        return []

    ts = datetime.now(timezone.utc)
    trades: list[Trade] = []

    # Compute desired value per ordered ticker (deltas vs current).
    sells: list[tuple] = []  # (ticker, delta_value, rationale)
    buys: list[tuple] = []  # (ticker, delta_value, conviction, rationale)

    for o in decision.orders:
        price = prices.get(o.ticker)
        if not price or price <= 0:
            continue

        if o.action == Action.HOLD:
            continue

        target_w = min(o.target_weight, SETTINGS.max_position_weight)
        if o.action == Action.SELL and o.target_weight <= 0:
            target_w = 0.0
        target_value = target_w * nav

        pos = portfolio.positions.get(o.ticker)
        current_value = pos.market_value(price) if pos else 0.0
        delta = target_value - current_value

        if abs(delta) < SETTINGS.min_trade_value:
            continue

        if delta < 0:
            sells.append((o.ticker, -delta, o.rationale))
        else:
            buys.append((o.ticker, delta, o.conviction, o.rationale))

    # 1) Sells first to free up cash.
    for ticker, value, rationale in sells:
        price = prices[ticker]
        fill = _slip(price, Action.SELL)
        shares = min(value / price, portfolio.positions[ticker].shares)
        if shares <= 0:
            continue
        proceeds = portfolio.apply_sell(ticker, shares, fill)
        trades.append(
            Trade(
                timestamp=ts,
                ticker=ticker,
                action=Action.SELL,
                shares=round(shares, 6),
                price=round(fill, 4),
                value=round(proceeds, 2),
                rationale=rationale,
            )
        )

    # 2) Buys, highest conviction first, each limited by remaining cash.
    buys.sort(key=lambda b: b[2], reverse=True)
    for ticker, value, _conv, rationale in buys:
        available = portfolio.cash - SETTINGS.cash_floor
        if available <= SETTINGS.min_trade_value:
            break
        price = prices[ticker]
        fill = _slip(price, Action.BUY)
        spend = min(value, available)
        shares = spend / fill
        if shares * fill < SETTINGS.min_trade_value:
            continue
        cost = portfolio.apply_buy(ticker, shares, fill)
        trades.append(
            Trade(
                timestamp=ts,
                ticker=ticker,
                action=Action.BUY,
                shares=round(shares, 6),
                price=round(fill, 4),
                value=round(cost, 2),
                rationale=rationale,
            )
        )

    return trades
