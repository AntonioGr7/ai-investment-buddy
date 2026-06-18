"""The quant funnel.

We cannot feed fundamentals + news for ~500 companies to the model every day —
it would be slow and expensive. The screener computes cheap technicals across
the whole universe from one bulk price download, then surfaces a diverse
shortlist of *candidates worth a deep look*:

  - momentum leaders (strong, trending names),
  - the day's biggest movers (news-driven situations, up and down),

plus the portfolio's current holdings are always carried through so the AI can
reconsider what it already owns.

Fundamentals and news are attached only to this shortlist (see ``enrich``).
"""

from __future__ import annotations

import math

import pandas as pd

from ..models import TickerData


def _pct_return(closes: pd.Series, lookback: int) -> float | None:
    if len(closes) <= lookback:
        return None
    past = closes.iloc[-1 - lookback]
    last = closes.iloc[-1]
    if past and not math.isnan(past):
        return round((last / past - 1) * 100, 2)
    return None


def compute_metrics(
    history: dict[str, pd.DataFrame], meta: dict[str, dict]
) -> dict[str, TickerData]:
    """Build TickerData with technicals for every ticker we have history for."""
    out: dict[str, TickerData] = {}
    for ticker, df in history.items():
        if df is None or df.empty or "Close" not in df:
            continue
        closes = df["Close"].dropna()
        if len(closes) < 30:
            continue
        last = float(closes.iloc[-1])
        prev = float(closes.iloc[-2]) if len(closes) >= 2 else last

        vol = df["Volume"].dropna() if "Volume" in df else pd.Series(dtype=float)
        avg_vol = float(vol.iloc[-21:].mean()) if len(vol) >= 21 else None
        last_vol = float(vol.iloc[-1]) if len(vol) else None
        vol_ratio = (
            round(last_vol / avg_vol, 2)
            if avg_vol and last_vol and avg_vol > 0
            else None
        )

        dma50 = float(closes.iloc[-50:].mean()) if len(closes) >= 50 else None
        dma200 = float(closes.iloc[-200:].mean()) if len(closes) >= 200 else None

        m = meta.get(ticker, {})
        out[ticker] = TickerData(
            ticker=ticker,
            name=m.get("name") or None,
            sector=m.get("sector") or None,
            price=round(last, 2),
            prev_close=round(prev, 2),
            change_pct=round((last / prev - 1) * 100, 2) if prev else None,
            ret_1m=_pct_return(closes, 21),
            ret_3m=_pct_return(closes, 63),
            ret_6m=_pct_return(closes, 126),
            above_50dma=(last > dma50) if dma50 else None,
            above_200dma=(last > dma200) if dma200 else None,
            vol_ratio=vol_ratio,
        )
    return out


def _momentum_score(td: TickerData) -> float:
    score = 0.0
    score += (td.ret_3m or 0) * 0.5
    score += (td.ret_6m or 0) * 0.3
    score += (td.ret_1m or 0) * 0.2
    if td.above_200dma:
        score += 10
    if td.above_50dma:
        score += 5
    return score


def screen(
    metrics: dict[str, TickerData],
    holdings: list[str],
    size: int,
) -> list[str]:
    """Return the shortlist of tickers (holdings always included)."""
    if not metrics:
        return list(holdings)

    ranked = list(metrics.values())

    # Bucket 1: momentum leaders.
    momentum = sorted(ranked, key=_momentum_score, reverse=True)
    leaders = [td.ticker for td in momentum[: size]]

    # Bucket 2: biggest absolute daily movers (news-driven), weighted by volume.
    movers = sorted(
        ranked,
        key=lambda td: abs(td.change_pct or 0) * (td.vol_ratio or 1),
        reverse=True,
    )
    mover_tickers = [td.ticker for td in movers[: size // 2]]

    # Blend: alternate leaders/movers for diversity, then top up to size.
    shortlist: list[str] = []
    for a, b in zip(leaders, mover_tickers):
        for t in (a, b):
            if t not in shortlist:
                shortlist.append(t)
    for t in leaders + mover_tickers:
        if t not in shortlist:
            shortlist.append(t)

    shortlist = shortlist[:size]

    # Holdings are always carried through (the AI must be able to sell/add).
    for h in holdings:
        if h not in shortlist:
            shortlist.append(h)

    return shortlist


def enrich(
    tickers: list[str],
    metrics: dict[str, TickerData],
    providers,
) -> list[TickerData]:
    """Attach fundamentals + headlines to the shortlist only."""
    enriched: list[TickerData] = []
    for t in tickers:
        td = metrics.get(t) or TickerData(ticker=t)
        f = providers.fundamentals.fundamentals(t)
        for k, v in f.items():
            if v is not None and getattr(td, k, None) in (None, ""):
                setattr(td, k, v)
        td.headlines = providers.news.headlines(t, limit=4)
        enriched.append(td)
    return enriched
