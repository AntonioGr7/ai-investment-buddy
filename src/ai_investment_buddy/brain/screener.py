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
from itertools import zip_longest

import pandas as pd

from ..config import SETTINGS
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

        # Drawdown from the trailing-1y high — the contrarian signal. A name down
        # hard from its high is one the market is punishing, even if it never
        # shows up as a momentum leader or a single-day mover.
        hi = float(closes.iloc[-252:].max()) if len(closes) else last
        drawdown = round((last / hi - 1) * 100, 1) if hi else None

        m = meta.get(ticker, {})
        out[ticker] = TickerData(
            ticker=ticker,
            name=m.get("name") or None,
            sector=m.get("sector") or None,
            industry=m.get("sub_industry") or m.get("industry") or None,
            price=round(last, 2),
            prev_close=round(prev, 2),
            change_pct=round((last / prev - 1) * 100, 2) if prev else None,
            ret_1m=_pct_return(closes, 21),
            ret_3m=_pct_return(closes, 63),
            ret_6m=_pct_return(closes, 126),
            ret_12m=_pct_return(closes, 252),
            above_50dma=(last > dma50) if dma50 else None,
            above_200dma=(last > dma200) if dma200 else None,
            vol_ratio=vol_ratio,
            drawdown_pct=drawdown,
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
    watchlist: list[str] | None = None,
    punished: list[str] | None = None,
    punished_industries: list[str] | None = None,
) -> list[str]:
    """Return the shortlist of tickers from a *balanced* set of buckets.

    The universe is all S&P 500 + Nasdaq-100 names, so every candidate is already
    large and liquid — no junk floor needed. We deliberately blend four lenses so
    contrarian setups are not drowned out by trend/news:

      - momentum  — trend leaders (what is working),
      - movers    — today's biggest news-driven jumps,
      - oversold  — the deepest drawdowns from their 1y high (what the market is
                    punishing — the SaaS-selloff bucket),
      - sector    — names inside the most beaten-down *sectors* (group repricing).

    Holdings and the user's watchlist are always carried through on top: the AI
    must be able to reconsider what it owns, and the watchlist is the user's
    explicit "always look at these" set."""
    watchlist = watchlist or []
    punished = punished or []
    punished_industries = punished_industries or []
    always = list(dict.fromkeys(list(holdings) + watchlist))
    if not metrics:
        return always

    ranked = list(metrics.values())
    mix = SETTINGS.screener_mix

    def n_for(bucket: str) -> int:
        return max(1, round(mix.get(bucket, 0.0) * size))

    # Bucket 1: momentum leaders.
    momentum = sorted(ranked, key=_momentum_score, reverse=True)
    leaders = [td.ticker for td in momentum[: n_for("momentum")]]

    # Bucket 2: biggest absolute daily movers (news-driven), weighted by volume.
    movers = sorted(
        ranked, key=lambda td: abs(td.change_pct or 0) * (td.vol_ratio or 1), reverse=True
    )
    mover_tickers = [td.ticker for td in movers[: n_for("movers")]]

    # Bucket 3: oversold — deepest drawdowns from the trailing high (most negative
    # first). Tie-break on worst 3m so a slow multi-month bleed scores too.
    oversold = sorted(
        (td for td in ranked if td.drawdown_pct is not None),
        key=lambda td: (td.drawdown_pct, td.ret_3m if td.ret_3m is not None else 0),
    )
    oversold_tickers = [td.ticker for td in oversold[: n_for("oversold")]]

    # Bucket 4: the most beaten-down names within the punished groups. Prefer the
    # finer sub-INDUSTRY grain (e.g. 'Application Software') over the whole sector,
    # so the contrarian bucket targets the part actually being repriced — not the
    # whole sector blob that averages a semis melt-up against a SaaS collapse.
    punished_set = set(punished)
    industry_set = set(punished_industries)

    def _contrarian_rank(td) -> tuple[int, float]:
        # 0 = name sits in a punished sub-industry (tightest), 1 = punished sector.
        in_ind = (td.industry or "") in industry_set
        return (0 if in_ind else 1, td.drawdown_pct)

    in_punished = sorted(
        (
            td for td in ranked
            if td.drawdown_pct is not None
            and ((td.industry or "") in industry_set or (td.sector or "") in punished_set)
        ),
        key=_contrarian_rank,
    )
    sector_tickers = [td.ticker for td in in_punished[: n_for("sector")]]

    # Interleave the buckets so the shortlist stays diverse, then top up to size.
    shortlist: list[str] = []
    buckets = [leaders, mover_tickers, oversold_tickers, sector_tickers]
    for row in zip_longest(*buckets):
        for t in row:
            if t and t not in shortlist:
                shortlist.append(t)
    for bucket in buckets:  # any leftovers if interleave fell short
        for t in bucket:
            if t not in shortlist:
                shortlist.append(t)

    shortlist = shortlist[:size]

    # Holdings + watchlist are always carried through.
    for t in always:
        if t not in shortlist:
            shortlist.append(t)

    return shortlist


def enrich(
    tickers: list[str],
    metrics: dict[str, TickerData],
    providers,
    with_news: bool = True,
) -> list[TickerData]:
    """Attach fundamentals (and optionally headlines) to the shortlist.

    In the daily pipeline ``with_news=False``: the strategist selects on trends +
    valuation, and news is fetched per-finalist *after* selection (targeted due
    diligence rather than an ambient dump)."""
    enriched: list[TickerData] = []
    for t in tickers:
        td = metrics.get(t) or TickerData(ticker=t)
        f = providers.fundamentals.fundamentals(t)
        for k, v in f.items():
            if v is not None and getattr(td, k, None) in (None, ""):
                setattr(td, k, v)
        if with_news:
            td.headlines = providers.news.headlines(t, limit=4)
        enriched.append(td)
    return enriched
