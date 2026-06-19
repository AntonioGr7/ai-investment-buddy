"""Top-down sector scan — the contrarian radar.

The single-name screener buckets (momentum, movers) are recency filters: they
see what is strong or what jumped *today*. They are structurally blind to a
whole sector being repriced lower over weeks — exactly the setup that produces
the best contrarian opportunities (e.g. software sold off en masse on an
AI-disruption narrative).

This module aggregates the technicals we already computed for the *entire*
universe into per-sector health stats, so the strategist can reason about which
groups the market is punishing — and the screener can deliberately fish in those
waters. It is pure computation: no extra data fetches, no model calls.
"""

from __future__ import annotations

import math
from statistics import median

from ..config import SETTINGS
from ..models import SectorStat, TickerData

# The SPDR sector ETFs — one per GICS sector. Their returns give the
# market-cap-weighted, top-down sector read (what the Finviz sector map shows),
# keyed by the same GICS sector names our universe uses.
SECTOR_ETFS: dict[str, str] = {
    "Information Technology": "XLK",
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}


def _median(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(median(vals), 1) if vals else None


def _ret(closes, lookback: int) -> float | None:
    if closes is None or len(closes) <= lookback:
        return None
    past = closes.iloc[-1 - lookback]
    last = closes.iloc[-1]
    if past and not math.isnan(past):
        return round((last / past - 1) * 100, 1)
    return None


def _ytd(closes) -> float | None:
    try:
        year = closes.index[-1].year
        yr = closes[closes.index.year == year]
        if len(yr) < 1:
            return None
        first, last = float(yr.iloc[0]), float(closes.iloc[-1])
        return round((last / first - 1) * 100, 1) if first else None
    except Exception:
        return None


def fetch_sector_performance(prices, lookback_days: int = 260) -> dict[str, dict]:
    """Market-cap-weighted sector performance from the SPDR sector ETFs.

    Returns ``{sector_name: {etf, ret_1w, ret_1m, ret_3m, ret_6m, ret_ytd}}``.
    Best-effort: a failed download yields an empty map (the scan still works off
    constituents)."""
    try:
        hist = prices.history(list(SECTOR_ETFS.values()), lookback_days=lookback_days)
    except Exception:
        return {}
    out: dict[str, dict] = {}
    for sector, etf in SECTOR_ETFS.items():
        df = hist.get(etf)
        if df is None or df.empty or "Close" not in df:
            continue
        closes = df["Close"].dropna()
        if len(closes) < 30:
            continue
        out[sector] = {
            "etf": etf,
            "ret_1w": _ret(closes, 5),
            "ret_1m": _ret(closes, 21),
            "ret_3m": _ret(closes, 63),
            "ret_6m": _ret(closes, 126),
            "ret_12m": _ret(closes, 252),
            "ret_ytd": _ytd(closes),
        }
    return out


def _trend_label(r3: float | None, r6: float | None, r12: float | None) -> str:
    """Classify a sector by its long-run trend vs its recent move, so we can lead
    with durable trends and pursue dislocations within them.

      durable-up      strong over 6-12m and not falling now
      dip-in-uptrend  strong over 6-12m but down recently — the contrarian entry
      durable-down    weak over 6-12m and still falling (secular decline / value trap)
      recovering      weak over 6-12m but turning up recently
      choppy/n.a.     mixed or insufficient data
    """
    if r12 is None or r6 is None:
        return "n/a"
    up_long = r12 > 5 and r6 > -2
    down_long = r12 < -5 and r6 < 2
    short_down = r3 is not None and r3 < -3
    short_up = r3 is not None and r3 > 3
    if up_long and short_down:
        return "dip-in-uptrend"
    if up_long:
        return "durable-up"
    if down_long and short_up:
        return "recovering"
    if down_long:
        return "durable-down"
    return "choppy"


def scan_sectors(
    metrics: dict[str, TickerData],
    etf_perf: dict[str, dict] | None = None,
) -> list[SectorStat]:
    """Aggregate per-GICS-sector health, ranked worst-performing first.

    Combines a bottom-up read (median returns + breadth across our constituents)
    with the top-down sector-ETF performance (``etf_perf``, market-cap-weighted)
    when available. Worst-first ordering puts the punished sectors at the top —
    the ones worth asking 'overreaction or value trap?' about."""
    etf_perf = etf_perf or {}
    buckets: dict[str, list[TickerData]] = {}
    for td in metrics.values():
        sector = (td.sector or "").strip()
        if not sector:
            continue
        buckets.setdefault(sector, []).append(td)

    stats: list[SectorStat] = []
    for sector, names in buckets.items():
        if len(names) < 3:  # too thin to be a meaningful aggregate
            continue
        above = [n.above_200dma for n in names if n.above_200dma is not None]
        breadth = round(100 * sum(1 for x in above if x) / len(above), 0) if above else None
        etf = etf_perf.get(sector, {})
        med_3m = _median([n.ret_3m for n in names])
        med_6m = _median([n.ret_6m for n in names])
        r3 = etf.get("ret_3m") if etf.get("ret_3m") is not None else med_3m
        r6 = etf.get("ret_6m") if etf.get("ret_6m") is not None else med_6m
        r12 = etf.get("ret_12m")
        stats.append(
            SectorStat(
                sector=sector,
                n=len(names),
                ret_1m=_median([n.ret_1m for n in names]),
                ret_3m=med_3m,
                ret_6m=med_6m,
                breadth_200dma=breadth,
                median_drawdown=_median([n.drawdown_pct for n in names]),
                etf=etf.get("etf"),
                etf_ret_1w=etf.get("ret_1w"),
                etf_ret_1m=etf.get("ret_1m"),
                etf_ret_3m=etf.get("ret_3m"),
                etf_ret_6m=etf.get("ret_6m"),
                etf_ret_12m=r12,
                etf_ret_ytd=etf.get("ret_ytd"),
                trend=_trend_label(r3, r6, r12),
            )
        )

    # Rank worst-first on the best 3m signal we have (ETF if present, else median).
    def rank_3m(s: SectorStat) -> float:
        v = s.etf_ret_3m if s.etf_ret_3m is not None else s.ret_3m
        return v if v is not None else 0.0

    stats.sort(key=rank_3m)
    return stats


def punished_sectors(stats: list[SectorStat]) -> list[str]:
    """The most beaten-down sectors — where the screener should go hunting."""
    return [s.sector for s in stats[: SETTINGS.punished_sector_count]]


def format_sector_scan(stats: list[SectorStat]) -> str:
    """Render the scan for the strategist prompt, worst-performing first."""
    if not stats:
        return "SECTOR SCAN: (insufficient data)"
    lines = [
        "SECTOR TREND MAP (sector-ETF market-cap-weighted returns incl. the long-run "
        "6-12m trend, + breadth; worst-3m first). The [trend] tag reads the long run "
        "vs the recent move: 'durable-up' = structurally strong; 'dip-in-uptrend' = "
        "strong long-run but sold off recently (the PRIME contrarian entry — the "
        "long-run trend says the dip is likely an overreaction); 'durable-down' = "
        "secular decline (value trap, avoid); 'recovering' = turning up off a weak base:"
    ]
    for s in stats:
        lines.append("  - " + s.one_line())
    return "\n".join(lines)
