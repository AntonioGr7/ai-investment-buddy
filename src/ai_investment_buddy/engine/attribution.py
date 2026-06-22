"""Risk-adjusted performance & return attribution.

`benchmark.py` answers "did we beat the index?" in raw total return. That is
*return*, not *skill*: a portfolio that is simply 1.4x long the Nasdaq will beat
it in a bull tape and give it all back in a drawdown. This module answers the
harder question — is there an edge, or are we just running high beta?

It works purely off what we already record each run (``nav_history.csv``: the NAV
series + benchmark index levels), so every committed day adds signal. From the
NAV series and each benchmark's level series we compute, per benchmark:

  beta            sensitivity to the benchmark (OLS slope of our returns on its
                  returns). beta≈1 means we move with it; >1 means we amplify it.
  alpha           Jensen's alpha, annualized — the return NOT explained by beta.
                  This is the skill term. Positive & persistent alpha is the bet.
  decomposition   splits total return into a market-beta part (beta × benchmark
                  return) and a selection/timing part (the residual). The headline
                  "X% of your return is beta, Y% is alpha" answer.
  up/down capture how much of the benchmark's up vs down moves we capture. <100%
                  downside capture with >100% upside capture is the goal.

Plus portfolio-level risk-adjusted ratios: annualized volatility, Sharpe,
Sortino, max drawdown, tracking error, information ratio.

IMPORTANT — sample size. With a handful of observations these numbers are noise.
Distinguishing skill from luck needs a real track record; we surface the
observation count and refuse to over-claim below ``MIN_MEANINGFUL`` returns.

Sampling note: runs are manual, so observations are irregularly spaced. We
annualize using the *actual* mean gap between observations (periods-per-year =
365.25 / mean_gap_days) rather than assuming 252 trading days. That keeps the
ratios honest when you run weekly vs daily.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

from ..config import SETTINGS

# Below this many period-returns, treat every ratio as indicative only — there is
# not enough data to separate skill from luck. (Roughly a month of daily runs.)
MIN_MEANINGFUL = 20
# Need at least this many returns to compute anything at all.
_MIN_RETURNS = 2
# A two-factor regression (market + size) needs a few more points than a single
# beta before the coefficients mean anything.
_MIN_FACTOR = 8


@dataclass
class SeriesPoint:
    d: date
    nav: float
    bench: dict[str, float] = field(default_factory=dict)


def _to_float(v) -> float | None:
    if v in (None, ""):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def build_series(
    nav_history: list[dict],
    benchmark_keys: list[str],
    current: SeriesPoint | None = None,
) -> list[SeriesPoint]:
    """Turn raw nav_history rows into a clean, chronologically-ordered series.

    Rows missing a usable NAV are dropped. `current` (today's live NAV + benchmark
    levels) may be appended so the latest, uncommitted point is reflected; it is
    skipped if it duplicates the last recorded date."""
    pts: list[SeriesPoint] = []
    for row in nav_history:
        nav = _to_float(row.get("nav"))
        if nav is None:
            continue
        try:
            d = date.fromisoformat(str(row.get("date")))
        except (TypeError, ValueError):
            continue
        bench = {k: v for k in benchmark_keys if (v := _to_float(row.get(k))) is not None}
        pts.append(SeriesPoint(d=d, nav=nav, bench=bench))

    pts.sort(key=lambda p: p.d)
    if current is not None and (not pts or current.d > pts[-1].d):
        pts.append(current)
    return pts


def _returns(values: list[float]) -> list[float]:
    """Simple period-over-period returns; skips non-positive bases."""
    out = []
    for prev, cur in zip(values, values[1:]):
        if prev and prev > 0:
            out.append(cur / prev - 1.0)
        else:
            out.append(0.0)
    return out


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    """Sample standard deviation (n-1)."""
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def _cov(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = _mean(xs), _mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - 1)


def _max_drawdown(navs: list[float]) -> float:
    """Largest peak-to-trough decline as a negative fraction (e.g. -0.18)."""
    peak = -math.inf
    mdd = 0.0
    for v in navs:
        peak = max(peak, v)
        if peak > 0:
            mdd = min(mdd, v / peak - 1.0)
    return mdd


def _periods_per_year(dates: list[date]) -> float:
    """Annualization factor from the actual mean spacing of observations."""
    if len(dates) < 2:
        return 252.0
    gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
    gaps = [g for g in gaps if g > 0]
    mean_gap = _mean(gaps) if gaps else 1.0
    return 365.25 / mean_gap if mean_gap > 0 else 252.0


def _capture(port: list[float], bench: list[float], up: bool) -> float | None:
    """Up/down capture: avg portfolio return when the benchmark was up (resp.
    down), divided by the benchmark's avg return over those same periods."""
    pairs = [(p, b) for p, b in zip(port, bench) if (b > 0 if up else b < 0)]
    if not pairs:
        return None
    bench_avg = _mean([b for _, b in pairs])
    if bench_avg == 0:
        return None
    return _mean([p for p, _ in pairs]) / bench_avg * 100.0


def compute_metrics(
    series: list[SeriesPoint],
    benchmark_keys: list[str],
    rf_annual: float = 0.0,
) -> dict:
    """Compute portfolio risk metrics and per-benchmark attribution.

    rf_annual: annual risk-free rate as a fraction (e.g. 0.04). Default 0 — a
    conservative choice that slightly understates Sharpe/alpha rather than
    flattering them."""
    n_obs = len(series)
    navs = [p.nav for p in series]
    dates = [p.d for p in series]
    rp = _returns(navs)
    n_ret = len(rp)

    out: dict = {
        "n_observations": n_obs,
        "n_returns": n_ret,
        "meaningful": n_ret >= MIN_MEANINGFUL,
        "span_days": (dates[-1] - dates[0]).days if n_obs >= 2 else 0,
        "first_date": dates[0].isoformat() if dates else None,
        "last_date": dates[-1].isoformat() if dates else None,
        "benchmarks": {},
    }
    if n_ret < _MIN_RETURNS:
        out["insufficient"] = True
        return out

    ppy = _periods_per_year(dates)
    rf_period = rf_annual / ppy if ppy else 0.0
    ann = math.sqrt(ppy)

    total_ret = navs[-1] / navs[0] - 1.0 if navs[0] else 0.0
    years = out["span_days"] / 365.25 if out["span_days"] > 0 else 0.0
    cagr = (navs[-1] / navs[0]) ** (1 / years) - 1.0 if years > 0 and navs[0] > 0 else None

    vol = _std(rp) * ann
    excess = [r - rf_period for r in rp]
    sharpe = _mean(excess) / _std(rp) * ann if _std(rp) > 0 else None
    downside = [min(0.0, r - rf_period) for r in rp]
    dd_dev = math.sqrt(_mean([d * d for d in downside]))
    sortino = _mean(excess) / dd_dev * ann if dd_dev > 0 else None

    out.update(
        {
            "periods_per_year": ppy,
            "total_return": total_ret,
            "cagr": cagr,
            "volatility": vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": _max_drawdown(navs),
        }
    )

    for key in benchmark_keys:
        # Pair our return with the benchmark's only over consecutive points where
        # this benchmark level is present on both ends (so partial columns don't
        # corrupt the regression).
        rp_b, rb = [], []
        for a, b in zip(series, series[1:]):
            la, lb = a.bench.get(key), b.bench.get(key)
            if None in (la, lb) or not (la and la > 0) or not (a.nav and a.nav > 0):
                continue
            rp_b.append(b.nav / a.nav - 1.0)
            rb.append(lb / la - 1.0)
        if len(rb) < _MIN_RETURNS:
            out["benchmarks"][key] = {"insufficient": True, "n_returns": len(rb)}
            continue

        var_b = _cov(rb, rb)
        beta = _cov(rp_b, rb) / var_b if var_b > 0 else None
        std_p, std_b = _std(rp_b), _std(rb)
        corr = _cov(rp_b, rb) / (std_p * std_b) if std_p > 0 and std_b > 0 else None

        # Jensen's alpha (annualized): the per-period return not explained by beta.
        alpha_ann = None
        if beta is not None:
            alpha_period = _mean(rp_b) - (rf_period + beta * (_mean(rb) - rf_period))
            alpha_ann = alpha_period * ppy

        # Full-period return decomposition over THIS paired window.
        port_win = math.prod(1 + r for r in rp_b) - 1.0
        bench_win = math.prod(1 + r for r in rb) - 1.0
        beta_part = beta * bench_win if beta is not None else None
        alpha_part = (port_win - beta_part) if beta_part is not None else None

        active = [p - b for p, b in zip(rp_b, rb)]
        te = _std(active) * ann
        ir = _mean(active) / _std(active) * ann if _std(active) > 0 else None

        out["benchmarks"][key] = {
            "n_returns": len(rb),
            "beta": beta,
            "alpha_annual": alpha_ann,
            "correlation": corr,
            "port_return_window": port_win,
            "bench_return_window": bench_win,
            "beta_contribution": beta_part,
            "alpha_contribution": alpha_part,
            "tracking_error": te,
            "information_ratio": ir,
            "up_capture": _capture(rp_b, rb, up=True),
            "down_capture": _capture(rp_b, rb, up=False),
        }
    return out


def factor_attribution(
    series: list[SeriesPoint],
    market_key: str,
    size_key: str,
    rf_annual: float = 0.0,
) -> dict:
    """Two-factor decomposition: regress portfolio excess returns on the market
    factor and the size factor (SMB = small-cap − market), so a small-cap tilt is
    no longer mistaken for skill.

      beta_market   exposure to the broad market, controlling for size
      beta_size     loading on the size factor. >0 = tilted small; the small-cap
                    'edge' shows up HERE, not as alpha
      alpha         intercept (annualized) — return left after BOTH market and
                    size are removed. THIS is genuine selection skill
      r_squared     how much of the variance the two factors explain

    Returns {available: False, ...} if the size series isn't recorded yet or there
    aren't enough paired observations."""
    rows: list[tuple[float, float, float]] = []
    dates: list[date] = []
    for a, b in zip(series, series[1:]):
        ma, mb = a.bench.get(market_key), b.bench.get(market_key)
        sa, sb = a.bench.get(size_key), b.bench.get(size_key)
        if None in (ma, mb, sa, sb) or not (a.nav and a.nav > 0 and ma > 0 and sa > 0):
            continue
        rows.append((b.nav / a.nav - 1.0, mb / ma - 1.0, sb / sa - 1.0))
        dates.append(b.d)

    if len(rows) < _MIN_FACTOR:
        return {"available": False, "n": len(rows), "size_key": size_key}

    import numpy as np

    ppy = _periods_per_year(dates)
    rf_p = rf_annual / ppy if ppy else 0.0
    rp = np.array([r[0] for r in rows]) - rf_p
    rm = np.array([r[1] for r in rows]) - rf_p  # market excess
    smb = np.array([r[2] - r[1] for r in rows])  # small − market (size factor)

    X = np.column_stack([np.ones(len(rows)), rm, smb])
    coef, *_ = np.linalg.lstsq(X, rp, rcond=None)
    alpha_p, beta_market, beta_size = (float(c) for c in coef)

    resid = rp - X @ coef
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((rp - rp.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

    # Window contributions (compounded factor returns × loading).
    mkt_win = float(np.prod(1 + rm) - 1)
    smb_win = float(np.prod(1 + smb) - 1)
    return {
        "available": True,
        "n": len(rows),
        "meaningful": len(rows) >= MIN_MEANINGFUL,
        "size_key": size_key,
        "market_key": market_key,
        "beta_market": beta_market,
        "beta_size": beta_size,
        "alpha_annual": alpha_p * ppy,
        "r_squared": r2,
        "market_contribution": beta_market * mkt_win,
        "size_contribution": beta_size * smb_win,
        "smb_window": smb_win,
    }


# --- Rendering ---------------------------------------------------------------
def _pct(v, places: int = 2) -> str:
    return f"{v * 100:+.{places}f}%" if v is not None else "n/a"


def _num(v, places: int = 2) -> str:
    return f"{v:.{places}f}" if v is not None else "n/a"


def attribution_report(
    nav_history: list[dict],
    current: SeriesPoint | None = None,
    rf_annual: float = 0.0,
) -> str:
    keys = list(SETTINGS.benchmarks.keys())
    factor_keys = list(SETTINGS.factor_proxies.keys())
    series = build_series(nav_history, keys + factor_keys, current=current)
    m = compute_metrics(series, keys, rf_annual=rf_annual)

    if m.get("insufficient"):
        return (
            f"Not enough history yet ({m['n_observations']} observation(s)). "
            "Attribution needs at least 2 recorded runs; meaningful ratios need "
            f"~{MIN_MEANINGFUL}."
        )

    lines = [
        f"Track record: {m['n_returns']} returns over {m['span_days']} days "
        f"({m['first_date']} → {m['last_date']}), "
        f"~{m['periods_per_year']:.0f} obs/yr annualization.",
    ]
    if not m["meaningful"]:
        lines.append(
            f"  ⚠ INDICATIVE ONLY — below {MIN_MEANINGFUL} returns these numbers "
            "are noise. Skill vs luck is not yet distinguishable."
        )

    lines += [
        "",
        "Risk-adjusted (portfolio):",
        f"  Total return {_pct(m['total_return'])}  |  "
        f"CAGR {_pct(m['cagr'])}  |  Volatility {_pct(m['volatility'])} ann.",
        f"  Sharpe {_num(m['sharpe'])}  |  Sortino {_num(m['sortino'])}  |  "
        f"Max drawdown {_pct(m['max_drawdown'])}",
    ]

    for key, b in m["benchmarks"].items():
        lines.append("")
        if b.get("insufficient"):
            lines.append(f"vs {key}: not enough paired data ({b['n_returns']} returns).")
            continue
        lines.append(f"vs {key}:")
        lines.append(
            f"  Beta {_num(b['beta'])}  |  Alpha {_pct(b['alpha_annual'])} ann.  |  "
            f"Correlation {_num(b['correlation'])}"
        )
        # The headline skill-vs-beta answer.
        if b["beta_contribution"] is not None:
            lines.append(
                f"  Return decomposition over window: portfolio {_pct(b['port_return_window'])} "
                f"= beta/market {_pct(b['beta_contribution'])} "
                f"+ selection/alpha {_pct(b['alpha_contribution'])}"
            )
        lines.append(
            f"  Tracking error {_pct(b['tracking_error'])}  |  "
            f"Information ratio {_num(b['information_ratio'])}"
        )
        lines.append(
            f"  Up-capture {_num(b['up_capture'], 0) + '%' if b['up_capture'] is not None else 'n/a'}  |  "
            f"Down-capture {_num(b['down_capture'], 0) + '%' if b['down_capture'] is not None else 'n/a'}"
        )

    # Two-factor decomposition (market + size) — separates a small-cap tilt from
    # genuine selection alpha. Only when a size proxy has been recorded long enough.
    if keys and factor_keys:
        fa = factor_attribution(series, keys[0], factor_keys[0], rf_annual=rf_annual)
        lines.append("")
        if not fa.get("available"):
            lines.append(
                f"Factor decomposition (market + size): not enough paired data yet "
                f"({fa.get('n', 0)}/{_MIN_FACTOR} obs incl. {factor_keys[0]})."
            )
        else:
            lines.append(f"Factor decomposition — market ({keys[0]}) vs size ({fa['size_key']}) vs selection:")
            if not fa["meaningful"]:
                lines.append("  ⚠ INDICATIVE ONLY — too few observations to trust the loadings.")
            lines.append(
                f"  Market beta {_num(fa['beta_market'])}  |  "
                f"Size beta {_num(fa['beta_size'])} ({'tilted SMALL' if fa['beta_size'] > 0.15 else 'tilted LARGE' if fa['beta_size'] < -0.15 else 'size-neutral'})  |  "
                f"R² {_num(fa['r_squared'])}"
            )
            lines.append(
                f"  Selection alpha (after market AND size removed) {_pct(fa['alpha_annual'])} ann."
            )
            lines.append(
                f"  Window attribution: market {_pct(fa['market_contribution'])} + "
                f"size-tilt {_pct(fa['size_contribution'])} + selection (residual)."
            )
            lines.append("  " + _factor_verdict(fa))

    # Plain-English verdict on the primary benchmark.
    primary = keys[0] if keys else None
    pb = m["benchmarks"].get(primary) if primary else None
    if pb and not pb.get("insufficient") and pb["beta"] is not None:
        lines.append("")
        lines.append(_verdict(pb, m["meaningful"], primary))
    return "\n".join(lines)


def _factor_verdict(fa: dict) -> str:
    bs, alpha = fa["beta_size"], fa["alpha_annual"]
    tilt = (
        f"a small-cap tilt (size beta {bs:+.2f}) contributing {_pct(fa['size_contribution'])}"
        if bs > 0.15
        else f"a large-cap tilt (size beta {bs:+.2f})"
        if bs < -0.15
        else "no meaningful size tilt"
    )
    skill = (
        "and genuine selection alpha on top"
        if alpha and alpha > 0.01
        else "with no selection alpha once the tilt is removed"
        if alpha is not None and abs(alpha) <= 0.01
        else "and negative selection alpha (the tilt flatters the raw return)"
    )
    return f"Read: the book shows {tilt} {skill}."


def _verdict(b: dict, meaningful: bool, key: str) -> str:
    beta = b["beta"]
    alpha = b["alpha_annual"]
    leverage = "amplifies" if beta and beta > 1.05 else ("dampens" if beta and beta < 0.95 else "tracks")
    edge = (
        "positive alpha (a selection edge so far)"
        if alpha and alpha > 0.01
        else "no real alpha — return is essentially the benchmark times beta"
        if alpha is not None and abs(alpha) <= 0.01
        else "negative alpha (lagging on a risk-adjusted basis)"
    )
    prefix = "Read: " if meaningful else "Early read (low confidence): "
    return f"{prefix}the book {leverage} {key} (beta {_num(beta)}) and shows {edge}."
