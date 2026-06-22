"""Portfolio-level (book) risk.

The execution guardrails are all *per-name* (max 20%/name, no leverage). That is
blind to the risk that actually sinks concentrated long books: five "different"
names that are really one bet. If the book is 70% long-duration AI beta, a 20%
cap on each name still leaves you implicitly leveraged to a single factor — and
"beating the market" in that state just means you out-ran it on the way up and
will give it all back on the way down.

This module looks at the WHOLE book at once, from the daily price history we
already pull each run, and answers:

  book volatility     ex-ante annualized vol of the equity sleeve (from the
                      holdings' return covariance — captures how they co-move,
                      not just each one's standalone vol).
  market exposure     book beta to the primary benchmark, and the NAV-level beta
                      after cash drag. NAV beta >> 1 = you're leveraged-long.
  concentration       HHI on book weights → effective number of names; the
                      diversification ratio (weighted-avg standalone vol / book
                      vol) → how many *independent* bets you really hold; the
                      single biggest sector weight; the most-correlated pairs.
  risk contribution   each holding's share of book variance (not just its weight)
                      — surfaces the name that is secretly driving your risk.
  drawdown            current peak-to-trough on NAV, and whether it has breached
                      the circuit-breaker.

Everything is surfaced to the PM as a soft guardrail (consistent with the
house style: execution enforces the hard limits, the PM is asked to respect the
soft ones with judgement) and shown by `aib risk`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import SETTINGS

_TRADING_DAYS = 252


@dataclass
class HoldingRisk:
    ticker: str
    weight: float  # fraction of NAV
    sector: str = "Unknown"
    vol: float | None = None  # standalone annualized vol
    beta: float | None = None  # to primary benchmark
    risk_contribution: float | None = None  # share of book variance (0..1)
    has_history: bool = True


@dataclass
class RiskReport:
    nav: float
    cash_weight: float
    invested_weight: float
    n_positions: int
    benchmark_label: str = ""
    book_vol: float | None = None  # annualized, equity sleeve
    book_beta: float | None = None  # beta of the invested book
    nav_beta: float | None = None  # book_beta * invested_weight (effective market exposure)
    hhi: float | None = None  # Herfindahl on book weights (0..1)
    effective_names: float | None = None  # 1/HHI
    diversification_ratio: float | None = None  # >1 good; ~1 = one big bet
    sector_weights: dict[str, float] = field(default_factory=dict)
    max_sector: tuple[str, float] | None = None
    top_correlations: list[tuple[str, str, float]] = field(default_factory=list)
    holdings: list[HoldingRisk] = field(default_factory=list)
    current_drawdown: float | None = None  # negative fraction
    drawdown_breach: bool = False
    flags: list[str] = field(default_factory=list)


def _close_series(df: pd.DataFrame | None) -> pd.Series | None:
    if df is None or df.empty:
        return None
    col = "Close" if "Close" in df.columns else ("close" if "close" in df.columns else None)
    if col is None:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return s if len(s) > 2 else None


def _current_drawdown(navs: list[float]) -> float | None:
    navs = [n for n in navs if n and n > 0]
    if len(navs) < 2:
        return None
    peak = max(navs)  # peak-to-DATE; current dd is vs the running peak up to now
    running_peak = -math.inf
    dd = 0.0
    for v in navs:
        running_peak = max(running_peak, v)
        dd = v / running_peak - 1.0
    return dd  # the drawdown AT the latest point


def build_risk_report(
    weights: dict[str, float],
    cash_weight: float,
    nav: float,
    history: dict[str, pd.DataFrame],
    benchmark_history: pd.DataFrame | None,
    benchmark_label: str,
    sectors: dict[str, str] | None = None,
    nav_navs: list[float] | None = None,
    lookback_days: int | None = None,
) -> RiskReport:
    """Compute book-level risk for the current holdings.

    weights: ticker -> fraction of NAV (invested names only; cash excluded).
    history: ticker -> OHLCV DataFrame (the daily pull the pipeline already has).
    benchmark_history: OHLCV for the primary benchmark (for beta); may be None.
    sectors: ticker -> GICS sector (from stored valuations); missing = Unknown.
    nav_navs: NAV series (oldest→newest) for the drawdown calc.
    """
    sectors = sectors or {}
    lookback = lookback_days or SETTINGS.risk_lookback_days
    invested = sum(weights.values())
    report = RiskReport(
        nav=nav,
        cash_weight=cash_weight,
        invested_weight=invested,
        n_positions=len(weights),
        benchmark_label=benchmark_label,
    )

    # Drawdown is independent of price history availability.
    report.current_drawdown = _current_drawdown(nav_navs or [])
    if report.current_drawdown is not None:
        report.drawdown_breach = report.current_drawdown <= -SETTINGS.drawdown_circuit_breaker

    # Sector exposure (fraction of NAV) — works even with no price history.
    sec: dict[str, float] = {}
    for t, w in weights.items():
        sec[sectors.get(t, "Unknown")] = sec.get(sectors.get(t, "Unknown"), 0.0) + w
    report.sector_weights = dict(sorted(sec.items(), key=lambda kv: kv[1], reverse=True))
    if report.sector_weights:
        top = max(report.sector_weights.items(), key=lambda kv: kv[1])
        report.max_sector = top

    # Concentration on the book (weights normalized to the invested sleeve).
    book_w = {t: w / invested for t, w in weights.items()} if invested > 0 else {}
    if book_w:
        report.hhi = sum(w * w for w in book_w.values())
        report.effective_names = 1.0 / report.hhi if report.hhi > 0 else None

    # --- Return-based metrics (need price history) ---------------------------
    closes: dict[str, pd.Series] = {}
    for t in weights:
        s = _close_series(history.get(t))
        if s is not None:
            closes[t] = s.tail(lookback + 5)

    holdings: list[HoldingRisk] = []
    bench_close = _close_series(benchmark_history)
    bench_ret = bench_close.pct_change().dropna() if bench_close is not None else None
    std_ann: dict[str, float] = {}
    betas: dict[str, float | None] = {}
    rc: dict[str, float] = {}

    if closes:
        rets = pd.DataFrame(closes).sort_index().pct_change().dropna(how="any")
        # Drop names that are flat / have no overlapping movement.
        usable = [t for t in rets.columns if rets[t].abs().sum() > 0]
        rets = rets[usable]
        enough = len(rets) >= 5
        cov_d = rets.cov() if enough and rets.shape[1] >= 1 else None

        # Standalone annualized vol + beta to the benchmark, per name.
        for t in rets.columns if enough else []:
            std_ann[t] = float(rets[t].std() * math.sqrt(_TRADING_DAYS))
            if bench_ret is not None:
                aligned = pd.concat([rets[t], bench_ret], axis=1, join="inner").dropna()
                if len(aligned) >= 5:
                    bvar = float(aligned.iloc[:, 1].var())
                    betas[t] = float(aligned.cov().iloc[0, 1] / bvar) if bvar > 0 else None

        # Book-level vol, risk contributions, diversification ratio, beta.
        if cov_d is not None and invested > 0:
            cols = list(cov_d.columns)
            wb = np.array([book_w.get(t, 0.0) for t in cols])
            Sigma = cov_d.values
            book_var_d = float(wb @ Sigma @ wb)
            if book_var_d > 0:
                report.book_vol = math.sqrt(book_var_d * _TRADING_DAYS)
                mrc = Sigma @ wb  # marginal risk contributions (daily units)
                rc = {cols[i]: float(wb[i] * mrc[i] / book_var_d) for i in range(len(cols))}
                wavg_vol_d = sum(book_w.get(t, 0.0) * float(rets[t].std()) for t in cols)
                if wavg_vol_d > 0:
                    report.diversification_ratio = (
                        wavg_vol_d * math.sqrt(_TRADING_DAYS) / report.book_vol
                    )

            book_beta = sum(book_w.get(t, 0.0) * betas[t] for t in cols if betas.get(t) is not None)
            if any(betas.get(t) is not None for t in cols):
                report.book_beta = book_beta
                report.nav_beta = book_beta * invested

        # Most-correlated pairs (concentration of co-movement).
        if enough and rets.shape[1] >= 2:
            corr = rets.corr()
            pairs = []
            for i, a in enumerate(corr.columns):
                for b in corr.columns[i + 1 :]:
                    c = float(corr.loc[a, b])
                    if not math.isnan(c):
                        pairs.append((a, b, c))
            pairs.sort(key=lambda x: x[2], reverse=True)
            report.top_correlations = pairs[:5]

    for t, w in weights.items():
        holdings.append(
            HoldingRisk(
                ticker=t,
                weight=w,
                sector=sectors.get(t, "Unknown"),
                vol=std_ann.get(t),
                beta=betas.get(t),
                risk_contribution=rc.get(t),
                has_history=t in closes,
            )
        )

    holdings.sort(
        key=lambda h: (h.risk_contribution if h.risk_contribution is not None else h.weight),
        reverse=True,
    )
    report.holdings = holdings
    report.flags = _flags(report)
    return report


def _flags(r: RiskReport) -> list[str]:
    flags: list[str] = []
    if (
        r.max_sector
        and r.max_sector[0] != "Unknown"  # all-Unknown is a data gap, not concentration
        and r.max_sector[1] > SETTINGS.max_sector_weight
    ):
        flags.append(
            f"SECTOR CONCENTRATION: {r.max_sector[1]:.0%} in {r.max_sector[0]} "
            f"(soft cap {SETTINGS.max_sector_weight:.0%}) — trim or diversify before adding more here."
        )
    if r.nav_beta is not None and r.nav_beta > SETTINGS.max_portfolio_beta:
        flags.append(
            f"HIGH MARKET EXPOSURE: effective NAV beta {r.nav_beta:.2f} to {r.benchmark_label} "
            f"(soft cap {SETTINGS.max_portfolio_beta:.2f}) — the book is leveraged-long; "
            "outperformance here is mostly market direction, not selection."
        )
    if (
        r.diversification_ratio is not None
        and r.n_positions >= 3
        and r.diversification_ratio < SETTINGS.min_diversification_ratio
    ):
        flags.append(
            f"FEW REAL BETS: diversification ratio {r.diversification_ratio:.2f} "
            f"(want ≥{SETTINGS.min_diversification_ratio:.2f}) — your names move together, "
            "so you hold fewer independent bets than positions. Adding a correlated name adds risk, not diversification."
        )
    hot = [p for p in r.top_correlations if p[2] >= SETTINGS.high_pair_correlation]
    if hot:
        pairs = ", ".join(f"{a}~{b} ({c:.2f})" for a, b, c in hot[:3])
        flags.append(
            f"CORRELATED CLUSTER: {pairs} — these are effectively one position; "
            "size them as a group, not independently."
        )
    if r.drawdown_breach:
        flags.append(
            f"DRAWDOWN CIRCUIT-BREAKER: book is {r.current_drawdown:+.1%} from its peak "
            f"(threshold {-SETTINGS.drawdown_circuit_breaker:.0%}) — raise the bar to add risk, "
            "prefer trimming the highest-beta / highest-risk-contribution names; preserve capital."
        )
    return flags


# --- Rendering ---------------------------------------------------------------
def _f(v, fmt="{:.2f}") -> str:
    return fmt.format(v) if v is not None else "n/a"


def format_risk(r: RiskReport, detailed: bool = True) -> str:
    if r.n_positions == 0:
        return (
            f"Book is {r.cash_weight:.0%} cash, no positions — no concentration or "
            "market-exposure risk to report."
        )
    lines = [
        f"Book: {r.invested_weight:.0%} invested across {r.n_positions} name(s), "
        f"{r.cash_weight:.0%} cash.",
        f"Volatility (equity sleeve): {_f(r.book_vol, '{:.0%}') if r.book_vol is not None else 'n/a'} ann.  |  "
        f"Effective market exposure (NAV beta to {r.benchmark_label}): {_f(r.nav_beta)} "
        f"(book beta {_f(r.book_beta)}).",
        f"Concentration: {_f(r.effective_names, '{:.1f}')} effective names "
        f"(of {r.n_positions})  |  diversification ratio {_f(r.diversification_ratio)} "
        f"(≈independent bets).",
    ]
    if r.max_sector:
        sect = ", ".join(f"{s} {w:.0%}" for s, w in list(r.sector_weights.items())[:4])
        lines.append(f"Sector exposure: {sect}.")
    if r.current_drawdown is not None:
        lines.append(f"Current drawdown from peak: {r.current_drawdown:+.1%}.")

    if detailed and r.holdings:
        lines.append("")
        lines.append("Risk contribution by holding (share of book variance):")
        for h in r.holdings:
            rc = f"{h.risk_contribution:.0%}" if h.risk_contribution is not None else "n/a"
            beta = _f(h.beta)
            vol = f"{h.vol:.0%}" if h.vol is not None else "n/a"
            note = "" if h.has_history else "  (no price history — concentration only)"
            lines.append(
                f"  {h.ticker:<6} w {h.weight:>5.1%} | risk {rc:>4} | beta {beta:>5} | "
                f"vol {vol:>4} | {h.sector}{note}"
            )

    if r.flags:
        lines.append("")
        lines.append("⚠ RISK FLAGS (soft guardrails — respect with judgement):")
        for f in r.flags:
            lines.append(f"  • {f}")
    else:
        lines.append("")
        lines.append("No book-level risk limits breached.")
    return "\n".join(lines)
