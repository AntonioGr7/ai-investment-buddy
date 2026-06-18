"""The daily run: ingest -> screen -> decide -> execute -> record.

One call to ``run_daily`` performs a full cycle. ``dry_run=True`` stops before
mutating any state (useful to preview what the AI would do)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from ..brain import screener
from ..brain.decide import DecisionEngine
from ..config import SETTINGS
from ..data import get_providers
from ..memory import Journal, MemoryToolkit, snapshot, store
from ..memory.portfolio import Portfolio
from ..models import (
    Decision,
    MacroSnapshot,
    StrategistView,
    TickerData,
    Trade,
    ValuationAssessment,
)
from ..universe import get_universe
from .benchmark import performance_summary
from .execute import execute


@dataclass
class RunResult:
    as_of: date
    portfolio: Portfolio
    decision: Decision
    trades: list[Trade]
    prices: dict[str, float]
    macro: MacroSnapshot
    shortlist: list[TickerData]
    performance_before: str
    nav_after: float
    strategy: StrategistView | None = None
    assessments: list[ValuationAssessment] = field(default_factory=list)
    dry_run: bool = False
    log: list[str] = field(default_factory=list)


def _portfolio_state(pf: Portfolio, prices: dict[str, float]) -> dict:
    nav = pf.nav(prices)
    positions = []
    for t, pos in pf.positions.items():
        px = prices.get(t)
        if px is None:
            continue
        mv = pos.market_value(px)
        positions.append(
            {
                "ticker": t,
                "shares": round(pos.shares, 4),
                "avg_cost": round(pos.avg_cost, 2),
                "price": round(px, 2),
                "market_value": round(mv, 2),
                "weight": round(mv / nav, 4) if nav else 0.0,
                "unrealized_pnl": round(pos.unrealized_pnl(px), 2),
            }
        )
    positions.sort(key=lambda p: p["market_value"], reverse=True)
    return {
        "nav": round(nav, 2),
        "cash": round(pf.cash, 2),
        "cash_weight": round(pf.cash / nav, 4) if nav else 1.0,
        "n_positions": len(pf.positions),
        "positions": positions,
    }


def _auto_export(progress) -> None:
    """Write a portable snapshot of all state after a committed run."""
    if not SETTINGS.auto_export:
        return
    try:
        path = SETTINGS.snapshot_path
        snapshot.export_state(path)
        progress(f"Auto-exported state snapshot → {path}")
    except Exception as e:
        progress(f"(auto-export skipped: {e})")


def _ensure_prices(providers, prices: dict[str, float], tickers: list[str]) -> None:
    for t in tickers:
        if t not in prices or not prices[t]:
            px = providers.prices.latest_price(t)
            if px:
                prices[t] = px


def run_daily(
    as_of: date | None = None,
    dry_run: bool = False,
    on_progress=None,
) -> RunResult:
    as_of = as_of or datetime.now(timezone.utc).date()
    log: list[str] = []

    def progress(msg: str) -> None:
        log.append(msg)
        if on_progress:
            on_progress(msg)

    portfolio = store.load_portfolio()
    providers = get_providers()

    progress("Loading universe (S&P 500 + Nasdaq 100)…")
    universe = get_universe()
    tickers = [c["ticker"] for c in universe]
    meta = {c["ticker"]: c for c in universe}

    progress("Sampling macro snapshot…")
    macro = providers.macro.snapshot()

    progress("Reading market & macro/Fed news…")
    market_news = providers.market_news.market_digest(days=4, per_feed=5)
    progress(f"Pulled {len(market_news)} market/macro headlines.")

    progress(f"Downloading price history for {len(tickers)} tickers…")
    history = providers.prices.history(tickers, lookback_days=260)
    metrics = screener.compute_metrics(history, meta)
    progress(f"Computed technicals for {len(metrics)} tickers.")

    holdings = list(portfolio.positions.keys())
    shortlist_tickers = screener.screen(metrics, holdings, SETTINGS.shortlist_size)
    progress(
        f"Screened to {len(shortlist_tickers)} candidates "
        f"({len(holdings)} current holdings included). Enriching with fundamentals + news…"
    )
    shortlist = screener.enrich(shortlist_tickers, metrics, providers)

    # Prices for valuation.
    prices = {t: td.price for t, td in metrics.items() if td.price}
    _ensure_prices(providers, prices, holdings)

    nav_history = store.load_nav_history()
    current_benchmarks = {
        label: macro.indicators[label]
        for label in SETTINGS.benchmarks
        if label in macro.indicators
    }
    perf = performance_summary(nav_history, portfolio.nav(prices), current_benchmarks)

    journal = Journal()
    toolkit = MemoryToolkit()
    narrative = toolkit.read_narrative()
    recent = journal.recent_entries(5)
    theses = journal.load_theses()
    state = _portfolio_state(portfolio, prices)

    progress("Running the 3-stage brain (strategist → analyst → PM)…")
    engine = DecisionEngine()
    brain = engine.decide(
        as_of=as_of,
        portfolio_state=state,
        macro=macro,
        shortlist=shortlist,
        recent_journal=recent,
        theses=theses,
        performance=perf,
        market_news=market_news,
        holdings=holdings,
        narrative=narrative,
        toolkit=toolkit,
        on_progress=progress,
    )
    decision = brain.decision
    progress(
        f"Decision: {len(decision.orders)} order(s); "
        f"target cash {decision.target_cash_weight:.0%}."
    )

    if dry_run:
        progress("Dry run — no state changed.")
        return RunResult(
            as_of=as_of,
            portfolio=portfolio,
            decision=decision,
            trades=[],
            prices=prices,
            macro=macro,
            shortlist=shortlist,
            performance_before=perf,
            nav_after=portfolio.nav(prices),
            strategy=brain.strategy,
            assessments=brain.assessments,
            dry_run=True,
            log=log,
        )

    # Make sure we have prices for any ticker the AI wants to trade.
    _ensure_prices(providers, prices, [o.ticker for o in decision.orders])

    trades = execute(portfolio, decision, prices)
    progress(f"Executed {len(trades)} trade(s).")

    store.save_portfolio(portfolio)
    store.append_trades(trades)
    journal.record_day(decision, brain.strategy, brain.assessments)
    journal.update_theses(decision)

    progress("Consolidating long-horizon memory narrative…")
    try:
        new_narrative = engine.consolidate(
            narrative, as_of, brain.strategy, decision, perf
        )
        toolkit.write_narrative(new_narrative)
    except Exception as e:
        progress(f"(narrative consolidation skipped: {e})")

    nav_after = portfolio.nav(prices)
    store.append_nav(
        as_of=as_of,
        nav=nav_after,
        cash=portfolio.cash,
        invested=portfolio.invested_value(prices),
        n_positions=len(portfolio.positions),
        benchmarks=current_benchmarks,
    )
    progress(f"Recorded NAV ${nav_after:,.2f}.")
    _auto_export(progress)

    return RunResult(
        as_of=as_of,
        portfolio=portfolio,
        decision=decision,
        trades=trades,
        prices=prices,
        macro=macro,
        shortlist=shortlist,
        performance_before=perf,
        nav_after=nav_after,
        strategy=brain.strategy,
        assessments=brain.assessments,
        dry_run=False,
        log=log,
    )


def commit(dry: RunResult, on_progress=None) -> RunResult:
    """Execute and persist a previously computed (dry-run) decision.

    Lets the CLI preview the AI's plan and only commit on confirmation, without
    paying for a second model call."""
    if not dry.dry_run:
        raise ValueError("commit() expects a dry-run RunResult.")

    def progress(msg: str) -> None:
        dry.log.append(msg)
        if on_progress:
            on_progress(msg)

    providers = get_providers()
    portfolio = dry.portfolio
    prices = dict(dry.prices)
    _ensure_prices(providers, prices, [o.ticker for o in dry.decision.orders])

    trades = execute(portfolio, dry.decision, prices)
    progress(f"Executed {len(trades)} trade(s).")

    store.save_portfolio(portfolio)
    store.append_trades(trades)
    journal = Journal()
    journal.record_day(dry.decision, dry.strategy, dry.assessments)
    journal.update_theses(dry.decision)

    progress("Consolidating long-horizon memory narrative…")
    try:
        toolkit = MemoryToolkit()
        engine = DecisionEngine()
        new_narrative = engine.consolidate(
            toolkit.read_narrative(), dry.as_of, dry.strategy, dry.decision,
            dry.performance_before,
        )
        toolkit.write_narrative(new_narrative)
    except Exception as e:
        progress(f"(narrative consolidation skipped: {e})")

    current_benchmarks = {
        label: dry.macro.indicators[label]
        for label in SETTINGS.benchmarks
        if label in dry.macro.indicators
    }
    nav_after = portfolio.nav(prices)
    store.append_nav(
        as_of=dry.as_of,
        nav=nav_after,
        cash=portfolio.cash,
        invested=portfolio.invested_value(prices),
        n_positions=len(portfolio.positions),
        benchmarks=current_benchmarks,
    )
    progress(f"Recorded NAV ${nav_after:,.2f}.")
    _auto_export(progress)

    dry.trades = trades
    dry.prices = prices
    dry.nav_after = nav_after
    dry.dry_run = False
    return dry
