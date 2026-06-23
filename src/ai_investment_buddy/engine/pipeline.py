"""The daily run: ingest -> screen -> decide -> execute -> record.

One call to ``run_daily`` performs a full cycle. ``dry_run=True`` stops before
mutating any state (useful to preview what the AI would do)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from .. import audit
from ..brain import screener, sectors
from ..brain.decide import DecisionEngine
from ..config import SETTINGS
from ..data import get_providers
from ..memory import Journal, MemoryToolkit, radar, snapshot, store, valuations
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
from ..watchlist import load_watchlist
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
    # ticker -> avg daily dollar volume, for realistic market-impact slippage at commit.
    liquidity: dict[str, float] = field(default_factory=dict)
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


def _valid_px(px) -> bool:
    # Reject None, NaN (px != px), and non-positive prices.
    return px is not None and px == px and px > 0


def _recent_activity(as_of: date) -> str:
    """A short turnover note for the PM, so it can feel its own churn and resist
    trading when it has just traded. This is a long-run game — patience compounds."""
    trades = store.load_trades()
    if not trades:
        return "No trades yet — the book is fresh; only act on a genuine fat pitch."
    last = max(t.timestamp.date() for t in trades)
    days_since = (as_of - last).days
    n7 = sum(1 for t in trades if (as_of - t.timestamp.date()).days <= 7)
    n30 = sum(1 for t in trades if (as_of - t.timestamp.date()).days <= 30)
    return (
        f"{n7} trade(s) in the last 7 days, {n30} in the last 30; "
        f"last trade {days_since} day(s) ago. "
        f"If you have been active recently, the bar to trade again is higher."
    )


def _book_risk_summary(state, history, providers, metrics, nav_history, progress) -> str:
    """Compute the whole-book risk read for the PM. Best-effort: any failure just
    yields an empty summary (the PM still has per-name discipline)."""
    weights = {p["ticker"]: p["weight"] for p in state.get("positions", [])}
    if not weights:
        return ""
    try:
        from .risk import build_risk_report, format_risk

        bench_label, bench_sym = next(iter(SETTINGS.benchmarks.items()))
        try:
            bench_hist = providers.prices.history(
                [bench_sym], lookback_days=SETTINGS.risk_lookback_days
            ).get(bench_sym)
        except Exception:
            bench_hist = None
        # Sector per holding: prefer the fresh screener read, fall back to stored valuations.
        sectors = {t: s for t in weights if (s := getattr(metrics.get(t), "sector", None))}
        if len(sectors) < len(weights):
            from ..memory import valuations

            for rec in valuations.load_all():
                if rec.ticker in weights and rec.ticker not in sectors and rec.latest.assessment.sector:
                    sectors[rec.ticker] = rec.latest.assessment.sector
        navs = [float(r["nav"]) for r in nav_history if r.get("nav")] + [state["nav"]]
        report = build_risk_report(
            weights, state.get("cash_weight", 0.0), state["nav"], history,
            bench_hist, bench_label, sectors=sectors, nav_navs=navs,
        )
        n = len(report.flags)
        progress(f"Computed book-level risk{f' — {n} flag(s)' if n else ' — no limits breached'}.")
        return format_risk(report, detailed=True)
    except Exception as e:
        progress(f"(book risk skipped: {e})")
        return ""


def _resolve_and_calibrate(providers, prices, as_of, progress) -> str:
    """Resolve due price-anchored predictions and return the calibration scorecard
    (compact) for prompt injection. Best-effort: any failure yields an empty string."""
    try:
        from ..memory import predictions as P

        due = P.due_predictions(as_of)
        price_kinds = {"price_above", "price_below", "return_above"}
        tickers = {p.ticker for p in due if p.resolve_kind in price_kinds and p.ticker}
        price_at: dict[str, float] = {}
        for t in tickers:
            px = prices.get(t)
            if not _valid_px(px):
                px = providers.prices.latest_price(t)
            if _valid_px(px):
                price_at[t] = px
        resolved = P.resolve_mechanical(as_of, price_at)
        if resolved:
            progress(f"Resolved {len(resolved)} due prediction(s) against real prices.")
        cal = P.compute_calibration()
        if cal.n_resolved == 0 and cal.n_open == 0:
            return ""
        return P.format_calibration(cal, detailed=False)
    except Exception as e:
        progress(f"(calibration skipped: {e})")
        return ""


def _persist_predictions(brain, progress) -> None:
    """Log the brain's fresh forecasts to the prediction ledger (idempotent)."""
    try:
        from ..memory import predictions as P

        preds = []
        for a in getattr(brain, "assessments", []) or []:
            preds.extend(getattr(a, "predictions", []) or [])
        fresh = P.add_many(preds)
        if fresh:
            progress(f"Logged {len(fresh)} new prediction(s) to the forecast ledger.")
    except Exception as e:
        progress(f"(prediction logging skipped: {e})")


def _recorded_levels(macro) -> dict[str, float]:
    """Index levels to store in nav_history each run: the beat-benchmarks PLUS the
    factor proxies (e.g. Russell 2000) that attribution needs to separate
    selection alpha from factor (size) exposure."""
    labels = list(SETTINGS.benchmarks) + list(SETTINGS.factor_proxies)
    return {label: macro.indicators[label] for label in labels if label in macro.indicators}


def _ensure_prices(providers, prices: dict[str, float], tickers: list[str]) -> None:
    for t in tickers:
        cur = prices.get(t)
        if not _valid_px(cur):
            px = providers.prices.latest_price(t)
            if _valid_px(px):
                prices[t] = px


def _refresh_live_prices(providers, shortlist, prices: dict[str, float], progress) -> None:
    """Overwrite the shortlist's prices with a freshly-fetched current price.

    The bulk daily download can be a day stale (the latest bar's close is often
    NaN), which silently values and trades names on yesterday's price — dangerous
    during a fast move. We refresh the small shortlist so both the analyst's
    valuation and execution use the *current* price."""
    by = {td.ticker: td for td in shortlist}
    if not by:
        return

    def fetch(t):
        return t, providers.prices.latest_price(t)

    n = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for t, px in ex.map(fetch, list(by)):
            if _valid_px(px):
                prices[t] = round(px, 2)
                by[t].price = round(px, 2)
                n += 1
    progress(f"Refreshed live prices for {n}/{len(by)} shortlist names.")


def run_daily(
    as_of: date | None = None,
    dry_run: bool = False,
    on_progress=None,
    force_revaluation: bool = False,
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

    watchlist = load_watchlist()
    if watchlist:
        progress(
            f"Watchlist: {len(watchlist)} favorite(s) always deep-dived "
            f"({', '.join(watchlist)})."
        )
    # Pull price history for any watchlist name outside the index universe too,
    # so favorites still get technicals.
    download_tickers = list(dict.fromkeys(tickers + watchlist))

    progress("Sampling macro snapshot…")
    macro = providers.macro.snapshot()

    progress("Reading macro/Fed policy news (regime only)…")
    market_news = providers.market_news.market_digest(days=4, per_feed=5, macro_only=True)
    progress(f"Pulled {len(market_news)} macro/policy headlines.")

    progress(f"Downloading price history for {len(download_tickers)} tickers…")
    history = providers.prices.history(download_tickers, lookback_days=260)
    metrics = screener.compute_metrics(history, meta)
    progress(f"Computed technicals for {len(metrics)} tickers.")

    # Sector scan: bottom-up (our constituents) + top-down sector-ETF performance
    # (market-cap-weighted, Finviz-style), to find the punished groups so we
    # deliberately hunt where the market may be overreacting.
    etf_perf = sectors.fetch_sector_performance(providers.prices)
    sector_stats = sectors.scan_sectors(metrics, etf_perf=etf_perf)
    punished = sectors.punished_sectors(sector_stats)
    sector_scan = sectors.format_sector_scan(sector_stats)
    if punished:
        progress(f"Sector scan: most punished → {', '.join(punished)}.")

    # Finer grain: GICS sub-industry dispersion (semis vs SaaS within Tech, etc.) —
    # the level where mispricing concentrates and the sector view averages away.
    industry_stats = sectors.scan_industries(metrics)
    punished_industries = sectors.punished_industries(industry_stats)
    industry_scan = sectors.format_industry_scan(industry_stats)
    if punished_industries:
        progress(f"Industry scan: most punished → {', '.join(punished_industries[:4])}.")

    # Liquidity map (avg daily $ volume) for realistic market-impact slippage —
    # essential now that the universe includes thinner small-caps.
    liquidity = {t: td.avg_dollar_volume for t, td in metrics.items() if td.avg_dollar_volume}

    holdings = list(portfolio.positions.keys())
    shortlist_tickers = screener.screen(
        metrics, holdings, SETTINGS.shortlist_size,
        watchlist=watchlist, punished=punished,
        punished_industries=punished_industries,
    )
    progress(
        f"Screened to {len(shortlist_tickers)} candidates "
        f"({len(holdings)} holdings + {len(watchlist)} watchlist included). "
        f"Enriching with fundamentals (news fetched per-finalist after selection)…"
    )
    shortlist = screener.enrich(shortlist_tickers, metrics, providers, with_news=False)

    # Prices for valuation. The bulk daily download can be a day stale (latest
    # bar's close NaN), so refresh the shortlist with fresh live prices before the
    # brain values or we trade anything.
    prices = {t: td.price for t, td in metrics.items() if td.price}
    progress("Refreshing live prices for the shortlist…")
    _refresh_live_prices(providers, shortlist, prices, progress)
    _ensure_prices(providers, prices, holdings + watchlist)

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
    investor_notes = journal.read_investor_notes()
    recent = journal.recent_entries(5)
    theses = journal.load_theses()
    state = _portfolio_state(portfolio, prices)

    # Book-level risk (concentration, market exposure, correlated clusters) so the
    # PM allocates with whole-portfolio risk in view, not just per-name caps. Reuses
    # the price `history` already in hand; only the benchmark series is extra.
    risk_summary = _book_risk_summary(
        state, history, providers, metrics, nav_history, progress
    )

    # Foresight loop: resolve any predictions whose horizon has passed (objective
    # scoring against real prices), then hand the agent its own calibration so it
    # forecasts with discipline this run (and discounts overconfident convictions).
    calibration_summary = _resolve_and_calibrate(providers, prices, as_of, progress)

    progress("Running the 3-stage brain (strategist → analyst → PM)…")
    engine = DecisionEngine()
    brain = engine.decide(
        as_of=as_of,
        portfolio_state=state,
        macro=macro,
        shortlist=shortlist,
        sector_scan=sector_scan,
        industry_scan=industry_scan,
        recent_journal=recent,
        theses=theses,
        performance=perf,
        market_news=market_news,
        holdings=holdings,
        watchlist=watchlist,
        narrative=narrative,
        investor_notes=investor_notes,
        recent_activity=_recent_activity(as_of),
        risk_summary=risk_summary,
        calibration_summary=calibration_summary,
        force_revaluation=force_revaluation,
        news_fetcher=providers.news.headlines,
        toolkit=toolkit,
        on_progress=progress,
    )
    decision = brain.decision
    progress(
        f"Decision: {len(decision.orders)} order(s); "
        f"target cash {decision.target_cash_weight:.0%}."
    )

    # Valuations + reasoning are ANALYSIS outputs — recorded on every run (dry
    # included), independent of whether we trade. Portfolio/ledger/journal below
    # are state and only change on a committed run.
    regime = brain.strategy.regime if brain.strategy else ""
    headlines = {td.ticker.upper(): td.headlines for td in shortlist}
    n_val = valuations.save_many(brain.assessments, as_of, regime, headlines)
    progress(f"Stored {n_val} fresh valuation(s) → data/valuations/.")
    if valuations.write_board():
        progress("Updated market opportunity board → data/opportunities.md.")
    radar_rows = radar.radar_rows(prices=prices)
    if radar.write_radar(prices=prices):
        n_trig = sum(1 for r in radar_rows if r["status"] == "TRIGGERED")
        progress(
            f"Refreshed radar ({len(radar_rows)} on watch, {n_trig} triggered) "
            "→ data/radar.md."
        )
    # Audit: macro read, the sector trend map, and the per-finalist news the agent
    # fetched after selection (headlines now attached to the finalist tickers).
    audit.write_news(as_of, macro, market_news, shortlist, sector_scan)
    audit.write_reasoning(as_of, brain.strategy, brain.assessments, decision)
    if SETTINGS.write_audit:
        progress("Saved macro + sector map + per-name news read → data/news/.")
    if SETTINGS.write_audit:
        progress(f"Wrote full reasoning → data/logs/{as_of.isoformat()}-reasoning.md.")

    if dry_run:
        progress("Dry run — no trades executed; analysis, valuations & reasoning recorded.")
        audit.write_log(as_of, log)
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
            liquidity=liquidity,
            dry_run=True,
            log=log,
        )

    # Make sure we have prices for any ticker the AI wants to trade.
    _ensure_prices(providers, prices, [o.ticker for o in decision.orders])

    trades = execute(portfolio, decision, prices, liquidity)
    progress(f"Executed {len(trades)} trade(s).")

    store.save_portfolio(portfolio)
    store.append_trades(trades)
    journal.record_day(decision, brain.strategy, brain.assessments)
    journal.update_theses(decision)
    # (valuations + reasoning already recorded above, before the dry-run gate.)

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
        benchmarks=_recorded_levels(macro),
    )
    progress(f"Recorded NAV ${nav_after:,.2f}.")
    _auto_export(progress)
    audit.write_log(as_of, log)

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
        liquidity=liquidity,
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

    trades = execute(portfolio, dry.decision, prices, dry.liquidity)
    progress(f"Executed {len(trades)} trade(s).")

    store.save_portfolio(portfolio)
    store.append_trades(trades)
    journal = Journal()
    journal.record_day(dry.decision, dry.strategy, dry.assessments)
    journal.update_theses(dry.decision)
    # Valuations + reasoning were already recorded during the dry analysis pass.
    # Log the agent's fresh forecasts to the prediction ledger (committed runs only,
    # so discarded previews don't create trackable obligations).
    _persist_predictions(dry, progress)

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

    current_benchmarks = _recorded_levels(dry.macro)
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
    audit.write_log(dry.as_of, dry.log)

    dry.trades = trades
    dry.prices = prices
    dry.nav_after = nav_after
    dry.dry_run = False
    return dry
