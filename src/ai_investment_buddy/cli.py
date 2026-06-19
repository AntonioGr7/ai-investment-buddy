"""Command-line interface for AI Investment Buddy.

  aib init      seed the paper portfolio with starting capital
  aib run       run today's decision (preview, then confirm to execute)
  aib status    show current portfolio and performance
  aib report    show the latest journal entry / decision rationale
  aib history   show NAV history vs benchmarks
"""

from __future__ import annotations

import sys
from datetime import date as date_cls
from datetime import datetime

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pathlib import Path

from .config import SETTINGS
from .engine import commit, run_daily
from .engine.benchmark import performance_summary
from .memory import Journal, snapshot, store

app = typer.Typer(
    add_completion=False,
    help="An AI that allocates a paper portfolio daily and tries to beat the S&P 500 and Nasdaq.",
    no_args_is_help=True,
)
console = Console()


def _require_key() -> None:
    if not SETTINGS.llm_api_key:
        console.print(
            f"[red]{SETTINGS.llm_key_env_name()} is not set[/red] "
            f"(required for AIB_LLM_PROVIDER={SETTINGS.llm_provider}). "
            "Add it to your environment or a .env file in the project root."
        )
        raise typer.Exit(1)


def _latest_prices_for(pf) -> dict[str, float]:
    """Best-effort current prices for held tickers (for status/history)."""
    from .data import get_providers

    if not pf.positions:
        return {}
    providers = get_providers()
    hist = providers.prices.history(list(pf.positions.keys()), lookback_days=10)
    prices = {}
    for t, df in hist.items():
        if df is not None and not df.empty and "Close" in df:
            prices[t] = float(df["Close"].dropna().iloc[-1])
    for t in pf.positions:
        if t not in prices:
            px = providers.prices.latest_price(t)
            if px:
                prices[t] = px
    return prices


# --- Commands ----------------------------------------------------------------
@app.command()
def init(
    capital: float = typer.Option(
        SETTINGS.starting_capital, help="Starting paper capital."
    ),
    force: bool = typer.Option(False, "--force", help="Reset an existing portfolio."),
):
    """Seed the paper portfolio."""
    try:
        pf = store.init_portfolio(capital=capital, force=force)
    except FileExistsError as e:
        console.print(f"[yellow]{e}[/yellow]")
        raise typer.Exit(1)
    console.print(
        Panel(
            f"Portfolio seeded with [bold green]${pf.cash:,.2f}[/bold green] cash.\n"
            f"Benchmarks to beat: {', '.join(SETTINGS.benchmarks)}.",
            title="AI Investment Buddy — initialized",
        )
    )


@app.command()
def run(
    date: str = typer.Option(None, help="Decision date (YYYY-MM-DD). Defaults to today."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Execute without confirmation."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview the AI's decision; never execute."
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-value every finalist even if a recent valuation still holds."
    ),
    no_feedback: bool = typer.Option(
        False, "--no-feedback", help="Skip the post-run feedback dialogue."
    ),
):
    """Run the daily decision cycle."""
    _require_key()
    if not store.is_initialized():
        console.print("[red]No portfolio yet.[/red] Run [bold]aib init[/bold] first.")
        raise typer.Exit(1)

    as_of = (
        datetime.strptime(date, "%Y-%m-%d").date() if date else date_cls.today()
    )

    with console.status("[bold]Running daily cycle…", spinner="dots") as status:
        def progress(msg: str) -> None:
            status.update(f"[bold]{msg}")
            console.log(msg)

        result = run_daily(
            as_of=as_of, dry_run=True, on_progress=progress, force_revaluation=force
        )

    _render_decision(result)
    console.print(
        f"\n[dim]Full reasoning saved to data/logs/{result.as_of.isoformat()}-reasoning.md "
        f"· news read in data/news/{result.as_of.isoformat()}/[/dim]"
    )

    # The agent invites discussion right after the analysis — before any trade.
    if not no_feedback and not yes and sys.stdin.isatty():
        _run_feedback(result)

    if dry_run:
        console.print("\n[dim]Dry run — nothing was executed.[/dim]")
        return

    if not result.decision.orders:
        console.print("\n[dim]No trades proposed. Nothing to execute.[/dim]")
        if yes or typer.confirm("Record this 'no-trade' day to the journal?", default=True):
            commit(result, on_progress=lambda m: console.log(m))
            console.print("[green]Logged.[/green]")
        return

    # Approve all at once, or pick trades individually.
    if yes:
        accepted = list(result.decision.orders)
    else:
        accepted = _select_orders(result.decision.orders)

    if not accepted:
        console.print("[yellow]No trades selected. Nothing executed.[/yellow]")
        return

    result.decision.orders = accepted
    commit(result, on_progress=lambda m: console.log(m))
    console.print("\n[bold green]Done.[/bold green] Executed and recorded.")
    _render_trades(result.trades)


@app.command()
def status():
    """Show the current portfolio and performance vs benchmarks."""
    if not store.is_initialized():
        console.print("[red]No portfolio yet.[/red] Run [bold]aib init[/bold] first.")
        raise typer.Exit(1)

    pf = store.load_portfolio()
    prices = _latest_prices_for(pf)
    nav = pf.nav(prices)

    table = Table(title="Portfolio")
    for col in ("Ticker", "Shares", "Avg cost", "Price", "Value", "Weight", "Unrl PnL"):
        table.add_column(col, justify="right")
    table.add_column("", justify="right")
    for t, pos in sorted(
        pf.positions.items(),
        key=lambda kv: kv[1].market_value(prices.get(kv[0], 0)),
        reverse=True,
    ):
        px = prices.get(t, 0.0)
        mv = pos.market_value(px)
        pnl = pos.unrealized_pnl(px)
        table.add_row(
            t,
            f"{pos.shares:.2f}",
            f"${pos.avg_cost:,.2f}",
            f"${px:,.2f}",
            f"${mv:,.0f}",
            f"{(mv/nav*100 if nav else 0):.1f}%",
            f"${pnl:,.0f}",
            "[green]▲[/green]" if pnl >= 0 else "[red]▼[/red]",
        )
    table.add_row(
        "CASH", "", "", "", f"${pf.cash:,.0f}",
        f"{(pf.cash/nav*100 if nav else 100):.1f}%", "", "",
    )
    console.print(table)

    nav_history = store.load_nav_history()
    benchmarks = _current_benchmark_levels()
    console.print(
        Panel(
            performance_summary(nav_history, nav, benchmarks),
            title=f"Performance — NAV ${nav:,.0f}",
        )
    )


@app.command()
def report():
    """Show the most recent decision rationale (journal entry)."""
    journal = Journal()
    entries = journal.recent_entries(1)
    if not entries:
        console.print("[yellow]No journal entries yet. Run [bold]aib run[/bold].[/yellow]")
        raise typer.Exit(0)
    console.print(Panel(entries[-1], title="Latest decision"))


@app.command()
def export(
    path: str = typer.Argument(
        None, help="Output file. Defaults to ./aib-state-<date>.json."
    ),
):
    """Serialize the entire bot state into one portable snapshot file."""
    if not store.is_initialized():
        console.print("[red]No state to export.[/red] Run [bold]aib init[/bold] first.")
        raise typer.Exit(1)
    out = Path(path) if path else Path.cwd() / f"aib-state-{date_cls.today()}.json"
    snap = snapshot.export_state(out)
    s = snap["summary"]
    console.print(
        Panel(
            f"Exported state to [bold]{out}[/bold]\n"
            f"cash=${s.get('cash', 0):,.0f} | positions={s.get('n_positions', 0)} | "
            f"trades={s.get('n_trades', 0)} | nav_rows={s.get('nav_rows', 0)} | "
            f"journal_days={s.get('journal_days', 0)}\n\n"
            f"[dim]Move this file to another machine, then `aib import {out.name}`.[/dim]",
            title="State exported",
        )
    )


@app.command("import")
def import_(
    path: str = typer.Argument(..., help="Snapshot file produced by `aib export`."),
    force: bool = typer.Option(
        False, "--force", help="Overwrite existing local state."
    ),
):
    """Restore bot state from a portable snapshot file and resume from there."""
    p = Path(path)
    if not p.exists():
        console.print(f"[red]File not found:[/red] {p}")
        raise typer.Exit(1)
    try:
        snap = snapshot.import_state(p, force=force)
    except FileExistsError as e:
        console.print(f"[yellow]{e}[/yellow]")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    s = snap.get("summary", {})
    console.print(
        Panel(
            f"Restored state from [bold]{p}[/bold] "
            f"(exported {snap.get('exported_at', '?')}).\n"
            f"cash=${s.get('cash', 0):,.0f} | positions={s.get('n_positions', 0)} | "
            f"trades={s.get('n_trades', 0)} | journal_days={s.get('journal_days', 0)}\n\n"
            f"[dim]Run `aib status` to confirm, then `aib run` to continue.[/dim]",
            title="State restored",
        )
    )


@app.command()
def history():
    """Show NAV history vs benchmarks."""
    rows = store.load_nav_history()
    if not rows:
        console.print("[yellow]No history yet. Run [bold]aib run[/bold].[/yellow]")
        raise typer.Exit(0)
    table = Table(title="NAV history")
    cols = list(rows[0].keys())
    for c in cols:
        table.add_column(c, justify="right")
    for r in rows:
        table.add_row(*[str(r.get(c, "")) for c in cols])
    console.print(table)


# --- Watchlist ---------------------------------------------------------------
watchlist_app = typer.Typer(
    no_args_is_help=True,
    help="Manage your favorite stocks. Every watchlist name is always run through "
    "the full daily process (enriched, made a finalist, and valued).",
)
app.add_typer(watchlist_app, name="watchlist")


def _print_watchlist(tickers: list[str]) -> None:
    if not tickers:
        console.print(
            "[yellow]Watchlist is empty.[/yellow] Add favorites with "
            "[bold]aib watchlist add TICKER…[/bold]"
        )
        return
    table = Table(title=f"Watchlist ({len(tickers)})")
    table.add_column("#", justify="right")
    table.add_column("ticker")
    for i, t in enumerate(tickers, 1):
        table.add_row(str(i), t)
    console.print(table)


@watchlist_app.command("list")
def watchlist_list():
    """Show the current watchlist."""
    from .watchlist import load_watchlist

    _print_watchlist(load_watchlist())


@watchlist_app.command("add")
def watchlist_add(
    tickers: list[str] = typer.Argument(..., help="One or more tickers, e.g. AAPL NVDA."),
):
    """Add one or more favorites to the watchlist."""
    from . import watchlist as wl

    added = wl.add(tickers)
    if added:
        console.print(f"[green]Added:[/green] {', '.join(added)}")
    else:
        console.print("[yellow]Nothing new to add (already on the watchlist).[/yellow]")
    _print_watchlist(wl.load_watchlist())


@watchlist_app.command("remove")
def watchlist_remove(
    tickers: list[str] = typer.Argument(..., help="One or more tickers to drop."),
):
    """Remove one or more favorites from the watchlist."""
    from . import watchlist as wl

    removed = wl.remove(tickers)
    if removed:
        console.print(f"[green]Removed:[/green] {', '.join(removed)}")
    else:
        console.print("[yellow]None of those were on the watchlist.[/yellow]")
    _print_watchlist(wl.load_watchlist())


# --- Valuations --------------------------------------------------------------
_REC_COLOR = {
    "BUY": "green", "ADD": "green", "HOLD": "yellow", "WATCH": "yellow",
    "TRIM": "red", "SELL": "red", "AVOID": "red",
}
_MKT_COLOR = {"OVERREACTING": "green", "UNDERREACTING": "red", "FAIR": "dim"}


def _assessment_table(assessments, title: str = "Analyst valuations (fair value vs price)") -> Table:
    vt = Table(title=title)
    for col in ("Ticker", "Type", "Rec", "Verdict", "Market", "Fair", "Price", "Upside", "Qual", "MoS", "Conf"):
        vt.add_column(col, justify="right")
    for a in assessments:
        c = _REC_COLOR.get(a.recommendation, "white")
        up = f"{a.upside_pct:+.0f}%" if a.upside_pct is not None else "?"
        up_c = "green" if (a.upside_pct or 0) > 0 else "red"
        mc = _MKT_COLOR.get(a.market_view, "white")
        vt.add_row(
            a.ticker,
            f"[dim]{a.archetype.title()}[/dim]" if a.archetype else "[dim]?[/dim]",
            f"[{c}]{a.recommendation}[/{c}]",
            a.valuation_verdict.replace("_", " ").title(),
            f"[{mc}]{a.market_view.title()}[/{mc}]" if a.market_view else "",
            f"${a.fair_value:.0f}" if a.fair_value else "?",
            f"${a.current_price:.0f}" if a.current_price else "?",
            f"[{up_c}]{up}[/{up_c}]",
            f"{a.quality_score}/5",
            "[green]Y[/green]" if a.margin_of_safety else "[dim]N[/dim]",
            f"{a.confidence}/5",
        )
    return vt


def _render_assessment_detail(a) -> None:
    body = []
    sr_color = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red", "SEVERE": "bold red"}
    up = f"{a.upside_pct:+.0f}%" if a.upside_pct is not None else "?"
    down = f"{a.downside_pct:+.0f}%" if a.downside_pct is not None else "?"
    rr = f"{a.risk_reward}" if a.risk_reward is not None else "?"
    body.append(
        f"[bold]Risk/reward {rr}[/bold]  ·  upside {up}  ·  downside {down}  ·  "
        f"structural risk [{sr_color.get(a.structural_risk,'white')}]{a.structural_risk}[/]"
    )
    if a.why_market_disagrees:
        body.append(f"[bold]Why the market disagrees:[/bold] {a.why_market_disagrees}")
    if a.rerating_catalyst:
        body.append(f"[bold]Catalyst / horizon:[/bold] {a.rerating_catalyst}")
    if a.valuation_method:
        body.append(f"[bold]Method:[/bold] {a.valuation_method}")
    if a.market_implied:
        body.append(f"[bold]Market implies:[/bold] {a.market_implied}")
    if a.mispricing_thesis:
        body.append(f"[bold]Mispricing thesis:[/bold] {a.mispricing_thesis}")
    if a.news_assessment:
        body.append(f"[bold]News ({a.news_sentiment or 'n/a'}):[/bold] {a.news_assessment}")
    body.append(f"[bold]Bull:[/bold] {a.bull_case}")
    body.append(f"[bold]Bear:[/bold] {a.bear_case}")
    body.append(f"[bold]Risks:[/bold] {a.key_risks}")
    mc = _MKT_COLOR.get(a.market_view, "white")
    console.print(
        Panel(
            "\n".join(body),
            title=f"{a.ticker} — {a.archetype.title()} · "
            f"[{mc}]{a.market_view.title()}[/{mc}] · {a.recommendation}",
        )
    )


@app.command()
def valuate(
    tickers: list[str] = typer.Argument(..., help="Ticker(s) to value now, e.g. CRM NOW."),
    watch: bool = typer.Option(False, "--watch", help="Also add these to your watchlist."),
):
    """Force a full fair-value analysis on specific ticker(s) right now.

    Runs the archetype-driven analyst on each name and stores the result to
    data/valuations/ (new file if unseen, updated with history if known)."""
    _require_key()
    from . import watchlist as wl
    from .brain import screener
    from .brain.decide import DecisionEngine
    from .data import get_providers
    from .memory import MemoryToolkit, valuations
    from .universe import get_universe

    tickers = [t for t in dict.fromkeys(wl.normalize(t) for t in tickers) if t]
    if not tickers:
        console.print("[red]No valid tickers given.[/red]")
        raise typer.Exit(1)

    as_of = date_cls.today()
    results = []
    with console.status("[bold]Valuing…", spinner="dots") as status:
        providers = get_providers()
        uni = {c["ticker"]: c for c in get_universe()}
        status.update("Downloading price history…")
        history = providers.prices.history(tickers, lookback_days=260)
        meta = {t: uni.get(t, {"ticker": t}) for t in tickers}
        metrics = screener.compute_metrics(history, meta)
        status.update("Fetching fundamentals + news…")
        enriched = screener.enrich(tickers, metrics, providers)

        regime, thesis = Journal().latest_strategy()
        positions = {}
        if store.is_initialized():
            pf = store.load_portfolio()
            positions = {
                t: {"ticker": t, "shares": round(p.shares, 4), "avg_cost": round(p.avg_cost, 2)}
                for t, p in pf.positions.items()
            }
        toolkit = MemoryToolkit()
        engine = DecisionEngine()
        for td in enriched:
            status.update(f"Analyst valuing {td.ticker}…")
            try:
                dossier = toolkit.ticker_dossier(td.ticker)
            except Exception:
                dossier = ""
            a = engine.valuate(td, regime, thesis, positions.get(td.ticker), dossier)
            if a is None:
                console.log(f"[yellow]{td.ticker}: valuation failed.[/yellow]")
                continue
            valuations.save_valuation(a, as_of=as_of, regime=regime)
            results.append(a)
            if watch:
                wl.add([td.ticker])

    if not results:
        console.print("[red]No valuations produced.[/red]")
        raise typer.Exit(1)
    from .memory import valuations as _v

    _v.write_board()  # keep the market board current
    console.print(_assessment_table(results, title="Valuation"))
    for a in results:
        _render_assessment_detail(a)
    console.print("[dim]Stored to data/valuations/. See the full board with [bold]aib opportunities[/bold].[/dim]")


@app.command()
def opportunities(
    limit: int = typer.Option(0, help="How many names to show (0 = all)."),
    buys_only: bool = typer.Option(False, "--buys", help="Only BUY/ADD-rated names."),
    sector: str = typer.Option(None, "--sector", help="Filter to one sector (substring match)."),
    min_upside: float = typer.Option(None, "--min-upside", help="Only names with at least this %% upside."),
    csv_path: str = typer.Option(None, "--csv", help="Also export the full board to this CSV file."),
):
    """The market-wide opportunity board: every name we've ever valued, ranked by
    cost (price) vs opportunity (fair value / upside / score), to decide what to
    watch. Also kept fresh at data/opportunities.md after every run."""
    from .memory import valuations

    rows = valuations.board_rows()
    if not rows:
        console.print(
            "[yellow]No valuations stored yet.[/yellow] Run [bold]aib run[/bold] or "
            "[bold]aib valuate TICKER[/bold]."
        )
        raise typer.Exit(0)

    if buys_only:
        rows = [r for r in rows if r["recommendation"] in ("BUY", "ADD")]
    if sector:
        rows = [r for r in rows if sector.lower() in (r["sector"] or "").lower()]
    if min_upside is not None:
        rows = [r for r in rows if (r["upside_pct"] or -999) >= min_upside]

    if csv_path:
        import csv as _csv

        with open(csv_path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=valuations.BOARD_COLUMNS)
            w.writeheader()
            w.writerows(rows)
        console.print(f"[green]Exported {len(rows)} rows → {csv_path}[/green]")

    total = len(rows)
    shown = rows if limit in (0, None) else rows[:limit]
    sr_color = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red", "SEVERE": "bold red"}
    table = Table(title=f"Opportunity board — {len(shown)} of {total} valued names (risk-adjusted)")
    for col in ("Score", "Ticker", "Sector", "Rec", "R/R", "Up", "Down", "StructRisk", "Price", "Fair", "MoS", "As of"):
        table.add_column(col, justify="right")
    for r in shown:
        c = _REC_COLOR.get(r["recommendation"], "white")
        up = f"{r['upside_pct']:+.0f}%" if r["upside_pct"] is not None else "?"
        up_c = "green" if (r["upside_pct"] or 0) > 0 else "red"
        down = f"{r['downside_pct']:+.0f}%" if r["downside_pct"] is not None else "?"
        rr = f"{r['risk_reward']:.1f}" if r["risk_reward"] is not None else "?"
        sr = r["structural_risk"] or "?"
        score_c = "green" if r["score"] > 0 else "dim"
        table.add_row(
            f"[{score_c}]{r['score']:+.1f}[/{score_c}]",
            r["ticker"],
            f"[dim]{(r['sector'] or '?')[:12]}[/dim]",
            f"[{c}]{r['recommendation']}[/{c}]",
            rr,
            f"[{up_c}]{up}[/{up_c}]",
            f"[red]{down}[/red]",
            f"[{sr_color.get(sr,'white')}]{sr}[/]",
            f"${r['current_price']:.0f}" if r["current_price"] else "?",
            f"${r['fair_value']:.0f}" if r["fair_value"] else "?",
            "[green]Y[/green]" if r["margin_of_safety"] else "[dim]—[/dim]",
            r["last_assessed"],
        )
    console.print(table)
    console.print(
        "[dim]Risk-adjusted score: reward/risk + downside − structural-risk penalty (a cheap "
        "value trap sinks). Watch one with [bold]aib watchlist add TICKER[/bold]; deep-dive with "
        "[bold]aib valuate TICKER[/bold]. Full board: data/opportunities.md[/dim]"
    )


# --- Trade approval ----------------------------------------------------------
def _order_label(o) -> str:
    return (
        f"{o.action.value} {o.ticker} → target {o.target_weight:.0%} "
        f"(conviction {o.conviction}/5)"
    )


def _select_orders(orders: list):
    """Let the user approve all proposed trades at once, pick them individually,
    or reject everything. Returns the approved subset."""
    console.print(
        f"\n[bold]{len(orders)} proposed trade(s).[/bold] "
        "[A]ll / [S]elect individually / [N]one?"
    )
    choice = typer.prompt("choice", default="A").strip().lower()
    if choice.startswith("n"):
        return []
    if choice.startswith("s"):
        kept = []
        for o in orders:
            if typer.confirm(f"  {_order_label(o)} — include?", default=True):
                kept.append(o)
        return kept
    return list(orders)  # "All" (default)


# --- Feedback dialogue -------------------------------------------------------
def _feedback_context(result) -> str:
    d = result.decision
    lines = [
        f"Regime: {result.strategy.regime if getattr(result, 'strategy', None) else '?'}",
        f"PM thesis: {d.market_thesis}",
    ]
    if d.orders:
        lines.append(
            "Orders: " + "; ".join(
                f"{o.action.value} {o.ticker}→{o.target_weight:.0%}" for o in d.orders
            )
        )
    else:
        lines.append("Orders: none (held).")
    if getattr(result, "assessments", None):
        lines.append("Key valuations:")
        for a in result.assessments[:12]:
            lines.append("  " + a.one_line())
    return "\n".join(lines)


def _run_feedback(result) -> None:
    """Interactive post-run dialogue: the PM discusses the decision, challenges or
    agrees, and stores any durable takeaways into memory."""
    from .brain.decide import DecisionEngine
    from .memory import valuations
    from .models import InvestorNote

    console.print(
        "\n[bold cyan]Feedback[/bold cyan] — tell the PM what you think "
        "(a name, the market, a thesis). It will engage and remember. Empty to skip."
    )
    first = typer.prompt("you", default="", show_default=False)
    if not first.strip():
        console.print("[dim]No feedback. Done.[/dim]")
        return

    engine = DecisionEngine()
    context = _feedback_context(result)
    transcript = [{"role": "investor", "text": first.strip()}]
    ticker_notes: dict[str, tuple[str, bool, str]] = {}  # ticker -> (note, changes, stance)
    market_notes: list[str] = []
    today = date_cls.today()

    while True:
        try:
            with console.status("[bold]PM is thinking…", spinner="dots"):
                payload = engine.discuss(context, transcript)
        except Exception as e:
            console.print(f"[red]Discussion failed: {e}[/red]")
            break
        resp = str(payload.get("response", "")).strip()
        stance = str(payload.get("stance", ""))
        console.print(Panel(resp or "_(no reply)_", title=f"PM · {stance}"))
        transcript.append({"role": "pm", "text": resp})
        for tn in payload.get("ticker_notes", []) or []:
            t = str(tn.get("ticker", "")).upper().strip()
            if t and tn.get("note"):
                ticker_notes[t] = (str(tn["note"]), bool(tn.get("changes_thesis", False)), stance)
        mn = str(payload.get("market_note", "")).strip()
        if mn:
            market_notes.append(mn)

        nxt = typer.prompt("you (empty to finish)", default="", show_default=False)
        if not nxt.strip():
            break
        transcript.append({"role": "investor", "text": nxt.strip()})

    investor_text = " / ".join(t["text"] for t in transcript if t["role"] == "investor")
    stored = []
    for t, (note, changes, stance) in ticker_notes.items():
        ok = valuations.add_note(
            t,
            InvestorNote(
                date=today, user_view=investor_text, agent_response=note,
                stance=stance, changes_thesis=changes,
            ),
        )
        if ok:
            stored.append(t + ("*" if changes else ""))
        else:
            market_notes.append(f"{t}: {note}")  # no valuation yet → keep as market note
    journal = Journal()
    for mn in market_notes:
        journal.append_investor_note(mn, today)

    if stored or market_notes:
        console.print(
            f"[green]Remembered.[/green] Per-name: {', '.join(stored) or 'none'}; "
            f"market notes: {len(market_notes)}.  [dim](* = forces a fresh valuation next run)[/dim]"
        )
    else:
        console.print("[dim]Nothing durable to store.[/dim]")


@app.command()
def feedback():
    """Discuss the latest decision with the PM and record any takeaways."""
    _require_key()
    entries = Journal().recent_entries(1)
    if not entries:
        console.print("[yellow]No decision logged yet.[/yellow] Run [bold]aib run[/bold] first.")
        raise typer.Exit(0)

    class _Ctx:  # minimal shim so _run_feedback can build context from the journal
        decision = type("D", (), {"market_thesis": entries[-1][:1500], "orders": []})()
        strategy = None
        assessments = []

    _run_feedback(_Ctx())


# --- Rendering helpers -------------------------------------------------------
def _current_benchmark_levels() -> dict[str, float]:
    from .data import get_providers

    macro = get_providers().macro.snapshot()
    return {
        label: macro.indicators[label]
        for label in SETTINGS.benchmarks
        if label in macro.indicators
    }


def _render_decision(result) -> None:
    d = result.decision
    console.print(Panel(result.performance_before, title="Where we stand"))

    if getattr(result, "strategy", None):
        s = result.strategy
        console.print(
            Panel(s.market_thesis or "_(none)_", title=f"Strategist — regime: {s.regime}")
        )
        if getattr(s, "sector_read", ""):
            console.print(Panel(s.sector_read, title="Sector read (overreaction vs value trap)"))

    if getattr(result, "assessments", None):
        console.print(_assessment_table(result.assessments))

    console.print(Panel(d.market_thesis or "_(none)_", title=f"PM thesis — {d.as_of}"))

    if d.orders:
        table = Table(title="Proposed orders")
        for col in ("Action", "Ticker", "Target wt", "Conv", "Rationale"):
            table.add_column(col)
        for o in d.orders:
            color = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}.get(o.action.value, "white")
            table.add_row(
                f"[{color}]{o.action.value}[/{color}]",
                o.ticker,
                f"{o.target_weight:.0%}",
                f"{o.conviction}/5",
                (o.rationale[:90] + "…") if len(o.rationale) > 90 else o.rationale,
            )
        console.print(table)
    else:
        console.print(Panel("[bold]No trades.[/bold] The AI chose to hold.", title="Orders"))

    console.print(
        f"[dim]Target cash weight: {d.target_cash_weight:.0%}[/dim]"
    )
    if d.notes:
        console.print(Panel(d.notes, title="Notes to future self"))


def _render_trades(trades) -> None:
    if not trades:
        return
    table = Table(title="Executed trades")
    for col in ("Action", "Ticker", "Shares", "Fill", "Value"):
        table.add_column(col, justify="right")
    for t in trades:
        color = "green" if t.action.value == "BUY" else "red"
        table.add_row(
            f"[{color}]{t.action.value}[/{color}]",
            t.ticker,
            f"{t.shares:.2f}",
            f"${t.price:,.2f}",
            f"${t.value:,.0f}",
        )
    console.print(table)


if __name__ == "__main__":
    app()
