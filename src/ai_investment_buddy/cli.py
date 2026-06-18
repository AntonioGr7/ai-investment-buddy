"""Command-line interface for AI Investment Buddy.

  aib init      seed the paper portfolio with starting capital
  aib run       run today's decision (preview, then confirm to execute)
  aib status    show current portfolio and performance
  aib report    show the latest journal entry / decision rationale
  aib history   show NAV history vs benchmarks
"""

from __future__ import annotations

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

        result = run_daily(as_of=as_of, dry_run=True, on_progress=progress)

    _render_decision(result)

    if dry_run:
        console.print("\n[dim]Dry run — nothing was executed.[/dim]")
        return

    if not result.decision.orders:
        console.print("\n[dim]No trades proposed. Nothing to execute.[/dim]")
        # Still record the journal + NAV so the day is logged.
        if yes or typer.confirm("Record this 'no-trade' day to the journal?", default=True):
            commit(result, on_progress=lambda m: console.log(m))
            console.print("[green]Logged.[/green]")
        return

    if not yes:
        if not typer.confirm("\nExecute these trades?", default=True):
            console.print("[yellow]Aborted. No state changed.[/yellow]")
            return

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

    if getattr(result, "assessments", None):
        vt = Table(title="Analyst valuations (fair value vs price)")
        for col in ("Ticker", "Rec", "Verdict", "Fair", "Price", "Upside", "Qual", "MoS", "Conf"):
            vt.add_column(col, justify="right")
        rec_color = {
            "BUY": "green", "ADD": "green", "HOLD": "yellow", "WATCH": "yellow",
            "TRIM": "red", "SELL": "red", "AVOID": "red",
        }
        for a in result.assessments:
            c = rec_color.get(a.recommendation, "white")
            up = f"{a.upside_pct:+.0f}%" if a.upside_pct is not None else "?"
            up_c = "green" if (a.upside_pct or 0) > 0 else "red"
            vt.add_row(
                a.ticker,
                f"[{c}]{a.recommendation}[/{c}]",
                a.valuation_verdict.replace("_", " ").title(),
                f"${a.fair_value:.0f}" if a.fair_value else "?",
                f"${a.current_price:.0f}" if a.current_price else "?",
                f"[{up_c}]{up}[/{up_c}]",
                f"{a.quality_score}/5",
                "[green]Y[/green]" if a.margin_of_safety else "[dim]N[/dim]",
                f"{a.confidence}/5",
            )
        console.print(vt)

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
