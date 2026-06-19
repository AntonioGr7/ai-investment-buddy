# AI Investment Buddy

An AI that manages a **paper portfolio**, deciding day by day how to allocate a
fixed pot of capital across the S&P 500 + Nasdaq-100 universe. The experiment:
**can it beat the S&P 500 and the Nasdaq 100 over time?**

Each run it ingests fresh data, updates its memory, and decides to buy, sell,
hold, or sit in cash — recording every decision and its reasoning so the
experiment is fully auditable.

> Paper trading only. No real money, no broker. Decisions are marked against
> real market prices. Not investment advice.

## How it works

A daily cycle (`aib run`):

1. **Ingest** — prices, fundamentals, macro, and news/headlines for the whole
   S&P 500 + Nasdaq-100 universe (~500 names).
2. **Sector scan** — a per-sector performance map (Finviz-style), combining a
   top-down read from the **SPDR sector ETFs** (market-cap-weighted 1w/1m/3m/6m/
   YTD returns for Technology, Communication Services, Consumer Cyclical,
   Healthcare, Financials, Energy, …) with a bottom-up aggregation of our
   constituents (breadth, median drawdown), ranked worst-first. This is the
   contrarian radar: it surfaces whole sectors the market is repricing lower
   (e.g. a software sell-off on AI-disruption fear) that single-name screens miss.
3. **Screen** — a cheap quant funnel narrows the universe to a shortlist (~25)
   from a *balanced* mix of buckets so contrarian setups aren't drowned out:
   momentum (trend leaders), movers (today's news-driven jumps), **oversold**
   (the deepest drawdowns from their 1y high), and **beaten-down-sector** names
   (from the punished sectors above). Current holdings **and your watchlist**
   are always carried through. *Only the shortlist gets the expensive
   fundamentals + news lookups.*
4. **Decide** — a **3-stage LangGraph brain** runs:
   - **Strategist** reads the macro/news regime *and the sector scan*, and
     decides which beaten-down areas are overreactions (opportunity) vs deserved
     de-ratings (value traps), then picks finalists accordingly. Queries memory.
   - **Analyst** classifies each name (hyper-growth / compounder / value /
     cyclical / financial / REIT / turnaround) and values it with the method
     that *fits*. It has real **valuation calculators** it drives with its own
     estimated inputs — a two-stage **DCF**, a **reverse-DCF** (what growth does
     today's price imply?), and an **exit-multiple** model — so the fair value is
     computed, not eyeballed. The reverse-DCF is the cleanest "is the market
     over/under-reacting?" test. Sell-side targets are a lagging cross-check, not
     an anchor.
   - **PM** allocates as target weights — and may only buy names with an
     acceptable valuation. Doing nothing / holding cash is always allowed.
5. **Execute (paper)** — orders become trades against live prices, enforcing
   risk guardrails (max 20% per name, no leverage, no shorting) plus slippage.
6. **Record & consolidate** — portfolio, trade ledger, NAV history (with
   benchmark levels), a written journal entry, and an updated rolling
   **narrative** are all persisted under `data/`.

### Memory

The experiment's edge is continuity. Everything lives in `data/`:

- `portfolio.json` — current cash + positions
- `trades.jsonl` — append-only trade ledger
- `nav_history.csv` — NAV + benchmark index levels per run (for honest scoring)
- `journal/YYYY-MM-DD.md` — the day's regime read, valuations, orders, notes
- `journal/theses.json` — living per-ticker theses the AI revises over time
- `journal/narrative.md` — a consolidated long-horizon story of the portfolio,
  rewritten each day and always loaded into the prompt as cheap long memory
- `valuations/<TICKER>.json` — one accumulating file per company we've ever
  valued (latest read + capped history + your notes); see below
- `opportunities.md` — the market-wide board (every valued name, ranked), rewritten each run
- `journal/investor_notes.md` — durable market-wide notes from your feedback
- `watchlist.jsonl` — your favorite stocks (see below)

Instead of blindly dumping history into every prompt, the agent has **read-only
memory tools** (`search_memory`, `read_journal`, `ticker_dossier`,
`list_journal_days`) and can *pull* exactly the past context it needs — the
filesystem-as-memory pattern. This scales as history accumulates.

### Audit trail — see exactly what the agent read

Each run also writes a diagnostic trail so you can understand *why* it decided
what it did:

- `data/logs/<date>.log` — the agent's step-by-step run log (incl. each DCF /
  reverse-DCF the analyst ran).
- `data/logs/<date>-reasoning.md` — the **full reasoning**: the strategist's
  regime + sector read, every per-name valuation (method, market-implied,
  mispricing thesis, bull/bear/risks), and the PM's rationale per order.
- `data/news/<date>/` — the raw inputs it pulled from the web: `macro_market_news.md`
  (Fed/macro + market headlines with sources and summaries), `company_news.md`
  (per-shortlist-name headlines), `macro_snapshot.md` (index/rate/vol levels), and
  `sector_performance.md` (the per-sector ETF returns + breadth map).

All of this is written on **every** run, including `--dry-run`.

These are **scratch**: gitignored, excluded from state snapshots, and safe to
delete anytime. Disable with `AIB_AUDIT=0`.

### Watchlist (favorites)

The quant screener picks a *different* shortlist each day. Your **watchlist** is
the set of names you always want looked at, no matter what the screener thinks.
Every watchlist ticker is forced through the **entire** daily process: price
history + fundamentals + news are fetched for it, it is always made a strategist
finalist, and the analyst always produces a full fair-value assessment — even if
it would never have made the quant cut. Watchlist names can sit outside the
S&P 500 / Nasdaq-100 universe too.

It is part of the bot's state (stored at `data/watchlist.jsonl`, included in
export/import snapshots) and is optional — an empty watchlist just means no
favorites. Manage it with `aib watchlist add|list|remove`.

### Valuation memory & the opportunity board

Every fair-value assessment is persisted to `data/valuations/<TICKER>.json` — a
new file the first time we look at a name, an update (with the prior read kept in
a capped history) thereafter. So our coverage of the market accumulates run
after run, and the files travel in export/import snapshots.

That corpus powers a **market-wide opportunity board**: `aib opportunities` lists
*every* name we've ever valued — cost (price) vs opportunity (fair value, upside,
score) — ranked so you can decide what to watch across the whole market, not just
today's shortlist. The score blends recommendation, upside to a conservative fair
value, quality, margin of safety, conviction, and whether the market looks to be
over-reacting. Filter with `--buys`, `--sector`, `--min-upside`, cap with
`--limit`, or dump the full thing with `--csv board.csv`. The same board is also
written to **`data/opportunities.md`** after every run, so there's always a
current table to open. Found one you like? `aib watchlist add TICKER`.

You can also force a deep-dive on demand: `aib valuate CRM NOW` runs the full
archetype-driven analyst on those names right now and stores the results (add
`--watch` to also pin them to your watchlist).

**Freshness / no wasted re-analysis.** Each valuation records its date and the
headlines it was based on. On the next run, a name is *not* re-valued if it was
assessed within `valuation_ttl_days` and nothing material changed — no new
headlines, price hasn't moved past `revaluation_price_move`, and no investor
feedback challenged the thesis. `aib run --force` overrides and re-values
everything; `aib valuate` is always a fresh look.

### Feedback dialogue — teaching the agent

After a committed run, the PM asks for your feedback and you can actually argue
with it: tell it _"I think the market is right on Adobe — the AI wave is eroding
their moat"_ and it will push back, concede, or refine, like a colleague. Durable
takeaways are stored:

- **Per-name views** attach to that ticker's valuation file and are injected into
  every future valuation of it — and if the view changes the thesis, the next run
  re-values the name from scratch.
- **Market-wide views** go to `journal/investor_notes.md`, always loaded into the
  strategist and PM prompts.

It prompts **right after the analysis** (before the trade step), so you can react
before anything executes — `--no-feedback` to skip, or run it anytime with
`aib feedback`. When you do execute, you can approve **all** trades at once or
**select** them individually.

### Portability

State is one thing you carry between machines:

```bash
uv run aib export                 # writes ./aib-state-<date>.json (the whole bot)
# ...move the file to another machine that has the code...
uv run aib import aib-state-<date>.json
uv run aib run                    # resumes exactly where it left off
```

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # then add your LLM API key
```

Pick your brain via `AIB_LLM_PROVIDER` (`anthropic` | `openai` | `gemini`) and
set the matching key. `openai` also covers any OpenAI-compatible endpoint
(OpenRouter, Together, Groq, local) via `AIB_OPENAI_BASE_URL`.

## Usage

```bash
uv run aib init                 # seed the portfolio ($100,000 by default)
uv run aib run                  # preview today's decision, confirm to execute
uv run aib run --dry-run        # preview only; never touches state
uv run aib run --yes            # execute without the confirmation prompt
uv run aib run --force          # re-value every finalist (ignore recent valuations)
uv run aib feedback             # discuss the latest decision with the PM
uv run aib status               # current portfolio + performance vs benchmarks
uv run aib report               # the latest decision's full rationale
uv run aib history              # NAV history vs benchmarks
uv run aib export [file]        # serialize the whole bot to a portable snapshot
uv run aib import <file>        # restore state on another machine and resume
uv run aib watchlist add NVDA AAPL   # add favorites (always deep-dived in full)
uv run aib watchlist list            # show the watchlist
uv run aib watchlist remove NVDA     # drop a favorite
uv run aib valuate CRM NOW           # force a full valuation on specific tickers
uv run aib opportunities             # the market-wide board: every name ever valued
uv run aib opportunities --buys --sector Tech --csv board.csv   # filter + export
```

Run `aib run` once per day (manually for now). When you trust it, it can be
scheduled to run automatically.

## Architecture

```
src/ai_investment_buddy/
  config.py        settings, capital, guardrails, screener mix, valuation freshness
  models.py        domain models (Position, Trade, Decision, TickerData, ValuationRecord, ...)
  universe.py      S&P 500 + Nasdaq-100 constituents (cached)
  watchlist.py     your favorite tickers, always deep-dived in full
  data/            swappable data layer (prices, fundamentals, macro, market news)
  memory/          portfolio, store, journal, valuations, memory tools, snapshot export/import
  brain/           screener + sector scan + LangGraph 3-stage brain (graph, prompts, llm, mem_tools, valuation_tools, consolidate)
  engine/          paper execution, benchmarking, daily pipeline
  cli.py           the `aib` command
```

Two seams are designed for swapping without touching callers:

- **Data providers** — implement the Protocols in `data/base.py` and register in
  `data/__init__.py` to add paid sources (Polygon, FMP, NewsAPI, FRED, ...).
- **LLM backends** — add a client in `brain/llm.py` satisfying `LLMClient`.

## Status

v1: end-to-end paper-trading loop with manual daily trigger. Roadmap ideas:
broker paper-account execution, richer data sources, automated scheduling, and a
performance dashboard.
