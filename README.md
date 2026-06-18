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
2. **Screen** — a cheap quant funnel ranks the universe on momentum + notable
   movers and narrows it to a shortlist (~25), always including current
   holdings. *This is what keeps each run fast and cheap — only the shortlist
   gets the expensive fundamentals + news lookups.*
3. **Decide** — a **3-stage LangGraph brain** runs:
   - **Strategist** reads the macro/news regime and picks finalists (always
     including current holdings). It can query its own memory first.
   - **Analyst** produces a disciplined per-name fair-value assessment in
     parallel (fair value vs price, margin of safety, bull/bear, BUY/WATCH/AVOID).
   - **PM** allocates as target weights — and may only buy names with an
     acceptable valuation. Doing nothing / holding cash is always allowed.
4. **Execute (paper)** — orders become trades against live prices, enforcing
   risk guardrails (max 20% per name, no leverage, no shorting) plus slippage.
5. **Record & consolidate** — portfolio, trade ledger, NAV history (with
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

Instead of blindly dumping history into every prompt, the agent has **read-only
memory tools** (`search_memory`, `read_journal`, `ticker_dossier`,
`list_journal_days`) and can *pull* exactly the past context it needs — the
filesystem-as-memory pattern. This scales as history accumulates.

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
uv run aib status               # current portfolio + performance vs benchmarks
uv run aib report               # the latest decision's full rationale
uv run aib history              # NAV history vs benchmarks
uv run aib export [file]        # serialize the whole bot to a portable snapshot
uv run aib import <file>        # restore state on another machine and resume
```

Run `aib run` once per day (manually for now). When you trust it, it can be
scheduled to run automatically.

## Architecture

```
src/ai_investment_buddy/
  config.py        settings, capital, guardrails, provider selection
  models.py        domain models (Position, Trade, Decision, TickerData, ...)
  universe.py      S&P 500 + Nasdaq-100 constituents (cached)
  data/            swappable data layer (prices, fundamentals, macro, market news)
  memory/          portfolio, store, journal, memory tools, snapshot export/import
  brain/           screener + LangGraph 3-stage brain (graph, prompts, llm, mem_tools, consolidate)
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
