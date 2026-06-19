"""Domain models shared across the system.

These are the lingua franca between the data layer, the brain, and the engine.
Kept as pydantic models so they serialize cleanly into memory/JSON and validate
whatever the LLM hands back.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field


# --- Market data -------------------------------------------------------------
class PriceBar(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float


class TickerData(BaseModel):
    """Everything we know about one company on a given run."""

    ticker: str
    name: str | None = None
    sector: str | None = None
    industry: str | None = None

    price: float | None = None
    prev_close: float | None = None
    change_pct: float | None = None  # 1-day % change

    # Derived momentum / technicals (filled by the screener).
    ret_1m: float | None = None
    ret_3m: float | None = None
    ret_6m: float | None = None
    above_50dma: bool | None = None
    above_200dma: bool | None = None
    vol_ratio: float | None = None  # today's volume / avg volume
    # How far below the trailing-1y high we sit, e.g. -35.0 = 35% off the high.
    # The key contrarian signal: a deep drawdown flags a name the market is
    # punishing (which the momentum/mover buckets are blind to).
    drawdown_pct: float | None = None

    # Fundamentals (best-effort; provider-dependent).
    market_cap: float | None = None
    pe: float | None = None
    forward_pe: float | None = None
    peg: float | None = None
    ps: float | None = None
    profit_margin: float | None = None
    revenue_growth: float | None = None
    earnings_growth: float | None = None
    debt_to_equity: float | None = None
    free_cashflow: float | None = None
    target_mean_price: float | None = None
    recommendation: str | None = None

    # Qualitative.
    headlines: list[str] = Field(default_factory=list)

    def one_line(self) -> str:
        """Compact human/LLM-readable summary used in screener output."""
        bits = [self.ticker]
        if self.price is not None:
            bits.append(f"${self.price:.2f}")
        if self.change_pct is not None:
            bits.append(f"{self.change_pct:+.1f}%d")
        if self.ret_3m is not None:
            bits.append(f"3m {self.ret_3m:+.0f}%")
        if self.pe is not None:
            bits.append(f"PE {self.pe:.0f}")
        return " | ".join(bits)


class SectorStat(BaseModel):
    """Aggregate health of one GICS sector, computed from the whole universe.

    This is the cheap, top-down signal that lets us spot a sector being sold off
    *as a group* (low breadth + deeply negative trailing returns) — the kind of
    repricing that single-name momentum/mover screens miss entirely."""

    sector: str
    n: int  # how many names contribute
    ret_1m: float | None = None  # median 1m return across the sector
    ret_3m: float | None = None
    ret_6m: float | None = None
    breadth_200dma: float | None = None  # % of names above their 200dma (0..100)
    median_drawdown: float | None = None  # median % off trailing high
    # Authoritative market-cap-weighted performance from the sector's SPDR ETF
    # (the Finviz-style top-down read), when available.
    etf: str | None = None
    etf_ret_1w: float | None = None
    etf_ret_1m: float | None = None
    etf_ret_3m: float | None = None
    etf_ret_6m: float | None = None
    etf_ret_12m: float | None = None
    etf_ret_ytd: float | None = None
    # Durability label from the long-run trend vs the recent move:
    # durable-up / durable-down / dip-in-uptrend / recovering / choppy.
    # 'dip-in-uptrend' is the prime contrarian entry (strong 6-12m, weak recently).
    trend: str = ""

    def one_line(self) -> str:
        def p(v):
            return f"{v:+.0f}%" if v is not None else "?"

        # Prefer the ETF (market-cap-weighted) numbers as the headline; fall back
        # to median-of-constituents when no ETF is mapped.
        if self.etf:
            out = (
                f"{self.sector} ({self.etf}): 1w {p(self.etf_ret_1w)}, "
                f"1m {p(self.etf_ret_1m)}, 3m {p(self.etf_ret_3m)}, "
                f"6m {p(self.etf_ret_6m)}, 12m {p(self.etf_ret_12m)}, "
                f"YTD {p(self.etf_ret_ytd)}"
            )
            if self.trend:
                out += f" [{self.trend}]"
        else:
            out = (
                f"{self.sector} (n={self.n}): 1m {p(self.ret_1m)}, "
                f"3m {p(self.ret_3m)}, 6m {p(self.ret_6m)}"
            )
        if self.breadth_200dma is not None:
            out += f" | breadth>200dma {self.breadth_200dma:.0f}%"
        if self.median_drawdown is not None:
            out += f" | median drawdown {self.median_drawdown:+.0f}%"
        return out


class MacroSnapshot(BaseModel):
    """Top-down context: indices, rates, vol, commodities, FX."""

    as_of: datetime
    indicators: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


# --- Portfolio & trades ------------------------------------------------------
class Position(BaseModel):
    ticker: str
    shares: float
    avg_cost: float  # average cost basis per share

    def market_value(self, price: float) -> float:
        return self.shares * price

    def unrealized_pnl(self, price: float) -> float:
        return (price - self.avg_cost) * self.shares


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class Trade(BaseModel):
    timestamp: datetime
    ticker: str
    action: Action
    shares: float
    price: float  # fill price including slippage
    value: float  # signed cash impact is handled by the ledger
    rationale: str = ""


# --- The AI's decision -------------------------------------------------------
class TradeOrder(BaseModel):
    """A single instruction emitted by the brain."""

    ticker: str
    action: Action
    # Target weight of NAV for this name AFTER the trade (0..1). The executor
    # translates target weights into share deltas. This is more robust than the
    # model trying to compute share counts itself.
    target_weight: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    conviction: int = Field(default=3, ge=1, le=5)


class Decision(BaseModel):
    """The full output of one daily decision cycle."""

    as_of: date
    market_thesis: str  # top-down read of the day
    orders: list[TradeOrder] = Field(default_factory=list)
    target_cash_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: str = ""  # anything the AI wants its future self to remember


# --- Structured analysis stages (LangGraph brain) ----------------------------
class StrategistView(BaseModel):
    """Stage 1 output: top-down regime read and which names to deep-dive."""

    regime: str  # short label, e.g. "risk-on, disinflationary"
    market_thesis: str
    finalists: list[str] = Field(default_factory=list)
    reasoning: str = ""
    # Top-down sector read: which beaten-down areas look like overreactions
    # (opportunity) vs deserved de-ratings (value traps), and where momentum is
    # crowded. This is the contrarian lens that decides where we go hunting.
    sector_read: str = ""


class ValuationAssessment(BaseModel):
    """Stage 2 output: a disciplined per-name valuation. The PM may only buy a
    name with an acceptable assessment, forcing fair-value discipline."""

    ticker: str
    sector: str = ""  # GICS sector, for the market-wide opportunity board
    # What kind of business this is, which dictates the valuation method:
    # e.g. HYPERGROWTH, COMPOUNDER, VALUE, CYCLICAL, FINANCIAL, REIT, TURNAROUND.
    archetype: str = ""
    valuation_method: str = ""  # the primary method used and why it fits
    fair_value: float | None = None  # probability-weighted expected intrinsic value/share
    current_price: float | None = None
    upside_pct: float | None = None  # (fair_value/price - 1) * 100
    # Downside scenario: the bear-case value and % to it from current price (the risk).
    bear_value: float | None = None
    downside_pct: float | None = None  # (bear_value/price - 1) * 100, negative
    # Reward/risk: expected upside vs downside magnitude. >1 = favourable asymmetry.
    risk_reward: float | None = None
    # UNDERVALUED | FAIRLY_VALUED | OVERVALUED
    valuation_verdict: str = "FAIRLY_VALUED"
    quality_score: int = Field(default=3, ge=1, le=5)  # business quality
    margin_of_safety: bool = False
    # Structural/existential risk to the business model (moat erosion, secular
    # disruption — e.g. AI displacing an incumbent): LOW | MEDIUM | HIGH | SEVERE.
    # A cheap price with SEVERE structural risk is a value trap, not an opportunity.
    structural_risk: str = "MEDIUM"
    # Why the market prices it where it does — the bear case the market is weighting
    # that we must address before claiming it's mispriced.
    why_market_disagrees: str = ""
    # What re-rates the stock and over what horizon (short <6m / medium 6-18m) — we
    # want disconnects that close in short-to-medium term, not someday.
    rerating_catalyst: str = ""
    # What today's price is implying (growth/margins/multiple) — the reverse
    # valuation — and whether the market looks right about it.
    market_implied: str = ""
    # OVERREACTING | UNDERREACTING | FAIR — our call on the market's pricing.
    market_view: str = "FAIR"
    mispricing_thesis: str = ""  # if mispriced: why, and what the market is missing
    # Per-name news due diligence (fetched AFTER selection): the material recent
    # news and how it affects the stock, plus the prevailing sentiment.
    news_sentiment: str = ""  # BULLISH | NEUTRAL | BEARISH
    news_assessment: str = ""  # material news + likely impact + is sentiment overdone?
    bull_case: str = ""
    bear_case: str = ""
    key_risks: str = ""
    # BUY | ADD | HOLD | WATCH | TRIM | SELL | AVOID
    recommendation: str = "WATCH"
    suggested_max_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: int = Field(default=3, ge=1, le=5)
    # Transient runtime flag: True if this came from the valuation cache (a recent
    # assessment reused because nothing material changed) rather than a fresh
    # model call. Not meaningful once persisted.
    from_cache: bool = False

    def one_line(self) -> str:
        fv = f"${self.fair_value:.0f}" if self.fair_value else "?"
        px = f"${self.current_price:.0f}" if self.current_price else "?"
        up = f"{self.upside_pct:+.0f}%" if self.upside_pct is not None else "?"
        arch = f"{self.archetype} | " if self.archetype else ""
        mkt = f" | market {self.market_view}" if self.market_view else ""
        rr = f" | R/R {self.risk_reward}" if self.risk_reward is not None else ""
        down = f" | downside {self.downside_pct:+.0f}%" if self.downside_pct is not None else ""
        sr = f" | struct-risk {self.structural_risk}" if self.structural_risk else ""
        return (
            f"{self.ticker}: {arch}{self.recommendation} | {self.valuation_verdict} | "
            f"fair {fv} vs {px} ({up}){down}{rr} | quality {self.quality_score}/5 | "
            f"MoS={'Y' if self.margin_of_safety else 'N'} | conf {self.confidence}/5{mkt}{sr}"
        )


# --- Persisted per-ticker valuation history ----------------------------------
class StoredValuation(BaseModel):
    """One dated valuation of a company, with the regime context it was made in."""

    as_of: date
    regime: str = ""
    assessment: ValuationAssessment
    # Headline titles considered at analysis time — used to detect whether *new*
    # news has appeared since, which would invalidate a cached valuation.
    news_seen: list[str] = Field(default_factory=list)


class InvestorNote(BaseModel):
    """A durable note from the human investor about a name, captured in dialogue.

    These are injected into future analyst valuations of the ticker, so the user
    can teach the agent (e.g. 'AI is eroding this moat') and have it stick."""

    date: date
    user_view: str
    agent_response: str = ""
    stance: str = ""  # the agent's stance: AGREE | PARTIALLY_AGREE | DISAGREE
    changes_thesis: bool = False  # if True, forces a fresh valuation next run


class ValuationRecord(BaseModel):
    """The full valuation file for ONE ticker — the latest read, a capped history,
    and any investor notes, so we can see how our thesis on a name evolved.

    Persisted one file per ticker (``data/valuations/<TICKER>.json``) so coverage
    of the market accumulates run after run and travels in state snapshots."""

    ticker: str
    first_assessed: date
    last_assessed: date
    latest: StoredValuation
    history: list[StoredValuation] = Field(default_factory=list)
    notes: list[InvestorNote] = Field(default_factory=list)
