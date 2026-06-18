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


class ValuationAssessment(BaseModel):
    """Stage 2 output: a disciplined per-name valuation. The PM may only buy a
    name with an acceptable assessment, forcing fair-value discipline."""

    ticker: str
    fair_value: float | None = None  # estimated intrinsic value per share
    current_price: float | None = None
    upside_pct: float | None = None  # (fair_value/price - 1) * 100
    # UNDERVALUED | FAIRLY_VALUED | OVERVALUED
    valuation_verdict: str = "FAIRLY_VALUED"
    quality_score: int = Field(default=3, ge=1, le=5)  # business quality
    margin_of_safety: bool = False
    bull_case: str = ""
    bear_case: str = ""
    key_risks: str = ""
    # BUY | ADD | HOLD | WATCH | TRIM | SELL | AVOID
    recommendation: str = "WATCH"
    suggested_max_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: int = Field(default=3, ge=1, le=5)

    def one_line(self) -> str:
        fv = f"${self.fair_value:.0f}" if self.fair_value else "?"
        px = f"${self.current_price:.0f}" if self.current_price else "?"
        up = f"{self.upside_pct:+.0f}%" if self.upside_pct is not None else "?"
        return (
            f"{self.ticker}: {self.recommendation} | {self.valuation_verdict} | "
            f"fair {fv} vs {px} ({up}) | quality {self.quality_score}/5 | "
            f"MoS={'Y' if self.margin_of_safety else 'N'} | conf {self.confidence}/5"
        )
