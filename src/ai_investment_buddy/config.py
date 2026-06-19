"""Central configuration for AI Investment Buddy.

Everything that might change (paths, capital, the decision model, screener
sizing, data-provider choices) lives here so the rest of the code reads cleanly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Filesystem layout -------------------------------------------------------
# Repo root = two levels up from this file (src/ai_investment_buddy/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("AIB_DATA_DIR", REPO_ROOT / "data"))
JOURNAL_DIR = DATA_DIR / "journal"
CACHE_DIR = DATA_DIR / "cache"
# Audit trail (the agent's step log + the raw news it read). Diagnostic scratch:
# excluded from state snapshots and safe to delete anytime.
LOGS_DIR = DATA_DIR / "logs"
NEWS_DIR = DATA_DIR / "news"


@dataclass(frozen=True)
class Settings:
    # --- Experiment terms ---
    starting_capital: float = 100_000.0
    base_currency: str = "USD"

    # Benchmarks we are trying to beat.
    benchmarks: dict[str, str] = field(
        default_factory=lambda: {"S&P 500": "^GSPC", "Nasdaq 100": "^NDX"}
    )

    # --- Decision engine (LLM-agnostic) ---
    # Which LLM backend powers the decision: "anthropic" | "openai" | "gemini".
    # "openai" also covers any OpenAI-compatible endpoint via AIB_OPENAI_BASE_URL
    # (OpenRouter, Together, Groq, local servers, ...).
    llm_provider: str = os.getenv("AIB_LLM_PROVIDER", "anthropic").lower()
    max_decision_tokens: int = 16_000

    @property
    def decision_model(self) -> str:
        explicit = os.getenv("AIB_MODEL")
        if explicit:
            return explicit
        return {
            "anthropic": "claude-opus-4-8",
            "openai": "gpt-4o",
            "gemini": "gemini-2.5-pro",
        }.get(self.llm_provider, "claude-opus-4-8")

    @property
    def openai_base_url(self) -> str | None:
        return os.getenv("AIB_OPENAI_BASE_URL")  # None => api.openai.com

    # --- Universe & screener funnel ---
    # The screener reduces the full S&P500 + Nasdaq100 universe down to a
    # shortlist that the model can reason about in depth each day.
    shortlist_size: int = 25
    # Always include current holdings in the deep-dive set, even if they don't
    # make the quant shortlist (the AI must be able to reconsider what it owns).
    always_review_holdings: bool = True

    # How the quant shortlist is split across buckets (fractions of shortlist_size).
    # A *balanced* mix so we surface beaten-down value (oversold + punished-sector
    # names) alongside trend/news, instead of drowning contrarian setups in
    # momentum noise. momentum=trend leaders, movers=today's news-driven jumps,
    # oversold=deepest drawdowns, sector=names inside the most-punished sectors.
    screener_mix: dict[str, float] = field(
        default_factory=lambda: {
            "momentum": 0.30,
            "movers": 0.20,
            "oversold": 0.25,
            "sector": 0.25,
        }
    )
    # How many of the worst-performing sectors count as "punished" (the sector
    # bucket fishes inside these).
    punished_sector_count: int = 4

    # Max tool-use rounds for the analyst's agentic valuation loop (it calls the
    # DCF / reverse-DCF / exit-multiple calculators, then submits). Caps cost.
    analyst_max_iters: int = 6

    # --- Valuation freshness ---
    # Don't re-run a full valuation on a name assessed within this many days IF
    # nothing material changed (no new headlines, price hasn't moved much, and no
    # investor feedback challenged the thesis). `aib run --force` overrides.
    valuation_ttl_days: int = 5
    # A price move beyond this (fraction) since the last valuation forces a fresh
    # look — the upside vs fair value has materially changed.
    revaluation_price_move: float = 0.10

    # --- Risk guardrails (the AI is asked to respect these; execution enforces) ---
    max_position_weight: float = 0.20  # no single name above 20% of NAV
    min_trade_value: float = 250.0  # ignore dust trades
    cash_floor: float = 0.0  # can go fully invested; never below 0 (no leverage)

    # --- Trading frictions (paper realism) ---
    commission_per_trade: float = 0.0
    slippage_bps: float = 5.0  # 0.05% applied against you on each fill

    # --- Data providers (swap implementations without touching callers) ---
    price_provider: str = os.getenv("AIB_PRICE_PROVIDER", "yfinance")
    fundamentals_provider: str = os.getenv("AIB_FUNDAMENTALS_PROVIDER", "yfinance")
    news_provider: str = os.getenv("AIB_NEWS_PROVIDER", "yfinance")
    macro_provider: str = os.getenv("AIB_MACRO_PROVIDER", "yfinance")
    market_news_provider: str = os.getenv("AIB_MARKET_NEWS_PROVIDER", "rss")

    # --- State snapshots ---
    # After each committed run, auto-export a portable snapshot of all state.
    auto_export: bool = os.getenv("AIB_AUTO_EXPORT", "1").lower() not in (
        "0", "false", "no", "off",
    )

    # Write a per-run audit trail (step log + raw news read) under data/logs and
    # data/news. Purely diagnostic; never affects decisions or snapshots.
    write_audit: bool = os.getenv("AIB_AUDIT", "1").lower() not in (
        "0", "false", "no", "off",
    )

    @property
    def snapshot_path(self) -> Path:
        return Path(os.getenv("AIB_SNAPSHOT_PATH", str(REPO_ROOT / "aib-state.json")))

    @property
    def anthropic_api_key(self) -> str | None:
        return os.getenv("ANTHROPIC_API_KEY")

    @property
    def openai_api_key(self) -> str | None:
        return os.getenv("OPENAI_API_KEY")

    @property
    def gemini_api_key(self) -> str | None:
        return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    @property
    def llm_api_key(self) -> str | None:
        return {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "gemini": self.gemini_api_key,
        }.get(self.llm_provider)

    def llm_key_env_name(self) -> str:
        return {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }.get(self.llm_provider, "ANTHROPIC_API_KEY")


SETTINGS = Settings()


def ensure_dirs() -> None:
    """Create the runtime data directories if they don't exist."""
    for d in (DATA_DIR, JOURNAL_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
