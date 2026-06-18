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
