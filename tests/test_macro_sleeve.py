"""Defensive macro-hedge sleeve: deterministic guardrail + wiring tests.

No LLM / network — these pin the mechanical behaviour: the aggregate sleeve cap,
that equity orders are untouched by it, that a held hedge eats the sleeve budget,
and that the asset_class tag flows from universe metadata into TickerData."""

from __future__ import annotations

from datetime import date

from ai_investment_buddy import macro_sleeve as ms
from ai_investment_buddy.brain import screener
from ai_investment_buddy.config import SETTINGS
from ai_investment_buddy.engine.execute import execute
from ai_investment_buddy.memory.portfolio import Portfolio
from ai_investment_buddy.models import Action, Decision, Position, TradeOrder

SLEEVE = ms.sleeve_set()
PRICES = {"GLD": 100.0, "SLV": 50.0, "DBC": 25.0, "AAPL": 200.0}


def _order(t, w, action=Action.BUY, conv=5):
    return TradeOrder(ticker=t, action=action, target_weight=w, conviction=conv)


def test_sleeve_set_and_helpers():
    assert ms.is_hedge("gld") and not ms.is_hedge("AAPL")
    assert set(ms.sleeve_tickers()) == SLEEVE
    block = ms.format_sleeve_for_strategist()
    assert "GLD" in block and f"{SETTINGS.max_macro_sleeve_weight:.0%}" in block


def test_sleeve_cap_scales_down_and_preserves_proportions():
    pf = Portfolio(cash=100_000.0, positions={})
    dec = Decision(as_of=date.today(), market_thesis="", orders=[
        _order("GLD", 0.12), _order("SLV", 0.10), _order("AAPL", 0.10),
    ])
    execute(pf, dec, PRICES, sleeve=SLEEVE)
    nav = pf.nav(PRICES)
    gld = pf.positions["GLD"].market_value(PRICES["GLD"]) / nav
    slv = pf.positions["SLV"].market_value(PRICES["SLV"]) / nav
    aapl = pf.positions["AAPL"].market_value(PRICES["AAPL"]) / nav
    # Combined sleeve within the cap (slightly under, from honest buy slippage).
    assert gld + slv <= SETTINGS.max_macro_sleeve_weight + 1e-3
    assert gld + slv > SETTINGS.max_macro_sleeve_weight - 0.01
    # 12:10 proportion preserved by the proportional scale.
    assert abs((gld / slv) - 1.2) < 0.02
    # The equity order is NOT touched by the sleeve cap.
    assert abs(aapl - 0.10) < 0.01


def test_under_cap_no_scaling():
    pf = Portfolio(cash=100_000.0, positions={})
    dec = Decision(as_of=date.today(), market_thesis="", orders=[_order("GLD", 0.05)])
    execute(pf, dec, PRICES, sleeve=SLEEVE)
    gld = pf.positions["GLD"].market_value(PRICES["GLD"]) / pf.nav(PRICES)
    assert abs(gld - 0.05) < 0.005


def test_held_untouched_hedge_eats_the_budget():
    # Already hold ~10% DBC and don't trade it; a new GLD buy gets only the ~5% left.
    pf = Portfolio(cash=90_000.0, positions={
        "DBC": Position(ticker="DBC", shares=400.0, avg_cost=25.0),  # $10k @ $25
    })
    dec = Decision(as_of=date.today(), market_thesis="", orders=[_order("GLD", 0.12)])
    execute(pf, dec, PRICES, sleeve=SLEEVE)
    nav = pf.nav(PRICES)
    total = sum(
        pf.positions[t].market_value(PRICES[t]) / nav
        for t in ("GLD", "DBC") if t in pf.positions
    )
    assert total <= SETTINGS.max_macro_sleeve_weight + 1e-3


def test_no_sleeve_arg_behaves_as_before():
    # Without a sleeve set, hedge tickers are treated like any equity (no cap).
    pf = Portfolio(cash=100_000.0, positions={})
    dec = Decision(as_of=date.today(), market_thesis="", orders=[_order("GLD", 0.18)])
    execute(pf, dec, PRICES)  # no sleeve kwarg
    gld = pf.positions["GLD"].market_value(PRICES["GLD"]) / pf.nav(PRICES)
    # Only the per-name cap (20%) applies.
    assert gld > SETTINGS.max_macro_sleeve_weight


def test_asset_class_flows_from_meta():
    import numpy as np
    import pandas as pd

    idx = pd.date_range("2025-01-01", periods=60, freq="D")
    df = pd.DataFrame({
        "Close": np.linspace(90, 100, 60),
        "Volume": np.full(60, 1_000_000.0),
    }, index=idx)
    history = {"GLD": df, "AAPL": df.copy()}
    meta = {
        "GLD": {"asset_class": "macro_hedge", "sector": "Macro Hedge", "name": "Gold"},
        "AAPL": {"sector": "Information Technology", "name": "Apple"},
    }
    m = screener.compute_metrics(history, meta)
    assert m["GLD"].asset_class == "macro_hedge"
    assert m["AAPL"].asset_class == "equity"
