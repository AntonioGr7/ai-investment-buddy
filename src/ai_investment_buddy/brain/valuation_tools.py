"""Deterministic valuation calculators the analyst can call mid-reasoning.

The model is good at *estimating inputs* (growth, margins, a sensible discount
rate) but bad at arithmetic over many years. So we hand it precise calculators —
DCF, reverse-DCF, exit-multiple — and let it drive them with its own judged
assumptions, run bull/base/bear sensitivities, and ground its fair value in the
math rather than eyeballing it.

All tools are pure functions: no network, no state. They are exposed to the brain
in the standard ``{name, description, input_schema}`` tool shape and dispatched by
``make_valuation_executor`` inside the analyst's agentic loop.

Unit convention (stated to the model in the schemas): cash flows, market cap,
net debt and share counts are all in **millions** and the *same* currency, so a
per-share value comes out in plain currency units. Rates are **percent**
(e.g. 9.5 means 9.5%).
"""

from __future__ import annotations

import json
from typing import Callable


# --- Core math ---------------------------------------------------------------
def _project_ev(
    base_cash_flow: float,
    growth_rate: float,
    growth_years: int,
    terminal_growth: float,
    discount_rate: float,
) -> tuple[float, float, float]:
    """Two-stage DCF enterprise value (before net debt). Returns
    (enterprise_value, pv_of_explicit_fcf, pv_of_terminal_value)."""
    dr = discount_rate / 100.0
    g = growth_rate / 100.0
    tg = terminal_growth / 100.0
    if dr <= tg:
        raise ValueError("discount_rate must exceed terminal_growth (else value is infinite).")
    if growth_years < 1:
        raise ValueError("growth_years must be >= 1.")
    cf = float(base_cash_flow)
    pv_explicit = 0.0
    for yr in range(1, int(growth_years) + 1):
        cf *= 1 + g
        pv_explicit += cf / (1 + dr) ** yr
    terminal_value = cf * (1 + tg) / (dr - tg)
    pv_terminal = terminal_value / (1 + dr) ** int(growth_years)
    return pv_explicit + pv_terminal, pv_explicit, pv_terminal


def dcf_two_stage(
    base_cash_flow: float,
    growth_rate: float,
    growth_years: int,
    terminal_growth: float,
    discount_rate: float,
    shares_outstanding: float,
    net_debt: float = 0.0,
) -> dict:
    """Two-stage discounted cash flow → intrinsic value per share."""
    if shares_outstanding <= 0:
        raise ValueError("shares_outstanding must be > 0.")
    ev, pv_explicit, pv_terminal = _project_ev(
        base_cash_flow, growth_rate, growth_years, terminal_growth, discount_rate
    )
    equity_value = ev - float(net_debt)
    return {
        "fair_value_per_share": round(equity_value / shares_outstanding, 2),
        "enterprise_value": round(ev, 1),
        "equity_value": round(equity_value, 1),
        "pv_explicit_fcf": round(pv_explicit, 1),
        "pv_terminal_value": round(pv_terminal, 1),
        "terminal_pct_of_value": round(100 * pv_terminal / ev, 1) if ev else None,
        "assumptions": {
            "base_cash_flow": base_cash_flow,
            "growth_rate_pct": growth_rate,
            "growth_years": growth_years,
            "terminal_growth_pct": terminal_growth,
            "discount_rate_pct": discount_rate,
            "net_debt": net_debt,
            "shares_outstanding": shares_outstanding,
        },
    }


def reverse_dcf(
    market_cap: float,
    base_cash_flow: float,
    growth_years: int,
    terminal_growth: float,
    discount_rate: float,
    net_debt: float = 0.0,
) -> dict:
    """Solve for the high-growth rate the CURRENT price is implying.

    The key 'is the market right?' tool: compare the implied growth to what the
    business can plausibly deliver. Implied >> plausible → priced for perfection;
    implied << plausible → the market may be over-reacting."""
    target_ev = float(market_cap) + float(net_debt)
    if target_ev <= 0:
        raise ValueError("market_cap + net_debt must be > 0.")

    def ev_of(g_pct: float) -> float:
        ev, _, _ = _project_ev(base_cash_flow, g_pct, growth_years, terminal_growth, discount_rate)
        return ev

    lo, hi = -50.0, 200.0
    # EV is monotonic increasing in growth; check the implied rate is bracketed.
    if ev_of(lo) > target_ev:
        return {"implied_growth_rate_pct": None, "note": "Price implies decline steeper than -50%/yr."}
    if ev_of(hi) < target_ev:
        return {"implied_growth_rate_pct": None, "note": "Price implies growth above 200%/yr."}
    for _ in range(100):
        mid = (lo + hi) / 2
        if ev_of(mid) < target_ev:
            lo = mid
        else:
            hi = mid
    implied = round((lo + hi) / 2, 1)
    return {
        "implied_growth_rate_pct": implied,
        "interpretation": (
            f"At the current price the market is pricing ~{implied}% annual cash-flow "
            f"growth for {growth_years} years (then {terminal_growth}% perpetual)."
        ),
        "assumptions": {
            "market_cap": market_cap,
            "base_cash_flow": base_cash_flow,
            "growth_years": growth_years,
            "terminal_growth_pct": terminal_growth,
            "discount_rate_pct": discount_rate,
            "net_debt": net_debt,
        },
    }


def probability_weighted_value(scenarios: list, current_price: float | None = None) -> dict:
    """Probability-weight bear/base/bull scenario values into one expected value —
    mirroring how the MARKET prices a stock (the consensus of all outcomes, weighted).

    Each scenario is {label, value, probability}; probabilities are normalised by
    their sum (percent or fraction both fine). Returns the expected value and, vs
    the current price, the expected upside, the DOWNSIDE to the worst scenario, and
    the reward/risk ratio — so a name is judged on risk-adjusted asymmetry, not raw
    upside."""
    if not scenarios:
        raise ValueError("provide at least one scenario")
    total_p = 0.0
    for s in scenarios:
        p = float(s.get("probability", 0) or 0)
        if p < 0:
            raise ValueError("probabilities must be non-negative")
        total_p += p
    if total_p <= 0:
        raise ValueError("probabilities must sum to > 0")

    ev = sum(float(s["value"]) * float(s.get("probability", 0) or 0) for s in scenarios) / total_p
    values = [float(s["value"]) for s in scenarios]
    out = {
        "expected_value": round(ev, 2),
        "worst_case_value": round(min(values), 2),
        "best_case_value": round(max(values), 2),
        "normalized_probabilities": {
            str(s.get("label", i)): round(float(s.get("probability", 0) or 0) / total_p, 3)
            for i, s in enumerate(scenarios)
        },
    }
    if current_price and current_price > 0:
        exp_up = (ev / current_price - 1) * 100
        downside = (min(values) / current_price - 1) * 100
        out["expected_upside_pct"] = round(exp_up, 1)
        out["downside_pct"] = round(downside, 1)
        out["best_case_upside_pct"] = round((max(values) / current_price - 1) * 100, 1)
        # Reward/risk: expected upside vs magnitude of downside to the worst case.
        down_mag = abs(min(0.0, downside))
        up_mag = max(0.0, exp_up)
        out["reward_risk"] = round(up_mag / down_mag, 2) if down_mag > 1e-9 else None
    return out


def exit_multiple(
    base_metric_per_share: float,
    growth_rate: float,
    years: int,
    exit_multiple: float,
    discount_rate: float,
) -> dict:
    """Earnings-power valuation: grow a per-share metric (EPS, FFO, FCF/share) for
    N years, apply a justified exit multiple, discount back to today. Good for
    financials (EPS × P/E), REITs (FFO × P/FFO) and mid-cycle cyclical earnings."""
    if years < 1:
        raise ValueError("years must be >= 1.")
    g = growth_rate / 100.0
    dr = discount_rate / 100.0
    metric_future = float(base_metric_per_share) * (1 + g) ** int(years)
    value_future = metric_future * float(exit_multiple)
    pv = value_future / (1 + dr) ** int(years)
    return {
        "fair_value_per_share": round(pv, 2),
        "metric_in_year_n": round(metric_future, 2),
        "undiscounted_value": round(value_future, 2),
        "assumptions": {
            "base_metric_per_share": base_metric_per_share,
            "growth_rate_pct": growth_rate,
            "years": years,
            "exit_multiple": exit_multiple,
            "discount_rate_pct": discount_rate,
        },
    }


# --- Tool specs (LLM-facing) -------------------------------------------------
_MILLIONS = "(in millions, same currency)"

VALUATION_TOOL_SPECS: list[dict] = [
    {
        "name": "dcf_two_stage",
        "description": (
            "Two-stage discounted cash flow → intrinsic value per share. Project free "
            "cash flow (or owner earnings) at your estimated growth for a high-growth "
            "phase, then a perpetual terminal growth, discounted at your required "
            "return. Use your own judged inputs; run bull/base/bear cases."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "base_cash_flow": {"type": "number", "description": f"Starting annual FCF/owner earnings {_MILLIONS}."},
                "growth_rate": {"type": "number", "description": "High-growth-phase annual growth, percent (e.g. 12)."},
                "growth_years": {"type": "integer", "description": "Length of the high-growth phase in years (e.g. 10)."},
                "terminal_growth": {"type": "number", "description": "Perpetual growth after the high-growth phase, percent (e.g. 2.5). Must be below discount_rate."},
                "discount_rate": {"type": "number", "description": "Required return / WACC, percent (e.g. 9)."},
                "shares_outstanding": {"type": "number", "description": f"Diluted shares {_MILLIONS}."},
                "net_debt": {"type": "number", "description": f"Net debt {_MILLIONS} (use a NEGATIVE value for net cash). Default 0."},
            },
            "required": ["base_cash_flow", "growth_rate", "growth_years", "terminal_growth", "discount_rate", "shares_outstanding"],
        },
    },
    {
        "name": "reverse_dcf",
        "description": (
            "Solve for the annual growth rate the CURRENT price implies. Compare it to "
            "what the business can plausibly deliver to judge whether the market is "
            "over-reacting (implied growth too low) or pricing perfection (too high)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "market_cap": {"type": "number", "description": f"Current equity market cap {_MILLIONS}."},
                "base_cash_flow": {"type": "number", "description": f"Starting annual FCF/owner earnings {_MILLIONS}."},
                "growth_years": {"type": "integer", "description": "High-growth-phase length in years."},
                "terminal_growth": {"type": "number", "description": "Perpetual growth after, percent. Must be below discount_rate."},
                "discount_rate": {"type": "number", "description": "Required return / WACC, percent."},
                "net_debt": {"type": "number", "description": f"Net debt {_MILLIONS} (negative for net cash). Default 0."},
            },
            "required": ["market_cap", "base_cash_flow", "growth_years", "terminal_growth", "discount_rate"],
        },
    },
    {
        "name": "probability_weighted_value",
        "description": (
            "Probability-weight your bear/base/bull scenario values into one expected "
            "value — the way the MARKET prices a stock. Returns expected value, the "
            "DOWNSIDE to your worst case, and the reward/risk ratio vs the current "
            "price. Use this to judge risk-adjusted asymmetry, and to check whether a "
            "big gap to the market price survives once you weight the bear case honestly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "scenarios": {
                    "type": "array",
                    "description": "Bear/base/bull (or more) outcomes. Probabilities should reflect "
                    "real likelihoods incl. structural-risk scenarios the market is pricing.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string", "description": "e.g. 'bear: AI erodes moat'"},
                            "value": {"type": "number", "description": "Fair value per share in this scenario."},
                            "probability": {"type": "number", "description": "Likelihood (percent or fraction)."},
                        },
                        "required": ["label", "value", "probability"],
                    },
                },
                "current_price": {"type": "number", "description": "Current share price, for upside/downside/RR."},
            },
            "required": ["scenarios", "current_price"],
        },
    },
    {
        "name": "exit_multiple",
        "description": (
            "Earnings-power valuation: grow a per-share metric (EPS, FFO, FCF/share) for "
            "N years, apply a justified exit multiple, discount back. Use for financials "
            "(EPS×P/E), REITs (FFO×P/FFO), or mid-cycle cyclical earnings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "base_metric_per_share": {"type": "number", "description": "Current per-share metric (EPS, FFO/sh, FCF/sh). For cyclicals use a mid-cycle figure."},
                "growth_rate": {"type": "number", "description": "Annual growth of the metric, percent."},
                "years": {"type": "integer", "description": "Years to project (e.g. 5)."},
                "exit_multiple": {"type": "number", "description": "Justified terminal multiple on the metric (e.g. 15 for P/E)."},
                "discount_rate": {"type": "number", "description": "Required return, percent."},
            },
            "required": ["base_metric_per_share", "growth_rate", "years", "exit_multiple", "discount_rate"],
        },
    },
]

_DISPATCH: dict[str, Callable[..., dict]] = {
    "dcf_two_stage": dcf_two_stage,
    "reverse_dcf": reverse_dcf,
    "exit_multiple": exit_multiple,
    "probability_weighted_value": probability_weighted_value,
}


def _summarize(name: str, result: dict) -> str:
    """A compact one-line summary for the progress/audit log."""
    if name == "reverse_dcf":
        return f"reverse_dcf → implied growth {result.get('implied_growth_rate_pct')}%"
    if name == "probability_weighted_value":
        return (
            f"scenarios → EV ${result.get('expected_value')}/sh, "
            f"downside {result.get('downside_pct')}%, R/R {result.get('reward_risk')}"
        )
    fv = result.get("fair_value_per_share")
    return f"{name} → ${fv}/sh" if fv is not None else f"{name} → ok"


def make_valuation_executor(on_call: Callable[[str, str], None] | None = None):
    """Return an executor(name, args)->str for the analyst's agentic loop.

    Returns a JSON string; on bad inputs returns an ``{"error": ...}`` JSON so the
    model can see what went wrong and retry with corrected assumptions."""

    def execute(name: str, args: dict) -> str:
        fn = _DISPATCH.get(name)
        if fn is None:
            return json.dumps({"error": f"unknown tool '{name}'"})
        try:
            result = fn(**args)
        except TypeError as e:
            return json.dumps({"error": f"bad arguments: {e}"})
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            return json.dumps({"error": str(e)})
        if on_call:
            try:
                on_call(name, _summarize(name, result))
            except Exception:
                pass
        return json.dumps(result)

    return execute
