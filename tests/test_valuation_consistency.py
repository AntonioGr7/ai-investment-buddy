"""Fair value must not swing on re-guessed inputs or the share price alone.

The regression: APP's stored fair value went $379 → $305 → $181 → $252 → $319 →
$252 over three weeks on prices of $262-$308. Each run valued from scratch with no
sight of the prior read, guessed base FCF ($1.8B one day, $3.0B two weeks later)
because the reported figure was never shown, and was re-run almost daily because
ANY new headline invalidated the cache. These tests pin the fixes: reported inputs
and the prior read reach the prompt, an unexplained move is reconciled or rejected,
and only material news forces a re-valuation.

No LLM / network: the client and the valuation store are stubbed."""

from __future__ import annotations

from datetime import date

import pytest

from ai_investment_buddy.brain import graph, prompts
from ai_investment_buddy.memory import valuations
from ai_investment_buddy.models import (
    StoredValuation,
    TickerData,
    ValuationAssessment,
    ValuationRecord,
)

TODAY = date(2026, 10, 15)
SCEN = [
    {"label": "bear", "value": 180.0, "probability": 0.25},
    {"label": "base", "value": 330.0, "probability": 0.55},
    {"label": "bull", "value": 450.0, "probability": 0.20},
]


def _prior(fair=330.0, price=308.0) -> ValuationRecord:
    a = ValuationAssessment(
        ticker="APP", fair_value=fair, current_price=price, bear_value=180.0,
        entry_price=265.0, risk_reward=2.8, recommendation="WATCH",
        valuation_method="two-stage FCF DCF",
        key_assumptions={"base_cash_flow_m": 3000, "scenarios": SCEN},
    )
    entry = StoredValuation(as_of=date(2026, 10, 14), assessment=a)
    return ValuationRecord(
        ticker="APP", first_assessed=date(2026, 9, 1),
        last_assessed=date(2026, 10, 14), latest=entry,
    )


def _payload(fair, fundamental=False, **kw):
    p = {
        "archetype": "HYPERGROWTH", "valuation_method": "DCF", "fair_value": fair,
        "entry_price": fair * 0.8, "bear_value": 120.0, "downside_pct": -54.0,
        "risk_reward": 0.7, "valuation_verdict": "FAIRLY_VALUED", "quality_score": 4,
        "margin_of_safety": False, "structural_risk": "MEDIUM", "recommendation": "WATCH",
        "suggested_max_weight": 0.05, "confidence": 3,
        "key_assumptions": {"base_cash_flow_m": 1800, "scenarios": []},
        "change_vs_prior": "price fell", "fundamental_change": fundamental,
    }
    p.update(kw)
    return p


class _Client:
    """Returns queued payloads and records the user prompt of each call."""

    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.prompts: list[str] = []

    def agentic_call(self, system, user, helper_tools, final_tool, executor, max_iters=6):
        self.prompts.append(user)
        return self.payloads.pop(0)


def _td():
    return TickerData(
        ticker="APP", name="AppLovin", price=262.0,
        free_cashflow=3.1e9, shares_outstanding=340e6, total_cash=1.0e9, total_debt=3.5e9,
    )


@pytest.fixture
def prior(monkeypatch):
    rec = _prior()
    monkeypatch.setattr(valuations, "prior_for_anchor", lambda t, d: rec)
    return rec


# --- Prompt ------------------------------------------------------------------
def test_reported_financials_and_prior_reach_the_prompt(prior):
    client = _Client(_payload(335.0))
    graph.assess_ticker(client, _td(), "", "", as_of=TODAY)
    user = client.prompts[0]
    assert "TTM free cash flow ($M): 3,100" in user
    assert "Shares outstanding (M): 340.0" in user
    assert "Net debt ($M, negative = net cash): 2,500" in user
    assert "fair $330.00" in user and '"base_cash_flow_m": 3000' in user


def test_first_valuation_says_so(monkeypatch):
    monkeypatch.setattr(valuations, "prior_for_anchor", lambda t, d: None)
    client = _Client(_payload(181.5))
    a = graph.assess_ticker(client, _td(), "", "", as_of=TODAY)
    assert "None on file — first valuation." in client.prompts[0]
    assert a.fair_value == 181.5 and len(client.prompts) == 1


def test_schema_requires_assumptions_and_change_explanation():
    req = prompts.ANALYST_TOOL["input_schema"]["required"]
    assert {"key_assumptions", "change_vs_prior", "fundamental_change"} <= set(req)


# --- Guard -------------------------------------------------------------------
def test_small_move_is_accepted_without_retry(prior):
    client = _Client(_payload(345.0))  # +4.5%
    a = graph.assess_ticker(client, _td(), "", "", as_of=TODAY)
    assert len(client.prompts) == 1 and a.fair_value == 345.0 and not a.anchored_to_prior


def test_explained_large_move_is_accepted(prior):
    client = _Client(_payload(250.0, fundamental=True, change_vs_prior="Q3 guide cut 20%"))
    a = graph.assess_ticker(client, _td(), "", "", as_of=TODAY)
    assert len(client.prompts) == 1 and a.fair_value == 250.0


def test_unexplained_move_gets_one_reconciliation(prior):
    client = _Client(_payload(181.5), _payload(328.0))
    a = graph.assess_ticker(client, _td(), "", "", as_of=TODAY)
    assert len(client.prompts) == 2
    assert "REVIEW OF YOUR PREVIOUS SUBMISSION" in client.prompts[1]
    assert a.fair_value == 328.0 and not a.anchored_to_prior


def test_persistent_unexplained_move_carries_prior_forward(prior):
    client = _Client(_payload(181.5), _payload(190.0))
    a = graph.assess_ticker(client, _td(), "", "", as_of=TODAY)
    assert a.anchored_to_prior
    assert (a.fair_value, a.bear_value, a.entry_price) == (330.0, 180.0, 265.0)
    # Re-measured at TODAY's price: the drop widens upside, it doesn't cut value.
    assert a.upside_pct == round((330 / 262 - 1) * 100, 1)
    assert a.key_assumptions["base_cash_flow_m"] == 3000
    # R/R re-weighted from the prior scenarios at the new price, not the rejected run's 0.7.
    assert a.risk_reward > 2.0


# --- Reuse gate ----------------------------------------------------------------
def test_routine_headlines_do_not_force_revaluation(monkeypatch):
    rec = _prior(price=262.0)
    rec.latest.news_seen = ["old headline"]
    monkeypatch.setattr(valuations, "load", lambda t: rec)
    routine = ["AppLovin stock slides 3% as tech sells off", "3 growth stocks to watch"]
    assert valuations.find_reusable("APP", 263.0, routine, TODAY) is not None


@pytest.mark.parametrize("h", [
    "AppLovin beats Q3 estimates, raises guidance",
    "SEC opens probe into AppLovin ad practices",
    "AppLovin announces $2B buyback",
])
def test_material_headline_forces_revaluation(monkeypatch, h):
    rec = _prior(price=262.0)
    monkeypatch.setattr(valuations, "load", lambda t: rec)
    assert valuations.find_reusable("APP", 263.0, [h], TODAY) is None
