"""Trades agreed in the feedback dialogue must reach the slate.

The regression: the PM upgraded RDDT in conversation, the investor said "ok, let's
buy it", the PM named 3% — and the run executed only the orders computed BEFORE the
conversation, because `submit_feedback` could return notes and nothing else. These
tests pin the whole path: schema → parse → dialogue return value → merge.

No LLM / network: the engine and the prompts are stubbed."""

from __future__ import annotations

import pytest

from ai_investment_buddy import cli
from ai_investment_buddy.brain import prompts
from ai_investment_buddy.brain.graph import parse_orders
from ai_investment_buddy.models import Action, Decision, TradeOrder

from datetime import date


def _order(ticker, weight, action=Action.BUY, conv=3):
    return TradeOrder(ticker=ticker, action=action, target_weight=weight, conviction=conv)


# --- Schema ------------------------------------------------------------------
def test_feedback_tool_can_emit_orders():
    props = prompts.FEEDBACK_TOOL["input_schema"]["properties"]
    assert "proposed_orders" in props
    # Required, so the model must consciously return [] rather than omit it.
    assert "proposed_orders" in prompts.FEEDBACK_TOOL["input_schema"]["required"]
    item = props["proposed_orders"]["items"]
    assert set(item["required"]) == {"ticker", "action", "target_weight", "rationale"}
    assert item["properties"]["action"]["enum"] == ["BUY", "SELL", "HOLD"]


def test_feedback_system_states_that_words_alone_do_nothing():
    for text in (prompts.feedback_system(False), prompts.feedback_system(True)):
        assert "proposed_orders" in text


# --- parse_orders ------------------------------------------------------------
def test_parse_orders_normalizes_and_skips_malformed():
    orders = parse_orders(
        [
            {"ticker": "rddt", "action": "buy", "target_weight": 0.03, "rationale": "post-print"},
            {"ticker": "TTD", "action": "NOPE", "target_weight": 0.02},  # bad action
            {"action": "BUY", "target_weight": 0.02},  # no ticker
            {"ticker": "AAPL", "action": "BUY", "target_weight": 1.5},  # weight out of range
            {"ticker": "MSFT", "action": "SELL", "target_weight": 0.0},
        ]
    )
    assert [(o.ticker, o.action, o.target_weight) for o in orders] == [
        ("RDDT", Action.BUY, 0.03),
        ("MSFT", Action.SELL, 0.0),
    ]
    assert orders[0].conviction == 3  # default when the model omits it


def test_parse_orders_handles_empty_and_none():
    assert parse_orders(None) == [] and parse_orders([]) == []


# --- merge -------------------------------------------------------------------
def _decision(orders):
    return Decision(as_of=date(2026, 7, 30), market_thesis="", orders=orders)


def test_merge_appends_new_names_and_keeps_existing():
    d = _decision([_order("TTD", 0.04), _order("MSFT", 0.10)])
    cli._merge_proposed_orders(d, [_order("RDDT", 0.03)])
    assert [(o.ticker, o.target_weight) for o in d.orders] == [
        ("TTD", 0.04), ("MSFT", 0.10), ("RDDT", 0.03),
    ]


def test_merge_supersedes_the_pre_conversation_order_for_the_same_name():
    d = _decision([_order("RDDT", 0.0, action=Action.SELL), _order("MSFT", 0.10)])
    cli._merge_proposed_orders(d, [_order("RDDT", 0.03, conv=4)])
    tickers = [o.ticker for o in d.orders]
    assert tickers.count("RDDT") == 1  # not two contradictory orders on one name
    rddt = next(o for o in d.orders if o.ticker == "RDDT")
    assert (rddt.action, rddt.target_weight, rddt.conviction) == (Action.BUY, 0.03, 4)


def test_merge_of_nothing_leaves_the_slate_alone():
    d = _decision([_order("MSFT", 0.10)])
    cli._merge_proposed_orders(d, [])
    assert [o.ticker for o in d.orders] == ["MSFT"]


# --- the dialogue returns what it agreed to ----------------------------------
class _Engine:
    """Replays a scripted PM: turn 1 argues, turn 2 proposes the 3% buy."""

    payloads = [
        {
            "response": "RDDT moves from WATCH to BUY after the print.",
            "stance": "AGREE", "ticker_notes": [], "market_note": "",
            "proposed_orders": [],
        },
        {
            "response": "Agreed — 3% of NAV.",
            "stance": "AGREE",
            "ticker_notes": [{"ticker": "RDDT", "note": "monetization intact", "changes_thesis": True}],
            "market_note": "",
            "proposed_orders": [
                {"ticker": "RDDT", "action": "BUY", "target_weight": 0.03,
                 "rationale": "post-earnings reset, search risk haircut", "conviction": 4},
            ],
        },
    ]

    def __init__(self, client=None):
        self.turn = 0

    def discuss(self, context, transcript, on_tool=None):
        self.turn += 1
        return self.payloads[min(self.turn, len(self.payloads)) - 1]


@pytest.fixture
def dialogue(monkeypatch):
    """_run_feedback with a scripted PM, scripted stdin, and memory writes stubbed."""
    from ai_investment_buddy.brain import decide
    from ai_investment_buddy.memory import valuations

    monkeypatch.setattr(decide, "DecisionEngine", _Engine)
    monkeypatch.setattr(valuations, "add_note", lambda *a, **k: True)
    monkeypatch.setattr(cli, "Journal", lambda: type(
        "J", (), {"append_investor_note": lambda self, *a, **k: None}
    )())
    answers = iter(["what about RDDT after earnings?", "ok, let's buy it", ""])
    monkeypatch.setattr(cli.typer, "prompt", lambda *a, **k: next(answers))
    return answers


class _Result:
    decision = Decision(as_of=date(2026, 7, 30), market_thesis="held", orders=[])
    strategy = None
    assessments: list = []
    portfolio = None
    prices: dict = {}


def test_run_feedback_returns_the_agreed_trade(dialogue):
    proposed = cli._run_feedback(_Result())
    assert len(proposed) == 1
    o = proposed[0]
    assert (o.ticker, o.action, o.target_weight, o.conviction) == (
        "RDDT", Action.BUY, 0.03, 4,
    )


def test_agreed_trade_survives_into_an_empty_slate(dialogue):
    """The exact reported bug: PM held (no orders), the discussion bought RDDT."""
    result = _Result()
    result.decision = Decision(as_of=date(2026, 7, 30), market_thesis="held", orders=[])
    proposed = cli._run_feedback(result)
    cli._merge_proposed_orders(result.decision, proposed)
    assert [(o.ticker, o.action.value) for o in result.decision.orders] == [("RDDT", "BUY")]


def test_dialogue_with_no_agreed_trade_returns_nothing(monkeypatch):
    from ai_investment_buddy.brain import decide
    from ai_investment_buddy.memory import valuations

    class _Chat(_Engine):
        payloads = [{
            "response": "Still a WATCH for me.", "stance": "DISAGREE",
            "ticker_notes": [], "market_note": "", "proposed_orders": [],
        }]

    monkeypatch.setattr(decide, "DecisionEngine", _Chat)
    monkeypatch.setattr(valuations, "add_note", lambda *a, **k: True)
    monkeypatch.setattr(cli, "Journal", lambda: type(
        "J", (), {"append_investor_note": lambda self, *a, **k: None}
    )())
    answers = iter(["thoughts on RDDT?", ""])
    monkeypatch.setattr(cli.typer, "prompt", lambda *a, **k: next(answers))
    assert cli._run_feedback(_Result()) == []


def test_skipped_dialogue_returns_no_orders(monkeypatch):
    monkeypatch.setattr(cli.typer, "prompt", lambda *a, **k: "")
    assert cli._run_feedback(_Result()) == []
