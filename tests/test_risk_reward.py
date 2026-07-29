"""Reward/risk must not charge the bear case twice.

The original formula divided the PROBABILITY-WEIGHTED expected upside by the RAW
worst case, so the bear both depressed the numerator and set the denominator. With
the analyst pushed toward deep bears, R/R almost never cleared the PM's bar and the
portfolio sat in 100% cash for eleven runs while rejecting its own BUY-rated names.
These tests pin the corrected gain/loss ratio.
"""

from ai_investment_buddy.brain.valuation_tools import probability_weighted_value


def _scenarios(bear, base, bull, p=(0.3, 0.5, 0.2)):
    return [
        {"label": "bear", "value": bear, "probability": p[0]},
        {"label": "base", "value": base, "probability": p[1]},
        {"label": "bull", "value": bull, "probability": p[2]},
    ]


def test_weighted_gain_and_loss_are_reported_separately():
    r = probability_weighted_value(_scenarios(60, 120, 160), current_price=100)
    # bear: -40% × 0.3 = -12 loss; base +20% × 0.5 = 10, bull +60% × 0.2 = 12 → 22 gain
    assert r["weighted_loss_pct"] == 12.0
    assert r["weighted_gain_pct"] == 22.0
    assert r["reward_risk"] == round(22.0 / 12.0, 2)  # 1.83


def test_worst_case_downside_still_reported():
    """The raw worst case remains visible — it just no longer sets the denominator."""
    r = probability_weighted_value(_scenarios(60, 120, 160), current_price=100)
    assert r["downside_pct"] == -40.0
    assert r["worst_case_value"] == 60.0


def test_old_formula_would_have_rejected_a_favourable_name():
    """Regression: the case that kept the book in cash.

    A name with an honest 30%-probability -40% bear and solid upside scores 1.83 on
    the weighted ratio — a fat pitch — where the old expected-upside/worst-case math
    produced 0.85 and failed the PM's 'R/R clearly >1' rule."""
    r = probability_weighted_value(_scenarios(60, 120, 160), current_price=100)
    old_rr = r["expected_upside_pct"] / abs(r["downside_pct"])
    assert old_rr < 1.0
    assert r["reward_risk"] > 1.5


def test_deep_bear_still_crushes_the_ratio():
    """The fix must not become a rubber stamp: a probable, severe bear still fails."""
    r = probability_weighted_value(_scenarios(20, 110, 140, p=(0.6, 0.3, 0.1)), current_price=100)
    assert r["reward_risk"] < 0.2


def test_benign_bear_is_flagged_not_rewarded():
    r = probability_weighted_value(_scenarios(99, 130, 160), current_price=100)
    assert "downside_warning" in r
    assert r["reward_risk"] is not None and r["reward_risk"] <= 10.0


def test_no_scenario_below_price_gives_no_ratio():
    r = probability_weighted_value(_scenarios(105, 130, 160), current_price=100)
    assert r["reward_risk"] is None
    assert "downside_warning" in r


def test_expected_value_unchanged_by_the_fix():
    r = probability_weighted_value(_scenarios(60, 120, 160), current_price=100)
    assert r["expected_value"] == round(60 * 0.3 + 120 * 0.5 + 160 * 0.2, 2)
