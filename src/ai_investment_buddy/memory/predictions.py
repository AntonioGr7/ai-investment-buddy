"""Forecast ledger + calibration scorecard — the agent's foresight, made scorable.

Every explicit prediction the brain makes is logged here with the probability we
assigned and the market-implied baseline we diverged from. As each prediction's
horizon passes it is RESOLVED (price-anchored ones mechanically; others by
judgement) and scored. The accumulated record yields a calibration scorecard:

  Brier score        mean (probability − outcome)^2. Lower is better; 0 = perfect,
                     0.25 = a coin flip, >0.25 = worse than guessing.
  Brier skill score  Brier vs always predicting the base rate. >0 = real skill;
                     ≤0 = no forecasting edge over knowing the base rate.
  hit rate           % correct at a 0.5 threshold.
  overconfidence     avg stated confidence − actual accuracy. >0 = the agent
                     believes its calls more than reality justifies.
  calibration curve  for each confidence bucket, predicted prob vs realized rate —
                     are the 70% calls right ~70% of the time?
  edge realization   on calls where we claimed an edge over consensus, did the
                     high-edge ones actually pay off? The test of variant perception.

This scorecard is injected back into the brain (so it can correct a systematic
bias) and shown by `aib calibration`. It is the empirical answer to "can this
thing actually foresee anything, or is it just telling good stories?".

Stored as one JSONL ledger (``data/predictions.jsonl``); resolution rewrites it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date

from ..config import DATA_DIR, ensure_dirs
from ..models import Prediction

_FILE = DATA_DIR / "predictions.jsonl"

# Resolution needs at least this much price cushion handled by the caller; here we
# only need the price at/after the horizon.


def make_id(ticker: str, created: date, statement: str) -> str:
    """Stable id from the claim, so the same prediction isn't logged twice."""
    h = hashlib.sha1(f"{ticker}|{statement}".encode()).hexdigest()[:8]
    return f"{(ticker or 'MKT')}-{created.isoformat()}-{h}"


def load_all() -> list[Prediction]:
    if not _FILE.exists():
        return []
    out: list[Prediction] = []
    for line in _FILE.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(Prediction.model_validate_json(line))
            except Exception:
                continue
    return out


def _write_all(preds: list[Prediction]) -> None:
    ensure_dirs()
    _FILE.write_text("".join(p.model_dump_json() + "\n" for p in preds))


def add_many(preds: list[Prediction]) -> list[Prediction]:
    """Append new predictions, skipping ids already on the ledger (idempotent)."""
    if not preds:
        return []
    existing = {p.id for p in load_all()}
    fresh = [p for p in preds if p.id not in existing]
    if not fresh:
        return []
    ensure_dirs()
    with _FILE.open("a") as f:
        for p in fresh:
            f.write(p.model_dump_json() + "\n")
    return fresh


def open_predictions(as_of: date | None = None) -> list[Prediction]:
    preds = [p for p in load_all() if p.status == "open"]
    return preds


def due_predictions(as_of: date) -> list[Prediction]:
    """Open predictions whose horizon has arrived."""
    return [p for p in load_all() if p.status == "open" and p.horizon_date <= as_of]


def _score(p: Prediction, outcome: bool, on: date, note: str) -> None:
    p.outcome = outcome
    p.status = "resolved"
    p.resolved_on = on
    p.resolution_note = note
    p.brier = (p.probability - (1.0 if outcome else 0.0)) ** 2


def resolve_mechanical(
    as_of: date, price_at: dict[str, float]
) -> list[Prediction]:
    """Resolve due, price-anchored predictions from actual prices. ``price_at`` maps
    ticker -> the resolving price (latest at/after the horizon). Returns the ones
    resolved (the ledger is rewritten in place). Manual/judged kinds are left open."""
    preds = load_all()
    changed: list[Prediction] = []
    for p in preds:
        if p.status != "open" or p.horizon_date > as_of:
            continue
        px = price_at.get(p.ticker)
        if px is None or px <= 0:
            continue
        outcome: bool | None = None
        note = ""
        if p.resolve_kind == "price_above" and p.resolve_price is not None:
            outcome = px >= p.resolve_price
            note = f"price ${px:.2f} vs ≥${p.resolve_price:.2f}"
        elif p.resolve_kind == "price_below" and p.resolve_price is not None:
            outcome = px <= p.resolve_price
            note = f"price ${px:.2f} vs ≤${p.resolve_price:.2f}"
        elif (
            p.resolve_kind == "return_above"
            and p.resolve_price is not None
            and p.resolve_reference_price
        ):
            ret = px / p.resolve_reference_price - 1.0
            outcome = ret >= p.resolve_price
            note = f"return {ret:+.1%} vs ≥{p.resolve_price:+.1%}"
        if outcome is not None:
            _score(p, outcome, as_of, note)
            changed.append(p)
    if changed:
        _write_all(preds)
    return changed


def resolve_manual(pred_id: str, outcome: bool, on: date, note: str = "") -> bool:
    """Resolve one prediction by judgement (agent or human). Returns True if found."""
    preds = load_all()
    for p in preds:
        if p.id == pred_id and p.status == "open":
            _score(p, outcome, on, note or "judged")
            _write_all(preds)
            return True
    return False


# --- Calibration scorecard ---------------------------------------------------
@dataclass
class Bucket:
    low: float
    high: float
    n: int = 0
    predicted_sum: float = 0.0
    actual_sum: float = 0.0  # count of True outcomes

    @property
    def predicted(self) -> float | None:
        return self.predicted_sum / self.n if self.n else None

    @property
    def actual(self) -> float | None:
        return self.actual_sum / self.n if self.n else None


@dataclass
class Calibration:
    n_resolved: int = 0
    n_open: int = 0
    brier: float | None = None
    brier_skill: float | None = None  # vs always predicting base rate
    hit_rate: float | None = None  # accuracy at 0.5
    base_rate: float | None = None  # fraction resolved True
    avg_confidence: float | None = None  # mean |p−0.5| mapped to confidence in the called direction
    overconfidence: float | None = None  # avg directional confidence − accuracy
    buckets: list[Bucket] = field(default_factory=list)
    by_category: dict[str, dict] = field(default_factory=dict)
    edge_hit_rate: float | None = None  # accuracy on calls where we claimed edge>0
    edge_n: int = 0


def _directional(p: float) -> tuple[float, bool]:
    """Map a probability to (confidence-it-happens-in-the-called-direction, called_yes).

    A 0.2 probability is a confident NO (0.8 confidence the claim is false)."""
    if p >= 0.5:
        return p, True
    return 1.0 - p, False


def compute_calibration(preds: list[Prediction] | None = None, n_buckets: int = 5) -> Calibration:
    preds = preds if preds is not None else load_all()
    resolved = [p for p in preds if p.status == "resolved" and p.outcome is not None]
    cal = Calibration(
        n_resolved=len(resolved),
        n_open=sum(1 for p in preds if p.status == "open"),
    )
    if not resolved:
        return cal

    outcomes = [1.0 if p.outcome else 0.0 for p in resolved]
    cal.base_rate = sum(outcomes) / len(outcomes)
    cal.brier = sum(p.brier for p in resolved) / len(resolved)
    # Brier of the naive base-rate forecaster, for the skill score.
    base_brier = sum((cal.base_rate - o) ** 2 for o in outcomes) / len(outcomes)
    cal.brier_skill = 1.0 - cal.brier / base_brier if base_brier > 0 else None

    # Accuracy: did the claim's called direction match the outcome?
    correct = 0
    conf_sum = 0.0
    for p, o in zip(resolved, outcomes):
        conf, called_yes = _directional(p.probability)
        conf_sum += conf
        if (called_yes and o == 1.0) or (not called_yes and o == 0.0):
            correct += 1
    cal.hit_rate = correct / len(resolved)
    cal.avg_confidence = conf_sum / len(resolved)
    cal.overconfidence = cal.avg_confidence - cal.hit_rate

    # Calibration curve on the raw probability (yes-claim) axis.
    edges = [i / n_buckets for i in range(n_buckets + 1)]
    buckets = [Bucket(edges[i], edges[i + 1]) for i in range(n_buckets)]
    for p, o in zip(resolved, outcomes):
        idx = min(int(p.probability * n_buckets), n_buckets - 1)
        b = buckets[idx]
        b.n += 1
        b.predicted_sum += p.probability
        b.actual_sum += o
    cal.buckets = [b for b in buckets if b.n]

    # By category.
    cats: dict[str, list] = {}
    for p, o in zip(resolved, outcomes):
        cats.setdefault(p.category or "uncategorized", []).append((p, o))
    for c, items in cats.items():
        n = len(items)
        br = sum(pp.brier for pp, _ in items) / n
        hr = sum(
            1
            for pp, oo in items
            if (_directional(pp.probability)[1] and oo == 1.0)
            or (not _directional(pp.probability)[1] and oo == 0.0)
        ) / n
        cal.by_category[c] = {"n": n, "brier": br, "hit_rate": hr}

    # Edge realization: on calls where we claimed a positive edge vs consensus, did
    # the YES claims actually come true more often? (variant-perception test).
    edge_calls = [
        (p, o)
        for p, o in zip(resolved, outcomes)
        if p.edge is not None and p.edge > 0.05 and p.probability >= 0.5
    ]
    cal.edge_n = len(edge_calls)
    if edge_calls:
        cal.edge_hit_rate = sum(o for _, o in edge_calls) / len(edge_calls)
    return cal


def format_calibration(cal: Calibration, detailed: bool = True) -> str:
    if cal.n_resolved == 0:
        return (
            f"No resolved predictions yet ({cal.n_open} open). The calibration "
            "scorecard fills in as forecasts reach their horizon and get scored."
        )

    def pc(v):
        return f"{v:.0%}" if v is not None else "n/a"

    def num(v):
        return f"{v:.3f}" if v is not None else "n/a"

    skill = (
        f"{cal.brier_skill:+.2f}"
        + (
            " (real skill)"
            if cal.brier_skill and cal.brier_skill > 0.02
            else " (no edge over base rate)"
            if cal.brier_skill is not None and cal.brier_skill <= 0.02
            else ""
        )
        if cal.brier_skill is not None
        else "n/a"
    )
    oc = cal.overconfidence
    oc_label = (
        "OVERCONFIDENT" if oc and oc > 0.05 else "underconfident" if oc and oc < -0.05 else "well-calibrated"
    )
    lines = [
        f"Resolved: {cal.n_resolved}  |  Open: {cal.n_open}  |  Base rate (things happened): {pc(cal.base_rate)}",
        f"Brier {num(cal.brier)} (0=perfect, 0.25=coin-flip)  |  Brier skill {skill}",
        f"Hit rate {pc(cal.hit_rate)}  |  Avg confidence {pc(cal.avg_confidence)}  |  "
        f"Overconfidence {f'{oc:+.0%}' if oc is not None else 'n/a'} → {oc_label}",
    ]
    if cal.edge_n:
        lines.append(
            f"Variant-perception test: {cal.edge_n} high-edge YES calls hit {pc(cal.edge_hit_rate)} "
            "of the time (want > the base rate — that's where alpha would come from)."
        )

    if detailed and cal.buckets:
        lines.append("")
        lines.append("Calibration curve (predicted prob → actual rate):")
        for b in cal.buckets:
            bar = "█" * round((b.actual or 0) * 10)
            lines.append(
                f"  {b.low:.1f}–{b.high:.1f}: n={b.n:<3} predicted {pc(b.predicted)} → actual {pc(b.actual)}  {bar}"
            )
    if detailed and cal.by_category:
        lines.append("")
        lines.append("By category (Brier / hit rate):")
        for c, s in sorted(cal.by_category.items()):
            lines.append(f"  {c:<12} n={s['n']:<3} Brier {s['brier']:.3f}  hit {s['hit_rate']:.0%}")
    return "\n".join(lines)
