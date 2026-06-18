"""Performance accounting vs the S&P 500 and Nasdaq 100.

NAV history rows store the benchmark index levels on each run, so 'beating the
benchmark' is measured the honest way: portfolio total return from inception vs
each index's price return over the same window.
"""

from __future__ import annotations

from ..config import SETTINGS


def _first_level(nav_history: list[dict], key: str) -> float | None:
    for row in nav_history:
        val = row.get(key)
        if val not in (None, ""):
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


def compute_returns(
    nav_history: list[dict],
    current_nav: float,
    current_benchmarks: dict[str, float],
    starting_capital: float | None = None,
) -> dict[str, float]:
    """Return inception-to-date % returns for portfolio and each benchmark."""
    cap = starting_capital if starting_capital is not None else SETTINGS.starting_capital
    out = {"Portfolio": (current_nav / cap - 1) * 100 if cap else 0.0}

    for label, level in current_benchmarks.items():
        base = _first_level(nav_history, label)
        if base is None:
            base = level  # first run: inception baseline is today.
        out[label] = (level / base - 1) * 100 if base else 0.0
    return out


def performance_summary(
    nav_history: list[dict],
    current_nav: float,
    current_benchmarks: dict[str, float],
    starting_capital: float | None = None,
) -> str:
    rets = compute_returns(
        nav_history, current_nav, current_benchmarks, starting_capital
    )
    cap = starting_capital if starting_capital is not None else SETTINGS.starting_capital
    days = len(nav_history)
    lines = [
        f"Inception capital: ${cap:,.0f} | Current NAV: ${current_nav:,.0f} | "
        f"Runs recorded: {days}",
        f"  Portfolio total return: {rets['Portfolio']:+.2f}%",
    ]
    for label in current_benchmarks:
        rel = rets["Portfolio"] - rets.get(label, 0.0)
        lines.append(
            f"  {label}: {rets.get(label, 0.0):+.2f}%  "
            f"(portfolio {rel:+.2f}% vs it)"
        )
    return "\n".join(lines)
