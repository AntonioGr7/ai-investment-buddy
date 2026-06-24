"""The defensive macro-hedge sleeve.

A small, curated set of macro/diversifier ETFs (gold, silver, broad commodities,
long Treasuries, the dollar) the agent can reach for as INSURANCE when the regime
is genuinely stressed — not as a momentum punt on a hot commodity. Unlike the
equity universe, these are always present to the brain (like a built-in
watchlist), tagged ``asset_class="macro_hedge"`` so the rest of the system can:

  - value them on a separate regime/role path (no DCF — they have no cash flows),
  - cap the whole sleeve as a fraction of NAV at execution,
  - surface them distinctly in the board / dashboard.

We trade the real ETFs at real prices, so contango/roll decay is already embedded
in the price series — there is no spot-commodity modelling and the paper P&L is
honest. The set lives in ``SETTINGS.macro_sleeve`` so it's configurable; an empty
dict disables the sleeve.
"""

from __future__ import annotations

from .config import SETTINGS

# The synthetic GICS-style "sector" we file the sleeve under, so it shows as its
# own bucket in the risk/sector views instead of polluting an equity sector.
SLEEVE_SECTOR = "Macro Hedge"


def sleeve_meta() -> dict[str, dict]:
    """The curated sleeve: ticker -> {name, role, drivers}."""
    return dict(SETTINGS.macro_sleeve or {})


def sleeve_tickers() -> list[str]:
    return list(sleeve_meta().keys())


def sleeve_set() -> set[str]:
    return set(sleeve_meta().keys())


def is_hedge(ticker: str) -> bool:
    return (ticker or "").upper() in sleeve_set()


def universe_entries() -> list[dict]:
    """Sleeve names shaped like ``universe.get_universe`` entries, so the pipeline
    can merge them into the universe metadata uniformly."""
    out = []
    for t, info in sleeve_meta().items():
        out.append(
            {
                "ticker": t,
                "name": info.get("name", ""),
                "sector": SLEEVE_SECTOR,
                "sub_industry": info.get("role", ""),
                "cap_tier": "large",
                "asset_class": "macro_hedge",
                "indices": ["Macro Hedge"],
            }
        )
    return out


def format_sleeve_for_strategist() -> str:
    """The sleeve block injected into the strategist prompt — what's available and
    what each instrument is FOR."""
    meta = sleeve_meta()
    if not meta:
        return ""
    lines = [
        "DEFENSIVE MACRO-HEDGE SLEEVE (available every day; a diversifier, NOT an "
        f"alpha source — total capped at {SETTINGS.max_macro_sleeve_weight:.0%} of NAV):"
    ]
    for t, info in meta.items():
        lines.append(f"  {t} ({info.get('name','')}): {info.get('role','')}")
        if info.get("drivers"):
            lines.append(f"      driven by: {info['drivers']}")
    return "\n".join(lines)
