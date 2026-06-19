"""Portable state snapshots: serialize the whole bot into one file and back.

The experiment's value is its accumulated state (portfolio, trade ledger, NAV
history, journal, theses, narrative). This bundles all of it into a single
JSON file you can move between machines: `aib export` here, copy the file,
`aib import` there, and the bot resumes exactly where it left off.

We capture each artifact's RAW contents (not parsed) so snapshots survive schema
changes. The re-fetchable universe cache is intentionally excluded.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..config import DATA_DIR, JOURNAL_DIR, ensure_dirs

SNAPSHOT_VERSION = 1

# Artifacts to include, as paths relative to DATA_DIR.
def _state_files() -> list[Path]:
    paths: list[Path] = []
    for name in ("portfolio.json", "trades.jsonl", "nav_history.csv", "watchlist.jsonl"):
        p = DATA_DIR / name
        if p.exists():
            paths.append(p)
    if JOURNAL_DIR.exists():
        paths.extend(sorted(JOURNAL_DIR.glob("*.md")))
        theses = JOURNAL_DIR / "theses.json"
        if theses.exists():
            paths.append(theses)
    valuations_dir = DATA_DIR / "valuations"
    if valuations_dir.exists():
        paths.extend(sorted(valuations_dir.glob("*.json")))
    return paths


def _summary(files: dict[str, str]) -> dict:
    summary: dict = {}
    pf = files.get("portfolio.json")
    if pf:
        try:
            data = json.loads(pf)
            summary["cash"] = data.get("cash")
            summary["n_positions"] = len(data.get("positions", {}))
        except Exception:
            pass
    trades = files.get("trades.jsonl", "")
    summary["n_trades"] = len([l for l in trades.splitlines() if l.strip()])
    nav = files.get("nav_history.csv", "")
    nav_rows = [l for l in nav.splitlines() if l.strip()]
    summary["nav_rows"] = max(0, len(nav_rows) - 1)  # minus header
    if len(nav_rows) >= 2:
        summary["last_nav_row"] = nav_rows[-1]
    summary["journal_days"] = len(
        [k for k in files if k.startswith("journal/") and k.endswith(".md")
         and not k.endswith("narrative.md")]
    )
    return summary


def build_snapshot() -> dict:
    files: dict[str, str] = {}
    for p in _state_files():
        rel = p.relative_to(DATA_DIR).as_posix()
        files[rel] = p.read_text()
    return {
        "aib_snapshot_version": SNAPSHOT_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "summary": _summary(files),
        "files": files,
    }


def export_state(path: Path) -> dict:
    snap = build_snapshot()
    path.write_text(json.dumps(snap, indent=2))
    return snap


def import_state(path: Path, force: bool = False) -> dict:
    snap = json.loads(path.read_text())
    if snap.get("aib_snapshot_version") != SNAPSHOT_VERSION:
        raise ValueError(
            f"Unsupported snapshot version {snap.get('aib_snapshot_version')} "
            f"(this build expects {SNAPSHOT_VERSION})."
        )

    existing = DATA_DIR / "portfolio.json"
    if existing.exists() and not force:
        raise FileExistsError(
            f"State already exists at {DATA_DIR}. Use --force to overwrite it."
        )

    ensure_dirs()
    for rel, content in snap.get("files", {}).items():
        # Guard against path traversal in untrusted snapshots.
        target = (DATA_DIR / rel).resolve()
        if not str(target).startswith(str(DATA_DIR.resolve())):
            raise ValueError(f"Refusing to write outside data dir: {rel}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return snap
