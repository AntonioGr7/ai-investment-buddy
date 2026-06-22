"""Execution engine: paper trading, benchmarking, and the daily pipeline."""

from .attribution import SeriesPoint, attribution_report, compute_metrics
from .benchmark import compute_returns, performance_summary
from .execute import execute
from .pipeline import RunResult, commit, run_daily

__all__ = [
    "compute_returns",
    "performance_summary",
    "attribution_report",
    "compute_metrics",
    "SeriesPoint",
    "execute",
    "RunResult",
    "run_daily",
    "commit",
]
