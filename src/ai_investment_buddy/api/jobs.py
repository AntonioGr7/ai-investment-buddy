"""On-demand valuation jobs — run ``aib valuate TICKER`` in a child process.

The trigger is deliberately a subprocess, not an in-process call: the valuation
is a multi-minute LLM pipeline, and a child process keeps it off the API event
loop entirely, reuses the exact CLI orchestration, and writes through the same
dual-write path (JSON files + SQLite) so the result is immediately readable via
the normal endpoints. Concurrency is capped so a burst of triggers can't spawn
an unbounded number of LLM runs; extra jobs wait in the "queued" state.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

# Most this many valuations run at once; the rest queue.
MAX_CONCURRENT = int(os.getenv("AIB_API_MAX_VALUATIONS", "2"))
# Hard ceiling on a single valuation before we give up on it.
VALUATION_TIMEOUT_S = int(os.getenv("AIB_API_VALUATION_TIMEOUT", "900"))

_sem = threading.BoundedSemaphore(MAX_CONCURRENT)
_jobs: dict[str, "Job"] = {}
_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    id: str
    ticker: str
    status: str = "queued"  # queued | running | done | error
    created_at: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    returncode: int | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def start_valuation(ticker: str) -> Job:
    job = Job(id=uuid.uuid4().hex[:12], ticker=ticker.strip().upper(), created_at=_now())
    with _lock:
        _jobs[job.id] = job
    threading.Thread(target=_run, args=(job,), daemon=True).start()
    return job


def _run(job: Job) -> None:
    # Block here (still "queued") until a concurrency slot frees up.
    with _sem:
        job.status = "running"
        job.started_at = _now()
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "ai_investment_buddy.cli", "valuate", job.ticker],
                capture_output=True,
                text=True,
                env=os.environ.copy(),  # same AIB_DATA_DIR + LLM key as the server
                timeout=VALUATION_TIMEOUT_S,
            )
            job.returncode = proc.returncode
            if proc.returncode == 0:
                job.status = "done"
            else:
                job.status = "error"
                job.error = (proc.stderr or proc.stdout or "valuation failed").strip()[-2000:]
        except subprocess.TimeoutExpired:
            job.status = "error"
            job.error = f"valuation timed out after {VALUATION_TIMEOUT_S}s"
        except Exception as e:  # pragma: no cover - defensive
            job.status = "error"
            job.error = str(e)
        finally:
            job.finished_at = _now()


def get_job(job_id: str) -> Job | None:
    with _lock:
        return _jobs.get(job_id)


def all_jobs() -> list[Job]:
    with _lock:
        return sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)
