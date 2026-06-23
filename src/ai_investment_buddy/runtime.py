"""Hang guards — make it impossible for a run to stall for hours.

Two independent layers:

1. ``apply_network_timeout()`` sets a global default socket timeout. This is the
   root-cause fix: libraries that open sockets without their own timeout (notably
   yfinance, which is how a single valuation once blocked for ~17h) will now raise
   after the timeout instead of waiting on a dead socket forever. It's applied at
   package import, so every entry point — CLI, API, and the valuation subprocess —
   inherits it. Clients that set an EXPLICIT socket timeout (the LLM SDKs, via
   httpx) are unaffected, so long model responses are not cut short.

2. ``deadline()`` is a whole-operation wall-clock cap (SIGALRM-based) wrapped
   around the long CLI commands — a catch-all for any unforeseen stall that the
   per-socket timeout doesn't cover. The API's on-demand trigger has its own
   subprocess timeout, so this covers the directly-invoked CLI path.
"""

from __future__ import annotations

import signal
import socket
import threading
from contextlib import contextmanager

from .config import SETTINGS


def apply_network_timeout(seconds: int | None = None) -> None:
    """Set the process-wide default socket timeout. Idempotent; 0/None disables."""
    secs = SETTINGS.network_timeout if seconds is None else seconds
    socket.setdefaulttimeout(secs if secs and secs > 0 else None)


@contextmanager
def deadline(seconds: int, label: str = "operation"):
    """Abort the wrapped block with TimeoutError after ``seconds`` wall-clock.

    Uses SIGALRM, so it only arms on the main thread of a Unix process (the CLI
    case). Where that's unavailable (a worker thread, or non-Unix) it's a no-op —
    the per-socket timeout still applies, this is just the extra backstop. 0/negative
    disables."""
    if seconds is None or seconds <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def _fire(signum, frame):
        raise TimeoutError(
            f"{label} exceeded its {seconds}s deadline and was aborted "
            "(AIB_*_DEADLINE / hang guard). Nothing was committed."
        )

    prev = signal.signal(signal.SIGALRM, _fire)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)
