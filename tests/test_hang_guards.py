"""Regression guards against the class of bug where a single run stalled for ~17h.

The cause was an unbounded network wait (yfinance opens sockets with no timeout,
and the process-wide default was None). These tests assert the defenses stay in
place: a default socket timeout is installed at import, a stalled socket actually
raises, and the wall-clock deadline aborts a runaway block.
"""

from __future__ import annotations

import socket
import time

import pytest


def test_default_socket_timeout_installed_on_import():
    """Importing the package must set a finite process-wide socket timeout."""
    import ai_investment_buddy  # noqa: F401 — import has the side effect

    from ai_investment_buddy.config import SETTINGS

    t = socket.getdefaulttimeout()
    assert t is not None, "no default socket timeout — a stalled read could hang forever"
    assert t == SETTINGS.network_timeout
    assert 0 < t <= 300


def test_apply_network_timeout_is_configurable_and_disableable():
    from ai_investment_buddy.runtime import apply_network_timeout

    try:
        apply_network_timeout(5)
        assert socket.getdefaulttimeout() == 5
        apply_network_timeout(0)  # 0 disables
        assert socket.getdefaulttimeout() is None
    finally:
        apply_network_timeout()  # restore configured default


def test_stalled_connect_aborts_quickly():
    """A connect to a non-routable host must raise, not block, within the timeout."""
    from ai_investment_buddy.runtime import apply_network_timeout

    apply_network_timeout(2)
    try:
        t0 = time.time()
        with pytest.raises((socket.timeout, TimeoutError, OSError)):
            socket.create_connection(("10.255.255.1", 80))
        assert time.time() - t0 < 10
    finally:
        apply_network_timeout()


def test_deadline_aborts_a_runaway_block():
    from ai_investment_buddy.runtime import deadline

    t0 = time.time()
    with pytest.raises(TimeoutError):
        with deadline(1, "test-op"):
            time.sleep(30)
    assert time.time() - t0 < 5


def test_deadline_noop_when_disabled():
    from ai_investment_buddy.runtime import deadline

    with deadline(0, "disabled"):  # must not raise or arm anything
        pass


def test_llm_clients_have_bounded_timeout(monkeypatch):
    """The LLM clients must carry an explicit finite timeout + retry cap."""
    from ai_investment_buddy.config import SETTINGS

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from ai_investment_buddy.brain.llm import AnthropicClient

    c = AnthropicClient()
    assert c.client.timeout == SETTINGS.llm_timeout
    assert c.client.max_retries == SETTINGS.llm_max_retries
    assert SETTINGS.llm_timeout and SETTINGS.llm_timeout <= 600
