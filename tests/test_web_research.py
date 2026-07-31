"""Live web research for the feedback dialogue: parsing, guardrails, wiring.

No network — the search backend and `requests` are stubbed. These pin the parts
that silently rot: DuckDuckGo redirect unwrapping, HTML→text extraction, the
refuse/degrade paths (bad scheme, dead link, backend exception), and that
`discuss()` actually hands the web tools to the model when enabled and falls
back to a single no-tools call when not."""

from __future__ import annotations

import pytest

from ai_investment_buddy.brain import prompts, web_tools
from ai_investment_buddy.brain.decide import DecisionEngine
from ai_investment_buddy.data import web

DDG_HTML = """
<html><body>
<div class="result results_links">
  <div class="result__body">
    <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Finvestor.fb.com%2Fq2&amp;rut=x">
      Meta Q2 2026 results</a>
    <a class="result__snippet">Revenue of $52.1B, up 14%.  Capex guidance raised.</a>
  </div>
</div>
<div class="result results_links">
  <div class="result__body">
    <a class="result__a" href="https://www.sec.gov/meta-8k">Meta 8-K</a>
    <a class="result__snippet">Form 8-K filed.</a>
  </div>
</div>
</body></html>
"""


def test_ddg_parse_unwraps_redirects_and_snippets():
    hits = web.DuckDuckGoSearch._parse(DDG_HTML, max_results=10)
    assert [h["url"] for h in hits] == [
        "https://investor.fb.com/q2",
        "https://www.sec.gov/meta-8k",
    ]
    assert hits[0]["title"] == "Meta Q2 2026 results"
    # Whitespace squeezed so the model sees one clean line per result.
    assert hits[0]["snippet"] == "Revenue of $52.1B, up 14%. Capex guidance raised."


def test_ddg_parse_respects_max_results():
    assert len(web.DuckDuckGoSearch._parse(DDG_HTML, max_results=1)) == 1


def test_unwrap_ddg_rejects_junk_schemes():
    assert web._unwrap_ddg("javascript:alert(1)") == ""
    assert web._unwrap_ddg("") == ""
    assert web._unwrap_ddg("//example.com/x") == "https://example.com/x"


def test_extract_text_drops_scripts_and_keeps_paragraphs():
    text = web.extract_text(
        "<html><head><style>b{}</style></head><body><nav>menu</nav>"
        "<p>Revenue rose   14%.</p><p>Capex guidance raised.</p>"
        "<script>evil()</script></body></html>"
    )
    assert "evil" not in text and "menu" not in text and "b{}" not in text
    assert "Revenue rose 14%." in text and "Capex guidance raised." in text


# --- WebResearch facade ------------------------------------------------------
class _Backend:
    def __init__(self, hits=None, boom=False):
        self.hits, self.boom, self.calls = hits or [], boom, []

    def search(self, query, max_results):
        self.calls.append((query, max_results))
        if self.boom:
            raise RuntimeError("rate limited")
        return self.hits[:max_results]


def test_search_formats_results_with_urls():
    backend = _Backend([web.SearchResult(title="T", url="https://x/y", snippet="S")])
    out = web.WebResearch(backend).search("meta earnings", max_results=3)
    assert "https://x/y" in out and "[1] T" in out
    assert backend.calls == [("meta earnings", 3)]


def test_search_degrades_to_a_message_instead_of_raising():
    out = web.WebResearch(_Backend(boom=True)).search("meta earnings")
    assert "Web search failed" in out and "rate limited" in out


def test_search_clamps_result_count_and_rejects_empty_query():
    backend = _Backend([web.SearchResult(title="T", url="https://x", snippet="")] * 50)
    web.WebResearch(backend).search("q", max_results=99)
    assert backend.calls[0][1] == 10
    assert web.WebResearch(backend).search("   ") == "Empty query."


def test_fetch_refuses_non_http_urls():
    out = web.WebResearch(_Backend()).fetch("file:///etc/passwd")
    assert "Refusing to fetch" in out


def test_fetch_extracts_text_and_truncates(monkeypatch):
    class _Raw:
        def read(self, n, decode_content=True):
            return b"<html><body><p>" + b"Revenue. " * 200 + b"</p></body></html>"

    class _Resp:
        headers = {"Content-Type": "text/html; charset=utf-8"}
        encoding = "utf-8"
        raw = _Raw()

        def raise_for_status(self):
            pass

    monkeypatch.setattr(web.requests, "get", lambda *a, **k: _Resp())
    out = web.WebResearch(_Backend()).fetch("https://investor.fb.com/q2", max_chars=500)
    assert out.startswith("=== https://investor.fb.com/q2 ===")
    assert "Revenue." in out and "[truncated]" in out


def test_fetch_reports_failure_as_text(monkeypatch):
    def _boom(*a, **k):
        raise TimeoutError("timed out")

    monkeypatch.setattr(web.requests, "get", _boom)
    out = web.WebResearch(_Backend()).fetch("https://example.com")
    assert "Fetch failed" in out and "timed out" in out


# --- Tool executor -----------------------------------------------------------
def test_executor_dispatches_and_reports_calls():
    seen = []
    research = web.WebResearch(_Backend([web.SearchResult(title="T", url="https://x", snippet="")]))
    execute = web_tools.make_web_executor(research, on_call=lambda n, a: seen.append((n, a)))
    assert "https://x" in execute("web_search", {"query": "meta"})
    assert "Refusing to fetch" in execute("fetch_url", {"url": "ftp://x"})
    assert "Unknown tool" in execute("rm_rf", {})
    assert [n for n, _ in seen] == ["web_search", "fetch_url", "rm_rf"]


def test_tool_specs_are_well_formed():
    names = {t["name"] for t in web_tools.WEB_TOOL_SPECS}
    assert names == {"web_search", "fetch_url"}
    for spec in web_tools.WEB_TOOL_SPECS:
        assert spec["description"] and spec["input_schema"]["type"] == "object"


# --- discuss() wiring --------------------------------------------------------
class _Client:
    def __init__(self):
        self.structured, self.agentic = [], []

    def structured_call(self, system, user, tool):
        self.structured.append((system, user, tool))
        return {"response": "ok", "stance": "AGREE", "ticker_notes": [], "market_note": ""}

    def agentic_call(self, system, user, helper_tools, final_tool, executor, max_iters=8):
        self.agentic.append((system, helper_tools, max_iters))
        executor("web_search", {"query": "meta earnings"})  # the model researches
        return {"response": "researched", "stance": "AGREE", "ticker_notes": [], "market_note": ""}


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr(DecisionEngine, "__init__", lambda self, client=None: None)
    eng = DecisionEngine()
    eng.client = _Client()
    return eng


def test_discuss_offers_web_tools_when_enabled(engine, monkeypatch):
    monkeypatch.setattr(web_tools, "web_tools_enabled", lambda: True)
    calls = []
    monkeypatch.setattr(
        web_tools, "make_web_executor",
        lambda research=None, on_call=None: lambda n, a: (calls.append((n, a)), "stub")[1],
    )
    out = engine.discuss("ctx", [{"role": "investor", "text": "META dropped 10%"}],
                         on_tool=lambda n, a: None)
    assert out["response"] == "researched"
    system, helper_tools, _ = engine.client.agentic[0]
    assert {t["name"] for t in helper_tools} == {"web_search", "fetch_url"}
    assert "web_search" in system  # the research contract is in the prompt
    assert calls == [("web_search", {"query": "meta earnings"})]
    assert not engine.client.structured


def test_discuss_falls_back_to_one_call_when_disabled(engine, monkeypatch):
    monkeypatch.setattr(web_tools, "web_tools_enabled", lambda: False)
    out = engine.discuss("ctx", [{"role": "investor", "text": "thoughts?"}])
    assert out["response"] == "ok"
    assert not engine.client.agentic
    # No web promise in the prompt when the tools aren't attached.
    system = engine.client.structured[0][0]
    assert "web_search" not in system and "live web access" not in system


def test_feedback_system_only_promises_research_when_armed():
    assert "RESEARCH." in prompts.feedback_system(True)
    assert prompts.feedback_system(False) == prompts.FEEDBACK_SYSTEM
