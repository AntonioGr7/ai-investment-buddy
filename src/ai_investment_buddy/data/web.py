"""Live web research: a search box and a page reader for the agent.

The RSS/price providers give the agent a curated, once-a-day view of the world.
That is enough for the daily cycle but useless the moment the investor says
"META just printed earnings and dropped 10%" — the numbers exist on the web and
nowhere in `data/`. This module is the missing hole in the wall: a keyless search
(DuckDuckGo HTML) with optional keyed backends (Tavily, Brave) when reliability
matters, plus a fetch-and-strip page reader.

Everything degrades gracefully: a failed search or fetch returns an explanatory
string, never an exception that kills a dialogue turn. Same contract as
``market_news``: swap a backend by implementing ``search`` and registering it.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qs, unquote, urlparse

import requests

from ..config import SETTINGS

_UA = "Mozilla/5.0 (compatible; ai-investment-buddy/0.1; research)"

# Pages that are pure boilerplate once stripped of JS — don't waste tokens.
_MIN_USEFUL_CHARS = 120


class SearchResult(dict):
    """{title, url, snippet} — a dict so backends stay trivially interchangeable."""


# --- Backends ----------------------------------------------------------------
class DuckDuckGoSearch:
    """Keyless: scrapes the no-JS HTML endpoints. Free, rate-limited, good enough
    for a handful of interactive lookups; prefer a keyed backend for volume."""

    _ENDPOINTS = (
        "https://html.duckduckgo.com/html/",
        "https://lite.duckduckgo.com/lite/",
    )

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        last_error: Exception | None = None
        for url in self._ENDPOINTS:
            try:
                resp = requests.post(
                    url,
                    data={"q": query},
                    headers={"User-Agent": _UA},
                    timeout=SETTINGS.web_timeout,
                )
                resp.raise_for_status()
                hits = self._parse(resp.text, max_results)
                if hits:
                    return hits
            except Exception as e:  # try the next endpoint
                last_error = e
        if last_error:
            raise last_error
        return []

    @staticmethod
    def _parse(html: str, max_results: int) -> list[SearchResult]:
        from lxml import html as lxml_html

        doc = lxml_html.fromstring(html)
        out: list[SearchResult] = []
        seen: set[str] = set()
        # The /html/ layout uses .result__a + .result__snippet; /lite/ uses a plain
        # table of .result-link + .result-snippet. Query both, in document order.
        # XPath, not cssselect — the latter is a separate package we don't depend on.
        for node in doc.xpath(
            "//a[contains(@class,'result__a') or contains(@class,'result-link')]"
        ):
            url = _unwrap_ddg(node.get("href") or "")
            if not url or url in seen:
                continue
            seen.add(url)
            out.append(
                SearchResult(
                    title=_squeeze(node.text_content()),
                    url=url,
                    snippet=_squeeze(_nearby_snippet(node)),
                )
            )
            if len(out) >= max_results:
                break
        return out


class TavilySearch:
    """Keyed (TAVILY_API_KEY): LLM-oriented search that returns content extracts,
    so one call often answers the question without a follow-up fetch."""

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": SETTINGS.tavily_api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": False,
            },
            timeout=SETTINGS.web_timeout,
        )
        resp.raise_for_status()
        return [
            SearchResult(
                title=_squeeze(r.get("title", "")),
                url=r.get("url", ""),
                snippet=_squeeze(r.get("content", ""))[:600],
            )
            for r in (resp.json().get("results") or [])
        ]


class BraveSearch:
    """Keyed (BRAVE_API_KEY): independent index, generous free tier."""

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": max_results},
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": SETTINGS.brave_api_key or "",
            },
            timeout=SETTINGS.web_timeout,
        )
        resp.raise_for_status()
        return [
            SearchResult(
                title=_squeeze(r.get("title", "")),
                url=r.get("url", ""),
                snippet=_squeeze(r.get("description", "")),
            )
            for r in (resp.json().get("web", {}).get("results") or [])
        ]


_BACKENDS = {"ddg": DuckDuckGoSearch, "tavily": TavilySearch, "brave": BraveSearch}


def get_search_backend():
    provider = SETTINGS.web_search_provider
    if provider not in _BACKENDS:
        raise ValueError(
            f"Unknown AIB_WEB_SEARCH_PROVIDER '{provider}'. "
            f"Choose one of: {', '.join(_BACKENDS)}."
        )
    return _BACKENDS[provider]()


# --- The agent-facing facade -------------------------------------------------
class WebResearch:
    """Two operations, both returning display-ready text for the LLM."""

    def __init__(self, backend=None) -> None:
        self._backend = backend

    @property
    def backend(self):
        if self._backend is None:
            self._backend = get_search_backend()
        return self._backend

    def search(self, query: str, max_results: int | None = None) -> str:
        query = (query or "").strip()
        if not query:
            return "Empty query."
        n = max(1, min(int(max_results or SETTINGS.web_search_results), 10))
        try:
            hits = self.backend.search(query, n)
        except Exception as e:
            return f"Web search failed ({type(e).__name__}: {e}). Say so rather than guessing."
        if not hits:
            return f"No results for '{query}'."
        lines = [f"{len(hits)} result(s) for '{query}' via {SETTINGS.web_search_provider}:"]
        for i, h in enumerate(hits, 1):
            lines.append(f"[{i}] {h['title']}\n    {h['url']}\n    {h['snippet']}")
        lines.append(
            "\n(Snippets are previews and may be stale or truncated — fetch_url the "
            "source before relying on a specific number.)"
        )
        return "\n".join(lines)

    def fetch(self, url: str, max_chars: int | None = None) -> str:
        url = (url or "").strip()
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return f"Refusing to fetch '{url}': only http(s) URLs are allowed."
        cap = max(500, min(int(max_chars or SETTINGS.web_fetch_max_chars), 20_000))
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": _UA, "Accept": "text/html,text/plain,*/*"},
                timeout=SETTINGS.web_timeout,
                stream=True,
            )
            resp.raise_for_status()
            ctype = resp.headers.get("Content-Type", "")
            if not any(t in ctype for t in ("html", "text", "xml", "json")):
                return f"{url}: unsupported content type '{ctype or 'unknown'}'."
            # Bound the download itself, not just the extracted text: a stray PDF or
            # huge page must not stall an interactive turn.
            raw = resp.raw.read(4_000_000, decode_content=True) or b""
        except Exception as e:
            return f"Fetch failed for {url} ({type(e).__name__}: {e})."

        # Hand lxml the raw bytes: it honours the document's own charset/BOM, which
        # beats requests' ISO-8859-1 default for headers with no charset (that
        # default turns every UTF-8 quote and dash into mojibake).
        text = extract_text(raw)
        if len(text) < _MIN_USEFUL_CHARS:
            return (
                f"{url}: no readable text extracted (likely JS-rendered or paywalled). "
                "Try another source."
            )
        truncated = "\n…[truncated]" if len(text) > cap else ""
        return f"=== {url} ===\n{text[:cap]}{truncated}"


def extract_text(html: str | bytes) -> str:
    """Visible text of an HTML document (str or raw bytes), chrome stripped and
    whitespace collapsed. Prefers the page's main/article region when it marks one
    — otherwise the token budget is spent on cookie banners and site navigation."""
    try:
        from lxml import html as lxml_html

        doc = lxml_html.fromstring(html)
        for bad in doc.xpath(
            "//script | //style | //noscript | //nav | //footer | //header"
            " | //aside | //form | //svg | //*[@role='navigation']"
            " | //*[@aria-hidden='true']"
        ):
            bad.drop_tree()
        main = doc.xpath("//main | //article | //*[@role='main']")
        root = main[0] if main else doc
        # text_content() concatenates blocks with no separator ("Key PointsMeta
        # shares slid…"), which mangles headline/figure boundaries. Give every block
        # element a trailing newline first.
        for block in root.xpath(
            ".//p | .//div | .//br | .//li | .//tr | .//section | .//h1 | .//h2"
            " | .//h3 | .//h4 | .//h5 | .//h6"
        ):
            block.tail = (block.tail or "") + "\n"
        text = root.text_content()
    except Exception:
        if isinstance(html, bytes):
            html = html.decode("utf-8", errors="replace")
        text = re.sub(r"<[^>]+>", " ", html)
    # Keep paragraph breaks (they carry structure), collapse everything else.
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n\s*", "\n\n", text)
    return text.strip()


def _unwrap_ddg(href: str) -> str:
    """DuckDuckGo wraps results as //duckduckgo.com/l/?uddg=<encoded>."""
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(target)
    return href if parsed.scheme in ("http", "https") else ""


def _nearby_snippet(link_node) -> str:
    """The snippet sits in a sibling of the link's result container in both layouts."""
    node = link_node
    for _ in range(4):  # walk up to the result block, then look inside it
        node = node.getparent()
        if node is None:
            return ""
        try:
            found = node.xpath(
                ".//*[contains(@class,'result__snippet')]"
                " | .//td[contains(@class,'result-snippet')]"
            )
        except Exception:
            return ""
        if found:
            return found[0].text_content()
    return ""


def _squeeze(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()
