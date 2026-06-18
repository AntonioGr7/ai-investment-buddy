"""Market-wide news & macro-event intelligence (free, no API key).

This is the top-down narrative the agent was missing: Fed / monetary-policy
releases and general market headlines, so a decision reflects the regime (e.g. a
Fed meeting) — not just price levels.

Feeds are pulled from public RSS. Anything that fails is skipped; the digest
degrades gracefully rather than blocking a run. Swap in a paid news API later by
implementing MarketNewsProvider and registering it in ``data/__init__.py``.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import feedparser
import requests

_UA = "Mozilla/5.0 (compatible; ai-investment-buddy/0.1; research)"

# (category, source label, url)
_MACRO_FEEDS = [
    ("MACRO/FED", "Federal Reserve (monetary policy)",
     "https://www.federalreserve.gov/feeds/press_monetary.xml"),
    ("MACRO/FED", "Federal Reserve (all press)",
     "https://www.federalreserve.gov/feeds/press_all.xml"),
    ("MACRO/FED", "CNBC Economy",
     "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258"),
]
_MARKET_FEEDS = [
    ("MARKETS", "CNBC Top News",
     "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"),
    ("MARKETS", "CNBC Markets",
     "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069"),
    ("MARKETS", "MarketWatch Top Stories",
     "http://feeds.marketwatch.com/marketwatch/topstories/"),
    ("MARKETS", "Yahoo Finance",
     "https://finance.yahoo.com/news/rssindex"),
]


def _parse_feed(url: str):
    """Fetch with a real UA (some feeds 403 the default) then parse."""
    try:
        resp = requests.get(url, headers={"User-Agent": _UA}, timeout=20)
        resp.raise_for_status()
        return feedparser.parse(resp.content)
    except Exception:
        try:
            return feedparser.parse(url, agent=_UA)
        except Exception:
            return None


def _entry_dt(entry) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        st = entry.get(key)
        if st:
            try:
                return datetime.fromtimestamp(time.mktime(st), tz=timezone.utc)
            except Exception:
                continue
    return None


class RSSMarketNews:
    def market_digest(self, days: int = 3, per_feed: int = 5) -> list[dict]:
        """Return recent market/macro headlines as dicts:
        {category, source, title, published, summary}. Newest first."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        seen_titles: set[str] = set()
        items: list[dict] = []

        for category, source, url in _MACRO_FEEDS + _MARKET_FEEDS:
            feed = _parse_feed(url)
            if not feed or not getattr(feed, "entries", None):
                continue
            count = 0
            for entry in feed.entries:
                title = (entry.get("title") or "").strip()
                if not title:
                    continue
                key = title.lower()
                if key in seen_titles:
                    continue
                dt = _entry_dt(entry)
                # Keep undated entries (some feeds omit dates) but drop clearly old ones.
                if dt is not None and dt < cutoff:
                    continue
                summary = (entry.get("summary") or "").strip()
                # RSS summaries can carry HTML; keep it short and crude-clean.
                if "<" in summary:
                    import re
                    summary = re.sub(r"<[^>]+>", "", summary)
                summary = summary[:240]
                seen_titles.add(key)
                items.append(
                    {
                        "category": category,
                        "source": source,
                        "title": title,
                        "published": dt.date().isoformat() if dt else "",
                        "summary": summary,
                    }
                )
                count += 1
                if count >= per_feed:
                    break

        # Sort: macro/Fed first, then newest first (undated entries last).
        def sort_key(it):
            cat_rank = 0 if it["category"].startswith("MACRO") else 1
            has_date = 0 if it["published"] else 1
            # ISO dates sort lexicographically; negate by using a high sentinel.
            date_desc = it["published"] or ""
            return (cat_rank, has_date, _invert(date_desc))

        items.sort(key=sort_key)
        return items


def _invert(iso_date: str) -> str:
    """Map an ISO date to a key that sorts newest-first ascending."""
    if not iso_date:
        return "0000-00-00"
    # Complement each digit so larger (later) dates produce smaller keys.
    return "".join(str(9 - int(c)) if c.isdigit() else c for c in iso_date)
