"""HTTP read API over the company corpus, plus an on-demand valuation trigger.

A thin FastAPI layer on top of the SQLite index (memory/db.py): safe concurrent
reads while a daily run writes (WAL), and filter/sort/paginate/full-text-search
that file-globbing can't do. The one write path — POST /valuate/{ticker} — runs
the existing `aib valuate` CLI in a separate process (true isolation for the
minutes-long LLM work) and lands its result in the same DB the API reads.

Launch with ``aib serve``.
"""
