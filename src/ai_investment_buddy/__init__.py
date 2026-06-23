"""AI Investment Buddy — an AI that allocates a paper portfolio daily and tries
to beat the S&P 500 and the Nasdaq 100."""

__version__ = "0.1.0"

# Hang guard: install a process-wide default socket timeout the moment the package
# is imported, so no network call (notably yfinance, which sets none) can block
# forever. Applies to every entry point — CLI, API, and the valuation subprocess.
from .runtime import apply_network_timeout as _apply_network_timeout

_apply_network_timeout()
