"""
stockkit — a small market data library.

The presence of this file is what makes the folder a PACKAGE rather
than just a directory of files. It defines the public surface: what
someone gets when they write `from stockkit import ...`.

Java's package + public/private modifiers, roughly.
"""

from .client import (
    MarketDataClient,
    MarketDataError,
    MarketDataUnavailableError,
    SymbolNotFoundError,
    YahooFinanceClient,
)
from .models import PriceBar, Quote, SymbolMatch

# __all__ declares the official public API of this package.
__all__ = [
    "MarketDataClient",
    "YahooFinanceClient",
    "PriceBar",
    "Quote",
    "SymbolMatch",
    "MarketDataError",
    "SymbolNotFoundError",
    "MarketDataUnavailableError",
]

__version__ = "0.1.0"
