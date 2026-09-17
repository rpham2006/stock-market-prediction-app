"""
Domain models — the data types the rest of the program speaks in.

If you know Java/C#, this maps directly:
    @dataclass      ->  a POJO / record / POCO
    field: float    ->  a typed field declaration
    @property       ->  a getter
    frozen=True     ->  final / readonly fields (immutable)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class PriceBar:
    """One trading day for one stock.

    @dataclass writes the constructor, __repr__, and __eq__ for you.
    In Java this class would be ~60 lines of boilerplate; here the
    field declarations ARE the class.

    frozen=True makes instances immutable — assigning to a field
    after construction raises an error. Market history is a fact that
    already happened, so it should not be editable.
    """

    day: date
    open: float
    high: float
    low: float
    close: float
    volume: int

    # Split/dividend-adjusted close. Optional because not every provider
    # sends it. Always model on THIS, never on `close`: a 4-for-1 split
    # shows up in the raw close as a -75% day that never actually happened.
    adj_close: float | None = None

    # ---- computed properties (getters, not stored fields) ----

    @property
    def model_close(self) -> float:
        """The close a model should learn from — adjusted if we have it."""
        return self.adj_close if self.adj_close is not None else self.close

    @property
    def change(self) -> float:
        """Dollar move from open to close."""
        return self.close - self.open

    @property
    def percent_change(self) -> float:
        """Percent move from open to close."""
        return (self.change / self.open) * 100.0

    @property
    def is_up_day(self) -> bool:
        return self.close > self.open

    @property
    def trading_range(self) -> float:
        """How far the price travelled during the day."""
        return self.high - self.low

    def __str__(self) -> str:
        """Human-readable form. Java's toString()."""
        arrow = "UP  " if self.is_up_day else "DOWN"
        return f"{self.day}  {arrow}  ${self.close:>8.2f}  ({self.percent_change:+.2f}%)"


@dataclass(frozen=True)
class Quote:
    """A point-in-time snapshot of a stock."""

    symbol: str
    company_name: str
    price: float
    day_high: float
    day_low: float
    currency: str = "USD"       # a default value, like an optional arg

    # Added later, all with defaults so existing callers keep working.
    previous_close: float | None = None
    volume: int | None = None
    exchange: str | None = None

    @property
    def position_in_day_range(self) -> float:
        """Where the price sits between the low and high, 0.0 to 1.0.

        Near 1.0 = closing at the top of the day's range (strength).
        """
        span = self.day_high - self.day_low
        if span == 0:
            return 0.5
        return (self.price - self.day_low) / span

    @property
    def change(self) -> float | None:
        """Dollar move since the previous session's close."""
        if self.previous_close is None:
            return None
        return self.price - self.previous_close

    @property
    def change_percent(self) -> float | None:
        """Percent move since the previous session's close."""
        if self.previous_close is None or self.previous_close == 0:
            return None
        return ((self.price - self.previous_close) / self.previous_close) * 100.0

    def value_of(self, shares: int) -> float:
        """A regular method — what N shares are worth."""
        return self.price * shares

    def __str__(self) -> str:
        return f"{self.symbol} ({self.company_name}): ${self.price:,.2f} {self.currency}"


@dataclass(frozen=True)
class SymbolMatch:
    """One result from a ticker search — enough to render a picker row."""

    symbol: str
    name: str
    exchange: str
    quote_type: str          # "EQUITY", "ETF", "INDEX", ...

    @property
    def is_tradeable_equity(self) -> bool:
        """Filter out indices and currencies, which have no usable OHLC."""
        return self.quote_type in {"EQUITY", "ETF"}

    def __str__(self) -> str:
        return f"{self.symbol} — {self.name} ({self.exchange})"
