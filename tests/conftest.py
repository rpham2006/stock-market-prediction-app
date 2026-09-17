"""
Shared test fixtures.

Nothing in the suite touches the network or the real database. A fake
market data client generates deterministic bars, and each test gets a
fresh in-memory SQLite database — which is the payoff for having put the
provider behind the MarketDataClient interface in the first place.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.models import Base
from stockkit import PriceBar, Quote, SymbolMatch
from stockkit.client import MarketDataClient, SymbolNotFoundError


# ---------------------------------------------------------------
# Synthetic market data
# ---------------------------------------------------------------

def make_bars(
    n: int = 300,
    start_price: float = 100.0,
    seed: int = 7,
    end: date | None = None,
    drift: float = 0.0004,
    sigma: float = 0.015,
) -> list[PriceBar]:
    """A deterministic geometric random walk, on consecutive weekdays."""
    rng = np.random.default_rng(seed)
    end = end or date(2026, 6, 30)

    days: list[date] = []
    day = end
    while len(days) < n:
        if day.weekday() < 5:
            days.append(day)
        day -= timedelta(days=1)
    days.reverse()

    returns = rng.normal(drift, sigma, n)
    closes = start_price * np.exp(np.cumsum(returns))

    bars: list[PriceBar] = []
    for i, close in enumerate(closes):
        wobble = abs(rng.normal(0, sigma / 2)) * close
        open_ = closes[i - 1] if i else start_price
        bars.append(
            PriceBar(
                day=days[i],
                open=float(open_),
                high=float(max(open_, close) + wobble),
                low=float(min(open_, close) - wobble),
                close=float(close),
                volume=int(rng.integers(1_000_000, 9_000_000)),
                adj_close=float(close),
            )
        )
    return bars


class FakeMarketDataClient(MarketDataClient):
    """In-memory stand-in for YahooFinanceClient.

    Implements the same interface, so anything that accepts a
    MarketDataClient accepts this — including the FastAPI dependency.
    """

    KNOWN = {"AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "KO": "Coca-Cola"}

    def __init__(self, n_bars: int = 300) -> None:
        self.n_bars = n_bars
        self.quote_calls = 0
        self.history_calls = 0

    def _check(self, symbol: str) -> str:
        symbol = symbol.upper()
        if symbol not in self.KNOWN:
            raise SymbolNotFoundError(f"No such symbol: {symbol!r}")
        return symbol

    def get_quote(self, symbol: str) -> Quote:
        symbol = self._check(symbol)
        self.quote_calls += 1
        bars = make_bars(self.n_bars)
        last, previous = bars[-1], bars[-2]
        return Quote(
            symbol=symbol,
            company_name=self.KNOWN[symbol],
            price=last.close,
            day_high=last.high,
            day_low=last.low,
            currency="USD",
            previous_close=previous.close,
            volume=last.volume,
            exchange="TestExchange",
        )

    def get_history(
        self, symbol: str, days: int = 30, period: str | None = None
    ) -> list[PriceBar]:
        self._check(symbol)
        self.history_calls += 1
        return make_bars(self.n_bars)

    def search_symbols(self, query: str, limit: int = 8) -> list[SymbolMatch]:
        query = query.upper()
        return [
            SymbolMatch(symbol=sym, name=name, exchange="TestExchange", quote_type="EQUITY")
            for sym, name in self.KNOWN.items()
            if query in sym or query in name.upper()
        ][:limit]


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture
def bars() -> list[PriceBar]:
    return make_bars()


@pytest.fixture
def client() -> FakeMarketDataClient:
    return FakeMarketDataClient()


@pytest.fixture
def session() -> Session:
    """A fresh in-memory database per test.

    StaticPool keeps every connection pointed at the same in-memory
    database; without it each connection would get its own empty one.
    """
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)

    db = factory()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()
