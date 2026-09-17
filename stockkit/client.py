"""
Market data clients.

Demonstrates the OOP structure you'd expect from Java or C#:

    ABC + @abstractmethod   ->  an interface / abstract class
    class Y(X)              ->  Y extends X
    _leading_underscore     ->  private by convention
    custom Exception types  ->  checked exceptions
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from .models import PriceBar, Quote, SymbolMatch   # '.' = "from this same package"


# ---------------------------------------------------------------
# Custom exceptions — a small hierarchy, exactly as in Java
# ---------------------------------------------------------------

class MarketDataError(Exception):
    """Base class for anything that goes wrong fetching data."""


class SymbolNotFoundError(MarketDataError):
    """The ticker doesn't exist."""


class MarketDataUnavailableError(MarketDataError):
    """The upstream service failed or was unreachable."""


# ---------------------------------------------------------------
# The interface
# ---------------------------------------------------------------

class MarketDataClient(ABC):
    """Interface every data provider must satisfy.

    ABC = Abstract Base Class. A subclass that doesn't implement
    every @abstractmethod cannot be instantiated — the error is
    raised at construction, not deep inside your program.

    Why bother: today this is Yahoo. In production the roadmap
    swaps in Finnhub or Polygon. Code that depends on THIS type
    keeps working when the implementation changes.
    """

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        """Return the current snapshot for one symbol."""
        ...

    @abstractmethod
    def get_history(
        self,
        symbol: str,
        days: int = 30,
        period: str | None = None,
    ) -> list[PriceBar]:
        """Return daily bars, oldest first.

        `period` is a provider-style range string ("1y", "6mo", "5y").
        When supplied it wins over `days`, which stays for older callers.
        """
        ...

    @abstractmethod
    def search_symbols(self, query: str, limit: int = 8) -> list[SymbolMatch]:
        """Look a ticker up by symbol or by company name."""
        ...

    # A concrete method on the abstract class — shared by all
    # subclasses, and written only once.
    def is_available(self) -> bool:
        """Check whether the provider is reachable."""
        try:
            self.get_quote("AAPL")
            return True
        except MarketDataError:
            return False


# ---------------------------------------------------------------
# A concrete implementation
# ---------------------------------------------------------------

class YahooFinanceClient(MarketDataClient):
    """Fetches from Yahoo Finance's public chart endpoint.

    This is Lesson 3's four steps, wrapped in a class.
    """

    BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"   # class constant
    SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"
    _USER_AGENT = "Mozilla/5.0"                                       # private constant

    def __init__(self, timeout_seconds: int = 20) -> None:
        """The constructor. 'self' is Java's implicit 'this', spelled out."""
        self.timeout_seconds = timeout_seconds
        self._request_count = 0          # private state

    @property
    def request_count(self) -> int:
        """Read-only view of a private field."""
        return self._request_count

    # ---- private helpers (the leading _ marks them internal) ----

    def _get_json(self, url: str) -> dict[str, Any]:
        """One HTTP GET returning parsed JSON.

        The single place in this class that touches the network, so
        every caller gets the same timeout and the same error mapping.
        """
        request = urllib.request.Request(url, headers={"User-Agent": self._USER_AGENT})

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise SymbolNotFoundError(f"Not found: {url}") from exc
            raise MarketDataUnavailableError(f"Yahoo returned HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise MarketDataUnavailableError(f"Could not reach Yahoo: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise MarketDataUnavailableError("Yahoo returned malformed JSON") from exc

        self._request_count += 1
        return payload

    def _fetch_chart(self, symbol: str, range_: str, interval: str) -> dict[str, Any]:
        """Fetch the chart payload for one symbol and unwrap the envelope."""
        url = (
            f"{self.BASE_URL}/{urllib.parse.quote(symbol)}"
            f"?range={range_}&interval={interval}&includeAdjustedClose=true"
        )

        try:
            payload = self._get_json(url)
        except SymbolNotFoundError as exc:
            # Re-raise with the symbol rather than the raw URL.
            raise SymbolNotFoundError(f"No such symbol: {symbol!r}") from exc

        results = payload.get("chart", {}).get("result")
        if not results:
            raise SymbolNotFoundError(f"No data returned for {symbol!r}")

        first: dict[str, Any] = results[0]
        return first

    # ---- the interface methods ----

    def get_quote(self, symbol: str) -> Quote:
        data = self._fetch_chart(symbol, range_="1d", interval="1d")
        meta = data["meta"]

        return Quote(
            symbol=meta["symbol"],
            company_name=meta.get("longName") or meta.get("shortName") or meta["symbol"],
            price=meta["regularMarketPrice"],
            day_high=meta.get("regularMarketDayHigh", meta["regularMarketPrice"]),
            day_low=meta.get("regularMarketDayLow", meta["regularMarketPrice"]),
            currency=meta.get("currency", "USD"),
            previous_close=meta.get("chartPreviousClose") or meta.get("previousClose"),
            volume=meta.get("regularMarketVolume"),
            exchange=meta.get("fullExchangeName") or meta.get("exchangeName"),
        )

    def get_history(
        self,
        symbol: str,
        days: int = 30,
        period: str | None = None,
    ) -> list[PriceBar]:
        range_ = period if period is not None else f"{days}d"
        data = self._fetch_chart(symbol, range_=range_, interval="1d")

        timestamps = data.get("timestamp") or []
        indicators = data["indicators"]
        quote = indicators["quote"][0]

        # adjclose arrives as a sibling list, and only because
        # _fetch_chart asks for includeAdjustedClose=true.
        adj_lists = indicators.get("adjclose") or []
        adj_series = adj_lists[0].get("adjclose") if adj_lists else None

        bars: list[PriceBar] = []
        for i, ts in enumerate(timestamps):
            # Yahoo sends null for days with incomplete data — skip those.
            if quote["close"][i] is None or quote["open"][i] is None:
                continue

            adj = adj_series[i] if adj_series and i < len(adj_series) else None

            bars.append(
                PriceBar(
                    day=datetime.fromtimestamp(ts, tz=timezone.utc).date(),
                    open=quote["open"][i],
                    high=quote["high"][i],
                    low=quote["low"][i],
                    close=quote["close"][i],
                    volume=int(quote["volume"][i] or 0),
                    adj_close=adj,
                )
            )
        return bars

    def search_symbols(self, query: str, limit: int = 8) -> list[SymbolMatch]:
        url = (
            f"{self.SEARCH_URL}?q={urllib.parse.quote(query)}"
            f"&quotesCount={limit}&newsCount=0"
        )
        payload = self._get_json(url)

        matches = [
            SymbolMatch(
                symbol=item["symbol"],
                name=item.get("shortname") or item.get("longname") or item["symbol"],
                exchange=item.get("exchDisp", ""),
                quote_type=item.get("quoteType", ""),
            )
            for item in payload.get("quotes", [])
            if item.get("symbol")
        ]
        # Indices and currencies come back too; they have no usable OHLC.
        return [match for match in matches if match.is_tradeable_equity][:limit]

    def __repr__(self) -> str:
        return f"YahooFinanceClient(timeout={self.timeout_seconds}s, requests={self._request_count})"
