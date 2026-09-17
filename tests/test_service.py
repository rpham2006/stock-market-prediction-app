"""
Service tests — the caching contract and the graceful-degradation path.
"""

from __future__ import annotations

import pytest

from app import repository as repo
from app import service
from app.config import settings
from app.models import utcnow
from stockkit import MarketDataUnavailableError, SymbolNotFoundError

from .conftest import FakeMarketDataClient


def test_first_load_fetches_and_stores(session, client):
    dashboard = service.get_dashboard(session, "aapl", client)

    assert dashboard.company.symbol == "AAPL"
    assert len(dashboard.bars) == 300
    assert dashboard.prediction is not None
    assert not dashboard.is_stale
    assert client.quote_calls == 1
    assert client.history_calls == 1


def test_second_load_is_served_from_the_database(session, client):
    """The whole point of storing the data: no repeat network calls."""
    service.get_dashboard(session, "AAPL", client)
    calls_after_first = (client.quote_calls, client.history_calls)

    service.get_dashboard(session, "AAPL", client)
    assert (client.quote_calls, client.history_calls) == calls_after_first


def test_force_refresh_goes_back_upstream(session, client):
    service.get_dashboard(session, "AAPL", client)
    service.get_dashboard(session, "AAPL", client, force=True)

    assert client.quote_calls == 2
    assert client.history_calls == 2


def test_stale_quote_triggers_a_refetch(session, client):
    service.get_dashboard(session, "AAPL", client)

    company = repo.get_company(session, "AAPL")
    company.quote_updated_at = utcnow().replace(year=2020)
    company.bars_updated_at = utcnow().replace(year=2020)
    session.flush()

    service.get_dashboard(session, "AAPL", client)
    assert client.quote_calls == 2


def test_unknown_symbol_raises(session, client):
    with pytest.raises(SymbolNotFoundError):
        service.get_dashboard(session, "NOPE", client)


def test_upstream_failure_falls_back_to_stored_data(session, client):
    """If the provider dies but we have rows, serve them and say so."""
    service.get_dashboard(session, "AAPL", client)

    class BrokenClient(FakeMarketDataClient):
        def get_quote(self, symbol):
            raise MarketDataUnavailableError("connection refused")

        def get_history(self, symbol, days=30, period=None):
            raise MarketDataUnavailableError("connection refused")

    dashboard = service.get_dashboard(session, "AAPL", BrokenClient(), force=True)

    assert dashboard.is_stale
    assert "stored data" in dashboard.notice
    assert len(dashboard.bars) == 300
    assert dashboard.prediction is not None


def test_upstream_failure_with_no_stored_data_raises(session):
    class BrokenClient(FakeMarketDataClient):
        def get_quote(self, symbol):
            raise MarketDataUnavailableError("connection refused")

    with pytest.raises(MarketDataUnavailableError):
        service.get_dashboard(session, "AAPL", BrokenClient())


def test_prediction_is_cached_per_base_day(session, client):
    first = service.predict_symbol(session, "AAPL")
    assert first is None          # nothing stored yet

    service.get_dashboard(session, "AAPL", client)
    cached = service.predict_symbol(session, "AAPL")
    again = service.predict_symbol(session, "AAPL")

    assert cached is not None
    assert cached.id == again.id      # same row, not recomputed


def test_short_history_yields_no_prediction(session):
    """A brand-new listing is a normal case, not an error."""
    thin_client = FakeMarketDataClient(n_bars=25)
    dashboard = service.get_dashboard(session, "AAPL", thin_client)

    assert dashboard.prediction is None
    assert "Not enough stored history" in dashboard.notice
    assert len(dashboard.bars) == 25


def test_search_passes_through(client):
    matches = service.search_symbols("AAPL", client)
    assert [m.symbol for m in matches] == ["AAPL"]


def test_search_swallows_upstream_errors():
    class BrokenClient(FakeMarketDataClient):
        def search_symbols(self, query, limit=8):
            raise MarketDataUnavailableError("down")

    assert service.search_symbols("AAPL", BrokenClient()) == []


def test_blank_search_makes_no_call(client):
    assert service.search_symbols("   ", client) == []


def test_history_window_matches_configuration():
    """The stored window is the configured one — 'up to a year ago'."""
    assert settings.history_period == "1y"
