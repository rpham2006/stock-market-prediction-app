"""
Persistence tests — idempotent ingestion and correct cache freshness.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from app import repository as repo
from app.models import Prediction, utcnow

from .conftest import make_bars


def _quote(client, symbol="AAPL"):
    return client.get_quote(symbol)


def test_upsert_creates_then_updates(session, client):
    company = repo.upsert_quote(session, _quote(client))
    assert company.symbol == "AAPL"
    assert company.name == "Apple Inc."
    assert company.quote_updated_at is not None

    again = repo.upsert_quote(session, _quote(client))
    assert again.symbol == "AAPL"
    assert len(repo.list_companies(session)) == 1      # not duplicated


def test_store_bars_is_idempotent(session, client):
    repo.upsert_quote(session, _quote(client))
    bars = make_bars(n=120)

    inserted, updated = repo.store_bars(session, "AAPL", bars)
    assert inserted == 120
    assert updated == 0

    # Re-running the same window must be a complete no-op.
    inserted, updated = repo.store_bars(session, "AAPL", bars)
    assert (inserted, updated) == (0, 0)
    assert repo.bar_count(session, "AAPL") == 120


def test_store_bars_corrects_changed_values(session, client):
    """A restated or split-adjusted bar should overwrite, not duplicate."""
    repo.upsert_quote(session, _quote(client))
    bars = make_bars(n=60)
    repo.store_bars(session, "AAPL", bars)

    revised = list(bars[:-1]) + [
        type(bars[-1])(
            day=bars[-1].day,
            open=bars[-1].open,
            high=bars[-1].high,
            low=bars[-1].low,
            close=bars[-1].close + 5.0,
            volume=bars[-1].volume + 1,
            adj_close=bars[-1].close + 5.0,
        )
    ]
    inserted, updated = repo.store_bars(session, "AAPL", revised)

    assert (inserted, updated) == (0, 1)
    assert repo.bar_count(session, "AAPL") == 60
    assert repo.latest_bar(session, "AAPL").close == pytest.approx(bars[-1].close + 5.0)


def test_bars_come_back_in_chronological_order(session, client):
    repo.upsert_quote(session, _quote(client))
    repo.store_bars(session, "AAPL", make_bars(n=80))

    stored = repo.get_bars(session, "AAPL")
    assert [b.day for b in stored] == sorted(b.day for b in stored)


def test_get_bars_limit_returns_the_newest(session, client):
    repo.upsert_quote(session, _quote(client))
    repo.store_bars(session, "AAPL", make_bars(n=80))

    everything = repo.get_bars(session, "AAPL")
    latest_ten = repo.get_bars(session, "AAPL", limit=10)

    assert len(latest_ten) == 10
    assert [b.day for b in latest_ten] == [b.day for b in everything[-10:]]


def test_freshness_checks(session, client):
    company = repo.upsert_quote(session, _quote(client))

    assert repo.quote_is_fresh(company, ttl_minutes=5)
    assert not repo.quote_is_fresh(None, ttl_minutes=5)

    company.quote_updated_at = utcnow() - timedelta(minutes=30)
    assert not repo.quote_is_fresh(company, ttl_minutes=5)

    assert not repo.bars_are_fresh(company, ttl_minutes=60)
    repo.store_bars(session, "AAPL", make_bars(n=30))
    assert repo.bars_are_fresh(company, ttl_minutes=60)


def test_naive_timestamps_do_not_crash_freshness(session, client):
    """SQLite strips tzinfo on write; the check must survive that."""
    company = repo.upsert_quote(session, _quote(client))
    company.quote_updated_at = utcnow().replace(tzinfo=None)

    assert repo.quote_is_fresh(company, ttl_minutes=5) is True


def test_save_prediction_replaces_same_target_day(session, client):
    repo.upsert_quote(session, _quote(client))
    bars = make_bars(n=40)
    repo.store_bars(session, "AAPL", bars)

    def build(price: float) -> Prediction:
        return Prediction(
            symbol="AAPL",
            target_day=bars[-1].day,
            base_day=bars[-2].day,
            base_close=100.0,
            predicted_close=price,
            predicted_return_pct=0.5,
            interval_low=price - 2,
            interval_high=price + 2,
            model_version="ridge-v1",
        )

    repo.save_prediction(session, build(101.0))
    repo.save_prediction(session, build(102.0))

    latest = repo.latest_prediction(session, "AAPL")
    assert latest.predicted_close == pytest.approx(102.0)
    assert len(session.query(Prediction).all()) == 1


def test_prediction_freshness_requires_matching_base_day(session, client):
    repo.upsert_quote(session, _quote(client))
    bars = make_bars(n=40)
    repo.store_bars(session, "AAPL", bars)

    prediction = repo.save_prediction(
        session,
        Prediction(
            symbol="AAPL",
            target_day=bars[-1].day,
            base_day=bars[-2].day,
            base_close=100.0,
            predicted_close=101.0,
            predicted_return_pct=1.0,
            interval_low=99.0,
            interval_high=103.0,
            model_version="ridge-v1",
        ),
    )

    assert repo.prediction_is_fresh(prediction, bars[-2].day, ttl_minutes=60)
    # A new close means the forecast is stale regardless of its age.
    assert not repo.prediction_is_fresh(prediction, bars[-1].day, ttl_minutes=60)
    assert not repo.prediction_is_fresh(None, bars[-2].day, ttl_minutes=60)


def test_prediction_drivers_round_trip(session, client):
    repo.upsert_quote(session, _quote(client))
    prediction = repo.save_prediction(
        session,
        Prediction(
            symbol="AAPL",
            target_day=make_bars(n=5)[-1].day,
            base_day=make_bars(n=5)[-2].day,
            base_close=100.0,
            predicted_close=101.0,
            predicted_return_pct=1.0,
            interval_low=99.0,
            interval_high=103.0,
            drivers_json='[["return_1d", 0.5], ["rsi_14", -0.25]]',
            model_version="ridge-v1",
        ),
    )
    assert prediction.drivers == [("return_1d", 0.5), ("rsi_14", -0.25)]


def test_malformed_drivers_json_degrades_gracefully():
    prediction = Prediction(
        symbol="AAPL",
        target_day=make_bars(n=5)[-1].day,
        base_day=make_bars(n=5)[-2].day,
        base_close=100.0,
        predicted_close=101.0,
        predicted_return_pct=1.0,
        interval_low=99.0,
        interval_high=103.0,
        drivers_json="not json at all",
        model_version="ridge-v1",
    )
    assert prediction.drivers == []
