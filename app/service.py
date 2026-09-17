"""
Orchestration — the layer that decides *when* to hit the network.

This is where the caching contract lives: the database is the source the
app reads from, and upstream is only consulted when what we hold has
aged past its TTL. Two consequences worth knowing:

  * A page view normally costs zero HTTP requests.
  * If Yahoo is down, the app still serves whatever it stored, clearly
    marked stale, instead of erroring out.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date

import numpy as np
from sqlalchemy.orm import Session

from stockkit import (
    MarketDataClient,
    MarketDataError,
    MarketDataUnavailableError,
    SymbolMatch,
    SymbolNotFoundError,
    YahooFinanceClient,
)

from . import repository as repo
from .config import settings
from .features import InsufficientHistoryError, build_features
from .models import Company, DailyBar, Prediction
from .predictor import MODEL_VERSION, Forecast, forecast_next_day

# One client per process; it is stateless apart from a request counter.
_default_client: MarketDataClient | None = None


def get_client() -> MarketDataClient:
    """The app's market data client (FastAPI dependency)."""
    global _default_client
    if _default_client is None:
        _default_client = YahooFinanceClient(
            timeout_seconds=settings.request_timeout_seconds
        )
    return _default_client


@dataclass
class Dashboard:
    """Everything one ticker page needs, assembled once."""

    company: Company
    bars: list[DailyBar]
    prediction: Prediction | None
    is_stale: bool = False           # served from cache after an upstream failure
    notice: str | None = None        # why, in words the UI can print

    @property
    def symbol(self) -> str:
        return self.company.symbol


# ---------------------------------------------------------------
# Refreshing
# ---------------------------------------------------------------

def refresh_symbol(
    session: Session,
    symbol: str,
    client: MarketDataClient,
    force: bool = False,
) -> tuple[Company, bool, str | None]:
    """Bring one symbol's stored data up to date.

    Returns (company, is_stale, notice). `is_stale` means we wanted to
    refresh, couldn't reach upstream, and fell back to stored rows.
    """
    symbol = symbol.upper()
    company = repo.get_company(session, symbol)

    need_quote = force or not repo.quote_is_fresh(company, settings.quote_ttl_minutes)
    need_bars = force or not repo.bars_are_fresh(company, settings.bar_ttl_minutes)

    if not need_quote and not need_bars:
        assert company is not None      # freshness checks imply existence
        return company, False, None

    try:
        if need_quote:
            quote = client.get_quote(symbol)
            company = repo.upsert_quote(session, quote)

        if need_bars:
            if company is None:
                # Bars need a company row to hang off (foreign key).
                company = repo.upsert_quote(session, client.get_quote(symbol))
            bars = client.get_history(symbol, period=settings.history_period)
            repo.store_bars(session, symbol, bars)

    except SymbolNotFoundError:
        # A genuinely unknown ticker is a 404, never a cache fallback.
        raise
    except MarketDataUnavailableError as exc:
        if company is None or repo.bar_count(session, symbol) == 0:
            raise
        return (
            company,
            True,
            f"Showing stored data — live refresh failed ({exc}).",
        )

    assert company is not None
    return company, False, None


# ---------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------

def features_from_bars(bars: list[DailyBar]):
    """Adapt ORM rows into the model's input arrays."""
    return build_features(
        days=[bar.day for bar in bars],
        closes=np.array([bar.model_close for bar in bars], dtype=float),
        highs=np.array([bar.high for bar in bars], dtype=float),
        lows=np.array([bar.low for bar in bars], dtype=float),
        volumes=np.array([bar.volume or 0 for bar in bars], dtype=float),
        raw_closes=np.array([bar.close for bar in bars], dtype=float),
    )


def _to_row(forecast: Forecast) -> Prediction:
    """Flatten a Forecast into the storable row."""
    backtest = forecast.backtest
    return Prediction(
        symbol=forecast.symbol,
        target_day=forecast.target_day,
        base_day=forecast.base_day,
        base_close=forecast.base_close,
        predicted_close=forecast.predicted_close,
        predicted_return_pct=forecast.predicted_return_pct,
        interval_low=forecast.interval_low,
        interval_high=forecast.interval_high,
        backtest_mae_pct=backtest.mae_pct if backtest else None,
        backtest_rmse_pct=backtest.rmse_pct if backtest else None,
        directional_accuracy=backtest.directional_accuracy if backtest else None,
        baseline_accuracy=backtest.baseline_accuracy if backtest else None,
        naive_mae_pct=backtest.naive_mae_pct if backtest else None,
        train_rows=forecast.n_train,
        test_rows=backtest.n_test if backtest else None,
        drivers_json=json.dumps([[name, weight] for name, weight in forecast.drivers]),
        model_version=forecast.model_version,
    )


def predict_symbol(
    session: Session,
    symbol: str,
    force: bool = False,
) -> Prediction | None:
    """Get the stored forecast for a symbol, computing it if needed.

    Returns None when there isn't enough stored history to model, which
    is a normal outcome for a recent IPO — not an error.
    """
    symbol = symbol.upper()
    bars = repo.get_bars(session, symbol)
    if not bars:
        return None

    base_day: date = bars[-1].day
    cached = repo.latest_prediction(session, symbol)
    if not force and repo.prediction_is_fresh(
        cached, base_day, settings.prediction_ttl_minutes
    ):
        return cached

    try:
        features = features_from_bars(bars)
    except InsufficientHistoryError:
        return None

    forecast = forecast_next_day(
        symbol,
        features,
        alpha=settings.ridge_lambda,
        min_train=settings.min_train_rows,
    )
    return repo.save_prediction(session, _to_row(forecast))


# ---------------------------------------------------------------
# The public entry points
# ---------------------------------------------------------------

def get_dashboard(
    session: Session,
    symbol: str,
    client: MarketDataClient,
    force: bool = False,
) -> Dashboard:
    """Refresh if needed, then assemble the full ticker payload."""
    company, is_stale, notice = refresh_symbol(session, symbol, client, force=force)
    bars = repo.get_bars(session, company.symbol)
    prediction = predict_symbol(session, company.symbol, force=force)

    if prediction is None and notice is None:
        notice = (
            f"Not enough stored history to model {company.symbol} yet "
            f"({len(bars)} daily bars)."
        )

    return Dashboard(
        company=company,
        bars=bars,
        prediction=prediction,
        is_stale=is_stale,
        notice=notice,
    )


def search_symbols(
    query: str,
    client: MarketDataClient,
    limit: int = 8,
) -> list[SymbolMatch]:
    """Ticker lookup. Returns [] rather than raising on upstream trouble."""
    query = query.strip()
    if not query:
        return []
    try:
        return client.search_symbols(query, limit=limit)
    except MarketDataError:
        return []


def tracked_companies(session: Session) -> list[Company]:
    """Every symbol already in the database — the homepage list."""
    return repo.list_companies(session)
