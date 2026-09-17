"""
JSON API routes.

Kept deliberately thin: parse the request, call the service, shape the
response. No SQL, no HTTP-to-Yahoo, no model code lives here — which is
what lets the same service functions back both this API and the HTML
pages in web.py.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from stockkit import MarketDataClient

from . import repository as repo
from . import scoring, service
from .config import settings
from .db import get_session
from .models import AccuracyRun, Company
from .schemas import (
    AccuracyRunOut,
    AccuracySummaryOut,
    BarOut,
    HealthOut,
    PredictionOut,
    QuoteOut,
    StockOut,
    SymbolMatchOut,
    TrackRecordOut,
)

router = APIRouter(prefix="/api", tags=["stocks"])


def _quote_out(company: Company) -> QuoteOut:
    """ORM row -> quote schema, including the computed change fields."""
    return QuoteOut(
        symbol=company.symbol,
        name=company.name,
        exchange=company.exchange,
        currency=company.currency or "USD",
        last_price=company.last_price,
        previous_close=company.previous_close,
        change=company.change,
        change_percent=company.change_percent,
        day_high=company.day_high,
        day_low=company.day_low,
        volume=company.volume,
        quote_updated_at=company.quote_updated_at,
    )


@router.get("/health", response_model=HealthOut, summary="Liveness and DB check")
def health(session: Session = Depends(get_session)) -> HealthOut:
    from . import __version__

    return HealthOut(
        status="ok",
        version=__version__,
        database="sqlite" if settings.is_sqlite else "postgresql",
        tracked_symbols=len(repo.list_companies(session)),
    )


@router.get("/search", response_model=list[SymbolMatchOut], summary="Find a ticker")
def search(
    q: str = Query(..., min_length=1, description="Ticker or company name"),
    limit: int = Query(8, ge=1, le=25),
    client: MarketDataClient = Depends(service.get_client),
) -> list[SymbolMatchOut]:
    matches = service.search_symbols(q, client, limit=limit)
    return [SymbolMatchOut.model_validate(match) for match in matches]


@router.get("/stocks", response_model=list[QuoteOut], summary="Tracked symbols")
def list_stocks(session: Session = Depends(get_session)) -> list[QuoteOut]:
    """Everything already stored locally — no upstream call."""
    return [_quote_out(company) for company in service.tracked_companies(session)]


@router.get("/stocks/{symbol}", response_model=StockOut, summary="Quote, history, forecast")
def get_stock(
    symbol: str,
    days: int = Query(365, ge=5, le=2000, description="Calendar days of history"),
    refresh: bool = Query(False, description="Force an upstream refetch"),
    session: Session = Depends(get_session),
    client: MarketDataClient = Depends(service.get_client),
) -> StockOut:
    dashboard = service.get_dashboard(session, symbol, client, force=refresh)
    session.commit()

    bars = dashboard.bars
    if days < 2000 and bars:
        cutoff = bars[-1].day.toordinal() - days
        bars = [bar for bar in bars if bar.day.toordinal() >= cutoff]

    return StockOut(
        quote=_quote_out(dashboard.company),
        bars=[BarOut.model_validate(bar) for bar in bars],
        prediction=(
            PredictionOut.from_row(dashboard.prediction) if dashboard.prediction else None
        ),
        is_stale=dashboard.is_stale,
        notice=dashboard.notice,
    )


@router.get(
    "/stocks/{symbol}/history",
    response_model=list[BarOut],
    summary="Stored daily bars",
)
def get_history(
    symbol: str,
    days: int = Query(365, ge=5, le=2000),
    session: Session = Depends(get_session),
    client: MarketDataClient = Depends(service.get_client),
) -> list[BarOut]:
    service.refresh_symbol(session, symbol, client)
    session.commit()

    bars = repo.get_bars(session, symbol)
    if not bars:
        raise HTTPException(status_code=404, detail=f"No stored history for {symbol.upper()}")

    cutoff = bars[-1].day.toordinal() - days
    return [BarOut.model_validate(bar) for bar in bars if bar.day.toordinal() >= cutoff]


@router.get(
    "/stocks/{symbol}/prediction",
    response_model=PredictionOut,
    summary="Next-day forecast",
)
def get_prediction(
    symbol: str,
    refresh: bool = Query(False),
    session: Session = Depends(get_session),
    client: MarketDataClient = Depends(service.get_client),
) -> PredictionOut:
    dashboard = service.get_dashboard(session, symbol, client, force=refresh)
    session.commit()

    if dashboard.prediction is None:
        raise HTTPException(
            status_code=422,
            detail=dashboard.notice or f"Not enough history to model {symbol.upper()}",
        )
    return PredictionOut.from_row(dashboard.prediction)


@router.post(
    "/stocks/{symbol}/refresh",
    response_model=StockOut,
    summary="Force a refetch and refit",
)
def refresh_stock(
    symbol: str,
    session: Session = Depends(get_session),
    client: MarketDataClient = Depends(service.get_client),
) -> StockOut:
    return get_stock(symbol, days=365, refresh=True, session=session, client=client)


# ---------------------------------------------------------------
# Accuracy tracking
# ---------------------------------------------------------------

def _run_out(run: AccuracyRun) -> AccuracyRunOut:
    return AccuracyRunOut(
        run_date=run.run_date,
        symbol=run.symbol,
        mae_pct=run.mae_pct,
        naive_mae_pct=run.naive_mae_pct,
        rmse_pct=run.rmse_pct,
        mae_ratio=run.mae_ratio,
        directional_accuracy=run.directional_accuracy,
        baseline_accuracy=run.baseline_accuracy,
        beats_naive=run.beats_naive,
        beats_baseline_direction=run.beats_baseline_direction,
        n_test=run.n_test,
        model_version=run.model_version,
    )


def _track_out(record: scoring.TrackRecord) -> TrackRecordOut:
    return TrackRecordOut(
        n=record.n,
        mae_pct=record.mae_pct,
        naive_mae_pct=record.naive_mae_pct,
        directional_accuracy=record.directional_accuracy,
        bias_pct=record.bias_pct,
        beats_naive=record.beats_naive,
        is_meaningful=record.is_meaningful,
    )


@router.get(
    "/accuracy",
    response_model=AccuracySummaryOut,
    summary="Latest audit plus the live track record",
)
def accuracy_summary(session: Session = Depends(get_session)) -> AccuracySummaryOut:
    """Read-only — reports what the daily job stored, and never runs it."""
    record = _track_out(scoring.track_record(session))

    dates = repo.audit_run_dates(session)
    if not dates:
        return AccuracySummaryOut(track_record=record)

    latest = dates[0]
    runs = repo.get_accuracy_runs(session, since=latest)
    runs = [run for run in runs if run.run_date == latest]

    total = sum(run.n_test for run in runs)
    hits = sum(run.directional_accuracy * run.n_test for run in runs)
    accuracy = hits / total if total else None
    z = (accuracy - 0.5) / ((0.25 / total) ** 0.5) if total and accuracy is not None else None

    import statistics

    return AccuracySummaryOut(
        run_date=latest,
        tickers_audited=len(runs),
        beat_naive=sum(1 for run in runs if run.beats_naive),
        beat_baseline_direction=sum(1 for run in runs if run.beats_baseline_direction),
        median_mae_ratio=statistics.median(run.mae_ratio for run in runs) if runs else None,
        mean_directional_accuracy=(
            statistics.fmean(run.directional_accuracy for run in runs) if runs else None
        ),
        mean_baseline_accuracy=(
            statistics.fmean(run.baseline_accuracy for run in runs) if runs else None
        ),
        pooled_direction_accuracy=accuracy,
        pooled_sample_size=total,
        pooled_z_score=z,
        pooled_is_significant=(abs(z) > 1.96) if z is not None else None,
        runs=[_run_out(run) for run in sorted(runs, key=lambda r: r.mae_ratio)],
        track_record=record,
    )


@router.get(
    "/accuracy/history",
    response_model=list[AccuracyRunOut],
    summary="Stored audit runs over time",
)
def accuracy_history(
    symbol: str | None = Query(None, description="Filter to one ticker"),
    days: int = Query(90, ge=1, le=3650, description="How far back to look"),
    session: Session = Depends(get_session),
) -> list[AccuracyRunOut]:
    from datetime import date as _date
    from datetime import timedelta

    since = _date.today() - timedelta(days=days)
    runs = repo.get_accuracy_runs(session, symbol=symbol, since=since)
    return [_run_out(run) for run in runs]


@router.get(
    "/accuracy/track-record",
    response_model=TrackRecordOut,
    summary="Hindsight-free performance on scored predictions",
)
def accuracy_track_record(
    symbol: str | None = Query(None),
    session: Session = Depends(get_session),
) -> TrackRecordOut:
    return _track_out(scoring.track_record(session, symbol))


@router.get(
    "/accuracy/predictions",
    response_model=list[PredictionOut],
    summary="Past predictions with their actual outcomes",
)
def scored_predictions(
    symbol: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    session: Session = Depends(get_session),
) -> list[PredictionOut]:
    rows = repo.scored_predictions(session, symbol=symbol, limit=limit)
    return [PredictionOut.from_row(row) for row in rows]
