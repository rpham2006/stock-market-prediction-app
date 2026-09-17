"""
Persistence layer — every SQL query in the app lives in this module.

Callers pass a Session in rather than opening their own. That keeps one
request inside one transaction and makes these functions trivial to test
against a throwaway in-memory database.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from stockkit import PriceBar, Quote

from .models import AccuracyRun, Company, DailyBar, Prediction, utcnow


def _aware(moment: datetime | None) -> datetime | None:
    """Normalise to UTC-aware.

    SQLite discards timezone information on write, so a value that went
    in aware comes back naive. Comparing those two raises TypeError —
    this is the single most common bug in a datetime cache check.
    """
    if moment is None:
        return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment


# ---------------------------------------------------------------
# Companies
# ---------------------------------------------------------------

def get_company(session: Session, symbol: str) -> Company | None:
    return session.get(Company, symbol.upper())


def list_companies(session: Session) -> list[Company]:
    """Every tracked ticker, most recently quoted first."""
    stmt = select(Company).order_by(Company.quote_updated_at.desc().nullslast())
    return list(session.scalars(stmt))


def upsert_quote(session: Session, quote: Quote) -> Company:
    """Insert or refresh the company row from a live quote."""
    company = session.get(Company, quote.symbol.upper())

    if company is None:
        company = Company(symbol=quote.symbol.upper(), name=quote.company_name)
        session.add(company)

    company.name = quote.company_name
    company.exchange = quote.exchange
    company.currency = quote.currency
    company.last_price = quote.price
    company.previous_close = quote.previous_close
    company.day_high = quote.day_high
    company.day_low = quote.day_low
    company.volume = quote.volume
    company.quote_updated_at = utcnow()

    session.flush()
    return company


def quote_is_fresh(company: Company | None, ttl_minutes: int) -> bool:
    if company is None or company.last_price is None:
        return False
    updated = _aware(company.quote_updated_at)
    if updated is None:
        return False
    return utcnow() - updated < timedelta(minutes=ttl_minutes)


def bars_are_fresh(company: Company | None, ttl_minutes: int) -> bool:
    if company is None:
        return False
    updated = _aware(company.bars_updated_at)
    if updated is None:
        return False
    return utcnow() - updated < timedelta(minutes=ttl_minutes)


# ---------------------------------------------------------------
# Daily bars
# ---------------------------------------------------------------

def store_bars(session: Session, symbol: str, bars: list[PriceBar]) -> tuple[int, int]:
    """Insert new bars and correct any that changed. Returns (inserted, updated).

    Safe to re-run over an overlapping window — that is the whole point
    of the (symbol, day) unique constraint. Existing rows are compared
    rather than blindly skipped because a split retroactively rewrites
    the adjusted close of every prior day.
    """
    symbol = symbol.upper()
    if not bars:
        return (0, 0)

    existing = {
        row.day: row
        for row in session.scalars(select(DailyBar).where(DailyBar.symbol == symbol))
    }

    inserted = updated = 0
    for bar in bars:
        row = existing.get(bar.day)

        if row is None:
            session.add(
                DailyBar(
                    symbol=symbol,
                    day=bar.day,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    adj_close=bar.adj_close,
                    volume=bar.volume,
                )
            )
            inserted += 1
            continue

        # Already stored — only write if the upstream numbers moved.
        if (
            row.close != bar.close
            or row.adj_close != bar.adj_close
            or row.volume != bar.volume
        ):
            row.open, row.high, row.low = bar.open, bar.high, bar.low
            row.close, row.adj_close, row.volume = bar.close, bar.adj_close, bar.volume
            updated += 1

    company = session.get(Company, symbol)
    if company is not None:
        company.bars_updated_at = utcnow()

    session.flush()
    return (inserted, updated)


def get_bars(
    session: Session,
    symbol: str,
    since: date | None = None,
    limit: int | None = None,
) -> list[DailyBar]:
    """Stored bars for a symbol, oldest first."""
    stmt = select(DailyBar).where(DailyBar.symbol == symbol.upper())
    if since is not None:
        stmt = stmt.where(DailyBar.day >= since)

    if limit is not None:
        # Take the newest N, then flip back to chronological order.
        stmt = stmt.order_by(DailyBar.day.desc()).limit(limit)
        return list(reversed(list(session.scalars(stmt))))

    return list(session.scalars(stmt.order_by(DailyBar.day)))


def bar_count(session: Session, symbol: str) -> int:
    stmt = select(func.count()).select_from(DailyBar).where(DailyBar.symbol == symbol.upper())
    return session.scalar(stmt) or 0


def latest_bar(session: Session, symbol: str) -> DailyBar | None:
    stmt = (
        select(DailyBar)
        .where(DailyBar.symbol == symbol.upper())
        .order_by(DailyBar.day.desc())
        .limit(1)
    )
    return session.scalars(stmt).first()


# ---------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------

def save_prediction(session: Session, prediction: Prediction) -> Prediction:
    """Store a forecast, replacing any earlier one for the same target day."""
    session.execute(
        delete(Prediction).where(
            Prediction.symbol == prediction.symbol,
            Prediction.target_day == prediction.target_day,
            Prediction.model_version == prediction.model_version,
        )
    )
    session.add(prediction)
    session.flush()
    return prediction


def latest_prediction(session: Session, symbol: str) -> Prediction | None:
    stmt = (
        select(Prediction)
        .where(Prediction.symbol == symbol.upper())
        .order_by(Prediction.target_day.desc(), Prediction.created_at.desc())
        .limit(1)
    )
    return session.scalars(stmt).first()


def prediction_is_fresh(
    prediction: Prediction | None,
    base_day: date,
    ttl_minutes: int,
) -> bool:
    """Reusable only if it was built on the same last close, and recently."""
    if prediction is None or prediction.base_day != base_day:
        return False
    created = _aware(prediction.created_at)
    if created is None:
        return False
    return utcnow() - created < timedelta(minutes=ttl_minutes)


def bar_on_or_after(session: Session, symbol: str, day: date) -> DailyBar | None:
    """The first stored session on or after `day`.

    On-or-after rather than exact: a forecast target that landed on a
    market holiday should be scored against the session that actually
    happened next, not left unscored forever.
    """
    stmt = (
        select(DailyBar)
        .where(DailyBar.symbol == symbol.upper(), DailyBar.day >= day)
        .order_by(DailyBar.day)
        .limit(1)
    )
    return session.scalars(stmt).first()


def unscored_predictions(session: Session, symbol: str | None = None) -> list[Prediction]:
    """Predictions whose target day has passed but which were never scored."""
    stmt = select(Prediction).where(Prediction.actual_close.is_(None))
    if symbol is not None:
        stmt = stmt.where(Prediction.symbol == symbol.upper())
    return list(session.scalars(stmt.order_by(Prediction.target_day)))


def scored_predictions(
    session: Session,
    symbol: str | None = None,
    limit: int | None = None,
) -> list[Prediction]:
    """Predictions that have been compared against reality, newest first."""
    stmt = select(Prediction).where(Prediction.actual_close.is_not(None))
    if symbol is not None:
        stmt = stmt.where(Prediction.symbol == symbol.upper())
    stmt = stmt.order_by(Prediction.target_day.desc())
    if limit is not None:
        stmt = stmt.limit(limit)
    return list(session.scalars(stmt))


# ---------------------------------------------------------------
# Accuracy audit runs
# ---------------------------------------------------------------

def save_accuracy_run(session: Session, run: AccuracyRun) -> AccuracyRun:
    """Store one ticker's scores for one day, replacing any earlier attempt."""
    session.execute(
        delete(AccuracyRun).where(
            AccuracyRun.run_date == run.run_date,
            AccuracyRun.symbol == run.symbol,
            AccuracyRun.model_version == run.model_version,
        )
    )
    session.add(run)
    session.flush()
    return run


def get_accuracy_runs(
    session: Session,
    symbol: str | None = None,
    since: date | None = None,
    limit: int | None = None,
) -> list[AccuracyRun]:
    """Audit history, newest first."""
    stmt = select(AccuracyRun)
    if symbol is not None:
        stmt = stmt.where(AccuracyRun.symbol == symbol.upper())
    if since is not None:
        stmt = stmt.where(AccuracyRun.run_date >= since)
    stmt = stmt.order_by(AccuracyRun.run_date.desc(), AccuracyRun.symbol)
    if limit is not None:
        stmt = stmt.limit(limit)
    return list(session.scalars(stmt))


def audit_run_dates(session: Session) -> list[date]:
    """Every distinct day the audit has been run, newest first."""
    stmt = select(AccuracyRun.run_date).distinct().order_by(AccuracyRun.run_date.desc())
    return list(session.scalars(stmt))
