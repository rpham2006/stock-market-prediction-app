"""
ORM models — the database schema, expressed as Python classes.

Do not confuse these with `stockkit.models`. Those are plain immutable
value objects describing market data; these are rows with primary keys
and indexes. The service layer translates between the two, which keeps
the storage schema free to change without touching the data client.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import json

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    """Timezone-aware UTC now. ROADMAP: ISO 8601 UTC everywhere."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Shared parent for every mapped class."""


class Company(Base):
    """One tracked ticker, plus its most recent quote snapshot.

    The quote columns are denormalised onto this row on purpose: there
    is exactly one "current price" per symbol, and storing it here means
    the dashboard reads a single row instead of a sorted subquery.
    """

    __tablename__ = "companies"

    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    exchange: Mapped[str | None] = mapped_column(String(80))
    currency: Mapped[str] = mapped_column(String(10), default="USD")

    # --- latest quote snapshot ---
    last_price: Mapped[float | None] = mapped_column(Float)
    previous_close: Mapped[float | None] = mapped_column(Float)
    day_high: Mapped[float | None] = mapped_column(Float)
    day_low: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[int | None] = mapped_column(Integer)

    # --- cache bookkeeping: what did we refresh, and when ---
    quote_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    bars_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    bars: Mapped[list[DailyBar]] = relationship(
        back_populates="company",
        cascade="all, delete-orphan",
        order_by="DailyBar.day",
    )
    predictions: Mapped[list[Prediction]] = relationship(
        back_populates="company",
        cascade="all, delete-orphan",
    )

    @property
    def change(self) -> float | None:
        if self.last_price is None or self.previous_close is None:
            return None
        return self.last_price - self.previous_close

    @property
    def change_percent(self) -> float | None:
        if self.last_price is None or not self.previous_close:
            return None
        return ((self.last_price - self.previous_close) / self.previous_close) * 100.0

    def __repr__(self) -> str:
        return f"<Company {self.symbol} {self.name!r}>"


class DailyBar(Base):
    """One trading day of OHLCV for one symbol.

    The (symbol, day) unique constraint is what makes ingestion safe to
    re-run: a day already stored can never be duplicated, so refreshing
    an overlapping window is a no-op rather than a mess.
    """

    __tablename__ = "daily_bars"
    __table_args__ = (
        UniqueConstraint("symbol", "day", name="uq_bar_symbol_day"),
        Index("ix_bar_symbol_day", "symbol", "day"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(
        String(20), ForeignKey("companies.symbol", ondelete="CASCADE"), nullable=False
    )
    day: Mapped[date] = mapped_column(Date, nullable=False)

    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    adj_close: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[int] = mapped_column(Integer, default=0)

    company: Mapped[Company] = relationship(back_populates="bars")

    @property
    def model_close(self) -> float:
        """Adjusted close when present — the series the model learns from."""
        return self.adj_close if self.adj_close is not None else self.close

    def __repr__(self) -> str:
        return f"<DailyBar {self.symbol} {self.day} close={self.close:.2f}>"


class Prediction(Base):
    """A stored next-day forecast, with the evidence behind it.

    Persisting predictions (rather than recomputing on every page view)
    is what later lets you score the model on what it *actually said at
    the time* instead of what it would say now knowing the outcome.
    """

    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("symbol", "target_day", "model_version", name="uq_pred_symbol_day"),
        Index("ix_pred_symbol_target", "symbol", "target_day"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(
        String(20), ForeignKey("companies.symbol", ondelete="CASCADE"), nullable=False
    )

    # The session being forecast, and the last close it was based on.
    target_day: Mapped[date] = mapped_column(Date, nullable=False)
    base_day: Mapped[date] = mapped_column(Date, nullable=False)
    base_close: Mapped[float] = mapped_column(Float, nullable=False)

    predicted_close: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_return_pct: Mapped[float] = mapped_column(Float, nullable=False)
    # 80% band, derived from the spread of walk-forward backtest errors.
    interval_low: Mapped[float] = mapped_column(Float, nullable=False)
    interval_high: Mapped[float] = mapped_column(Float, nullable=False)

    # --- how much to trust the above ---
    backtest_mae_pct: Mapped[float | None] = mapped_column(Float)
    backtest_rmse_pct: Mapped[float | None] = mapped_column(Float)
    directional_accuracy: Mapped[float | None] = mapped_column(Float)
    baseline_accuracy: Mapped[float | None] = mapped_column(Float)
    naive_mae_pct: Mapped[float | None] = mapped_column(Float)
    train_rows: Mapped[int | None] = mapped_column(Integer)
    test_rows: Mapped[int | None] = mapped_column(Integer)

    # Top standardised coefficients, as JSON [[name, weight], ...]. Stored
    # so a cached prediction can still explain itself without a refit.
    drivers_json: Mapped[str | None] = mapped_column(Text)

    # --- outcome: filled in once target_day has actually closed ---
    # This is the only measurement that scores the model on what it said
    # *at the time*, with no hindsight. A backtest re-run today can only
    # tell you what the model would say now.
    actual_close: Mapped[float | None] = mapped_column(Float)
    error_pct: Mapped[float | None] = mapped_column(Float)      # signed: + = overshot
    direction_correct: Mapped[bool | None] = mapped_column(Boolean)
    scored_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    model_version: Mapped[str] = mapped_column(String(40), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    company: Mapped[Company] = relationship(back_populates="predictions")

    @property
    def drivers(self) -> list[tuple[str, float]]:
        """The features that moved this forecast most, strongest first."""
        if not self.drivers_json:
            return []
        try:
            return [(str(n), float(w)) for n, w in json.loads(self.drivers_json)]
        except (ValueError, TypeError):
            return []

    @property
    def direction(self) -> str:
        """Coarse call, with a dead zone so noise isn't dressed up as signal."""
        if self.predicted_return_pct > 0.05:
            return "up"
        if self.predicted_return_pct < -0.05:
            return "down"
        return "flat"

    @property
    def is_scored(self) -> bool:
        """Has the target day closed and been compared against?"""
        return self.actual_close is not None

    @property
    def beats_naive(self) -> bool | None:
        """Did the model actually improve on 'tomorrow equals today'?"""
        if self.backtest_mae_pct is None or self.naive_mae_pct is None:
            return None
        return self.backtest_mae_pct < self.naive_mae_pct

    @property
    def beats_baseline_direction(self) -> bool | None:
        """Did it call direction better than always guessing 'up'?"""
        if self.directional_accuracy is None or self.baseline_accuracy is None:
            return None
        return self.directional_accuracy > self.baseline_accuracy

    @property
    def verdict(self) -> str:
        """The honest one-liner the dashboard leads with."""
        price_ok, direction_ok = self.beats_naive, self.beats_baseline_direction
        if price_ok is None or direction_ok is None:
            return "Not enough out-of-sample history to score this model."
        if price_ok and direction_ok:
            return "Beats both the no-change and always-up baselines out of sample."
        if price_ok:
            return "Slightly more accurate than assuming no change, but no better at direction."
        if direction_ok:
            return "Calls direction better than chance, but is no more accurate on price."
        return "Does not beat a no-change baseline. Treat this forecast as noise."

    @property
    def verdict_level(self) -> str:
        """'good' | 'mixed' | 'poor' — drives the badge colour in the UI."""
        price_ok, direction_ok = self.beats_naive, self.beats_baseline_direction
        if price_ok is None or direction_ok is None:
            return "mixed"
        if price_ok and direction_ok:
            return "good"
        if price_ok or direction_ok:
            return "mixed"
        return "poor"

    def __repr__(self) -> str:
        return (
            f"<Prediction {self.symbol} {self.target_day} "
            f"{self.predicted_close:.2f} ({self.predicted_return_pct:+.2f}%)>"
        )


class AccuracyRun(Base):
    """One ticker's backtest scores, as measured on one day.

    A row per (run_date, symbol). Re-running the audit on the same day
    overwrites rather than accumulating, so the table stays one clean
    time series you can plot model drift from.
    """

    __tablename__ = "accuracy_runs"
    __table_args__ = (
        UniqueConstraint("run_date", "symbol", "model_version", name="uq_audit_day_symbol"),
        Index("ix_audit_symbol_date", "symbol", "run_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_date: Mapped[date] = mapped_column(Date, nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)

    mae_pct: Mapped[float] = mapped_column(Float, nullable=False)
    naive_mae_pct: Mapped[float] = mapped_column(Float, nullable=False)
    rmse_pct: Mapped[float | None] = mapped_column(Float)
    directional_accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    baseline_accuracy: Mapped[float] = mapped_column(Float, nullable=False)

    n_test: Mapped[int] = mapped_column(Integer, nullable=False)
    n_train: Mapped[int | None] = mapped_column(Integer)
    bars_used: Mapped[int | None] = mapped_column(Integer)

    model_version: Mapped[str] = mapped_column(String(40), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    @property
    def mae_ratio(self) -> float:
        """MAE relative to the naive baseline. Below 1.0 means it helped."""
        if not self.naive_mae_pct:
            return float("nan")
        return self.mae_pct / self.naive_mae_pct

    @property
    def beats_naive(self) -> bool:
        return self.mae_pct < self.naive_mae_pct

    @property
    def beats_baseline_direction(self) -> bool:
        return self.directional_accuracy > self.baseline_accuracy

    def __repr__(self) -> str:
        return (
            f"<AccuracyRun {self.run_date} {self.symbol} "
            f"mae={self.mae_pct:.2f}% ratio={self.mae_ratio:.3f}>"
        )
