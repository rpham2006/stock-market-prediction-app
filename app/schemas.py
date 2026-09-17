"""
Pydantic response models — the public shape of the JSON API.

These exist so the API contract is declared in one place rather than
emerging accidentally from whatever the ORM happens to expose. They also
give FastAPI what it needs to generate the OpenAPI docs at /docs.

Convention (ROADMAP §API): snake_case keys, ISO 8601 dates, and a
consistent error envelope.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field

_FROM_ORM = ConfigDict(from_attributes=True)


class SymbolMatchOut(BaseModel):
    """One ticker-search hit."""

    model_config = _FROM_ORM

    symbol: str
    name: str
    exchange: str
    quote_type: str


class QuoteOut(BaseModel):
    """Current snapshot for one company."""

    model_config = _FROM_ORM

    symbol: str
    name: str
    exchange: str | None = None
    currency: str = "USD"
    last_price: float | None = None
    previous_close: float | None = None
    change: float | None = None
    change_percent: float | None = None
    day_high: float | None = None
    day_low: float | None = None
    volume: int | None = None
    quote_updated_at: datetime | None = None


class BarOut(BaseModel):
    """One stored trading day."""

    model_config = _FROM_ORM

    day: date
    open: float
    high: float
    low: float
    close: float
    adj_close: float | None = None
    volume: int


class BacktestOut(BaseModel):
    """How the model scored out of sample. Read this before the forecast."""

    mae_pct: float | None = Field(
        None, description="Mean absolute error of the predicted next-day close, in percent."
    )
    rmse_pct: float | None = None
    naive_mae_pct: float | None = Field(
        None, description="Error of assuming tomorrow's close equals today's."
    )
    directional_accuracy: float | None = Field(
        None, description="Share of days the predicted sign was correct, 0..1."
    )
    baseline_accuracy: float | None = Field(
        None, description="Share of days that simply rose — the always-up baseline."
    )
    beats_naive: bool | None = Field(
        None, description="Whether the model is more accurate than no-change."
    )
    train_rows: int | None = None
    test_rows: int | None = None


class PredictionOut(BaseModel):
    """A next-day forecast plus the evidence for trusting it."""

    model_config = _FROM_ORM

    symbol: str
    base_day: date
    base_close: float
    target_day: date
    predicted_close: float
    predicted_return_pct: float
    interval_low: float
    interval_high: float
    direction: str
    model_version: str
    created_at: datetime
    drivers: list[tuple[str, float]] = []
    backtest: BacktestOut | None = None

    # --- outcome, once target_day has closed ---
    actual_close: float | None = None
    error_pct: float | None = Field(
        None, description="Signed error in percent; positive means the model overshot."
    )
    direction_correct: bool | None = None

    @classmethod
    def from_row(cls, row) -> PredictionOut:
        """Build from the ORM row, flattening the backtest columns."""
        change = row.predicted_return_pct
        direction = "up" if change > 0.05 else "down" if change < -0.05 else "flat"

        return cls(
            symbol=row.symbol,
            base_day=row.base_day,
            base_close=row.base_close,
            target_day=row.target_day,
            predicted_close=row.predicted_close,
            predicted_return_pct=change,
            interval_low=row.interval_low,
            interval_high=row.interval_high,
            direction=direction,
            model_version=row.model_version,
            created_at=row.created_at,
            drivers=row.drivers,
            actual_close=row.actual_close,
            error_pct=row.error_pct,
            direction_correct=row.direction_correct,
            backtest=BacktestOut(
                mae_pct=row.backtest_mae_pct,
                rmse_pct=row.backtest_rmse_pct,
                naive_mae_pct=row.naive_mae_pct,
                directional_accuracy=row.directional_accuracy,
                baseline_accuracy=row.baseline_accuracy,
                beats_naive=row.beats_naive,
                train_rows=row.train_rows,
                test_rows=row.test_rows,
            ),
        )


class StockOut(BaseModel):
    """The full ticker payload: quote, history, forecast."""

    quote: QuoteOut
    bars: list[BarOut]
    prediction: PredictionOut | None = None
    is_stale: bool = False
    notice: str | None = None
    disclaimer: str = (
        "For educational and informational purposes only. Not investment advice. "
        "Predictions are model output, not forecasts of actual prices."
    )


class AccuracyRunOut(BaseModel):
    """One ticker's backtest scores as measured on one day."""

    model_config = _FROM_ORM

    run_date: date
    symbol: str
    mae_pct: float
    naive_mae_pct: float
    rmse_pct: float | None = None
    mae_ratio: float = Field(
        ..., description="MAE divided by the naive baseline. Below 1.0 means it helped."
    )
    directional_accuracy: float
    baseline_accuracy: float
    beats_naive: bool
    beats_baseline_direction: bool
    n_test: int
    model_version: str


class TrackRecordOut(BaseModel):
    """Hindsight-free performance: what the model said, versus what happened."""

    n: int = Field(..., description="Predictions scored against a real close.")
    mae_pct: float
    naive_mae_pct: float
    directional_accuracy: float
    bias_pct: float = Field(
        ..., description="Mean signed error. Away from zero means a systematic skew."
    )
    beats_naive: bool
    is_meaningful: bool = Field(
        ..., description="False below ~100 samples, where direction is a coin flip."
    )


class AccuracySummaryOut(BaseModel):
    """The latest audit, its aggregate, and the live track record."""

    run_date: date | None = None
    tickers_audited: int = 0
    beat_naive: int = 0
    beat_baseline_direction: int = 0
    median_mae_ratio: float | None = None
    mean_directional_accuracy: float | None = None
    mean_baseline_accuracy: float | None = None
    pooled_direction_accuracy: float | None = None
    pooled_sample_size: int = 0
    pooled_z_score: float | None = None
    pooled_is_significant: bool | None = None
    runs: list[AccuracyRunOut] = []
    track_record: TrackRecordOut | None = None


class HealthOut(BaseModel):
    status: str
    version: str
    database: str
    tracked_symbols: int


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorOut(BaseModel):
    """The single error envelope every failure uses."""

    error: ErrorBody
