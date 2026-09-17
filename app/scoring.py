"""
Accuracy tracking — two different measurements, deliberately kept apart.

**Scoring** compares each stored prediction against the close that
actually happened. This is the only measurement with no hindsight in it:
the forecast was written down before the outcome existed. It is the
number that will still be true in a year.

**Auditing** re-runs the walk-forward backtest on today's data. It tells
you what the model would say *now*, which is useful for spotting drift
but is not evidence about the future — the same 253-bar window shifts by
one day between runs, so consecutive audits are nearly identical.

Run both daily and you get one honest track record plus one drift chart.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
from sqlalchemy.orm import Session

from stockkit import MarketDataClient, MarketDataError

from . import repository as repo
from . import service
from .config import settings
from .features import InsufficientHistoryError
from .models import AccuracyRun, Prediction, utcnow
from .predictor import MODEL_VERSION, walk_forward_backtest

logger = logging.getLogger("stockapp.scoring")

# Used when the database has no tracked tickers yet — a spread of sectors
# so the aggregate isn't dominated by one correlated cluster.
DEFAULT_UNIVERSE = (
    "SPY", "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "ORCL",
    "JPM", "BAC", "V", "JNJ", "UNH", "LLY", "XOM", "CVX",
    "WMT", "PG", "KO", "CAT", "HON",
)


# ---------------------------------------------------------------
# 1. Score past predictions against reality
# ---------------------------------------------------------------

def score_predictions(session: Session, symbol: str | None = None) -> list[Prediction]:
    """Fill in the outcome of every prediction whose target day has closed.

    Returns the predictions newly scored. Safe to run repeatedly: an
    already-scored row is never revisited.
    """
    newly_scored: list[Prediction] = []

    for prediction in repo.unscored_predictions(session, symbol):
        bar = repo.bar_on_or_after(session, prediction.symbol, prediction.target_day)
        if bar is None:
            continue                     # that session hasn't happened yet

        actual = bar.close
        prediction.actual_close = actual
        # Signed, so a positive error means the model overshot.
        prediction.error_pct = ((prediction.predicted_close - actual) / actual) * 100.0

        predicted_move = prediction.predicted_close - prediction.base_close
        actual_move = actual - prediction.base_close
        if actual_move == 0:
            prediction.direction_correct = None      # unmovable; nobody wins
        else:
            prediction.direction_correct = (predicted_move > 0) == (actual_move > 0)

        prediction.scored_at = utcnow()
        newly_scored.append(prediction)

    session.flush()
    return newly_scored


@dataclass
class TrackRecord:
    """Aggregate of every prediction that has been checked against reality."""

    n: int = 0
    mae_pct: float = 0.0
    naive_mae_pct: float = 0.0
    directional_accuracy: float = 0.0
    bias_pct: float = 0.0            # mean signed error; non-zero = systematic skew

    @property
    def beats_naive(self) -> bool:
        return self.mae_pct < self.naive_mae_pct

    @property
    def is_meaningful(self) -> bool:
        """Below ~100 scored days, direction is indistinguishable from a coin."""
        return self.n >= 100


def track_record(session: Session, symbol: str | None = None) -> TrackRecord:
    """Summarise real, hindsight-free performance so far."""
    scored = repo.scored_predictions(session, symbol)
    if not scored:
        return TrackRecord()

    errors = np.array([p.error_pct for p in scored], dtype=float)
    # The naive comparison: what the error would have been predicting no change.
    naive = np.array(
        [abs((p.base_close - p.actual_close) / p.actual_close) * 100.0 for p in scored],
        dtype=float,
    )
    directional = [p.direction_correct for p in scored if p.direction_correct is not None]

    return TrackRecord(
        n=len(scored),
        mae_pct=float(np.mean(np.abs(errors))),
        naive_mae_pct=float(np.mean(naive)),
        directional_accuracy=(
            float(np.mean(directional)) if directional else 0.0
        ),
        bias_pct=float(np.mean(errors)),
    )


# ---------------------------------------------------------------
# 2. Re-run the backtest and record the scores
# ---------------------------------------------------------------

@dataclass
class AuditReport:
    """Everything one audit run produced."""

    run_date: date
    runs: list[AccuracyRun] = field(default_factory=list)
    scored: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.runs)

    @property
    def beat_naive(self) -> int:
        return sum(1 for run in self.runs if run.beats_naive)

    @property
    def beat_direction(self) -> int:
        return sum(1 for run in self.runs if run.beats_baseline_direction)

    @property
    def median_ratio(self) -> float:
        if not self.runs:
            return float("nan")
        return float(np.median([run.mae_ratio for run in self.runs]))

    @property
    def mean_directional(self) -> float:
        if not self.runs:
            return float("nan")
        return float(np.mean([run.directional_accuracy for run in self.runs]))

    @property
    def mean_baseline(self) -> float:
        if not self.runs:
            return float("nan")
        return float(np.mean([run.baseline_accuracy for run in self.runs]))

    @property
    def pooled_direction(self) -> tuple[float, int, float]:
        """(accuracy, total predictions, z-score against a 50/50 coin).

        The z-score is the guard against reading noise as skill: with a
        few hundred samples, anything under |z| = 1.96 is not evidence.
        """
        total = sum(run.n_test for run in self.runs)
        if total == 0:
            return (float("nan"), 0, float("nan"))
        hits = sum(run.directional_accuracy * run.n_test for run in self.runs)
        accuracy = hits / total
        z = (accuracy - 0.5) / ((0.25 / total) ** 0.5)
        return (accuracy, total, z)


def audit_universe(
    session: Session,
    client: MarketDataClient,
    symbols: list[str] | None = None,
    run_date: date | None = None,
) -> AuditReport:
    """Refresh, score, then backtest every symbol and store the results."""
    run_date = run_date or date.today()
    symbols = [s.upper() for s in (symbols or default_universe(session))]
    report = AuditReport(run_date=run_date)

    for symbol in symbols:
        try:
            # Refresh first so the backtest sees the newest close, and so
            # yesterday's prediction has a bar to be scored against.
            service.refresh_symbol(session, symbol, client)
        except MarketDataError as exc:
            logger.warning("%s: refresh failed — %s", symbol, exc)
            report.failures.append((symbol, str(exc)))
            continue

        bars = repo.get_bars(session, symbol)
        try:
            features = service.features_from_bars(bars)
        except InsufficientHistoryError as exc:
            report.failures.append((symbol, str(exc)))
            continue

        backtest = walk_forward_backtest(
            features,
            alpha=settings.ridge_lambda,
            min_train=settings.min_train_rows,
        )
        if backtest is None:
            report.failures.append((symbol, "not enough rows to hold out a test set"))
            continue

        report.runs.append(
            repo.save_accuracy_run(
                session,
                AccuracyRun(
                    run_date=run_date,
                    symbol=symbol,
                    mae_pct=backtest.mae_pct,
                    naive_mae_pct=backtest.naive_mae_pct,
                    rmse_pct=backtest.rmse_pct,
                    directional_accuracy=backtest.directional_accuracy,
                    baseline_accuracy=backtest.baseline_accuracy,
                    n_test=backtest.n_test,
                    n_train=backtest.n_train,
                    bars_used=len(bars),
                    model_version=MODEL_VERSION,
                ),
            )
        )

        # Also refresh today's forecast so there is something to score
        # tomorrow. Without this the track record never starts.
        service.predict_symbol(session, symbol)

    report.scored = len(score_predictions(session))
    return report


def default_universe(session: Session) -> list[str]:
    """Audit whatever is tracked; fall back to a fixed spread of sectors."""
    tracked = [company.symbol for company in repo.list_companies(session)]
    return tracked or list(DEFAULT_UNIVERSE)
