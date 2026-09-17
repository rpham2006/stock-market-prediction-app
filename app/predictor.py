"""
The model: ridge regression on technical features, validated walk-forward.

Why ridge and not something fancier. Daily returns are mostly noise, the
signal-to-noise ratio is brutal, and there are ~230 usable rows in a year
of data. A high-capacity model memorises that noise perfectly and
generalises not at all. A linear model with L2 shrinkage is the honest
choice at this sample size, and — critically — it is auditable: you can
read the coefficients and see what it believes.

Why walk-forward and not a random train/test split. Shuffling time series
lets the model train on Thursday to predict Wednesday. Every score you
get that way is a lie. Here, each prediction is made using only rows
strictly before it, which is the same information the live model has.

Read `BacktestResult.beats_naive` before believing any forecast.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from .features import FeatureSet

MODEL_VERSION = "ridge-v1"


# ---------------------------------------------------------------
# The estimator
# ---------------------------------------------------------------

class RidgeRegressor:
    """Least squares with an L2 penalty, solved in closed form.

    Standardising inside fit() matters: the penalty shrinks every
    coefficient by the same amount, so features on wildly different
    scales (a 0.01 return vs a 0.6 RSI) would otherwise be penalised
    unequally for no principled reason.
    """

    def __init__(self, alpha: float = 3.0) -> None:
        self.alpha = alpha
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._coefficients: np.ndarray | None = None
        self._intercept: float = 0.0

    @property
    def is_fitted(self) -> bool:
        return self._coefficients is not None

    def fit(self, x: np.ndarray, y: np.ndarray) -> RidgeRegressor:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        self._mean = x.mean(axis=0)
        scale = x.std(axis=0)
        # A constant column has zero spread; dividing by 1 leaves it at
        # zero, which the intercept then absorbs.
        self._scale = np.where(scale > 1e-12, scale, 1.0)
        x_scaled = (x - self._mean) / self._scale

        # Centre the target so the intercept isn't penalised.
        self._intercept = float(y.mean())
        y_centred = y - self._intercept

        n_features = x_scaled.shape[1]
        gram = x_scaled.T @ x_scaled + self.alpha * np.eye(n_features)
        self._coefficients = np.linalg.solve(gram, x_scaled.T @ y_centred)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._coefficients is None or self._mean is None or self._scale is None:
            raise RuntimeError("model must be fitted before predicting")

        x = np.atleast_2d(np.asarray(x, dtype=float))
        x_scaled = (x - self._mean) / self._scale
        return x_scaled @ self._coefficients + self._intercept

    def coefficients(self, names: tuple[str, ...]) -> list[tuple[str, float]]:
        """Feature weights, largest absolute effect first.

        These are on the standardised scale, so they are directly
        comparable: each is the effect of a one-standard-deviation move
        in that feature on the predicted next-day return.
        """
        if self._coefficients is None:
            return []
        pairs = list(zip(names, (float(c) for c in self._coefficients)))
        return sorted(pairs, key=lambda pair: abs(pair[1]), reverse=True)


# ---------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------

@dataclass(frozen=True)
class BacktestResult:
    """Out-of-sample scores, all percentages in percentage points."""

    mae_pct: float
    rmse_pct: float
    directional_accuracy: float      # share of days the sign was right
    baseline_accuracy: float         # share of days that were simply up
    naive_mae_pct: float             # error of "tomorrow's close = today's"
    residual_low_pct: float          # 10th percentile of signed error
    residual_high_pct: float         # 90th percentile of signed error
    n_test: int
    n_train: int

    @property
    def beats_naive(self) -> bool:
        """Is the model more accurate than assuming no change at all?

        This is the bar that matters. A model that loses to it is worse
        than useless — it is adding error to a random walk.
        """
        return self.mae_pct < self.naive_mae_pct

    @property
    def beats_baseline_direction(self) -> bool:
        """Does it call direction better than always guessing 'up'?"""
        return self.directional_accuracy > self.baseline_accuracy

    @property
    def verdict(self) -> str:
        """One-line honest summary for the UI."""
        if self.beats_naive and self.beats_baseline_direction:
            return "Beats both the no-change and always-up baselines out of sample."
        if self.beats_naive:
            return "Slightly more accurate than assuming no change, but no better at direction."
        if self.beats_baseline_direction:
            return "Calls direction better than chance, but is no more accurate on price."
        return "Does not beat a no-change baseline. Treat this forecast as noise."


def walk_forward_backtest(
    features: FeatureSet,
    alpha: float = 3.0,
    min_train: int = 120,
) -> BacktestResult | None:
    """Refit on every expanding window and score the one-step-ahead call.

    Returns None when there isn't enough history to hold out a test set.
    """
    x, y = features.x, features.y
    n = x.shape[0]

    # Keep a meaningful test set even on short histories.
    min_train = min(min_train, max(30, int(n * 0.6)))
    if n - min_train < 20:
        return None

    predictions = np.empty(n - min_train)
    actuals = y[min_train:]

    model = RidgeRegressor(alpha=alpha)
    for offset, i in enumerate(range(min_train, n)):
        # Train strictly on the past, predict exactly one day forward.
        model.fit(x[:i], y[:i])
        predictions[offset] = model.predict(x[i])[0]

    errors_pct = (predictions - actuals) * 100.0
    absolute_pct = np.abs(errors_pct)

    # The naive comparison: predict zero return every day. Its error is
    # simply the size of the move that actually happened.
    naive_mae_pct = float(np.mean(np.abs(actuals)) * 100.0)

    # Days the market didn't move are excluded from the direction score;
    # neither the model nor the baseline can be right about a zero.
    moved = actuals != 0
    if moved.any():
        directional = float(np.mean(np.sign(predictions[moved]) == np.sign(actuals[moved])))
        baseline = float(np.mean(actuals[moved] > 0))
    else:
        directional = baseline = 0.0

    return BacktestResult(
        mae_pct=float(absolute_pct.mean()),
        rmse_pct=float(np.sqrt(np.mean(errors_pct**2))),
        directional_accuracy=directional,
        baseline_accuracy=baseline,
        naive_mae_pct=naive_mae_pct,
        residual_low_pct=float(np.percentile(errors_pct, 10)),
        residual_high_pct=float(np.percentile(errors_pct, 90)),
        n_test=int(n - min_train),
        n_train=int(min_train),
    )


# ---------------------------------------------------------------
# The forecast
# ---------------------------------------------------------------

def next_trading_day(day: date) -> date:
    """The next weekday.

    Deliberately naive: it skips weekends but not market holidays, so a
    forecast made the day before Thanksgiving is labelled with a date the
    market is shut. Wiring in an exchange calendar is the fix — see
    ROADMAP for the market-calendar item.
    """
    nxt = day + timedelta(days=1)
    while nxt.weekday() >= 5:      # 5 = Saturday, 6 = Sunday
        nxt += timedelta(days=1)
    return nxt


@dataclass(frozen=True)
class Forecast:
    """A next-day price estimate with the evidence attached."""

    symbol: str
    base_day: date
    base_close: float
    target_day: date
    predicted_close: float
    predicted_return_pct: float
    interval_low: float
    interval_high: float
    backtest: BacktestResult | None
    drivers: list[tuple[str, float]]
    n_train: int
    model_version: str = MODEL_VERSION

    @property
    def direction(self) -> str:
        if self.predicted_return_pct > 0.05:
            return "up"
        if self.predicted_return_pct < -0.05:
            return "down"
        return "flat"


def forecast_next_day(
    symbol: str,
    features: FeatureSet,
    alpha: float = 3.0,
    min_train: int = 120,
) -> Forecast:
    """Fit on all available history and predict the next session's close.

    The interval is the 10th–90th percentile of the *backtest's own*
    errors, applied around the point estimate. It is an empirical claim —
    "80% of the time this model was this wrong" — not a theoretical
    confidence interval, and it makes no distributional assumption.
    """
    backtest = walk_forward_backtest(features, alpha=alpha, min_train=min_train)

    model = RidgeRegressor(alpha=alpha).fit(features.x, features.y)
    predicted_return = float(model.predict(features.x_live)[0])
    predicted_return_pct = predicted_return * 100.0

    base = features.base_close
    predicted_close = base * (1.0 + predicted_return)

    if backtest is not None:
        # Subtract, because residual = predicted - actual: an error band
        # skewed high means the model tends to overshoot.
        low = base * (1.0 + (predicted_return_pct - backtest.residual_high_pct) / 100.0)
        high = base * (1.0 + (predicted_return_pct - backtest.residual_low_pct) / 100.0)
    else:
        # No test set — fall back to the in-sample spread of daily moves.
        spread = float(np.std(features.y)) * 1.2816      # ~80% of a normal
        low = base * (1.0 + predicted_return - spread)
        high = base * (1.0 + predicted_return + spread)

    return Forecast(
        symbol=symbol.upper(),
        base_day=features.base_day,
        base_close=base,
        target_day=next_trading_day(features.base_day),
        predicted_close=predicted_close,
        predicted_return_pct=predicted_return_pct,
        interval_low=min(low, high),
        interval_high=max(low, high),
        backtest=backtest,
        drivers=model.coefficients(features.feature_names)[:5],
        n_train=features.n_samples,
    )
