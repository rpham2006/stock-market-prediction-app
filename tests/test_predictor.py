"""
Model tests — does the estimator learn, and is the scoring honest?
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from app.features import build_features
from app.predictor import (
    RidgeRegressor,
    forecast_next_day,
    next_trading_day,
    walk_forward_backtest,
)

from .conftest import make_bars


def _features(bars):
    return build_features(
        days=[b.day for b in bars],
        closes=np.array([b.model_close for b in bars]),
        highs=np.array([b.high for b in bars]),
        lows=np.array([b.low for b in bars]),
        volumes=np.array([b.volume for b in bars]),
        raw_closes=np.array([b.close for b in bars]),
    )


# ---------------------------------------------------------------
# The estimator
# ---------------------------------------------------------------

def test_ridge_recovers_a_known_signal():
    """Given y built from x0 alone, x0 must dominate the coefficients."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 4))
    y = 2.5 * x[:, 0] + rng.normal(0, 0.1, size=400)

    model = RidgeRegressor(alpha=1.0).fit(x, y)
    weights = model.coefficients(("a", "b", "c", "d"))

    assert weights[0][0] == "a"
    assert weights[0][1] > 0
    # The real driver should outweigh every pure-noise feature.
    assert abs(weights[0][1]) > 5 * max(abs(w) for name, w in weights[1:])


def test_ridge_predicts_close_to_truth():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, 3))
    y = 1.5 * x[:, 0] - 0.8 * x[:, 1]

    model = RidgeRegressor(alpha=0.01).fit(x, y)
    predictions = model.predict(x)
    assert np.mean(np.abs(predictions - y)) < 0.15


def test_penalty_shrinks_coefficients():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(200, 5))
    y = x[:, 0] * 3 + rng.normal(0, 0.5, size=200)

    weak = RidgeRegressor(alpha=0.01).fit(x, y).coefficients(tuple("abcde"))
    strong = RidgeRegressor(alpha=500.0).fit(x, y).coefficients(tuple("abcde"))

    assert abs(strong[0][1]) < abs(weak[0][1])


def test_predicting_before_fitting_raises():
    with pytest.raises(RuntimeError):
        RidgeRegressor().predict(np.zeros((1, 3)))


def test_constant_feature_column_does_not_blow_up():
    """A zero-variance column would divide by zero if unguarded."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(100, 3))
    x[:, 1] = 7.0
    y = rng.normal(size=100)

    model = RidgeRegressor().fit(x, y)
    assert np.isfinite(model.predict(x)).all()


# ---------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------

def test_backtest_metrics_are_sane(bars):
    result = walk_forward_backtest(_features(bars))

    assert result is not None
    assert result.mae_pct > 0
    assert result.rmse_pct >= result.mae_pct       # RMSE never below MAE
    assert 0.0 <= result.directional_accuracy <= 1.0
    assert 0.0 <= result.baseline_accuracy <= 1.0
    assert result.naive_mae_pct > 0
    assert result.n_test > 0
    assert result.residual_low_pct <= result.residual_high_pct


def test_backtest_returns_none_without_enough_rows():
    """A short series can't hold out a test set, and says so."""
    short = make_bars(n=60)
    assert walk_forward_backtest(_features(short), min_train=55) is None


def test_verdict_matches_the_numbers(bars):
    result = walk_forward_backtest(_features(bars))
    assert result is not None

    assert result.beats_naive == (result.mae_pct < result.naive_mae_pct)
    assert result.beats_baseline_direction == (
        result.directional_accuracy > result.baseline_accuracy
    )
    if not result.beats_naive and not result.beats_baseline_direction:
        assert "noise" in result.verdict


def test_random_walk_does_not_beat_the_naive_baseline():
    """Sanity check on the scoring itself.

    The synthetic series is a pure random walk with no learnable
    structure. If the backtest claimed the model beat a no-change
    baseline here, the scoring would be the thing that's broken.
    """
    result = walk_forward_backtest(_features(make_bars(n=400, seed=99)))
    assert result is not None
    assert not result.beats_naive


# ---------------------------------------------------------------
# The forecast
# ---------------------------------------------------------------

def test_forecast_is_internally_consistent(bars):
    forecast = forecast_next_day("TEST", _features(bars))

    assert forecast.symbol == "TEST"
    assert forecast.base_day == bars[-1].day
    assert forecast.target_day > forecast.base_day
    assert forecast.interval_low <= forecast.predicted_close <= forecast.interval_high

    # The stated return must actually reproduce the stated price.
    implied = forecast.base_close * (1 + forecast.predicted_return_pct / 100)
    assert forecast.predicted_close == pytest.approx(implied, rel=1e-9)


def test_forecast_stays_in_a_plausible_range(bars):
    """A one-day move of more than a few percent from this model is a bug."""
    forecast = forecast_next_day("TEST", _features(bars))
    assert abs(forecast.predicted_return_pct) < 10.0


def test_forecast_direction_has_a_dead_zone(bars):
    forecast = forecast_next_day("TEST", _features(bars))
    if forecast.direction == "flat":
        assert abs(forecast.predicted_return_pct) <= 0.05
    elif forecast.direction == "up":
        assert forecast.predicted_return_pct > 0
    else:
        assert forecast.predicted_return_pct < 0


def test_forecast_lists_its_drivers(bars):
    forecast = forecast_next_day("TEST", _features(bars))
    assert 1 <= len(forecast.drivers) <= 5

    names = [name for name, _ in forecast.drivers]
    assert all(name in _features(bars).feature_names for name in names)

    # Sorted by descending absolute weight.
    weights = [abs(w) for _, w in forecast.drivers]
    assert weights == sorted(weights, reverse=True)


def test_forecast_is_deterministic(bars):
    """Same input, same output — no hidden randomness in the pipeline."""
    first = forecast_next_day("TEST", _features(bars))
    second = forecast_next_day("TEST", _features(bars))
    assert first.predicted_close == second.predicted_close


# ---------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------

@pytest.mark.parametrize(
    "given, expected",
    [
        (date(2026, 9, 3), date(2026, 9, 4)),    # Thu -> Fri
        (date(2026, 9, 4), date(2026, 9, 7)),    # Fri -> Mon
        (date(2026, 9, 5), date(2026, 9, 7)),    # Sat -> Mon
        (date(2026, 9, 6), date(2026, 9, 7)),    # Sun -> Mon
    ],
)
def test_next_trading_day_skips_weekends(given, expected):
    assert next_trading_day(given) == expected
