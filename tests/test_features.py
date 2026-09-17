"""
Feature tests — most importantly, the one that proves no lookahead bias.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.features import (
    LOOKBACK,
    MIN_BARS,
    FeatureSet,
    InsufficientHistoryError,
    build_features,
)

from .conftest import make_bars


def _build(bars) -> FeatureSet:
    return build_features(
        days=[b.day for b in bars],
        closes=np.array([b.model_close for b in bars]),
        highs=np.array([b.high for b in bars]),
        lows=np.array([b.low for b in bars]),
        volumes=np.array([b.volume for b in bars]),
        raw_closes=np.array([b.close for b in bars]),
    )


def test_shapes_line_up(bars):
    features = _build(bars)

    # One row per usable day: total minus the lookback trim, minus the
    # final bar whose target hasn't happened yet.
    assert features.x.shape[0] == len(bars) - LOOKBACK - 1
    assert features.x.shape[1] == len(features.feature_names)
    assert features.y.shape[0] == features.x.shape[0]
    assert len(features.days) == features.x.shape[0]
    assert features.x_live.shape == (len(features.feature_names),)


def test_no_nan_or_inf(bars):
    features = _build(bars)
    assert np.isfinite(features.x).all()
    assert np.isfinite(features.y).all()
    assert np.isfinite(features.x_live).all()


def test_targets_are_next_day_returns(bars):
    features = _build(bars)
    closes = np.array([b.model_close for b in bars])

    # Row 0 corresponds to bar LOOKBACK; its target is the move into LOOKBACK+1.
    expected = closes[LOOKBACK + 1] / closes[LOOKBACK] - 1.0
    assert features.y[0] == pytest.approx(expected, rel=1e-12)

    # The last training row is the second-to-last bar, targeting the final one.
    expected_last = closes[-1] / closes[-2] - 1.0
    assert features.y[-1] == pytest.approx(expected_last, rel=1e-12)


def test_live_row_is_the_latest_bar(bars):
    features = _build(bars)
    assert features.base_day == bars[-1].day
    assert features.base_close == pytest.approx(bars[-1].close)
    # The live row must not be one of the training rows.
    assert features.days[-1] == bars[-2].day


def test_no_lookahead_bias(bars):
    """Rewriting the future must not change any past feature row.

    This is the test that actually matters. If a feature accidentally
    used forward-looking data, mutating the tail of the series would
    ripple backwards and this comparison would fail.
    """
    baseline = _build(bars)

    tampered = list(bars[:-3]) + [
        type(bar)(
            day=bar.day,
            open=bar.open * 3,
            high=bar.high * 3,
            low=bar.low * 3,
            close=bar.close * 3,
            volume=bar.volume * 9,
            adj_close=(bar.adj_close or bar.close) * 3,
        )
        for bar in bars[-3:]
    ]
    after = _build(tampered)

    # Every row whose target predates the tampering must be identical.
    unaffected = baseline.x.shape[0] - 4
    np.testing.assert_allclose(baseline.x[:unaffected], after.x[:unaffected], rtol=1e-12)
    np.testing.assert_allclose(baseline.y[:unaffected], after.y[:unaffected], rtol=1e-12)


def test_rsi_stays_in_range(bars):
    features = _build(bars)
    rsi_index = features.feature_names.index("rsi_14")
    column = features.x[:, rsi_index]
    assert column.min() >= 0.0
    assert column.max() <= 1.0


def test_range_position_stays_in_range(bars):
    features = _build(bars)
    index = features.feature_names.index("range_position")
    column = features.x[:, index]
    assert column.min() >= 0.0
    assert column.max() <= 1.0


def test_short_history_is_rejected():
    short = make_bars(n=MIN_BARS - 5)
    with pytest.raises(InsufficientHistoryError):
        _build(short)


def test_flat_series_does_not_divide_by_zero():
    """A stock that never moves must produce zeros, not NaN."""
    n = 120
    closes = np.full(n, 50.0)
    days = [b.day for b in make_bars(n=n)]

    features = build_features(
        days=days,
        closes=closes,
        highs=closes.copy(),
        lows=closes.copy(),
        volumes=np.full(n, 1000.0),
    )
    assert np.isfinite(features.x).all()
    assert np.allclose(features.y, 0.0)
