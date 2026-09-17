"""
Feature engineering.

The one rule that matters here: **a feature for day D may only use data
known at the close of day D.** Break it and your backtest will look
spectacular and your live predictions will be worthless. That mistake
has a name — lookahead bias — and it is the single most common reason a
model that "worked" in research does nothing in production.

Everything below is derived from the adjusted close, so splits and
dividends don't masquerade as price moves.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

# Longest window any feature looks back over. The first LOOKBACK bars of
# any series can't produce a complete row and are dropped.
LOOKBACK = 20

FEATURE_NAMES: tuple[str, ...] = (
    "return_1d",
    "return_2d",
    "return_3d",
    "return_5d",
    "return_10d",
    "close_vs_sma5",
    "close_vs_sma10",
    "close_vs_sma20",
    "volatility_10d",
    "volatility_20d",
    "rsi_14",
    "range_position",
    "volume_vs_avg20",
)

# Minimum bars needed before a model can be fit at all: enough for the
# lookback, one target, and a training set worth the name.
MIN_BARS = LOOKBACK + 30


@dataclass(frozen=True)
class FeatureSet:
    """A model-ready view of one symbol's price history.

    `x_live` is deliberately separate: it is the feature row for the most
    recent bar, whose target hasn't happened yet. It is the only row the
    live forecast uses, and it must never appear in training.
    """

    x: np.ndarray            # (n_samples, n_features) — rows with known targets
    y: np.ndarray            # (n_samples,) next-day simple return, as a decimal
    days: list[date]         # decision day for each row of x
    x_live: np.ndarray       # (n_features,) features at the latest close
    base_day: date           # the latest close's date
    base_close: float        # the latest (unadjusted) close, for display
    base_model_close: float  # the latest adjusted close, what y is relative to
    feature_names: tuple[str, ...] = FEATURE_NAMES

    @property
    def n_samples(self) -> int:
        return int(self.x.shape[0])


def _rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's RSI, returned as 0..100 and aligned to `closes`.

    RSI answers "how one-sided has recent movement been?". Values are
    undefined until `period` deltas exist; those slots hold 50 (neutral)
    and get dropped anyway by the LOOKBACK trim.
    """
    rsi = np.full(closes.shape, 50.0)
    if closes.size <= period:
        return rsi

    deltas = np.diff(closes)
    gains = np.clip(deltas, 0.0, None)
    losses = np.clip(-deltas, 0.0, None)

    # Seed with a simple average, then smooth exponentially (Wilder).
    avg_gain = float(gains[:period].mean())
    avg_loss = float(losses[:period].mean())

    for i in range(period, deltas.size + 1):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period

        if avg_loss == 0.0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean aligned so index i covers values[i-window+1 : i+1]."""
    out = np.full(values.shape, np.nan)
    if values.size < window:
        return out
    cumulative = np.cumsum(np.insert(values.astype(float), 0, 0.0))
    out[window - 1:] = (cumulative[window:] - cumulative[:-window]) / window
    return out


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing standard deviation, same alignment as _rolling_mean."""
    out = np.full(values.shape, np.nan)
    for i in range(window - 1, values.size):
        out[i] = float(np.std(values[i - window + 1: i + 1]))
    return out


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """numerator/denominator - 1, with zeros where the denominator vanishes."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denominator != 0, numerator / denominator - 1.0, 0.0)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


class InsufficientHistoryError(ValueError):
    """Not enough bars to build a usable feature matrix."""


def build_features(
    days: list[date],
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    raw_closes: np.ndarray | None = None,
) -> FeatureSet:
    """Turn raw OHLCV series into a FeatureSet.

    `closes` must be adjusted closes. `raw_closes` is only carried through
    for display, so the dashboard can show the price a user would actually
    see quoted.
    """
    closes = np.asarray(closes, dtype=float)
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    n = closes.size

    if n < MIN_BARS:
        raise InsufficientHistoryError(
            f"need at least {MIN_BARS} daily bars to build features, got {n}"
        )
    if len(days) != n:
        raise ValueError("days and closes must be the same length")

    # --- momentum: return over k sessions, ending at each index ---
    def trailing_return(k: int) -> np.ndarray:
        out = np.zeros(n)
        out[k:] = _safe_ratio(closes[k:], closes[:-k])
        return out

    ret_1 = trailing_return(1)
    ret_2 = trailing_return(2)
    ret_3 = trailing_return(3)
    ret_5 = trailing_return(5)
    ret_10 = trailing_return(10)

    # --- trend: where price sits relative to its own moving averages ---
    sma5 = _rolling_mean(closes, 5)
    sma10 = _rolling_mean(closes, 10)
    sma20 = _rolling_mean(closes, 20)
    # Before a window fills, fall back to the close itself, which makes
    # the ratio 0 ("price is exactly at its average") rather than NaN.
    close_vs_sma5 = _safe_ratio(closes, np.where(np.isnan(sma5), closes, sma5))
    close_vs_sma10 = _safe_ratio(closes, np.where(np.isnan(sma10), closes, sma10))
    close_vs_sma20 = _safe_ratio(closes, np.where(np.isnan(sma20), closes, sma20))

    # --- risk: how noisy has the recent path been ---
    daily_returns = np.zeros(n)
    daily_returns[1:] = _safe_ratio(closes[1:], closes[:-1])
    vol_10 = np.nan_to_num(_rolling_std(daily_returns, 10), nan=0.0)
    vol_20 = np.nan_to_num(_rolling_std(daily_returns, 20), nan=0.0)

    # --- oscillator + intraday position + participation ---
    rsi_14 = _rsi(closes, 14) / 100.0
    span = highs - lows
    range_position = np.where(span != 0, (closes - lows) / np.where(span != 0, span, 1.0), 0.5)
    avg_volume_20 = _rolling_mean(volumes, 20)
    volume_vs_avg = _safe_ratio(
        volumes, np.where(np.isnan(avg_volume_20), volumes, avg_volume_20)
    )
    # Volume ratios have a long right tail; clip so one frenzied session
    # doesn't dominate the fit.
    volume_vs_avg = np.clip(volume_vs_avg, -1.0, 3.0)

    matrix = np.column_stack(
        [
            ret_1, ret_2, ret_3, ret_5, ret_10,
            close_vs_sma5, close_vs_sma10, close_vs_sma20,
            vol_10, vol_20,
            rsi_14, range_position, volume_vs_avg,
        ]
    )
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # --- align features to targets ---
    # Row i predicts the return from close[i] to close[i+1]. The last
    # usable training row is therefore n-2; row n-1 is the live forecast.
    first = LOOKBACK
    last_trainable = n - 2
    if last_trainable < first:
        raise InsufficientHistoryError("not enough bars after the lookback trim")

    x = matrix[first: last_trainable + 1]
    y = _safe_ratio(closes[first + 1:], closes[first: last_trainable + 1])
    sample_days = list(days[first: last_trainable + 1])

    display_closes = raw_closes if raw_closes is not None else closes

    return FeatureSet(
        x=x,
        y=y,
        days=sample_days,
        x_live=matrix[n - 1],
        base_day=days[n - 1],
        base_close=float(np.asarray(display_closes, dtype=float)[n - 1]),
        base_model_close=float(closes[n - 1]),
    )
