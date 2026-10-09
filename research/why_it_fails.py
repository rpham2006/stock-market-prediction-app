r"""
Why the model doesn't predict well — a diagnostic, not a model.

`baselines.py` establishes what score would be good. This script answers the
follow-up: the model doesn't clear that bar, so is the problem the model, the
horizon, the sample size, or the data itself?

Run:  .venv\Scripts\python.exe research\why_it_fails.py
      (fetches 5y of daily bars for 10 tickers; takes ~30s)

------------------------------------------------------------------------
FINDINGS (10 tickers, 5y, ~8,300 out-of-sample predictions per horizon)
------------------------------------------------------------------------

Out-of-sample R2 is NEGATIVE at every horizon: -0.018 (1d), -0.013 (5d),
-0.038 (20d). Below zero means the model is worse than predicting the
average return — it is adding error, not explaining variance. Everything
below is why.

1. SIGNAL-TO-NOISE. Expected daily move is 2-8% of a typical daily move
   (SNR 0.017-0.076). The noise is 13-60x the signal.

2. A LONGER HORIZON DOES NOT HELP, and this is the subtle one. Raw
   directional accuracy climbs 50.6% -> 57.7% from 1d to 20d, which looks
   like a fix. But the always-up baseline climbs faster, 53.8% -> 59.8%,
   because drift makes "up" genuinely more likely over 20 days. The
   model's EDGE stays negative at every horizon. Reporting the 57.7%
   without its baseline would mean shipping a model that is worse than
   always guessing up.

3. MORE DATA DOES NOT HELP. Growing the training set 120 -> 900 rows moves
   the MAE ratio 1.019 -> 1.003: it converges TOWARD tying the naive
   baseline, never below it. That is the signature of no signal existing,
   as opposed to too little data to find one.

4. THE 13 FEATURES ARE REALLY ABOUT 7. 95% of their variance sits in 7
   principal components, with a max pairwise correlation of 0.95. They are
   all transformations of one price series, so adding more indicators of
   the same kind adds almost no independent information.

CONCLUSION: the binding constraint is information, not modelling. Price
history is the most heavily analysed data that exists; what is learnable
from it is already in the price. Improving this needs different inputs
(earnings surprises, options-implied volatility, cross-sectional ranking,
news sentiment) — not a bigger model, a longer horizon, or more bars.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.features import LOOKBACK, build_features      # noqa: E402
from app.predictor import RidgeRegressor               # noqa: E402
from stockkit import MarketDataError, YahooFinanceClient  # noqa: E402

# --- Configuration -----------------------------------------------------------

TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "JPM", "JNJ", "XOM", "KO", "PG"]
HORIZONS = [1, 5, 20]
TRAIN_SIZES = [120, 250, 500, 900]
MIN_TRAIN = 400          # 5y of data affords a real training set
YEARS = "5y"


# --- Helpers -----------------------------------------------------------------

def load_series() -> dict[str, list]:
    """Fetch daily bars for the universe, skipping anything unavailable."""
    client = YahooFinanceClient(timeout_seconds=25)
    series: dict[str, list] = {}

    for symbol in TICKERS:
        try:
            series[symbol] = client.get_history(symbol, period=YEARS)
        except MarketDataError as exc:
            print(f"  {symbol}: skipped — {exc}")
    return series


def features_for(bars: list):
    """Build the standard feature matrix from a list of PriceBar."""
    return build_features(
        days=[bar.day for bar in bars],
        closes=np.array([bar.model_close for bar in bars]),
        highs=np.array([bar.high for bar in bars]),
        lows=np.array([bar.low for bar in bars]),
        volumes=np.array([bar.volume for bar in bars]),
    )


def retarget(bars: list, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Same features, but predicting the return `horizon` sessions ahead.

    Row j of the feature matrix is decision day index LOOKBACK+j, so the
    target is simply the return from that day to day+horizon. Rows whose
    target would fall past the end of the series are dropped.
    """
    features = features_for(bars)
    closes = np.array([bar.model_close for bar in bars])

    index = np.arange(LOOKBACK, LOOKBACK + features.x.shape[0])
    usable = index + horizon < len(closes)

    x = features.x[usable]
    y = closes[index[usable] + horizon] / closes[index[usable]] - 1.0
    return x, y


def walk_forward(
    x: np.ndarray,
    y: np.ndarray,
    min_train: int,
    alpha: float = 3.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Expanding-window one-step-ahead predictions. None if too few rows."""
    n = x.shape[0]
    if n - min_train < 50:
        return None

    predictions = np.empty(n - min_train)
    model = RidgeRegressor(alpha=alpha)
    for offset, i in enumerate(range(min_train, n)):
        model.fit(x[:i], y[:i])          # strictly the past
        predictions[offset] = model.predict(x[i])[0]

    return predictions, y[min_train:]


def pooled_scores(predictions: np.ndarray, actuals: np.ndarray) -> dict[str, float]:
    """Directional edge, error ratio, and out-of-sample R2."""
    moved = actuals != 0
    directional = float(np.mean(np.sign(predictions[moved]) == np.sign(actuals[moved])))
    baseline = float(np.mean(actuals[moved] > 0))

    mae = float(np.mean(np.abs(predictions - actuals)))
    naive_mae = float(np.mean(np.abs(actuals)))

    # R2 against predicting the unconditional mean. Negative => worse
    # than guessing the average, which is the result that matters here.
    residual = float(np.sum((actuals - predictions) ** 2))
    total = float(np.sum((actuals - actuals.mean()) ** 2))

    return {
        "directional": directional,
        "baseline": baseline,
        "edge": directional - baseline,
        "mae_ratio": mae / naive_mae,
        "r2": 1 - residual / total,
        "n": int(actuals.size),
    }


def run_across(series: dict[str, list], horizon: int, min_train: int):
    """Pool walk-forward predictions over every ticker."""
    predictions, actuals = [], []

    for bars in series.values():
        x, y = retarget(bars, horizon)
        result = walk_forward(x, y, min_train)
        if result is None:
            continue
        predictions.append(result[0])
        actuals.append(result[1])

    if not predictions:
        return None
    return pooled_scores(np.concatenate(predictions), np.concatenate(actuals))


# --- The four questions ------------------------------------------------------

def report_signal_to_noise(series: dict[str, list]) -> None:
    print("=" * 74)
    print("1. SIGNAL-TO-NOISE OF DAILY RETURNS (the fundamental constraint)")
    print("=" * 74)
    print(f"{'sym':<7}{'drift/day':>12}{'sigma/day':>12}"
          f"{'SNR 1d':>10}{'SNR 5d':>10}{'SNR 20d':>10}")
    print("-" * 74)

    for symbol, bars in series.items():
        closes = np.array([bar.model_close for bar in bars])
        returns = closes[1:] / closes[:-1] - 1
        drift, sigma = returns.mean(), returns.std()

        # Mean scales with h, volatility with sqrt(h), so SNR ~ sqrt(h).
        def snr(h: int) -> float:
            return (drift * h) / (sigma * math.sqrt(h))

        print(f"{symbol:<7}{drift:>12.5f}{sigma:>12.5f}"
              f"{snr(1):>10.3f}{snr(5):>10.3f}{snr(20):>10.3f}")

    print("\n  SNR = expected move / typical move. Below ~0.1 the drift is")
    print("  invisible inside the noise.\n")


def report_horizons(series: dict[str, list]) -> None:
    print("=" * 74)
    print("2. DOES A LONGER HORIZON HELP?")
    print("=" * 74)
    print(f"{'horizon':<10}{'dir acc':>10}{'baseline':>10}{'edge':>8}"
          f"{'MAE ratio':>12}{'OOS R2':>10}{'n':>7}")
    print("-" * 74)

    for horizon in HORIZONS:
        s = run_across(series, horizon, MIN_TRAIN)
        if s is None:
            continue
        print(f"{str(horizon) + 'd':<10}{s['directional'] * 100:>9.1f}%"
              f"{s['baseline'] * 100:>9.1f}%{s['edge'] * 100:>+7.1f}"
              f"{s['mae_ratio']:>12.3f}{s['r2']:>10.4f}{s['n']:>7}")

    print("\n  edge = directional accuracy minus the always-up baseline.")
    print("  Accuracy rises with horizon but so does the baseline, and faster.")
    print("  Reporting accuracy alone would hide a negative edge.\n")


def report_sample_size(series: dict[str, list]) -> None:
    print("=" * 74)
    print("3. IS IT A SAMPLE-SIZE PROBLEM?  (1-day horizon)")
    print("=" * 74)
    print(f"{'train rows':<14}{'dir acc':>10}{'baseline':>10}"
          f"{'MAE ratio':>12}{'n test':>9}")
    print("-" * 74)

    for min_train in TRAIN_SIZES:
        s = run_across(series, horizon=1, min_train=min_train)
        if s is None:
            continue
        print(f"{min_train:<14}{s['directional'] * 100:>9.1f}%"
              f"{s['baseline'] * 100:>9.1f}%{s['mae_ratio']:>12.3f}{s['n']:>9}")

    print("\n  The ratio converges toward 1.000 but never below: more data")
    print("  teaches the model that the best guess is 'no change'.\n")


def report_feature_redundancy(series: dict[str, list]) -> None:
    print("=" * 74)
    print("4. ARE THE 13 FEATURES REALLY 13 FEATURES?")
    print("=" * 74)

    symbol, bars = next(iter(series.items()))
    features = features_for(bars)

    standardised = (features.x - features.x.mean(0)) / features.x.std(0)
    eigenvalues = np.linalg.eigvalsh(np.cov(standardised.T))[::-1]
    cumulative = np.cumsum(eigenvalues) / eigenvalues.sum()

    n_features = features.x.shape[1]
    correlations = np.corrcoef(standardised.T)
    off_diagonal = correlations[~np.eye(n_features, dtype=bool)]

    print(f"  ({symbol})")
    print(f"  variance explained by top components: 1={cumulative[0]:.0%}  "
          f"3={cumulative[2]:.0%}  5={cumulative[4]:.0%}  8={cumulative[7]:.0%}")
    print(f"  effective rank (components for 95%): "
          f"{int(np.searchsorted(cumulative, 0.95)) + 1} of {n_features}")
    print(f"  mean |correlation| between features: {np.abs(off_diagonal).mean():.2f}")
    print(f"  max  |correlation| between features: {np.abs(off_diagonal).max():.2f}")
    print("\n  Every feature is a transformation of the same price series.\n")


def main() -> int:
    print(f"Fetching {YEARS} of daily bars for {len(TICKERS)} tickers...\n")
    series = load_series()
    if not series:
        print("No data available — cannot run the diagnostic.")
        return 1

    report_signal_to_noise(series)
    report_horizons(series)
    report_sample_size(series)
    report_feature_redundancy(series)

    print("=" * 74)
    print("The constraint is information, not modelling. See the module")
    print("docstring for the full conclusion.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
