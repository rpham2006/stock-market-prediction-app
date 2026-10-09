r"""
Does an error band that scales with volatility beat a fixed one?

The dashboard's "80% range" is the 10th-90th percentile of the backtest's
errors, applied around the point forecast. That band has one width for
every day. But volatility clusters: a wild week is usually followed by
another, a calm one by calm. A fixed band is therefore too wide on calm
days and too narrow on wild ones — it can hit 80% overall while missing
far more often exactly when it matters.

The alternative: divide each past error by the volatility estimate known
at the time, take percentiles of those standardised errors, and multiply
back by today's volatility. Same data, no new inputs. `beta` sets how hard
the band reacts: the scale is sigma**beta, so beta=0 is the fixed band and
beta=1 is full proportional scaling.

Every band below is built strictly from errors already realised:
  - point forecast = ridge at the shipped alpha, walk-forward
  - band at step t uses only the last RESIDUAL_WINDOW errors before t
    (~110, which is what the app gets from 1y of bars minus min_train)

Scores:
  coverage   share of outcomes inside the band. Target 80%.
  by vol     coverage on the calmest / middle / wildest third of days for
             that ticker. A good band hits ~80% in all three.
  width      average band width, % of price. Narrower is better ONLY at
             equal coverage.
  score      Winkler interval score at alpha=0.2: width plus 10x any miss
             distance. A proper scoring rule — lower is better, and it
             can't be gamed by trading width against coverage.

Run:  .venv\Scripts\python.exe research\intervals.py
      (fetches 5y of daily bars for 10 tickers; takes ~15s)

------------------------------------------------------------------------
FINDINGS (10 tickers, 5y, 7,720 out-of-sample days)
------------------------------------------------------------------------

1. VOLATILITY DOES PREDICT THE SIZE OF TOMORROW'S MOVE: correlation
   between EWMA sigma and |next return| is 0.11-0.24 per ticker. Unlike
   the direction of the move, its size is forecastable.

2. FULL SCALING OVERREACTS. beta=1 covers 71.7% on calm days and 84.3%
   on wild ones, and scores worse than the fixed band (5.929 vs 5.917).
   The fixed band isn't as naive as it looks: its ~110-error window
   already drifts with the volatility regime.

3. HALF SCALING WINS, MODESTLY. beta=0.5: score 5.858 (-1.0%), coverage
   77.7% -> 78.3%, wild-day coverage 75.4% -> 80.2%, and it beats the
   fixed band on 10 of 10 tickers. Every beta from 0.25 to 0.75 beats
   fixed, so the result doesn't hinge on the exact value picked.

CONCLUSION: ship beta=0.5. The gain is small but consistent, and it is
concentrated where a band matters most — volatile days, where the fixed
band was the most overconfident.
"""

from __future__ import annotations

import numpy as np

# why_it_fails puts the project root on sys.path, so app/stockkit resolve.
from why_it_fails import LOOKBACK, MIN_TRAIN, TICKERS, YEARS, features_for, load_series

from app.predictor import RidgeRegressor

# --- Configuration -----------------------------------------------------------

ALPHA = 3.0                  # the shipped penalty
LEVEL = 0.80
QUANTILES = (10.0, 90.0)
RESIDUAL_WINDOW = 110        # errors the app's backtest yields from 1y of bars
WARMUP = 60                  # errors needed before the first band
EWMA_LAMBDA = 0.94           # RiskMetrics' daily decay
BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)


# --- Volatility estimates known at each decision day -------------------------

def daily_returns(bars: list) -> np.ndarray:
    closes = np.array([bar.model_close for bar in bars])
    returns = np.zeros(closes.size)
    returns[1:] = closes[1:] / closes[:-1] - 1.0
    return returns


def rolling_vol(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """Std of the last `window` returns, ending at (and including) each day."""
    out = np.full(returns.size, np.nan)
    for i in range(window, returns.size):
        out[i] = returns[i - window + 1: i + 1].std()
    return out


def ewma_vol(returns: np.ndarray, decay: float = EWMA_LAMBDA) -> np.ndarray:
    """Exponentially weighted volatility, updated with each day's return."""
    out = np.empty(returns.size)
    variance = float(np.var(returns[1:21]))      # seed on the first month
    for i in range(returns.size):
        variance = decay * variance + (1 - decay) * returns[i] ** 2
        out[i] = np.sqrt(variance)
    return out


# --- Walk-forward bands -------------------------------------------------------

def walk_forward_bands(bars: list) -> dict | None:
    """One-step forecasts with a band per beta, scaled by EWMA volatility.

    Returns arrays over the scored steps: actual, tercile (calm/middle/wild
    by 20-day volatility within this ticker), and (low, high) per beta.
    """
    features = features_for(bars)
    x, y = features.x, features.y
    n = x.shape[0]
    if n - MIN_TRAIN < WARMUP + 50:
        return None

    returns = daily_returns(bars)
    # Row j's decision day is bar LOOKBACK+j, so its volatility is the one
    # computed through that day's close — known when the forecast is made.
    decision = np.arange(LOOKBACK, LOOKBACK + n)
    sigma = ewma_vol(returns)[decision]
    regime = rolling_vol(returns)[decision]

    predictions = np.empty(n)
    model = RidgeRegressor(alpha=ALPHA)
    for i in range(MIN_TRAIN, n):
        model.fit(x[:i], y[:i])
        predictions[i] = model.predict(x[i])[0]

    steps = np.arange(MIN_TRAIN + WARMUP, n)
    bands = {beta: (np.empty(steps.size), np.empty(steps.size)) for beta in BETAS}
    for k, i in enumerate(steps):
        past = np.arange(max(MIN_TRAIN, i - RESIDUAL_WINDOW), i)
        errors = predictions[past] - y[past]              # predicted - actual
        for beta in BETAS:
            scale = sigma ** beta
            z_lo, z_hi = np.percentile(errors / scale[past], QUANTILES)
            bands[beta][0][k] = predictions[i] - z_hi * scale[i]
            bands[beta][1][k] = predictions[i] - z_lo * scale[i]

    ranks = regime[steps].argsort().argsort()
    return {"actual": y[steps], "tercile": ranks * 3 // ranks.size, "bands": bands}


def score(low: np.ndarray, high: np.ndarray, actual: np.ndarray, tercile: np.ndarray):
    inside = (actual >= low) & (actual <= high)
    width = high - low
    miss = np.where(actual < low, low - actual, 0.0) + np.where(actual > high, actual - high, 0.0)
    winkler = width + (2.0 / (1.0 - LEVEL)) * miss
    return {
        "coverage": inside.mean(),
        "by_vol": [inside[tercile == t].mean() for t in range(3)],
        "width": width.mean(),
        "winkler": winkler.mean(),
    }


def main() -> int:
    print(f"Fetching {YEARS} of daily bars for {len(TICKERS)} tickers...\n")
    series = load_series()
    if not series:
        print("No data available — cannot run the comparison.")
        return 1

    results = [r for r in (walk_forward_bands(bars) for bars in series.values()) if r]
    actual = np.concatenate([r["actual"] for r in results])
    tercile = np.concatenate([r["tercile"] for r in results])

    def per_ticker(beta: float) -> list[float]:
        return [score(*r["bands"][beta], r["actual"], r["tercile"])["winkler"] for r in results]

    fixed_scores = per_ticker(0.0)

    print("=" * 78)
    print(f"{LEVEL:.0%} NEXT-DAY BANDS — band scale = ewma_sigma ** beta "
          f"({actual.size} out-of-sample days)")
    print("=" * 78)
    print(f"{'beta':<8}{'coverage':>10}{'calm':>8}{'middle':>8}{'wild':>8}"
          f"{'width':>9}{'score':>9}{'wins':>8}")
    print("-" * 78)
    for beta in BETAS:
        low = np.concatenate([r["bands"][beta][0] for r in results])
        high = np.concatenate([r["bands"][beta][1] for r in results])
        s = score(low, high, actual, tercile)
        calm, middle, wild = s["by_vol"]
        wins = sum(b < f for b, f in zip(per_ticker(beta), fixed_scores))
        label = f"{beta:g}" + (" (fixed)" if beta == 0 else "")
        print(f"{label:<8}{s['coverage']:>10.1%}{calm:>8.1%}{middle:>8.1%}{wild:>8.1%}"
              f"{s['width'] * 100:>8.2f}%{s['winkler'] * 100:>9.3f}"
              f"{'' if beta == 0 else f'{wins}/{len(results)}':>8}")

    print("\n  calm/middle/wild = coverage on each third of days, ranked by")
    print("  20-day volatility within each ticker. Target is 80% in every column.")
    print("  score = Winkler interval score (% of price); lower is better.")
    print("  wins  = tickers where this beta scores better than the fixed band.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
