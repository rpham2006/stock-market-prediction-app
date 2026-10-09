r"""
Cheap fixes to the existing model — do any of them clear the baselines?

`why_it_fails.py` concluded the binding constraint is information, not
modelling. Before accepting that, this script tries the three changes that
cost nothing in new data, scored exactly the same way (same tickers, same
walk-forward loop, same `pooled_scores`):

  A. THE PENALTY. Features are standardised before the L2 penalty, so the
     diagonal of X'X is ~n (hundreds of rows). alpha=3 against that is
     almost no shrinkage at all — the "ridge" is close to plain OLS. Sweep
     alpha, and also pick it walk-forward: at each step use whichever alpha
     has had the lowest error on the predictions already scored.

  B. THE TARGET. Predict the return in excess of SPY rather than the raw
     return. That strips out the market-wide move — most of the noise, and
     all of the drift that makes "always up" hard to beat.

  C. THE TAILS. Winsorise the training target at its own 1st/99th
     percentiles, so one earnings gap can't dominate a least-squares fit.
     Only training targets are clipped; scoring is on the real returns.

Every variant must be read against its own baselines: "mean only" (predict
the trailing average return — what infinite shrinkage converges to) and
the naive no-change forecast behind the MAE ratio.

A caution on the standard errors: pooled rows from different tickers on the
same day are correlated, so the true uncertainty is larger than `se` shows.
Treat any edge smaller than about 2 se as zero.

Run:  .venv\Scripts\python.exe research\experiments.py
      (fetches 5y of daily bars for 10 tickers plus SPY; takes ~30s)

------------------------------------------------------------------------
FINDINGS (10 tickers, 5y, 8,320 pooled out-of-sample predictions, 1d)
------------------------------------------------------------------------

A. MORE SHRINKAGE IS STRICTLY BETTER — all the way to "mean only".
   alpha=3 (shipped): edge -3.2, MAE ratio 1.009, R2 -0.018.
   alpha=3000:        edge -1.2, MAE ratio 0.998, R2 -0.002.
   mean only:         edge -0.5, MAE ratio 0.998, R2 -0.001.
   Every step up in alpha moves the model toward predicting the trailing
   average, and every step improves it. Walk-forward alpha selection lands
   in between (-1.3, 1.000). The penalty was too weak, but the fix only
   stops the model adding error; it never finds signal.

B. EXCESS RETURN REMOVES THE DRIFT, NOT THE NOISE. The baseline falls to
   49.8% and the edge turns slightly positive (+1.0 at alpha=1000), but
   that is 2 understated standard errors, the MAE ratio stays above 1.000
   and R2 stays negative. Not a usable edge.

C. WINSORISING CHANGES NOTHING MEANINGFUL — a few tenths of a point
   either way, inside the noise.

CONCLUSION: the cheap fixes confirm `why_it_fails.py`. The best price-only
next-day forecast is "the stock's average daily return"; ridge on these
features can at most be shrunk into agreeing with that. Raising the
shipped alpha to ~1,000-3,000 is still worth doing, because it stops the
forecast being worse than no-change. Real gains need the volatility
forecast or new inputs (ROADMAP-level work), not model tuning.
"""

from __future__ import annotations

import math

import numpy as np

# why_it_fails puts the project root on sys.path, so app/stockkit resolve.
from why_it_fails import (
    LOOKBACK,
    MIN_TRAIN,
    TICKERS,
    YEARS,
    RidgeRegressor,
    features_for,
    load_series,
    pooled_scores,
)
from stockkit import MarketDataError, YahooFinanceClient

# --- Configuration -----------------------------------------------------------

BENCHMARK = "SPY"
ALPHAS = [3.0, 30.0, 100.0, 300.0, 1_000.0, 3_000.0, 10_000.0]
CURRENT_ALPHA = 3.0          # what app/predictor.py ships with
SELECTION_WARMUP = 20        # scored predictions needed before picking alpha
CLIP_QUANTILES = (1.0, 99.0)


# --- Data --------------------------------------------------------------------

def load_benchmark() -> dict | None:
    """Benchmark adjusted closes keyed by day, or None if unavailable."""
    client = YahooFinanceClient(timeout_seconds=25)
    try:
        bars = client.get_history(BENCHMARK, period=YEARS)
    except MarketDataError as exc:
        print(f"  {BENCHMARK}: unavailable — {exc}")
        return None
    return {bar.day: bar.model_close for bar in bars}


def one_day_rows(bars: list, benchmark: dict | None):
    """Feature rows with raw and excess next-day returns.

    Row j is decision day LOOKBACK+j and its target is the return to the
    following session — the same alignment as `build_features`. Rows where
    the benchmark is missing either day get NaN excess returns.
    """
    features = features_for(bars)
    x, y = features.x, features.y

    excess = np.full(y.shape, np.nan)
    if benchmark is not None:
        for j in range(y.size):
            today = bars[LOOKBACK + j].day
            tomorrow = bars[LOOKBACK + j + 1].day
            if today in benchmark and tomorrow in benchmark:
                market = benchmark[tomorrow] / benchmark[today] - 1.0
                excess[j] = y[j] - market
    return x, y, excess


# --- Walk-forward over many variants at once ---------------------------------

def walk_forward_variants(
    x: np.ndarray,
    y: np.ndarray,
    min_train: int,
    alphas: list[float],
    clip: bool = False,
) -> dict[str, np.ndarray] | None:
    """Expanding-window one-step-ahead predictions for every alpha.

    Returns {variant name: predictions} plus "actual". Includes:
      - "alpha=<a>" for each fixed alpha
      - "adaptive": at step t, the alpha with the lowest MAE over steps
        before t. Step t-1's target is the return into day t, which is
        known at the close of day t, so this uses no future information.
      - "mean only": the trailing mean of the training target.
    """
    n = x.shape[0]
    if n - min_train < 50:
        return None

    n_test = n - min_train
    fixed = np.empty((len(alphas), n_test))
    adaptive = np.empty(n_test)
    mean_only = np.empty(n_test)
    abs_errors = np.zeros(len(alphas))
    default = alphas.index(CURRENT_ALPHA) if CURRENT_ALPHA in alphas else 0

    for offset, i in enumerate(range(min_train, n)):
        y_train = y[:i]                    # strictly the past
        if clip:
            low, high = np.percentile(y_train, CLIP_QUANTILES)
            y_train = np.clip(y_train, low, high)

        for k, alpha in enumerate(alphas):
            model = RidgeRegressor(alpha=alpha).fit(x[:i], y_train)
            fixed[k, offset] = model.predict(x[i])[0]

        best = int(np.argmin(abs_errors)) if offset >= SELECTION_WARMUP else default
        adaptive[offset] = fixed[best, offset]
        mean_only[offset] = y_train.mean()

        # Only now is this step's outcome allowed into the selection score.
        abs_errors += np.abs(fixed[:, offset] - y[i])

    out = {f"alpha={a:g}": fixed[k] for k, a in enumerate(alphas)}
    out["adaptive"] = adaptive
    out["mean only"] = mean_only
    out["actual"] = y[min_train:]
    return out


def pool(results: list[dict[str, np.ndarray]]) -> dict[str, dict[str, float]]:
    """Concatenate each variant across tickers and score it."""
    actuals = np.concatenate([r["actual"] for r in results])
    names = [name for name in results[0] if name != "actual"]
    return {
        name: pooled_scores(np.concatenate([r[name] for r in results]), actuals)
        for name in names
    }


# --- Reporting ---------------------------------------------------------------

def print_table(title: str, scores: dict[str, dict[str, float]]) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)
    print(f"{'variant':<14}{'dir acc':>9}{'baseline':>10}{'edge':>8}{'se':>6}"
          f"{'MAE ratio':>11}{'OOS R2':>10}{'n':>7}")
    print("-" * 78)
    for name, s in scores.items():
        se = math.sqrt(s["directional"] * (1 - s["directional"]) / s["n"])
        print(f"{name:<14}{s['directional'] * 100:>8.1f}%{s['baseline'] * 100:>9.1f}%"
              f"{s['edge'] * 100:>+8.1f}{se * 100:>6.1f}"
              f"{s['mae_ratio']:>11.3f}{s['r2']:>10.4f}{s['n']:>7}")
    print()


def run_experiment(rows, target: str, clip: bool = False):
    """Walk every ticker forward on one target and pool the scores."""
    results = []
    for x, y_raw, y_excess in rows:
        y = y_raw if target == "raw" else y_excess
        keep = ~np.isnan(y)
        result = walk_forward_variants(x[keep], y[keep], MIN_TRAIN, ALPHAS, clip=clip)
        if result is not None:
            results.append(result)
    return pool(results) if results else None


def main() -> int:
    print(f"Fetching {YEARS} of daily bars for {len(TICKERS)} tickers + {BENCHMARK}...\n")
    series = load_series()
    benchmark = load_benchmark()
    if not series:
        print("No data available — cannot run the experiments.")
        return 1

    rows = [one_day_rows(bars, benchmark) for bars in series.values()]

    experiments = [
        ("A. RAW NEXT-DAY RETURN — alpha sweep", "raw", False),
        ("C. RAW RETURN, WINSORISED TRAINING TARGET", "raw", True),
    ]
    if benchmark is not None:
        experiments.insert(1, (f"B. RETURN IN EXCESS OF {BENCHMARK}", "excess", False))
    else:
        print(f"Skipping experiment B: no {BENCHMARK} data.\n")

    for title, target, clip in experiments:
        scores = run_experiment(rows, target, clip)
        if scores is None:
            print(f"{title}: not enough history.\n")
            continue
        print_table(title, scores)

    print("Reading the tables:")
    print("  edge      = directional accuracy minus always-guessing-up (for the")
    print("              excess target: always guessing 'beats SPY').")
    print("  MAE ratio = model error / error of predicting zero. Below 1.000 helps.")
    print("  OOS R2    = vs predicting the test-period mean. Below 0 hurts.")
    print("  A variant only counts if it beats 'mean only' AND has MAE ratio < 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
