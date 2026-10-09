r"""
Does market context help the next-day model?

Every one of the 13 features is a transformation of the stock's own price.
`why_it_fails.py` showed they carry about 7 features' worth of information,
and none of it predicts tomorrow. This adds the first inputs that come from
outside the stock:

  spy_ret_1d     the market's move today
  spy_ret_5d     the market's move this week
  rel_5d         the stock's 5-day return minus SPY's (relative strength)
  vix_level      the VIX, /100 — the options market's own fear gauge
  vix_change_1d  today's change in the VIX

Same tickers, horizon, walk-forward loop and scoring as `experiments.py`,
so the tables compare line for line. Each feature set runs at the shipped
alpha (3) and a heavy one (1000), since `experiments.py` found 3 too weak.

Run:  .venv\Scripts\python.exe research\market_context.py
      (fetches 5y of daily bars for 10 tickers, SPY and ^VIX; ~1 min)

------------------------------------------------------------------------
FINDINGS (10 tickers, 5y, 8,330 out-of-sample next-day predictions)
------------------------------------------------------------------------

MARKET CONTEXT MAKES IT WORSE, NOT BETTER.
  price only   @3:    edge -3.2, MAE ratio 1.009, R2 -0.018
  price+market @3:    edge -3.7, MAE ratio 1.015, R2 -0.025
  market only  @1000: edge -2.0, MAE ratio 0.999, R2 -0.002
  mean only:          edge -0.5, MAE ratio 0.998, R2 -0.001
Five more inputs give the fit five more ways to chase noise. At heavy
shrinkage every set converges on "mean only" and none passes it. The VIX
is public and real-time; whatever it says about tomorrow is already in
today's price.
"""

from __future__ import annotations

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
from experiments import print_table
from stockkit import MarketDataError, YahooFinanceClient

# --- Configuration -----------------------------------------------------------

MARKET = "SPY"
FEAR = "^VIX"
ALPHAS = (3.0, 1_000.0)
MARKET_FEATURES = ("spy_ret_1d", "spy_ret_5d", "rel_5d", "vix_level", "vix_change_1d")


# --- Data --------------------------------------------------------------------

def load_closes(symbol: str) -> dict | None:
    """Adjusted closes keyed by day, or None if unavailable."""
    client = YahooFinanceClient(timeout_seconds=25)
    try:
        bars = client.get_history(symbol, period=YEARS)
    except MarketDataError as exc:
        print(f"  {symbol}: unavailable — {exc}")
        return None
    return {bar.day: bar.model_close for bar in bars}


def trailing_return(closes: dict, k: int) -> dict:
    """k-session return ending on each day, using that series' own sessions."""
    days = sorted(closes)
    return {
        days[i]: closes[days[i]] / closes[days[i - k]] - 1.0
        for i in range(k, len(days))
    }


def market_rows(bars: list, spy: dict, vix: dict):
    """Base features, market features and next-day target, aligned by day.

    Every market value is keyed to the decision day itself — its close is
    known when the forecast is made. Rows missing any market value are
    dropped rather than filled, so nothing is invented.
    """
    features = features_for(bars)
    closes = np.array([bar.model_close for bar in bars])

    spy_1d, spy_5d = trailing_return(spy, 1), trailing_return(spy, 5)
    vix_1d = trailing_return(vix, 1)

    base, extra, target = [], [], []
    for j in range(features.y.size):
        i = LOOKBACK + j                      # bar index of the decision day
        day = bars[i].day
        if i < 5 or not all(day in s for s in (spy_1d, spy_5d, vix_1d)):
            continue
        stock_5d = closes[i] / closes[i - 5] - 1.0
        extra.append([
            spy_1d[day],
            spy_5d[day],
            stock_5d - spy_5d[day],
            vix[day] / 100.0,
            vix_1d[day],
        ])
        base.append(features.x[j])
        target.append(features.y[j])

    return np.array(base), np.array(extra), np.array(target)


# --- Walk-forward ------------------------------------------------------------

def walk_forward(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray | None:
    n = x.shape[0]
    if n - MIN_TRAIN < 50:
        return None
    predictions = np.empty(n - MIN_TRAIN)
    model = RidgeRegressor(alpha=alpha)
    for offset, i in enumerate(range(MIN_TRAIN, n)):
        model.fit(x[:i], y[:i])                # strictly the past
        predictions[offset] = model.predict(x[i])[0]
    return predictions


def main() -> int:
    print(f"Fetching {YEARS} of daily bars for {len(TICKERS)} tickers, {MARKET} and {FEAR}...\n")
    series = load_series()
    spy, vix = load_closes(MARKET), load_closes(FEAR)
    if not series or spy is None or vix is None:
        print("Missing data — cannot run the comparison.")
        return 1

    rows = [market_rows(bars, spy, vix) for bars in series.values()]
    feature_sets = {
        "price only": lambda base, extra: base,
        "market only": lambda base, extra: extra,
        "price+market": lambda base, extra: np.hstack([base, extra]),
    }

    scores: dict[str, dict[str, float]] = {}
    for label, build in feature_sets.items():
        for alpha in ALPHAS:
            predictions, actuals = [], []
            for base, extra, y in rows:
                result = walk_forward(build(base, extra), y, alpha)
                if result is not None:
                    predictions.append(result)
                    actuals.append(y[MIN_TRAIN:])
            scores[f"{label} a={alpha:g}"] = pooled_scores(
                np.concatenate(predictions), np.concatenate(actuals)
            )

    # The reference every variant has to beat: the trailing mean return.
    means, actuals = [], []
    for _, _, y in rows:
        if y.size - MIN_TRAIN >= 50:
            means.append(np.array([y[:i].mean() for i in range(MIN_TRAIN, y.size)]))
            actuals.append(y[MIN_TRAIN:])
    scores["mean only"] = pooled_scores(np.concatenate(means), np.concatenate(actuals))

    print_table(f"NEXT-DAY RETURN WITH MARKET CONTEXT ({', '.join(MARKET_FEATURES)})",
                {name.replace(" a=", " @"): s for name, s in scores.items()})
    print("  @N = ridge alpha. A feature set only helps if it beats 'mean only'")
    print("  on edge AND has MAE ratio below 1.000 and OOS R2 above 0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
