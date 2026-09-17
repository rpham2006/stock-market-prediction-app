r"""
Phase 0: Establish baselines before building any model.

The question this answers: if a model later scores 56% directional accuracy,
is that good? You cannot know without the numbers below.

The most important one is the BASE RATE -- the share of periods where the
forward return was positive. That is exactly the accuracy of a "model" that
ignores every input and always says UP. Equities drift upward, so this is
meaningfully above 50%, and any real model has to clear it to be worth having.

Run:  .venv\Scripts\python.exe research\baselines.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# --- Configuration -----------------------------------------------------------

HORIZON_DAYS = 5          # predict direction of the 5-trading-day forward return
YEARS_OF_HISTORY = 10
CACHE = Path(__file__).resolve().parent.parent / "data" / "prices.csv"

UNIVERSE = [
    "SPY",                                          # market benchmark
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",        # tech
    "META", "AVGO", "ORCL",
    "JPM", "BAC", "V",                              # financials
    "JNJ", "UNH", "LLY",                            # healthcare
    "XOM", "CVX",                                   # energy
    "WMT", "PG", "KO",                              # consumer staples
    "CAT", "HON",                                   # industrials
]


# --- Data --------------------------------------------------------------------

def load_prices() -> pd.DataFrame:
    """Return a (date x ticker) frame of adjusted closes. Cached to disk."""
    if CACHE.exists():
        print(f"Loading cached prices from {CACHE.relative_to(CACHE.parent.parent)}")
        return pd.read_csv(CACHE, index_col=0, parse_dates=True)

    print(f"Downloading {YEARS_OF_HISTORY}y of daily data for "
          f"{len(UNIVERSE)} tickers from Yahoo Finance...")
    raw = yf.download(
        UNIVERSE,
        period=f"{YEARS_OF_HISTORY}y",
        interval="1d",
        auto_adjust=True,     # adjusts for splits and dividends
        progress=False,
        threads=True,
    )
    if raw is None or raw.empty:
        sys.exit("ERROR: download returned no data. Check your connection.")

    # yfinance returns MultiIndex columns (Field, Ticker) for multiple symbols
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close = close.dropna(how="all").sort_index()

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    close.to_csv(CACHE)
    print(f"Cached to {CACHE.relative_to(CACHE.parent.parent)} "
          f"(delete this file to re-download)")
    return close


# --- Baselines ---------------------------------------------------------------

def evaluate_ticker(prices: pd.Series, horizon: int) -> dict | None:
    """Compute baseline accuracies for a single ticker's price series."""
    px = prices.dropna()
    if len(px) < 300:
        return None

    # TARGET: did the price rise over the NEXT `horizon` trading days?
    # shift(-horizon) looks FORWARD. This is the only place we may look forward.
    forward_return = px.shift(-horizon) / px - 1.0
    target_up = (forward_return > 0).astype(float)

    # FEATURES: only backward-looking information, known at prediction time.
    trailing_5d = px.pct_change(5)
    sma_50 = px.rolling(50).mean()
    above_sma50 = (px > sma_50).astype(float)
    momentum_up = (trailing_5d > 0).astype(float)

    # Align and drop rows where the target is unknown (last `horizon` rows)
    df = pd.DataFrame({
        "target": target_up,
        "forward_return": forward_return,
        "momentum_up": momentum_up,
        "reversion_up": 1.0 - momentum_up,
        "above_sma50": above_sma50,
    }).dropna()

    if df.empty:
        return None

    n = len(df)
    y = df["target"]

    return {
        "n": n,
        # A model that always says UP is right exactly base_rate of the time.
        "base_rate": y.mean(),
        "momentum": (df["momentum_up"] == y).mean(),
        "reversion": (df["reversion_up"] == y).mean(),
        "sma50": (df["above_sma50"] == y).mean(),
        "mean_fwd_return": df["forward_return"].mean(),
        "volatility": df["forward_return"].std(),
    }


def main() -> None:
    close = load_prices()
    horizon = HORIZON_DAYS

    results = {}
    for ticker in close.columns:
        r = evaluate_ticker(close[ticker], horizon)
        if r:
            results[ticker] = r

    if not results:
        sys.exit("ERROR: no ticker had enough history to evaluate.")

    table = pd.DataFrame(results).T.sort_values("base_rate", ascending=False)

    span = f"{close.index.min():%Y-%m-%d} to {close.index.max():%Y-%m-%d}"
    print()
    print("=" * 74)
    print(f"  BASELINES — {horizon}-day forward direction")
    print(f"  {len(results)} tickers | {span}")
    print("=" * 74)
    print()
    print("  Accuracy of each naive strategy, per ticker:")
    print()

    header = f"  {'':<7}{'obs':>7}{'ALWAYS UP':>12}{'momentum':>11}{'reversion':>11}{'>SMA50':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for tk, row in table.iterrows():
        print(f"  {tk:<7}{int(row['n']):>7}"
              f"{row['base_rate']:>11.1%}"
              f"{row['momentum']:>11.1%}"
              f"{row['reversion']:>11.1%}"
              f"{row['sma50']:>9.1%}")

    print()
    print("=" * 74)
    print("  THE NUMBER THAT MATTERS")
    print("=" * 74)

    avg_base = table["base_rate"].mean()
    best_naive = max(
        ("always-up", avg_base),
        ("momentum", table["momentum"].mean()),
        ("mean-reversion", table["reversion"].mean()),
        ("price > SMA50", table["sma50"].mean()),
        key=lambda kv: kv[1],
    )

    print(f"""
  Average base rate (always predict UP):   {avg_base:.1%}
  Best naive strategy:                     {best_naive[0]} at {best_naive[1]:.1%}

  Range of base rates across tickers:      {table['base_rate'].min():.1%} to {table['base_rate'].max():.1%}
  SPY (whole-market) base rate:            {table.loc['SPY', 'base_rate']:.1%}   <- the market itself
""")

    bar = best_naive[1]
    print(f"  ==> Any model you build must beat {bar:.1%} on out-of-sample,")
    print( "      walk-forward data to have earned its existence.")
    print()
    print(f"      If your first model scores 55-58%, that is a genuinely good result.")
    print(f"      If it scores above 70%, you have a data leak. Go find it.")
    print()


if __name__ == "__main__":
    main()
