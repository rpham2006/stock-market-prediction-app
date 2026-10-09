r"""
Ranking stocks against each other — the one framing the diagnostics
haven't ruled out.

Every model so far asked "will this stock go up tomorrow?". Most of that
answer is the market's move, which nothing here can predict. This asks a
different question: across a universe of large caps, which will do better
than the others over the next week? Market-wide moves cancel out by
construction, the drift that makes "always up" hard to beat is gone, and
each week supplies ~55 comparisons instead of one.

Factors (each known at the decision day's close, ranked across stocks):
  rev_5       minus the 5-day return       short-term reversal
  mom_21      21-day return                1-month momentum
  mom_12_1    return from t-252 to t-21    classic 12-1 momentum
  low_vol     minus 20-day volatility      the low-volatility anomaly
  vol_surge   5-day / 60-day avg volume    attention / news flow
  dist_high   close / 52-week high - 1     anchoring on the high

Target: the next 5-session return minus that week's cross-sectional mean.
Decision days are every 5th session, so target windows never overlap and
the t-statistics below are not inflated by reusing the same days.

Scored two ways, both with no hindsight:
  - each factor alone (no fitting at all)
  - a pooled ridge on all six, walk-forward: retrained every 4 weeks on
    weeks whose outcomes were already known, after a 1-year warm-up

Metrics, per week then averaged:
  IC        Spearman rank correlation between score and outcome. 0 = none;
            0.02-0.05 is what published factors typically show.
  t         mean IC / its standard error. Below ~2 is not evidence.
  spread    top-fifth minus bottom-fifth return, % per week.
  net       spread after costs: turnover x 4 sides x COST_BPS.

CAVEAT — survivorship bias. The universe is today's large caps, chosen
knowing they survived and grew. That flatters momentum in particular: a
stock that ran up and then collapsed out of the index isn't here. Treat
positive momentum results as an upper bound.

Run:  .venv\Scripts\python.exe research\cross_section.py
      (fetches 10y of daily bars for ~55 tickers; takes ~30s)

------------------------------------------------------------------------
FINDINGS (55 stocks, 10y, 399 non-overlapping out-of-sample weeks)
------------------------------------------------------------------------

NOTHING CLEARS |t| > 2.
  mom_12_1   IC +0.020  t +1.35  both halves positive   net +0.12%/wk
  low_vol    IC -0.025  t -1.79  sign flips by half     (high vol won)
  ridge (6)  IC +0.006  t +0.45  halves -0.031 / +0.042
  random     IC +0.008  t +1.16  <- the noise floor for this sample

1. 12-1 MOMENTUM is the only signal pointing the same way in both halves,
   in line with the published effect — but t=1.35 isn't evidence, and the
   survivorship bias in this universe flatters exactly this factor.
   Re-run at its standard monthly horizon (95 months): IC +0.023, t +0.79.

2. LOW VOLATILITY RAN BACKWARDS: volatile stocks beat calm ones, almost
   all in the second half. That's the decade's mega-cap tech rally in a
   universe picked for having survived it — a regime, not a rule.

3. COMBINING THE SIX DIDN'T HELP. The ridge's IC flips sign between the
   halves; it learned whichever factor had worked lately.

4. A RANDOM RANKING SCORED t=1.16. That is how large a t this sample
   throws up by chance — the momentum result sits barely above it.

CONCLUSION: with free daily price data on ~55 survivorship-biased large
caps, there is no ranking edge worth shipping. A real test of momentum
needs a point-in-time universe (constituents as they were, delisted names
included) — a paid-data problem, not a modelling one.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np

# why_it_fails puts the project root on sys.path, so app/stockkit resolve.
import why_it_fails  # noqa: F401
from app.predictor import RidgeRegressor
from stockkit import MarketDataError, YahooFinanceClient

# --- Configuration -----------------------------------------------------------

UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "ORCL", "ADBE",
    "CRM", "CSCO", "INTC", "AMD", "QCOM", "TXN", "IBM",               # tech
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP", "BLK",         # financials
    "JNJ", "UNH", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT", "AMGN",  # health
    "XOM", "CVX", "COP", "SLB",                                       # energy
    "WMT", "PG", "KO", "PEP", "COST", "HD", "MCD", "NKE",             # consumer
    "CAT", "HON", "GE", "UPS", "BA", "LMT",                           # industrials
    "NEE", "DUK", "SO",                                               # utilities
]
PERIOD = "10y"
HORIZON = 5                  # sessions per holding period
YEAR = 252
WARMUP_WEEKS = 52            # weeks of history before the first model score
REFIT_EVERY = 4              # weeks between refits
ALPHA = 1_000.0
QUANTILE = 0.2               # top and bottom fifth
COST_BPS = 5.0               # per side, large-cap commission + half-spread

FACTORS = ("rev_5", "mom_21", "mom_12_1", "low_vol", "vol_surge", "dist_high")


# --- Data --------------------------------------------------------------------

def load_panel() -> tuple[list[date], list[str], np.ndarray, np.ndarray]:
    """(days, symbols, closes, volumes) as day x symbol arrays, NaN if absent.

    Today's bar is dropped: during market hours it is a partial session,
    not a close.
    """
    client = YahooFinanceClient(timeout_seconds=25)
    history: dict[str, dict] = {}
    for symbol in UNIVERSE:
        try:
            bars = client.get_history(symbol, period=PERIOD)
        except MarketDataError as exc:
            print(f"  {symbol}: skipped — {exc}")
            continue
        history[symbol] = {
            bar.day: (bar.model_close, bar.volume) for bar in bars if bar.day < date.today()
        }

    symbols = list(history)
    days = sorted({day for rows in history.values() for day in rows})
    closes = np.full((len(days), len(symbols)), np.nan)
    volumes = np.full((len(days), len(symbols)), np.nan)
    index = {day: i for i, day in enumerate(days)}
    for j, symbol in enumerate(symbols):
        for day, (close, volume) in history[symbol].items():
            closes[index[day], j] = close
            volumes[index[day], j] = volume

    # Keep only sessions most of the universe traded.
    enough = np.sum(~np.isnan(closes), axis=1) >= len(symbols) // 2
    return [d for d, keep in zip(days, enough) if keep], symbols, closes[enough], volumes[enough]


# --- Factors -----------------------------------------------------------------

def factors_at(t: int, closes: np.ndarray, volumes: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """(n_symbols, n_factors) raw factor values using rows <= t only."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rev_5 = -(closes[t] / closes[t - 5] - 1.0)
        mom_21 = closes[t] / closes[t - 21] - 1.0
        mom_12_1 = closes[t - 21] / closes[t - YEAR] - 1.0
        low_vol = -np.std(returns[t - 19: t + 1], axis=0)
        vol_surge = volumes[t - 4: t + 1].mean(axis=0) / volumes[t - 59: t + 1].mean(axis=0)
        dist_high = closes[t] / np.max(closes[t - YEAR + 1: t + 1], axis=0) - 1.0
    return np.column_stack([rev_5, mom_21, mom_12_1, low_vol, vol_surge, dist_high])


def rank_centre(values: np.ndarray) -> np.ndarray:
    """Cross-sectional ranks mapped to -0.5..0.5. Robust to outliers."""
    ranks = values.argsort(axis=0).argsort(axis=0).astype(float)
    return ranks / (values.shape[0] - 1) - 0.5


def build_weeks(closes: np.ndarray, volumes: np.ndarray):
    """One (x, y) cross-section per non-overlapping decision day."""
    returns = np.full(closes.shape, np.nan)
    returns[1:] = closes[1:] / closes[:-1] - 1.0

    weeks = []
    for t in range(YEAR, closes.shape[0] - HORIZON, HORIZON):
        raw = factors_at(t, closes, volumes, returns)
        forward = closes[t + HORIZON] / closes[t] - 1.0
        valid = np.all(np.isfinite(raw), axis=1) & np.isfinite(forward)
        if valid.sum() < 20:
            continue
        x = rank_centre(raw[valid])
        y = forward[valid] - forward[valid].mean()     # beat the average stock?
        weeks.append((x, y, np.flatnonzero(valid)))
    return weeks


# --- Scoring -----------------------------------------------------------------

def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def evaluate(scores: list[np.ndarray], weeks: list) -> dict[str, float]:
    """IC, long-short spread and turnover across the scored weeks."""
    ics, spreads, turnover = [], [], []
    previous_top = previous_bottom = None
    for score, (_, y, members) in zip(scores, weeks):
        ics.append(spearman(score, y))
        k = max(1, int(len(y) * QUANTILE))
        order = score.argsort()
        top, bottom = order[-k:], order[:k]
        spreads.append(y[top].mean() - y[bottom].mean())

        top_names, bottom_names = set(members[top]), set(members[bottom])
        if previous_top is not None:
            turnover.append(
                (len(top_names - previous_top) + len(bottom_names - previous_bottom)) / (2 * k)
            )
        previous_top, previous_bottom = top_names, bottom_names

    ics, spreads = np.array(ics), np.array(spreads)
    half = ics.size // 2
    mean_turnover = float(np.mean(turnover)) if turnover else 0.0
    cost = mean_turnover * 4 * COST_BPS / 10_000
    return {
        "ic": float(ics.mean()),
        "t": float(ics.mean() / ics.std(ddof=1) * math.sqrt(ics.size)),
        "ic_first": float(ics[:half].mean()),
        "ic_second": float(ics[half:].mean()),
        "hit": float(np.mean(ics > 0)),
        "spread": float(spreads.mean()),
        "net": float(spreads.mean() - cost),
        "turnover": mean_turnover,
        "weeks": int(ics.size),
    }


def walk_forward_model(weeks: list) -> list[np.ndarray]:
    """Pooled ridge scores for every week after the warm-up."""
    scores = []
    model = None
    for k in range(WARMUP_WEEKS, len(weeks)):
        if model is None or (k - WARMUP_WEEKS) % REFIT_EVERY == 0:
            # Week k-1's outcome ends on week k's decision day, so it is
            # known by then. Nothing later is touched.
            x = np.vstack([w[0] for w in weeks[:k]])
            y = np.concatenate([w[1] for w in weeks[:k]])
            model = RidgeRegressor(alpha=ALPHA).fit(x, y)
        scores.append(model.predict(weeks[k][0]))
    return scores


def main() -> int:
    print(f"Fetching {PERIOD} of daily bars for {len(UNIVERSE)} tickers...\n")
    days, symbols, closes, volumes = load_panel()
    weeks = build_weeks(closes, volumes)
    if len(weeks) <= WARMUP_WEEKS + 20:
        print("Not enough history to score anything.")
        return 1

    test = weeks[WARMUP_WEEKS:]
    results = {name: evaluate([w[0][:, f] for w in test], test) for f, name in enumerate(FACTORS)}
    results["ridge (all 6)"] = evaluate(walk_forward_model(weeks), test)
    results["random"] = evaluate(
        [np.random.default_rng(k).random(len(w[1])) for k, w in enumerate(test)], test
    )

    print("=" * 92)
    print(f"WEEKLY CROSS-SECTIONAL RANKING — {len(symbols)} stocks, "
          f"{results['random']['weeks']} out-of-sample weeks")
    print("=" * 92)
    print(f"{'signal':<16}{'IC':>8}{'t':>7}{'1st half':>10}{'2nd half':>10}{'IC>0':>7}"
          f"{'spread':>9}{'turnover':>10}{'net':>9}")
    print("-" * 92)
    for name, r in results.items():
        print(f"{name:<16}{r['ic']:>+8.3f}{r['t']:>+7.2f}{r['ic_first']:>+10.3f}"
              f"{r['ic_second']:>+10.3f}{r['hit']:>7.0%}{r['spread'] * 100:>+8.2f}%"
              f"{r['turnover']:>10.0%}{r['net'] * 100:>+8.2f}%")

    print(f"\n  spread/net = top fifth minus bottom fifth, % per week; net assumes")
    print(f"  {COST_BPS:g} bps per side. Multiply by ~52 for a rough annual figure.")
    print("  A signal is only interesting if |t| > 2 AND both halves agree in sign.")
    print("  Survivorship bias flatters momentum — see the module docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
