# Stock Market Prediction App

A full-stack Python app that finds a stock, stores a year of daily history in a
database, and shows the current quote alongside a next-day price estimate — with
the backtest that tells you whether that estimate is worth anything.

Everything is Python: FastAPI serves both the JSON API and the server-rendered
HTML. No Node, no CDN, no build step.

> **Educational project.** Not investment advice, not a recommendation to buy or
> sell any security. See [Honest expectations](#honest-expectations).

---

## Quick start

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

```bash
.venv/Scripts/python.exe -m uvicorn app.main:app --reload
```

Then open:

| URL | What it is |
|---|---|
| <http://127.0.0.1:8000> | Search page + locally stored tickers |
| <http://127.0.0.1:8000/stock/AAPL> | Full dashboard for one ticker |
| <http://127.0.0.1:8000/docs> | Generated OpenAPI console |

Run the tests:

```bash
.venv/Scripts/python.exe -m pytest tests -q
```

---

## What it does

1. **Find a stock** — search by ticker or company name (`/search?q=microsoft`).
   An exact ticker match jumps straight to the dashboard.
2. **Backdate a year** — on first view it fetches ~253 daily bars (1 year) and
   writes them to the database. Later views read from the database; a page load
   normally costs zero HTTP requests.
3. **Show current information** — live price, day change, day range, 52-week
   range, volume, and an interactive 1-year price chart with 1M/3M/6M/1Y ranges.
4. **Predict the next day** — a ridge-regression model estimates the next
   session's close, shown with an 80% error band, the features that drove it,
   and its out-of-sample scores.

---

## Architecture

```
                    ┌──────────────────────────────┐
  browser  ───────► │  web.py      Jinja2 + SVG    │
  curl/JS  ───────► │  api.py      JSON, /docs     │   HTTP layer
                    └───────────────┬──────────────┘
                                    ▼
                    ┌──────────────────────────────┐
                    │  service.py                  │   when to fetch,
                    │  refresh policy, assembly    │   what to assemble
                    └───────┬───────────────┬──────┘
                            ▼               ▼
              ┌───────────────────┐  ┌──────────────────┐
              │ repository.py     │  │ features.py      │
              │ every SQL query   │  │ predictor.py     │
              └─────────┬─────────┘  └──────────────────┘
                        ▼
              ┌───────────────────┐  ┌──────────────────┐
              │ SQLite / Postgres │  │ stockkit/        │  market data,
              └───────────────────┘  └──────────────────┘  knows nothing
                                                           of this app
```

Each layer calls inward, never outward. That is what makes the model testable
without a database and the database testable without a network.

| File | Responsibility |
|---|---|
| `app/config.py` | Settings, all overridable by environment variable |
| `app/db.py` | Engine + session factory — the only place that knows the dialect |
| `app/models.py` | ORM: `Company`, `DailyBar`, `Prediction` |
| `app/repository.py` | Every SQL query in the app |
| `app/features.py` | 13 technical features, strictly no lookahead |
| `app/predictor.py` | Ridge regression + walk-forward backtest |
| `app/service.py` | Refresh policy, caching, payload assembly |
| `app/api.py` / `app/web.py` | JSON routes / HTML routes |
| `stockkit/` | Market data client (pre-existing, extended) |

---

## Database

Three tables, SQLite by default:

- **`companies`** — one row per ticker, with the latest quote denormalised onto
  it and two cache timestamps (`quote_updated_at`, `bars_updated_at`).
- **`daily_bars`** — OHLCV plus adjusted close, with a `UNIQUE(symbol, day)`
  constraint. That constraint is what makes ingestion idempotent: re-running an
  overlapping window inserts nothing and only rewrites bars whose values
  actually changed (which happens after a split).
- **`predictions`** — each forecast with its backtest scores and drivers, keyed
  on `(symbol, target_day, model_version)`. Storing them means you can later
  score the model on what it *actually said at the time*.

Switch to Postgres with one environment variable — no code change:

```bash
DATABASE_URL=postgresql+psycopg://user:pw@host/dbname
```

### Freshness

| Data | Default TTL | Env var |
|---|---|---|
| Quote | 5 minutes | `QUOTE_TTL_MINUTES` |
| Daily bars | 6 hours | `BAR_TTL_MINUTES` |
| Prediction | 1 hour, or until a new close | `PREDICTION_TTL_MINUTES` |

If the provider is unreachable but rows are stored, the app serves them and
labels the page stale rather than erroring out.

---

## The model

**Features (13).** Trailing returns over 1/2/3/5/10 sessions, close relative to
its 5/10/20-day moving averages, 10- and 20-day realised volatility, RSI(14),
intraday range position, and volume versus its 20-day average. All computed from
the **adjusted** close, so a split doesn't look like a −75% day.

**Estimator.** Ridge regression, solved in closed form, features standardised
before the penalty is applied. Chosen deliberately: with ~230 usable rows and a
brutal signal-to-noise ratio, a high-capacity model memorises noise. A linear
model with L2 shrinkage is the honest choice at this sample size — and you can
read its coefficients, which is why the dashboard can show what drove a forecast.

**Validation.** Walk-forward, never a random split. For each held-out day the
model is refit on *only* the days before it, then predicts one step ahead. A
shuffled split would let the model train on Thursday to predict Wednesday, and
every score it produced would be a lie.

**Scored against two baselines:**

| Baseline | Question it answers |
|---|---|
| No-change ("tomorrow = today") | Is the model more accurate than a random walk? |
| Always-up | Does it call direction better than exploiting upward drift? |

**The 80% band** is the 10th–90th percentile of the backtest's own errors. It is
an empirical claim — "80% of the time the model was this wrong" — not a
theoretical confidence interval, and assumes nothing about the distribution.

---

## Accuracy tracking

`scripts/daily_audit.py` runs on a schedule and records two *different*
measurements. Keeping them apart is the whole point:

| | **Scoring** | **Auditing** |
|---|---|---|
| What it does | Compares each stored prediction to the close that actually happened | Re-runs the walk-forward backtest on today's data |
| Hindsight? | **None** — the forecast was written down first | Yes — it only says what the model would claim *now* |
| Stored in | `predictions.actual_close`, `.error_pct`, `.direction_correct` | `accuracy_runs` |
| Good for | The real track record | Spotting model drift |

Scoring is the one that matters. Auditing daily is *nearly redundant* on its own
— the 253-bar window shifts by one day between runs, so consecutive audits are
almost identical. It earns its place as a drift chart over months, not as a
daily verdict.

Run it by hand:

```bash
.venv\Scripts\python.exe -m scripts.daily_audit
```

Output goes to the console and to `logs/daily_audit.log`. Exit code 1 if every
symbol failed, so a scheduler can alert on it.

### Schedule it (Windows)

```bash
schtasks /create /tn "StockForecast daily audit" /tr "'C:\Users\Ryan\Stock market prediction app\run_daily_audit.bat'" /sc daily /st 18:30
```

18:30 local is after the US close, so the day's final bar exists. The task only
runs while your machine is on; add `/ru SYSTEM` to run it logged-out, or move
the job to a small VPS if you want unbroken daily coverage.

Inspect or remove it:

```bash
schtasks /query /tn "StockForecast daily audit"
```

```bash
schtasks /delete /tn "StockForecast daily audit" /f
```

### Reading the results

The track record needs **~100 scored predictions** before direction is
distinguishable from a coin flip — roughly 5 months of daily runs on one ticker,
or a few weeks across 22. Until then `is_meaningful` is `false`, and the API
says so rather than letting you over-read a small sample.

Measured across 22 tickers (2,464 held-out predictions): the model beat the
no-change baseline on price for **2 of 22**, pooled directional accuracy was
**51.1%** (z = 1.06, not significant). It has no demonstrated edge — see below.

## Honest expectations

Running this on real tickers, the model typically **loses to the no-change
baseline on price accuracy** and sometimes beats the always-up baseline on
direction. That is the expected result, and the dashboard says so in plain
English rather than hiding it behind a confident-looking number.

Public equity markets are close to efficient at a one-day horizon. A model built
from public price history should not be expected to beat them. The value here is
the pipeline, the honest measurement, and the transparency — not alpha.

Known limitations, all visible in the UI:

- `next_trading_day` skips weekends but **not market holidays**.
- Price history only — no fundamentals, no news, no order-flow.
- `create_all` on boot, not migrations. Adding a table is fine; changing a
  column means switching to Alembic.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Liveness + tracked symbol count |
| `GET` | `/api/search?q=` | Ticker lookup |
| `GET` | `/api/stocks` | Locally stored tickers (no network) |
| `GET` | `/api/stocks/{symbol}` | Quote + history + prediction |
| `GET` | `/api/stocks/{symbol}/history?days=365` | Stored daily bars |
| `GET` | `/api/stocks/{symbol}/prediction` | Forecast only |
| `POST` | `/api/stocks/{symbol}/refresh` | Force refetch and refit |
| `GET` | `/api/accuracy` | Latest audit + live track record |
| `GET` | `/api/accuracy/history?symbol=&days=` | Audit scores over time |
| `GET` | `/api/accuracy/track-record` | Hindsight-free performance |
| `GET` | `/api/accuracy/predictions` | Past predictions vs actual outcomes |

Conventions: `snake_case` keys, ISO 8601 dates, and one error envelope
everywhere:

```json
{ "error": { "code": "symbol_not_found", "message": "No such symbol: 'FAKE'" } }
```

---

## Tests

72 tests, no network and no real database. The market data client is behind an
interface, so a fake implementation drops straight into the FastAPI dependency.

The one that matters most is `test_no_lookahead_bias`: it rewrites the tail of a
price series and asserts every earlier feature row is byte-identical. If a
feature ever leaks future data, that test fails.

---

## Roadmap

See `ROADMAP.md` for the full plan. Nearest next steps: LLM explanations of a
forecast, a news feed with sentiment, auth + watchlists, and an exchange
calendar to replace the naive next-weekday logic.
