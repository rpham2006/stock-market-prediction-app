"""
HTTP tests — the JSON API and the rendered pages, end to end.

The app is wired to the in-memory database and the fake data client via
FastAPI's dependency_overrides, so these exercise real routing, real
serialisation and real templates without touching the network.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import service
from app.db import get_session
from app.main import app


@pytest.fixture
def api(session, client) -> TestClient:
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[service.get_client] = lambda: client
    yield TestClient(app)
    app.dependency_overrides.clear()


# ---------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------

def test_health(api):
    response = api.get("/api/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "sqlite"


def test_search(api):
    response = api.get("/api/search", params={"q": "AAPL"})
    assert response.status_code == 200
    assert response.json()[0]["symbol"] == "AAPL"


def test_search_requires_a_query(api):
    assert api.get("/api/search").status_code == 422


def test_get_stock_returns_quote_history_and_prediction(api):
    response = api.get("/api/stocks/AAPL")
    assert response.status_code == 200
    body = response.json()

    assert body["quote"]["symbol"] == "AAPL"
    assert body["quote"]["last_price"] > 0
    assert len(body["bars"]) > 200

    prediction = body["prediction"]
    assert prediction["target_day"] > prediction["base_day"]
    assert prediction["interval_low"] <= prediction["predicted_close"] <= prediction["interval_high"]
    assert prediction["direction"] in {"up", "down", "flat"}
    assert prediction["backtest"]["test_rows"] > 0
    assert "not investment advice" in body["disclaimer"].lower()


def test_get_stock_is_case_insensitive(api):
    assert api.get("/api/stocks/aapl").json()["quote"]["symbol"] == "AAPL"


def test_history_window_is_respected(api):
    api.get("/api/stocks/AAPL")          # populate

    full = api.get("/api/stocks/AAPL/history", params={"days": 2000}).json()
    month = api.get("/api/stocks/AAPL/history", params={"days": 30}).json()

    assert len(month) < len(full)
    assert all(bar["day"] >= month[0]["day"] for bar in month)


def test_bars_are_chronological(api):
    bars = api.get("/api/stocks/AAPL").json()["bars"]
    assert [b["day"] for b in bars] == sorted(b["day"] for b in bars)


def test_prediction_endpoint(api):
    response = api.get("/api/stocks/AAPL/prediction")
    assert response.status_code == 200

    body = response.json()
    assert body["symbol"] == "AAPL"
    assert body["model_version"] == "ridge-v1"
    assert len(body["drivers"]) > 0


def test_unknown_symbol_uses_the_error_envelope(api):
    response = api.get("/api/stocks/FAKE")
    assert response.status_code == 404

    body = response.json()
    assert body["error"]["code"] == "symbol_not_found"
    assert "FAKE" in body["error"]["message"]


def test_tracked_list_grows_after_a_lookup(api):
    assert api.get("/api/stocks").json() == []
    api.get("/api/stocks/AAPL")
    assert [q["symbol"] for q in api.get("/api/stocks").json()] == ["AAPL"]


def test_refresh_endpoint(api, client):
    api.get("/api/stocks/AAPL")
    before = client.history_calls

    response = api.post("/api/stocks/AAPL/refresh")
    assert response.status_code == 200
    assert client.history_calls > before


def test_openapi_schema_is_generated(api):
    schema = api.get("/openapi.json").json()
    assert "/api/stocks/{symbol}" in schema["paths"]


# ---------------------------------------------------------------
# Accuracy tracking
# ---------------------------------------------------------------

def test_accuracy_summary_is_empty_before_any_audit(api):
    body = api.get("/api/accuracy").json()
    assert body["tickers_audited"] == 0
    assert body["run_date"] is None
    assert body["track_record"]["n"] == 0


def test_accuracy_summary_after_an_audit(api, session, client):
    from app import scoring

    scoring.audit_universe(session, client, symbols=["AAPL", "MSFT"])
    session.commit()

    body = api.get("/api/accuracy").json()
    assert body["tickers_audited"] == 2
    assert body["run_date"] is not None
    assert len(body["runs"]) == 2
    assert body["pooled_sample_size"] > 0
    assert body["pooled_is_significant"] in (True, False)
    # Sorted best-first by MAE ratio.
    ratios = [run["mae_ratio"] for run in body["runs"]]
    assert ratios == sorted(ratios)


def test_accuracy_history_endpoint(api, session, client):
    from app import scoring

    scoring.audit_universe(session, client, symbols=["AAPL", "MSFT"])
    session.commit()

    everything = api.get("/api/accuracy/history").json()
    just_aapl = api.get("/api/accuracy/history", params={"symbol": "AAPL"}).json()

    assert len(everything) == 2
    assert len(just_aapl) == 1
    assert just_aapl[0]["symbol"] == "AAPL"


def test_track_record_endpoint(api, session, client):
    from app import repository as repo
    from app import scoring, service
    from app.models import Prediction

    service.get_dashboard(session, "AAPL", client)
    bars = repo.get_bars(session, "AAPL")
    base, target = bars[-3], bars[-2]

    repo.save_prediction(
        session,
        Prediction(
            symbol="AAPL", target_day=target.day, base_day=base.day,
            base_close=base.close, predicted_close=target.close,
            predicted_return_pct=0.0,
            interval_low=target.close * 0.98, interval_high=target.close * 1.02,
            model_version="ridge-v1",
        ),
    )
    scoring.score_predictions(session)
    session.commit()

    body = api.get("/api/accuracy/track-record").json()
    assert body["n"] == 1
    assert body["is_meaningful"] is False        # one sample proves nothing
    assert body["mae_pct"] == pytest.approx(0.0, abs=1e-6)


def test_scored_predictions_endpoint_exposes_outcomes(api, session, client):
    from app import repository as repo
    from app import scoring, service
    from app.models import Prediction

    service.get_dashboard(session, "AAPL", client)
    bars = repo.get_bars(session, "AAPL")
    base, target = bars[-3], bars[-2]

    repo.save_prediction(
        session,
        Prediction(
            symbol="AAPL", target_day=target.day, base_day=base.day,
            base_close=base.close, predicted_close=target.close * 1.01,
            predicted_return_pct=1.0,
            interval_low=target.close, interval_high=target.close * 1.02,
            model_version="ridge-v1",
        ),
    )
    scoring.score_predictions(session)
    session.commit()

    rows = api.get("/api/accuracy/predictions").json()
    assert len(rows) == 1
    assert rows[0]["actual_close"] == pytest.approx(target.close)
    assert rows[0]["error_pct"] == pytest.approx(1.0, abs=0.01)
    assert rows[0]["direction_correct"] in (True, False)


# ---------------------------------------------------------------
# HTML pages
# ---------------------------------------------------------------

def test_home_page(api):
    response = api.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Next-day price estimates" in response.text


def test_stock_page_renders_the_dashboard(api):
    response = api.get("/stock/AAPL")
    assert response.status_code == 200

    html = response.text
    assert "Apple Inc." in html
    assert 'id="price-chart"' in html
    assert 'id="chart-data"' in html
    assert "Next-day estimate" in html
    assert "80% error band" in html
    assert "Does this model actually work?" in html


def test_stock_page_always_carries_a_disclaimer(api):
    html = api.get("/stock/AAPL").text
    assert "Not investment advice" in html
    assert "Past performance does not indicate future results" in html


def test_stock_page_embeds_chart_json(api):
    import json
    import re

    html = api.get("/stock/AAPL").text
    match = re.search(r'<script id="chart-data" type="application/json">(.*?)</script>', html, re.S)
    assert match

    series = json.loads(match.group(1))
    assert len(series) > 200
    assert set(series[0]) == {"d", "c"}


def test_unknown_symbol_page_is_a_404(api):
    response = api.get("/stock/FAKE")
    assert response.status_code == 404
    assert "Ticker not found" in response.text


def test_search_page(api):
    response = api.get("/search", params={"q": "Microsoft"})
    assert response.status_code == 200
    assert "MSFT" in response.text


def test_search_page_with_no_hits(api):
    response = api.get("/search", params={"q": "zzzznothing"})
    assert response.status_code == 200
    assert "No tradeable equities" in response.text


def test_exact_ticker_search_redirects(api):
    response = api.get("/search", params={"q": "AAPL"}, follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/stock/AAPL"


def test_refresh_form_redirects(api):
    response = api.post("/stock/AAPL/refresh", follow_redirects=False)
    assert response.status_code == 303
    assert "refresh=true" in response.headers["location"]
