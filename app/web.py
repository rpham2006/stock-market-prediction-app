"""
HTML routes — the server-rendered frontend.

Same service layer as api.py, different renderer. The page ships a small
JSON blob for the chart and no framework; the SVG is drawn by
static/chart.js. That keeps the whole stack Python and the page fast, and
leaves the JSON API free to serve a JavaScript frontend later.
"""

from __future__ import annotations

import json
from datetime import date

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from stockkit import MarketDataClient, MarketDataError, SymbolNotFoundError

from . import service
from .config import PROJECT_ROOT
from .db import get_session
from .models import DailyBar

router = APIRouter(tags=["web"])
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "app" / "templates"))

DISCLAIMER = (
    "For educational and informational purposes only. Not investment advice, "
    "not a recommendation to buy or sell any security. Model output is not a "
    "forecast of actual prices. Past performance does not indicate future results."
)


def _chart_payload(bars: list[DailyBar]) -> str:
    """Compact JSON series for the client-side SVG chart."""
    return json.dumps(
        [{"d": bar.day.isoformat(), "c": round(bar.close, 4)} for bar in bars]
    )


def _fmt_compact(value: int | float | None) -> str:
    """Humanise large numbers — 23,177,785 becomes 23.2M."""
    if value is None:
        return "—"
    number = float(value)
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(number) >= threshold:
            return f"{number / threshold:.1f}{suffix}"
    return f"{number:,.0f}"


templates.env.filters["compact"] = _fmt_compact


def _base_context(request: Request) -> dict:
    return {"request": request, "disclaimer": DISCLAIMER, "today": date.today()}


@router.get("/", response_class=HTMLResponse, summary="Home")
def home(request: Request, session: Session = Depends(get_session)) -> HTMLResponse:
    context = _base_context(request)
    context["companies"] = service.tracked_companies(session)
    return templates.TemplateResponse(request, "index.html", context)


@router.get("/search", response_class=HTMLResponse, summary="Ticker search results")
def search_results(
    request: Request,
    q: str = Query("", description="Ticker or company name"),
    client: MarketDataClient = Depends(service.get_client),
) -> HTMLResponse:
    query = q.strip()
    matches = service.search_symbols(query, client) if query else []

    # An exact ticker hit is almost always what was meant — skip the list.
    if len(matches) == 1 and matches[0].symbol.upper() == query.upper():
        return RedirectResponse(url=f"/stock/{matches[0].symbol}", status_code=303)

    context = _base_context(request)
    context.update({"query": query, "matches": matches})
    return templates.TemplateResponse(request, "search.html", context)


@router.get("/stock/{symbol}", response_class=HTMLResponse, summary="Ticker dashboard")
def stock_page(
    request: Request,
    symbol: str,
    refresh: bool = Query(False),
    session: Session = Depends(get_session),
    client: MarketDataClient = Depends(service.get_client),
) -> HTMLResponse:
    context = _base_context(request)

    try:
        dashboard = service.get_dashboard(session, symbol, client, force=refresh)
        session.commit()
    except SymbolNotFoundError:
        context.update(
            {
                "title": "Ticker not found",
                "message": f"No market data exists for “{symbol.upper()}”.",
            }
        )
        return templates.TemplateResponse(request, "error.html", context, status_code=404)
    except MarketDataError as exc:
        context.update(
            {
                "title": "Market data unavailable",
                "message": f"Could not reach the data provider: {exc}",
            }
        )
        return templates.TemplateResponse(request, "error.html", context, status_code=503)

    prediction = dashboard.prediction
    context.update(
        {
            "company": dashboard.company,
            "bars": dashboard.bars,
            "prediction": prediction,
            "is_stale": dashboard.is_stale,
            "notice": dashboard.notice,
            "chart_json": _chart_payload(dashboard.bars),
            "prediction_json": json.dumps(
                {
                    "target_day": prediction.target_day.isoformat(),
                    "predicted_close": round(prediction.predicted_close, 4),
                    "interval_low": round(prediction.interval_low, 4),
                    "interval_high": round(prediction.interval_high, 4),
                }
                if prediction
                else None
            ),
        }
    )
    return templates.TemplateResponse(request, "stock.html", context)


@router.post("/stock/{symbol}/refresh", summary="Force refresh from the UI")
def refresh_stock(symbol: str) -> RedirectResponse:
    return RedirectResponse(url=f"/stock/{symbol}?refresh=true", status_code=303)
