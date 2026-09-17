"""
FastAPI application factory and entrypoint.

Run it:
    .venv\\Scripts\\python.exe -m uvicorn app.main:app --reload

Then open http://127.0.0.1:8000 for the UI, or http://127.0.0.1:8000/docs
for the generated OpenAPI console.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from stockkit import MarketDataUnavailableError, SymbolNotFoundError

from . import __version__, api, web
from .config import PROJECT_ROOT, settings
from .db import init_db

logger = logging.getLogger("stockapp")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup and shutdown. Tables are created once, on boot."""
    init_db()
    logger.info("database ready at %s", settings.database_url)
    yield


def _error(status: int, code: str, message: str) -> JSONResponse:
    """The single error envelope, per ROADMAP §API conventions."""
    return JSONResponse(
        status_code=status, content={"error": {"code": code, "message": message}}
    )


def create_app() -> FastAPI:
    app = FastAPI(
        title="Stock Market Prediction App",
        version=__version__,
        description=(
            "Stores a year of daily market data per ticker and produces a "
            "next-day price estimate from a walk-forward-validated ridge "
            "model. Educational use only — not investment advice."
        ),
        lifespan=lifespan,
    )

    # --- upstream failures become clean HTTP responses ---
    @app.exception_handler(SymbolNotFoundError)
    async def _not_found(request: Request, exc: SymbolNotFoundError) -> JSONResponse:
        return _error(404, "symbol_not_found", str(exc))

    @app.exception_handler(MarketDataUnavailableError)
    async def _unavailable(
        request: Request, exc: MarketDataUnavailableError
    ) -> JSONResponse:
        return _error(503, "market_data_unavailable", str(exc))

    app.include_router(api.router)
    app.include_router(web.router)

    app.mount(
        "/static",
        StaticFiles(directory=str(PROJECT_ROOT / "app" / "static")),
        name="static",
    )
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
