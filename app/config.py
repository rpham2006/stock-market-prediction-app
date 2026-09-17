"""
Application settings.

Everything tunable lives here and is overridable by an environment
variable, so the same code runs on your laptop against SQLite and on a
server against Postgres without a source change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable config snapshot, built once at import time."""

    # --- storage -------------------------------------------------
    # Swap to Postgres with:
    #   set DATABASE_URL=postgresql+psycopg://user:pw@host/dbname
    # Nothing else in the app changes; SQLAlchemy handles the dialect.
    database_url: str = os.environ.get(
        "DATABASE_URL", f"sqlite:///{DATA_DIR / 'stockapp.db'}"
    )

    # --- data window --------------------------------------------
    # "up to a year ago" — the history window we store per ticker.
    history_period: str = os.environ.get("HISTORY_PERIOD", "1y")

    # --- freshness ----------------------------------------------
    # How long stored rows stay usable before we re-fetch upstream.
    # Bars are immutable once a session closes, so they age slowly;
    # a live quote goes stale in minutes.
    bar_ttl_minutes: int = _env_int("BAR_TTL_MINUTES", 6 * 60)
    quote_ttl_minutes: int = _env_int("QUOTE_TTL_MINUTES", 5)

    # --- model ---------------------------------------------------
    ridge_lambda: float = float(os.environ.get("RIDGE_LAMBDA", "3.0"))
    min_train_rows: int = _env_int("MIN_TRAIN_ROWS", 120)
    # Predictions are cached per target trading day.
    prediction_ttl_minutes: int = _env_int("PREDICTION_TTL_MINUTES", 60)

    # --- http ----------------------------------------------------
    request_timeout_seconds: int = _env_int("REQUEST_TIMEOUT_SECONDS", 20)

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


settings = Settings()

# SQLite needs the containing folder to exist before it will open a file.
if settings.is_sqlite:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
