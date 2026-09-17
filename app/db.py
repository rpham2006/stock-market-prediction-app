"""
Database engine and session plumbing (SQLAlchemy 2.0 style).

Two things live here and nowhere else: the engine (one per process,
owns the connection pool) and the session factory (one session per
request). Keeping them in one module is what lets the rest of the app
stay ignorant of whether it's talking to SQLite or Postgres.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from .config import settings

logger = logging.getLogger("stockapp.db")

# check_same_thread=False: FastAPI serves requests on a threadpool, and
# SQLite otherwise refuses to be touched from a thread it didn't open.
_connect_args = {"check_same_thread": False} if settings.is_sqlite else {}

engine = create_engine(
    settings.database_url,
    connect_args=_connect_args,
    pool_pre_ping=True,      # drop dead connections instead of erroring
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def _add_missing_columns() -> list[str]:
    """Add nullable columns present in the models but missing from the DB.

    `create_all` creates missing *tables* and then stops — it will never
    alter one that already exists, so a new column would silently not be
    there and every read of it would fail.

    This handles the one schema change that is safe to automate: adding
    a nullable column. Renames, type changes and new constraints are not
    attempted, because guessing at those corrupts data. When you need
    those, that is the moment to adopt Alembic properly.
    """
    from . import models

    inspector = inspect(engine)
    existing = set(inspector.get_table_names())
    added: list[str] = []

    with engine.begin() as connection:
        for table in models.Base.metadata.sorted_tables:
            if table.name not in existing:
                continue                    # create_all will build it

            present = {col["name"] for col in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in present:
                    continue
                if not column.nullable:
                    logger.warning(
                        "cannot auto-add NOT NULL column %s.%s — needs a migration",
                        table.name, column.name,
                    )
                    continue

                column_type = column.type.compile(engine.dialect)
                connection.execute(
                    text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {column_type}')
                )
                added.append(f"{table.name}.{column.name}")

    if added:
        logger.info("added columns: %s", ", ".join(added))
    return added


def init_db() -> None:
    """Bring the database schema up to date.

    Adds any newly-declared nullable columns to existing tables, then
    creates whatever tables don't exist yet.
    """
    from . import models  # noqa: F401 — import registers the mappings

    _add_missing_columns()
    models.Base.metadata.create_all(bind=engine)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for scripts and background work.

    Commits on success, rolls back on any exception, always closes.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Iterator[Session]:
    """FastAPI dependency — one session per request."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
