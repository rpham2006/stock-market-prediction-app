"""
Database copy tests — the one-time move from local SQLite to hosted Postgres.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app import service
from app.config import _database_url
from app.models import Base, DailyBar, Prediction
from scripts.copy_database import copy_database


@pytest.mark.parametrize(
    "given, expected",
    [
        ("postgres://u:p@host/db", "postgresql+psycopg://u:p@host/db"),
        ("postgresql://u:p@host/db?sslmode=require",
         "postgresql+psycopg://u:p@host/db?sslmode=require"),
        ("postgresql+psycopg://u:p@host/db", "postgresql+psycopg://u:p@host/db"),
        ("sqlite:///data/stockapp.db", "sqlite:///data/stockapp.db"),
    ],
)
def test_hosted_postgres_urls_use_the_installed_driver(given, expected):
    assert _database_url(given) == expected


def _file_db(path, client):
    """A SQLite file holding one ticker's bars and a stored prediction."""
    url = f"sqlite:///{path}"
    engine = create_engine(url)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        service.get_dashboard(session, "AAPL", client)
        session.commit()
    engine.dispose()
    return url


def test_copy_moves_every_row(tmp_path, client):
    source = _file_db(tmp_path / "source.db", client)
    target = f"sqlite:///{tmp_path / 'target.db'}"

    copied = copy_database(source, target)

    engine = create_engine(target)
    with engine.connect() as conn:
        assert conn.scalar(select(func.count()).select_from(DailyBar)) == copied["daily_bars"] > 0
        assert conn.scalar(select(func.count()).select_from(Prediction)) == copied["predictions"] == 1
    engine.dispose()


def test_copy_refuses_a_target_that_already_has_rows(tmp_path, client):
    source = _file_db(tmp_path / "source.db", client)
    target = f"sqlite:///{tmp_path / 'target.db'}"
    copy_database(source, target)

    with pytest.raises(SystemExit, match="refusing to merge"):
        copy_database(source, target)
