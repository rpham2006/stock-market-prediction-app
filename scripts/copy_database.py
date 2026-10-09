r"""
One-time copy of every row from one database to another — e.g. the local
SQLite file to the hosted Postgres the scheduled audit writes to, so the
track record carries on instead of starting from zero.

    .venv\Scripts\python.exe -m scripts.copy_database --to "postgresql://..."

The source defaults to the local SQLite file. The target must be empty:
merging two track records row by row would mean guessing which copy of a
prediction is the real one, so this refuses rather than guesses.

Integer ids are not copied — nothing references them, and letting the
target assign its own keeps its sequences consistent.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import Integer, create_engine, func, insert, select

# Allow `python scripts\copy_database.py` as well as `-m scripts.copy_database`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DATA_DIR, _database_url  # noqa: E402
from app.models import Base  # noqa: E402

DEFAULT_SOURCE = f"sqlite:///{DATA_DIR / 'stockapp.db'}"


def _copyable_columns(table):
    """Every column except an auto-assigned integer primary key."""
    return [
        column for column in table.columns
        if not (column.primary_key and column.autoincrement is True
                and isinstance(column.type, Integer))
    ]


def _utc(value):
    # SQLite stores datetimes without a zone; the app always writes UTC.
    # Say so explicitly, or Postgres would read them in its own zone.
    if isinstance(value, datetime) and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def copy_database(source_url: str, target_url: str) -> dict[str, int]:
    """Copy every table; returns rows copied per table."""
    source = create_engine(_database_url(source_url))
    target = create_engine(_database_url(target_url))
    Base.metadata.create_all(target)

    copied: dict[str, int] = {}
    with source.connect() as read, target.begin() as write:
        for table in Base.metadata.sorted_tables:
            if write.scalar(select(func.count()).select_from(table)):
                raise SystemExit(
                    f"target table {table.name!r} already has rows — refusing to merge."
                )

        # sorted_tables is dependency order: companies before the tables
        # whose foreign keys point at it.
        for table in Base.metadata.sorted_tables:
            columns = _copyable_columns(table)
            rows = [
                {key: _utc(value) for key, value in row._mapping.items()}
                for row in read.execute(select(*columns))
            ]
            if rows:
                write.execute(insert(table), rows)
            copied[table.name] = len(rows)

    source.dispose()
    target.dispose()
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy every row to an empty database.")
    parser.add_argument("--from", dest="source", default=DEFAULT_SOURCE,
                        help="source database URL (default: the local SQLite file)")
    parser.add_argument("--to", dest="target", required=True,
                        help="target database URL, e.g. the Neon connection string")
    args = parser.parse_args()

    for table, count in copy_database(args.source, args.target).items():
        print(f"  {table:<16}{count:>7} rows")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
