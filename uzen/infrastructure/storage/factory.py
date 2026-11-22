"""Database backend selection for the control plane.

A deployment chooses its metadata store by the ``db_path`` it passes to
:class:`~uzen.jobs.service.JobService` (or ``UZEN_DB`` for the CLI/API):

* ``postgresql://…`` / ``postgres://…`` -> :class:`PostgresDatabase` (multi-host)
* any other value -> the single-node SQLite :class:`Database`

The factory keeps the choice in one place so every caller (service, worker daemon,
scripts) selects the same backend from the same string.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from uzen.infrastructure.storage.db import Database


def is_postgres_dsn(location: str | Path | None) -> bool:
    return isinstance(location, str) and location.startswith(("postgresql://", "postgres://"))


def open_database(location: str | Path, *, setup: bool = True) -> Any:
    """Open the metadata store named by ``location``."""
    if is_postgres_dsn(location):
        from uzen.infrastructure.storage.postgres import PostgresDatabase

        return PostgresDatabase(str(location), setup=setup)
    return Database(location)


def database_location(default: str | Path) -> str | Path:
    """Resolve the configured store location (``UZEN_DB`` wins)."""
    configured = os.environ.get("UZEN_DB")
    return configured if configured else default


__all__ = ["database_location", "is_postgres_dsn", "open_database"]
