"""PostgreSQL-backed durable queue (multi-host alternative to the Redis queue).

One ``uzen_queue`` table holds ready, leased, done and dead messages. Claiming is a
single ``UPDATE … WHERE id = (SELECT … FOR UPDATE SKIP LOCKED) RETURNING *``, so any
number of control-plane processes or workers can pull from the same table without
handing the same message to two consumers.

Semantics:

* ``attempts`` is incremented on every claim. ``nack`` returns a message to ``ready``
  (optionally with a delay) until it reaches ``max_attempts``, then moves it to
  ``dead`` with the last error — nothing is dropped or retried forever.
* A lease that is not renewed expires; ``reap`` makes expired messages claimable
  again (or dead when out of attempts).
* ``enqueue`` accepts an ``idempotency_key``; a duplicate key returns the existing
  message id instead of enqueuing a second copy.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from uzen.errors import RuntimeUnavailable
from uzen.util.canonical import utc_now_iso

try:  # optional dependency
    import psycopg
    from psycopg.rows import dict_row

    PSYCOPG_AVAILABLE = True
except Exception as exc:  # pragma: no cover - exercised only without the extra
    psycopg = None  # type: ignore[assignment]
    dict_row = None  # type: ignore[assignment]
    PSYCOPG_AVAILABLE = False
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

QUEUE_SCHEMA = """
CREATE TABLE IF NOT EXISTS uzen_queue (
    id BIGSERIAL PRIMARY KEY,
    topic TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    idempotency_key TEXT UNIQUE,
    priority INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 5,
    available_at DOUBLE PRECISION NOT NULL,
    lease_expires_at DOUBLE PRECISION,
    worker_id TEXT,
    state TEXT NOT NULL DEFAULT 'ready',
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_uzen_queue_ready
    ON uzen_queue(state, topic, priority DESC, id ASC);
CREATE INDEX IF NOT EXISTS idx_uzen_queue_lease
    ON uzen_queue(state, lease_expires_at);
"""


def _require_psycopg() -> None:
    if not PSYCOPG_AVAILABLE:
        raise RuntimeUnavailable(
            "the PostgreSQL queue needs the optional 'postgres' extra "
            "(pip install 'psycopg[binary]')"
        )


@dataclass(frozen=True)
class QueueMessage:
    id: int
    topic: str
    payload: dict[str, Any]
    attempts: int
    max_attempts: int
    priority: int
    worker_id: str | None
    available_at: float
    lease_expires_at: float | None
    state: str
    last_error: str | None
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "payload": self.payload,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "priority": self.priority,
            "worker_id": self.worker_id,
            "available_at": self.available_at,
            "lease_expires_at": self.lease_expires_at,
            "state": self.state,
            "last_error": self.last_error,
            "created_at": self.created_at,
        }


def _message(row: Mapping[str, Any]) -> QueueMessage:
    return QueueMessage(
        id=int(row["id"]),
        topic=str(row["topic"]),
        payload=json.loads(row["payload_json"]),
        attempts=int(row["attempts"]),
        max_attempts=int(row["max_attempts"]),
        priority=int(row["priority"]),
        worker_id=row["worker_id"],
        available_at=float(row["available_at"]),
        lease_expires_at=(
            float(row["lease_expires_at"]) if row["lease_expires_at"] is not None else None
        ),
        state=str(row["state"]),
        last_error=row["last_error"],
        created_at=str(row["created_at"]),
    )


class PostgresQueue:
    """A durable, multi-consumer queue on PostgreSQL."""

    def __init__(
        self,
        dsn: str,
        *,
        connect_timeout: float = 10.0,
        statement_timeout_ms: int | None = 30_000,
    ) -> None:
        _require_psycopg()
        if not str(dsn).startswith(("postgresql://", "postgres://")):
            raise ValueError("PostgresQueue needs a postgresql:// DSN")
        self.dsn = str(dsn)
        self._conn = psycopg.connect(
            self.dsn, connect_timeout=connect_timeout, row_factory=dict_row
        )
        # See PostgresDatabase: autocommit + explicit transaction blocks, so a claim
        # never leaves the connection idle-in-transaction holding row locks.
        self._conn.autocommit = True
        if statement_timeout_ms:
            self._conn.execute(f"SET statement_timeout = {int(statement_timeout_ms)}")
        with self._conn.cursor() as cursor:
            cursor.execute(QUEUE_SCHEMA)

    # ------------------------------------------------------------------ #

    def close(self) -> None:
        self._conn.close()

    def _one(self, sql: str, params: Any = None) -> dict[str, Any] | None:
        with self._conn.cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.fetchone()

    def enqueue(
        self,
        topic: str,
        payload: Mapping[str, Any],
        *,
        priority: int = 0,
        available_at: float | None = None,
        idempotency_key: str | None = None,
        max_attempts: int = 5,
    ) -> int:
        if int(max_attempts) <= 0:
            raise ValueError("max_attempts must be positive")
        now = time.time()
        text = json.dumps(dict(payload), separators=(",", ":"), sort_keys=True)
        with self._conn.transaction():
            row = self._one(
                "INSERT INTO uzen_queue(topic, payload_json, idempotency_key, priority, "
                "max_attempts, available_at, created_at, updated_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (idempotency_key) DO NOTHING "
                "RETURNING id",
                (
                    str(topic),
                    text,
                    idempotency_key,
                    int(priority),
                    int(max_attempts),
                    float(available_at if available_at is not None else now),
                    utc_now_iso(),
                    utc_now_iso(),
                ),
            )
            if row is not None:
                return int(row["id"])
            existing = self._one(
                "SELECT id FROM uzen_queue WHERE idempotency_key = %s", (idempotency_key,)
            )
        if existing is None:  # pragma: no cover - conflict implies a row exists
            raise RuntimeError("idempotent enqueue lost its row")
        return int(existing["id"])

    def claim(
        self,
        worker_id: str,
        *,
        topics: Sequence[str] | None = None,
        lease_seconds: float = 60.0,
        limit: int = 1,
        now: float | None = None,
    ) -> list[QueueMessage]:
        now = now if now is not None else time.time()
        if int(limit) <= 0:
            return []
        topic_clause = ""
        topic_params: dict[str, Any] = {}
        if topics:
            names = tuple(str(topic) for topic in topics)
            placeholders = ", ".join(f"%(topic{index})s" for index in range(len(names)))
            topic_clause = f"AND topic IN ({placeholders})"
            topic_params = {f"topic{index}": name for index, name in enumerate(names)}
        statement = f"""
            WITH candidate AS (
                SELECT id FROM uzen_queue
                WHERE state = 'ready' AND available_at <= %(now)s {topic_clause}
                ORDER BY priority DESC, id ASC
                FOR UPDATE SKIP LOCKED
                LIMIT %(limit)s
            )
            UPDATE uzen_queue AS q SET state = 'leased', worker_id = %(worker)s,
                attempts = q.attempts + 1, lease_expires_at = %(expires)s,
                updated_at = %(now_iso)s
            FROM candidate WHERE q.id = candidate.id
            RETURNING q.*
        """
        with self._conn.transaction():
            with self._conn.cursor() as cursor:
                cursor.execute(
                    statement,
                    {
                        "now": float(now),
                        "limit": int(limit),
                        "worker": str(worker_id),
                        "expires": float(now) + float(lease_seconds),
                        "now_iso": utc_now_iso(),
                        **topic_params,
                    },
                )
                rows = list(cursor.fetchall())
        return [_message(row) for row in rows]

    def renew(self, message_id: int, worker_id: str, *, lease_seconds: float = 60.0) -> bool:
        with self._conn.transaction():
            row = self._one(
                "UPDATE uzen_queue SET lease_expires_at = %s, updated_at = %s "
                "WHERE id = %s AND worker_id = %s AND state = 'leased' RETURNING id",
                (time.time() + float(lease_seconds), utc_now_iso(), int(message_id), str(worker_id)),
            )
        return row is not None

    def ack(self, message_id: int, worker_id: str) -> bool:
        with self._conn.transaction():
            row = self._one(
                "UPDATE uzen_queue SET state = 'done', lease_expires_at = NULL, last_error = NULL, "
                "updated_at = %s WHERE id = %s AND worker_id = %s AND state = 'leased' "
                "RETURNING id",
                (utc_now_iso(), int(message_id), str(worker_id)),
            )
        return row is not None

    def nack(
        self,
        message_id: int,
        worker_id: str,
        *,
        retry_in: float = 0.0,
        error: str | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        with self._conn.transaction():
            row = self._one(
                "SELECT * FROM uzen_queue WHERE id = %s AND worker_id = %s AND state = 'leased' "
                "FOR UPDATE",
                (int(message_id), str(worker_id)),
            )
            if row is None:
                return {"status": "not_leased", "id": int(message_id)}
            if int(row["attempts"]) >= int(row["max_attempts"]):
                state = "dead"
                available_at = float(row["available_at"])
            else:
                state = "ready"
                available_at = now + max(0.0, float(retry_in))
            self._one(
                "UPDATE uzen_queue SET state = %s, worker_id = NULL, lease_expires_at = NULL, "
                "available_at = %s, last_error = %s, updated_at = %s WHERE id = %s RETURNING id",
                (state, available_at, error, utc_now_iso(), int(message_id)),
            )
        return {
            "status": "dead" if state == "dead" else "requeued",
            "id": int(message_id),
            "state": state,
            "attempts": int(row["attempts"]),
            "max_attempts": int(row["max_attempts"]),
            "available_at": available_at,
        }

    def reap(self, *, now: float | None = None) -> dict[str, int]:
        """Recover messages whose lease expired or whose worker died."""
        now = now if now is not None else time.time()
        with self._conn.transaction():
            with self._conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE uzen_queue SET state = CASE WHEN attempts >= max_attempts "
                    "THEN 'dead' ELSE 'ready' END, worker_id = NULL, lease_expires_at = NULL, "
                    "available_at = %s, last_error = COALESCE(last_error, 'lease expired'), "
                    "updated_at = %s WHERE state = 'leased' AND lease_expires_at IS NOT NULL "
                    "AND lease_expires_at <= %s RETURNING state",
                    (now, utc_now_iso(), now),
                )
                states = [row["state"] for row in cursor.fetchall()]
        return {
            "requeued": states.count("ready"),
            "dead": states.count("dead"),
            "total": len(states),
        }

    def depth(self, *, topic: str | None = None) -> dict[str, int]:
        query = "SELECT state, COUNT(*) AS n FROM uzen_queue"
        params: list[Any] = []
        if topic is not None:
            query += " WHERE topic = %s"
            params.append(str(topic))
        query += " GROUP BY state"
        with self._conn.cursor() as cursor:
            cursor.execute(query, params or None)
            rows = list(cursor.fetchall())
        return {str(row["state"]): int(row["n"]) for row in rows}

    def get(self, message_id: int) -> QueueMessage | None:
        with self._conn.cursor() as cursor:
            cursor.execute("SELECT * FROM uzen_queue WHERE id = %s", (int(message_id),))
            row = cursor.fetchone()
        return None if row is None else _message(row)

    def purge(self, *, states: Iterable[str] = ("done", "dead")) -> int:
        wanted = tuple(str(state) for state in states)
        if not wanted:
            return 0
        placeholders = ", ".join(["%s"] * len(wanted))
        with self._conn.transaction():
            with self._conn.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM uzen_queue WHERE state IN ({placeholders})", wanted
                )
                return int(cursor.rowcount)

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "postgresql",
            "table": "uzen_queue",
            "depth": self.depth(),
        }


__all__ = ["QUEUE_SCHEMA", "PostgresQueue", "QueueMessage"]
