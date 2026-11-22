"""PostgreSQL control-plane store.

This is the multi-host deployment's metadata store: several control-plane processes
and worker fleets can share one database, which SQLite cannot do across hosts. The
public surface is the same one :class:`~uzen.infrastructure.storage.db.Database`
offers for jobs, events, artifacts, models, workers, leases, tokens and ACLs, so
:class:`~uzen.jobs.service.JobService` and :class:`~uzen.jobs.scheduler.JobScheduler`
run unchanged against either backend.

The concurrency primitives are the real PostgreSQL ones, not emulation:

* :meth:`PostgresDatabase.claim_job` selects its candidate with
  ``FOR UPDATE SKIP LOCKED`` inside the same statement that takes the lease, so two
  workers on different hosts can never receive the same job.
* :meth:`PostgresDatabase.finish_from_worker`, :meth:`request_cancel` and
  :meth:`transition` lock the job row (``SELECT … FOR UPDATE``) before deciding, which
  makes the "late worker report cannot overwrite a cancellation" guarantee hold
  across processes exactly as the SQLite ``BEGIN IMMEDIATE`` version does locally.
* Event sequence numbers are assigned under the same row lock, so ``(job_id, seq)``
  stays gap-free and unique.

The module imports ``psycopg`` lazily: the core engine still runs with only NumPy and
Pydantic installed. Constructing the store without the driver raises
:class:`~uzen.errors.RuntimeUnavailable` naming the extra to install.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from uzen.errors import IdempotencyConflict, JobNotFound, JobStateError, RuntimeUnavailable
from uzen.infrastructure.storage.db import (
    ALLOWED_TRANSITIONS,
    JOB_STATES,
    TERMINAL_STATES,
    JobRow,
    _ambient_project,
    _row_to_job as _sqlite_row_to_job,
)
from uzen.util.canonical import canonical_json, sha256_text, utc_now_iso

try:  # optional dependency
    import psycopg
    from psycopg.rows import dict_row

    PSYCOPG_AVAILABLE = True
    PSYCOPG_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - exercised only without the extra
    psycopg = None  # type: ignore[assignment]
    dict_row = None  # type: ignore[assignment]
    PSYCOPG_AVAILABLE = False
    PSYCOPG_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    task_scope TEXT NOT NULL,
    objective TEXT NOT NULL DEFAULT '',
    request_json TEXT NOT NULL,
    plan_json TEXT,
    result_json TEXT,
    error_json TEXT,
    idempotency_key TEXT UNIQUE,
    request_hash TEXT,
    worker_id TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    priority INTEGER NOT NULL DEFAULT 0,
    job_kind TEXT NOT NULL DEFAULT '',
    enqueued_at DOUBLE PRECISION,
    available_at DOUBLE PRECISION,
    dispatch_token TEXT,
    lease_expires_at DOUBLE PRECISION,
    min_cpu_count INTEGER,
    min_memory_bytes INTEGER,
    progress DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
    project_id TEXT NOT NULL DEFAULT 'default',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);
CREATE INDEX IF NOT EXISTS idx_jobs_kind ON jobs(job_kind);
CREATE INDEX IF NOT EXISTS idx_jobs_project ON jobs(project_id, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_claimable ON jobs(state, lease_expires_at, available_at);

CREATE TABLE IF NOT EXISTS job_events (
    id BIGSERIAL PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(job_id, seq)
);

CREATE TABLE IF NOT EXISTS artifacts_index (
    artifact_id TEXT PRIMARY KEY,
    name TEXT,
    family TEXT,
    operation TEXT,
    path TEXT NOT NULL,
    numel INTEGER,
    num_tensors INTEGER,
    job_id TEXT,
    project_id TEXT NOT NULL DEFAULT 'default',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_artifacts_project ON artifacts_index(project_id, created_at);

CREATE TABLE IF NOT EXISTS models (
    model_id TEXT PRIMARY KEY,
    artifact_id TEXT NOT NULL UNIQUE,
    family TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    project_id TEXT NOT NULL DEFAULT 'default',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS workers (
    worker_id TEXT PRIMARY KEY,
    protocol_version TEXT NOT NULL,
    capabilities_json TEXT NOT NULL,
    token_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    registered_at TEXT NOT NULL,
    cpu_count INTEGER,
    memory_bytes INTEGER
);

CREATE TABLE IF NOT EXISTS worker_capabilities (
    worker_id TEXT NOT NULL REFERENCES workers(worker_id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY(worker_id, kind, value)
);
CREATE INDEX IF NOT EXISTS idx_worker_capabilities ON worker_capabilities(kind, value);

CREATE TABLE IF NOT EXISTS job_requirements (
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    value TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'all',
    PRIMARY KEY(job_id, kind, value)
);

CREATE TABLE IF NOT EXISTS dispatch_records (
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    attempt INTEGER NOT NULL,
    worker_id TEXT NOT NULL,
    token TEXT NOT NULL,
    outcome TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    reported_at TEXT,
    PRIMARY KEY(job_id, attempt)
);

CREATE TABLE IF NOT EXISTS leases (
    job_id TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
    worker_id TEXT NOT NULL,
    expires_at DOUBLE PRECISION NOT NULL,
    attempt INTEGER NOT NULL DEFAULT 1,
    token TEXT
);

CREATE TABLE IF NOT EXISTS idempotency (
    key TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS api_tokens (
    token_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    token_hash TEXT NOT NULL UNIQUE,
    scopes_json TEXT NOT NULL,
    description TEXT,
    created_at TEXT NOT NULL,
    revoked_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_api_tokens_project ON api_tokens(project_id);

CREATE TABLE IF NOT EXISTS artifact_acl (
    artifact_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    permission TEXT NOT NULL,
    granted_at TEXT NOT NULL,
    PRIMARY KEY(artifact_id, project_id, permission)
);
"""


def _require_psycopg() -> None:
    if not PSYCOPG_AVAILABLE:
        raise RuntimeUnavailable(
            "the PostgreSQL store needs the optional 'postgres' extra "
            "(pip install 'psycopg[binary]')",
            reason=PSYCOPG_IMPORT_ERROR,
        )


def _job_from_row(row: Mapping[str, Any]) -> JobRow:
    """Convert a psycopg dict row into the shared :class:`JobRow`."""

    class _KeysRow(dict):  # the shared converter only uses ``row["x"]``/``row.keys()``
        pass

    return _sqlite_row_to_job(_KeysRow(row))


def _loads(value: Any) -> Any:
    return json.loads(value) if value else None


@dataclass
class _Claimed:
    job: JobRow
    token: str


class PostgresDatabase:
    """A psycopg-backed metadata store with the same surface as ``Database``."""

    backend = "postgresql"

    def __init__(
        self,
        dsn: str,
        *,
        connect_timeout: float = 10.0,
        statement_timeout_ms: int | None = 30_000,
        setup: bool = True,
    ) -> None:
        _require_psycopg()
        if not str(dsn).startswith(("postgresql://", "postgres://")):
            raise ValueError("PostgresDatabase needs a postgresql:// DSN")
        self.dsn = str(dsn)
        self.path = self.dsn
        self._lock = threading.RLock()
        self._conn = psycopg.connect(
            self.dsn, connect_timeout=connect_timeout, row_factory=dict_row
        )
        # Autocommit plus explicit ``transaction()`` blocks: reads never leave an idle
        # transaction holding locks, which would otherwise stall another process's
        # schema creation or a concurrent claim. Multi-statement operations (claim,
        # settle, append_event, create_job) still run inside one real transaction.
        self._conn.autocommit = True
        if statement_timeout_ms:
            self._conn.execute(f"SET statement_timeout = {int(statement_timeout_ms)}")
        if setup:
            with self._lock:
                with self._conn.cursor() as cursor:
                    cursor.execute(POSTGRES_SCHEMA)

    # ------------------------------------------------------------------ #
    # plumbing
    # ------------------------------------------------------------------ #

    def _fetchone(self, sql: str, params: Mapping[str, Any] | Sequence[Any] | None = None) -> dict[str, Any] | None:
        with self._conn.cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.fetchone()

    def _fetchall(self, sql: str, params: Mapping[str, Any] | Sequence[Any] | None = None) -> list[dict[str, Any]]:
        with self._conn.cursor() as cursor:
            cursor.execute(sql, params)
            return list(cursor.fetchall())

    def _execute(self, sql: str, params: Mapping[str, Any] | Sequence[Any] | None = None) -> int:
        with self._conn.cursor() as cursor:
            cursor.execute(sql, params)
            return int(cursor.rowcount)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ #
    # jobs
    # ------------------------------------------------------------------ #

    def create_job(
        self,
        request: Mapping[str, Any],
        *,
        job_id: str | None = None,
        idempotency_key: str | None = None,
        state: str = "QUEUED",
        requirements: Any | None = None,
        max_attempts: int = 3,
        priority: int = 0,
        project_id: str = "default",
    ) -> JobRow:
        job_id = job_id or str(uuid.uuid4())
        now = utc_now_iso()
        priority = max(-1000, min(1000, int(priority)))
        request_json = json.dumps(request, separators=(",", ":"), sort_keys=True)
        request_hash = sha256_text(request_json)
        options = request.get("options") if isinstance(request, Mapping) else None
        job_kind = str(options.get("task_kind") or "") if isinstance(options, Mapping) else ""
        with self._lock:
            if idempotency_key:
                existing = self._fetchone(
                    "SELECT * FROM idempotency WHERE key = %s", (idempotency_key,)
                )
                if existing:
                    if existing["request_hash"] != request_hash:
                        raise IdempotencyConflict(
                            "idempotency key was reused with a different request body",
                            key=idempotency_key,
                            existing_job=existing["job_id"],
                        )
                    return self.get_job(existing["job_id"])
            try:
                with self._conn.transaction():
                    self._execute(
                        "INSERT INTO jobs(id, state, task_scope, objective, request_json, "
                        "idempotency_key, request_hash, max_attempts, priority, job_kind, "
                        "enqueued_at, min_cpu_count, min_memory_bytes, project_id, created_at, "
                        "updated_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                        (
                            job_id,
                            state,
                            str(request.get("task_scope", "inspection")),
                            str(request.get("objective", "")),
                            request_json,
                            idempotency_key,
                            request_hash,
                            int(max_attempts),
                            priority,
                            job_kind,
                            time.time(),
                            getattr(requirements, "min_cpu_count", None) or None,
                            getattr(requirements, "min_memory_bytes", None),
                            str(project_id or "default"),
                            now,
                            now,
                        ),
                    )
                    if idempotency_key:
                        self._execute(
                            "INSERT INTO idempotency(key, job_id, request_hash, created_at) "
                            "VALUES (%s,%s,%s,%s)",
                            (idempotency_key, job_id, request_hash, now),
                        )
                    if requirements is not None:
                        self._replace_job_requirements(job_id, requirements)
            except Exception as exc:
                # A concurrent insert of the same idempotency key: the loser reads the
                # winner's row and either returns it or reports the hash conflict.
                if idempotency_key and psycopg is not None and isinstance(
                    exc, psycopg.errors.UniqueViolation
                ):
                    existing = self._fetchone(
                        "SELECT * FROM idempotency WHERE key = %s", (idempotency_key,)
                    )
                    if existing and existing["request_hash"] == request_hash:
                        return self.get_job(existing["job_id"])
                    raise IdempotencyConflict(
                        "idempotency key was reused with a different request body",
                        key=idempotency_key,
                        existing_job=existing["job_id"] if existing else None,
                    ) from exc
                raise
        self.append_event(job_id, "created", {"state": state, "task_scope": request.get("task_scope")})
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> JobRow:
        with self._lock:
            row = self._fetchone("SELECT * FROM jobs WHERE id = %s", (job_id,))
        if row is None:
            raise JobNotFound(f"job '{job_id}' not found")
        return _job_from_row(row)

    def project_of_job(self, job_id: str) -> str:
        with self._lock:
            row = self._fetchone("SELECT project_id FROM jobs WHERE id = %s", (job_id,))
        if row is None:
            raise JobNotFound(f"job '{job_id}' not found")
        return str(row["project_id"] or "default")

    def list_jobs(
        self,
        *,
        state: str | None = None,
        job_kind: str | None = None,
        project_id: str | None = None,
        before: tuple[str, str] | None = None,
        limit: int = 50,
    ) -> list[JobRow]:
        clauses: list[str] = []
        params: list[Any] = []
        if state:
            clauses.append("state = %s")
            params.append(state)
        if job_kind is not None:
            clauses.append("job_kind = %s")
            params.append(job_kind)
        if project_id is not None:
            clauses.append("project_id = %s")
            params.append(str(project_id))
        if before is not None:
            clauses.append("(created_at < %s OR (created_at = %s AND id < %s))")
            params.extend([before[0], before[0], before[1]])
        query = "SELECT * FROM jobs"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC, id DESC LIMIT %s"
        params.append(int(limit))
        with self._lock:
            rows = self._fetchall(query, params)
        return [_job_from_row(row) for row in rows]

    def transition(
        self,
        job_id: str,
        new_state: str,
        *,
        payload: Mapping[str, Any] | None = None,
        progress: float | None = None,
        force: bool = False,
    ) -> JobRow:
        if new_state not in JOB_STATES:
            raise JobStateError(f"unknown job state '{new_state}'")
        with self._lock:
            with self._conn.transaction():
                row = self._fetchone("SELECT * FROM jobs WHERE id = %s FOR UPDATE", (job_id,))
                if row is None:
                    raise JobNotFound(f"job '{job_id}' not found")
                current = row["state"]
                if (
                    not force
                    and new_state != current
                    and new_state not in ALLOWED_TRANSITIONS.get(current, set())
                ):
                    raise JobStateError(
                        f"illegal job transition {current} -> {new_state}", job_id=job_id
                    )
                self._execute(
                    "UPDATE jobs SET state = %s, progress = COALESCE(%s, progress), "
                    "updated_at = %s WHERE id = %s",
                    (new_state, None if progress is None else float(progress), utc_now_iso(), job_id),
                )
        self.append_event(job_id, "state_changed", {"from": current, "to": new_state, **(payload or {})})
        if new_state != current:
            self.append_event(
                job_id, new_state.lower(), {"from": current, "to": new_state, **(payload or {})}
            )
        return self.get_job(job_id)

    def set_plan(self, job_id: str, plan: Mapping[str, Any]) -> None:
        with self._lock, self._conn.transaction():
            self._execute(
                "UPDATE jobs SET plan_json = %s, updated_at = %s WHERE id = %s",
                (json.dumps(plan, separators=(",", ":"), sort_keys=True), utc_now_iso(), job_id),
            )

    def set_result(self, job_id: str, result: Mapping[str, Any], *, state: str = "SUCCEEDED") -> None:
        with self._lock, self._conn.transaction():
            self._execute(
                "UPDATE jobs SET result_json = %s, state = %s, progress = 1.0, updated_at = %s "
                "WHERE id = %s",
                (
                    json.dumps(result, separators=(",", ":"), sort_keys=True),
                    state,
                    utc_now_iso(),
                    job_id,
                ),
            )
        self.append_event(job_id, "completed", {"state": state})

    def set_error(self, job_id: str, error: Mapping[str, Any]) -> None:
        with self._lock, self._conn.transaction():
            self._execute(
                "UPDATE jobs SET error_json = %s, state = 'FAILED', updated_at = %s WHERE id = %s",
                (
                    json.dumps(error, separators=(",", ":"), sort_keys=True),
                    utc_now_iso(),
                    job_id,
                ),
            )
        self.append_event(job_id, "failed", dict(error))

    def request_cancel(self, job_id: str) -> JobRow:
        with self._lock:
            with self._conn.transaction():
                row = self._fetchone("SELECT * FROM jobs WHERE id = %s FOR UPDATE", (job_id,))
                if row is None:
                    raise JobNotFound(f"job '{job_id}' not found")
                previous = _job_from_row(row)
                if previous.state in TERMINAL_STATES:
                    return previous
                state = (
                    "CANCELLED"
                    if previous.state in ("CREATED", "QUEUED")
                    else "CANCEL_REQUESTED"
                )
                self._execute(
                    "UPDATE jobs SET cancel_requested = TRUE, state = %s, updated_at = %s "
                    "WHERE id = %s",
                    (state, utc_now_iso(), job_id),
                )
        self.append_event(job_id, "cancel_requested", {})
        if state == "CANCELLED":
            self.append_event(job_id, "state_changed", {"from": previous.state, "to": "CANCELLED"})
            self.append_event(job_id, "cancelled", {"from": previous.state, "to": "CANCELLED"})
        return self.get_job(job_id)

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            row = self._fetchone("SELECT cancel_requested FROM jobs WHERE id = %s", (job_id,))
        return bool(row and row["cancel_requested"])

    # ------------------------------------------------------------------ #
    # events
    # ------------------------------------------------------------------ #

    def append_event(self, job_id: str, kind: str, payload: Mapping[str, Any]) -> None:
        with self._lock:
            with self._conn.transaction():
                # The job row lock serialises sequence assignment across processes.
                self._fetchone("SELECT id FROM jobs WHERE id = %s FOR UPDATE", (job_id,))
                row = self._fetchone(
                    "SELECT COALESCE(MAX(seq), 0) AS m FROM job_events WHERE job_id = %s",
                    (job_id,),
                )
                seq = int(row["m"]) + 1
                self._execute(
                    "INSERT INTO job_events(job_id, seq, kind, payload_json, created_at) "
                    "VALUES (%s,%s,%s,%s,%s)",
                    (
                        job_id,
                        seq,
                        kind,
                        json.dumps(payload, separators=(",", ":"), sort_keys=True),
                        utc_now_iso(),
                    ),
                )

    def events(self, job_id: str, *, after: int = 0, limit: int = 500) -> list[dict[str, Any]]:
        self.get_job(job_id)
        with self._lock:
            rows = self._fetchall(
                "SELECT * FROM job_events WHERE job_id = %s AND seq > %s ORDER BY seq ASC LIMIT %s",
                (job_id, int(after), int(limit)),
            )
        return [
            {
                "seq": row["seq"],
                "kind": row["kind"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    # ------------------------------------------------------------------ #
    # artifacts / models
    # ------------------------------------------------------------------ #

    def index_artifact(
        self,
        *,
        artifact_id: str,
        path: str,
        name: str | None = None,
        family: str | None = None,
        operation: str | None = None,
        numel: int | None = None,
        num_tensors: int | None = None,
        job_id: str | None = None,
        project_id: str | None = None,
    ) -> None:
        if project_id is None:
            project_id = self.project_of_job(job_id) if job_id is not None else _ambient_project()
        with self._lock, self._conn.transaction():
            self._execute(
                "INSERT INTO artifacts_index(artifact_id, name, family, operation, path, numel, "
                "num_tensors, job_id, project_id, created_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (artifact_id) DO UPDATE SET name = EXCLUDED.name, "
                "family = EXCLUDED.family, operation = EXCLUDED.operation, path = EXCLUDED.path, "
                "numel = EXCLUDED.numel, num_tensors = EXCLUDED.num_tensors, "
                "job_id = EXCLUDED.job_id, project_id = EXCLUDED.project_id, "
                "created_at = EXCLUDED.created_at",
                (
                    artifact_id,
                    name,
                    family,
                    operation,
                    path,
                    numel,
                    num_tensors,
                    job_id,
                    str(project_id or "default"),
                    utc_now_iso(),
                ),
            )

    def artifacts(
        self,
        limit: int = 100,
        *,
        project_id: str | None = None,
        before: tuple[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if project_id is not None:
            clauses.append("project_id = %s")
            params.append(str(project_id))
        if before is not None:
            clauses.append("(created_at < %s OR (created_at = %s AND artifact_id < %s))")
            params.extend([before[0], before[0], before[1]])
        query = "SELECT * FROM artifacts_index"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC, artifact_id DESC LIMIT %s"
        params.append(int(limit))
        with self._lock:
            return self._fetchall(query, params)

    def artifact_project(self, artifact_id: str) -> str | None:
        with self._lock:
            row = self._fetchone(
                "SELECT project_id FROM artifacts_index WHERE artifact_id = %s", (artifact_id,)
            )
        return None if row is None else str(row["project_id"] or "default")

    def grant_artifact_access(
        self, artifact_id: str, project_id: str, *, permission: str = "read"
    ) -> dict[str, Any]:
        if permission not in ("read", "write"):
            raise ValueError("artifact permission must be 'read' or 'write'")
        grant = {
            "artifact_id": artifact_id,
            "project_id": str(project_id),
            "permission": permission,
            "granted_at": utc_now_iso(),
        }
        with self._lock, self._conn.transaction():
            self._execute(
                "INSERT INTO artifact_acl(artifact_id, project_id, permission, granted_at) "
                "VALUES (%s,%s,%s,%s) ON CONFLICT (artifact_id, project_id, permission) "
                "DO UPDATE SET granted_at = EXCLUDED.granted_at",
                (artifact_id, grant["project_id"], permission, grant["granted_at"]),
            )
        return grant

    def artifact_acls(self, artifact_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return self._fetchall(
                "SELECT artifact_id, project_id, permission, granted_at FROM artifact_acl "
                "WHERE artifact_id = %s ORDER BY project_id, permission",
                (artifact_id,),
            )

    def revoke_artifact_access(self, artifact_id: str, project_id: str) -> int:
        with self._lock, self._conn.transaction():
            return self._execute(
                "DELETE FROM artifact_acl WHERE artifact_id = %s AND project_id = %s",
                (artifact_id, str(project_id)),
            )

    def register_model(
        self, *, artifact_id: str, family: str, spec: Mapping[str, Any], project_id: str | None = None
    ) -> str:
        model_id = f"model_{sha256_text(artifact_id)[:16]}"
        if project_id is None:
            project_id = self.artifact_project(artifact_id) or _ambient_project()
        with self._lock, self._conn.transaction():
            self._execute(
                "INSERT INTO models(model_id, artifact_id, family, spec_json, project_id, created_at) "
                "VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT (artifact_id) DO UPDATE SET "
                "model_id = EXCLUDED.model_id, family = EXCLUDED.family, "
                "spec_json = EXCLUDED.spec_json, project_id = EXCLUDED.project_id, "
                "created_at = EXCLUDED.created_at",
                (model_id, artifact_id, family, canonical_json(dict(spec)), str(project_id), utc_now_iso()),
            )
        return model_id

    def models(
        self,
        limit: int = 100,
        *,
        project_id: str | None = None,
        before: tuple[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if project_id is not None:
            clauses.append("project_id = %s")
            params.append(str(project_id))
        if before is not None:
            clauses.append("(created_at < %s OR (created_at = %s AND model_id < %s))")
            params.extend([before[0], before[0], before[1]])
        query = "SELECT * FROM models"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC, model_id DESC LIMIT %s"
        params.append(int(limit))
        with self._lock:
            rows = self._fetchall(query, params)
        return [
            {
                "model_id": row["model_id"],
                "artifact_id": row["artifact_id"],
                "family": row["family"],
                "spec": json.loads(row["spec_json"]),
                "project_id": row["project_id"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    # ------------------------------------------------------------------ #
    # tokens
    # ------------------------------------------------------------------ #

    def create_api_token(
        self,
        project_id: str,
        scopes: Iterable[str],
        *,
        description: str | None = None,
        token_id: str | None = None,
    ) -> dict[str, Any]:
        import secrets as _secrets

        token_id = token_id or f"tok_{uuid.uuid4().hex[:16]}"
        secret = f"uzt_{_secrets.token_urlsafe(32)}"
        scope_list = sorted({str(scope) for scope in scopes})
        now = utc_now_iso()
        with self._lock, self._conn.transaction():
            self._execute(
                "INSERT INTO api_tokens(token_id, project_id, token_hash, scopes_json, description, "
                "created_at) VALUES (%s,%s,%s,%s,%s,%s)",
                (token_id, str(project_id), sha256_text(secret), canonical_json(scope_list), description, now),
            )
        return {
            "token_id": token_id,
            "token": secret,
            "project_id": str(project_id),
            "scopes": scope_list,
            "description": description,
            "created_at": now,
        }

    def authenticate_api_token(self, secret: str) -> dict[str, Any] | None:
        if not isinstance(secret, str) or not secret:
            return None
        with self._lock:
            row = self._fetchone(
                "SELECT * FROM api_tokens WHERE token_hash = %s", (sha256_text(secret),)
            )
        if row is None or row["revoked_at"]:
            return None
        return {
            "token_id": row["token_id"],
            "project_id": str(row["project_id"]),
            "scopes": json.loads(row["scopes_json"]),
            "description": row["description"],
            "created_at": row["created_at"],
            "revoked_at": row["revoked_at"],
        }

    def revoke_api_token(self, token_id: str) -> bool:
        with self._lock, self._conn.transaction():
            updated = self._execute(
                "UPDATE api_tokens SET revoked_at = %s WHERE token_id = %s AND revoked_at IS NULL",
                (utc_now_iso(), token_id),
            )
        return updated > 0

    def list_api_tokens(self, *, project_id: str | None = None) -> list[dict[str, Any]]:
        query = (
            "SELECT token_id, project_id, scopes_json, description, created_at, revoked_at "
            "FROM api_tokens"
        )
        params: list[Any] = []
        if project_id is not None:
            query += " WHERE project_id = %s"
            params.append(str(project_id))
        query += " ORDER BY created_at DESC"
        with self._lock:
            rows = self._fetchall(query, params)
        return [
            {
                "token_id": row["token_id"],
                "project_id": row["project_id"],
                "scopes": json.loads(row["scopes_json"]),
                "description": row["description"],
                "created_at": row["created_at"],
                "revoked_at": row["revoked_at"],
            }
            for row in rows
        ]

    # ------------------------------------------------------------------ #
    # workers
    # ------------------------------------------------------------------ #

    def register_worker(
        self,
        *,
        worker_id: str,
        protocol_version: str,
        capabilities: Mapping[str, Any],
        token_hash: str,
    ) -> None:
        now = utc_now_iso()
        with self._lock:
            with self._conn.transaction():
                self._execute(
                    "INSERT INTO workers(worker_id, protocol_version, capabilities_json, token_hash, "
                    "status, last_seen, registered_at) VALUES (%s,%s,%s,%s,%s,%s,%s) "
                    "ON CONFLICT (worker_id) DO UPDATE SET "
                    "capabilities_json = EXCLUDED.capabilities_json, "
                    "token_hash = EXCLUDED.token_hash, last_seen = EXCLUDED.last_seen, "
                    "status = CASE WHEN workers.status = 'draining' THEN 'draining' ELSE 'online' END",
                    (worker_id, protocol_version, canonical_json(dict(capabilities)), token_hash, "online", now, now),
                )
                self._set_worker_capabilities(worker_id, capabilities)

    def worker_token_hash(self, worker_id: str) -> str | None:
        with self._lock:
            row = self._fetchone("SELECT token_hash FROM workers WHERE worker_id = %s", (worker_id,))
        return None if row is None else str(row["token_hash"])

    def heartbeat(self, worker_id: str, status: str | None = None) -> None:
        with self._lock, self._conn.transaction():
            if status is None:
                self._execute(
                    "UPDATE workers SET last_seen = %s WHERE worker_id = %s",
                    (utc_now_iso(), worker_id),
                )
            else:
                self._execute(
                    "UPDATE workers SET last_seen = %s, status = %s WHERE worker_id = %s",
                    (utc_now_iso(), status, worker_id),
                )

    def _set_worker_capabilities(self, worker_id: str, capabilities: Mapping[str, Any]) -> None:
        from uzen.jobs.scheduling import capability_index

        index = capability_index(dict(capabilities))
        now = utc_now_iso()
        self._execute(
            "INSERT INTO workers(worker_id, protocol_version, capabilities_json, token_hash, status, "
            "last_seen, registered_at) VALUES (%s,%s,%s,%s,%s,%s,%s) "
            "ON CONFLICT (worker_id) DO NOTHING",
            (worker_id, "uzen.worker/1", "{}", "", "unregistered", now, now),
        )
        self._execute(
            "UPDATE workers SET capabilities_json = %s, cpu_count = %s, memory_bytes = %s "
            "WHERE worker_id = %s",
            (
                canonical_json(dict(capabilities)),
                int(capabilities["cpu_count"]) if capabilities.get("cpu_count") is not None else None,
                int(capabilities["memory_bytes"]) if capabilities.get("memory_bytes") is not None else None,
                worker_id,
            ),
        )
        self._execute("DELETE FROM worker_capabilities WHERE worker_id = %s", (worker_id,))
        for kind, values in index.items():
            for value in values:
                self._execute(
                    "INSERT INTO worker_capabilities(worker_id, kind, value) VALUES (%s,%s,%s) "
                    "ON CONFLICT DO NOTHING",
                    (worker_id, kind, value),
                )

    def set_worker_capabilities(self, worker_id: str, capabilities: Mapping[str, Any]) -> None:
        with self._lock, self._conn.transaction():
            self._set_worker_capabilities(worker_id, capabilities)

    def worker_capabilities(self, worker_id: str) -> dict[str, list[str]]:
        with self._lock:
            rows = self._fetchall(
                "SELECT kind, value FROM worker_capabilities WHERE worker_id = %s "
                "ORDER BY kind, value",
                (worker_id,),
            )
        out: dict[str, list[str]] = {}
        for row in rows:
            out.setdefault(row["kind"], []).append(row["value"])
        return out

    def _replace_job_requirements(self, job_id: str, requirements: Any) -> None:
        self._execute("DELETE FROM job_requirements WHERE job_id = %s", (job_id,))
        for kind, value, mode in requirements.rows():
            self._execute(
                "INSERT INTO job_requirements(job_id, kind, value, mode) VALUES (%s,%s,%s,%s) "
                "ON CONFLICT DO NOTHING",
                (job_id, kind, value, mode),
            )

    def set_job_requirements(self, job_id: str, requirements: Any) -> None:
        with self._lock, self._conn.transaction():
            self._replace_job_requirements(job_id, requirements)
            self._execute(
                "UPDATE jobs SET min_cpu_count = %s, min_memory_bytes = %s, updated_at = %s "
                "WHERE id = %s",
                (
                    getattr(requirements, "min_cpu_count", None) or None,
                    getattr(requirements, "min_memory_bytes", None),
                    utc_now_iso(),
                    job_id,
                ),
            )

    def job_requirements(self, job_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return self._fetchall(
                "SELECT kind, value, mode FROM job_requirements WHERE job_id = %s "
                "ORDER BY kind, value",
                (job_id,),
            )

    def workers(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._fetchall("SELECT * FROM workers ORDER BY registered_at DESC")
        return [
            {
                "worker_id": row["worker_id"],
                "protocol_version": row["protocol_version"],
                "capabilities": json.loads(row["capabilities_json"]),
                "status": row["status"],
                "last_seen": row["last_seen"],
                "registered_at": row["registered_at"],
                "cpu_count": row["cpu_count"],
                "memory_bytes": row["memory_bytes"],
            }
            for row in rows
        ]

    # ------------------------------------------------------------------ #
    # claiming and settlement
    # ------------------------------------------------------------------ #

    _CAPABILITY_PREDICATE = """
        NOT EXISTS (
            SELECT 1 FROM job_requirements r
            WHERE r.job_id = j.id AND r.mode = 'all'
              AND NOT EXISTS (
                  SELECT 1 FROM worker_capabilities c
                  WHERE c.worker_id = %(worker)s AND c.kind = r.kind AND c.value = r.value
              )
        )
        AND NOT EXISTS (
            SELECT 1 FROM job_requirements a
            WHERE a.job_id = j.id AND a.mode = 'any'
              AND NOT EXISTS (
                  SELECT 1 FROM worker_capabilities c
                  WHERE c.worker_id = %(worker)s AND c.kind = a.kind
                    AND c.value IN (
                        SELECT value FROM job_requirements
                        WHERE job_id = j.id AND kind = a.kind AND mode = 'any'
                    )
              )
        )
        AND (j.min_cpu_count IS NULL OR j.min_cpu_count <= COALESCE(
            (SELECT cpu_count FROM workers WHERE worker_id = %(worker)s), 0))
        AND (j.min_memory_bytes IS NULL OR j.min_memory_bytes <= COALESCE(
            (SELECT memory_bytes FROM workers WHERE worker_id = %(worker)s), 0))
    """

    def claim_job(
        self,
        worker_id: str,
        *,
        lease_seconds: float = 120.0,
        now: float | None = None,
        capabilities: Mapping[str, Any] | None = None,
        aging_seconds: float = 0.0,
        max_in_flight_per_worker: int = 0,
        job_kinds: Sequence[str] | None = None,
    ) -> JobRow | None:
        if capabilities is not None:
            self.set_worker_capabilities(worker_id, capabilities)
        now = now if now is not None else time.time()
        token = uuid.uuid4().hex
        expires_at = now + float(lease_seconds)
        kind_clause = ""
        kind_parameters: dict[str, Any] = {}
        if job_kinds is not None:
            kinds = tuple(str(kind) for kind in job_kinds)
            if kinds:
                placeholders = ", ".join(f"%(kind{index})s" for index in range(len(kinds)))
                kind_clause = f"AND j.job_kind IN ({placeholders})"
                kind_parameters = {f"kind{index}": kind for index, kind in enumerate(kinds)}
            else:
                kind_clause = "AND FALSE"
        statement = f"""
            WITH candidate AS (
                SELECT j.id FROM jobs j
                WHERE j.state IN ('QUEUED','DISPATCHED')
                  AND j.cancel_requested = FALSE
                  AND (j.lease_expires_at IS NULL OR j.lease_expires_at <= %(now)s)
                  AND (j.available_at IS NULL OR j.available_at <= %(now)s)
                  {kind_clause}
                  AND COALESCE(
                      (SELECT status FROM workers w WHERE w.worker_id = %(worker)s), 'online'
                  ) <> 'draining'
                  AND (%(max_in_flight)s <= 0 OR (
                      SELECT COUNT(*) FROM jobs x
                      WHERE x.worker_id = %(worker)s
                        AND x.state IN ('DISPATCHED','RUNNING','VALIDATING')
                  ) < %(max_in_flight)s)
                  AND {self._CAPABILITY_PREDICATE}
                ORDER BY (
                    j.priority + CASE WHEN %(aging)s > 0
                        THEN CAST((%(now)s - COALESCE(j.enqueued_at, %(now)s)) / %(aging)s AS INTEGER)
                        ELSE 0 END
                ) DESC, j.created_at ASC, j.id ASC
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            UPDATE jobs SET state = 'DISPATCHED', worker_id = %(worker)s,
                attempts = attempts + 1, dispatch_token = %(token)s,
                lease_expires_at = %(expires)s, available_at = NULL, updated_at = %(now_iso)s
            WHERE id = (SELECT id FROM candidate)
            RETURNING *
        """
        with self._lock:
            with self._conn.transaction():
                row = self._fetchone(
                    statement,
                    {
                        "worker": worker_id,
                        "token": token,
                        "expires": expires_at,
                        "now": now,
                        "now_iso": utc_now_iso(),
                        "aging": float(aging_seconds),
                        "max_in_flight": int(max_in_flight_per_worker),
                        **kind_parameters,
                    },
                )
                if row is None:
                    return None
                job = _job_from_row(row)
                self._execute(
                    "INSERT INTO leases(job_id, worker_id, expires_at, attempt, token) "
                    "VALUES (%s,%s,%s,%s,%s) ON CONFLICT (job_id) DO UPDATE SET "
                    "worker_id = EXCLUDED.worker_id, expires_at = EXCLUDED.expires_at, "
                    "attempt = EXCLUDED.attempt, token = EXCLUDED.token",
                    (job.id, worker_id, expires_at, job.attempts, token),
                )
                self._execute(
                    "INSERT INTO dispatch_records(job_id, attempt, worker_id, token, outcome, "
                    "recorded_at) VALUES (%s,%s,%s,%s,%s,%s) "
                    "ON CONFLICT (job_id, attempt) DO UPDATE SET worker_id = EXCLUDED.worker_id, "
                    "token = EXCLUDED.token, outcome = EXCLUDED.outcome, "
                    "recorded_at = EXCLUDED.recorded_at, reported_at = NULL",
                    (job.id, job.attempts, worker_id, token, "dispatched", utc_now_iso()),
                )
        self.append_event(
            job.id,
            "dispatched",
            {
                "worker_id": worker_id,
                "lease_seconds": float(lease_seconds),
                "attempt": job.attempts,
                "dispatch_token": token,
            },
        )
        return self.get_job(job.id)

    def renew_lease(
        self,
        job_id: str,
        worker_id: str,
        *,
        lease_seconds: float = 120.0,
        token: str | None = None,
        now: float | None = None,
    ) -> bool:
        now = now if now is not None else time.time()
        expires_at = now + float(lease_seconds)
        with self._lock:
            with self._conn.transaction():
                row = self._fetchone("SELECT * FROM jobs WHERE id = %s FOR UPDATE", (job_id,))
                if row is None or row["worker_id"] != worker_id:
                    return False
                if row["state"] not in ("DISPATCHED", "RUNNING", "VALIDATING"):
                    return False
                if token is not None and row["dispatch_token"] != token:
                    return False
                if (
                    row["lease_expires_at"] is not None
                    and row["lease_expires_at"] <= now
                    and token is None
                ):
                    return False
                self._execute(
                    "UPDATE jobs SET lease_expires_at = %s, updated_at = %s WHERE id = %s",
                    (expires_at, utc_now_iso(), job_id),
                )
                self._execute(
                    "UPDATE leases SET expires_at = %s WHERE job_id = %s AND worker_id = %s",
                    (expires_at, job_id, worker_id),
                )
        return True

    def lease_holder(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._fetchone("SELECT * FROM leases WHERE job_id = %s", (job_id,))
        if row is None:
            return None
        return {
            "job_id": row["job_id"],
            "worker_id": row["worker_id"],
            "expires_at": row["expires_at"],
            "attempt": row["attempt"],
            "token": row["token"],
        }

    def release_lease(self, job_id: str) -> None:
        with self._lock, self._conn.transaction():
            self._execute("DELETE FROM leases WHERE job_id = %s", (job_id,))
            self._execute(
                "UPDATE jobs SET lease_expires_at = NULL WHERE id = %s AND state IN "
                "('SUCCEEDED','FAILED','CANCELLED','QUEUED')",
                (job_id,),
            )

    def reap_expired_leases(
        self,
        *,
        now: float | None = None,
        max_attempts: int | None = None,
        retry_backoff_seconds: float = 0.0,
        retry_backoff_cap_seconds: float = 300.0,
    ) -> list[dict[str, Any]]:
        now = now if now is not None else time.time()
        with self._lock:
            rows = self._fetchall(
                "SELECT * FROM jobs WHERE lease_expires_at IS NOT NULL AND lease_expires_at <= %s "
                "AND state IN ('DISPATCHED','RUNNING','VALIDATING','CANCEL_REQUESTED') "
                "ORDER BY created_at ASC",
                (now,),
            )
        outcomes: list[dict[str, Any]] = []
        for row in rows:
            job = _job_from_row(row)
            limit = int(max_attempts or job.max_attempts or 3)
            with self._lock, self._conn.transaction():
                self._execute("DELETE FROM leases WHERE job_id = %s", (job.id,))
            outcome: dict[str, Any] = {
                "job_id": job.id,
                "worker_id": job.worker_id,
                "attempts": job.attempts,
                "max_attempts": limit,
            }
            if job.cancel_requested:
                self.transition(job.id, "CANCELLED", payload={"reason": "cancel requested while leased"})
                outcome["action"] = "cancelled"
            elif job.attempts >= limit:
                self.set_error(
                    job.id,
                    {
                        "error": {
                            "code": "job.lease_expired",
                            "message": (
                                f"worker '{job.worker_id}' stopped renewing the lease and the job "
                                f"exhausted its {limit} attempt(s)"
                            ),
                            "details": {
                                "worker_id": job.worker_id,
                                "attempts": job.attempts,
                                "max_attempts": limit,
                                "dispatch_token": job.dispatch_token,
                            },
                        },
                        "timestamp": utc_now_iso(),
                    },
                )
                self._clear_lease_fields(job.id)
                outcome["action"] = "failed"
                outcome["error"] = "job.lease_expired"
            else:
                delay = 0.0
                if retry_backoff_seconds > 0:
                    delay = min(
                        float(retry_backoff_cap_seconds),
                        float(retry_backoff_seconds) * (2 ** max(0, job.attempts - 1)),
                    )
                self.transition(
                    job.id,
                    "QUEUED",
                    payload={
                        "reason": "lease expired",
                        "worker_id": job.worker_id,
                        "attempts": job.attempts,
                        "retry_in_seconds": delay,
                    },
                )
                self._clear_lease_fields(job.id)
                if delay:
                    with self._lock, self._conn.transaction():
                        self._execute(
                            "UPDATE jobs SET available_at = %s, updated_at = %s WHERE id = %s",
                            (now + delay, utc_now_iso(), job.id),
                        )
                outcome["action"] = "requeued"
                outcome["retry_in_seconds"] = delay
            outcomes.append(outcome)
        return outcomes

    def preempt_job(
        self, job_id: str, *, reason: str = "preempted by operator", now: float | None = None
    ) -> dict[str, Any]:
        now = now if now is not None else time.time()
        job = self.get_job(job_id)
        if job.state in TERMINAL_STATES:
            return {"status": "terminal", "job_id": job_id, "state": job.state}
        if job.state not in ("DISPATCHED", "RUNNING"):
            return {"status": "not_leased", "job_id": job_id, "state": job.state}
        with self._lock, self._conn.transaction():
            self._execute(
                "UPDATE jobs SET state = 'QUEUED', worker_id = NULL, dispatch_token = NULL, "
                "lease_expires_at = NULL, available_at = %s, updated_at = %s WHERE id = %s",
                (now, utc_now_iso(), job_id),
            )
            self._execute("DELETE FROM leases WHERE job_id = %s", (job_id,))
        self.append_event(
            job_id,
            "preempted",
            {"reason": reason, "previous_worker": job.worker_id, "attempts": job.attempts},
        )
        return {
            "status": "preempted",
            "job_id": job_id,
            "state": "QUEUED",
            "previous_worker": job.worker_id,
            "attempts": job.attempts,
        }

    def set_worker_status(self, worker_id: str, status: str) -> bool:
        if status not in ("online", "draining"):
            raise ValueError(f"unsupported worker status '{status}'")
        with self._lock, self._conn.transaction():
            updated = self._execute(
                "UPDATE workers SET status = %s, last_seen = %s WHERE worker_id = %s",
                (status, utc_now_iso(), worker_id),
            )
        return updated > 0

    def _clear_lease_fields(self, job_id: str) -> None:
        with self._lock, self._conn.transaction():
            self._execute(
                "UPDATE jobs SET lease_expires_at = NULL, dispatch_token = NULL, worker_id = NULL, "
                "updated_at = %s WHERE id = %s",
                (utc_now_iso(), job_id),
            )

    def finish_from_worker(
        self,
        job_id: str,
        *,
        worker_id: str,
        token: str,
        state: str,
        result: Mapping[str, Any] | None = None,
        error: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if state not in ("SUCCEEDED", "FAILED", "CANCELLED"):
            raise JobStateError(f"worker reports must be terminal states, got '{state}'", job_id=job_id)
        previous_state = ""
        final_state = ""
        attempts = 0
        with self._lock:
            with self._conn.transaction():
                row = self._fetchone("SELECT * FROM jobs WHERE id = %s FOR UPDATE", (job_id,))
                if row is None:
                    raise JobNotFound(f"job '{job_id}' not found")
                job = _job_from_row(row)
                if job.dispatch_token != token or job.worker_id != worker_id:
                    return {
                        "status": "stale",
                        "job_id": job_id,
                        "worker_id": worker_id,
                        "detail": (
                            "dispatch token does not belong to this worker's current attempt; the "
                            "job was requeued or taken over"
                        ),
                        "current_worker": job.worker_id,
                        "current_attempt": job.attempts,
                    }
                recorded = self._fetchone(
                    "SELECT outcome FROM dispatch_records WHERE job_id = %s AND attempt = %s",
                    (job_id, job.attempts),
                )
                if recorded is None or recorded["outcome"] != "dispatched":
                    return {
                        "status": "duplicate",
                        "job_id": job_id,
                        "recorded_outcome": recorded["outcome"] if recorded else None,
                        "attempt": job.attempts,
                    }
                if job.state in TERMINAL_STATES:
                    return {
                        "status": "rejected",
                        "job_id": job_id,
                        "state": job.state,
                        "detail": "job is already terminal; the report was not applied",
                    }
                previous_state = job.state
                attempts = job.attempts
                final_state = "CANCELLED" if job.cancel_requested and state != "CANCELLED" else state
                now = utc_now_iso()
                self._execute(
                    "UPDATE dispatch_records SET outcome = %s, reported_at = %s "
                    "WHERE job_id = %s AND attempt = %s AND token = %s AND outcome = 'dispatched'",
                    (final_state, now, job_id, job.attempts, token),
                )
                if final_state == "SUCCEEDED":
                    self._execute(
                        "UPDATE jobs SET state = 'SUCCEEDED', result_json = %s, progress = 1.0, "
                        "lease_expires_at = NULL, updated_at = %s WHERE id = %s",
                        (
                            json.dumps(dict(result or {"worker_id": worker_id}), separators=(",", ":"), sort_keys=True),
                            now,
                            job_id,
                        ),
                    )
                elif final_state == "FAILED":
                    recorded_error = {
                        "error": dict(error or {"code": "job.failed", "message": "worker reported failure"}),
                        "timestamp": now,
                        "worker_id": worker_id,
                    }
                    self._execute(
                        "UPDATE jobs SET state = 'FAILED', error_json = %s, lease_expires_at = NULL, "
                        "updated_at = %s WHERE id = %s",
                        (json.dumps(recorded_error, separators=(",", ":"), sort_keys=True), now, job_id),
                    )
                else:
                    self._execute(
                        "UPDATE jobs SET state = 'CANCELLED', lease_expires_at = NULL, "
                        "updated_at = %s WHERE id = %s",
                        (now, job_id),
                    )
                self._execute("DELETE FROM leases WHERE job_id = %s", (job_id,))
        if final_state != previous_state:
            self.append_event(
                job_id,
                "state_changed",
                {"from": previous_state, "to": final_state, "worker_id": worker_id},
            )
            self.append_event(
                job_id, final_state.lower(), {"from": previous_state, "to": final_state, "worker_id": worker_id}
            )
        if final_state == "SUCCEEDED":
            self.append_event(job_id, "completed", {"state": "SUCCEEDED"})
        elif final_state == "FAILED":
            self.append_event(
                job_id,
                "failed",
                {
                    "error": dict(error or {"code": "job.failed", "message": "worker reported failure"}),
                    "timestamp": utc_now_iso(),
                    "worker_id": worker_id,
                },
            )
        payload: dict[str, Any] = {
            "status": "accepted",
            "job_id": job_id,
            "state": final_state,
            "attempt": attempts,
            "worker_id": worker_id,
        }
        if final_state == "CANCELLED" and previous_state != "CANCELLED":
            payload["detail"] = "job was cancelled before the late report could be applied"
        return payload

    def dispatch_records(self, job_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return self._fetchall(
                "SELECT * FROM dispatch_records WHERE job_id = %s ORDER BY attempt", (job_id,)
            )

    def stats(self) -> dict[str, Any]:
        with self._lock:
            states = self._fetchall("SELECT state, COUNT(*) AS n FROM jobs GROUP BY state")
            assets = self._fetchone("SELECT COUNT(*) AS n FROM artifacts_index")
            workers = self._fetchone("SELECT COUNT(*) AS n FROM workers")
        return {
            "jobs_by_state": {row["state"]: int(row["n"]) for row in states},
            "artifacts_indexed": int(assets["n"]) if assets else 0,
            "workers": int(workers["n"]) if workers else 0,
            "db": "postgresql",
        }


__all__ = ["POSTGRES_SCHEMA", "PSYCOPG_AVAILABLE", "PostgresDatabase", "_loads"]
