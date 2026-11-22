"""SQLite metadata store for the control plane.

Artifacts live on disk (content addressed). This database holds the *control* data:
jobs and their state transitions, progress events with monotonic sequence numbers,
model/artifact indexes, worker registrations and leases, and idempotency keys.

SQLite is used for the single-node control plane; the schema is written so it can be
ported to PostgreSQL without redesign (text ids, explicit timestamps, no SQLite-only
types).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from uzen.errors import IdempotencyConflict, JobNotFound, JobStateError
from uzen.util.canonical import canonical_json, sha256_text, utc_now_iso

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

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
    enqueued_at REAL,
    available_at REAL,
    dispatch_token TEXT,
    lease_expires_at REAL,
    min_cpu_count INTEGER,
    min_memory_bytes INTEGER,
    progress REAL NOT NULL DEFAULT 0.0,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    project_id TEXT NOT NULL DEFAULT 'default',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);
CREATE INDEX IF NOT EXISTS idx_jobs_kind ON jobs(job_kind);

CREATE TABLE IF NOT EXISTS job_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE,
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
    artifact_id TEXT NOT NULL,
    family TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    project_id TEXT NOT NULL DEFAULT 'default',
    created_at TEXT NOT NULL,
    UNIQUE(artifact_id)
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
    worker_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY(worker_id, kind, value),
    FOREIGN KEY(worker_id) REFERENCES workers(worker_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_worker_capabilities ON worker_capabilities(kind, value);

CREATE TABLE IF NOT EXISTS job_requirements (
    job_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    value TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'all',
    PRIMARY KEY(job_id, kind, value),
    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_job_requirements ON job_requirements(kind, value);

CREATE TABLE IF NOT EXISTS dispatch_records (
    job_id TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    worker_id TEXT NOT NULL,
    token TEXT NOT NULL,
    outcome TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    reported_at TEXT,
    PRIMARY KEY(job_id, attempt),
    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
);

-- The authoritative lease expiry is jobs.lease_expires_at: it is read and written
-- by the same statement that takes the lease, which is what makes concurrent claims
-- across processes safe. This table mirrors the current holders for inspection.
CREATE TABLE IF NOT EXISTS leases (
    job_id TEXT PRIMARY KEY,
    worker_id TEXT NOT NULL,
    expires_at REAL NOT NULL,
    attempt INTEGER NOT NULL DEFAULT 1,
    token TEXT,
    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS idempotency (
    key TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Only the SHA-256 of an API token is stored; the plaintext is returned once by
-- ``create_api_token`` and is unrecoverable afterwards.
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

-- Cross-project artifact sharing. The owning project is artifacts_index.project_id;
-- a row here grants another project read (or write) access to one artifact.
CREATE TABLE IF NOT EXISTS artifact_acl (
    artifact_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    permission TEXT NOT NULL,
    granted_at TEXT NOT NULL,
    PRIMARY KEY(artifact_id, project_id, permission)
);
"""

def _ambient_project() -> str:
    """The authenticated project, for rows created outside an explicit id."""
    try:
        from uzen.infrastructure.security.access import current_project

        return current_project()
    except Exception:  # pragma: no cover - engine without the security layer
        return "default"


JOB_STATES = (
    "CREATED",
    "QUEUED",
    "DISPATCHED",
    "RUNNING",
    "VALIDATING",
    "SUCCEEDED",
    "FAILED",
    "CANCEL_REQUESTED",
    "CANCELLED",
)

TERMINAL_STATES = {"SUCCEEDED", "FAILED", "CANCELLED"}

ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "CREATED": {"QUEUED", "CANCELLED", "FAILED"},
    "QUEUED": {"DISPATCHED", "RUNNING", "CANCEL_REQUESTED", "CANCELLED", "FAILED"},
    # QUEUED is reachable from DISPATCHED/RUNNING only through lease expiry: a worker
    # that stops heartbeating loses the job and the attempt is returned to the queue.
    "DISPATCHED": {"RUNNING", "QUEUED", "VALIDATING", "CANCEL_REQUESTED", "FAILED", "CANCELLED"},
    "RUNNING": {"VALIDATING", "QUEUED", "FAILED", "CANCEL_REQUESTED", "CANCELLED", "SUCCEEDED"},
    "VALIDATING": {"SUCCEEDED", "FAILED", "CANCEL_REQUESTED", "CANCELLED"},
    "CANCEL_REQUESTED": {"CANCELLED", "FAILED"},
    "SUCCEEDED": set(),
    "FAILED": set(),
    "CANCELLED": set(),
}


@dataclass
class JobRow:
    id: str
    state: str
    task_scope: str
    objective: str
    request: dict[str, Any]
    plan: dict[str, Any] | None
    result: dict[str, Any] | None
    error: dict[str, Any] | None
    progress: float
    cancel_requested: bool
    created_at: str
    updated_at: str
    worker_id: str | None = None
    attempts: int = 0
    max_attempts: int = 3
    dispatch_token: str | None = None
    lease_expires_at: float | None = None
    min_cpu_count: int | None = None
    min_memory_bytes: int | None = None
    priority: int = 0
    job_kind: str = ""
    enqueued_at: float | None = None
    available_at: float | None = None
    project_id: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state,
            "task_scope": self.task_scope,
            "objective": self.objective,
            "request": self.request,
            "plan": self.plan,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
            "cancel_requested": self.cancel_requested,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "worker_id": self.worker_id,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "dispatch_token": self.dispatch_token,
            "lease_expires_at": self.lease_expires_at,
            "min_cpu_count": self.min_cpu_count,
            "min_memory_bytes": self.min_memory_bytes,
            "priority": self.priority,
            "job_kind": self.job_kind,
            "enqueued_at": self.enqueued_at,
            "available_at": self.available_at,
            "project_id": self.project_id,
        }


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._migrate()
            self._conn.commit()

    def _migrate(self) -> None:
        """Add columns introduced after a database was first created.

        ``CREATE TABLE IF NOT EXISTS`` never alters an existing table, so a control
        plane upgraded from an earlier build would otherwise fail its first claim. The
        added columns are all nullable or defaulted, so an in-place ALTER is safe.
        """
        additions = {
            "jobs": {
                "attempts": "INTEGER NOT NULL DEFAULT 0",
                "max_attempts": "INTEGER NOT NULL DEFAULT 3",
                "dispatch_token": "TEXT",
                "lease_expires_at": "REAL",
                "min_cpu_count": "INTEGER",
                "min_memory_bytes": "INTEGER",
                "priority": "INTEGER NOT NULL DEFAULT 0",
                "job_kind": "TEXT NOT NULL DEFAULT ''",
                "enqueued_at": "REAL",
                "available_at": "REAL",
                "project_id": "TEXT NOT NULL DEFAULT 'default'",
            },
            "artifacts_index": {
                "project_id": "TEXT NOT NULL DEFAULT 'default'",
            },
            "models": {
                "project_id": "TEXT NOT NULL DEFAULT 'default'",
            },
            "workers": {
                "cpu_count": "INTEGER",
                "memory_bytes": "INTEGER",
            },
            "leases": {
                "attempt": "INTEGER NOT NULL DEFAULT 1",
                "token": "TEXT",
            },
        }
        for table, columns in additions.items():
            existing = {
                row["name"] for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for name, declaration in columns.items():
                if name not in existing:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")

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
        import time as _time

        job_id = job_id or str(uuid.uuid4())
        now = utc_now_iso()
        # Priority is a small operator-supplied bias, not a deadline. Clamping keeps a
        # typo (1e9) from starving the queue indefinitely; aging still applies on top.
        priority = max(-1000, min(1000, int(priority)))
        request_json = json.dumps(request, separators=(",", ":"), sort_keys=True)
        request_hash = sha256_text(request_json)
        options = request.get("options") if isinstance(request, Mapping) else None
        job_kind = ""
        if isinstance(options, Mapping):
            job_kind = str(options.get("task_kind") or "")
        with self._lock:
            if idempotency_key:
                existing = self._conn.execute(
                    "SELECT * FROM idempotency WHERE key = ?", (idempotency_key,)
                ).fetchone()
                if existing:
                    if existing["request_hash"] != request_hash:
                        raise IdempotencyConflict(
                            "idempotency key was reused with a different request body",
                            key=idempotency_key,
                            existing_job=existing["job_id"],
                        )
                    row = self._conn.execute(
                        "SELECT * FROM jobs WHERE id = ?", (existing["job_id"],)
                    ).fetchone()
                    return _row_to_job(row)
                self._conn.execute(
                    "INSERT INTO idempotency(key, job_id, request_hash, created_at) VALUES (?,?,?,?)",
                    (idempotency_key, job_id, request_hash, now),
                )
            self._conn.execute(
                "INSERT INTO jobs(id, state, task_scope, objective, request_json, idempotency_key, "
                "request_hash, max_attempts, priority, job_kind, enqueued_at, min_cpu_count, "
                "min_memory_bytes, project_id, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
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
                    _time.time(),
                    getattr(requirements, "min_cpu_count", None) or None,
                    getattr(requirements, "min_memory_bytes", None),
                    str(project_id or "default"),
                    now,
                    now,
                ),
            )
            if requirements is not None:
                self._replace_job_requirements(job_id, requirements)
            self._conn.commit()
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        self.append_event(job_id, "created", {"state": state, "task_scope": request.get("task_scope")})
        return _row_to_job(row)

    def get_job(self, job_id: str) -> JobRow:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise JobNotFound(f"job '{job_id}' not found")
        return _row_to_job(row)

    def list_jobs(
        self,
        *,
        state: str | None = None,
        job_kind: str | None = None,
        project_id: str | None = None,
        before: tuple[str, str] | None = None,
        limit: int = 50,
    ) -> list[JobRow]:
        query = "SELECT * FROM jobs"
        params: list[Any] = []
        clauses: list[str] = []
        if state:
            clauses.append("state = ?")
            params.append(state)
        if job_kind is not None:
            clauses.append("job_kind = ?")
            params.append(job_kind)
        if project_id is not None:
            clauses.append("project_id = ?")
            params.append(str(project_id))
        if before is not None:
            # Keyset pagination on the (created_at, id) sort key: stable while rows are
            # inserted, unlike OFFSET, and it uses the created_at index.
            clauses.append("(created_at < ? OR (created_at = ? AND id < ?))")
            params.extend([before[0], before[0], before[1]])
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [_row_to_job(r) for r in rows]

    def project_of_job(self, job_id: str) -> str:
        with self._lock:
            row = self._conn.execute(
                "SELECT project_id FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        if row is None:
            raise JobNotFound(f"job '{job_id}' not found")
        return str(row["project_id"] or "default")

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
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise JobNotFound(f"job '{job_id}' not found")
            current = row["state"]
            if not force and new_state != current and new_state not in ALLOWED_TRANSITIONS.get(current, set()):
                raise JobStateError(
                    f"illegal job transition {current} -> {new_state}", job_id=job_id
                )
            now = utc_now_iso()
            updates = ["state = ?", "updated_at = ?"]
            params: list[Any] = [new_state, now]
            if progress is not None:
                updates.append("progress = ?")
                params.append(float(progress))
            params.append(job_id)
            self._conn.execute(f"UPDATE jobs SET {', '.join(updates)} WHERE id = ?", params)
            self._conn.commit()
        self.append_event(job_id, "state_changed", {"from": current, "to": new_state, **(payload or {})})
        if new_state != current:
            # State-named event so subscribers can stream lifecycle progress without
            # decoding every state_changed payload (CREATED -> queued -> running ...).
            self.append_event(
                job_id,
                new_state.lower(),
                {"from": current, "to": new_state, **(payload or {})},
            )
        return self.get_job(job_id)

    def set_plan(self, job_id: str, plan: Mapping[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET plan_json = ?, updated_at = ? WHERE id = ?",
                (json.dumps(plan, separators=(",", ":"), sort_keys=True), utc_now_iso(), job_id),
            )
            self._conn.commit()

    def set_result(self, job_id: str, result: Mapping[str, Any], *, state: str = "SUCCEEDED") -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET result_json = ?, state = ?, progress = 1.0, updated_at = ? WHERE id = ?",
                (json.dumps(result, separators=(",", ":"), sort_keys=True), state, utc_now_iso(), job_id),
            )
            self._conn.commit()
        self.append_event(job_id, "completed", {"state": state})

    def set_error(self, job_id: str, error: Mapping[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET error_json = ?, state = 'FAILED', updated_at = ? WHERE id = ?",
                (json.dumps(error, separators=(",", ":"), sort_keys=True), utc_now_iso(), job_id),
            )
            self._conn.commit()
        self.append_event(job_id, "failed", dict(error))

    def request_cancel(self, job_id: str) -> JobRow:
        """Durably request cancellation without reopening a terminal job.

        A worker can settle a job on another connection at the same moment. The read
        and write therefore happen under one ``BEGIN IMMEDIATE`` transaction: checking
        terminal state first with a plain SELECT lets a concurrent success commit in
        between, after which this update would overwrite ``SUCCEEDED`` with
        ``CANCEL_REQUESTED``. Queued jobs become terminal immediately; a leased job is
        flagged so its worker (or the lease reaper) settles it.
        """
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
                if row is None:
                    raise JobNotFound(f"job '{job_id}' not found")
                previous = _row_to_job(row)
                if previous.state in TERMINAL_STATES:
                    self._conn.rollback()
                    return previous
                state = (
                    "CANCELLED"
                    if previous.state in ("CREATED", "QUEUED")
                    else "CANCEL_REQUESTED"
                )
                self._conn.execute(
                    "UPDATE jobs SET cancel_requested = 1, state = ?, updated_at = ? WHERE id = ?",
                    (state, utc_now_iso(), job_id),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        self.append_event(job_id, "cancel_requested", {})
        if state == "CANCELLED":
            self.append_event(
                job_id,
                "state_changed",
                {"from": previous.state, "to": "CANCELLED"},
            )
            self.append_event(
                job_id,
                "cancelled",
                {"from": previous.state, "to": "CANCELLED"},
            )
        return self.get_job(job_id)

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return bool(row and row["cancel_requested"])

    # ------------------------------------------------------------------ #
    # events
    # ------------------------------------------------------------------ #

    def append_event(self, job_id: str, kind: str, payload: Mapping[str, Any]) -> None:
        # ``seq`` is per job. A plain SELECT MAX + INSERT races when a control-plane
        # process and a worker append to the same job through different connections;
        # BEGIN IMMEDIATE takes the writer lock before reading the sequence.
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT COALESCE(MAX(seq), 0) AS m FROM job_events WHERE job_id = ?", (job_id,)
                ).fetchone()
                seq = int(row["m"]) + 1
                self._conn.execute(
                    "INSERT INTO job_events(job_id, seq, kind, payload_json, created_at) VALUES (?,?,?,?,?)",
                    (job_id, seq, kind, json.dumps(payload, separators=(",", ":"), sort_keys=True), utc_now_iso()),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def events(self, job_id: str, *, after: int = 0, limit: int = 500) -> list[dict[str, Any]]:
        self.get_job(job_id)
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM job_events WHERE job_id = ? AND seq > ? ORDER BY seq ASC LIMIT ?",
                (job_id, int(after), int(limit)),
            ).fetchall()
        return [
            {
                "seq": r["seq"],
                "kind": r["kind"],
                "payload": json.loads(r["payload_json"]),
                "created_at": r["created_at"],
            }
            for r in rows
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
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO artifacts_index(artifact_id, name, family, operation, path, "
                "numel, num_tensors, job_id, project_id, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
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
            self._conn.commit()

    def artifacts(
        self,
        limit: int = 100,
        *,
        project_id: str | None = None,
        before: tuple[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM artifacts_index"
        params: list[Any] = []
        clauses: list[str] = []
        if project_id is not None:
            clauses.append("project_id = ?")
            params.append(str(project_id))
        if before is not None:
            clauses.append("(created_at < ? OR (created_at = ? AND artifact_id < ?))")
            params.extend([before[0], before[0], before[1]])
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC, artifact_id DESC LIMIT ?"
        params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def artifact_project(self, artifact_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT project_id FROM artifacts_index WHERE artifact_id = ?", (artifact_id,)
            ).fetchone()
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
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO artifact_acl(artifact_id, project_id, permission, granted_at) "
                "VALUES (?,?,?,?)",
                (artifact_id, str(project_id), permission, grant["granted_at"]),
            )
            self._conn.commit()
        return grant

    def artifact_acls(self, artifact_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT artifact_id, project_id, permission, granted_at FROM artifact_acl "
                "WHERE artifact_id = ? ORDER BY project_id, permission",
                (artifact_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def revoke_artifact_access(self, artifact_id: str, project_id: str) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM artifact_acl WHERE artifact_id = ? AND project_id = ?",
                (artifact_id, str(project_id)),
            )
            self._conn.commit()
        return int(cursor.rowcount)

    # ------------------------------------------------------------------ #
    # api tokens (project scoped)
    # ------------------------------------------------------------------ #

    def create_api_token(
        self,
        project_id: str,
        scopes: Iterable[str],
        *,
        description: str | None = None,
        token_id: str | None = None,
    ) -> dict[str, Any]:
        """Mint a token; the plaintext secret is returned once and never stored."""
        import secrets as _secrets

        token_id = token_id or f"tok_{uuid.uuid4().hex[:16]}"
        secret = f"uzt_{_secrets.token_urlsafe(32)}"
        scope_list = sorted({str(scope) for scope in scopes})
        now = utc_now_iso()
        with self._lock:
            self._conn.execute(
                "INSERT INTO api_tokens(token_id, project_id, token_hash, scopes_json, description, "
                "created_at) VALUES (?,?,?,?,?,?)",
                (token_id, str(project_id), sha256_text(secret), canonical_json(scope_list), description, now),
            )
            self._conn.commit()
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
            row = self._conn.execute(
                "SELECT * FROM api_tokens WHERE token_hash = ?", (sha256_text(secret),)
            ).fetchone()
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
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE api_tokens SET revoked_at = ? WHERE token_id = ? AND revoked_at IS NULL",
                (utc_now_iso(), token_id),
            )
            self._conn.commit()
        return cursor.rowcount > 0

    def list_api_tokens(self, *, project_id: str | None = None) -> list[dict[str, Any]]:
        query = (
            "SELECT token_id, project_id, scopes_json, description, created_at, revoked_at "
            "FROM api_tokens"
        )
        params: list[Any] = []
        if project_id is not None:
            query += " WHERE project_id = ?"
            params.append(str(project_id))
        query += " ORDER BY created_at DESC"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
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

    def register_model(
        self, *, artifact_id: str, family: str, spec: Mapping[str, Any], project_id: str | None = None
    ) -> str:
        model_id = f"model_{sha256_text(artifact_id)[:16]}"
        if project_id is None:
            project_id = self.artifact_project(artifact_id) or _ambient_project()
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO models(model_id, artifact_id, family, spec_json, project_id, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (model_id, artifact_id, family, canonical_json(dict(spec)), str(project_id), utc_now_iso()),
            )
            self._conn.commit()
        return model_id

    def models(
        self,
        limit: int = 100,
        *,
        project_id: str | None = None,
        before: tuple[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM models"
        params: list[Any] = []
        clauses: list[str] = []
        if project_id is not None:
            clauses.append("project_id = ?")
            params.append(str(project_id))
        if before is not None:
            clauses.append("(created_at < ? OR (created_at = ? AND model_id < ?))")
            params.extend([before[0], before[0], before[1]])
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC, model_id DESC LIMIT ?"
        params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [
            {
                "model_id": r["model_id"],
                "artifact_id": r["artifact_id"],
                "family": r["family"],
                "spec": json.loads(r["spec_json"]),
                "project_id": r["project_id"],
                "created_at": r["created_at"],
            }
            for r in rows
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
            self._conn.execute(
                "INSERT INTO workers(worker_id, protocol_version, capabilities_json, token_hash, status, "
                "last_seen, registered_at) VALUES (?,?,?,?,?,?,?) "
                "ON CONFLICT(worker_id) DO UPDATE SET capabilities_json=excluded.capabilities_json, "
                "token_hash=excluded.token_hash, last_seen=excluded.last_seen, "
                "status=CASE WHEN workers.status='draining' THEN 'draining' ELSE 'online' END",
                (worker_id, protocol_version, canonical_json(dict(capabilities)), token_hash, "online", now, now),
            )
            self.set_worker_capabilities(worker_id, capabilities)
            self._conn.commit()

    def worker_token_hash(self, worker_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT token_hash FROM workers WHERE worker_id = ?", (worker_id,)
            ).fetchone()
        return None if row is None else str(row["token_hash"])

    def heartbeat(self, worker_id: str, status: str | None = None) -> None:
        """Record liveness; only an explicit status changes the routing state."""
        now = utc_now_iso()
        with self._lock:
            if status is None:
                self._conn.execute(
                    "UPDATE workers SET last_seen = ? WHERE worker_id = ?", (now, worker_id)
                )
            else:
                self._conn.execute(
                    "UPDATE workers SET last_seen = ?, status = ? WHERE worker_id = ?",
                    (now, status, worker_id),
                )
            self._conn.commit()

    def set_worker_capabilities(self, worker_id: str, capabilities: Mapping[str, Any]) -> None:
        """Store a worker's advertised capabilities as matchable (kind, value) rows.

        The columns and rows are derived from the capability document through
        :func:`uzen.jobs.scheduling.capability_index`, so the SQL claim predicate and
        the Python explanation can never disagree about what a worker can do.
        """
        from uzen.jobs.scheduling import capability_index

        index = capability_index(dict(capabilities))
        with self._lock:
            # A worker that claims through the database directly (single-process embed,
            # tests, a pre-registered pool) has no credential row yet. It gets a
            # placeholder whose empty token hash can never authenticate over HTTP.
            self._conn.execute(
                "INSERT OR IGNORE INTO workers(worker_id, protocol_version, capabilities_json, "
                "token_hash, status, last_seen, registered_at) VALUES (?,?,?,?,?,?,?)",
                (worker_id, "uzen.worker/1", "{}", "", "unregistered", utc_now_iso(), utc_now_iso()),
            )
            self._conn.execute(
                "UPDATE workers SET capabilities_json = ?, cpu_count = ?, memory_bytes = ? "
                "WHERE worker_id = ?",
                (
                    canonical_json(dict(capabilities)),
                    int(capabilities["cpu_count"]) if capabilities.get("cpu_count") is not None else None,
                    int(capabilities["memory_bytes"]) if capabilities.get("memory_bytes") is not None else None,
                    worker_id,
                ),
            )
            self._conn.execute("DELETE FROM worker_capabilities WHERE worker_id = ?", (worker_id,))
            for kind, values in index.items():
                for value in values:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO worker_capabilities(worker_id, kind, value) VALUES (?,?,?)",
                        (worker_id, kind, value),
                    )
            self._conn.commit()

    def worker_capabilities(self, worker_id: str) -> dict[str, list[str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT kind, value FROM worker_capabilities WHERE worker_id = ? ORDER BY kind, value",
                (worker_id,),
            ).fetchall()
        out: dict[str, list[str]] = {}
        for row in rows:
            out.setdefault(row["kind"], []).append(row["value"])
        return out

    def _replace_job_requirements(self, job_id: str, requirements: Any) -> None:
        self._conn.execute("DELETE FROM job_requirements WHERE job_id = ?", (job_id,))
        for kind, value, mode in requirements.rows():
            self._conn.execute(
                "INSERT OR IGNORE INTO job_requirements(job_id, kind, value, mode) VALUES (?,?,?,?)",
                (job_id, kind, value, mode),
            )

    def set_job_requirements(self, job_id: str, requirements: Any) -> None:
        with self._lock:
            self._replace_job_requirements(job_id, requirements)
            self._conn.execute(
                "UPDATE jobs SET min_cpu_count = ?, min_memory_bytes = ?, updated_at = ? WHERE id = ?",
                (
                    getattr(requirements, "min_cpu_count", None) or None,
                    getattr(requirements, "min_memory_bytes", None),
                    utc_now_iso(),
                    job_id,
                ),
            )
            self._conn.commit()

    def job_requirements(self, job_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT kind, value, mode FROM job_requirements WHERE job_id = ? ORDER BY kind, value",
                (job_id,),
            ).fetchall()
        return [{"kind": r["kind"], "value": r["value"], "mode": r["mode"]} for r in rows]

    def workers(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM workers ORDER BY registered_at DESC").fetchall()
        return [
            {
                "worker_id": r["worker_id"],
                "protocol_version": r["protocol_version"],
                "capabilities": json.loads(r["capabilities_json"]),
                "status": r["status"],
                "last_seen": r["last_seen"],
                "registered_at": r["registered_at"],
                "cpu_count": r["cpu_count"] if "cpu_count" in r.keys() else None,
                "memory_bytes": r["memory_bytes"] if "memory_bytes" in r.keys() else None,
            }
            for r in rows
        ]

    #: SQL predicate (over alias ``j`` for the jobs row, and ``:worker`` taken from the
    #: binding) that is true only when the worker advertised every capability the job
    #: recorded. It runs inside the claiming UPDATE, so the capability decision and the
    #: lease acquisition are one atomic step.
    _CAPABILITY_PREDICATE = """
        NOT EXISTS (
            SELECT 1 FROM job_requirements r
            WHERE r.job_id = j.id AND r.mode = 'all'
              AND NOT EXISTS (
                  SELECT 1 FROM worker_capabilities c
                  WHERE c.worker_id = :worker AND c.kind = r.kind AND c.value = r.value
              )
        )
        AND NOT EXISTS (
            SELECT 1 FROM job_requirements a
            WHERE a.job_id = j.id AND a.mode = 'any'
              AND NOT EXISTS (
                  SELECT 1 FROM worker_capabilities c
                  WHERE c.worker_id = :worker AND c.kind = a.kind
                    AND c.value IN (
                        SELECT value FROM job_requirements
                        WHERE job_id = j.id AND kind = a.kind AND mode = 'any'
                    )
              )
        )
        AND (j.min_cpu_count IS NULL OR j.min_cpu_count <= COALESCE(
            (SELECT cpu_count FROM workers WHERE worker_id = :worker), 0))
        AND (j.min_memory_bytes IS NULL OR j.min_memory_bytes <= COALESCE(
            (SELECT memory_bytes FROM workers WHERE worker_id = :worker), 0))
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
        """Atomically lease the oldest job this worker is capable of executing.

        The whole decision is a single ``UPDATE`` whose sub-select applies the
        capability predicate and skips jobs with a live lease. SQLite serializes
        writers, so when several control-plane processes race on the same database
        exactly one of them can match a given job: there is no read-then-write window
        in which two workers can receive the same row. Each claim increments
        ``attempts`` and installs a fresh ``dispatch_token``; every later report must
        present that token, so a worker whose lease expired cannot publish a result
        over the worker that replaced it.
        """
        import time as _time

        if capabilities is not None:
            self.set_worker_capabilities(worker_id, capabilities)
        now = now if now is not None else _time.time()
        token = uuid.uuid4().hex
        expires_at = now + float(lease_seconds)
        kind_clause = ""
        kind_parameters: dict[str, Any] = {}
        if job_kinds is not None:
            kinds = tuple(str(kind) for kind in job_kinds)
            if kinds:
                placeholders = ", ".join(f":kind{index}" for index in range(len(kinds)))
                kind_clause = f"AND j.job_kind IN ({placeholders})"
                kind_parameters = {
                    f"kind{index}": kind for index, kind in enumerate(kinds)
                }
            else:
                kind_clause = "AND 0"
        statement = f"""
            UPDATE jobs SET state = 'DISPATCHED', worker_id = :worker, attempts = attempts + 1,
                dispatch_token = :token, lease_expires_at = :expires, available_at = NULL,
                updated_at = :now_iso
            WHERE id = (
                SELECT j.id FROM jobs j
                WHERE j.state IN ('QUEUED','DISPATCHED')
                  AND j.cancel_requested = 0
                  AND (j.lease_expires_at IS NULL OR j.lease_expires_at <= :now)
                  AND (j.available_at IS NULL OR j.available_at <= :now)
                  {kind_clause}
                  AND COALESCE(
                      (SELECT status FROM workers w WHERE w.worker_id = :worker), 'online'
                  ) != 'draining'
                  AND (:max_in_flight <= 0 OR (
                      SELECT COUNT(*) FROM jobs x
                      WHERE x.worker_id = :worker
                        AND x.state IN ('DISPATCHED','RUNNING','VALIDATING')
                  ) < :max_in_flight)
                  AND {self._CAPABILITY_PREDICATE}
                ORDER BY (
                    j.priority + CASE WHEN :aging > 0
                        THEN CAST((:now - COALESCE(j.enqueued_at, :now)) / :aging AS INTEGER)
                        ELSE 0 END
                ) DESC, j.created_at ASC, j.id ASC
                LIMIT 1
            )
            AND state IN ('QUEUED','DISPATCHED')
        """
        with self._lock:
            cursor = self._conn.execute(
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
            self._conn.commit()
            if cursor.rowcount == 0:
                return None
            row = self._conn.execute("SELECT * FROM jobs WHERE dispatch_token = ?", (token,)).fetchone()
            if row is None:  # pragma: no cover - the UPDATE just wrote this token
                return None
            job = _row_to_job(row)
            self._conn.execute(
                "INSERT OR REPLACE INTO leases(job_id, worker_id, expires_at, attempt, token) VALUES (?,?,?,?,?)",
                (job.id, worker_id, expires_at, job.attempts, token),
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO dispatch_records(job_id, attempt, worker_id, token, outcome, "
                "recorded_at) VALUES (?,?,?,?,?,?)",
                (job.id, job.attempts, worker_id, token, "dispatched", utc_now_iso()),
            )
            self._conn.commit()
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
        """Extend a lease; false if the caller no longer holds it (expired or replaced)."""
        import time as _time

        now = now if now is not None else _time.time()
        expires_at = now + float(lease_seconds)
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None or row["worker_id"] != worker_id:
                return False
            if row["state"] not in ("DISPATCHED", "RUNNING", "VALIDATING"):
                return False
            if token is not None and row["dispatch_token"] != token:
                return False
            if row["lease_expires_at"] is not None and row["lease_expires_at"] <= now and token is None:
                return False
            self._conn.execute(
                "UPDATE jobs SET lease_expires_at = ?, updated_at = ? WHERE id = ?",
                (expires_at, utc_now_iso(), job_id),
            )
            self._conn.execute(
                "UPDATE leases SET expires_at = ? WHERE job_id = ? AND worker_id = ?",
                (expires_at, job_id, worker_id),
            )
            self._conn.commit()
        return True

    def lease_holder(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM leases WHERE job_id = ?", (job_id,)).fetchone()
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
        with self._lock:
            self._conn.execute("DELETE FROM leases WHERE job_id = ?", (job_id,))
            self._conn.execute(
                "UPDATE jobs SET lease_expires_at = NULL WHERE id = ? AND state IN "
                "('SUCCEEDED','FAILED','CANCELLED','QUEUED')",
                (job_id,),
            )
            self._conn.commit()

    def reap_expired_leases(
        self,
        *,
        now: float | None = None,
        max_attempts: int | None = None,
        retry_backoff_seconds: float = 0.0,
        retry_backoff_cap_seconds: float = 300.0,
    ) -> list[dict[str, Any]]:
        """Return jobs whose lease expired to the queue, or fail them out of attempts.

        A worker that dies mid-job never reports. Its lease simply stops being renewed,
        and this reaper (called on every claim and by the scheduler) turns the expired
        lease into either another attempt or a terminal ``FAILED`` carrying
        ``job.lease_expired``. Nothing is silently requeued forever.
        """
        import time as _time

        now = now if now is not None else _time.time()
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE lease_expires_at IS NOT NULL AND lease_expires_at <= ? "
                "AND state IN ('DISPATCHED','RUNNING','VALIDATING','CANCEL_REQUESTED') "
                "ORDER BY created_at ASC",
                (now,),
            ).fetchall()
        outcomes: list[dict[str, Any]] = []
        for row in rows:
            job = _row_to_job(row)
            limit = int(max_attempts or job.max_attempts or 3)
            with self._lock:
                self._conn.execute("DELETE FROM leases WHERE job_id = ?", (job.id,))
                self._conn.commit()
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
                # Exponential backoff from the attempt that just failed, capped so a
                # long-lived queue cannot grow the delay without bound. The delay is
                # stored on the row, so the next claim (possibly on another host) sees
                # the same rule instead of retrying immediately.
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
                    with self._lock:
                        self._conn.execute(
                            "UPDATE jobs SET available_at = ?, updated_at = ? WHERE id = ?",
                            (now + delay, utc_now_iso(), job.id),
                        )
                        self._conn.commit()
                outcome["action"] = "requeued"
                outcome["retry_in_seconds"] = delay
            outcomes.append(outcome)
        return outcomes

    def preempt_job(
        self,
        job_id: str,
        *,
        reason: str = "preempted by operator",
        now: float | None = None,
    ) -> dict[str, Any]:
        """Return a leased job to the queue and revoke the current worker's token.

        Preemption is cooperative: the row stops being renewable immediately and is
        claimable at once, but a worker already inside a kernel only notices when it
        next renews or reports. Its report is then answered ``stale``, so it cannot
        overwrite the attempt that replaces it.
        """
        import time as _time

        now = now if now is not None else _time.time()
        job = self.get_job(job_id)
        if job.state in TERMINAL_STATES:
            return {"status": "terminal", "job_id": job_id, "state": job.state}
        if job.state not in ("DISPATCHED", "RUNNING"):
            return {"status": "not_leased", "job_id": job_id, "state": job.state}
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET state = 'QUEUED', worker_id = NULL, dispatch_token = NULL, "
                "lease_expires_at = NULL, available_at = ?, updated_at = ? WHERE id = ?",
                (now, utc_now_iso(), job_id),
            )
            self._conn.execute("DELETE FROM leases WHERE job_id = ?", (job_id,))
            self._conn.commit()
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
        """Mark a worker online or draining; a draining worker cannot claim."""
        if status not in ("online", "draining"):
            raise ValueError(f"unsupported worker status '{status}'")
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE workers SET status = ?, last_seen = ? WHERE worker_id = ?",
                (status, utc_now_iso(), worker_id),
            )
            self._conn.commit()
        return cursor.rowcount > 0

    def _clear_lease_fields(self, job_id: str) -> None:
        """Drop the lease grant so a replacement attempt gets a fresh token."""
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET lease_expires_at = NULL, dispatch_token = NULL, worker_id = NULL, "
                "updated_at = ? WHERE id = ?",
                (utc_now_iso(), job_id),
            )
            self._conn.commit()

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
        """Apply one worker report exactly once.

        Returns ``status``: ``accepted`` (this call settled the job), ``duplicate``
        (an identical report was already applied — the recorded outcome is returned and
        nothing changes), ``stale`` (the token belongs to a superseded attempt, so the
        report is refused rather than overwriting the current holder) or ``rejected``
        (the job is already terminal for another reason).
        """
        if state not in ("SUCCEEDED", "FAILED", "CANCELLED"):
            raise JobStateError(f"worker reports must be terminal states, got '{state}'", job_id=job_id)
        # Settlement and cancellation race on separate connections. One immediate
        # transaction re-reads the job, decides the outcome and closes the dispatch
        # record together, so a cancellation that commits first wins and a late report
        # can never overwrite it (nor can a report reopen a terminal job).
        previous_state = ""
        final_state = ""
        attempts = 0
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
                if row is None:
                    raise JobNotFound(f"job '{job_id}' not found")
                job = _row_to_job(row)
                if job.dispatch_token != token or job.worker_id != worker_id:
                    self._conn.rollback()
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
                recorded = self._conn.execute(
                    "SELECT outcome FROM dispatch_records WHERE job_id = ? AND attempt = ?",
                    (job_id, job.attempts),
                ).fetchone()
                if recorded is None or recorded["outcome"] != "dispatched":
                    self._conn.rollback()
                    return {
                        "status": "duplicate",
                        "job_id": job_id,
                        "recorded_outcome": recorded["outcome"] if recorded else None,
                        "attempt": job.attempts,
                    }
                if job.state in TERMINAL_STATES:
                    self._conn.rollback()
                    return {
                        "status": "rejected",
                        "job_id": job_id,
                        "state": job.state,
                        "detail": "job is already terminal; the report was not applied",
                    }
                previous_state = job.state
                attempts = job.attempts
                final_state = (
                    "CANCELLED"
                    if job.cancel_requested and state != "CANCELLED"
                    else state
                )
                now = utc_now_iso()
                self._conn.execute(
                    "UPDATE dispatch_records SET outcome = ?, reported_at = ? "
                    "WHERE job_id = ? AND attempt = ? AND token = ? AND outcome = 'dispatched'",
                    (final_state, now, job_id, job.attempts, token),
                )
                if final_state == "SUCCEEDED":
                    self._conn.execute(
                        "UPDATE jobs SET state = 'SUCCEEDED', result_json = ?, progress = 1.0, "
                        "lease_expires_at = NULL, updated_at = ? WHERE id = ?",
                        (
                            json.dumps(
                                dict(result or {"worker_id": worker_id}),
                                separators=(",", ":"),
                                sort_keys=True,
                            ),
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
                    self._conn.execute(
                        "UPDATE jobs SET state = 'FAILED', error_json = ?, lease_expires_at = NULL, "
                        "updated_at = ? WHERE id = ?",
                        (
                            json.dumps(recorded_error, separators=(",", ":"), sort_keys=True),
                            now,
                            job_id,
                        ),
                    )
                else:
                    self._conn.execute(
                        "UPDATE jobs SET state = 'CANCELLED', lease_expires_at = NULL, "
                        "updated_at = ? WHERE id = ?",
                        (now, job_id),
                    )
                self._conn.execute("DELETE FROM leases WHERE job_id = ?", (job_id,))
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        if final_state != previous_state:
            self.append_event(
                job_id,
                "state_changed",
                {"from": previous_state, "to": final_state, "worker_id": worker_id},
            )
            self.append_event(
                job_id,
                final_state.lower(),
                {"from": previous_state, "to": final_state, "worker_id": worker_id},
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
        result_payload: dict[str, Any] = {
            "status": "accepted",
            "job_id": job_id,
            "state": final_state,
            "attempt": attempts,
            "worker_id": worker_id,
        }
        if final_state == "CANCELLED" and previous_state != "CANCELLED":
            result_payload["detail"] = "job was cancelled before the late report could be applied"
        return result_payload

    def dispatch_records(self, job_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM dispatch_records WHERE job_id = ? ORDER BY attempt", (job_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            jobs = {row["state"]: row["n"] for row in self._conn.execute(
                "SELECT state, COUNT(*) AS n FROM jobs GROUP BY state"
            ).fetchall()}
            assets = self._conn.execute("SELECT COUNT(*) AS n FROM artifacts_index").fetchone()["n"]
            workers = self._conn.execute("SELECT COUNT(*) AS n FROM workers").fetchone()["n"]
        return {"jobs_by_state": jobs, "artifacts_indexed": assets, "workers": workers, "db": str(self.path)}


def _row_to_job(row: sqlite3.Row) -> JobRow:
    return JobRow(
        id=row["id"],
        state=row["state"],
        task_scope=row["task_scope"],
        objective=row["objective"],
        request=json.loads(row["request_json"]),
        plan=json.loads(row["plan_json"]) if row["plan_json"] else None,
        result=json.loads(row["result_json"]) if row["result_json"] else None,
        error=json.loads(row["error_json"]) if row["error_json"] else None,
        progress=float(row["progress"]),
        cancel_requested=bool(row["cancel_requested"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        worker_id=row["worker_id"],
        attempts=int(row["attempts"] or 0),
        max_attempts=int(row["max_attempts"] or 3),
        dispatch_token=row["dispatch_token"],
        lease_expires_at=row["lease_expires_at"],
        min_cpu_count=row["min_cpu_count"],
        min_memory_bytes=row["min_memory_bytes"],
        priority=int(row["priority"] or 0) if "priority" in row.keys() else 0,
        job_kind=str(row["job_kind"] or "") if "job_kind" in row.keys() else "",
        enqueued_at=row["enqueued_at"] if "enqueued_at" in row.keys() else None,
        available_at=row["available_at"] if "available_at" in row.keys() else None,
        project_id=str(row["project_id"] or "default") if "project_id" in row.keys() else "default",
    )
