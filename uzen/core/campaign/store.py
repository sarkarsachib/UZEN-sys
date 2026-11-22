"""Durable campaign state in SQLite.

The store is intentionally simple and local: one database per workspace, WAL mode,
one short-lived connection per operation. Campaign runs are idempotent at trial
granularity, so a process killed between batches can resume without re-evaluating
or duplicating completed candidates.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from uzen.core.campaign.models import Candidate, Trial
from uzen.core.campaign.spec import CampaignSpec
from uzen.errors import CampaignConflict, CampaignNotFound
from uzen.util.canonical import canonical_json, utc_now_iso

_SCHEMA = """
CREATE TABLE IF NOT EXISTS campaigns (
    campaign_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    mode TEXT NOT NULL,
    spec_digest TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    status TEXT NOT NULL,
    stop_reason TEXT NOT NULL DEFAULT '',
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    run_job_id TEXT NOT NULL DEFAULT '',
    seed INTEGER NOT NULL,
    budget INTEGER NOT NULL,
    rounds INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trials (
    campaign_id TEXT NOT NULL,
    digest TEXT NOT NULL,
    round_index INTEGER NOT NULL,
    sequence INTEGER NOT NULL,
    candidate_json TEXT NOT NULL,
    status TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    error TEXT NOT NULL DEFAULT '',
    failures_json TEXT NOT NULL DEFAULT '[]',
    package_fingerprint TEXT NOT NULL DEFAULT '',
    graph_digest TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    PRIMARY KEY (campaign_id, digest),
    FOREIGN KEY (campaign_id) REFERENCES campaigns(campaign_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS trials_order
    ON trials (campaign_id, round_index, sequence);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_order ON events (campaign_id, id);
CREATE TABLE IF NOT EXISTS campaign_jobs (
    campaign_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    candidate_digest TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    PRIMARY KEY (campaign_id, job_id),
    FOREIGN KEY (campaign_id) REFERENCES campaigns(campaign_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS campaign_jobs_order ON campaign_jobs (campaign_id, created_at);
"""


class CampaignStore:
    def __init__(self, root: str | Path) -> None:
        self.directory = Path(root) / "campaigns"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db_path = self.directory / "campaigns.sqlite3"
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(_SCHEMA)
            self._migrate(connection)

    def _migrate(self, connection: sqlite3.Connection) -> None:
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(campaigns)").fetchall()
        }
        if "cancel_requested" not in columns:
            connection.execute(
                "ALTER TABLE campaigns ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0"
            )
        if "run_job_id" not in columns:
            connection.execute(
                "ALTER TABLE campaigns ADD COLUMN run_job_id TEXT NOT NULL DEFAULT ''"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    # ------------------------------------------------------------------ #
    # campaigns
    # ------------------------------------------------------------------ #

    def create_or_load(self, spec: CampaignSpec, *, resume: bool = True) -> dict[str, Any]:
        campaign_id = spec.candidate_id
        digest = spec.digest()
        now = utc_now_iso()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM campaigns WHERE campaign_id = ?", (campaign_id,)
            ).fetchone()
            if row is not None:
                if row["spec_digest"] != digest:
                    raise CampaignConflict(
                        f"campaign '{campaign_id}' already exists with a different specification",
                        campaign_id=campaign_id,
                        existing_digest=row["spec_digest"],
                        requested_digest=digest,
                    )
                if not resume:
                    count = connection.execute(
                        "SELECT COUNT(*) FROM trials WHERE campaign_id = ?", (campaign_id,)
                    ).fetchone()[0]
                    if count:
                        raise CampaignConflict(
                            f"campaign '{campaign_id}' already has {count} trials; "
                            "resume it or use a new campaign id",
                            campaign_id=campaign_id,
                            trials=int(count),
                        )
                return dict(row)
            connection.execute(
                "INSERT INTO campaigns "
                "(campaign_id, name, mode, spec_digest, spec_json, status, stop_reason, "
                "cancel_requested, seed, budget, rounds, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 'running', '', 0, ?, ?, 0, ?, ?)",
                (
                    campaign_id,
                    spec.name,
                    spec.mode,
                    digest,
                    canonical_json(spec.to_dict()),
                    spec.seed,
                    spec.budget,
                    now,
                    now,
                ),
            )
            connection.execute(
                "INSERT INTO events (campaign_id, kind, payload_json, created_at) "
                "VALUES (?, 'campaign.created', ?, ?)",
                (campaign_id, canonical_json({"spec_digest": digest}), now),
            )
            return dict(
                connection.execute(
                    "SELECT * FROM campaigns WHERE campaign_id = ?", (campaign_id,)
                ).fetchone()
            )

    def campaign(self, campaign_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM campaigns WHERE campaign_id = ?", (campaign_id,)
            ).fetchone()
        if row is None:
            raise CampaignNotFound(f"campaign '{campaign_id}' not found", campaign_id=campaign_id)
        return dict(row)

    def list_campaigns(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM campaigns ORDER BY updated_at DESC, campaign_id LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [dict(row) for row in rows]

    def cancel_requested(self, campaign_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT cancel_requested FROM campaigns WHERE campaign_id = ?", (campaign_id,)
            ).fetchone()
        if row is None:
            raise CampaignNotFound(f"campaign '{campaign_id}' not found", campaign_id=campaign_id)
        return bool(row["cancel_requested"])

    def request_cancel(self, campaign_id: str) -> dict[str, Any]:
        """Durably request cancellation; the running engine observes this flag."""
        now = utc_now_iso()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM campaigns WHERE campaign_id = ?", (campaign_id,)
            ).fetchone()
            if row is None:
                raise CampaignNotFound(
                    f"campaign '{campaign_id}' not found", campaign_id=campaign_id
                )
            if row["status"] in ("completed", "cancelled"):
                return dict(row)
            if not row["cancel_requested"]:
                connection.execute(
                    "UPDATE campaigns SET cancel_requested = 1, status = 'cancelling', "
                    "updated_at = ? WHERE campaign_id = ?",
                    (now, campaign_id),
                )
                connection.execute(
                    "INSERT INTO events (campaign_id, kind, payload_json, created_at) "
                    "VALUES (?, 'campaign.cancel_requested', ?, ?)",
                    (campaign_id, canonical_json({}), now),
                )
        return self.campaign(campaign_id)

    def finish_cancel(self, campaign_id: str, *, rounds: int) -> None:
        now = utc_now_iso()
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT status FROM campaigns WHERE campaign_id = ?", (campaign_id,)
            ).fetchone()
            if existing is None:
                raise CampaignNotFound(
                    f"campaign '{campaign_id}' not found", campaign_id=campaign_id
                )
            if existing["status"] == "cancelled":
                return
            connection.execute(
                "UPDATE campaigns SET cancel_requested = 1, status = 'cancelled', "
                "stop_reason = 'cancelled', rounds = ?, updated_at = ? WHERE campaign_id = ?",
                (int(rounds), now, campaign_id),
            )
            connection.execute(
                "INSERT INTO events (campaign_id, kind, payload_json, created_at) "
                "VALUES (?, 'campaign.cancelled', ?, ?)",
                (campaign_id, canonical_json({"rounds": int(rounds)}), now),
            )

    def link_job(
        self,
        campaign_id: str,
        job_id: str,
        *,
        candidate_digest: str = "",
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO campaign_jobs "
                "(campaign_id, job_id, candidate_digest, created_at) VALUES (?, ?, ?, ?)",
                (campaign_id, job_id, candidate_digest, utc_now_iso()),
            )

    def run_job_id(self, campaign_id: str) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT run_job_id FROM campaigns WHERE campaign_id = ?", (campaign_id,)
            ).fetchone()
        if row is None:
            raise CampaignNotFound(f"campaign '{campaign_id}' not found", campaign_id=campaign_id)
        return str(row["run_job_id"] or "")

    def set_run_job_id(self, campaign_id: str, job_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE campaigns SET run_job_id = ?, updated_at = ? WHERE campaign_id = ?",
                (job_id, utc_now_iso(), campaign_id),
            )

    def job_ids(self, campaign_id: str) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT job_id FROM campaign_jobs WHERE campaign_id = ? "
                "ORDER BY created_at, job_id",
                (campaign_id,),
            ).fetchall()
        return [str(row["job_id"]) for row in rows]

    def update_campaign(
        self,
        campaign_id: str,
        *,
        status: str,
        stop_reason: str,
        rounds: int,
    ) -> bool:
        """Advance status without overwriting a cancellation that already won.

        Cancellation and the coordinator's final ``completed`` write can race. Both
        are writers, so the conditional UPDATE makes the first committed transition
        authoritative: a cancellation flag turns later status writes into no-ops
        instead of replacing ``cancelled`` with ``completed``.
        """
        now = utc_now_iso()
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE campaigns SET status = ?, stop_reason = ?, rounds = ?, updated_at = ? "
                "WHERE campaign_id = ? AND cancel_requested = 0",
                (status, stop_reason, int(rounds), now, campaign_id),
            )
            if cursor.rowcount == 0:
                return False
            connection.execute(
                "INSERT INTO events (campaign_id, kind, payload_json, created_at) "
                "VALUES (?, 'campaign.updated', ?, ?)",
                (
                    campaign_id,
                    canonical_json(
                        {"status": status, "stop_reason": stop_reason, "rounds": int(rounds)}
                    ),
                    now,
                ),
            )
        return True

    # ------------------------------------------------------------------ #
    # trials
    # ------------------------------------------------------------------ #

    def record_trial(self, campaign_id: str, trial: Trial) -> bool:
        with self._connect() as connection:
            # Serialize the cancellation check with the insert. A deferred read could
            # observe cancel_requested=0, let a cancellation commit, and then append a
            # trial after the cancel call returned.
            connection.execute("BEGIN IMMEDIATE")
            campaign = connection.execute(
                "SELECT cancel_requested FROM campaigns WHERE campaign_id = ?", (campaign_id,)
            ).fetchone()
            if campaign is not None and campaign["cancel_requested"]:
                # Cancellation is a durable barrier: a batch that was already being
                # evaluated when the request arrived may not append new trials.
                return False
            cursor = connection.execute(
                "INSERT OR IGNORE INTO trials "
                "(campaign_id, digest, round_index, sequence, candidate_json, status, "
                "metrics_json, error, failures_json, package_fingerprint, graph_digest, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    campaign_id,
                    trial.candidate.digest,
                    int(trial.round_index),
                    int(trial.sequence),
                    canonical_json(trial.candidate.to_dict()),
                    trial.status,
                    canonical_json(trial.metrics),
                    trial.error,
                    canonical_json(list(trial.constraint_failures)),
                    trial.package_fingerprint,
                    trial.graph_digest,
                    trial.created_at,
                ),
            )
            inserted = cursor.rowcount == 1
            if inserted:
                connection.execute(
                    "INSERT INTO events (campaign_id, kind, payload_json, created_at) "
                    "VALUES (?, 'trial.recorded', ?, ?)",
                    (
                        campaign_id,
                        canonical_json(
                            {
                                "digest": trial.candidate.digest,
                                "round": trial.round_index,
                                "sequence": trial.sequence,
                                "status": trial.status,
                            }
                        ),
                        trial.created_at,
                    ),
                )
        return inserted

    def trials(self, campaign_id: str) -> list[Trial]:
        row = self.campaign(campaign_id)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM trials WHERE campaign_id = ? "
                "ORDER BY round_index, sequence, digest",
                (campaign_id,),
            ).fetchall()
        campaign_spec = json.loads(row["spec_json"])
        family = str(campaign_spec.get("family", ""))
        trials: list[Trial] = []
        for record in rows:
            payload = json.loads(record["candidate_json"])
            candidate = Candidate.build(
                str(row["mode"]),
                family,
                tuple(payload["values"]),
                payload["config"],
            )
            trials.append(
                Trial(
                    candidate=candidate,
                    round_index=int(record["round_index"]),
                    sequence=int(record["sequence"]),
                    status=str(record["status"]),
                    metrics=json.loads(record["metrics_json"]),
                    error=str(record["error"]),
                    package_fingerprint=str(record["package_fingerprint"]),
                    graph_digest=str(record["graph_digest"]),
                    created_at=str(record["created_at"]),
                    constraint_failures=tuple(json.loads(record["failures_json"])),
                )
            )
        return trials

    def events(self, campaign_id: str, *, after: int = 0, limit: int = 1000) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM events WHERE campaign_id = ? AND id > ? ORDER BY id LIMIT ?",
                (campaign_id, int(after), int(limit)),
            ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "kind": row["kind"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
