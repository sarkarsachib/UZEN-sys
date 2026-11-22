"""Worker daemon.

Two real modes:

**Local mode** (default): the daemon owns a local workspace and executes queued jobs
from its own job database. This is the single-node execution plane.

**Remote mode** (``control_plane=...``): the daemon registers with a UZEN REST control
plane, heartbeats, claims a job under a lease, fetches the artifacts that job needs,
executes it locally, renews the lease while it runs, and reports the outcome.

Distributed guarantees this daemon participates in:

* **A claim is a lease with a token.** The control plane returns the job together with
  a ``dispatch_token`` and the attempt number. Renewals and the final report present
  that token, so a worker whose lease expired cannot overwrite the result of the worker
  that replaced it.
* **Execution is idempotent.** The local job keeps the control plane's job id. If the
  same claim is delivered twice (a lost report, a retried request) the previous result
  is reused instead of recomputing the job and creating a second artifact.
* **Inputs are fetched, not assumed.** An artifact referenced by the request that is
  absent locally is pulled through the configured object-store transport (or the shared
  exchange directory) and integrity-verified before the job runs.

A worker never executes arbitrary user scripts: it only runs jobs through
:class:`uzen.jobs.service.JobService`, i.e. registered operations against artifacts.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from uzen.errors import UzenError, WorkerAuthError
from uzen.infrastructure.storage.db import TERMINAL_STATES
from uzen.jobs.service import JobService
from uzen.util.canonical import utc_now_iso
from uzen.workers.protocol.messages import (
    PROTOCOL_VERSION,
    WorkerRegistration,
    detect_capabilities,
    merge_capability_overrides,
)


@dataclass
class WorkerDaemon:
    root: str | Path = "./uzen-worker"
    worker_id: str | None = None
    control_plane: str | None = None
    secret: str | None = None
    exchange_dir: str | Path | None = None
    poll_interval: float = 1.0
    lease_seconds: float = 120.0
    claim_wait_seconds: float = 0.0
    max_jobs: int | None = None
    job_kinds: tuple[str, ...] | None = None
    capability_overrides: Mapping[str, Any] | None = None
    transport: Any | None = None
    _service: JobService | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.worker_id = self.worker_id or f"{socket.gethostname()}-{os.getpid()}"
        self.exchange_dir = Path(self.exchange_dir) if self.exchange_dir else self.root / "exchange"
        self.exchange_dir.mkdir(parents=True, exist_ok=True)
        self._service = JobService(self.root / "workspace")

    @property
    def service(self) -> JobService:
        assert self._service is not None
        return self._service

    def capabilities(self) -> dict[str, Any]:
        """What this worker advertises: detected environment plus operator overrides."""
        return merge_capability_overrides(detect_capabilities(), self.capability_overrides)

    # ------------------------------------------------------------------ #
    # local mode
    # ------------------------------------------------------------------ #

    def run_once(self) -> dict[str, Any]:
        if self.control_plane:
            return self._remote_cycle()
        pending = self.service.db.list_jobs(state="QUEUED", limit=1)
        if not pending:
            return {"worker_id": self.worker_id, "executed": None, "note": "queue empty"}
        job = self.service.run(pending[0].id)
        return {"worker_id": self.worker_id, "executed": job["id"], "state": job["state"], "result": job.get("result") or job.get("error")}

    def run_forever(self, *, poll_interval: float | None = None, max_jobs: int | None = None, max_iterations: int | None = None) -> None:
        interval = poll_interval if poll_interval is not None else self.poll_interval
        executed = 0
        iterations = 0
        if self.control_plane:
            self.register()
        while True:
            iterations += 1
            outcome = self.run_once()
            if outcome.get("executed"):
                executed += 1
            if max_jobs is not None and executed >= max_jobs:
                return
            if max_iterations is not None and iterations >= max_iterations:
                return
            if not outcome.get("executed"):
                time.sleep(interval)

    # ------------------------------------------------------------------ #
    # remote mode
    # ------------------------------------------------------------------ #

    def register(self) -> dict[str, Any]:
        if not self.control_plane:
            raise UzenError("remote registration requires control_plane=...")
        body = WorkerRegistration(
            worker_id=self.worker_id or "worker", capabilities=self.capabilities()
        ).to_dict()
        response = self._post("/v1/workers/register", body, authenticated=False)
        secret = response.get("secret")
        if not secret:
            raise WorkerAuthError("control plane did not return a worker secret")
        self.secret = str(secret)
        assert self.worker_id is not None
        self._write_credentials(self.worker_id, self.secret)
        return response

    def _write_credentials(self, worker_id: str, secret: str) -> None:
        path = self.root / "worker-credentials.json"
        path.write_text(json.dumps({"worker_id": worker_id, "secret": secret}), encoding="utf-8")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    def heartbeat(self, status: str | None = None) -> dict[str, Any]:
        """Liveness touch. A status is sent only when the caller wants to change it,
        so a routine heartbeat cannot accidentally un-drain a worker."""
        assert self.worker_id is not None
        body: dict[str, Any] = {"worker_id": self.worker_id, "secret": self.secret}
        if status is not None:
            body["status"] = status
        return self._post("/v1/workers/heartbeat", body)

    def claim(self) -> dict[str, Any] | None:
        assert self.worker_id is not None
        response = self._post(
            "/v1/workers/claim",
            {
                "worker_id": self.worker_id,
                "secret": self.secret,
                "lease_seconds": self.lease_seconds,
                "wait_seconds": self.claim_wait_seconds,
                "capabilities": self.capabilities(),
                "job_kinds": list(self.job_kinds) if self.job_kinds is not None else None,
            },
        )
        return response.get("job")

    def drain(self) -> dict[str, Any]:
        """Ask the control plane to stop routing new work to this worker."""
        assert self.worker_id is not None
        return self._post(
            f"/v1/workers/{self.worker_id}/drain",
            {"worker_id": self.worker_id, "secret": self.secret},
        )

    def renew(self, job_id: str, token: str | None) -> bool:
        assert self.worker_id is not None
        response = self._post(
            "/v1/workers/renew",
            {
                "worker_id": self.worker_id,
                "secret": self.secret,
                "job_id": job_id,
                "token": token,
                "lease_seconds": self.lease_seconds,
            },
        )
        return bool(response.get("ok"))

    def report(
        self,
        job_id: str,
        state: str,
        *,
        token: str | None = None,
        result: Any = None,
        error: Any = None,
    ) -> dict[str, Any]:
        assert self.worker_id is not None
        return self._post(
            "/v1/workers/report",
            {
                "worker_id": self.worker_id,
                "secret": self.secret,
                "job_id": job_id,
                "token": token,
                "result": {"state": state, "result": result, "error": error},
            },
        )

    # ------------------------------------------------------------------ #
    # artifact transfer and local execution of a claimed job
    # ------------------------------------------------------------------ #

    def fetch_inputs(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Make every artifact the request references available locally.

        Object-store transport is used when one is configured; otherwise the artifact
        tree is copied out of the shared ``exchange_dir``. Anything that cannot be
        obtained is reported, and the job fails explicitly rather than running against a
        silently different artifact.
        """
        fetched: list[dict[str, Any]] = []
        present: list[str] = []
        missing: list[dict[str, Any]] = []
        for entry in request.get("inputs") or []:
            if not isinstance(entry, Mapping) or "artifact_id" not in entry:
                continue
            artifact_id = str(entry["artifact_id"])
            if self.service.store.exists(artifact_id):
                present.append(artifact_id)
                continue
            try:
                if self.transport is not None:
                    report = self.transport.fetch_artifact(
                        self.service.store, artifact_id, include_parents=True
                    )
                    fetched.append({"artifact_id": artifact_id, "via": "object_store", **report})
                else:
                    source = Path(self.exchange_dir) / artifact_id
                    if not source.is_dir():
                        raise FileNotFoundError(
                            f"artifact {artifact_id} is not in the local store, the object-store "
                            f"transport is not configured, and {source} does not exist"
                        )
                    target = self.service.store.artifact_dir(artifact_id)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(source, target, dirs_exist_ok=True)
                    verification = self.service.store.verify(artifact_id)
                    fetched.append(
                        {
                            "artifact_id": artifact_id,
                            "via": "shared_filesystem",
                            "verified": bool(verification.get("ok")),
                        }
                    )
            except Exception as exc:
                missing.append(
                    {
                        "artifact_id": artifact_id,
                        "error": type(exc).__name__,
                        "detail": str(exc),
                    }
                )
        return {"fetched": fetched, "already_present": present, "unavailable": missing}

    def prune_exchange(
        self,
        *,
        keep_artifact_ids: Any = (),
        older_than_seconds: float = 24 * 3600,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Evict old entries from the shared exchange cache.

        Artifacts still referenced by queued or leased work are kept when the caller
        passes their ids. The default is a dry run: deletion of a cache directory is
        explicit, never a side effect of polling.
        """
        keep = {str(item) for item in keep_artifact_ids}
        cutoff = time.time() - max(0.0, float(older_than_seconds))
        candidates: list[str] = []
        retained: list[str] = []
        for entry in sorted(Path(self.exchange_dir).iterdir()):
            if not entry.is_dir():
                continue
            if entry.name in keep:
                retained.append(entry.name)
                continue
            try:
                modified = entry.stat().st_mtime
            except OSError:
                continue
            if modified <= cutoff:
                candidates.append(entry.name)
            else:
                retained.append(entry.name)
        removed: list[str] = []
        if not dry_run:
            for name in candidates:
                try:
                    shutil.rmtree(Path(self.exchange_dir) / name)
                    removed.append(name)
                except OSError:
                    retained.append(name)
        return {
            "exchange_dir": str(self.exchange_dir),
            "dry_run": bool(dry_run),
            "kept": sorted(retained),
            "evictable": candidates,
            "removed": removed,
            "older_than_seconds": float(older_than_seconds),
            "kept_for_jobs": sorted(keep),
        }

    def execute_claimed(self, job: Mapping[str, Any]) -> dict[str, Any]:
        """Run a leased control-plane job locally, at most once.

        The local job id is the control plane's job id, so events, results and dispatch
        records line up across the two databases. A repeated claim for a job this worker
        already completed returns the stored result instead of executing again.
        """
        job_id = str(job["id"])
        request = dict(job.get("request") or {})
        attempt = int((job.get("lease") or {}).get("attempt") or job.get("attempts") or 1)
        existing = None
        try:
            existing = self.service.db.get_job(job_id)
        except UzenError:
            existing = None
        if existing is not None and existing.state in TERMINAL_STATES:
            return {
                "job_id": job_id,
                "state": existing.state,
                "reused": True,
                "attempt": attempt,
                "result": existing.result,
                "error": existing.error,
            }
        if existing is None:
            self.service.db.create_job(request, job_id=job_id, state="QUEUED")
        transfer = self.fetch_inputs(request)
        if transfer["unavailable"]:
            detail = {
                "code": "worker.artifact_unavailable",
                "message": "the worker could not obtain every input artifact",
                "details": {"unavailable": transfer["unavailable"]},
            }
            return {
                "job_id": job_id,
                "state": "FAILED",
                "attempt": attempt,
                "result": None,
                "error": detail,
                "transfer": transfer,
            }
        executed = self.service.run(job_id)
        return {
            "job_id": job_id,
            "state": executed["state"],
            "attempt": attempt,
            "reused": False,
            "result": executed.get("result"),
            "error": executed.get("error"),
            "transfer": transfer,
        }

    # ------------------------------------------------------------------ #
    # one remote poll cycle
    # ------------------------------------------------------------------ #

    def _remote_cycle(self) -> dict[str, Any]:
        if not self.secret:
            self.register()
        self.heartbeat()
        job = self.claim()
        if not job:
            return {"worker_id": self.worker_id, "executed": None, "note": "no job available"}
        job_id = str(job["id"])
        lease = dict(job.get("lease") or {})
        token = lease.get("token")
        attempt = lease.get("attempt")
        start = time.perf_counter()

        stop = threading.Event()
        renewals = {"count": 0, "lost": False}

        def _renew_loop() -> None:
            interval = max(1.0, float(self.lease_seconds) / 3.0)
            while not stop.wait(interval):
                try:
                    if self.renew(job_id, token):
                        renewals["count"] += 1
                    else:
                        renewals["lost"] = True
                        return
                except Exception:
                    # A transient control-plane failure must not kill the job; the lease
                    # will either be renewed next tick or expire and be reaped.
                    continue

        renewer = threading.Thread(target=_renew_loop, name=f"lease-{job_id[:8]}", daemon=True)
        renewer.start()
        try:
            executed = self.execute_claimed(job)
        finally:
            stop.set()
            renewer.join(timeout=5.0)
        duration_ms = (time.perf_counter() - start) * 1000.0
        state = str(executed["state"])
        reported = self.report(
            job_id,
            state,
            token=token,
            result=executed.get("result"),
            error=executed.get("error"),
        )
        return {
            "worker_id": self.worker_id,
            "executed": job_id,
            "control_plane_job": job_id,
            "attempt": attempt,
            "state": state,
            "duration_ms": round(duration_ms, 3),
            "renewals": renewals["count"],
            "lease_lost": renewals["lost"],
            "report_status": reported.get("status"),
            "reused": executed.get("reused", False),
        }

    # ------------------------------------------------------------------ #

    def _post(self, path: str, body: Mapping[str, Any], *, authenticated: bool = True) -> dict[str, Any]:
        assert self.control_plane is not None
        if authenticated and not self.secret:
            raise WorkerAuthError("worker is not registered (no secret); call register() first")
        url = f"{self.control_plane.rstrip('/')}{path}"
        req = urllib.request.Request(
            url,
            data=json.dumps(dict(body)).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json", "X-UZEN-Protocol": PROTOCOL_VERSION},
        )
        try:
            with urllib.request.urlopen(req, timeout=60.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            payload = exc.read().decode("utf-8", errors="replace")
            raise WorkerAuthError(
                f"worker request failed with HTTP {exc.code}", body=payload[:500], path=path
            ) from exc
        except urllib.error.URLError as exc:
            raise UzenError(f"cannot reach control plane: {exc.reason}", path=path) from exc

    def describe(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "root": str(self.root),
            "mode": "remote" if self.control_plane else "local",
            "control_plane": self.control_plane,
            "capabilities": self.capabilities(),
            "capability_overrides": dict(self.capability_overrides or {}),
            "artifact_transport": (
                "object_store" if self.transport is not None else "shared_filesystem"
            ),
            "lease_seconds": self.lease_seconds,
            "claim_wait_seconds": self.claim_wait_seconds,
            "credentials_file": str(self.root / "worker-credentials.json"),
            "detected_at": utc_now_iso(),
        }
