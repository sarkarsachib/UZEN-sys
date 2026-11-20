"""Distributed scheduler for the control plane.

The scheduler owns everything that is *not* the job's own execution: lease expiry,
capability-aware assignment, attempt accounting, and the operational view an operator
needs when a queue stops draining.

Workers pull; the scheduler never pushes work at a worker that has not asked for it.
Two decisions are therefore separated on purpose:

* **Claim time** (:meth:`JobScheduler.claim`) — reaps expired leases, then lets the
  database atomically pick the oldest job this worker is provably able to run. The
  eligibility predicate lives in SQL, so a fleet of control-plane processes sharing one
  database cannot hand the same job to two workers.
* **Reporting** (:meth:`JobScheduler.report`) — applies a worker's terminal report once,
  guarded by the dispatch token issued with the lease.

``status()`` is the honesty surface: a job whose requirements no registered worker
satisfies is reported as *unroutable* with the exact capability each worker lacked,
instead of appearing as a healthy queued job forever.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Mapping

from uzen.errors import JobNotFound
from uzen.infrastructure.storage.db import Database, JobRow, TERMINAL_STATES
from uzen.jobs.scheduling import (
    CapabilityRequirements,
    explain_assignment,
    match_worker,
    requirements_from_request,
)
from uzen.util.canonical import utc_now_iso

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_WORKER_TTL_SECONDS = 90.0
DEFAULT_CLAIM_POLL_SECONDS = 0.25


def _age_seconds(timestamp: str | None, *, now: datetime | None = None) -> float | None:
    if not timestamp:
        return None
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    return max(0.0, (now - parsed).total_seconds())


class JobScheduler:
    """Lease, attempt and capability management over a shared job database."""

    def __init__(
        self,
        db: Database,
        *,
        lease_seconds: float = 120.0,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        worker_ttl_seconds: float = DEFAULT_WORKER_TTL_SECONDS,
        retry_backoff_seconds: float = 0.0,
        retry_backoff_cap_seconds: float = 300.0,
        aging_seconds: float = 0.0,
        max_in_flight_per_worker: int = 0,
    ) -> None:
        self.db = db
        self.lease_seconds = float(lease_seconds)
        self.max_attempts = int(max_attempts)
        self.worker_ttl_seconds = float(worker_ttl_seconds)
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self.retry_backoff_cap_seconds = max(0.0, float(retry_backoff_cap_seconds))
        # Aging raises a waiting job's effective priority by one every ``aging_seconds``.
        # Off by default because it changes strict priority order; operators opt in.
        self.aging_seconds = max(0.0, float(aging_seconds))
        self.max_in_flight_per_worker = max(0, int(max_in_flight_per_worker))

    # ------------------------------------------------------------------ #
    # assignment
    # ------------------------------------------------------------------ #

    def reap(self, *, now: float | None = None) -> list[dict[str, Any]]:
        return self.db.reap_expired_leases(
            now=now,
            max_attempts=self.max_attempts,
            retry_backoff_seconds=self.retry_backoff_seconds,
            retry_backoff_cap_seconds=self.retry_backoff_cap_seconds,
        )

    def claim(
        self,
        worker_id: str,
        *,
        capabilities: Mapping[str, Any] | None = None,
        lease_seconds: float | None = None,
        now: float | None = None,
        wait_seconds: float = 0.0,
        poll_interval: float = DEFAULT_CLAIM_POLL_SECONDS,
    ) -> dict[str, Any] | None:
        """Lease the next eligible job for ``worker_id``.

        Expired leases are reaped first, so a worker that polls after another worker
        crashed can pick the orphaned job up in the same call. With ``wait_seconds`` the
        call long-polls: it retries the same atomic claim until a job appears or the
        deadline passes, so an idle worker does not need a fixed sleep between polls.

        The returned document carries the ``dispatch_token`` the worker must present on
        renew/report.
        """
        deadline = time.monotonic() + max(0.0, float(wait_seconds))
        while True:
            reaped = self.reap(now=now)
            job = self.db.claim_job(
                worker_id,
                lease_seconds=self.lease_seconds if lease_seconds is None else float(lease_seconds),
                now=now,
                capabilities=capabilities,
                aging_seconds=self.aging_seconds,
                max_in_flight_per_worker=self.max_in_flight_per_worker,
            )
            if job is not None:
                break
            remaining = deadline - time.monotonic()
            if wait_seconds <= 0 or remaining <= 0 or now is not None:
                return None
            time.sleep(min(max(0.01, float(poll_interval)), remaining))
        document = job.to_dict()
        document["lease"] = {
            "worker_id": worker_id,
            "token": job.dispatch_token,
            "attempt": job.attempts,
            "expires_at": job.lease_expires_at,
            "lease_seconds": self.lease_seconds if lease_seconds is None else float(lease_seconds),
        }
        if reaped:
            document["reaped_on_claim"] = reaped
        return document

    def renew(
        self,
        job_id: str,
        worker_id: str,
        *,
        token: str | None = None,
        lease_seconds: float | None = None,
    ) -> bool:
        return self.db.renew_lease(
            job_id,
            worker_id,
            lease_seconds=self.lease_seconds if lease_seconds is None else float(lease_seconds),
            token=token,
        )

    def preempt(self, job_id: str, *, reason: str = "preempted by operator") -> dict[str, Any]:
        """Return a leased job to the queue and revoke the current token."""
        return self.db.preempt_job(job_id, reason=reason)

    def drain_worker(self, worker_id: str) -> dict[str, Any]:
        """Stop a worker from taking new work; its current lease keeps running."""
        updated = self.db.set_worker_status(worker_id, "draining")
        return {
            "worker_id": worker_id,
            "status": "draining" if updated else "unknown",
            "updated": updated,
        }

    def resume_worker(self, worker_id: str) -> dict[str, Any]:
        updated = self.db.set_worker_status(worker_id, "online")
        return {
            "worker_id": worker_id,
            "status": "online" if updated else "unknown",
            "updated": updated,
        }

    def report(
        self,
        job_id: str,
        *,
        worker_id: str,
        token: str,
        state: str,
        result: Mapping[str, Any] | None = None,
        error: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.db.finish_from_worker(
            job_id,
            worker_id=worker_id,
            token=token,
            state=state,
            result=result,
            error=error,
        )

    # ------------------------------------------------------------------ #
    # operational view
    # ------------------------------------------------------------------ #

    def requirements_for(self, job: JobRow) -> CapabilityRequirements:
        rows = self.db.job_requirements(job.id)
        if not rows:
            return requirements_from_request(job.request)
        return CapabilityRequirements.from_rows(
            rows,
            min_cpu_count=job.min_cpu_count,
            min_memory_bytes=job.min_memory_bytes,
        )

    def unroutable_jobs(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """Queued jobs no registered worker can execute, with the reason for each."""
        workers = self.db.workers()
        out: list[dict[str, Any]] = []
        for job in self.db.list_jobs(state="QUEUED", limit=limit):
            if job.cancel_requested:
                continue
            requirements = self.requirements_for(job)
            explanation = explain_assignment(requirements, workers)
            if explanation["routable"]:
                continue
            out.append(
                {
                    "job_id": job.id,
                    "task_scope": job.task_scope,
                    "objective": job.objective,
                    "requirements": requirements.to_dict(),
                    "attempts": job.attempts,
                    "created_at": job.created_at,
                    "detail": (
                        "no registered worker advertises every required capability"
                        if workers
                        else "no worker is registered"
                    ),
                    "considered": explanation["considered"],
                }
            )
        return out

    def eligible_workers(self, job_id: str) -> dict[str, Any]:
        """Explain which workers could take a specific job, and what each one lacks."""
        job = self.db.get_job(job_id)
        requirements = self.requirements_for(job)
        explanation = explain_assignment(requirements, self.db.workers())
        return {"job_id": job.id, "requirements": requirements.to_dict(), **explanation}

    def status(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for job in self.db.list_jobs(limit=10_000):
            counts[job.state] = counts.get(job.state, 0) + 1
        workers = self.db.workers()
        now = datetime.now(timezone.utc)
        worker_states = []
        for worker in workers:
            age = _age_seconds(worker.get("last_seen"), now=now)
            worker_states.append(
                {
                    "worker_id": worker["worker_id"],
                    "status": worker["status"],
                    "last_seen_seconds": age,
                    "alive": age is not None and age <= self.worker_ttl_seconds,
                    "draining": worker["status"] == "draining",
                    "capability_kinds": sorted(
                        {kind for kind, values in self.db.worker_capabilities(worker["worker_id"]).items() if values}
                    ),
                }
            )
        leased = [
            job.to_dict()
            for job in self.db.list_jobs(limit=10_000)
            if job.state in ("DISPATCHED", "RUNNING", "VALIDATING")
        ]
        retried = [
            job.id for job in self.db.list_jobs(limit=10_000) if job.attempts > 1 and job.state not in TERMINAL_STATES
        ]
        return {
            "jobs_by_state": counts,
            "queued": counts.get("QUEUED", 0),
            "leased": len(leased),
            "retried_in_flight": len(retried),
            "workers": worker_states,
            "alive_workers": sum(1 for w in worker_states if w["alive"]),
            "unroutable": self.unroutable_jobs(limit=20),
            "policy": {
                "lease_seconds": self.lease_seconds,
                "max_attempts": self.max_attempts,
                "retry_backoff_seconds": self.retry_backoff_seconds,
                "retry_backoff_cap_seconds": self.retry_backoff_cap_seconds,
                "aging_seconds": self.aging_seconds,
                "max_in_flight_per_worker": self.max_in_flight_per_worker,
            },
            "generated_at": utc_now_iso(),
        }

    def history(self, job_id: str) -> dict[str, Any]:
        """Every dispatch attempt for a job: who held it, with which token, and how it ended."""
        try:
            job = self.db.get_job(job_id)
        except JobNotFound:
            raise
        return {
            "job_id": job.id,
            "state": job.state,
            "attempts": job.attempts,
            "max_attempts": job.max_attempts,
            "records": self.db.dispatch_records(job_id),
            "events": [
                event
                for event in self.db.events(job_id)
                if event["kind"] in (
                    "dispatched",
                    "lease_expired",
                    "queued",
                    "preempted",
                    "failed",
                    "succeeded",
                    "cancelled",
                )
            ],
        }

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "job_scheduler",
            "queue": "job database (atomic capability-aware claim)",
            "lease_seconds": self.lease_seconds,
            "max_attempts": self.max_attempts,
            "retry_backoff_seconds": self.retry_backoff_seconds,
            "retry_backoff_cap_seconds": self.retry_backoff_cap_seconds,
            "aging_seconds": self.aging_seconds,
            "max_in_flight_per_worker": self.max_in_flight_per_worker,
            "priority": "operator bias + optional aging",
            "preemption": "cooperative (token revoked, job requeued)",
            "worker_draining": True,
            "long_poll_claim": True,
            "reap_on_claim": True,
            "idempotent_reports": True,
            "capability_matching": "sql",
        }

    # ------------------------------------------------------------------ #
    # convenience for single-process deployments
    # ------------------------------------------------------------------ #

    def match(self, requirements: CapabilityRequirements, worker: Mapping[str, Any]) -> dict[str, Any]:
        return match_worker(
            requirements,
            dict(worker.get("capabilities") or {}),
            worker_id=str(worker.get("worker_id", "")),
        ).to_dict()
