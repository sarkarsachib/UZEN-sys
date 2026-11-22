"""N7 distributed campaign trials over the N4 job queue.

The campaign coordinator owns planning, ranking and durable campaign state. Trial
evaluation is submitted as ordinary control-plane jobs and executed by workers that
claim them under the normal capability/lease rules. This module deliberately does not
execute a trial locally while waiting: a distributed run either has workers or fails
with a typed timeout.

A trial job is read-only (scope ``inspection``): it builds a candidate package/graph in
worker memory, runs a real NumPy forward pass, and returns metrics. It never persists
an artifact, modifies a source package or performs a gradient step.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from uzen.core.campaign.evaluate import Evaluator
from uzen.core.campaign.models import Candidate
from uzen.core.campaign.spec import CampaignSpec
from uzen.core.campaign.store import CampaignStore
from uzen.errors import (
    CampaignCancelled,
    CampaignTimeout,
    GraphValidationError,
    InvalidOperationParameters,
    ShapeMismatch,
    UzenError,
)
from uzen.infrastructure.storage.db import TERMINAL_STATES
from uzen.jobs.scheduler import JobScheduler
from uzen.jobs.service import JobService
from uzen.workers.protocol.messages import detect_capabilities

CAMPAIGN_TRIAL_KIND = "campaign_trial"
CAMPAIGN_RUN_KIND = "campaign_run"


def _open_job_service(root: str, *, timeout_seconds: float = 15.0) -> JobService:
    """Open a worker-side service, waiting out a short schema/writer lock.

    Several worker processes may start while the coordinator is creating the job
    database. SQLite's busy timeout covers normal statements, but schema setup can
    still report a lock at process start; retrying is preferable to losing a worker
    without an error the scheduler can see.
    """
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while True:
        try:
            return JobService(root)
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def campaign_trial_request(spec: CampaignSpec, candidate: Candidate) -> dict[str, Any]:
    """Build the control-plane request for one read-only trial evaluation."""
    return {
        "task_scope": "inspection",
        "objective": f"campaign trial {spec.candidate_id}",
        "inputs": [],
        "operations": [],
        "policy": "STRICT_PLUS",
        "seed": spec.seed,
        "options": {
            "task_kind": CAMPAIGN_TRIAL_KIND,
            "campaign_spec": spec.to_dict(),
            "candidate": candidate.to_dict(),
        },
    }


def campaign_trial_idempotency_key(spec: CampaignSpec, candidate: Candidate) -> str:
    """Stable key so a crashed coordinator can resubmit a batch without duplicates."""
    return f"campaign-trial:{spec.candidate_id}:{candidate.digest}"


def validate_campaign_trial_request(request: Mapping[str, Any]) -> tuple[CampaignSpec, Candidate]:
    """Validate a campaign trial request at submit time, before it enters the queue.

    The digest is recomputed from the candidate values/config rather than trusted from
    the payload, so a forged candidate cannot be submitted under another trial's key.
    """
    options = dict(request.get("options") or {})
    if options.get("task_kind") != CAMPAIGN_TRIAL_KIND:
        raise InvalidOperationParameters("not a campaign trial request")
    scope = str(request.get("task_scope", "inspection"))
    if scope != "inspection":
        raise InvalidOperationParameters("campaign trials must use the inspection scope")
    if request.get("inputs"):
        raise InvalidOperationParameters("campaign trials must not consume input artifacts")
    if request.get("operations"):
        raise InvalidOperationParameters("campaign trials must not submit engineering operations")
    spec_doc = options.get("campaign_spec")
    candidate_doc = options.get("candidate")
    if not isinstance(spec_doc, Mapping) or not isinstance(candidate_doc, Mapping):
        raise InvalidOperationParameters(
            "campaign trials require options.campaign_spec and options.candidate objects"
        )
    spec = CampaignSpec.from_dict(dict(spec_doc))
    config = candidate_doc.get("config")
    values = candidate_doc.get("values")
    if not isinstance(config, Mapping) or not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise InvalidOperationParameters("candidate requires config (object) and values (list)")
    candidate = Candidate.build(spec.mode, spec.family, tuple(values), dict(config))
    if str(candidate_doc.get("digest")) != candidate.digest:
        raise InvalidOperationParameters(
            "candidate digest does not match its values/config",
            expected=candidate.digest,
            received=candidate_doc.get("digest"),
        )
    return spec, candidate


def campaign_trial_outcome(spec: CampaignSpec, candidate: Candidate) -> dict[str, Any]:
    """Evaluate one candidate and normalize the result for the campaign store.

    Invalid candidates are returned as a successful job with ``trial.status=invalid``;
    the campaign records them rather than losing the batch to a worker failure.
    """
    try:
        metrics, fingerprint, graph_digest = Evaluator(spec).evaluate(candidate)
        return {
            "status": "ok",
            "metrics": metrics,
            "package_fingerprint": fingerprint,
            "graph_digest": graph_digest,
            "error": "",
        }
    except (InvalidOperationParameters, GraphValidationError, ShapeMismatch) as exc:
        return {"status": "invalid", "error": f"{type(exc).__name__}: {exc}"}
    except UzenError as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:  # noqa: BLE001 - recorded as a failed trial, not a process kill
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def _job_error(row: Mapping[str, Any]) -> str:
    error = row.get("error")
    if isinstance(error, Mapping):
        code = error.get("code") or "job.failed"
        message = error.get("message") or ""
        return f"{code}: {message}".strip()
    return str(error or "job did not produce a trial result")


def _outcome_from_job(row: Mapping[str, Any]) -> dict[str, Any]:
    state = str(row.get("state"))
    if state == "SUCCEEDED":
        result = row.get("result")
        trial = result.get("trial") if isinstance(result, Mapping) else None
        if isinstance(trial, Mapping):
            return {
                "status": str(trial.get("status", "error")),
                "metrics": dict(trial.get("metrics") or {}),
                "package_fingerprint": str(trial.get("package_fingerprint", "")),
                "graph_digest": str(trial.get("graph_digest", "")),
                "error": str(trial.get("error", "")),
            }
        return {"status": "error", "error": "job succeeded without a trial result"}
    if state == "CANCELLED":
        return {"status": "error", "error": "campaign trial was cancelled"}
    return {"status": "error", "error": _job_error(row)}


@dataclass
class DistributedTrialExecutor:
    """Submit campaign trials to N4 and wait for workers to settle them.

    ``timeout_seconds`` is a hard deadline for the whole batch, not for each job. A
    timeout is typed and leaves the campaign resumable; jobs that finish later remain
    available to the next run through their idempotency keys.
    """

    service: JobService
    timeout_seconds: float = 600.0
    poll_interval_seconds: float = 0.1
    submitter_id: str = "campaign-coordinator"
    store: CampaignStore | None = None
    should_stop: Callable[[], bool] | None = None

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise InvalidOperationParameters("trial timeout must be positive")
        if self.poll_interval_seconds <= 0:
            raise InvalidOperationParameters("poll interval must be positive")
        if self.store is None:
            self.store = CampaignStore(self.service.root)

    def _stopped(self, spec: CampaignSpec) -> bool:
        if self.should_stop is not None and bool(self.should_stop()):
            return True
        assert self.store is not None
        return self.store.cancel_requested(spec.candidate_id)

    def _cancel_jobs(self, job_ids: Sequence[str]) -> None:
        for job_id in job_ids:
            try:
                self.service.cancel(job_id)
            except UzenError:
                # A job that already settled is not an error for cancellation; the
                # campaign state remains the authority.
                pass

    def __call__(self, spec: CampaignSpec, candidates: Sequence[Candidate]) -> list[dict[str, Any]]:
        assert self.store is not None
        jobs: list[tuple[Candidate, str]] = []
        for candidate in candidates:
            row = self.service.submit(
                campaign_trial_request(spec, candidate),
                idempotency_key=campaign_trial_idempotency_key(spec, candidate),
            )
            job_id = str(row["id"])
            self.store.link_job(spec.candidate_id, job_id, candidate_digest=candidate.digest)
            jobs.append((candidate, job_id))

        outcomes: list[dict[str, Any]] = []
        pending: dict[str, Candidate] = {job_id: candidate for candidate, job_id in jobs}
        settled: dict[str, dict[str, Any]] = {}
        deadline = time.monotonic() + float(self.timeout_seconds)
        while pending:
            if self._stopped(spec):
                self._cancel_jobs(sorted(pending))
                raise CampaignCancelled(
                    "campaign was cancelled while waiting for trial jobs",
                    campaign_id=spec.candidate_id,
                    pending_jobs=sorted(pending),
                )
            for job_id in list(pending):
                row = self.service.get(job_id)
                if str(row.get("state")) in TERMINAL_STATES:
                    settled[job_id] = _outcome_from_job(row)
                    pending.pop(job_id, None)
            if pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CampaignTimeout(
                        "distributed campaign batch did not settle before its deadline",
                        campaign_id=spec.candidate_id,
                        pending_jobs=sorted(pending),
                        timeout_seconds=float(self.timeout_seconds),
                    )
                time.sleep(min(float(self.poll_interval_seconds), remaining))
        for _, job_id in jobs:
            outcomes.append(settled[job_id])
        return outcomes


@dataclass
class CampaignWorker:
    """A minimal N4 worker loop for campaign-trial jobs.

    It uses the same atomic claim and token-guarded ``JobService.run`` path as a remote
    worker. It is intentionally not a generic daemon: use one on a control plane whose
    queue contains campaign-trial jobs.
    """

    root: str
    worker_id: str
    lease_seconds: float = 120.0
    claim_wait_seconds: float = 0.25
    poll_interval_seconds: float = 0.1
    max_in_flight_per_worker: int = 1
    capabilities: Mapping[str, Any] | None = None
    job_kinds: Sequence[str] | None = (CAMPAIGN_TRIAL_KIND,)
    _executed: int = field(default=0, init=False)
    _capability_cache: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def warmup(self) -> None:
        """Register capabilities before a coordinator starts submitting trials."""
        service = _open_job_service(self.root)
        try:
            self._capability_cache = dict(self.capabilities or detect_capabilities())
            service.db.set_worker_capabilities(self.worker_id, self._capability_cache)
        finally:
            service.db.close()

    def run_once(self, *, service: JobService | None = None) -> dict[str, Any]:
        owned = service is None
        service = service or _open_job_service(self.root)
        scheduler = JobScheduler(
            service.db,
            lease_seconds=self.lease_seconds,
            max_in_flight_per_worker=self.max_in_flight_per_worker,
        )
        if self._capability_cache is None:
            self._capability_cache = dict(self.capabilities or detect_capabilities())
        job = scheduler.claim(
            self.worker_id,
            capabilities=self._capability_cache,
            lease_seconds=self.lease_seconds,
            wait_seconds=self.claim_wait_seconds,
            poll_interval=self.poll_interval_seconds,
            job_kinds=self.job_kinds,
        )
        if job is None:
            if owned:
                service.db.close()
            return {"worker_id": self.worker_id, "executed": None}
        token = (job.get("lease") or {}).get("token")
        result = service.run(
            str(job["id"]),
            worker_id=self.worker_id,
            token=token,
            lease_seconds=self.lease_seconds,
        )
        self._executed += 1
        if owned:
            service.db.close()
        return {
            "worker_id": self.worker_id,
            "executed": job["id"],
            "state": result.get("state"),
            "attempt": (job.get("lease") or {}).get("attempt"),
        }

    def run_forever(self, *, stop: Any = None, max_jobs: int | None = None) -> int:
        service = _open_job_service(self.root)
        try:
            while stop is None or not stop.is_set():
                if max_jobs is not None and self._executed >= max_jobs:
                    break
                outcome = self.run_once(service=service)
                if outcome.get("executed") is None and self.claim_wait_seconds <= 0:
                    time.sleep(self.poll_interval_seconds)
        finally:
            service.db.close()
        return self._executed
