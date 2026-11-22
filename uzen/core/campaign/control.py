"""N8 campaign control plane: durable background runs and cancellation.

A background campaign is an ordinary N4 job with ``options.task_kind=campaign_run``.
A coordinator worker claims it and runs the same :class:`CampaignEngine` used by a
synchronous request; a distributed coordinator submits each trial as a nested
``campaign_trial`` job. Because the run job is leased and recorded like any other
job, it survives a lost HTTP connection, a coordinator crash and a retry.

This module owns only the request contract and submission helper. Job execution lives
in :class:`uzen.jobs.service.JobService`, and cancellation lives in
:class:`uzen.core.campaign.engine.CampaignEngine`.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Mapping

from uzen.core.campaign.distributed import CAMPAIGN_RUN_KIND
from uzen.core.campaign.spec import CampaignSpec
from uzen.errors import CampaignConflict, InvalidOperationParameters, JobNotFound
from uzen.infrastructure.storage.db import TERMINAL_STATES
from uzen.jobs.service import JobService

CAMPAIGN_RUN_CANDIDATE = "__campaign_run__"
#: How long a duplicate submitter waits for the winning coordinator job to appear.
RUN_ADMISSION_WAIT_SECONDS = 5.0
#: A reservation older than this may be recovered if its creator never created the job.
RUN_RESERVATION_GRACE_SECONDS = 30.0


def campaign_run_request(
    spec: CampaignSpec,
    *,
    distributed: bool = False,
    resume: bool = True,
    max_rounds: int | None = None,
    trial_timeout_seconds: float = 600.0,
    poll_interval_seconds: float = 0.1,
) -> dict[str, Any]:
    """Build the durable request for one campaign execution."""
    if max_rounds is not None and (
        isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds < 1
    ):
        raise InvalidOperationParameters("max_rounds must be a positive integer or null")
    if trial_timeout_seconds <= 0:
        raise InvalidOperationParameters("trial_timeout_seconds must be positive")
    if poll_interval_seconds <= 0:
        raise InvalidOperationParameters("poll_interval_seconds must be positive")
    return {
        "task_scope": "inspection",
        "objective": f"campaign run {spec.candidate_id}",
        "inputs": [],
        "operations": [],
        "policy": "STRICT_PLUS",
        "seed": spec.seed,
        "options": {
            "task_kind": CAMPAIGN_RUN_KIND,
            "campaign_spec": spec.to_dict(),
            "distributed": bool(distributed),
            "resume": bool(resume),
            "max_rounds": max_rounds,
            "trial_timeout_seconds": float(trial_timeout_seconds),
            "poll_interval_seconds": float(poll_interval_seconds),
        },
    }


def validate_campaign_run_request(
    request: Mapping[str, Any],
) -> tuple[CampaignSpec, dict[str, Any]]:
    """Validate a campaign-run request before it is queued or executed."""
    options = dict(request.get("options") or {})
    if options.get("task_kind") != CAMPAIGN_RUN_KIND:
        raise InvalidOperationParameters("not a campaign run request")
    if str(request.get("task_scope", "inspection")) != "inspection":
        raise InvalidOperationParameters("campaign runs must use the inspection scope")
    if request.get("inputs"):
        raise InvalidOperationParameters("campaign runs must not consume input artifacts")
    if request.get("operations"):
        raise InvalidOperationParameters("campaign runs must not submit engineering operations")
    spec_doc = options.get("campaign_spec")
    if not isinstance(spec_doc, Mapping):
        raise InvalidOperationParameters("campaign runs require options.campaign_spec object")
    spec = CampaignSpec.from_dict(dict(spec_doc))
    max_rounds = options.get("max_rounds")
    if max_rounds is not None and (
        isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds < 1
    ):
        raise InvalidOperationParameters("max_rounds must be a positive integer or null")
    distributed = options.get("distributed", False)
    resume = options.get("resume", True)
    if not isinstance(distributed, bool) or not isinstance(resume, bool):
        raise InvalidOperationParameters("distributed and resume must be booleans")
    trial_timeout = float(options.get("trial_timeout_seconds", 600.0))
    poll_interval = float(options.get("poll_interval_seconds", 0.1))
    if trial_timeout <= 0 or poll_interval <= 0:
        raise InvalidOperationParameters("trial timeout and poll interval must be positive")
    return spec, {
        "distributed": distributed,
        "resume": resume,
        "max_rounds": max_rounds,
        "trial_timeout_seconds": trial_timeout,
        "poll_interval_seconds": poll_interval,
    }


def submit_campaign_run(
    service: JobService,
    spec: CampaignSpec,
    *,
    distributed: bool = False,
    resume: bool = True,
    max_rounds: int | None = None,
    trial_timeout_seconds: float = 600.0,
    poll_interval_seconds: float = 0.1,
) -> dict[str, Any]:
    """Create or reuse the durable run job for a campaign.

    A completed or cancelled campaign is returned as a report with no new job. An
    already queued/leased run job is returned instead of starting a second coordinator.
    Concurrent callers are serialized by an atomic reservation in :class:`CampaignStore`;
    the winner creates a job under the reserved id and every loser returns that same job.
    """
    from uzen.core.campaign.engine import CampaignEngine
    from uzen.core.campaign.store import CampaignStore

    store = CampaignStore(service.root)
    request = campaign_run_request(
        spec,
        distributed=distributed,
        resume=resume,
        max_rounds=max_rounds,
        trial_timeout_seconds=trial_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
    deadline = time.monotonic() + RUN_ADMISSION_WAIT_SECONDS
    first_lookup = True
    while True:
        row = store.create_or_load(spec, resume=resume if first_lookup else True)
        first_lookup = False
        campaign_id = str(row["campaign_id"])

        def result(job: Mapping[str, Any] | None) -> dict[str, Any]:
            return {
                "campaign_id": campaign_id,
                "job": job,
                "campaign": CampaignEngine(service.root, store=store, job_service=service)
                .report(campaign_id)
                .to_dict(),
            }

        if row["status"] in ("completed", "cancelled") or row["cancel_requested"]:
            # Cancellation is a durable barrier: do not reopen a campaign that is still
            # settling its coordinator or trial jobs.
            return result(None)

        reservation = store.run_reservation(campaign_id)
        existing_job_id = str(reservation["job_id"])
        if existing_job_id:
            try:
                existing = service.get(existing_job_id)
            except JobNotFound:
                existing = None
            if existing is not None and str(existing.get("state")) not in TERMINAL_STATES:
                return result(existing)
            # The reservation is either terminal (a failed/paused run may be retried) or
            # names a job that never reached the queue. A fresh missing job belongs to a
            # submitter that is still between reservation and creation: wait for it
            # rather than stealing the slot. A stale one is safe to recover.
            reservation_age = time.time() - float(reservation["reserved_at"])
            if existing is None and reservation_age < RUN_RESERVATION_GRACE_SECONDS:
                if time.monotonic() >= deadline:
                    raise CampaignConflict(
                        "campaign run admission is still in progress; retry the submission",
                        campaign_id=campaign_id,
                    )
                time.sleep(0.02)
                continue
            if store.release_run(campaign_id, existing_job_id):
                continue
            if time.monotonic() >= deadline:
                raise CampaignConflict(
                    "campaign run admission changed while waiting; retry the submission",
                    campaign_id=campaign_id,
                )
            continue

        candidate_job_id = uuid.uuid4().hex
        if not store.reserve_run(campaign_id, candidate_job_id):
            if time.monotonic() >= deadline:
                raise CampaignConflict(
                    "campaign run admission is contended; retry the submission",
                    campaign_id=campaign_id,
                )
            time.sleep(0.01)
            continue

        try:
            job = service.submit(request, job_id=candidate_job_id)
        except Exception:
            store.release_run(campaign_id, candidate_job_id)
            raise
        store.link_job(campaign_id, str(job["id"]), candidate_digest=CAMPAIGN_RUN_CANDIDATE)
        return result(job)
