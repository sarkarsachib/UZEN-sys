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

from typing import Any, Mapping

from uzen.core.campaign.distributed import CAMPAIGN_RUN_KIND
from uzen.core.campaign.spec import CampaignSpec
from uzen.errors import InvalidOperationParameters, JobNotFound
from uzen.infrastructure.storage.db import TERMINAL_STATES
from uzen.jobs.service import JobService

CAMPAIGN_RUN_CANDIDATE = "__campaign_run__"


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
    """
    from uzen.core.campaign.engine import CampaignEngine
    from uzen.core.campaign.store import CampaignStore

    store = CampaignStore(service.root)
    row = store.create_or_load(spec, resume=resume)
    campaign_id = str(row["campaign_id"])
    if row["status"] in ("completed", "cancelled"):
        return {
            "campaign_id": campaign_id,
            "job": None,
            "campaign": CampaignEngine(service.root, store=store, job_service=service)
            .report(campaign_id)
            .to_dict(),
        }

    existing_job_id = store.run_job_id(campaign_id)
    if existing_job_id:
        try:
            existing = service.get(existing_job_id)
        except JobNotFound:
            existing = None
        if existing is not None and str(existing.get("state")) not in TERMINAL_STATES:
            return {
                "campaign_id": campaign_id,
                "job": existing,
                "campaign": CampaignEngine(service.root, store=store, job_service=service)
                .report(campaign_id)
                .to_dict(),
            }

    request = campaign_run_request(
        spec,
        distributed=distributed,
        resume=resume,
        max_rounds=max_rounds,
        trial_timeout_seconds=trial_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
    # A campaign may legitimately need a new run job after a prior run failed or was
    # paused. ``run_job_id`` prevents a second live coordinator; no request-level
    # idempotency key is used because the same specification can be retried.
    job = service.submit(request)
    store.set_run_job_id(campaign_id, str(job["id"]))
    store.link_job(campaign_id, str(job["id"]), candidate_digest=CAMPAIGN_RUN_CANDIDATE)
    return {
        "campaign_id": campaign_id,
        "job": job,
        "campaign": CampaignEngine(service.root, store=store, job_service=service)
        .report(campaign_id)
        .to_dict(),
    }
