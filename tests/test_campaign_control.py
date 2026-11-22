"""N8 campaign control: durable background runs, cancellation and job-kind isolation."""

from __future__ import annotations

import json
import threading
import time

import pytest

from uzen.core.campaign import (
    CAMPAIGN_RUN_KIND,
    CampaignEngine,
    CampaignSpec,
    CampaignWorker,
    Trial,
    campaign_run_request,
    submit_campaign_run,
)
from uzen.core.campaign.control import validate_campaign_run_request
from uzen.core.campaign.distributed import CAMPAIGN_TRIAL_KIND, campaign_trial_request
from uzen.core.campaign.space import propose
from uzen.errors import CampaignConflict, InvalidOperationParameters
from uzen.infrastructure.storage.db import TERMINAL_STATES
from uzen.jobs.scheduler import JobScheduler
from uzen.jobs.service import JobService


def make_spec(**overrides):
    base = {
        "name": "n8-control",
        "family": "mlp",
        "mode": "architecture",
        "base_spec": {
            "input_dim": 64,
            "hidden_dims": [64, 64],
            "output_dim": 16,
            "activation": "relu",
        },
        "axes": [
            {
                "name": "activation",
                "kind": "choice",
                "values": ["relu", "gelu", "silu", "tanh"],
            },
            {"name": "input_dim", "kind": "int", "low": 8, "high": 128, "step": 8},
            {"name": "output_dim", "kind": "int", "low": 8, "high": 32, "step": 8},
        ],
        "objectives": [{"name": "parameters", "direction": "min"}],
        "budget": 4,
        "batch_size": 2,
        "strategy": "evolution",
        "seed": 3,
        "write_corpus": False,
    }
    base.update(overrides)
    return CampaignSpec.from_dict(base)


def first_candidate(spec):
    return propose(spec, [], round_index=0, count=1, ranked=[])[0]


def start_workers(root, *, coordinators=0, trials=0):
    stop = threading.Event()
    workers = [
        CampaignWorker(
            str(root),
            f"coordinator-{index}",
            job_kinds=(CAMPAIGN_RUN_KIND,),
            claim_wait_seconds=0.1,
        )
        for index in range(coordinators)
    ] + [
        CampaignWorker(
            str(root),
            f"trial-{index}",
            claim_wait_seconds=0.1,
        )
        for index in range(trials)
    ]
    for worker in workers:
        worker.warmup()
    threads = [
        threading.Thread(target=worker.run_forever, kwargs={"stop": stop}, daemon=True)
        for worker in workers
    ]
    for thread in threads:
        thread.start()
    return stop, threads


def stop_workers(stop, threads):
    stop.set()
    for thread in threads:
        thread.join(timeout=3.0)
    assert not any(thread.is_alive() for thread in threads)


def wait_for(predicate, *, timeout=10.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError("condition was not reached before the deadline")


def test_campaign_worker_kind_filter_does_not_claim_engineering_jobs(tmp_path):
    service = JobService(tmp_path)
    engineering = service.submit(
        {"task_scope": "inspection", "objective": "ordinary inspection job"}
    )
    worker = CampaignWorker(str(tmp_path), "trial-only", claim_wait_seconds=0.01)
    assert worker.job_kinds == (CAMPAIGN_TRIAL_KIND,)
    assert worker.run_once()["executed"] is None
    assert service.get(engineering["id"])["state"] == "QUEUED"

    any_worker = CampaignWorker(
        str(tmp_path), "plain-worker", job_kinds=None, claim_wait_seconds=0.01
    )
    any_worker.warmup()
    claimed = any_worker.run_once()
    assert claimed["executed"] == engineering["id"]


def test_campaign_run_job_executes_the_campaign_and_links_the_job(tmp_path):
    service = JobService(tmp_path)
    spec = make_spec(budget=2, batch_size=1)
    submitted = submit_campaign_run(service, spec)
    job = submitted["job"]
    assert job["state"] == "QUEUED"
    assert job["job_kind"] == CAMPAIGN_RUN_KIND
    assert "campaign_spec" in job["request"]["options"]

    worker = CampaignWorker(
        str(tmp_path),
        "coordinator",
        job_kinds=(CAMPAIGN_RUN_KIND,),
        claim_wait_seconds=0.01,
    )
    worker.warmup()
    assert worker.run_once()["executed"] == job["id"]

    completed = service.get(job["id"])
    assert completed["state"] == "SUCCEEDED"
    assert completed["result"]["task_kind"] == CAMPAIGN_RUN_KIND
    assert completed["result"]["status"] == "completed"
    assert completed["result"]["report"]["trials_evaluated"] == 2
    engine = CampaignEngine(tmp_path, job_service=service)
    linked = engine.jobs(spec.candidate_id)
    assert [item["id"] for item in linked] == [job["id"]]
    assert engine.get(spec.candidate_id).status == "completed"


def test_campaign_run_request_rejects_operations_and_bad_options(tmp_path):
    spec = make_spec()
    request = campaign_run_request(spec)
    request["operations"] = [{"op": "w.arithmetic.scale", "params": {"factor": 1.0}}]
    with pytest.raises(InvalidOperationParameters, match="engineering operations"):
        validate_campaign_run_request(request)

    request = campaign_run_request(spec)
    request["task_scope"] = "weight_modification"
    with pytest.raises(InvalidOperationParameters, match="inspection scope"):
        validate_campaign_run_request(request)

    request = campaign_run_request(spec)
    request["options"]["poll_interval_seconds"] = 0
    with pytest.raises(InvalidOperationParameters, match="poll interval"):
        validate_campaign_run_request(request)


def test_cancel_stops_a_distributed_campaign_and_its_jobs(tmp_path):
    spec = make_spec(budget=64, batch_size=4)
    service = JobService(tmp_path)
    stop, threads = start_workers(tmp_path, coordinators=1, trials=3)
    try:
        submitted = submit_campaign_run(
            service,
            spec,
            distributed=True,
            trial_timeout_seconds=30.0,
            poll_interval_seconds=0.02,
        )
        engine = CampaignEngine(tmp_path, job_service=service)
        wait_for(lambda: len(engine.get(spec.candidate_id).trials) >= 2)
        before = len(engine.get(spec.candidate_id).trials)
        cancelled = engine.cancel(spec.candidate_id)
        assert cancelled["campaign"]["status"] == "cancelled"
        assert cancelled["campaign"]["stop_reason"] == "cancelled"
        assert cancelled["campaign"]["trials_evaluated"] <= before + 4
        wait_for(
            lambda: service.get(submitted["job"]["id"])["state"] == "CANCELLED"
        )
        time.sleep(0.3)
        after = engine.get(spec.candidate_id)
        assert after.status == "cancelled"
        assert after.stop_reason == "cancelled"
        assert len(after.trials) == cancelled["campaign"]["trials_evaluated"]
        assert all(
            job["state"] in TERMINAL_STATES for job in engine.jobs(spec.candidate_id)
        )
    finally:
        stop_workers(stop, threads)


def test_cancel_is_idempotent_and_completed_campaign_is_a_conflict(tmp_path):
    service = JobService(tmp_path)
    spec = make_spec(budget=1, batch_size=1)
    engine = CampaignEngine(tmp_path, job_service=service)
    engine.store.create_or_load(spec)
    first = engine.cancel(spec.candidate_id)
    assert first["campaign"]["status"] == "cancelled"
    second = engine.cancel(spec.candidate_id)
    assert second["campaign"]["status"] == "cancelled"

    complete = make_spec(name="n8-completed", budget=1, batch_size=1)
    completed = CampaignEngine(tmp_path).run(complete)
    assert completed.status == "completed"
    with pytest.raises(CampaignConflict, match="already completed"):
        engine.cancel(complete.candidate_id)


def test_cancel_barrier_rejects_trial_inserts_after_the_request(tmp_path):
    spec = make_spec(budget=2, batch_size=1)
    engine = CampaignEngine(tmp_path)
    engine.store.create_or_load(spec)
    first = first_candidate(spec)
    assert engine.store.record_trial(
        spec.candidate_id,
        Trial(candidate=first, round_index=0, sequence=0, status="ok", metrics={}),
    )
    engine.cancel(spec.candidate_id)
    second = propose(
        spec, engine.store.trials(spec.candidate_id), round_index=1, count=1, ranked=[]
    )[0]
    assert not engine.store.record_trial(
        spec.candidate_id,
        Trial(candidate=second, round_index=1, sequence=0, status="ok", metrics={}),
    )
    assert len(engine.store.trials(spec.candidate_id)) == 1


def test_cancel_before_leased_execution_settles_and_releases_the_lease(tmp_path):
    service = JobService(tmp_path)
    spec = make_spec(budget=1, batch_size=1)
    row = service.submit(campaign_trial_request(spec, first_candidate(spec)))
    scheduler = JobScheduler(service.db, lease_seconds=30.0)
    from uzen.workers.protocol.messages import detect_capabilities

    claimed = scheduler.claim("leased-worker", capabilities=detect_capabilities())
    assert claimed is not None
    token = claimed["lease"]["token"]
    service.db.request_cancel(row["id"])
    completed = service.run(row["id"], worker_id="leased-worker", token=token)
    assert completed["state"] == "CANCELLED"
    assert service.db.lease_holder(row["id"]) is None
    assert service.db.dispatch_records(row["id"])[0]["outcome"] == "CANCELLED"


def test_late_worker_success_cannot_overwrite_cancellation(tmp_path):
    service = JobService(tmp_path)
    spec = make_spec(budget=1, batch_size=1)
    candidate = first_candidate(spec)
    row = service.submit(campaign_trial_request(spec, candidate))
    scheduler = JobScheduler(service.db, lease_seconds=30.0)
    from uzen.workers.protocol.messages import detect_capabilities

    claimed = scheduler.claim("late-worker", capabilities=detect_capabilities())
    assert claimed is not None
    token = claimed["lease"]["token"]
    service.db.request_cancel(row["id"])
    reported = service.db.finish_from_worker(
        row["id"],
        worker_id="late-worker",
        token=token,
        state="SUCCEEDED",
        result={"task_kind": "campaign_trial", "trial": {"status": "ok"}},
    )
    assert reported["status"] == "accepted"
    assert reported["state"] == "CANCELLED"
    assert service.get(row["id"])["state"] == "CANCELLED"
    assert service.db.dispatch_records(row["id"])[0]["outcome"] == "CANCELLED"


def test_rest_cancel_endpoint_cancels_the_run_job(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from uzen.interfaces.api.app import create_app

    service = JobService(tmp_path)
    spec = make_spec(budget=2, batch_size=1)
    client = TestClient(create_app(service=service, auto_run=False))
    submitted = client.post("/v1/campaigns", json={"spec": spec.to_dict(), "background": True})
    assert submitted.status_code == 200, submitted.text
    job_id = submitted.json()["job"]["id"]
    cancelled = client.post(f"/v1/campaigns/{spec.candidate_id}/cancel")
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["campaign"]["status"] == "cancelled"
    assert service.get(job_id)["state"] == "CANCELLED"
    linked = client.get(f"/v1/campaigns/{spec.candidate_id}/jobs")
    assert linked.status_code == 200
    assert [job["id"] for job in linked.json()["jobs"]] == [job_id]


def test_sdk_rest_and_mcp_can_submit_background_campaigns(tmp_path):
    from uzen.interfaces.mcp.server import MCPServer
    from uzen.interfaces.python_sdk.client import Uzen

    service = JobService(tmp_path)
    spec = make_spec(budget=1, batch_size=1)
    sdk = Uzen(root=tmp_path).run_campaign(spec.to_dict(), background=True)
    assert sdk["job"]["job_kind"] == CAMPAIGN_RUN_KIND

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from uzen.interfaces.api.app import create_app

    client = TestClient(create_app(service=service, auto_run=False))
    rest = client.post(
        "/v1/campaigns",
        json={"spec": make_spec(name="n8-rest", budget=1, batch_size=1).to_dict(), "background": True},
    )
    assert rest.status_code == 200, rest.text
    assert rest.json()["job"]["job_kind"] == CAMPAIGN_RUN_KIND

    mcp = MCPServer(root=tmp_path).handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "uzen.run_campaign",
                "arguments": {
                    "spec": make_spec(name="n8-mcp", budget=1, batch_size=1).to_dict(),
                    "background": True,
                },
            },
        }
    )
    mcp_payload = json.loads(mcp["result"]["content"][0]["text"])
    assert mcp_payload["job"]["job_kind"] == CAMPAIGN_RUN_KIND
    listed = MCPServer(root=tmp_path).handle(
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    )
    names = {tool["name"] for tool in listed["result"]["tools"]}
    assert {"uzen.cancel_campaign", "uzen.campaign_jobs"} <= names

    worker = CampaignWorker(
        str(tmp_path),
        "coordinator",
        job_kinds=(CAMPAIGN_RUN_KIND,),
        claim_wait_seconds=0.01,
    )
    worker.warmup()
    for _ in range(3):
        assert worker.run_once()["executed"] is not None

    assert CampaignEngine(tmp_path, job_service=service).get(spec.candidate_id).status == "completed"
    assert client.get(f"/v1/campaigns/{rest.json()['campaign_id']}").json()["status"] == "completed"
    jobs = client.get(f"/v1/campaigns/{spec.candidate_id}/jobs")
    assert jobs.status_code == 200
    assert jobs.json()["jobs"][0]["job_kind"] == CAMPAIGN_RUN_KIND
