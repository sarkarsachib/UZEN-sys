"""N7 distributed campaign trials over the N4 queue."""

from __future__ import annotations

import threading
import time

import pytest

from uzen.core.campaign import (
    CampaignEngine,
    CampaignSpec,
    CampaignTimeout,
    CampaignWorker,
    DistributedTrialExecutor,
    campaign_trial_request,
)
from uzen.core.campaign.evaluate import evaluate_candidate
from uzen.core.campaign.space import propose
from uzen.errors import InvalidOperationParameters
from uzen.jobs.service import JobService


def make_spec(**overrides):
    base = {
        "name": "n7-distributed",
        "family": "mlp",
        "mode": "architecture",
        "base_spec": {
            "input_dim": 8,
            "hidden_dims": [8],
            "output_dim": 4,
            "activation": "relu",
        },
        "axes": [
            {
                "name": "activation",
                "kind": "choice",
                "values": ["relu", "gelu", "silu", "tanh"],
            }
        ],
        "objectives": [
            {"name": "parameters", "direction": "min"},
            {"name": "output_diversity", "direction": "max"},
        ],
        "budget": 4,
        "batch_size": 2,
        "strategy": "grid",
        "seed": 7,
        "write_corpus": False,
    }
    base.update(overrides)
    return CampaignSpec.from_dict(base)


def first_candidate(spec):
    return propose(spec, [], round_index=0, count=1, ranked=[])[0]


def test_campaign_trial_job_is_read_only_and_matches_local_evaluation(tmp_path):
    spec = make_spec(budget=1, batch_size=1)
    candidate = first_candidate(spec)
    service = JobService(tmp_path)
    row = service.submit(
        campaign_trial_request(spec, candidate),
        idempotency_key="n7-read-only",
    )
    assert row["task_scope"] == "inspection"
    completed = service.run(row["id"])
    assert completed["state"] == "SUCCEEDED"
    trial = completed["result"]["trial"]
    expected = evaluate_candidate(spec, candidate)
    assert trial["status"] == "ok"
    assert trial["metrics"] == expected["metrics"]
    assert trial["package_fingerprint"] == expected["package_fingerprint"]
    assert trial["graph_digest"] == expected["graph_digest"]
    # A campaign trial is an evaluation, not a new model artifact.
    assert service.store.list() == []


def test_campaign_trial_rejects_a_forged_candidate_digest(tmp_path):
    spec = make_spec(budget=1, batch_size=1)
    candidate = first_candidate(spec)
    request = campaign_trial_request(spec, candidate)
    request["options"]["candidate"]["digest"] = "0" * 64
    with pytest.raises(InvalidOperationParameters, match="digest does not match"):
        JobService(tmp_path).submit(request)


def test_campaign_trial_requires_inspection_scope(tmp_path):
    spec = make_spec(budget=1, batch_size=1)
    candidate = first_candidate(spec)
    request = campaign_trial_request(spec, candidate)
    request["task_scope"] = "weight_modification"
    with pytest.raises(InvalidOperationParameters, match="inspection scope"):
        JobService(tmp_path).submit(request)


def _start_workers(root, count, stop):
    workers = [
        CampaignWorker(str(root), f"campaign-worker-{index}", claim_wait_seconds=0.2)
        for index in range(count)
    ]
    threads = [
        threading.Thread(target=worker.run_forever, kwargs={"stop": stop}, daemon=True)
        for worker in workers
    ]
    for thread in threads:
        thread.start()
    return threads


def _stop_workers(threads, stop):
    stop.set()
    for thread in threads:
        thread.join(timeout=3.0)


def test_distributed_campaign_matches_the_local_deterministic_result(tmp_path):
    spec = make_spec()
    local_root = tmp_path / "local"
    distributed_root = tmp_path / "distributed"
    local = CampaignEngine(local_root).run(spec)

    stop = threading.Event()
    threads = _start_workers(distributed_root, 2, stop)
    try:
        time.sleep(0.2)
        control = JobService(distributed_root)
        executor = DistributedTrialExecutor(
            control, timeout_seconds=20.0, poll_interval_seconds=0.02
        )
        report = CampaignEngine(distributed_root, trial_executor=executor).run(spec)
    finally:
        _stop_workers(threads, stop)

    assert report.status == "completed"
    assert report.stop_reason == "budget"
    assert len(report.trials) == 4
    assert report.deterministic_digest() == local.deterministic_digest()
    jobs = control.list(limit=20)
    assert len(jobs) == 4
    assert all(job["state"] == "SUCCEEDED" for job in jobs)
    assert all(job["worker_id"] for job in jobs)
    assert {job["worker_id"] for job in jobs}.issubset(
        {"campaign-worker-0", "campaign-worker-1"}
    )


def test_sdk_distributed_campaign_uses_worker_queue(tmp_path):
    from uzen.interfaces.python_sdk.client import Uzen

    spec = make_spec(budget=2, batch_size=2)
    stop = threading.Event()
    threads = _start_workers(tmp_path, 1, stop)
    try:
        time.sleep(0.2)
        report = Uzen(root=tmp_path).run_campaign(
            spec.to_dict(), distributed=True, trial_timeout_seconds=20.0
        )
    finally:
        _stop_workers(threads, stop)
    assert report["status"] == "completed"
    assert report["trials_evaluated"] == 2


def test_rest_distributed_campaign_uses_worker_queue(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from uzen.interfaces.api.app import create_app

    spec = make_spec(budget=2, batch_size=2)
    stop = threading.Event()
    threads = _start_workers(tmp_path, 1, stop)
    try:
        time.sleep(0.2)
        response = TestClient(create_app(root=tmp_path)).post(
            "/v1/campaigns",
            json={
                "spec": spec.to_dict(),
                "distributed": True,
                "trial_timeout_seconds": 20.0,
                "poll_interval_seconds": 0.02,
            },
        )
    finally:
        _stop_workers(threads, stop)
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "completed"
    assert response.json()["trials_evaluated"] == 2


def test_mcp_distributed_campaign_uses_worker_queue(tmp_path):
    import json

    from uzen.interfaces.mcp.server import MCPServer

    spec = make_spec(budget=2, batch_size=2)
    stop = threading.Event()
    threads = _start_workers(tmp_path, 1, stop)
    try:
        time.sleep(0.2)
        response = MCPServer(root=tmp_path).handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "uzen.run_campaign",
                    "arguments": {
                        "spec": spec.to_dict(),
                        "distributed": True,
                        "trial_timeout_seconds": 20.0,
                        "poll_interval_seconds": 0.02,
                    },
                },
            }
        )
    finally:
        _stop_workers(threads, stop)
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["status"] == "completed"
    assert payload["trials_evaluated"] == 2


def test_rest_distributed_timeout_is_a_typed_504(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from uzen.interfaces.api.app import create_app

    response = TestClient(create_app(root=tmp_path)).post(
        "/v1/campaigns",
        json={
            "spec": make_spec(budget=1, batch_size=1).to_dict(),
            "distributed": True,
            "trial_timeout_seconds": 0.2,
            "poll_interval_seconds": 0.02,
        },
    )
    assert response.status_code == 504
    assert response.json()["detail"]["code"] == "campaign.timeout"


def test_distributed_timeout_pauses_and_resume_reuses_queued_trials(tmp_path):
    spec = make_spec(budget=2, batch_size=2)
    local = CampaignEngine(tmp_path / "local").run(spec)
    root = tmp_path / "distributed"
    control = JobService(root)
    engine = CampaignEngine(
        root,
        trial_executor=DistributedTrialExecutor(
            control, timeout_seconds=0.2, poll_interval_seconds=0.02
        ),
    )
    with pytest.raises(CampaignTimeout):
        engine.run(spec)
    paused = engine.get(spec.candidate_id)
    assert paused.status == "paused"
    assert paused.stop_reason == "worker_timeout"
    assert len(control.list(limit=10)) == 2

    stop = threading.Event()
    threads = _start_workers(root, 1, stop)
    try:
        executor = DistributedTrialExecutor(
            control, timeout_seconds=20.0, poll_interval_seconds=0.02
        )
        resumed = CampaignEngine(root, trial_executor=executor).run(spec)
    finally:
        _stop_workers(threads, stop)

    assert resumed.status == "completed"
    assert len(resumed.trials) == 2
    assert resumed.deterministic_digest() == local.deterministic_digest()
