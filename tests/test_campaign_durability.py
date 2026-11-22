"""N9 campaign durability: single-coordinator admission and lease keepalive."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from uzen.core.campaign import (
    CAMPAIGN_RUN_KIND,
    CampaignSpec,
    CampaignStore,
    submit_campaign_run,
)
from uzen.jobs.service import JobService
from uzen.workers.protocol.messages import detect_capabilities


def make_spec(**overrides):
    base = {
        "name": "n9-durability",
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
                "values": ["relu", "gelu"],
            },
            {"name": "input_dim", "kind": "int", "low": 8, "high": 64, "step": 8},
        ],
        "objectives": [{"name": "parameters", "direction": "min"}],
        "budget": 2,
        "batch_size": 1,
        "strategy": "grid",
        "seed": 3,
        "write_corpus": False,
    }
    base.update(overrides)
    return CampaignSpec.from_dict(base)


def wait_for(predicate, *, timeout=5.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError("condition was not reached before the deadline")


class SlowJobService(JobService):
    """A real service whose work deliberately outlives the initial lease."""

    delay_seconds = 1.2

    def _execute(self, job_id, request):
        time.sleep(self.delay_seconds)
        return {"ok": True, "job_id": job_id}


def test_concurrent_submissions_admit_exactly_one_coordinator(tmp_path):
    spec = make_spec()
    # Create the schema before the race so every thread only exercises admission.
    JobService(tmp_path)
    CampaignStore(tmp_path).create_or_load(spec)
    services = [JobService(tmp_path) for _ in range(4)]
    barrier = threading.Barrier(len(services))

    def submit(service):
        barrier.wait()
        return submit_campaign_run(service, spec)

    with ThreadPoolExecutor(max_workers=len(services)) as pool:
        results = list(pool.map(submit, services))

    job_ids = {str(result["job"]["id"]) for result in results}
    assert len(job_ids) == 1
    admitted = job_ids.pop()
    jobs = [job for job in services[0].list(limit=20) if job["job_kind"] == CAMPAIGN_RUN_KIND]
    assert [job["id"] for job in jobs] == [admitted]
    campaign = CampaignStore(tmp_path).campaign(spec.candidate_id)
    assert campaign["run_job_id"] == admitted


def test_cancelled_campaign_is_not_reopened(tmp_path):
    spec = make_spec(name="n9-cancelled-submit")
    service = JobService(tmp_path)
    store = CampaignStore(tmp_path)
    store.create_or_load(spec)
    store.request_cancel(spec.candidate_id)

    result = submit_campaign_run(service, spec)

    assert result["job"] is None
    assert result["campaign"]["status"] == "cancelling"
    assert store.run_job_id(spec.candidate_id) == ""


def test_stale_missing_reservation_is_recovered(tmp_path):
    spec = make_spec()
    service = JobService(tmp_path)
    store = CampaignStore(tmp_path)
    store.create_or_load(spec)
    assert store.reserve_run(spec.candidate_id, "missing-job")
    with store._connect() as connection:
        connection.execute(
            "UPDATE campaigns SET run_reserved_at = ? WHERE campaign_id = ?",
            (time.time() - 60.0, spec.candidate_id),
        )

    result = submit_campaign_run(service, spec)

    assert result["job"] is not None
    assert result["job"]["id"] != "missing-job"
    assert store.campaign(spec.candidate_id)["run_job_id"] == result["job"]["id"]


def test_leased_execution_renews_its_lease_while_work_runs(tmp_path):
    service = SlowJobService(tmp_path)
    job = service.submit(
        {"task_scope": "inspection", "objective": "slow inspection"},
    )
    claimed = service.db.claim_job(
        "slow-worker",
        capabilities=detect_capabilities(),
        lease_seconds=0.5,
    )
    assert claimed is not None and claimed.id == job["id"]

    outcome: dict[str, object] = {}

    def run():
        outcome.update(
            service.run(
                job["id"],
                worker_id="slow-worker",
                token=claimed.dispatch_token,
                lease_seconds=0.5,
            )
        )

    thread = threading.Thread(target=run)
    thread.start()
    wait_for(lambda: service.get(job["id"])["state"] == "RUNNING")

    # Without a keepalive this is well past the 0.5s lease and the reaper would
    # return the job to the queue while the original execution is still running.
    time.sleep(0.7)
    service.db.reap_expired_leases()
    assert service.get(job["id"])["state"] != "QUEUED"

    thread.join(timeout=5.0)
    assert not thread.is_alive()
    final = service.get(job["id"])
    assert final["state"] == "SUCCEEDED"
    assert final["attempts"] == 1
    assert outcome["state"] == "SUCCEEDED"
