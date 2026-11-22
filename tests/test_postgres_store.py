"""P11: PostgreSQL control-plane store and durable queue against a real server.

The fixture starts a real PostgreSQL server (``pgserver`` ships the PostgreSQL
binaries); when neither ``pgserver`` nor ``psycopg`` is installed the tests skip
instead of pretending the backend works.
"""

from __future__ import annotations

import threading
import warnings

import pytest

from uzen.infrastructure.storage.postgres import PostgresDatabase
from uzen.infrastructure.storage.postgres_queue import PostgresQueue
from uzen.jobs.scheduling import CapabilityRequirements

TRUNCATE = (
    "TRUNCATE jobs, job_events, artifacts_index, models, workers, worker_capabilities, "
    "job_requirements, dispatch_records, leases, idempotency, api_tokens, artifact_acl "
    "RESTART IDENTITY CASCADE"
)


@pytest.fixture(scope="module")
def pg_dsn(tmp_path_factory):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pgserver = pytest.importorskip("pgserver")
        pytest.importorskip("psycopg")
        directory = tmp_path_factory.mktemp("pgdata")
        server = pgserver.get_server(directory)
    try:
        yield server.get_uri()
    finally:
        server.cleanup()


@pytest.fixture()
def db(pg_dsn):
    database = PostgresDatabase(pg_dsn)
    with database._conn.transaction():
        database._execute(TRUNCATE)
    yield database
    database.close()


def test_job_lifecycle_and_events_on_postgres(db: PostgresDatabase) -> None:
    created = db.create_job({"task_scope": "inspection", "objective": "pg"}, project_id="alpha")
    assert created.project_id == "alpha"
    assert db.project_of_job(created.id) == "alpha"
    db.transition(created.id, "RUNNING")
    db.set_result(created.id, {"ok": True})
    finished = db.get_job(created.id)
    assert finished.state == "SUCCEEDED"
    assert finished.result == {"ok": True}
    kinds = [event["kind"] for event in db.events(created.id)]
    assert kinds[0] == "created"
    assert "running" in kinds and "completed" in kinds
    assert [event["seq"] for event in db.events(created.id)] == list(range(1, len(kinds) + 1))
    assert db.stats()["jobs_by_state"]["SUCCEEDED"] == 1


def test_idempotency_key_replays_and_conflicts(db: PostgresDatabase) -> None:
    request = {"task_scope": "inspection", "objective": "same"}
    first = db.create_job(request, idempotency_key="pg-key")
    second = db.create_job(request, idempotency_key="pg-key")
    assert first.id == second.id
    with pytest.raises(Exception) as excinfo:
        db.create_job({**request, "objective": "different"}, idempotency_key="pg-key")
    assert "idempotency" in str(excinfo.value).lower()


def test_two_connections_never_claim_the_same_job(db: PostgresDatabase, pg_dsn: str) -> None:
    ids = [db.create_job({"task_scope": "inspection"}).id for _ in range(12)]
    other = PostgresDatabase(pg_dsn)
    results: dict[int, list[str]] = {}
    errors: list[BaseException] = []

    def consume(index: int, database: PostgresDatabase) -> None:
        claimed: list[str] = []
        try:
            while True:
                row = database.claim_job(f"worker-{index}", lease_seconds=120)
                if row is None:
                    break
                claimed.append(row.id)
        except BaseException as exc:  # surface thread failures in the main thread
            errors.append(exc)
        results[index] = claimed

    threads = [
        threading.Thread(target=consume, args=(0, db)),
        threading.Thread(target=consume, args=(1, other)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    other.close()
    assert not errors
    claimed_all = results[0] + results[1]
    assert sorted(claimed_all) == sorted(ids)
    assert len(set(claimed_all)) == len(ids)
    for job_id in ids:
        job = db.get_job(job_id)
        assert job.state == "DISPATCHED"
        assert job.attempts == 1
        assert db.lease_holder(job_id)["worker_id"] == job.worker_id


def test_capability_and_resource_filtering(db: PostgresDatabase) -> None:
    requirements = CapabilityRequirements(
        operation=("w.arithmetic.scale",), min_cpu_count=4
    )
    db.create_job({"task_scope": "weight_modification"}, requirements=requirements)
    weak = {"cpu_count": 1, "supported_operations": ["w.arithmetic.scale"], "strict_plus": True}
    assert db.claim_job("weak-worker", capabilities=weak) is None
    narrow = {"cpu_count": 8, "supported_operations": ["w.merging.linear"], "strict_plus": True}
    assert db.claim_job("narrow-worker", capabilities=narrow) is None
    capable = {"cpu_count": 8, "supported_operations": ["w.arithmetic.scale"], "strict_plus": True}
    claimed = db.claim_job("capable-worker", capabilities=capable)
    assert claimed is not None
    assert claimed.worker_id == "capable-worker"
    assert db.worker_capabilities("capable-worker")["operation"] == ["w.arithmetic.scale"]


def test_worker_reports_are_exactly_once(db: PostgresDatabase) -> None:
    db.create_job({"task_scope": "inspection"})
    claimed = db.claim_job("reporter")
    assert claimed is not None
    token = claimed.dispatch_token
    accepted = db.finish_from_worker(
        claimed.id, worker_id="reporter", token=token, state="SUCCEEDED", result={"n": 1}
    )
    assert accepted["status"] == "accepted"
    duplicate = db.finish_from_worker(
        claimed.id, worker_id="reporter", token=token, state="SUCCEEDED", result={"n": 1}
    )
    assert duplicate["status"] in ("duplicate", "rejected")
    stale = db.finish_from_worker(
        claimed.id, worker_id="reporter", token="not-the-token", state="FAILED"
    )
    assert stale["status"] == "stale"
    assert db.get_job(claimed.id).state == "SUCCEEDED"


def test_cancellation_wins_over_a_late_success(db: PostgresDatabase) -> None:
    db.create_job({"task_scope": "inspection"})
    claimed = db.claim_job("slow-worker")
    token = claimed.dispatch_token
    cancelled = db.request_cancel(claimed.id)
    assert cancelled.state == "CANCEL_REQUESTED"
    outcome = db.finish_from_worker(
        claimed.id, worker_id="slow-worker", token=token, state="SUCCEEDED", result={"late": True}
    )
    assert outcome["status"] == "accepted"
    assert outcome["state"] == "CANCELLED"
    assert db.get_job(claimed.id).state == "CANCELLED"


def test_expired_lease_is_recovered(db: PostgresDatabase) -> None:
    db.create_job({"task_scope": "inspection"})
    claimed = db.claim_job("doomed-worker", lease_seconds=0.01, now=1000.0)
    assert claimed is not None
    outcomes = db.reap_expired_leases(now=1001.0)
    assert outcomes and outcomes[0]["action"] == "requeued"
    again = db.claim_job("fresh-worker", now=1002.0)
    assert again is not None and again.id == claimed.id and again.attempts == 2


def test_tokens_and_artifact_acl_roundtrip(db: PostgresDatabase) -> None:
    minted = db.create_api_token("alpha", ["jobs:read", "jobs:submit"], description="ci")
    record = db.authenticate_api_token(minted["token"])
    assert record is not None and record["project_id"] == "alpha"
    assert set(record["scopes"]) == {"jobs:read", "jobs:submit"}
    assert db.revoke_api_token(minted["token_id"]) is True
    assert db.authenticate_api_token(minted["token"]) is None
    db.index_artifact(artifact_id="art_pg", path="/tmp/art_pg", project_id="alpha")
    assert db.artifact_project("art_pg") == "alpha"
    db.grant_artifact_access("art_pg", "beta", permission="read")
    assert db.artifact_acls("art_pg")[0]["project_id"] == "beta"
    assert db.revoke_artifact_access("art_pg", "beta") == 1


def test_queue_claim_ack_nack_and_reap(pg_dsn: str, db: PostgresDatabase) -> None:
    queue = PostgresQueue(pg_dsn)
    other = PostgresQueue(pg_dsn)
    try:
        with queue._conn.transaction():
            with queue._conn.cursor() as cursor:
                cursor.execute("TRUNCATE uzen_queue RESTART IDENTITY")
        ids = [queue.enqueue("jobs", {"n": n}, priority=n) for n in range(6)]
        assert queue.enqueue("dedupe", {"n": 0}, idempotency_key="dedupe") == queue.enqueue(
            "dedupe", {"n": 0}, idempotency_key="dedupe"
        )

        claimed: list[int] = []
        lock = threading.Lock()

        def consume(q: PostgresQueue, worker: str) -> None:
            while True:
                batch = q.claim(worker, topics=["jobs"], lease_seconds=60, limit=2)
                if not batch:
                    return
                with lock:
                    claimed.extend(message.id for message in batch)
                for message in batch:
                    if message.payload["n"] % 2:
                        # Parked far in the future so the other consumer cannot race
                        # this test by re-claiming it immediately.
                        q.nack(message.id, worker, retry_in=3600, error="odd")
                    else:
                        assert q.ack(message.id, worker) is True

        threads = [
            threading.Thread(target=consume, args=(queue, "q1")),
            threading.Thread(target=consume, args=(other, "q2")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        jobs_claimed = [message for message in claimed if message in ids]
        assert sorted(jobs_claimed) == sorted(ids)
        assert len(jobs_claimed) == len(set(jobs_claimed))
        depth = queue.depth(topic="jobs")
        assert depth["done"] == 3 and depth["ready"] == 3
        assert queue.depth(topic="dedupe") == {"ready": 1}
        # A message that is not leased by this worker cannot be nacked by it.
        assert queue.nack(ids[1], "q1")["status"] == "not_leased"
        assert queue.reap()["total"] == 0
        assert queue.describe()["table"] == "uzen_queue"
    finally:
        other.close()
        queue.close()


def test_job_service_runs_a_real_job_on_postgres(pg_dsn: str, tmp_path) -> None:
    from uzen.core.nws.families import build_architecture
    from uzen.interfaces.api.app import create_app
    from uzen.jobs.service import JobService

    service = JobService(tmp_path / "workspace", db_path=pg_dsn)
    with service.db._conn.transaction():
        service.db._execute(TRUNCATE)
    definition = build_architecture("mlp", dict(input_dim=6, hidden_dims=[8], output_dim=3))
    package = definition.build_package(seed=4)
    artifact = service.store.put(package, operation="init").artifact_id
    job = service.submit(
        {
            "task_scope": "weight_modification",
            "inputs": [{"name": "base", "artifact_id": artifact}],
            "operations": [{"op": "w.arithmetic.scale", "params": {"factor": 0.5}}],
        },
        project_id="alpha",
    )
    finished = service.run(job["id"])
    assert finished["state"] == "SUCCEEDED", finished
    assert finished["project_id"] == "alpha"
    page = service.list_page(limit=10)
    assert [row["id"] for row in page["jobs"]] == [job["id"]]
    assert service.db.stats()["db"] == "postgresql"

    from fastapi.testclient import TestClient

    client = TestClient(create_app(service=service, auto_run=False))
    listed = client.get("/v1/jobs", params={"limit": 5}).json()
    assert listed["jobs"][0]["id"] == job["id"]
    assert listed["has_more"] is False
    service.db.close()
