"""Distributed queue and scheduler: capability matching, leases, idempotent dispatch."""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from uzen.core.nws.families import build_architecture
from uzen.errors import JobStateError
from uzen.infrastructure.storage.db import Database
from uzen.infrastructure.storage.redis_queue import (
    RedisDurableQueue,
    RedisProtocolError,
    RedisQueueConfig,
    RespConnection,
    encode_command,
)
from uzen.jobs.scheduler import JobScheduler
from uzen.jobs.scheduling import (
    CapabilityRequirements,
    capability_index,
    explain_assignment,
    match_worker,
    requirements_from_request,
)
from uzen.jobs.service import JobService


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #


def _capabilities(**overrides: Any) -> dict[str, Any]:
    base = {
        "cpu_count": 4,
        "memory_bytes": 8 * 1024**3,
        "device": "cpu",
        "cuda_available": False,
        "supported_operations": ["w.arithmetic.scale", "w.merge.linear"],
        "supported_operators": ["linear", "layernorm"],
        "supported_architectures": ["mlp", "transformer_decoder"],
        "artifact_transport": ["shared_filesystem", "inline_tensors"],
        "representations": ["dense_blocked"],
        "strict_plus": True,
    }
    base.update(overrides)
    return base


@pytest.fixture()
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "control.sqlite3")
    yield database
    database.close()


@pytest.fixture()
def service(tmp_path: Path) -> JobService:
    return JobService(tmp_path / "plane")


def _scale_request(artifact_id: str, factor: float = 0.5, **extra: Any) -> dict[str, Any]:
    request = {
        "task_scope": "weight_modification",
        "objective": f"scale by {factor}",
        "inputs": [{"name": "base", "artifact_id": artifact_id}],
        "operations": [{"op": "w.arithmetic.scale", "params": {"factor": factor}}],
    }
    request.update(extra)
    return request


def _base_artifact(service: JobService) -> str:
    definition = build_architecture("mlp", dict(input_dim=8, hidden_dims=[16], output_dim=4))
    record = service.store.put(definition.build_package(seed=5), operation="init")
    return record.artifact_id


# --------------------------------------------------------------------------- #
# capability model
# --------------------------------------------------------------------------- #


def test_capability_index_maps_the_advertised_document():
    index = capability_index(_capabilities(device="cuda", cuda_available=True))
    assert index["operation"] == ("w.arithmetic.scale", "w.merge.linear")
    assert index["family"] == ("mlp", "transformer_decoder")
    assert index["device"] == ("cpu", "cuda")
    assert index["policy"] == ("STRICT_PLUS",)
    assert index["transport"] == ("inline_tensors", "shared_filesystem")


def test_requirements_are_derived_from_the_request_it_will_run():
    request = {
        "task_scope": "architecture_synthesis",
        "objective": "make a cnn",
        "operations": [{"op": "arch.synthesize", "params": {}}],
        "options": {"family": "cnn"},
        "compute_requirements": {"device": "cuda", "min_cpu_count": 8, "min_memory_bytes": 16},
        "evaluation_plan": {"graph": {"family": "cnn"}},
    }
    requirements = requirements_from_request(request)
    assert requirements.family == ("cnn",)
    assert requirements.device == ("cuda",)
    assert requirements.operation == ("arch.synthesize",)
    assert requirements.min_cpu_count == 8
    assert requirements.min_memory_bytes == 16
    assert requirements.policy == ("STRICT_PLUS",)


def test_requirements_default_to_policy_only_when_the_request_is_silent():
    requirements = requirements_from_request({"task_scope": "inspection", "objective": "look"})
    assert requirements.policy == ("STRICT_PLUS",)
    assert requirements.family == () and requirements.device == ()


def test_match_worker_explains_each_missing_capability():
    requirements = CapabilityRequirements(
        family=("cnn",), device=("cuda",), operation=("arch.synthesize",), min_cpu_count=64
    )
    decision = match_worker(requirements, _capabilities(), worker_id="w1")
    assert decision.eligible is False
    assert decision.missing["family"] == ["cnn"]
    assert decision.missing["device"] == ["cuda"]
    assert decision.missing["operation"] == ["arch.synthesize"]
    assert decision.missing["cpu_count"] == ["64"]
    assert len(decision.reasons) == 4


def test_match_worker_accepts_any_of_transport_and_respects_budgets():
    requirements = CapabilityRequirements(
        transport=("object_store", "inline_tensors"), min_memory_bytes=1024
    )
    assert match_worker(requirements, _capabilities()).eligible is True
    strict = CapabilityRequirements(transport=("object_store",))
    assert match_worker(strict, _capabilities()).eligible is False


def test_explain_assignment_lists_considered_workers():
    workers = [
        {"worker_id": "a", "capabilities": _capabilities()},
        {"worker_id": "b", "capabilities": _capabilities(supported_architectures=["cnn"])},
    ]
    explanation = explain_assignment(CapabilityRequirements(family=("cnn",)), workers)
    assert explanation["routable"] is True
    assert explanation["eligible_workers"] == ["b"]
    assert [entry["worker_id"] for entry in explanation["considered"]] == ["a", "b"]


# --------------------------------------------------------------------------- #
# capability-aware claiming
# --------------------------------------------------------------------------- #


def test_claim_skips_a_job_the_worker_cannot_execute(db: Database):
    cnn_job = db.create_job(
        {"task_scope": "architecture_synthesis", "options": {"family": "cnn"}},
        requirements=CapabilityRequirements(family=("cnn",)),
    )
    mlp_job = db.create_job(
        {"task_scope": "architecture_synthesis", "options": {"family": "mlp"}},
        requirements=CapabilityRequirements(family=("mlp",)),
    )
    claimed = db.claim_job("w-mlp", capabilities=_capabilities())
    assert claimed is not None and claimed.id == mlp_job.id
    assert db.claim_job("w-mlp", capabilities=_capabilities()) is None
    assert db.get_job(cnn_job.id).state == "QUEUED"


def test_claim_respects_numeric_budgets(db: Database):
    db.create_job({"task_scope": "inspection"}, requirements=CapabilityRequirements(min_cpu_count=16))
    assert db.claim_job("small", capabilities=_capabilities(cpu_count=4)) is None
    claimed = db.claim_job("big", capabilities=_capabilities(cpu_count=32))
    assert claimed is not None and claimed.attempts == 1


def test_claim_returns_a_token_and_records_the_attempt(db: Database):
    job = db.create_job(
        {"task_scope": "inspection"}, requirements=CapabilityRequirements(policy=("STRICT_PLUS",))
    )
    claimed = db.claim_job("w1", capabilities=_capabilities(), lease_seconds=30)
    assert claimed is not None and claimed.id == job.id
    assert claimed.state == "DISPATCHED"
    assert claimed.attempts == 1
    assert claimed.dispatch_token
    assert claimed.worker_id == "w1"
    assert db.lease_holder(job.id)["token"] == claimed.dispatch_token
    records = db.dispatch_records(job.id)
    assert records[0]["outcome"] == "dispatched" and records[0]["worker_id"] == "w1"


def test_two_database_connections_never_claim_the_same_job(tmp_path: Path):
    path = tmp_path / "shared.sqlite3"
    first = Database(path)
    second = Database(path)
    try:
        jobs = [
            first.create_job(
                {"task_scope": "inspection", "objective": f"job {index}"},
                requirements=CapabilityRequirements(policy=("STRICT_PLUS",)),
            )
            for index in range(20)
        ]
        claimed: list[str] = []
        lock = threading.Lock()

        def worker(connection: Database, worker_id: str) -> None:
            local: list[str] = []
            while True:
                row = connection.claim_job(worker_id, capabilities=_capabilities())
                if row is None:
                    break
                local.append(row.id)
            with lock:
                claimed.extend(local)

        threads = [
            threading.Thread(target=worker, args=(first, "a")),
            threading.Thread(target=worker, args=(second, "b")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert sorted(claimed) == sorted(job.id for job in jobs)
        assert len(claimed) == len(set(claimed))
    finally:
        first.close()
        second.close()


# --------------------------------------------------------------------------- #
# leases, expiry and attempts
# --------------------------------------------------------------------------- #


def test_renew_requires_the_token_and_a_held_lease(db: Database):
    db.create_job({"task_scope": "inspection"}, requirements=CapabilityRequirements())
    claimed = db.claim_job("w1", capabilities=_capabilities(), lease_seconds=10, now=1000.0)
    assert claimed is not None
    assert db.renew_lease(claimed.id, "w1", lease_seconds=10, token="wrong", now=1001.0) is False
    assert db.renew_lease(claimed.id, "w2", lease_seconds=10, token=claimed.dispatch_token,
                          now=1001.0) is False
    assert db.renew_lease(claimed.id, "w1", lease_seconds=30, token=claimed.dispatch_token,
                          now=1001.0) is True
    assert db.get_job(claimed.id).lease_expires_at == pytest.approx(1031.0)


def test_reap_requeues_an_expired_lease_without_a_second_token(db: Database):
    db.create_job({"task_scope": "inspection"}, requirements=CapabilityRequirements())
    claimed = db.claim_job("w-doomed", capabilities=_capabilities(), lease_seconds=5, now=100.0)
    assert claimed is not None
    assert db.reap_expired_leases(now=104.0) == []
    outcomes = db.reap_expired_leases(now=200.0)
    assert outcomes[0]["action"] == "requeued"
    requeued = db.get_job(claimed.id)
    assert requeued.state == "QUEUED"
    assert requeued.dispatch_token is None
    assert requeued.worker_id is None
    second = db.claim_job("w-replacement", capabilities=_capabilities(), now=201.0)
    assert second is not None and second.id == claimed.id and second.attempts == 2
    assert second.dispatch_token != claimed.dispatch_token


def test_attempts_are_exhausted_into_a_typed_failure(db: Database):
    job = db.create_job(
        {"task_scope": "inspection"},
        requirements=CapabilityRequirements(),
        max_attempts=2,
    )
    for attempt in range(2):
        claimed = db.claim_job("w1", capabilities=_capabilities(), lease_seconds=1, now=100.0 * (attempt + 1))
        assert claimed is not None
        db.reap_expired_leases(now=100.0 * (attempt + 1) + 2)
    final = db.get_job(job.id)
    assert final.state == "FAILED"
    assert final.error["error"]["code"] == "job.lease_expired"
    assert final.error["error"]["details"]["attempts"] == 2


def test_cancel_during_a_lease_cancels_instead_of_requeueing(db: Database):
    job = db.create_job({"task_scope": "inspection"}, requirements=CapabilityRequirements())
    db.claim_job("w1", capabilities=_capabilities(), lease_seconds=1, now=10.0)
    db.request_cancel(job.id)
    outcomes = db.reap_expired_leases(now=50.0)
    assert outcomes[0]["action"] == "cancelled"
    assert db.get_job(job.id).state == "CANCELLED"


# --------------------------------------------------------------------------- #
# idempotent reporting
# --------------------------------------------------------------------------- #


def test_worker_report_is_applied_once(db: Database):
    job = db.create_job({"task_scope": "inspection"}, requirements=CapabilityRequirements())
    claimed = db.claim_job("w1", capabilities=_capabilities())
    assert claimed is not None
    first = db.finish_from_worker(
        job.id, worker_id="w1", token=claimed.dispatch_token, state="SUCCEEDED",
        result={"output_artifact_id": "a" * 64},
    )
    assert first["status"] == "accepted"
    assert db.get_job(job.id).state == "SUCCEEDED"
    second = db.finish_from_worker(
        job.id, worker_id="w1", token=claimed.dispatch_token, state="FAILED",
        error={"code": "worker.retry"},
    )
    assert second["status"] == "duplicate"
    assert second["recorded_outcome"] == "SUCCEEDED"
    assert db.get_job(job.id).state == "SUCCEEDED"


def test_a_superseded_worker_cannot_publish_a_result(db: Database):
    job = db.create_job({"task_scope": "inspection"}, requirements=CapabilityRequirements())
    first = db.claim_job("w1", capabilities=_capabilities(), lease_seconds=1, now=10.0)
    assert first is not None
    db.reap_expired_leases(now=20.0)
    second = db.claim_job("w2", capabilities=_capabilities(), now=21.0)
    assert second is not None
    stale = db.finish_from_worker(
        job.id, worker_id="w1", token=first.dispatch_token, state="SUCCEEDED", result={"x": 1}
    )
    assert stale["status"] == "stale"
    assert db.get_job(job.id).state == "DISPATCHED"
    accepted = db.finish_from_worker(
        job.id, worker_id="w2", token=second.dispatch_token, state="SUCCEEDED", result={"x": 2}
    )
    assert accepted["status"] == "accepted"
    assert db.get_job(job.id).result == {"x": 2}


def test_report_must_be_terminal(db: Database):
    job = db.create_job({"task_scope": "inspection"}, requirements=CapabilityRequirements())
    claimed = db.claim_job("w1", capabilities=_capabilities())
    assert claimed is not None
    with pytest.raises(JobStateError):
        db.finish_from_worker(job.id, worker_id="w1", token=claimed.dispatch_token, state="RUNNING")


# --------------------------------------------------------------------------- #
# scheduler
# --------------------------------------------------------------------------- #


def test_scheduler_reports_unroutable_jobs_with_reasons(db: Database):
    db.register_worker(
        worker_id="cpu-only",
        protocol_version="uzen.worker/1",
        capabilities=_capabilities(),
        token_hash="0" * 64,
    )
    db.create_job(
        {"task_scope": "architecture_synthesis", "options": {"family": "cnn"}},
        requirements=CapabilityRequirements(family=("cnn",)),
    )
    scheduler = JobScheduler(db)
    unroutable = scheduler.unroutable_jobs()
    assert len(unroutable) == 1
    assert unroutable[0]["considered"][0]["missing"]["family"] == ["cnn"]
    status = scheduler.status()
    assert status["queued"] == 1 and status["unroutable"][0]["job_id"] == unroutable[0]["job_id"]
    assert status["alive_workers"] == 1


def test_scheduler_claim_returns_the_lease_document_and_reaps_first(db: Database):
    job = db.create_job({"task_scope": "inspection"}, requirements=CapabilityRequirements())
    first = db.claim_job("w-dead", capabilities=_capabilities(), lease_seconds=1, now=time.time() - 100)
    assert first is not None
    scheduler = JobScheduler(db, lease_seconds=5)
    claimed = scheduler.claim("w-alive", capabilities=_capabilities())
    assert claimed is not None and claimed["id"] == job.id
    assert claimed["lease"]["worker_id"] == "w-alive"
    assert claimed["lease"]["attempt"] == 2
    assert claimed["reaped_on_claim"][0]["action"] == "requeued"
    assert "w-alive" in scheduler.eligible_workers(job.id)["eligible_workers"]
    history = scheduler.history(job.id)
    assert [record["attempt"] for record in history["records"]] == [1, 2]


def test_service_records_requirements_on_submit(service: JobService):
    artifact_id = _base_artifact(service)
    job = service.submit(_scale_request(artifact_id))
    rows = service.db.job_requirements(job["id"])
    assert {"kind": "operation", "value": "w.arithmetic.scale", "mode": "all"} in rows
    assert {"kind": "policy", "value": "STRICT_PLUS", "mode": "all"} in rows


# --------------------------------------------------------------------------- #
# worker daemon: claimed execution and artifact transfer
# --------------------------------------------------------------------------- #


def test_daemon_executes_a_claimed_job_once(tmp_path: Path):
    from uzen.workers.daemon import WorkerDaemon

    plane = JobService(tmp_path / "plane")
    artifact_id = _base_artifact(plane)
    job = plane.submit(_scale_request(artifact_id))
    claimed = plane.db.claim_job("w1", capabilities=_capabilities())
    assert claimed is not None

    # Share the artifact store through the exchange directory, as a remote worker would.
    daemon = WorkerDaemon(root=tmp_path / "worker", worker_id="w1")
    exchange = Path(daemon.exchange_dir) / artifact_id
    exchange.parent.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copytree(plane.store.artifact_dir(artifact_id), exchange)
    payload = claimed.to_dict()
    first = daemon.execute_claimed(payload)
    assert first["state"] == "SUCCEEDED", first
    assert first["transfer"]["already_present"] == [] or first["transfer"]["fetched"]
    artifacts_after_first = len(daemon.service.db.artifacts(limit=100))
    second = daemon.execute_claimed(payload)
    assert second["reused"] is True
    assert len(daemon.service.db.artifacts(limit=100)) == artifacts_after_first
    assert second["result"] == first["result"]


def test_daemon_refuses_a_job_whose_inputs_it_cannot_obtain(tmp_path: Path):
    from uzen.workers.daemon import WorkerDaemon

    daemon = WorkerDaemon(root=tmp_path / "worker", worker_id="w1")
    payload = {
        "id": "job-without-artifact",
        "request": _scale_request("b" * 64),
        "lease": {"attempt": 1, "token": "t"},
    }
    result = daemon.execute_claimed(payload)
    assert result["state"] == "FAILED"
    assert result["error"]["code"] == "worker.artifact_unavailable"
    assert result["error"]["details"]["unavailable"][0]["artifact_id"] == "b" * 64


# --------------------------------------------------------------------------- #
# REST worker flow: claim -> execute under lease -> report
# --------------------------------------------------------------------------- #


def test_rest_worker_claims_executes_and_reports_once(service: JobService):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from uzen.interfaces.api.app import create_app

    artifact_id = _base_artifact(service)
    api = fastapi_testclient.TestClient(create_app(service=service, auto_run=False))
    registered = api.post(
        "/v1/workers/register",
        json={"worker_id": "http-worker", "capabilities": _capabilities()},
    ).json()
    assert registered["secret"]
    job = api.post("/v1/jobs", json=_scale_request(artifact_id)).json()
    assert job["state"] == "QUEUED"  # auto_run=False: waiting for a worker

    claimed = api.post(
        "/v1/workers/claim",
        json={
            "worker_id": "http-worker",
            "secret": registered["secret"],
            "lease_seconds": 60,
            "capabilities": _capabilities(),
        },
    ).json()["job"]
    assert claimed["id"] == job["id"]
    token = claimed["lease"]["token"]

    executed = service.run(job["id"], worker_id="http-worker", token=token)
    assert executed["state"] == "SUCCEEDED", executed

    report = {
        "worker_id": "http-worker",
        "secret": registered["secret"],
        "job_id": job["id"],
        "token": token,
        "result": {"state": "SUCCEEDED", "result": executed["result"]},
    }
    first = api.post("/v1/workers/report", json=report).json()
    assert first["status"] == "duplicate"
    second = api.post("/v1/workers/report", json=report).json()
    assert second["status"] == "duplicate"
    assert service.get(job["id"])["state"] == "SUCCEEDED"

    stale = api.post(
        "/v1/workers/report", json={**report, "token": "0" * 32}
    ).json()
    assert stale["status"] == "stale"
    assert api.post(
        "/v1/workers/report", json={k: v for k, v in report.items() if k != "token"}
    ).status_code == 400

    status = api.get("/v1/scheduler").json()
    assert status["alive_workers"] == 1
    history = api.get(f"/v1/jobs/{job['id']}/dispatch").json()
    assert history["records"][0]["outcome"] == "SUCCEEDED"


# --------------------------------------------------------------------------- #
# Redis durable queue: protocol level
# --------------------------------------------------------------------------- #


def test_resp_command_encoding_is_bulk_string_frames():
    assert encode_command("PING") == b"*1\r\n$4\r\nPING\r\n"
    assert encode_command("HSET", "k", 1) == b"*3\r\n$4\r\nHSET\r\n$1\r\nk\r\n$1\r\n1\r\n"


class _ScriptedRedis:
    """Speaks RESP2 on a real socket and replays canned replies.

    This verifies the client's framing and reply parsing. It is **not** a Redis
    emulator: no key space, no Lua, no semantics.
    """

    def __init__(self, replies: list[bytes]) -> None:
        self.replies = replies
        self.commands: list[list[str]] = []
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(1)
        self.port = self._server.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def __enter__(self) -> "_ScriptedRedis":
        self._thread.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        try:
            self._server.close()
        except OSError:
            pass
        self._thread.join(timeout=5)

    def _read_command(self, conn: socket.socket) -> list[str] | None:
        buffer = b""
        while b"\r\n" not in buffer:
            chunk = conn.recv(4096)
            if not chunk:
                return None
            buffer += chunk
        count = int(buffer[1 : buffer.index(b"\r\n")])
        buffer = buffer[buffer.index(b"\r\n") + 2 :]
        arguments: list[str] = []
        for _ in range(count):
            while b"\r\n" not in buffer:
                buffer += conn.recv(4096)
            length = int(buffer[1 : buffer.index(b"\r\n")])
            buffer = buffer[buffer.index(b"\r\n") + 2 :]
            while len(buffer) < length + 2:
                buffer += conn.recv(4096)
            arguments.append(buffer[:length].decode())
            buffer = buffer[length + 2 :]
        return arguments

    def _serve(self) -> None:
        conn, _ = self._server.accept()
        try:
            while True:
                command = self._read_command(conn)
                if command is None:
                    return
                self.commands.append(command)
                reply = self.replies.pop(0) if self.replies else b"+OK\r\n"
                conn.sendall(reply)
        finally:
            conn.close()


def test_resp_connection_parses_every_reply_type():
    replies = [
        b"+PONG\r\n",
        b":42\r\n",
        b"$5\r\nhello\r\n",
        b"*2\r\n$1\r\na\r\n:7\r\n",
        b"$-1\r\n",
        b"-ERR unknown command\r\n",
    ]
    with _ScriptedRedis(replies) as server:
        config = RedisQueueConfig.from_url(f"redis://127.0.0.1:{server.port}/0")
        with RespConnection(config) as connection:
            assert connection.command("PING") == "PONG"
            assert connection.command("INCR", "k") == 42
            assert connection.command("GET", "k") == b"hello"
            assert connection.command("LRANGE", "l", 0, -1) == [b"a", 7]
            assert connection.command("GET", "missing") is None
            with pytest.raises(RedisProtocolError, match="unknown command"):
                connection.command("NOPE")
        assert server.commands[0] == ["PING"]


def test_resp_connection_is_refused_by_a_malformed_frame():
    replies = [b"!bogus\r\n"]
    with _ScriptedRedis(replies) as server:
        config = RedisQueueConfig.from_url(f"redis://127.0.0.1:{server.port}/0")
        with RespConnection(config) as connection:
            with pytest.raises(RedisProtocolError):
                connection.command("PING")


def test_redis_queue_sends_the_claim_script_with_its_keys_and_args():
    claim_reply = (
        b"*2\r\n$6\r\njob-01\r\n"
        b"*8\r\n$5\r\nstate\r\n$10\r\nDISPATCHED\r\n$8\r\nattempts\r\n$1\r\n1\r\n"
        b"$7\r\nrequest\r\n$2\r\n{}\r\n$14\r\ndispatch_token\r\n$4\r\ntok1\r\n"
    )
    # DEL, SADD, HSET, HSETNX x2 (register_worker), HGET x2 (worker budgets),
    # the reap script, then the claim script.
    replies = [
        b":0\r\n",
        b":6\r\n",
        b":3\r\n",
        b"+OK\r\n",
        b"+OK\r\n",
        b"$1\r\n4\r\n",
        b"$10\r\n8589934592\r\n",
        b"*0\r\n",
        claim_reply,
    ]
    with _ScriptedRedis(replies) as server:
        queue = RedisDurableQueue(RedisQueueConfig.from_url(f"redis://127.0.0.1:{server.port}/0"))
        queue.register_worker("w1", _capabilities())
        job = queue.claim("w1", lease_seconds=30, now=1000.0)
        queue.close()
    assert job is not None and job["id"] == "job-01"
    assert job["state"] == "DISPATCHED"
    assert len(job["lease"]["token"]) == 32
    assert job["lease"]["attempt"] == 1
    register = server.commands[1]
    claim = server.commands[-1]
    assert register[0] == "SADD" and register[1].endswith(":caps:w1")
    assert claim[0] == "EVAL" and claim[1].startswith("\nlocal prefix = KEYS[1]")
    # The token the server is told to record is the one handed back to the worker.
    assert claim[5] == job["lease"]["token"]
    assert claim[2] == "1" and claim[3] == "uzen:q"
    assert claim[4] == "w1" and float(claim[7]) == 1000.0 and claim[8] == "4"
    assert claim[10] == "0.0" and claim[11] == "0"


def test_redis_queue_delivers_requirements_and_emits_lease_keys():
    with _ScriptedRedis([b"+OK\r\n"] * 12) as server:
        queue = RedisDurableQueue(RedisQueueConfig.from_url(f"redis://127.0.0.1:{server.port}/0"))
        requirements = CapabilityRequirements(
            family=("cnn",), transport=("object_store", "inline_tensors"), min_cpu_count=2
        )
        queue.enqueue("job-02", {"task_scope": "architecture_synthesis"}, requirements, max_attempts=4)
        ack = queue.ack("job-02", worker_id="w1", token="t", state="SUCCEEDED", payload={"ok": True})
        queue.close()
    assert ack == "OK"
    commands = [command[0] for command in server.commands]
    assert commands[:2] == ["HSET", "SADD"]
    sadd = server.commands[1]
    assert "family:cnn" in sadd and "min_cpu_count" not in " ".join(sadd)
    any_sets = [command for command in server.commands if command[0] == "SADD" and ":reqany:" in command[1]]
    assert any_sets and "transport:object_store" in any_sets[0]
    ack_command = server.commands[-1]
    assert ack_command[0] == "EVAL" and ack_command[4] == "job-02"
    assert "job.lease_expired" not in " ".join(commands)


def test_redis_queue_describes_its_scheduling_semantics():
    described = RedisDurableQueue(RedisQueueConfig()).describe()
    assert described["adapter"] == "redis"
    assert described["atomic_claim"] == "lua"
    assert described["preemption"].startswith("cooperative")
    assert described["worker_draining"] is True
    assert "verified_against_live_server" not in described


# --------------------------------------------------------------------------- #
# priority, backoff, preemption, draining, fairness and long-poll
# --------------------------------------------------------------------------- #


def _submit_scale(service: JobService, name: str, **extra: Any) -> dict[str, Any]:
    return service.submit(_scale_request(f"artifact-{name}", 0.5, **extra))


def test_priority_orders_the_queue_before_arrival_time(service: JobService):
    scheduler = JobScheduler(service.db, lease_seconds=30)
    low = _submit_scale(service, "low")
    high = _submit_scale(service, "high", priority=10)
    first = scheduler.claim("w-priority", capabilities=_capabilities())
    second = scheduler.claim("w-priority", capabilities=_capabilities())
    assert first is not None and first["id"] == high["id"]
    assert second is not None and second["id"] == low["id"]


def test_aging_prevents_a_low_priority_job_from_starving(service: JobService):
    scheduler = JobScheduler(service.db, lease_seconds=30, aging_seconds=2.0)
    old = _submit_scale(service, "old")
    # Pretend the low-priority job has waited for two minutes: its effective priority
    # is 60, well above the fresh job's 10.
    with service.db._lock:
        service.db._conn.execute(
            "UPDATE jobs SET enqueued_at = ? WHERE id = ?", (time.time() - 120.0, old["id"])
        )
        service.db._conn.commit()
    fresh = _submit_scale(service, "fresh", priority=10)
    claimed = scheduler.claim("w-aging", capabilities=_capabilities())
    assert claimed is not None and claimed["id"] == old["id"], fresh


def test_expired_lease_waits_for_the_backoff_window(service: JobService):
    scheduler = JobScheduler(
        service.db, lease_seconds=5, retry_backoff_seconds=10, retry_backoff_cap_seconds=60
    )
    job = _submit_scale(service, "backoff")
    claimed = scheduler.claim("w-backoff", capabilities=_capabilities())
    assert claimed is not None and claimed["id"] == job["id"]
    expiry = claimed["lease"]["expires_at"]
    reaped = scheduler.reap(now=expiry + 0.1)
    assert reaped and reaped[0]["action"] == "requeued"
    assert reaped[0]["retry_in_seconds"] == 10.0
    assert scheduler.claim("w-backoff", capabilities=_capabilities(), now=expiry + 1.0) is None
    retried = scheduler.claim("w-backoff", capabilities=_capabilities(), now=expiry + 11.0)
    assert retried is not None and retried["id"] == job["id"]
    assert retried["attempts"] == 2


def test_backoff_is_capped(service: JobService):
    scheduler = JobScheduler(
        service.db,
        lease_seconds=10,
        max_attempts=6,
        retry_backoff_seconds=30,
        retry_backoff_cap_seconds=45,
    )
    job = _submit_scale(service, "cap")
    claimed = scheduler.claim("w-cap", capabilities=_capabilities())
    assert claimed is not None
    # Force the attempt counter up, then expire the lease: 30 * 2^4 would be 480s,
    # but the cap reports 45s.
    with service.db._lock:
        service.db._conn.execute("UPDATE jobs SET attempts = 5 WHERE id = ?", (job["id"],))
        service.db._conn.commit()
    reaped = scheduler.reap(now=claimed["lease"]["expires_at"] + 0.1)
    assert reaped[0]["retry_in_seconds"] == 45.0


def test_preemption_requeues_and_makes_the_old_report_stale(service: JobService):
    scheduler = JobScheduler(service.db, lease_seconds=30)
    job = _submit_scale(service, "preempt")
    first = scheduler.claim("w-pre", capabilities=_capabilities())
    assert first is not None and first["id"] == job["id"]
    token = first["lease"]["token"]
    outcome = scheduler.preempt(job["id"], reason="test preemption")
    assert outcome["status"] == "preempted" and outcome["state"] == "QUEUED"
    stale = scheduler.report(
        job["id"], worker_id="w-pre", token=token, state="SUCCEEDED", result={"ok": True}
    )
    assert stale["status"] == "stale"
    second = scheduler.claim("w-pre", capabilities=_capabilities())
    assert second is not None and second["id"] == job["id"]
    assert second["lease"]["token"] != token
    assert second["attempts"] == 2
    accepted = scheduler.report(
        job["id"],
        worker_id="w-pre",
        token=second["lease"]["token"],
        state="SUCCEEDED",
        result={"ok": True},
    )
    assert accepted["status"] == "accepted"
    assert accepted["state"] == "SUCCEEDED"


def test_draining_worker_cannot_claim_until_resumed(service: JobService):
    scheduler = JobScheduler(service.db, lease_seconds=30)
    service.db.register_worker(
        worker_id="w-drain",
        protocol_version="uzen.worker/1",
        capabilities=_capabilities(),
        token_hash="hash",
    )
    _submit_scale(service, "drain")
    assert scheduler.drain_worker("w-drain")["updated"] is True
    assert scheduler.claim("w-drain", capabilities=_capabilities()) is None
    # Re-registration (a reconnect) must not silently un-drain the worker.
    service.db.register_worker(
        worker_id="w-drain",
        protocol_version="uzen.worker/1",
        capabilities=_capabilities(),
        token_hash="hash",
    )
    status = scheduler.status()
    assert status["workers"][0]["draining"] is True
    assert scheduler.resume_worker("w-drain")["updated"] is True
    claimed = scheduler.claim("w-drain", capabilities=_capabilities())
    assert claimed is not None and claimed["state"] == "DISPATCHED"


def test_routine_heartbeat_does_not_undrain_a_worker(service: JobService):
    service.db.register_worker(
        worker_id="w-heart",
        protocol_version="uzen.worker/1",
        capabilities=_capabilities(),
        token_hash="hash",
    )
    service.db.set_worker_status("w-heart", "draining")
    service.db.heartbeat("w-heart")
    row = [w for w in service.db.workers() if w["worker_id"] == "w-heart"][0]
    assert row["status"] == "draining"
    service.db.heartbeat("w-heart", status="online")
    row = [w for w in service.db.workers() if w["worker_id"] == "w-heart"][0]
    assert row["status"] == "online"


def test_in_flight_cap_stops_one_worker_hoarding_the_queue(service: JobService):
    scheduler = JobScheduler(service.db, lease_seconds=30, max_in_flight_per_worker=1)
    first = _submit_scale(service, "cap-a")
    _submit_scale(service, "cap-b")
    claimed = scheduler.claim("w-inflight", capabilities=_capabilities())
    assert claimed is not None and claimed["id"] == first["id"]
    assert scheduler.claim("w-inflight", capabilities=_capabilities()) is None
    scheduler.report(
        claimed["id"],
        worker_id="w-inflight",
        token=claimed["lease"]["token"],
        state="SUCCEEDED",
        result={"ok": True},
    )
    second = scheduler.claim("w-inflight", capabilities=_capabilities())
    assert second is not None and second["id"] != claimed["id"]


def test_long_poll_claim_returns_as_soon_as_work_arrives(service: JobService):
    scheduler = JobScheduler(service.db, lease_seconds=30)

    def producer() -> None:
        time.sleep(0.4)
        _submit_scale(service, "late")

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()
    started = time.monotonic()
    claimed = scheduler.claim("w-poll", capabilities=_capabilities(), wait_seconds=5.0)
    elapsed = time.monotonic() - started
    thread.join(timeout=5)
    assert claimed is not None and claimed["task_scope"] == "weight_modification"
    assert elapsed < 5.0


def test_worker_exchange_prune_is_dry_by_default_and_keeps_referenced_artifacts(tmp_path: Path):
    from uzen.workers.daemon import WorkerDaemon

    daemon = WorkerDaemon(root=tmp_path / "worker", worker_id="w-prune")
    exchange = Path(daemon.exchange_dir)
    old = exchange / "art-old"
    fresh = exchange / "art-fresh"
    referenced = exchange / "art-referenced"
    for path in (old, fresh, referenced):
        path.mkdir(parents=True)
        (path / "manifest.json").write_text("{}")
    past = time.time() - 3600
    os.utime(old, (past, past))
    os.utime(referenced, (past, past))
    dry = daemon.prune_exchange(
        keep_artifact_ids=["art-referenced"], older_than_seconds=60, dry_run=True
    )
    assert dry["evictable"] == ["art-old"]
    assert old.is_dir() and referenced.is_dir()
    applied = daemon.prune_exchange(
        keep_artifact_ids=["art-referenced"], older_than_seconds=60, dry_run=False
    )
    assert applied["removed"] == ["art-old"]
    assert not old.exists()
    assert fresh.is_dir() and referenced.is_dir()


@pytest.mark.skipif(
    not os.environ.get("UZEN_REDIS_URL"),
    reason="set UZEN_REDIS_URL to run the live Redis/Valkey integration test",
)
def test_redis_queue_against_a_live_server():
    url = os.environ["UZEN_REDIS_URL"]
    prefix = f"uzen:test:{uuid.uuid4().hex[:8]}"
    queue = RedisDurableQueue(RedisQueueConfig.from_url(url), prefix=prefix)
    try:
        assert queue.ping() is True
        requirements = requirements_from_request(
            {"task_scope": "weight_modification", "operations": [{"op": "w.arithmetic.scale"}]}
        )
        queue.enqueue("live-low", {"task_scope": "weight_modification"}, requirements)
        queue.enqueue(
            "live-high", {"task_scope": "weight_modification"}, requirements, priority=10
        )
        queue.register_worker("live-w", _capabilities())
        first = queue.claim("live-w", lease_seconds=30, now=1000.0)
        assert first is not None and first["id"] == "live-high"
        assert (
            queue.renew("live-high", "live-w", first["lease"]["token"], lease_seconds=30, now=1001.0)
            is True
        )
        assert (
            queue.ack(
                "live-high",
                worker_id="live-w",
                token=first["lease"]["token"],
                state="SUCCEEDED",
                payload={"ok": True},
            )
            == "accepted"
        )
        live_low = queue.claim("live-w", lease_seconds=30, now=1002.0)
        assert live_low is not None and live_low["id"] == "live-low"
        assert (
            queue.ack(
                "live-low",
                worker_id="live-w",
                token=live_low["lease"]["token"],
                state="SUCCEEDED",
                payload={"ok": True},
            )
            == "accepted"
        )

        # Draining blocks new claims, and resuming restores them.
        queue.enqueue("live-drain", {"task_scope": "weight_modification"}, requirements)
        assert queue.set_worker_status("live-w", "draining") is True
        assert queue.claim("live-w", lease_seconds=30, now=1003.0) is None
        queue.set_worker_status("live-w", "online")

        # Preemption requeues and the old token is stale.
        claimed = queue.claim("live-w", lease_seconds=30, now=1004.0)
        assert claimed is not None and claimed["id"] == "live-drain"
        token = claimed["lease"]["token"]
        assert queue.preempt("live-drain", now=1005.0) == "preempted"
        assert queue.ack(
            "live-drain", worker_id="live-w", token=token, state="SUCCEEDED"
        ) == "stale"
        reclaimed = queue.claim("live-w", lease_seconds=30, now=1006.0)
        assert reclaimed is not None and int(reclaimed["attempts"]) == 2

        # Lease expiry honours the retry backoff before the job is claimable again.
        reaped = queue.reap(
            now=reclaimed["lease"]["expires_at"] + 0.1,
            retry_backoff_seconds=5.0,
            retry_backoff_cap_seconds=60.0,
        )
        assert any(item.endswith(":requeued") for item in reaped)
        assert queue.claim("live-w", lease_seconds=30, now=reclaimed["lease"]["expires_at"] + 1) is None
        # Attempt 2 backs off 5 * 2^(2-1) = 10s, so +12 is safely past the window.
        retried = queue.claim(
            "live-w", lease_seconds=30, now=reclaimed["lease"]["expires_at"] + 12
        )
        assert retried is not None and int(retried["attempts"]) == 3
    finally:
        keys = queue.connection.command("KEYS", f"{prefix}*") or []
        if keys:
            queue.connection.command("DEL", *[_decode_key(key) for key in keys])
        queue.close()


def _decode_key(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)
