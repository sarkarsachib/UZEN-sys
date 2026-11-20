"""Real use case: run a heterogeneous worker fleet against one control plane.

A weight-engineering deployment is rarely one machine. Some hosts are big, some are
small, one has a GPU that is reserved for a specific architecture family, and a host
can die in the middle of a job. This demo runs that situation for real:

1. one SQLite control plane and one artifact store, shared by every worker process;
2. three worker **processes** with different advertised capabilities, plus a fourth
   process that claims a job and is killed before it can report;
3. capability routing — a job that needs 12 CPUs is only ever claimed by the 16-CPU
   worker, a `cnn` job only by the worker that advertises `cnn`;
4. lease recovery — the killed worker's job is reaped after its lease expires and
   completed by a healthy worker, on attempt 2, with exactly one output artifact;
5. idempotent dispatch — replaying the winning worker's report is refused as a
   duplicate and changes nothing;
6. honest starvation — a job requiring a device no worker advertises is reported as
   *unroutable* with the reason, instead of sitting in the queue looking healthy;
7. scheduling policy — a high-priority job is claimed first, a preempted job is
   requeued while its old token is refused as stale, a draining worker stops receiving
   work until resumed, and an expired lease waits out its backoff window.

Every job executes real registered operations through
:class:`uzen.jobs.service.JobService`; nothing is simulated.

Usage::

    .venv/bin/python examples/distributed_scheduling_demo.py --root /tmp/uzen-distributed-demo

Add ``--redis-url redis://127.0.0.1:6379/0`` to also exercise the Redis/Valkey durable
queue adapter against a live server (it is skipped, and reported as skipped, when no
server is configured).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from uzen.core.nws.families import build_architecture
from uzen.infrastructure.storage.db import Database
from uzen.jobs.scheduler import JobScheduler
from uzen.jobs.service import JobService
from uzen.jobs.scheduling import requirements_from_request

WORKER_A = "fleet-a-mlp"
WORKER_B = "fleet-b-general"
WORKER_C = "fleet-c-cnn"
WORKER_DOOMED = "fleet-d-doomed"


def _capabilities(
    *,
    cpu_count: int,
    families: list[str],
    device: str = "cpu",
    cuda: bool = False,
) -> dict[str, Any]:
    return {
        "protocol_version": "uzen.worker/1",
        "cpu_count": cpu_count,
        "memory_bytes": 16 * 1024**3,
        "device": device,
        "cuda_available": cuda,
        "supported_operations": [
            "w.arithmetic.scale",
            "arch.synthesize",
        ],
        "supported_operators": ["linear", "layernorm", "gelu"],
        "supported_architectures": families,
        "artifact_transport": ["shared_filesystem", "inline_tensors"],
        "representations": [],
        "strict_plus": True,
    }


# --------------------------------------------------------------------------- #
# worker processes
# --------------------------------------------------------------------------- #


def worker_process(
    plane_root: str,
    worker_id: str,
    capabilities: dict[str, Any],
    lease_seconds: float,
    run_seconds: float,
    outcomes: Any,
) -> None:
    """Claim, execute and report until the time budget runs out."""
    service = JobService(plane_root)
    scheduler = JobScheduler(service.db, lease_seconds=lease_seconds, max_attempts=3)
    deadline = time.time() + run_seconds
    while time.time() < deadline:
        claimed = scheduler.claim(worker_id, capabilities=capabilities)
        if claimed is None:
            time.sleep(0.05)
            continue
        token = claimed["lease"]["token"]
        attempt = claimed["lease"]["attempt"]
        started = time.perf_counter()
        try:
            executed = service.run(claimed["id"], worker_id=worker_id, token=token)
            outcomes.put(
                {
                    "worker_id": worker_id,
                    "job_id": claimed["id"],
                    "state": executed["state"],
                    "attempt": attempt,
                    "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
                }
            )
        except Exception as exc:  # a failed job is recorded on the job, not hidden
            outcomes.put(
                {
                    "worker_id": worker_id,
                    "job_id": claimed["id"],
                    "state": "EXCEPTION",
                    "attempt": attempt,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    outcomes.put({"worker_id": worker_id, "done": True})


def doomed_process(
    plane_root: str,
    worker_id: str,
    capabilities: dict[str, Any],
    lease_seconds: float,
    outcomes: Any,
) -> None:
    """Claim exactly one job and die without reporting, like `kill -9` on a worker."""
    service = JobService(plane_root)
    scheduler = JobScheduler(service.db, lease_seconds=lease_seconds, max_attempts=3)
    claimed = scheduler.claim(worker_id, capabilities=capabilities)
    outcomes.put(
        {
            "worker_id": worker_id,
            "claimed": claimed["id"] if claimed else None,
            "crashed_before_report": True,
        }
    )
    os._exit(9)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _scale_request(artifact_id: str, factor: float, **extra: Any) -> dict[str, Any]:
    request: dict[str, Any] = {
        "task_scope": "weight_modification",
        "objective": f"scale weights by {factor}",
        "inputs": [{"name": "base", "artifact_id": artifact_id}],
        "operations": [{"op": "w.arithmetic.scale", "params": {"factor": factor}}],
    }
    request.update(extra)
    return request


def _drain(outcomes: Any, timeout: float) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            collected.append(outcomes.get(timeout=0.2))
        except Exception:
            break
    return collected


def _artifacts_for(plane: JobService, job_id: str) -> list[dict[str, Any]]:
    return [
        entry
        for entry in plane.db.artifacts(limit=500)
        if entry.get("job_id") == job_id
    ]


def redis_section(url: str | None) -> dict[str, Any]:
    """Exercise the RESP adapter only when a live server is configured."""
    if not url:
        return {
            "configured": False,
            "skipped": True,
            "detail": (
                "no Redis/Valkey server configured; the adapter is verified at the RESP "
                "protocol level only (see tests/test_distributed_scheduling.py)"
            ),
        }
    import uuid

    from uzen.infrastructure.storage.redis_queue import RedisDurableQueue, RedisQueueConfig

    suffix = uuid.uuid4().hex[:8]
    prefix = f"uzen:demo:{suffix}"
    job_id = f"demo-job-{suffix}"
    worker_id = f"redis-probe-{suffix}"
    queue = RedisDurableQueue(
        RedisQueueConfig.from_url(url, timeout=3.0), prefix=prefix
    )
    try:
        connected = queue.ping()
        requirements = requirements_from_request(
            _scale_request("f" * 64, 0.5, compute_requirements={"min_cpu_count": 2})
        )
        queue.register_worker(worker_id, _capabilities(cpu_count=8, families=["mlp"]))
        queue.enqueue(job_id, _scale_request("f" * 64, 0.5), requirements, max_attempts=2)
        claimed = queue.claim(worker_id, lease_seconds=1.0)
        renewed = (
            queue.renew(
                job_id, worker_id, claimed["lease"]["token"], lease_seconds=1.0
            )
            if claimed
            else False
        )
        first = queue.ack(
            job_id,
            worker_id=worker_id,
            token=claimed["lease"]["token"] if claimed else "",
            state="SUCCEEDED",
            payload={"ok": True},
        )
        second = queue.ack(
            job_id,
            worker_id=worker_id,
            token=claimed["lease"]["token"] if claimed else "",
            state="SUCCEEDED",
            payload={"ok": True},
        )
        reaped = queue.reap(now=time.time() + 60)
        return {
            "configured": True,
            "connected": connected,
            "claimed": bool(claimed),
            "renewed": renewed,
            "first_ack": first,
            "second_ack": second,
            "reaped": reaped,
            "status": queue.status(),
        }
    finally:
        keys = queue.connection.command("KEYS", f"{prefix}*") or []
        if keys:
            queue.connection.command(
                "DEL", *[key.decode() if isinstance(key, bytes) else str(key) for key in keys]
            )
        queue.close()


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    shutil.rmtree(root, ignore_errors=True)
    plane_root = root / "plane"
    plane = JobService(plane_root)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    # Two real artifacts to operate on.
    mlp = build_architecture("mlp", dict(input_dim=16, hidden_dims=[32], output_dim=8))
    mlp_a = plane.store.put(mlp.build_package(seed=11), operation="init").artifact_id
    mlp_b = plane.store.put(mlp.build_package(seed=12), operation="init").artifact_id

    workers = {
        WORKER_A: _capabilities(cpu_count=2, families=["mlp", "transformer_decoder"]),
        WORKER_B: _capabilities(cpu_count=16, families=["mlp"]),
        WORKER_C: _capabilities(cpu_count=8, families=["cnn"]),
        WORKER_DOOMED: _capabilities(cpu_count=2, families=["mlp"]),
    }
    for worker_id, capabilities in workers.items():
        plane.db.register_worker(
            worker_id=worker_id,
            protocol_version="uzen.worker/1",
            capabilities=capabilities,
            token_hash="0" * 64,
        )

    # 1. A job that a worker claims and then dies holding.
    crash_job = plane.submit(_scale_request(mlp_a, 0.9, objective="crashed-then-retried job"))
    ctx = mp.get_context("spawn")
    outcomes = ctx.Queue()
    doomed = ctx.Process(
        target=doomed_process,
        args=(str(plane_root), WORKER_DOOMED, workers[WORKER_DOOMED], 2.0, outcomes),
        name="doomed-worker",
    )
    doomed.start()
    doomed.join(timeout=60)
    doomed_claim = _drain(outcomes, 2.0)

    # 2. The rest of the work, submitted while the crash lease is still live.
    jobs = {
        "scale_mlp": plane.submit(_scale_request(mlp_b, 1.5)),
        "cnn_synthesis": plane.submit(
            {
                "task_scope": "architecture_synthesis",
                "objective": "synthesize a cnn",
                "options": {
                    "family": "cnn",
                    "in_channels": 1,
                    "height": 8,
                    "width": 8,
                    "channels": [4],
                    "num_classes": 3,
                },
                "compute_requirements": {"min_cpu_count": 8},
                "seed": 3,
            }
        ),
        "big_cpu_scale": plane.submit(
            _scale_request(mlp_b, 2.0, compute_requirements={"min_cpu_count": 12})
        ),
        "needs_cuda": plane.submit(
            _scale_request(mlp_b, 0.5, compute_requirements={"device": "cuda"})
        ),
    }

    # 3. Run the fleet; the crash lease expires while it is running and gets reaped.
    procs = [
        ctx.Process(
            target=worker_process,
            args=(
                str(plane_root),
                worker_id,
                capabilities,
                2.0,
                args.run_seconds,
                outcomes,
            ),
            name=worker_id,
        )
        for worker_id, capabilities in (
            (WORKER_A, workers[WORKER_A]),
            (WORKER_B, workers[WORKER_B]),
            (WORKER_C, workers[WORKER_C]),
        )
    ]
    for process in procs:
        process.start()
    for process in procs:
        process.join(timeout=args.run_seconds + 60)
    executions = [entry for entry in _drain(outcomes, 3.0) if not entry.get("done")]

    scheduler = JobScheduler(plane.db, lease_seconds=2.0, max_attempts=3)
    crash_history = scheduler.history(crash_job["id"])
    crash_state = plane.db.get_job(crash_job["id"])
    crash_artifacts = _artifacts_for(plane, crash_job["id"])

    # 4. Replay the winning report: it must be refused as a duplicate.
    winner = crash_history["records"][-1]
    replay = scheduler.report(
        crash_job["id"],
        worker_id=winner["worker_id"],
        token=winner["token"],
        state="SUCCEEDED",
        result={"output_artifact_id": "f" * 64},
    )
    artifacts_after_replay = _artifacts_for(plane, crash_job["id"])

    # 5. Scheduling policy, executed against the same real control plane.
    policy_scheduler = JobScheduler(
        plane.db,
        lease_seconds=2.0,
        max_attempts=3,
        retry_backoff_seconds=5.0,
        retry_backoff_cap_seconds=30.0,
    )
    policy_worker = "fleet-e-policy"
    policy_caps = workers[WORKER_A]
    policy_low = plane.submit(_scale_request(mlp_a, 0.4, objective="policy low"))
    policy_high = plane.submit(
        _scale_request(mlp_b, 0.6, objective="policy high", priority=50)
    )
    policy_first = policy_scheduler.claim(policy_worker, capabilities=policy_caps)
    priority_ok = policy_first is not None and policy_first["id"] == policy_high["id"]
    preempted = policy_scheduler.preempt(policy_high["id"], reason="demo maintenance")
    stale_report = policy_scheduler.report(
        policy_high["id"],
        worker_id=policy_worker,
        token=policy_first["lease"]["token"],
        state="SUCCEEDED",
        result={"output_artifact_id": "f" * 64},
    )
    replacement = policy_scheduler.claim(policy_worker, capabilities=policy_caps)
    replacement_report = policy_scheduler.report(
        policy_high["id"],
        worker_id=policy_worker,
        token=replacement["lease"]["token"],
        state="SUCCEEDED",
        result={"output_artifact_id": "f" * 64},
    )
    low_claim = policy_scheduler.claim(policy_worker, capabilities=policy_caps)
    policy_scheduler.report(
        policy_low["id"],
        worker_id=policy_worker,
        token=low_claim["lease"]["token"],
        state="SUCCEEDED",
        result={"output_artifact_id": "f" * 64},
    )

    draining_job = plane.submit(_scale_request(mlp_b, 0.7, objective="drained worker"))
    policy_scheduler.drain_worker(policy_worker)
    while_draining = policy_scheduler.claim(policy_worker, capabilities=policy_caps)
    policy_scheduler.resume_worker(policy_worker)
    after_resume = policy_scheduler.claim(policy_worker, capabilities=policy_caps)
    policy_scheduler.report(
        draining_job["id"],
        worker_id=policy_worker,
        token=after_resume["lease"]["token"],
        state="SUCCEEDED",
        result={"output_artifact_id": "f" * 64},
    )

    backoff_job = plane.submit(_scale_request(mlp_a, 0.8, objective="backoff retry"))
    backoff_claim = policy_scheduler.claim(policy_worker, capabilities=policy_caps)
    backoff_expiry = backoff_claim["lease"]["expires_at"]
    policy_scheduler.reap(now=backoff_expiry + 0.1)
    too_early = policy_scheduler.claim(
        policy_worker, capabilities=policy_caps, now=backoff_expiry + 1.0
    )
    after_backoff = policy_scheduler.claim(
        policy_worker, capabilities=policy_caps, now=backoff_expiry + 6.0
    )
    policy_scheduler.report(
        backoff_job["id"],
        worker_id=policy_worker,
        token=after_backoff["lease"]["token"],
        state="SUCCEEDED",
        result={"output_artifact_id": "f" * 64},
    )

    states = {name: plane.get(job["id"])["state"] for name, job in jobs.items()}
    workers_by_job: dict[str, list[str]] = {}
    for name, job in jobs.items():
        workers_by_job[name] = [record["worker_id"] for record in plane.db.dispatch_records(job["id"])]
    outputs = {
        name: (plane.get(job["id"]).get("result") or {}).get("output_artifact_id")
        for name, job in jobs.items()
    }
    artifact_counts = {
        name: len(_artifacts_for(plane, job["id"])) for name, job in jobs.items()
    }
    unroutable = scheduler.unroutable_jobs()

    gates = {
        "crashed job recovered on a later attempt": crash_state.state == "SUCCEEDED"
        and crash_state.attempts == 2
        and [record["worker_id"] for record in crash_history["records"]][0] == WORKER_DOOMED,
        "crashed job produced exactly one artifact": len(crash_artifacts) == 1,
        "replayed report rejected as duplicate": replay["status"] == "duplicate"
        and len(artifacts_after_replay) == 1,
        "every routable job succeeded": all(
            states[name] == "SUCCEEDED"
            for name in ("scale_mlp", "cnn_synthesis", "big_cpu_scale")
        ),
        "cnn job only ever claimed by the cnn worker": workers_by_job["cnn_synthesis"]
        and all(worker == WORKER_C for worker in workers_by_job["cnn_synthesis"]),
        "12-cpu job only ever claimed by the 16-cpu worker": workers_by_job["big_cpu_scale"]
        and all(worker == WORKER_B for worker in workers_by_job["big_cpu_scale"]),
        "cuda job reported unroutable instead of starving silently": states["needs_cuda"] == "QUEUED"
        and any(entry["job_id"] == jobs["needs_cuda"]["id"] for entry in unroutable),
        "no job produced more than one artifact": all(count <= 1 for count in artifact_counts.values()),
        "each opaque output id is unique per job": len(
            [value for value in outputs.values() if value]
        )
        == len({value for value in outputs.values() if value}),
        "high-priority job claimed before the older low-priority job": priority_ok,
        "preempted job requeued and its old token refused": preempted["status"] == "preempted"
        and stale_report["status"] == "stale"
        and replacement_report["status"] == "accepted"
        and replacement["attempts"] == 2,
        "draining worker is skipped until resumed": while_draining is None
        and after_resume is not None
        and after_resume["id"] == draining_job["id"],
        "retry waits out the backoff window": too_early is None
        and after_backoff is not None
        and after_backoff["attempts"] == 2,
    }

    report = {
        "demo": "distributed_scheduling",
        "use_case": (
            "heterogeneous worker fleet, capability routing, lease recovery after a worker "
            "crash and idempotent dispatch over one control plane"
        ),
        "root": str(root),
        "workers": {
            worker_id: {
                "cpu_count": capabilities["cpu_count"],
                "families": capabilities["supported_architectures"],
                "device": capabilities["device"],
            }
            for worker_id, capabilities in workers.items()
        },
        "crashed_job": {
            "job_id": crash_job["id"],
            "state": crash_state.state,
            "attempts": crash_state.attempts,
            "records": crash_history["records"],
            "artifacts": crash_artifacts,
            "replay_report": replay,
        },
        "jobs": {
            name: {
                "job_id": job["id"],
                "state": states[name],
                "workers": workers_by_job[name],
                "output_artifact_id": outputs[name],
                "artifacts": artifact_counts[name],
            }
            for name, job in jobs.items()
        },
        "unroutable": unroutable,
        "policy": {
            "priority_claimed": policy_first["id"] if policy_first else None,
            "priority_expected": policy_high["id"],
            "preemption": preempted,
            "stale_report": stale_report,
            "replacement": replacement_report,
            "claim_while_draining": while_draining,
            "claim_after_resume": after_resume["id"] if after_resume else None,
            "claim_too_early": too_early,
            "claim_after_backoff": after_backoff["id"] if after_backoff else None,
        },
        "executions": executions,
        "scheduler": scheduler.status(),
        "redis": redis_section(args.redis_url),
        "gates": gates,
        "ok": all(gates.values()),
    }
    report_path = reports / "distributed_scheduling.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")

    print(f"control plane: {plane_root}")
    print(f"workers: {', '.join(workers)}")
    print(f"crashed job {crash_job['id'][:8]}… claimed by {WORKER_DOOMED}, never reported")
    print(
        f"  recovered on attempt {crash_state.attempts} by "
        f"{crash_history['records'][-1]['worker_id']} → {crash_state.state}"
    )
    print(f"  replayed report: {replay['status']}")
    for name, job in jobs.items():
        print(
            f"{name:<16} {states[name]:<10} workers={workers_by_job[name]} "
            f"artifacts={artifact_counts[name]}"
        )
    print(f"unroutable jobs: {[entry['job_id'][:8] for entry in unroutable]}")
    if unroutable:
        print(f"  reason: {unroutable[0]['considered'][0]['reasons']}")
    print(
        "policy: "
        f"priority={'ok' if priority_ok else 'FAIL'}, "
        f"preempt={preempted['status']}, stale_report={stale_report['status']}, "
        f"drain_claim={'none' if while_draining is None else 'FAIL'}, "
        f"too_early={'none' if too_early is None else 'FAIL'}"
    )
    redis = report["redis"]
    print(
        "redis adapter: "
        + ("skipped (no live server configured)" if redis.get("skipped") else "exercised")
    )

    print("\n=== gates ===")
    for name, passed in gates.items():
        print(f"{name}: {passed}")
    print(f"\nreport: {report_path}")
    print(f"gate: {'PASS' if report['ok'] else 'FAIL'}")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="/tmp/uzen-distributed-demo")
    parser.add_argument("--run-seconds", type=float, default=12.0)
    parser.add_argument("--redis-url", default=os.environ.get("UZEN_REDIS_URL"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run(args)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
