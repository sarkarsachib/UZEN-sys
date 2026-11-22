"""N7 demo: a campaign evaluated by real worker processes over the N4 queue.

Run:
    python examples/distributed_campaign_demo.py --root /tmp/uzen-n7-demo

The demo starts two worker processes on a shared control-plane database, runs a
four-trial architecture campaign through the distributed executor, compares its
deterministic digest with a local run, and replays every corpus record. It exits
non-zero if any gate fails.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

from uzen.core.campaign import (
    CampaignEngine,
    CampaignSpec,
    CampaignWorker,
    DistributedTrialExecutor,
)
from uzen.jobs.service import JobService


def _spec() -> CampaignSpec:
    return CampaignSpec.from_dict(
        {
            "name": "n7-distributed-architecture",
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
            "constraints": {"output_rms": {"min": 1e-6}},
            "budget": 4,
            "batch_size": 4,
            "strategy": "grid",
            "seed": 19,
            "write_corpus": True,
        }
    )


def _worker_main(root: str, worker_id: str, ready, stop) -> None:
    worker = CampaignWorker(root, worker_id, claim_wait_seconds=0.25)
    worker.warmup()
    ready.wait()
    worker.run_forever(stop=stop)


def main() -> int:
    parser = argparse.ArgumentParser(description="UZEN N7 distributed campaign demo")
    parser.add_argument("--root", default="/tmp/uzen-n7-demo")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    spec = _spec()
    local_root = root / "local"
    distributed_root = root / "distributed"
    local_report = CampaignEngine(local_root).run(spec)

    ctx = mp.get_context("fork")
    ready = ctx.Barrier(3)  # two workers plus the coordinator
    stop = ctx.Event()
    processes = [
        ctx.Process(
            target=_worker_main,
            args=(str(distributed_root), f"n7-worker-{index}", ready, stop),
            name=f"n7-worker-{index}",
        )
        for index in range(2)
    ]
    for process in processes:
        process.start()
    ready.wait(timeout=10.0)
    time.sleep(0.2)

    control = JobService(distributed_root)
    try:
        report = CampaignEngine(
            distributed_root,
            trial_executor=DistributedTrialExecutor(
                control, timeout_seconds=args.timeout, poll_interval_seconds=0.02
            ),
        ).run(spec)
    finally:
        stop.set()
        for process in processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)

    jobs = control.list(limit=100)
    workers_seen = sorted({str(job.get("worker_id")) for job in jobs if job.get("worker_id")})
    worker_exits = [[process.name, process.exitcode] for process in processes]
    engine = CampaignEngine(distributed_root)
    replay = engine.verify_corpus()

    gates = {
        "distributed_completed": report.status == "completed" and report.stop_reason == "budget",
        "all_trials_ok": len(report.trials) == 4 and all(trial.status == "ok" for trial in report.trials),
        "jobs_succeeded": len(jobs) == 4 and all(job["state"] == "SUCCEEDED" for job in jobs),
        "multiple_worker_processes": len(workers_seen) >= 2,
        "matches_local_digest": report.deterministic_digest() == local_report.deterministic_digest(),
        "corpus_replays": replay["matched"] == 4 and not replay["mismatches"],
    }

    print("UZEN N7 - distributed research campaigns")
    print(f"control plane: {distributed_root}")
    print(f"workers seen: {workers_seen}; exits: {worker_exits}")
    print(f"trials: {len(report.trials)}; digest: {report.deterministic_digest()[:16]}...")
    print(f"jobs: {[(job['state'], job.get('worker_id')) for job in jobs]}")
    print(f"corpus: {replay['matched']}/{replay['checked']} verified")
    print("gates:")
    for name, passed in gates.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    report_path = root / "reports" / "n7_distributed.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "schema": "uzen.n7.demo/1",
                "spec": spec.to_dict(),
                "distributed": report.to_dict(),
                "local_digest": local_report.deterministic_digest(),
                "workers": workers_seen,
                "worker_exits": worker_exits,
                "jobs": jobs,
                "corpus": replay,
                "gates": gates,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"report: {report_path}")
    print(f"gate: {'PASS' if all(gates.values()) else 'FAIL'}")
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
