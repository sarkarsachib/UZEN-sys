"""N8 demo: durable background campaigns, cancellation and job-kind isolation.

Run:
    python examples/campaign_control_demo.py --root /tmp/uzen-n8-demo

The demo starts one coordinator process and three trial-worker processes on one
shared control-plane database. It submits a campaign as a durable N4 job, waits for
it to complete, submits a second campaign, cancels it while workers are evaluating,
and verifies that linked jobs stop without adding new trials. A decoy engineering job
stays queued because campaign workers are restricted to their own task kinds.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any

from uzen.core.campaign import (
    CAMPAIGN_RUN_KIND,
    CampaignEngine,
    CampaignSpec,
    CampaignWorker,
    submit_campaign_run,
)
from uzen.infrastructure.storage.db import TERMINAL_STATES
from uzen.jobs.service import JobService


def _spec(name: str, *, budget: int, batch_size: int) -> CampaignSpec:
    return CampaignSpec.from_dict(
        {
            "name": name,
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
            "objectives": [
                {"name": "parameters", "direction": "min"},
                {"name": "output_diversity", "direction": "max"},
            ],
            "constraints": {"output_rms": {"min": 1e-6}},
            "budget": budget,
            "batch_size": batch_size,
            "strategy": "evolution",
            "seed": 31,
            "write_corpus": True,
        }
    )


def _worker_main(root: str, worker_id: str, job_kinds, ready, stop) -> None:
    worker = CampaignWorker(
        root,
        worker_id,
        claim_wait_seconds=0.2,
        job_kinds=tuple(job_kinds) if job_kinds else None,
    )
    worker.warmup()
    ready.wait()
    worker.run_forever(stop=stop)


def _wait_for(predicate, *, timeout: float) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise TimeoutError("demo condition was not reached before the deadline")


def main() -> int:
    parser = argparse.ArgumentParser(description="UZEN N8 campaign control demo")
    parser.add_argument("--root", default="/tmp/uzen-n8-demo")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    service = JobService(root)
    decoy = service.submit(
        {"task_scope": "inspection", "objective": "decoy non-campaign job"}
    )

    ctx = mp.get_context("fork")
    ready = ctx.Barrier(5)  # one coordinator, three trial workers and the parent
    stop = ctx.Event()
    worker_specs = [
        ("n8-coordinator", (CAMPAIGN_RUN_KIND,)),
        ("n8-trial-a", ("campaign_trial",)),
        ("n8-trial-b", ("campaign_trial",)),
        ("n8-trial-c", ("campaign_trial",)),
    ]
    processes = [
        ctx.Process(
            target=_worker_main,
            args=(str(root), worker_id, kinds, ready, stop),
            name=worker_id,
        )
        for worker_id, kinds in worker_specs
    ]
    for process in processes:
        process.start()
    ready.wait(timeout=10.0)
    time.sleep(0.2)

    short = _spec("n8-background-short", budget=6, batch_size=2)
    long = _spec("n8-cancelled-long", budget=512, batch_size=8)
    engine = CampaignEngine(root, job_service=service)

    try:
        short_submit = submit_campaign_run(
            service,
            short,
            distributed=True,
            trial_timeout_seconds=args.timeout,
            poll_interval_seconds=0.02,
        )
        short_report = _wait_for(
            lambda: engine.get(short.candidate_id)
            if engine.get(short.candidate_id).status == "completed"
            else None,
            timeout=args.timeout,
        )
        short_job_id = short_submit["job"]["id"]
        _wait_for(
            lambda: service.get(short_job_id)["state"] in TERMINAL_STATES,
            timeout=args.timeout,
        )
        short_job = service.get(short_job_id)

        long_submit = submit_campaign_run(
            service,
            long,
            distributed=True,
            trial_timeout_seconds=args.timeout,
            poll_interval_seconds=0.02,
        )
        _wait_for(
            lambda: len(engine.get(long.candidate_id).trials) >= 2,
            timeout=args.timeout,
        )
        cancellation = engine.cancel(long.candidate_id)
        after_cancel_count = len(engine.get(long.candidate_id).trials)
        long_job_id = long_submit["job"]["id"]
        _wait_for(
            lambda: service.get(long_job_id)["state"] in TERMINAL_STATES,
            timeout=args.timeout,
        )
        _wait_for(
            lambda: all(
                job["state"] in TERMINAL_STATES for job in engine.jobs(long.candidate_id)
            ),
            timeout=args.timeout,
        )
        time.sleep(0.15)
        long_report = engine.get(long.candidate_id)
        linked_long = engine.jobs(long.candidate_id)
        replay = engine.verify_corpus()
    finally:
        stop.set()
        for process in processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)

    short_trials = [
        job for job in service.list(limit=200) if job["job_kind"] == "campaign_trial"
    ]
    short_workers = sorted(
        {
            str(job["worker_id"])
            for job in short_trials
            if job.get("worker_id") and job["state"] == "SUCCEEDED"
        }
    )
    decoy_state = service.get(decoy["id"])["state"]
    worker_exits = [[process.name, process.exitcode] for process in processes]

    gates = {
        "background_job_is_campaign_run": short_submit["job"]["job_kind"]
        == CAMPAIGN_RUN_KIND,
        "background_campaign_completed": short_report.status == "completed"
        and len(short_report.trials) == 6,
        "background_job_succeeded": short_job["state"] == "SUCCEEDED"
        and short_job["result"]["status"] == "completed",
        "campaign_workers_are_observed": len(short_workers) >= 2,
        "cancel_status_is_terminal": long_report.status == "cancelled"
        and long_report.stop_reason == "cancelled",
        "cancel_propagated_to_run_job": service.get(long_job_id)["state"] == "CANCELLED",
        "cancel_left_no_live_jobs": all(
            job["state"] in TERMINAL_STATES for job in linked_long
        ),
        "cancel_did_not_add_trials": len(long_report.trials) == after_cancel_count
        and cancellation["campaign"]["trials_evaluated"] <= after_cancel_count,
        "decoy_job_not_stolen": decoy_state == "QUEUED",
        "corpus_replays": replay["matched"] == replay["checked"]
        and not replay["mismatches"],
    }

    print("UZEN N8 - campaign control and isolation")
    print(f"control plane: {root}")
    print(
        f"short campaign: {short_report.status}/{short_report.stop_reason} "
        f"trials={len(short_report.trials)} workers={short_workers}"
    )
    print(
        f"cancelled campaign: {long_report.status}/{long_report.stop_reason} "
        f"trials={len(long_report.trials)} linked_jobs={len(linked_long)}"
    )
    print(f"decoy job: {decoy_state}; workers: {worker_exits}")
    print(f"corpus: {replay['matched']}/{replay['checked']} verified")
    print("gates:")
    for name, passed in gates.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    report_path = root / "reports" / "n8_campaign_control.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "schema": "uzen.n8.demo/1",
                "short_spec": short.to_dict(),
                "short_report": short_report.to_dict(),
                "cancelled_spec": long.to_dict(),
                "cancelled_report": long_report.to_dict(),
                "cancellation": cancellation,
                "linked_jobs": linked_long,
                "decoy_job": service.get(decoy["id"]),
                "workers": short_workers,
                "worker_exits": worker_exits,
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
