"""N9 demo: single-coordinator admission and a lease that outlives its claim.

Run:
    python examples/campaign_durability_demo.py --root /tmp/uzen-n9-demo

Four independent submitter processes race to start the same background campaign. The
campaign store's atomic reservation must admit exactly one coordinator job. A reaper
then expires leases aggressively while one coordinator executes a campaign that lasts
longer than its initial lease; the keepalive must preserve attempt 1 and a single
successful execution.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import threading
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
from uzen.jobs.service import JobService


def _spec() -> CampaignSpec:
    # A unique name keeps repeated demo runs from reusing a completed campaign in the
    # same control-plane directory.
    run_id = f"{os.getpid()}-{int(time.time())}"
    return CampaignSpec.from_dict(
        {
            "name": f"n9-admission-and-lease-{run_id}",
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
            "budget": 64,
            "batch_size": 4,
            "strategy": "grid",
            "patience": 1000,
            "seed": 19,
            "write_corpus": False,
        }
    )


def _submit_main(
    root: str,
    spec_doc: dict[str, Any],
    gate: Any,
    results: Any,
) -> None:
    service = JobService(root)
    gate.wait()
    try:
        outcome = submit_campaign_run(service, CampaignSpec.from_dict(spec_doc))
        results.put(
            {
                "job_id": str(outcome["job"]["id"]),
                "campaign_id": str(outcome["campaign_id"]),
            }
        )
    except Exception as exc:  # surfaced by the parent as a failed gate
        results.put({"error": f"{type(exc).__name__}: {exc}"})


def main() -> int:
    parser = argparse.ArgumentParser(description="UZEN N9 campaign durability demo")
    parser.add_argument("--root", default="/tmp/uzen-n9-demo")
    parser.add_argument("--submitters", type=int, default=4)
    parser.add_argument("--lease-seconds", type=float, default=0.15)
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    spec = _spec()
    service = JobService(root)
    # Create the campaign row before the submitters race; admission is the only
    # concurrent operation being exercised.
    CampaignEngine(root, job_service=service).store.create_or_load(spec)

    ctx = mp.get_context("spawn")
    gate = ctx.Barrier(args.submitters + 1)
    results = ctx.Queue()
    processes = [
        ctx.Process(
            target=_submit_main,
            args=(str(root), spec.to_dict(), gate, results),
            name=f"submitter-{index}",
        )
        for index in range(args.submitters)
    ]
    for process in processes:
        process.start()
    gate.wait(timeout=30.0)

    outcomes = [results.get(timeout=30.0) for _ in processes]
    for process in processes:
        process.join(timeout=10.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)

    submitted = [outcome for outcome in outcomes if "job_id" in outcome]
    admitted_ids = {outcome["job_id"] for outcome in submitted}
    admitted_id = next(iter(admitted_ids)) if len(admitted_ids) == 1 else ""

    reaped: list[dict[str, Any]] = []
    reap_stop = threading.Event()

    def reap() -> None:
        while not reap_stop.wait(0.01):
            reaped.extend(service.db.reap_expired_leases())

    reaper = threading.Thread(target=reap, name="n9-reaper", daemon=True)
    reaper.start()

    worker = CampaignWorker(
        str(root),
        "n9-coordinator",
        lease_seconds=args.lease_seconds,
        claim_wait_seconds=5.0,
        poll_interval_seconds=0.01,
        job_kinds=(CAMPAIGN_RUN_KIND,),
    )
    worker.warmup()
    started = time.monotonic()
    try:
        execution = worker.run_once(service=service)
    finally:
        elapsed = time.monotonic() - started
        reap_stop.set()
        reaper.join(timeout=2.0)

    engine = CampaignEngine(root, job_service=service)
    report = engine.get(spec.candidate_id)
    job = service.get(admitted_id) if admitted_id else None
    campaign_row = engine.store.campaign(spec.candidate_id)
    run_jobs = [
        linked for linked in engine.jobs(spec.candidate_id) if linked["job_kind"] == CAMPAIGN_RUN_KIND
    ]
    attempted = [
        outcome for outcome in reaped if str(outcome.get("job_id")) == admitted_id
    ]

    gates = {
        "all_submitters_returned": len(submitted) == args.submitters,
        "one_admitted_job": len(admitted_ids) == 1,
        "one_run_job_in_store": len(run_jobs) == 1,
        "reservation_matches_job": campaign_row["run_job_id"] == admitted_id,
        "coordinator_succeeded": job is not None and job["state"] == "SUCCEEDED",
        "single_attempt": job is not None and job["attempts"] == 1,
        "campaign_completed": report.status == "completed"
        and len(report.trials) == spec.budget,
        "run_outlived_initial_lease": elapsed > args.lease_seconds,
        "reaper_never_moved_the_job": not attempted,
    }

    print("UZEN N9 - campaign run durability")
    print(f"control plane: {root}")
    print(
        f"submitters: {len(submitted)}/{args.submitters} returned; "
        f"admitted job: {admitted_id or '-'}"
    )
    print(
        f"campaign: {report.status}/{report.stop_reason} trials={len(report.trials)}; "
        f"run jobs={len(run_jobs)}"
    )
    print(
        f"coordinator: state={job['state'] if job else '-'} "
        f"attempts={job['attempts'] if job else '-'} elapsed={elapsed:.3f}s "
        f"lease={args.lease_seconds:.3f}s"
    )
    print("gates:")
    for name, passed in gates.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")

    report_path = root / "reports" / "n9_campaign_durability.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "schema": "uzen.n9.demo/1",
                "spec": spec.to_dict(),
                "submitters": outcomes,
                "admitted_job_id": admitted_id,
                "execution": execution,
                "job": job,
                "campaign": report.to_dict(),
                "run_jobs": run_jobs,
                "reaped": reaped,
                "elapsed_seconds": elapsed,
                "lease_seconds": args.lease_seconds,
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
