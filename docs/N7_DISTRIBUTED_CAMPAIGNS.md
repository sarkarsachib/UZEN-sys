# N7 — Distributed research campaigns

N7 extends N6 without changing its planners, scoring, campaign schema or corpus. The
only thing that moves is **trial evaluation**: instead of the coordinator's local
thread pool, every candidate is submitted as a read-only job to the N4 control-plane
queue and executed by a worker that claims it under a lease.

```text
CampaignEngine (planner, score, Pareto, SQLite)
        │
        │  DistributedTrialExecutor: submit candidate job + idempotency key
        ▼
N4 job database ── claim/lease/token ──► CampaignWorker process
        ▲                                      │
        │        SUCCEEDED {trial: metrics}    │
        └──────────────────────────────────────┘
        │
        ▼
same Trial rows, same content-addressed corpus, local-equivalent deterministic digest
```

N7 is still STRICT_PLUS: a trial does not train, fine-tune, backpropagate, update
weights or persist a model artifact. It builds a candidate in worker memory and runs a
real NumPy forward pass.

## Trial job contract

A trial request is deliberately narrow:

```json
{
  "task_scope": "inspection",
  "objective": "campaign trial <campaign-id>",
  "inputs": [],
  "operations": [],
  "policy": "STRICT_PLUS",
  "options": {
    "task_kind": "campaign_trial",
    "campaign_spec": { "...": "immutable CampaignSpec" },
    "candidate": { "digest": "...", "values": [], "config": {} }
  }
}
```

`JobService.submit` validates the task kind, rejects any input artifacts or engineering
operations, and **recomputes the candidate digest** from `values` + `config`; a client
cannot submit a forged candidate under another trial's digest. The job is stored in the
normal job/event/lease tables and is visible through `/v1/jobs`.

`JobService` handles the task kind before its ANE path. It returns:

```json
{
  "task_kind": "campaign_trial",
  "campaign_id": "...",
  "candidate_digest": "...",
  "trial": {
    "status": "ok|invalid|error",
    "metrics": {},
    "package_fingerprint": "...",
    "graph_digest": "...",
    "error": ""
  }
}
```

Invalid candidates are recorded as `status=invalid` in a successful worker job, so one
bad candidate does not poison the batch. A worker still returns a normal failed job if
the request itself is invalid or the worker cannot execute it.

## Coordinator

`DistributedTrialExecutor`:

1. submits each candidate with idempotency key
   `campaign-trial:<campaign-id>:<candidate-digest>`;
2. waits for every job to reach a terminal state;
3. converts its result into the same outcome mapping the local evaluator returns;
4. returns the outcomes in proposal order, so `sequence` and the deterministic report
   digest do not depend on which worker finished first.

It never falls back to running a trial locally while waiting. If no worker settles the
batch within `timeout_seconds`, it raises `CampaignTimeout` (HTTP 504). The engine marks
the campaign `paused` with `stop_reason=worker_timeout`; jobs that settle later remain
in the queue, and a resume resubmits the same idempotency keys and collects them. A
coordinator-side submission failure pauses with `stop_reason=trial_executor_error`
instead of leaving the campaign marked active.

## Worker

`CampaignWorker` is a small N4 worker loop for a dedicated campaign queue:

```python
from uzen.core.campaign import CampaignWorker

worker = CampaignWorker(
    "/srv/uzen-control-plane",
    "campaign-worker-a",
    claim_wait_seconds=0.25,
    max_in_flight_per_worker=1,
)
worker.warmup()          # register detected capabilities
worker.run_forever()     # claim, execute under token, report through JobService.run
```

It uses the same atomic claim and dispatch token as a remote worker. A normal
`uzen worker`/REST worker can also execute trial jobs because the task kind is handled by
`JobService`; it is not a separate protocol. The worker must run a UZEN checkout that
contains N7.

A campaign worker claims any eligible job on the queue, so it should use a dedicated
control plane. Capability routing remains N4's job; this phase does not add a second
scheduler.

## Determinism and resume

The demo runs the same four-candidate campaign twice:

- a local N6 run on one workspace;
- a distributed N7 run on a separate workspace with two real worker processes.

Both reports have the same `deterministic_digest`, which covers candidate digests,
statuses, metrics, fingerprints, graph digests, Pareto entries and the corpus merkle.
Wall-clock latency and worker identity are not in the digest. The distributed run
replays 4/4 corpus records and reports jobs from both worker processes.

Resume after a timeout or crash reuses queued/settled jobs via their idempotency keys.
Completed trial rows are never duplicated; an in-flight batch may have jobs that settle
after the coordinator has paused, which the next run collects.

## Interfaces

- **SDK**: `run_campaign(..., distributed=True, trial_timeout_seconds=..., poll_interval_seconds=...)`
- **CLI**: `uzen campaign run --spec-file spec.json --distributed --trial-timeout 120`
- **REST**: `POST /v1/campaigns` with `"distributed": true`, `trial_timeout_seconds`,
  `poll_interval_seconds`
- **MCP**: `uzen.run_campaign` accepts the same three fields
- **Worker**: `CampaignWorker` (Python) or the existing `/v1/workers/*` REST loop

The server-side coordinator remains synchronous; there is no campaign daemon that keeps
running after the HTTP request returns.

## Verification

```bash
.venv/bin/python -m pytest -o addopts="" -q tests/test_distributed_campaigns.py
.venv/bin/python examples/distributed_campaign_demo.py --root /tmp/uzen-n7-demo
```

The focused file covers: a read-only trial job, forged-digest refusal, inspection-scope
enforcement, local/distributed deterministic equality, two worker processes, SDK, REST
and MCP distributed paths, and timeout → paused → resume with reused queued jobs.

Measured on this workspace:

```text
full suite: 543 passed, 37 skipped
N7 focused: 9 passed
N7 demo: gate PASS
workers: n7-worker-0, n7-worker-1
trials: 4; corpus: 4/4 verified
```

The 37 suite skips are the documented opt-in/hardware gates: 27 live Hugging Face,
1 live S3, 1 live Redis, 4 CUDA GPU parity, and 4 CUDA compilation checks because the
optional NVRTC wheel is absent in this environment.

## Limits

1. **Coordinator is single-host.** Campaign state is SQLite in the coordinator's
   workspace. N4 jobs can be claimed by processes on one host sharing that database;
   there is no distributed campaign database or multi-coordinator election. Redis is an
   N4 job-queue adapter, not campaign storage.
2. **Synchronous API.** A REST campaign call blocks until completion, timeout or
   failure. There is no background campaign daemon.
3. **Trial evaluation is CPU NumPy.** Workers run the same `Evaluator` as N6; no CUDA
   path is implied and no task-quality metric exists.
4. **No campaign-level cancellation propagation.** Cancelling the HTTP request does not
   cancel already-submitted trial jobs; operators can cancel jobs through `/v1/jobs`.
5. **Batch-granular resume.** Trials are persisted after the batch settles. A crash
   inside a batch preserves the jobs, not partially written campaign rows.
6. **Dedicated queue recommended.** `CampaignWorker` claims any eligible queued job;
   use a dedicated control plane on shared installations.
7. **Same code version required.** Workers must understand the N7 task kind; older
   workers will fail the job rather than silently evaluating something else.
