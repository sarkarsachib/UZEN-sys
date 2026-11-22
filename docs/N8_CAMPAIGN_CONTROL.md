# N8 — Campaign control, cancellation and isolation

N8 is the production-control layer around the N6 campaign engine and the N7 trial
workers. It does not change campaign planning, scoring, corpus records or the read-only
trial payload. It adds three operational capabilities:

1. a campaign can be submitted as a **durable background job** instead of blocking an
   HTTP request;
2. **cancellation propagates** from the campaign to its coordinator job and every linked
   trial job, with a durable barrier against late trial inserts and late worker reports;
3. workers can be restricted to a **job kind**, so a dedicated trial worker cannot claim
   ordinary engineering work and a coordinator cannot claim trials.

```text
POST /v1/campaigns (background=true)
        │
        ▼
campaign_run job ──claim/lease──► coordinator worker
                                      │
                                      │ CampaignEngine.run
                                      ▼
                       campaign_trial jobs ──► trial workers
                                      │
                                      ▼
                         same campaign SQLite store and corpus

POST /v1/campaigns/{id}/cancel
        │
        ├─ durable cancel_requested flag + campaign status
        ├─ cancel linked campaign_run and campaign_trial jobs
        └─ record_trial refuses any insert after cancellation
```

## Background run job

`submit_campaign_run` creates or loads the campaign, then submits a normal N4 job:

```json
{
  "task_scope": "inspection",
  "objective": "campaign run <campaign-id>",
  "inputs": [],
  "operations": [],
  "policy": "STRICT_PLUS",
  "options": {
    "task_kind": "campaign_run",
    "campaign_spec": { "...": "immutable CampaignSpec" },
    "distributed": true,
    "resume": true,
    "max_rounds": null,
    "trial_timeout_seconds": 600.0,
    "poll_interval_seconds": 0.1
  }
}
```

`JobService` validates the request before queueing it. A coordinator worker claims it
through the usual atomic claim and token-guarded `JobService.run` path. The job result
contains the campaign report:

```json
{
  "task_kind": "campaign_run",
  "campaign_id": "...",
  "status": "completed",
  "stop_reason": "budget",
  "report": { "...": "CampaignReport" }
}
```

The job is durable: a lost HTTP client, a coordinator crash or a retry does not lose it.
An expired lease returns it to the queue. Re-running the campaign is idempotent at trial
granularity because N7 still uses one idempotency key per candidate. A coordinator worker
that handles `campaign_run` does not execute trial jobs itself; distributed mode needs at
least one worker restricted to `campaign_trial`, otherwise the trials remain queued and
the coordinator hits its typed trial timeout.

`campaigns.run_job_id` points at the current coordinator job. A second background request
for a campaign with a live run returns that job instead of starting another coordinator.
A run that failed or paused can get a new job.

## Cancellation

`CampaignEngine.cancel(campaign_id)`:

1. sets a durable `cancel_requested` flag and status `cancelling` in the campaign store;
2. cancels every job linked to the campaign (`campaign_jobs`): the coordinator and all
   trial evaluations;
3. finishes the campaign as `cancelled` with `stop_reason=cancelled`;
4. returns the report and the linked job documents.

A queued job becomes `CANCELLED` immediately. A leased job becomes
`CANCEL_REQUESTED`; its worker observes the flag before publishing. `JobService.run`
checks cancellation after execution and reports `CANCELLED` instead of publishing the
result. If a remote worker reports success late, `finish_from_worker` closes its dispatch
record as `CANCELLED` and refuses the result, so a late report cannot overwrite the
cancellation.

The store serializes the cancellation check and trial insert in
`CampaignStore.record_trial` with `BEGIN IMMEDIATE`. Once `cancel` has returned, a trial
cannot be appended: an in-flight batch may have started before the request, but its
outcomes are discarded if the flag is set. Already committed trials remain in the report
and corpus; cancellation is not a rollback.

Job cancellation and worker settlement are also one transaction each. A plain
read-then-write could observe a running job, let the worker commit `SUCCEEDED`, and then
overwrite it with `CANCEL_REQUESTED`; `request_cancel` and `finish_from_worker` therefore
use `BEGIN IMMEDIATE` and re-read the job inside that transaction. Campaign status writes
are conditional on `cancel_requested=0`, so a coordinator finishing at the same instant
cannot replace `cancelled` with `completed`. The distributed cancel test was run 100
times in a loop after the change, with every iteration reaching `CANCELLED` and no
worker threads left behind; the focused test additionally asserts every linked job is
terminal.

A completed campaign cannot be cancelled (`campaign.conflict`, HTTP 409). Cancelling an
already-cancelled campaign is idempotent.

## Job-kind isolation

The job schema now records `job_kind` from `options.task_kind`. The claiming SQL accepts
an optional kind list. `CampaignWorker` defaults to `campaign_trial`; a coordinator must
explicitly claim `campaign_run`:

```python
from uzen.core.campaign import CAMPAIGN_RUN_KIND, CampaignWorker

coordinator = CampaignWorker(
    "/srv/uzen-control-plane",
    "campaign-coordinator",
    job_kinds=(CAMPAIGN_RUN_KIND,),
)
coordinator.warmup()
coordinator.run_forever()
```

Trial workers use the default:

```python
from uzen.core.campaign import CampaignWorker

worker = CampaignWorker("/srv/uzen-control-plane", "campaign-trial-a")
```

The normal worker daemon can also restrict itself:

```bash
uzen worker run --root /srv/uzen-worker --job-kind campaign_trial
```

and `/v1/workers/claim` accepts `job_kinds`. A job with no task kind remains claimable by
an unrestricted worker; a kind-restricted worker never sees it. This is a queue filter,
not an authentication or attestation mechanism.

## Interfaces

- **SDK**: `run_campaign(..., background=True)` returns `{campaign_id, job, campaign}`;
  `cancel_campaign(id)` and `campaign_jobs(id)`.
- **CLI**: `campaign run --background`; `campaign cancel <id>`; `campaign jobs <id>`.
- **REST**: `POST /v1/campaigns` with `background=true`;
  `POST /v1/campaigns/{id}/cancel`; `GET /v1/campaigns/{id}/jobs`.
- **MCP**: `uzen.run_campaign` accepts `background`; `uzen.cancel_campaign`;
  `uzen.campaign_jobs`. The server now exposes 28 tools.

The synchronous path remains available and still blocks until the campaign completes,
times out or fails. A background request returns a queued job immediately; poll
`/v1/jobs/{id}` or `/v1/campaigns/{id}` for completion.

## Verification

```bash
.venv/bin/python -m pytest -o addopts="" -q tests/test_campaign_control.py
.venv/bin/python examples/campaign_control_demo.py --root /tmp/uzen-n8-demo
```

Focused N8 tests (10) cover job-kind filtering, background execution, request validation,
distributed cancellation, idempotent cancellation, completed-campaign conflict, the
trial-insert barrier, cancel-before-lease settlement, late worker reports, REST
cancellation, and SDK/REST/MCP background paths. The N7 suite remains green (9 tests).

Measured in this workspace:

```text
full suite: 553 passed, 37 skipped
N8 focused: 10 passed
N7 focused: 9 passed
N8 demo: gate PASS
short campaign: completed/budget, 6 trials, 3 trial workers observed
cancelled campaign: cancelled/cancelled, 7-8 trials, 9-17 linked jobs
decoy engineering job: QUEUED (not claimed by campaign workers)
corpus: 13/13-14/14 verified (all recorded benchmarks matched)
```

The 37 skips are environment gates: 27 live Hugging Face, 1 live S3, 1 live Redis,
4 CUDA GPU parity, and 4 CUDA compilation tests because the optional NVRTC wheel is not
installed in this restored environment. No N8 test is skipped.

## Limits

1. **Campaign state is still single-host SQLite.** Background execution is durable
   through the N4 job database, but the campaign store, corpus and coordinator are not a
   distributed campaign database and there is no multi-coordinator election.
2. **A background run needs a coordinator worker.** Without
   `CampaignWorker(job_kinds=(CAMPAIGN_RUN_KIND,))` the job stays `QUEUED`. Distributed
   campaigns also need trial workers; one coordinator-only worker is not enough.
3. **Cancellation is cooperative.** A running NumPy forward pass is not interrupted; its
   result is discarded. The cancellation call does not wait for a worker to finish.
4. **No partial-batch resume after cancellation.** The campaign is terminal. Already
   committed trials remain; trials evaluated after the barrier are not stored.
5. **No push notification.** Operators poll the campaign or job. There is no webhook or
   persistent campaign-event stream beyond the job/campaign event tables.
6. **One live coordinator per campaign in normal sequential use.** `run_job_id` is
   checked before submission, but two simultaneous submissions from different processes
   can still create two run jobs. N7's per-candidate idempotency keys prevent duplicate
   trial work; the campaign-level duplicate job is a coordination race, not a data-loss
   path.
7. **No task-quality metric and no training.** Metrics remain inference-only proxies;
   cancellation changes scheduling, not the mathematical result of a completed trial.
8. **A worker's kind filter is self-declared.** It prevents accidental work stealing; it
   is not a security boundary and does not replace worker authentication.
