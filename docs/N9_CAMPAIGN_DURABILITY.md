# N9 — Campaign run durability: admission and lease keepalive

N9 closes two coordination gaps left by N8 without changing the external campaign
contract:

1. concurrent background submissions could each create a `campaign_run` job because
   `run_job_id` was read before a job was written;
2. an in-process coordinator did not renew its job lease, so a campaign lasting longer
   than the lease could be reaped and executed a second time.

Both are control-plane concerns. Campaign evaluation remains inference-only, and N9 adds
no training, gradient update or task-quality metric.

## Single-coordinator admission

`CampaignStore` now owns a compare-and-set reservation:

```text
submit_campaign_run(service, spec)
  create/load campaign
  if completed/cancelled: return report
  reservation = campaigns.run_job_id + run_reserved_at
  if reservation names a live job: return that job
  if reservation is terminal or stale: release only if it still matches
  candidate_id = uuid4()
  reserve_run(campaign_id, candidate_id)   -- BEGIN IMMEDIATE, one winner
  submit(request, job_id=candidate_id)     -- job id is reserved before it exists
  link_job(campaign_id, candidate_id)
```

`reserve_run` runs under `BEGIN IMMEDIATE`, re-reads `run_job_id` inside the
transaction, and succeeds only when the slot is empty. The winner already knows the job
id it will create, so a loser can wait for that exact job rather than creating an orphan
job of its own. A fresh reservation whose job has not appeared yet is treated as an
in-progress winner; a reservation older than 30 seconds whose job never appeared is
recovered. The wait is bounded at five seconds and returns a typed
`CampaignConflict` instead of hanging.

`JobService.submit(..., job_id=...)` is the internal seam that makes the reservation
usable. It is not a new REST/MCP field: callers still submit a campaign specification,
and the normal request validation and capability requirements are applied.

The reservation does not replace job state. A terminal coordinator job still permits a
new run after a failed or paused campaign; the old reservation is released by
compare-and-set, and concurrent releasers race into a single new reservation. The same
transaction refuses to reserve when the campaign's durable `cancel_requested` flag is
set, so a cancellation that lands during admission cannot be reopened by a later
submission.

## Lease keepalive

`JobService.run(..., lease_seconds=N)` starts a token-guarded keepalive thread after the
job enters `RUNNING` and stops it before returning. The thread renews every `N/3`
seconds (bounded to 0.02–30 seconds) through `Database.renew_lease`. If renewal returns
false — the lease was reaped or another worker holds the token — the thread exits; the
original execution still cannot publish because settlement is token-guarded.

`CampaignWorker.run_once` passes its configured `lease_seconds`, so a long local
campaign keeps attempt 1 instead of being requeued mid-run. The remote worker daemon
already renews its own lease; it is unchanged. A local caller that omits
`lease_seconds` keeps the old behavior.

The keepalive does not make cancellation preemptive: a running NumPy forward pass is
still allowed to finish and its result is discarded if cancellation won.

## Verification

Focused tests are in `tests/test_campaign_durability.py`:

- four threads with independent `JobService` connections release from a barrier and
  submit the same campaign; exactly one job id is returned and stored;
- a stale reservation naming a missing job is recovered;
- a campaign with a durable cancellation request is not reopened by a new submission;
- a slow leased execution runs for 1.2 seconds under a 0.5-second lease while the
  reaper runs at 0.7 seconds; the job remains leased, succeeds, and records one attempt.

The real-process demo starts four submitter processes, admits one coordinator, and runs
a 64-trial campaign for longer than its initial lease while a reaper thread expires
leases every 10 ms.

```bash
.venv/bin/python -m pytest -o addopts="" -q tests/test_campaign_durability.py
.venv/bin/python examples/campaign_durability_demo.py --root /tmp/uzen-n9-demo
```

Measured in this workspace:

```text
full suite: 557 passed, 37 skipped
N9 focused: 4 passed
N8 focused: 10 passed
N7 focused: 9 passed
demo: 9/9 gates PASS
submitters: 4/4 returned one admitted job
campaign: completed/budget, 64 trials
coordinator: SUCCEEDED, attempts=1, elapsed ~0.4s, lease 0.15s
reaper: no requeue of the admitted job
```

The demo is timing-sensitive only in that the campaign must outlive the initial lease;
it reports that condition as a gate rather than asserting it silently.

## Limits

1. **Admission is cross-process on one SQLite database.** It uses SQLite's
   `BEGIN IMMEDIATE` serialization, not a distributed lock. The campaign store remains
   single-host and is not safe on a network filesystem.
2. **There is still no multi-coordinator election.** Admission prevents duplicate
   `campaign_run` jobs submitted through `submit_campaign_run`; an operator who
   bypasses that helper and creates a campaign job directly is outside the protocol.
3. **The keepalive protects an in-process leased execution.** A worker process that
   dies still loses its lease and the job is reaped normally. Renewal is not a
   distributed heartbeat service, and the job remains single-owner.
4. **Reservation recovery is time-based.** A submitter that pauses for more than 30
   seconds between reserving and creating the job can have its slot recovered. This is
   intentional liveness behavior; the resulting job id is still returned to every
   waiter that observes the reservation.
5. **No push notification.** Operators still poll campaign/job events; N9 does not add
   webhooks or a persistent subscription stream.
6. **Cancellation remains cooperative.** The keepalive extends a lease; it does not
   interrupt a trial or make a cancellation synchronous.
7. **No task-quality claim.** Campaign metrics remain architecture/weight proxies from
   real inference. N9 changes coordination, not the mathematical result.
