# API Contracts

All interfaces share one engine. Responses are JSON; errors carry a machine-readable
`code` and HTTP status.

## Error shape

```json
{
  "detail": {
    "code": "policy.violation",
    "message": "STRICT_PLUS prohibits effect 'gradient_update'",
    "details": {"operation": "w.train.finetune"}
  }
}
```

| Code | HTTP | Meaning |
| --- | --- | --- |
| `policy.violation` | 403 | prohibited effect or unauthorized plan |
| `policy.scope_violation` | 403 | effect outside the authorized task scope |
| `registry.not_registered` | 400 | operation code unknown |
| `operation.unavailable` | 409 | operation planned/unsupported or wrong scope |
| `params.invalid` | 400 | missing/unknown/out-of-range parameter |
| `tensor.shape_mismatch` | 400 | incompatible shapes or missing parameters |
| `tensor.invalid` | 400 | non-finite or corrupt tensors |
| `npa.no_valid_alignment` | 422 | no defensible parameter correspondence |
| `graph.invalid` | 400 | invalid NIR graph or parameter bindings |
| `format.unsupported` | 415 | format not supported (e.g. pickle checkpoints) |
| `artifact.not_found` | 404 | unknown artifact id |
| `artifact.immutable` | 409 | attempted rewrite of an existing artifact |
| `job.state` | 409 | illegal job transition |
| `auth.invalid` / `auth.missing` | 401 | bad or missing bearer token |
| `worker.auth_failed` | 401 | invalid worker credentials |
| `runtime.unavailable` | 503 | optional backend missing (e.g. torch) |
| `budget.exhausted` | 429 | search budget exhausted |
| `campaign.conflict` | 409 | campaign id reused with a different spec, or a completed campaign cannot be cancelled |
| `campaign.not_found` | 404 | unknown campaign id |
| `campaign.cancelled` | 409 | the campaign was cancelled before the operation could finish |
| `campaign.timeout` | 504 | a distributed trial batch did not settle before its deadline |

## REST (v1)

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | liveness, policy name |
| GET | `/v1/stats` | store/db stats + detected capabilities |
| GET | `/v1/operations` | catalogue (`family`, `executable_only`) |
| GET | `/v1/operations/{code}` | one operation spec |
| GET | `/v1/policies/strict-plus` | policy description, effects, scopes |
| GET | `/v1/architectures/families` | supported families + specs |
| POST | `/v1/architectures/synthesize` | build a model from a family spec |
| POST | `/v1/architectures/research` | compare architecture variants |
| POST | `/v1/architectures/compose` | compose stored graphs (sequential/parallel/residual), validate, execute and persist with lineage |
| POST | `/v1/architectures/{id}/edit` | bypass, truncate or substitute a stored graph and persist the surviving parameters |
| POST | `/v1/campaigns` | run/resume an N6 campaign synchronously (`spec`, `resume`, `max_rounds`); optional `distributed=true` submits N7 trial jobs to workers (`trial_timeout_seconds`, `poll_interval_seconds`); `background=true` enqueues an N8 `campaign_run` job and returns `{campaign_id, job, campaign}` |
| GET | `/v1/campaigns` | list campaigns |
| POST | `/v1/campaigns/verify` | replay benchmark records (optional `digests`) |
| GET | `/v1/campaigns/{id}` | campaign report and trials |
| GET | `/v1/campaigns/{id}/events` | campaign event history |
| GET | `/v1/campaigns/{id}/jobs` | linked coordinator and trial jobs |
| POST | `/v1/campaigns/{id}/cancel` | cancel a campaign and its linked queued/leased jobs |
| GET | `/v1/campaigns/{id}/corpus` | corpus merkle root and record digests |
| GET | `/v1/benchmarks/{digest}` | replay one content-addressed benchmark record |
| POST | `/v1/models/import` | import a local HF directory/file, weights-only `.pt/.pth/.bin/.ckpt`, single-file `.onnx`, or `repo_id` + `revision`; returns the compatibility report or provenance report, plus graph/reconstruction status when available |
| POST | `/v1/models/{id}/quantize` | persist a weight-only int8 artifact variant; returns the new artifact id and quantization report |
| POST | `/v1/models/{id}/forward` | execute a stored reconstructed graph on tokens/positions; optional `capture` node ids or `["layers"]`; `quantize: "int8"` returns the quantization report |
| POST | `/v1/models/{id}/forward-text` | tokenize text with the artifact's stored `tokenizer.json`, then execute the graph; accepts `quantize: "int8"` |
| POST | `/v1/models/{id}/generate-text` | greedy, seeded sampling or beam generation with EOS/context stopping; `quantize: "int8"` quantizes once and returns the report |
| POST | `/v1/models/compare` | compare source and weight-modified model outputs with recorded tolerances |
| POST | `/v1/models/compare-quantized` | compare a floating-point source with its persisted quantized variant; reports per-output and aggregate drift, optional failure thresholds, and the quantization report |
| GET | `/v1/models` | model registry |
| GET | `/v1/weights` | weight artifact registry |
| POST | `/v1/weights/analyze` | WIE inspection (`artifact_id` or `path`) |
| POST | `/v1/weights/generate` | weight generation |
| POST | `/v1/weights/modify` | registered modification |
| POST | `/v1/weights/reconstruct` | reconstruction job |
| POST | `/v1/weights/evaluate` | VEL verification |
| GET | `/v1/artifacts/{id}` | manifest |
| GET | `/v1/artifacts/{id}/verify` | integrity verification |
| GET | `/v1/artifacts/{id}/lineage` | lineage chain |
| GET | `/v1/artifacts/{id}/export` | export (safetensors download) |
| POST | `/v1/jobs` | submit (`Idempotency-Key` supported) |
| GET | `/v1/jobs` | list (`state`, `limit`) |
| GET | `/v1/jobs/{id}` | job detail |
| POST | `/v1/jobs/{id}/run` | run synchronously |
| POST | `/v1/jobs/{id}/cancel` | cancel |
| GET | `/v1/jobs/{id}/events` | progress events |
| GET | `/v1/workers` | worker registry |
| POST | `/v1/workers/register` | register, returns one-time secret |
| POST | `/v1/workers/heartbeat` | liveness |
| POST | `/v1/workers/claim` | claim the highest-priority eligible job (capability + budget matched), returns the job plus `lease.token`; optional `wait_seconds` long-polls and `job_kinds` restricts the claim |
| POST | `/v1/workers/renew` | renew lease (requires `token`) |
| POST | `/v1/workers/report` | report a terminal state for a held lease (requires `token`); answers `accepted`, `duplicate`, `stale` or `rejected` |
| GET | `/v1/scheduler` | queue depth, leased/retried counts, worker liveness, unroutable jobs with reasons |
| POST | `/v1/scheduler/reap` | return expired leases to the queue with their backoff window (also runs on every claim) |
| POST | `/v1/jobs/{id}/preempt` | revoke the lease, requeue at once; the old token is answered `stale` |
| POST | `/v1/workers/{id}/drain` | stop routing new claims to a worker; its current lease continues |
| POST | `/v1/workers/{id}/resume` | put a draining worker back into rotation |
| GET | `/v1/jobs/{id}/dispatch` | every dispatch attempt: worker, token, outcome |
| GET | `/v1/jobs/{id}/eligibility` | which registered workers could take this job, and what each one lacks |

Auth: `Authorization: Bearer <UZEN_API_TOKEN>` when the variable is set. Worker routes
carry `worker_id` + `secret` in the body; the control plane compares
`SHA-256(secret)` against the stored hash in constant time.

## Job request

```json
{
  "task_scope": "weight_modification",
  "objective": "blend two compatible parameter sets",
  "inputs": [{"name": "base", "artifact_id": "..."}, {"name": "other", "path": "model.safetensors"}],
  "operations": [{"op": "w.arithmetic.linear_interpolate", "params": {"t": 0.4}}],
  "constraints": {"max_tensors": 512},
  "policy": "STRICT_PLUS",
  "evaluation_plan": {"metric": "distance_to_target", "target_artifact_id": "..."},
  "output_format": "safetensors",
  "priority": 10,
  "seed": 0,
  "budget": {"candidates": 8, "evaluations": 16, "search_algorithm": "cma_es"}
}
```

`priority` is an optional scheduling bias (clamped to ±1000; higher runs first, with
optional aging against starvation). Inputs accept `artifact_id`, `path` or `inline`
arrays. The job result contains the
plan, per-stage operation records, policy decisions, candidates with metrics, the best
candidate's artifact id, the VEL report, trace, warnings and diagnosis.

## Python SDK

```python
from uzen import Uzen

local  = Uzen(root="./ws")
remote = Uzen(base_url="http://localhost:8080", token="...")

local.inspect(path)                     # WIE report (no pickle)
local.import_weights(path, family=...)  # artifact id, format + provenance report
local.import_model(path_or_repo_id, revision="main")  # artifact id, compatibility, graph/reconstruction report
local.forward_model(artifact_id, tokens=[[0, 1, 2, 3]], capture="layers")
local.forward_text(artifact_id, text="hello world")
local.generate_text(artifact_id, text="hello world", max_new_tokens=8)
local.quantize_model(artifact_id)
local.compare_models(source_id, candidate_id, tokens=[[0, 1, 2, 3]])
local.compare_quantized_model(source_id, quantized_id, tokens=[[0, 1, 2, 3]])
local.analyze(artifact_id)
local.generate(family=..., spec=..., seed=0)
local.modify(artifact_id, "svd_truncate", params={"rank": 32})
local.evaluate(artifact_id, family=..., spec=...)
local.submit_job(request); local.run_job(job_id); local.job_events(job_id)
local.run_local(request)                # submit + run, returns full result
local.run_campaign(spec, background=True)  # durable N8 campaign-run job
local.cancel_campaign(campaign_id); local.campaign_jobs(campaign_id)
local.verify_artifact(artifact_id); local.lineage(artifact_id)
local.operations(); local.policy(); local.families()
```

## CLI

`uzen --help` lists groups: `ops`, `policy`, `families`, `inspect`, `import`,
`analyze`, `modify`, `generate`, `evaluate`, `quantize`, `forward`, `generate-text`,
`compare-models`, `compare-quantized`, `align`, `research`, `architecture`, `campaign`,
`artifacts`, `job`, `serve`, `worker`, `mcp`, `sdk`. `uzen forward` accepts exactly one of `--tokens` or `--text`; `--text` uses the
imported tokenizer. Errors print `{"error": {...}}` to stderr and exit 2.

## MCP

Tools: `uzen.import_model`, `uzen.quantize_model`, `uzen.forward_model`,
`uzen.forward_text`, `uzen.generate_text`, `uzen.compare_models`,
`uzen.compare_quantized_model`, `uzen.inspect_model`, `uzen.analyze_weights`,
`uzen.plan_engineering`,
`uzen.generate_weights`, `uzen.modify_weights`, `uzen.reconstruct_weights`,
`uzen.research_weight_architecture`, `uzen.research_model_architecture`,
`uzen.modify_architecture`, `uzen.compose_architectures`, `uzen.create_model`,
`uzen.evaluate_candidate`, `uzen.inspect_provenance`, `uzen.get_job_status`,
`uzen.run_campaign` (including `distributed` trial execution and `background` durable
runs), `uzen.get_campaign`, `uzen.list_campaigns`, `uzen.cancel_campaign`,
`uzen.campaign_jobs`, `uzen.verify_campaign_corpus`, `uzen.replay_benchmark`.
The server exposes 28 tools.

Transport: JSON-RPC 2.0 over stdio; `initialize`, `tools/list`, `tools/call`, `ping`.
Tool failures return `isError: true` with the typed error payload rather than a
fabricated success.

## Worker protocol (`uzen.worker/1`)

1. `register` → `{worker_id, secret}` (secret shown once, stored as a hash).
2. `heartbeat` with detected capabilities.
3. `claim` → `{job, lease}` or `job: null`. The lease carries `token`, `attempt` and
   `expires_at`; a job is only returned when the worker's advertised capabilities
   satisfy the requirements stored with the job (operations, operators, family, policy,
   device, native layouts, artifact transport, CPU/memory budgets).
4. Renew with the token while executing (the daemon renews every `lease_seconds / 3`).
5. `report` with state, result/error and the **token**. The first terminal report
   applies; a repeat is `duplicate`, a report for a superseded attempt is `stale`.
6. Inputs are fetched and verified before execution (`ObjectStoreArtifactTransport` or
   the shared exchange directory); the local job id equals the control-plane job id, so
   a re-delivered claim reuses the stored result instead of recomputing.

Artifact transport in this build: shared filesystem (`exchange_dir`), inline
tensors, or an S3-compatible object store.
`ObjectStoreArtifactTransport` (`uzen/infrastructure/storage/object_store.py`)
pushes and fetches content-addressed artifacts with their parent lineage and
stored tokenizer sidecar. It signs every request with AWS Signature Version 4,
streams single and multipart uploads, and verifies SHA-256 plus the UZEN tensor
fingerprint before publishing a fetched artifact.

Configure it with `UZEN_S3_BUCKET`, `UZEN_S3_ACCESS_KEY`, `UZEN_S3_SECRET_KEY` and
optionally `UZEN_S3_ENDPOINT`, `UZEN_S3_REGION`, `UZEN_S3_PREFIX`,
`UZEN_S3_PATH_STYLE` and `UZEN_S3_SESSION_TOKEN`. Missing credentials raise
`storage.object_store_unavailable`; the transport is invoked explicitly by a
worker or caller, not implicitly by the daemon.
