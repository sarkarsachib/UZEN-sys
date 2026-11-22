# Architecture

UZEN is a layered computational substrate. Each layer has a narrow contract; the only
path that mutates parameters runs through the operation registry, and every produced
artifact carries provenance and validation evidence.

```text
            interfaces            REST · MCP · CLI · Python SDK
                 │
        ┌────────┴─────────┐
        ▼                  ▼
      jobs/          core/campaign (N6 local, N7 distributed trials, N8 control)
 control plane       scheduler · Pareto · replay corpus · cancellation
        │                  │
        └────────┬─────────┘
                 ▼
        core/ane (planner)        objective → plan → candidates → search → diagnosis
                 │
   ┌─────────────┼──────────────┬──────────────┐
   ▼             ▼              ▼              ▼
core/wie      core/npa       core/nwpc      core/nase
inspection    alignment      registry       synthesis
   │             │              │              │
   └─────────────┴──────┬───────┴──────────────┘
                        ▼
                 core/nir (typed graph + kernels)
                        │
                        ▼
                 core/nws (tensors + store)
                        │
                        ▼
        policies/strict_plus + infrastructure (storage, provenance, security)
```

## NWS — Neural Weight Substrate

`WeightPackage` is an **enforced-immutable** bundle of

* `tensors: Mapping[str, np.ndarray]` — real arrays, no lazy placeholders. Buffers
  are copied into read-only byte-backed arrays, mappings/attributes are frozen,
  and `assert_unmodified()` re-hashes tensors at store/execution boundaries;
* `metas: dict[str, TensorMeta]` — shape, dtype, inferred **role**, **origin**
  (`inherited`, `extracted`, `transformed`, `analytically_derived`,
  `randomly_initialized`, `search_selected`), source artifact, applied operations;
* `family` + `family_spec` — the architecture this parameter set belongs to;
* `fingerprint()` — SHA-256 over canonical tensor bytes (value identity);
* `structure_digest()` — SHA-256 over names/shapes/dtypes (architecture identity).

Weight-only operations must preserve `structure_digest()`; this is asserted in tests.

`ArtifactStore` is content-addressed:

```text
<root>/artifacts/<aa>/<sha256>/{manifest.json, tensors.safetensors, lineage.json}
```

`artifact_id` is derived from schema version, tensor fingerprint, family, operation,
parents, operations and seeds — so retries are idempotent and re-running a job cannot
mutate an existing artifact. Publication is atomic (write temp dir, then rename), and
`verify()` re-hashes the payload against the manifest checksum.

Remote transport (`uzen.infrastructure.storage.object_store`) moves the same
artifact directories through any S3-compatible endpoint. The client implements
SigV4 with the standard library, streams uploads (multipart above a configurable
threshold), retries transient failures with a fresh signature, and verifies both
the object SHA-256 and the tensor fingerprint before publishing a downloaded
artifact. Parent lineage and the artifact's stored tokenizer are transported with
it, so a fetched artifact verifies and executes exactly like a local one.

## NIR — Neural Intermediate Representation

A typed dataflow graph, not a layer list:

* `Node(id, op, inputs: dict[port, Ref], params)`, `Ref("node:port")` / `Ref("input:name")`;
* symbolic dims (`"B"`, `"T"`) resolved per execution; concrete dims checked exactly;
* per-operator shape inference (`OperatorSpec.infer_types`) and kernels
  (`OperatorSpec.run`) in one registry, so a graph cannot validate but fail to run;
* cycle detection via Kahn topological sort, with an explicit cycle error;
* `Graph.validate(package)` checks declared inputs, parameter binding, shapes and
  dtype compatibility; `require_valid` raises `GraphValidationError`.

The interpreter (`executor.execute`) binds parameters **by tensor name**, resolves
symbols, runs kernels in topological order, and raises `ShapeMismatch` with node id and
input shapes on failure. `capture=` returns intermediate activations for VEL/NPA.

The 24 registered families construct graphs from specs, allocate parameter shapes, and
initialize analytically where a closed form exists (orthogonal rows, unit norms, zero
bias) or label tensors `randomly_initialized` otherwise.

## NWPC — operation registry

The single choke point for parameter change. Each `OperationSpec` declares:

* `code` (`w.<family>.<name>`), human title, effect (`param_transform`,
  `param_generate`, `param_select`, `arch_mutation`, `analysis`, ...);
* `maturity` (`implemented` / `prototype` / `planned` / `unsupported`);
* allowed `scopes` (task-scope matrix);
* typed `params` (`ParamSpec` with ranges/choices) — unknown parameters are rejected,
  not ignored;
* `assumptions` — the mathematical conditions under which the operation is defensible;
* `requires_alignment`, `mutates_architecture`, `min/max_inputs`.

`invoke(code, inputs, params, scope=..., seed=...)` performs: registration check →
maturity check → scope check → STRICT_PLUS authorization → parameter validation →
function execution → origin/provenance composition. Inputs are never mutated; the
output is a new `WeightPackage` with per-tensor origins and an `OperationRecord`
containing policy decisions, seed, details and parent artifact ids.

`invoke_search` runs derivative-free search (CMA-ES, evolution strategies, Bayesian
optimization, coordinate descent, simulated annealing, beam search) over
low-dimensional transformation parameters only — never over billions of individual
weights.

## NPA — alignment

Two models can share shapes yet live in different learned bases. `ParameterAligner`
tries `identity`, `procrustes` (closed-form SVD), `linear_map` (ridge least squares),
`permutation` (Hungarian assignment on a cosine cost matrix, plus attention-head
grouping), `basis` (whitening/colouring) and `activation` (forward-pass CKA + neuron
matching). `auto` evaluates candidates and selects the best-scoring defensible one.

If the best score is below the threshold, the report is `ok=False` and `require()`
raises `NoValidAlignment` (`npa.no_valid_alignment`). The report always states
`semantic_equivalence_claimed: false`: a numeric score is evidence, not proof.

## NASE — architecture synthesis

Build graphs from specs, compose definitions (`sequential`/`parallel`/`residual`),
substitute operators, bypass nodes and truncate layers. Every transformation validates
the resulting graph and reports invalidated parameters, so callers know exactly which
tensors must be re-bound. `from_graph` rebuilds an editable definition from a stored
NIR graph and preserves its family spec; composition selects the semantic output
(`out`, `logits`, `y`, `reconstruction`, `eps_pred`) rather than relying on JSON key
order. The service persists composed/edited graphs with parent lineage through SDK,
CLI, REST and MCP.

## WComp / WPL

WPL is a small declarative language with a real lexer and recursive-descent parser:

```text
program merge_and_scale
  input base: weights
  input other: weights
  param t: float = 0.5
  param factor: float = 1.0
  step blend = linear_interpolate(base, other, t=t)
  step out = scale(blend, factor=factor)
  output result = out
end
```

`compile_program` resolves short operation names against the registry, type-checks
declared parameters, and produces a `CompiledPipeline` whose `CompilationReport`
records per-stage effects, maturity, policy decisions and compatibility. Execution
returns a `PipelineResult` with the output artifact id, provenance and per-stage
records.

## VEL — verification

Independent sections, each reported separately (they are not interchangeable):

* **structural** — tensor validity, shape/dtype consistency, architecture unchanged;
* **numerical** — finite values, distribution statistics, outlier/anomaly flags;
* **functional** — real forward pass, output shapes, finiteness, optional comparison
  against a reference (relative L2, allclose);
* **regression** — worst relative L2 difference across outputs;
* **performance** — latency statistics over repeated runs, device, memory estimate;
* **provenance** — lineage parents and operation records;
* **policy** — STRICT_PLUS decisions attached to the producing operations.

## ANE — autonomous engineer

`plan()` interprets the request, applies the scope matrix, selects operations by
objective keywords and available operations, and emits typed `PlanStage`s with search
spaces. `execute()` runs each stage, generates candidates (grid/random low-dimensional
sampling), scores them with an `Objective` (fidelity, distance-to-target, sparsity,
activation agreement, size, custom callables), searches within the stage's search
space, keeps the best candidate per stage, verifies the final package, and returns a
trace plus diagnosis. When alignment fails it tries the next defensible strategy; when
none exists it reports `NO_VALID_ALIGNMENT` rather than inventing a result.

`LLMPlanner` optionally consults an OpenAI-compatible endpoint, but every proposed
stage is re-validated through the same registry/scope/policy path.

## Campaigns (N6/N7/N8/N9)

`uzen.core.campaign` is a separate research loop over the deterministic engines. A
seeded planner (`grid`/`random`/`evolution`) proposes architecture specs or weight
transforms; an evaluator builds the package and executes a real NumPy forward pass;
the engine scores objectives, applies constraints, computes a Pareto front and
persists every trial in SQLite. `max_rounds` pauses at a batch boundary and a later
run resumes from the same seed. Successful trials are written to a content-addressed
corpus whose `verify`/`replay` paths rebuild and execute the candidate instead of
trusting stored numbers. Campaigns never train, fine-tune or update weights by
gradients, and their proxy metrics are not task quality.

N7 (`uzen.core.campaign.distributed`) passes a `DistributedTrialExecutor` to the same
engine. It submits each proposal as a read-only `inspection` job with
`options.task_kind=campaign_trial`; `CampaignWorker` claims those jobs through the N4
scheduler and returns metrics through the token-guarded job result. The coordinator
waits for the batch, stores the same trials and corpus records as a local run, and
pauses on a typed timeout. The campaign database remains local to the coordinator.

N8 (`uzen.core.campaign.control`) makes a campaign run itself an N4 job with
`options.task_kind=campaign_run`. A coordinator worker executes it through the same
lease/token path as any other job, so the run survives a lost HTTP request or coordinator
crash. Campaign jobs are linked in `campaign_jobs`; `CampaignEngine.cancel` sets a
durable cancel flag, cancels the coordinator and trial jobs, and blocks new trial inserts
at the SQLite write boundary. Late worker reports are settled as `CANCELLED`, never as
successful results. `job_kind` is stored on job rows and can be used as a claim filter, so
`CampaignWorker` defaults to `campaign_trial` while a coordinator claims only
`campaign_run`.

N9 (`uzen.core.campaign.control` and `uzen.core.campaign.store`) admits one coordinator
per campaign across processes. A submission reserves `run_job_id` atomically under
`BEGIN IMMEDIATE` before creating the job under that id, so a concurrent submitter
waits for and returns the same job instead of creating a second one. `JobService.run`
can also keep a leased execution alive with a token-guarded renewal thread;
`CampaignWorker` enables it with its configured lease so a long campaign is not reaped
mid-run. The campaign database and reservation remain single-host SQLite.

## Control and execution planes

`JobService` owns a SQLite job database (states, events, leases, idempotency keys,
artifact index) plus the artifact store and engines. States:
`CREATED → QUEUED → DISPATCHED → RUNNING → VALIDATING → SUCCEEDED|FAILED`, with
`CANCEL_REQUESTED → CANCELLED`. Job requests are persisted before execution, so a
retry re-reads the same authorized scope.

`WorkerDaemon` runs jobs either locally or against the REST control plane
(register → heartbeat → claim under lease → fetch inputs → execute → report). Workers
advertise detected capabilities (CPU/memory/Python/torch, supported operation codes,
operators, families, artifact transports, native weight layouts). They never execute
user-supplied scripts: only registered operations over artifacts.

`JobScheduler` owns distribution: it derives a requirement set from each job request,
stores it with the job, and lets the claiming statement take a lease only when the
worker's advertised capability rows satisfy it and the worker is not draining. Claims
order by operator priority plus optional aging, and mint a `dispatch_token`; renewals
and reports must present it, and `dispatch_records` makes the first terminal report the
only one that applies. A worker that stops renewing loses its lease: the reaper returns
the job to the queue after an exponential backoff, or fails it with `job.lease_expired`
once attempts are exhausted. An operator can drain a worker or preempt a job
cooperatively, and a long-poll claim returns as soon as work appears. Queued jobs no
registered worker can run are reported as unroutable with the missing capability, never
left to starve silently.

Worker authentication uses versioned HMAC envelopes (`uzen.worker.envelope/2`) with
signing-key rotation, an optional signed expiry, a bounded timestamp window and either
an in-process or durable SQLite nonce guard. Scoped hashed credentials support
rotation and revocation; plaintext secrets are never persisted.

## Design boundaries

* Base44 (or any host) can provide control-plane metadata, orchestration and API
  exposure. It must never pretend to execute tensor math.
* Tensor computation happens in Python/NumPy (reference), PyTorch (adapter) or the
  native C++ CPU runtime. The native runtime has two paths: a direct executor that
  calls one kernel per node, and a compiled plan that lowers the graph into steps
  with fusion, prepacked weights and a liveness buffer plan. Both implement only
  their documented float32 operator sets and raise `UnsupportedOperator` for
  everything else; layout operators are NumPy byte movement and are reported as
  such.
* Advanced parameter representations (N3) are per-weight native layouts on the
  compiled path — blocked dense, transposed, group-wise int8/int4, low-rank factors
  and CSR sparse — each packed once with a measured cost and error report, and
  dispatched by the plan to its own kernel. The NIR graph language is unchanged.
* Distributed execution is a control-plane concern: one job database (SQLite across
  processes) or a Redis/Valkey queue, capability-routed claims, token-guarded leases and
  idempotent reports. Compute stays where it was — NumPy, torch, the native CPU
  runtime or the CUDA runtime — and a device requirement only routes work; it never
  implies a kernel.
* CUDA (N5) is a real execution backend, not a routing label. `kernels.cu` is compiled
  to PTX with NVRTC or nvcc, loaded through the driver API and launched per node;
  weights are resident, activations are pooled, host layout operators and transfers are
  reported separately, and `fallback="error"` refuses without a device. On the
  authoring host (no driver/device) the kernels compile and assemble but are not
  executed, and no GPU speed is claimed. HIP is not implemented.
* Rust integration remains a future layer, not a stub module.
