# Phased completion plan

This is the implementation backlog derived from the STRICT_PLUS audit. A phase is
marked **complete** only when its code, tests, interfaces and documentation exist
and the checks could run in this environment. Infrastructure phases that require a
device, database server, container runtime or multi-host network are marked
**blocked** rather than claimed.

## Phase table

| Phase | Scope | Status |
| --- | --- | --- |
| P7 | Non-causal checkpoint ingestion: 2-D normalization operator and explicit sequential CNN reconstruction with forward parity | **complete** |
| P8 | Claim integrity: remove weakened acceptance assertions, make acceptance test 13 create a real job, wire `LLMPlanner` through a real job path | **complete** |
| P9 | Tenant/project isolation: project ids on jobs and artifacts, scoped tokens, artifact ACLs, mandatory-auth deployment mode | **complete** ([P9_P10_TENANCY_OPS.md](P9_P10_TENANCY_OPS.md)) |
| P10 | Operations: keyset pagination on list endpoints, W3C trace context and OTLP-compatible spans | **complete** ([P9_P10_TENANCY_OPS.md](P9_P10_TENANCY_OPS.md)) |
| P11 | Multi-host persistence: PostgreSQL store and queue adapter with row-locked claims | **complete and tested against a real PostgreSQL 16.2 server**; not yet exercised across physical hosts ([P11_P13_INFRASTRUCTURE.md](P11_P13_INFRASTRUCTURE.md)) |
| P12 | Worker isolation: process sandbox with rlimits/process-group kill, Docker isolation, capability reporting | **complete**; process sandbox verified, Docker/network-namespace paths implemented but unavailable in this VM ([P11_P13_INFRASTRUCTURE.md](P11_P13_INFRASTRUCTURE.md)) |
| P13 | Hardware and formats: CUDA batchnorm kernel, Rust kernel runtime, embedded ONNX executor, Q2_K/Q3_K/IQ4_NL GGUF decoders | **complete for build/compile**; CUDA device execution and remaining IQ grids remain open ([P11_P13_INFRASTRUCTURE.md](P11_P13_INFRASTRUCTURE.md)) |
| P14 | WPL control flow: compile-time conditionals, constraints and `expect` postconditions | **complete** ([P14_WPL_CONTROL_FLOW.md](P14_WPL_CONTROL_FLOW.md)) |

## P7 — Non-causal checkpoint ingestion

Implemented in `uzen/core/wie/cnn_reconstruction.py` and the NIR/torch operator
tables:

* `batchnorm` inference kernel using stored running mean/variance, optional affine
  parameters, configurable epsilon and channel/last-axis normalization.
* `reconstruct_cnn_model`: an explicit sequential CNN builder. The caller supplies
  the layer specification and an optional canonical-to-source tensor map. It builds
  a real NIR graph, refuses missing or shape-mismatched tensors, validates the
  graph, and executes a real forward pass with finite-output checks.
* PyTorch inference metadata such as `num_batches_tracked` is recorded as an
  ignored metadata warning, never invented as a graph parameter.
* Tests: `tests/test_cnn_reconstruction.py` compares UZEN logits against a real
  `torch` CNN with batch normalization and exercises missing/shape-mismatch
  refusals.
* Demo: `examples/cnn_reconstruction_demo.py` reports mapped tensors, graph size,
  and parity error (observed `2.384e-07`).

Limits: this is an explicit adapter, not automatic architecture discovery. Residual
blocks, global/adaptive pooling, transposed convolution and segmentation heads are
not included yet. ViT/CNN checkpoints from the Hub still need a per-architecture
adapter or a supported `config.json` path.

## P8 — Claim integrity and planner wiring

* Acceptance test 5 no longer contains the tautological `or True`; it checks the
  structure digest and exact parameter-name set for a weight-only merge.
* Acceptance test 13 now creates a model through MCP and runs a real
  `uzen.research_weight_architecture` job, asserting the job id, terminal state,
  task scope and non-empty measurements.
* `AutonomousNeuralEngineer.execute` accepts a pre-validated plan and rejects a plan
  built for a different request.
* `JobService` accepts `options.planner = "llm"` and calls `LLMPlanner.propose`
  before execution. The deterministic planner remains the default; an unconfigured
  LLM endpoint raises a typed error instead of silently falling back.
* Test: `tests/test_llm_planner.py` runs the planner through a real local
  OpenAI-compatible HTTP endpoint, verifies the job succeeds with the LLM plan, and
  checks the unconfigured error path.

## P9 — Tenancy and authorization

Projects own jobs, artifacts and models; scoped bearer tokens are stored hashed and
minted/redeemed through `POST /v1/tokens` (plaintext returned once); artifact ACLs
grant another project read or write on one artifact; the middleware authenticates and
authorizes before the handler, so a foreign job id is a 403 rather than an existence
oracle. `UZEN_API_TOKEN` or `UZEN_REQUIRE_AUTH=1` makes authentication mandatory;
health, metrics, the operation catalogue, the policy document and the worker plane
stay reachable. Tests: `tests/test_tenancy.py`.

Limits: no per-token rate limiting, no OIDC/SSO, and expiry is the only token
lifetime control. The full route/scope table is in `docs/P9_P10_TENANCY_OPS.md`.

## P10 — Pagination and tracing

Keyset pagination over `(created_at DESC, id DESC)` with an opaque cursor on
`/v1/jobs`, `/v1/models` and `/v1/weights`; malformed cursors are a typed 400, limits
clamp to 500. W3C `traceparent` is parsed and propagated, every response carries the
server span, `GET /v1/traces` and `/v1/traces/otlp` expose recent spans, and job
execution runs inside a `uzen.job.execute` span. Tests:
`tests/test_pagination_tracing.py`.

Limits: descending order only; spans are process-local (durability is the
exporter's job).

## P11 — PostgreSQL store and queue

`uzen/infrastructure/storage/postgres.py` implements the same `Database` contract as
the SQLite store — jobs, events, dispatch records, tokens, artifact ACLs, models —
using `FOR UPDATE SKIP LOCKED` claims, row-locked settlement/cancel/event appends and
per-session `statement_timeout`. `postgres_queue.py` adds the queue adapter over the
same database, and `open_database()/is_postgres_dsn()` select the backend from the
DSN so `JobService(db_path="postgresql://…")` works without code changes. Connections
run in autocommit mode and use explicit `transaction()` blocks, because an
idle-in-transaction session blocked a second connection's schema setup.

Verified here: `tests/test_postgres_store.py` starts a real bundled PostgreSQL 16.2
(`pgserver`), creates the schema, and exercises claims, settlement, cancellation,
event ordering, tokens/ACLs and queue depth across separate connections. Not
verified: physical multi-host operation, failover, partitions, or tuning under load.

## P12 — Worker sandbox

`uzen/workers/sandbox.py` defines `SandboxSpec`, `SandboxResult`, `ProcessSandbox`,
`DockerSandbox` and `default_sandbox()`. The process sandbox applies CPU/address-space/
file-size/process rlimits, a timeout, a scrubbed environment, a per-run temporary
directory and process-group termination (including children); it probes whether a
network namespace can be created and reports `not_isolated` honestly when it cannot.
The Docker sandbox builds a real isolation argv (network none, read-only root,
capability drop, memory/CPU/pids limits, no-new-privileges). `detect_capabilities()`
advertises `sandbox` and its isolation level.

Verified here: `tests/test_worker_sandbox.py` (12 passed, 1 skipped) covers rlimit and
timeout enforcement, environment scrubbing, child-process kill, refusal of invalid
specs, the Docker argv construction and `available()` reporting. Skipped: a real
container run, because the Docker daemon socket is not accessible in this VM. The
network namespace is unavailable here, and the report says so instead of claiming
isolation.

## P13 — Hardware and formats

* **CUDA.** `uzen_batchnorm_f32` was added to `kernels.cu`, `KERNELS`, `DEVICE_OPS`
  and the executor dispatch (`_batchnorm`). The PTX was compiled for real with
  NVRTC 12.9: `python -m uzen.runtimes.cuda.build --arch compute_80 --force`
  reports 18 kernels and produces a PTX module containing batchnorm.
  **Device execution is not verified here**: there is no GPU or driver, and the
  device-marked parity tests skip. No speed is claimed.
* **Rust.** `uzen/runtimes/rust/` is a real `cdylib` crate (linear/norm/softmax/
  GELU/SiLU/ReLU/tanh/sigmoid/scale/add/mul) with a ctypes `runtime.py` and a
  `build.py` that rebuilds when sources change. `cargo build --release` succeeded
  (0.245 s) and `tests/test_rust_runtime.py` (9 tests) checks parity with NumPy, an
  MLP, and typed refusal of unsupported operators.
* **Embedded ONNX executor.** `uzen/runtimes/onnx_embedded.py` decodes the ONNX
  protobuf itself (no `onnxruntime`) and executes the supported NIR subset on NumPy.
  `tests/test_onnx_embedded.py` (8 tests) covers graphs produced by UZEN's own
  exporter, initializers, attributes, shape ops and refusal of unsupported ops.
  Only the exported NIR subset is implemented, not arbitrary ONNX.
* **GGUF codecs.** Q2_K (84-byte super-block, 4-bit scales/mins), Q3_K (110-byte
  super-block, packed 6-bit scales, high-bit mask) and IQ4_NL (18-byte block,
  non-linear lookup) are decoded in `ggml_quant.py` and compared value-for-value
  against the reference `gguf` package on randomised blocks
  (`tests/test_gguf_codecs.py`). IQ2/IQ3/IQ1 still require their published grid
  tables and stay metadata-only with a typed refusal.

## P14 — WPL control flow

`if`/`else` over parameter expressions, `constraint` declarations evaluated at
compile time, and `expect` postconditions (`finite`, `tensors=N`, `max_abs`, `min_abs`)
enforced after execution. Both branches are compiled and validated — operation
resolution, parameter typing, dataflow and policy effects — while only the selected
branch enters the plan; a condition that references a tensor is refused because
control flow may only depend on declared parameters and constants. Details and the
grammar are in [P14_WPL_CONTROL_FLOW.md](P14_WPL_CONTROL_FLOW.md).

## Remaining work

1. **CUDA device execution.** Kernels and PTX build are real and compile-verified,
   but no GPU/driver exists in the authoring environment. Device parity and timing
   remain unverified; there is no speed claim and no HIP backend.
2. **Docker/network isolation.** The Docker sandbox argv and availability probing are
   implemented; actual container execution is skipped here because the daemon socket
   is inaccessible, and `unshare --net` is unavailable, so the process sandbox reports
   `not_isolated`.
3. **PostgreSQL across hosts.** The store and queue are tested against a real local
   server, not across physical hosts, partitions or failover.
4. **Remaining GGUF IQ types.** IQ2_XXS/IQ2_XS/IQ3_XXS/IQ3_S/IQ1_S need their
   published grid tables; they remain inspectable and are refused for reads by name.
5. **WPL scope.** Control flow is compile-time only (no data-dependent branching), and
   there is no loop construct; the language remains a pipeline description, not a
   general-purpose interpreter.
