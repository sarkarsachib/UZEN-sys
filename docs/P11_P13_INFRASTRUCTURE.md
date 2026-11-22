# P11–P13 — PostgreSQL, worker sandbox, runtimes and codecs

This note records what was built for the infrastructure phases and, just as
importantly, what could actually be executed in the authoring environment.

## P11 — PostgreSQL store and queue

`uzen/infrastructure/storage/postgres.py` implements the `Database` contract used by
`JobService`:

* schema creation/migration for jobs, job events, dispatch records, `api_tokens`,
  `artifact_acl`, artifacts and models, including `project_id`;
* atomic claims with `SELECT … FOR UPDATE SKIP LOCKED` (capability filter, priority
  and aging, lease and attempt bookkeeping);
* row-locked settlement, cancellation, event append and dispatch records, so two
  workers cannot settle the same attempt;
* per-session `statement_timeout` and an explicit `transaction()` context manager.

`postgres_queue.py` exposes the queue adapter (`claim`, `ack`, `nack`, `reap`,
`depth`) over the same database. `factory.py` provides `is_postgres_dsn()` and
`open_database()`; `JobService(db_path="postgresql://…")` therefore selects
PostgreSQL without code changes.

Connections run with `autocommit=True` and explicit transactions. This is not
cosmetic: the first implementation left an idle-in-transaction session holding the
schema lock, and a second connection's schema setup hung until the timeout.

**Verified:** `tests/test_postgres_store.py` (10 tests) starts a real bundled
PostgreSQL 16.2 via `pgserver`, then exercises schema setup from two connections,
claims, settlement, cancellation, event ordering, token/ACL operations and queue
depth. **Not verified:** physical multi-host operation, failover, partitions or
load tuning.

## P12 — Worker sandbox

`uzen/workers/sandbox.py`:

* `SandboxSpec` — command, working directory, environment allowlist, timeout, CPU
  seconds, address-space and file-size limits, process limit.
* `ProcessSandbox` — `setrlimit` in a pre-exec hook, scrubbed environment, per-run
  temporary directory, process-group creation and kill (children included), captured
  stdout/stderr and an explicit `timed_out` flag.
* `DockerSandbox` — builds the isolation argv: `--network none`, read-only root
  filesystem, `--cap-drop ALL`, `no-new-privileges`, memory/CPU/pids limits, a
  writable tmpfs and the job directory mounted read-only.
* `default_sandbox()` and `describe_isolation()` report which isolation level is in
  effect. `detect_capabilities()` advertises `sandbox` with that level.

Network-namespace isolation is probed at runtime (`unshare --net`). In this VM the
probe fails, and the report says `not_isolated` with the reason instead of claiming
isolation.

**Verified:** `tests/test_worker_sandbox.py` — 12 passed, 1 skipped. Covered:
timeout and rlimit enforcement, environment scrubbing, child-process termination,
invalid-spec refusal, Docker argv construction, availability reporting. The skipped
test runs a real container and is gated on `UZEN_TEST_DOCKER`: `/usr/bin/docker`
exists but its daemon socket is not accessible here, and `DockerSandbox.available()`
correctly returns false.

## P13 — CUDA, Rust, embedded ONNX and GGUF codecs

### CUDA batchnorm

`uzen_batchnorm_f32` was added to `uzen/runtimes/cuda/kernels.cu` (channel-wise
inference normalization with optional affine terms and epsilon), registered in
`KERNELS`/`DEVICE_OPS`, and dispatched by the executor (`_batchnorm`). The PTX was
compiled for real:

```bash
python -m uzen.runtimes.cuda.build --arch compute_80 --force --output /tmp/uzen_kernels.ptx
# report: kernels: 18, NVRTC 12.9, batchnorm present in the module
```

No GPU or driver exists here, so device execution and timing are **not verified**;
the device-marked parity tests skip. No speed claim is made.

### Rust kernel runtime

`uzen/runtimes/rust/` is a real `cdylib` crate:

* `src/lib.rs` — float32 linear (with bias), layernorm/rmsnorm, softmax,
  GELU/SiLU/ReLU/tanh/sigmoid, scale/add/mul;
* `runtime.py` — ctypes binding and operator dispatch with typed refusal;
* `build.py` — locates `cargo`, rebuilds when sources are newer than the artifact.

`cargo build --release` succeeded in 0.245 s. `tests/test_rust_runtime.py` (9 tests)
checks per-operator parity with NumPy, an MLP end to end, and refusal of unsupported
operations. The crate is not wired into the default NIR executor; it is available as
`uzen.runtimes.rust`.

### Embedded ONNX executor

`uzen/runtimes/onnx_embedded.py` decodes the ONNX protobuf wire format itself (no
`onnxruntime`, no `onnx` dependency) and executes UZEN's exportable NIR subset on
NumPy: initializers, MatMul/Add, activations, normalization, reshape/transpose,
concat/slice and softmax-style graphs. Unsupported operators raise a typed error.

`tests/test_onnx_embedded.py` (8 tests) runs models produced by UZEN's own exporter
and compares them with the NumPy reference. Output names differ between NIR and the
ONNX graph (the exporter names outputs `node:port`), so the parity test compares the
declared graph outputs in order. This is an executor for the exported subset, not a
general ONNX runtime.

### GGUF codecs

`uzen/infrastructure/storage/ggml_quant.py` now decodes:

| Type | Block | Layout |
| --- | --- | --- |
| `Q2_K` | 84 bytes / 256 values | 16 sub-blocks of 16; 4-bit scale and min per sub-block; 2-bit quants; `d`/`dmin` fp16 |
| `Q3_K` | 110 bytes / 256 values | 16 sub-blocks of 16; 12 packed bytes unpacked into 16 six-bit scales; 2-bit quants plus a 256-bit high mask (cleared bit ⇒ −4) |
| `IQ4_NL` | 18 bytes / 32 values | fp16 `d` plus 32 nibble indices into GGML's fixed non-linear table |

`tests/test_gguf_codecs.py` compares every decoded value against the reference
`gguf` package (`gguf.quants.dequantize`) on randomised blocks with pinned finite
scales, which exercises all scale fields, mask bits and nibble positions. The
existing GGUF tests were updated: `q2_k`/`q3_k`/`iq4_nl` now report
`decoder_available: true`, and the metadata-only inspection test uses `iq2_xxs`.

**Not implemented:** IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S and IQ1_S still need their
published grid/lookup tables. They remain inspectable and are refused by name on
read (`UnsupportedFormat`), never simulated.
