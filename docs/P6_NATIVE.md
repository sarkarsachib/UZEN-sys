# P6 — Native computational substrate

This document records UZEN's own phase plan for the master roadmap's **Phase 6:
Advanced computational substrate**, and exactly what is implemented. The master
specification lists Rust integration, a native compiler, C++/CUDA kernels,
additional runtime backends, advanced parameter representations and large-scale
autonomous research. Those are too large to pretend to finish in one step, so
they are split into six core phases, plus the N7, N8 and N9 post-P6 campaign
extensions.

No phase below is claimed because a directory or interface name exists. A phase is
`implemented` only when its kernels/engine execute real computation and the
listed gate passes.

## Phase plan

| Phase | Deliverable | Status |
| --- | --- | --- |
| **N1** | Native CPU arithmetic: C++17 kernels, ctypes ABI, NIR executor, native runtime adapter, parity tests, working demo | **implemented** |
| **N2** | Native graph compiler: lower NIR to a native execution plan, fuse linear+activation and residual+norm, prepack weights, add conv/pool/RNN/SSM/GCN/MoE/int8 kernels, single-pass attention, liveness memory planner | **implemented** |
| **N3** | Advanced parameter representations: block-wise int4/int8 with scales, low-rank/sparse native layouts, packed transposed weights, representation-aware execution | **implemented** |
| **N4** | Distributed execution: durable queue adapter, capability-aware worker scheduling, leases with expiry and attempts, priority and backoff, preemption and draining, idempotent dispatch, remote artifact fetch | **implemented** (SQLite verified across processes; Redis/Valkey RESP2 adapter verified against a live Redis 7.4 server on localhost) |
| **N5** | GPU/native boundary: real CUDA kernels, device allocator, transfer-aware dispatch, CPU fallback policy that is explicit rather than silent | **implemented at the CUDA boundary** (17 kernels compile to PTX with NVRTC and assemble with `ptxas`; driver-API runtime and GPU tests present; execution not verified in the authoring environment because it has no driver/device; HIP not implemented) |
| **N6** | Large-scale autonomous research: candidate-evaluation batches, experiment scheduler, architecture/parameter search campaigns, reproducible benchmark corpus | **implemented** (in-process batch scheduler; inference-only proxy metrics; see [N6_CAMPAIGNS.md](N6_CAMPAIGNS.md)) |
| **N7** | Distributed campaign trials over the N4 worker queue | **implemented** (post-P6 extension; see [N7_DISTRIBUTED_CAMPAIGNS.md](N7_DISTRIBUTED_CAMPAIGNS.md)) |
| **N8** | Campaign control: durable background runs, cancellation propagation and job-kind isolation | **implemented** (post-P6 extension; see [N8_CAMPAIGN_CONTROL.md](N8_CAMPAIGN_CONTROL.md)) |
| **N9** | Campaign run durability: atomic single-coordinator admission and lease keepalive | **implemented** (post-P6 extension; see [N9_CAMPAIGN_DURABILITY.md](N9_CAMPAIGN_DURABILITY.md)) |

N5 is implemented as code and compile-verified; its GPU tests remain skipped until run on
a machine with a device. N6 runs real architecture and weight-transform campaigns with
SQLite resume and a replayable corpus; it is not a distributed training or task-quality
system.

## N1 — implemented native CPU layer

### What runs natively

The checked-in `uzen/runtimes/native/kernels.cpp` is compiled with the local C++
toolchain into a shared library and called through a small C ABI with `ctypes`.
Compilation is cached by a hash of the source, flags, platform and machine
architecture.

| Operator | Native implementation | Limits |
| --- | --- | --- |
| `linear` | float32 GEMM with optional bias, double accumulators | float32 only; any batch rank flattened to rows |
| `matmul` | float32 GEMM | rank-2 operands only |
| `layernorm` | mean/variance over the last axis, double accumulation | last axis; optional weight/bias |
| `rmsnorm` | root-mean-square over the last axis | last axis; optional weight |
| `softmax` | max-shifted stable softmax | last axis only |
| `gelu`, `gelu_exact`, `silu`, `relu`, `tanh`, `sigmoid` | elementwise kernels | float32 |
| `add`, `sub`, `mul` | broadcasting elementwise kernels | float32 |
| `scale`, `clip` | elementwise kernels | float32 |
| `embedding` | int64 gather from a float32 table | duplicate indices use last write |
| `mean`, `sum` | reduction over the last axis | last axis; optional keepdims |
| `attention` | causal, sliding-window and ALiBi masked attention; optional scale; double accumulation | float32; no dropout; zero denominator yields zero |
| layout operators (`reshape`, `transpose`, `slice`, `flatten`, `concat`, `split_qkv`, `merge_heads`, `rope`, `repeat_kv`, `pos_embedding`, `identity`, `dropout`, `select`) | NumPy byte movement, visibly reported as layout | no arithmetic claim |

The native executor covers the generated MLP and transformer-decoder families end
to end. Other NIR operators (conv2d, pool2d, rnn/lstm/gru, ssm, gcn, moe,
quantized_linear, sampling) raise `UnsupportedOperator` with the operator and node
id. There is no silent NumPy fallback in the native path.

### Build and use

```bash
# Build into the per-user cache (~/.cache/uzen/native), or pass --output.
python -m uzen.runtimes.native.build

# Force a rebuild after changing kernels.cpp or the compiler.
python -m uzen.runtimes.native.build --force

# Environment:
#   CXX                 explicit compiler (default: g++, clang++, c++)
#   UZEN_NATIVE_CACHE   cache directory
#   UZEN_NATIVE_LIBRARY explicit prebuilt shared library
```

```python
from uzen.runtimes.native import NativeRuntime

runtime = NativeRuntime()              # builds on first use if needed
result = runtime.execute(graph, inputs, package)
print(runtime.describe()["kernel_library"]["kernels"])
```

The C ABI exports `uzen_linear_f32`, `uzen_matmul_f32`, `uzen_layernorm_f32`,
`uzen_rmsnorm_f32`, `uzen_softmax_f32`, the unary/binary/scale/clip kernels,
`uzen_embedding_f32`, `uzen_reduce_last_f32` and `uzen_attention_f32`. Every
wrapper validates rank, shape and dtype in Python before calling C; a non-zero C
status becomes `uzen.errors.NativeKernelError` (`runtime.native_kernel`).

### Verification

```bash
.venv/bin/python -m pytest -o addopts="" -q tests/test_native_runtime.py
.venv/bin/python examples/native_backend_demo.py --root /tmp/uzen-native-demo
```

The 16 focused tests cover the build, ctypes bindings, every kernel family,
attention masks, MLP and transformer parity against the NumPy reference, capture
parity, float64 and quantized refusal, and capability reporting. The demo builds
the library, compares a generated MLP and a two-layer transformer decoder against
NumPy, checks GGUF round-trip execution, measures latency, and verifies that
`quantized_linear` is refused.

Verified in this workspace (g++ 13.3, x86_64, Linux):

```text
16 passed
mlp max_abs_error=5.588e-09
transformer max_abs_error=1.118e-08
attention max_abs_error: causal=0.0, bidirectional_window=0.0, alibi=0.0
gguf round-trip max_abs_error=1.118e-08
matmul 256^3 native=0.44 ms, numpy=0.09 ms
attention native=0.03 ms, numpy=0.07 ms  (small 2x4x6x9x16 case)
quantized_linear refused=True
gate: PASS
```

The latency numbers are single-process wall-clock measurements on this machine.
They are **not** a speedup claim: the naive native GEMM is slower than NumPy's BLAS
for this shape, while the small attention kernel is faster. N2's blocked GEMM,
fused plans and prepacking target the matmul gap.

### Honest limitations

1. **float32 only.** The ABI and executor reject float64, float16 and bfloat16
   arithmetic; no silent upcast/downcast.
2. **The direct executor has no graph compiler.** Nodes call C functions one by one
   and layout operators use NumPy. Fusion, prepacking and the buffer planner live
   in the N2 plan path below.
3. **The direct path's GEMM is naive.** Correctness-first triple loop with OpenMP;
   it is not competitive with OpenBLAS at medium/large sizes. The plan path uses
   the prepacked blocked GEMM instead.
4. **Attention is correct but not optimized.** Scores are recomputed for the
   softmax pass; there is no flash-attention tiling, KV cache or fused kernel.
5. **The direct path's coverage is partial.** Conv, recurrent, SSM, GNN, MoE,
   sampling and int8 `quantized_linear` are unsupported there. The N2 plan path
   implements all of them except the two seeded sampling operators.
6. **OpenMP is optional.** The build retries without `-fopenmp` if the compiler
   rejects it; the capability report does not claim a thread count.
7. **No CUDA.** `capabilities()["extra"]["cuda"]` is always `False`. A GPU claim
   requires real kernels and measurement, which do not exist yet.
8. **No mobile/Windows validation.** The library is built and tested on Linux
   x86_64; macOS uses `.dylib` naming, but was not exercised here.

## N2 — implemented native plan compiler

`uzen.runtimes.native.compiler.compile_native_plan` lowers a validated NIR graph
into a linear plan of native steps over explicitly planned buffers;
`plan_executor.execute_plan` runs it. The N1 direct executor is unchanged and
remains available.

### Pipeline

| Stage | Implementation |
| --- | --- |
| Lowering | each node becomes one plan step over the compiled kernel set; `split_qkv` writes all three head tensors from one step |
| Fusion — linear | `linear` + bias + the single activation that consumes it become one `gemm_packed` call with the activation code compiled in |
| Fusion — residual | an `add` feeding a `layernorm`/`rmsnorm` becomes one `add_norm` pass; the sum is still stored when another consumer or a graph output needs it |
| Prepacking | linear weights are packed once per package fingerprint into 64x64 (k x n) panels, zero-padded to whole blocks |
| Memory planning | buffers are allocated by liveness; a freed slot is reused only for an identical element count, and a target is allocated *before* the step's dying inputs are released, so no kernel can read the buffer it writes |
| Shape polymorphism | a plan is compiled once and bound per input signature; symbolic dims — including dims an operator left unknown, such as a symbolic `slice` — are resolved at bind time |
| Reporting | `plan.describe()` and `NativePlanExecution.summary()` expose steps, fusions, operators, prepacked weights, peak bytes, slot reuse and capturable intermediates |

### Kernel coverage

| Plan operator | Implementation | Limits |
| --- | --- | --- |
| `gemm_packed` | prepacked blocked GEMM, optional bias and fused activation (none/relu/gelu/gelu_exact/silu/tanh/sigmoid) | float32; any batch rank flattened to rows |
| `add_norm` | residual add + layernorm/rmsnorm in one pass, optional residual write | operands must have identical shapes (the compiler only fuses then) |
| `conv2d` | direct NCHW convolution with stride/padding | no dilation or groups |
| `maxpool2d` | NCHW max pooling; padding contributes -inf like the NumPy reference | max only (the NIR operator is max pooling) |
| `recurrent` | tanh RNN, LSTM (i,f,g,o) and GRU (r,z,n) with per-batch state | float32 in/out, double state |
| `ssm` | diagonal S4D-style recurrence | float32 |
| `gcn` | symmetric degree-normalized adjacency, then projection | dense adjacency |
| `moe` | top-k routed experts, deterministic selection, gate renormalization | unused experts are never evaluated |
| `quantized_linear` | int8 weights with per-output-channel scales, dequantized inside the accumulation | int8 weights only |
| `reduce` | mean/sum over any single axis | one axis per node |
| `attention` | single-pass online-softmax attention; the score matrix is never materialized | float32; no dropout |
| `layout` | NumPy byte movement from the direct executor, reported separately | no arithmetic claim |

Operators outside this set — the seeded `sample_normal`/`add_noise` and multi-axis
reductions — are refused at compile time with the node id. There is no fallback.

### Build and use

```bash
# Both translation units are compiled into one cached shared library.
python -m uzen.runtimes.native.build --force
python examples/native_compiler_demo.py --root /tmp/uzen-compiler-demo
```

```python
from uzen.runtimes.native import NativeRuntime

runtime = NativeRuntime()
plan = runtime.compile(graph, package)              # fusions + prepacking + memory plan
result = runtime.execute_compiled_graph(plan, inputs, package)
print(result.summary())                             # steps, fusions, peak bytes
```

### Verification

```bash
.venv/bin/python -m pytest -o addopts="" -q tests/test_native_runtime.py tests/test_native_compiler.py
.venv/bin/python examples/native_compiler_demo.py --root /tmp/uzen-compiler-demo
```

`41 passed` (16 direct-executor, 25 compiler). The compiler tests check every
prepared kernel against its NumPy reference, fusion detection, shape polymorphism,
slot reuse against the unfused path, all seven supported families, int8 plans,
capture/fusion interaction and typed refusal. Demo output on this machine
(g++ 13.3, x86_64):

```text
mlp                  nodes=  5 steps=  4 fused= 1 max_abs_error=5.588e-09
transformer_decoder  nodes= 28 steps= 22 fused= 6 max_abs_error=1.118e-08
cnn                  nodes=  8 steps=  8          max_abs_error=1.863e-08
recurrent (lstm)     nodes=  5 steps=  5          max_abs_error=1.185e-09
moe                  nodes=  4 steps=  4          max_abs_error=2.330e-10
state_space          nodes=  4 steps=  4          max_abs_error=1.490e-08
gnn                  nodes=  6 steps=  6          max_abs_error=1.863e-09
memory (transformer) 29 buffers in 7 slots, peak 112 KiB of 160 KiB allocated, 22 reused
prepack padding overhead: 1.00x on every weight
int8 plan: 9 rewritten nodes, max_abs_error=7.451e-09
latency, batch=4 seq=16 d_model=64 2 layers (median of 5):
  numpy 3.578 ms | direct native 3.847 ms | compiled plan 2.432 ms
gate: PASS
```

The latency row is measured, not predicted: on this machine the compiled plan took
0.63x the direct executor's time and 0.68x NumPy's for that shape, from fusing the
residual/norm and activation steps and from contiguous prepacked GEMM panels. It is
still a CPU float32 backend and makes no GPU claim.

### Honest limitations

1. **float32 only**, with the same typed refusals as N1.
2. **Seeded stochastic operators are not planned.** `sample_normal` and `add_noise`
   would need a bit-exact NumPy PCG64 stream; the plan refuses them rather than
   changing the numbers. Use the NumPy executor for VAE/diffusion graphs.
3. **Fusion is intentionally narrow**: linear+bias+activation and residual+norm.
   No conv+activation, attention+rope or multi-op elementwise chains yet.
4. **The default prepacked weight is dense float32.** Alternative layouts exist in
   N3 below and are selected per weight at compile time.
5. **One reduction axis per node**, and reductions do not fuse into consumers.
6. **No thread-pool ownership.** OpenMP remains optional; the plan does not pick a
   thread count.
7. **No CUDA.** `capabilities()["extra"]["cuda"]` is still always `False`.

## N3 — implemented advanced parameter representations

N3 changes how a linear weight is stored and which kernel reads it; the NIR graph
language is untouched. `compile_native_plan(..., representations={weight: kind})`
records the request on the plan, packs that weight once, and the plan executor
dispatches to the kernel for its layout. Unlisted weights keep the exact blocked
float32 layout.

| Kind | Native storage | Kernel | Lossy |
| --- | --- | --- | --- |
| `dense_blocked` | 64x64 float32 panels (N2 prepack) | `uzen_gemm_packed_f32` | no |
| `transposed` | float32 (in, out) contiguous, no padding | `uzen_gemm_transposed_f32` | no |
| `int8_block` | int8 panels + one float32 scale per (row, 64-value group) | `uzen_gemm_int8_block_f32` | weights |
| `int4_block` | two's-complement nibbles, two per byte + per-group scales | `uzen_gemm_int4_block_f32` | weights |
| `low_rank` | float32 factors u (out, r), v (r, in) from a float64 SVD | `uzen_gemm_low_rank_f32` | yes |
| `sparse_csr` | float32 values + int32 indices, CSR rows over outputs | `uzen_sparse_gemm_f32` | unless threshold is 0 |

Quantization is symmetric and group-wise (`[-127, 127]` for int8, `[-7, 7]` for
int4, zero point 0), with the group equal to the GEMM's 64-value reduction panel, so
a panel is exactly one quantized group of one output block. Dequantization happens
inside the accumulation; a float weight matrix is never materialized.

### Provenance

Every prepared weight carries a `RepresentationReport`: layout kind, shape, source
dtype, packed bytes, bits per weight, compression ratio against the source tensor,
max/mean/max-relative reconstruction error, group and scale statistics (count, min,
max, SHA-256), saturation count, rank and captured spectral energy for `low_rank`,
nonzero count and density for `sparse_csr`, and a SHA-256 digest over the packed
buffers. The plan exposes `describe()["representations"]` and
`representation_summary()`; each execution summary embeds the same reports.

```python
from uzen.runtimes.native import NativeRuntime

runtime = NativeRuntime()
plan = runtime.compile(
    graph,
    package,
    representations={
        "layers.0.qkv.weight": "int8_block",
        "layers.0.mlp_up.weight": "int4_block",
        "layers.1.attn_out.weight": {"kind": "low_rank", "rank": 32},
        "layers.1.mlp_down.weight": {"kind": "sparse_csr", "threshold": 0.0},
    },
)
result = runtime.execute_compiled_graph(plan, inputs, package)
print(result.summary()["representations"])
```

### Build and verify

```bash
python -m uzen.runtimes.native.build --force
.venv/bin/python -m pytest -o addopts="" -q tests/test_native_representations.py
.venv/bin/python examples/native_representation_demo.py --root /tmp/uzen-representation-demo
```

`25 passed` for the representation suite, plus the unchanged N1/N2 suites. Tests
compare each kernel against the matrix its packed buffers reconstruct through an
independent NumPy unpacking path, so a packing/indexing error cannot cancel out;
they also cover group-scale placement, signed nibble round-trip, full-rank SVD
exactness, CSR structure validation, request normalization and per-weight caching.

Demo output on this machine (g++ 13.3, x86_64), weight 70x130:

```text
layout parity and cost (weight 70x130, structured random):
  dense_blocked  bits/weight= 86.421 bytes= 98304 reconstruction=0.000e+00 kernel=0.000e+00
  transposed     bits/weight= 32.000 bytes= 36400 reconstruction=0.000e+00 kernel=0.000e+00
  int8_block     bits/weight= 22.956 bytes= 26112 reconstruction=1.444e-02 kernel=1.907e-06
  int4_block     bits/weight= 12.153 bytes= 13824 reconstruction=2.634e-01 kernel=1.907e-06
  low_rank       bits/weight= 14.066 bytes= 16000 reconstruction=2.438e+00 kernel=2.861e-06
  sparse_csr     bits/weight=  2.810 bytes=  3196 reconstruction=0.000e+00 kernel=0.000e+00
representations on whole compiled plans (max abs error vs float32 NumPy):
  mlp                  weights= 3 dense=5.59e-09 transposed=5.59e-09 int8=2.38e-04 int4=2.71e-03
  transformer_decoder  weights= 9 dense=1.12e-08 transposed=1.12e-08 int8=3.51e-04 int4=6.66e-03
  cnn                  weights= 1 dense=1.86e-08 transposed=1.86e-08 int8=1.99e-05 int4=3.36e-04
  recurrent            weights= 1 dense=1.18e-09 transposed=1.18e-09 int8=1.36e-05 int4=5.17e-04
mixed plan (transformer decoder): 9 weights, max_abs_error=5.679e-04, stored=103424 bytes
latency (median of 5, batch=4 seq=16 d_model=64 2 layers):
  dense_blocked   2.629 ms  stored= 294912 bytes  error=2.24e-08
  int8_block      3.254 ms  stored=  78336 bytes  error=4.78e-04
  int4_block      2.866 ms  stored=  41472 bytes  error=8.64e-03
gate: PASS
```

The storage column is the measured sum of packed buffers, not a projection: int4
stores 7.1x less than the padded float32 plan and int8 3.8x less. The latency column
is also measured and **is not a speedup claim** — both quantized paths were a little
slower than float32 on this run because these kernels dequantize with scalar
accumulation rather than SIMD int8 dot products, and the medians move by a few
percent between runs. The win is storage and explicit error control.

The 70x130 example is deliberately not block-aligned: 2 output blocks x 3 reduction
blocks cost 2.7x padding, which is why `dense_blocked` reports 86 bits/weight there.
On block-aligned shapes the same layout reports 32 bits/weight, and the report always
shows the real number.

### Honest limitations

1. **Group size is fixed at 64 values**, tied to the GEMM reduction panel. Arbitrary
   group sizes (32/128) are refused rather than silently rounded.
2. **No SIMD integer dot products.** int8/int4 kernels decode and accumulate in
   scalar double precision; throughput is not better than float32, only storage is.
3. **int4 is symmetric `[-7, 7]` with zero point 0.** No asymmetric int4 or per-token
   (activation) quantization; activations stay float32.
4. **Sparse CSR is unstructured magnitude pruning** with float32 values and int32
   indices: 8 bytes per nonzero, so it only beats float32 below 50% density and int4
   below about 6%. There is no 2:4 structured sparsity, no sparse tensor core claim,
   and pruning is a one-shot transform, never training.
5. **`low_rank` is a one-shot float64 SVD.** On randomly initialized weights it
   captures only about `rank / min(out, in)` of the spectral energy, so its reported
   error is large by construction; it is a representation, not a compression pipeline
   or a fine-tuning method.
6. **Only `linear` weights can change layout.** Conv, embedding, recurrent, SSM, GCN
   and MoE parameters keep their N2 storage.
7. **Padding is part of the cost.** Block layouts pad to whole 64x64 panels; the
   report's `bits_per_weight` includes that padding instead of quoting a theoretical
   4 or 8 bits.
8. **Still float32 activations and still CPU.** No CUDA, no mixed-precision
   hardware path, no quantized attention.

## N4 — implemented distributed scheduling

N4 keeps the NIR/graph work untouched and hardens the *control plane* around it: which
worker may take a job, for how long, what happens when it dies, and what stops a second
report from changing an outcome. Workers still pull; nothing is pushed at a worker that
did not ask.

### Guarantees and the mechanism behind each

| Guarantee | Mechanism |
| --- | --- |
| A job only goes to a worker that can run it | requirements are derived from the request (`requirements_from_request`) into `job_requirements` rows plus numeric budgets; the worker's advertised document becomes `worker_capabilities` rows through the same `capability_index`, so the SQL predicate and the Python explanation cannot disagree |
| Two workers can never hold one job | the eligibility predicate is inside the claiming `UPDATE` … `WHERE id = (SELECT …)`, a single SQLite statement; concurrent control-plane processes are serialized by the writer lock, and the claim installs `lease_expires_at` in the same statement |
| A dead worker's job comes back | `reap_expired_leases` returns expired leases to `QUEUED` (incrementing nothing but re-using the attempt counter), or fails the job with `job.lease_expired` once attempts are exhausted; the reaper runs inside every claim, so a polling fleet heals itself |
| A superseded worker cannot publish | every claim mints a `dispatch_token`; renew and report must present it, and the report compares it against the *current* token, so a zombie worker is answered `stale` instead of overwriting the replacement |
| A repeated report changes nothing | `dispatch_records(job_id, attempt)` is unique and flips from `dispatched` to a terminal outcome exactly once; later reports answer `duplicate` with the recorded outcome |
| Re-execution does not duplicate artifacts | the worker's local job keeps the control-plane job id and returns the stored result for an already-terminal job instead of running it again |
| Hungry jobs are visible | `JobScheduler.unroutable_jobs()` reports queued jobs no registered worker satisfies, with the exact capability each worker lacked |
| Urgent work goes first without starving the rest | `jobs.priority` is an operator bias (clamped to ±1000) inside the claim's `ORDER BY`; optional `aging_seconds` adds one effective priority point per aging interval waited, so an old low-priority job eventually overtakes a stream of fresh ones |
| A failed attempt is not retried in a hot loop | the reaper stores `available_at = now + min(cap, base * 2^(attempt-1))`; the claim predicate refuses the row until that instant, and the delay is recorded on the lease-expiry event |
| An operator can take a job back | `JobScheduler.preempt` clears the lease and token, returns the row to `QUEUED` at once, and records a `preempted` event; the old worker's next renew or report is refused as `stale` |
| A worker can leave the fleet without killing work | a `draining` worker is excluded inside the claiming predicate but keeps renewing its current lease; `resume_worker` puts it back. A routine heartbeat only touches `last_seen`, so it cannot silently un-drain a worker |
| Idle polling is bounded | `claim(wait_seconds=...)` long-polls the same atomic claim and returns as soon as a job appears; the REST claim and the worker daemon expose the same window |
| A skewed worker clock cannot move a Redis lease | the Redis adapter reads the server's `TIME` for claim, renew, reap and preempt instead of trusting the caller's clock |

Job states gained one transition: `DISPATCHED → QUEUED` (and `RUNNING → QUEUED`) for lease
expiry and preemption, and `DISPATCHED → VALIDATING` for a worker that reports success
without an intermediate poll. A worker never chooses its own next state.

### Durable queue

`JobScheduler` talks to the job database, and the database is the queue: WAL SQLite with
a single atomic claim statement. That is a real multi-process deployment on one host (or
one host with the database on local disk), which is what this build verifies.

`uzen/infrastructure/storage/redis_queue.py` is the multi-node adapter: a RESP2 client
with no third-party dependency, and Lua scripts that keep claim, renew, report and reap
atomic on the server (`uzen:q:ready`, `uzen:q:job:<id>`, `uzen:q:caps:<worker>`,
`uzen:q:lease` sorted set, `uzen:q:reports`). It mirrors the SQLite semantics — same
token, attempt and duplicate-report rules — plus priority selection, backoff, draining,
preemption and the server-side clock.

> **Status:** the Redis adapter is exercised against a **live Redis 7.4.2 server** in
> this workspace (built from source; the integration test is gated by `UZEN_REDIS_URL`).
> It is not yet exercised across physical hosts, network partitions or a failover pair,
> so the Redis topology itself remains a deployment concern. Do not run a multi-node
> fleet on SQLite over NFS: WAL needs shared memory.

### Code

```python
from uzen.jobs.scheduler import JobScheduler
from uzen.jobs.service import JobService

service = JobService("./uzen-plane")
scheduler = JobScheduler(
    service.db,
    lease_seconds=120,
    max_attempts=3,
    retry_backoff_seconds=5,
    retry_backoff_cap_seconds=60,
    aging_seconds=30,
    max_in_flight_per_worker=2,
)

# control plane
job = service.submit({"task_scope": "weight_modification", ..., "priority": 10})

# worker
claimed = scheduler.claim(
    "fleet-a", capabilities=my_capabilities, wait_seconds=15.0
)
token = claimed["lease"]["token"]
executed = service.run(claimed["id"], worker_id="fleet-a", token=token)

# operator
scheduler.drain_worker("fleet-a")
scheduler.preempt(job["id"], reason="maintenance")
scheduler.resume_worker("fleet-a")
```

`service.run(job_id, worker_id=..., token=...)` settles the job through the same
dispatch record a remote worker uses; without a token it is the local execution path.

REST: `POST /v1/workers/{register,heartbeat,claim,renew,report}`,
`POST /v1/workers/{id}/{drain,resume}`, `GET /v1/scheduler`, `POST /v1/scheduler/reap`,
`POST /v1/jobs/{id}/preempt`, `GET /v1/jobs/{id}/dispatch`,
`GET /v1/jobs/{id}/eligibility`. `uzen serve --no-auto-run` keeps submitted jobs `QUEUED`
for workers instead of running them inside the API process. `uzen serve` also accepts
`--retry-backoff-seconds`, `--aging-seconds` and `--max-in-flight-per-worker`;
`uzen worker run --wait-seconds` enables long polling.

### Verification

```bash
.venv/bin/python -m pytest -o addopts="" -q tests/test_distributed_scheduling.py
.venv/bin/python examples/distributed_scheduling_demo.py --root /tmp/uzen-distributed-demo
```

`40 passed` (with `UZEN_REDIS_URL` pointing at a live server; 39 otherwise). The suite
covers the capability model, capability-aware and budget-aware claims, two database
connections racing over 20 jobs (every job claimed exactly once), token-guarded
renewals, expiry requeue with exponential backoff and its cap, priority ordering,
priority aging, preemption and the stale report that follows it, draining and resuming,
the per-worker in-flight cap, long-poll claim, attempt exhaustion into
`job.lease_expired`, cancellation during a lease, duplicate/stale/rejected reports, the
unroutable report, the REST claim → execute → report loop, the daemon's idempotent
re-execution and artifact refusal, exchange-cache pruning, the RESP client against a
scripted socket, and the full priority/backoff/preempt/drain flow against live Redis.

Demo output on this machine (4 worker processes, 2 s leases, live Redis configured):

```text
control plane: /tmp/uzen-distributed-demo/plane
workers: fleet-a-mlp, fleet-b-general, fleet-c-cnn, fleet-d-doomed
crashed job 24011ce9… claimed by fleet-d-doomed, never reported
  recovered on attempt 2 by fleet-a-mlp → SUCCEEDED
  replayed report: duplicate
scale_mlp        SUCCEEDED  workers=['fleet-a-mlp'] artifacts=1
cnn_synthesis    SUCCEEDED  workers=['fleet-c-cnn', 'fleet-c-cnn'] artifacts=1
big_cpu_scale    SUCCEEDED  workers=['fleet-b-general', 'fleet-b-general'] artifacts=1
needs_cuda       QUEUED     workers=[] artifacts=0
unroutable jobs: ['7abdff04']
  reason: ["worker does not advertise device ['cuda']"]
policy: priority=ok, preempt=preempted, stale_report=stale, drain_claim=none, too_early=none
redis adapter: exercised
gate: PASS
```

Thirteen gates pass: crash recovery on a later attempt, exactly one artifact for the
recovered job, duplicate report refused, every routable job succeeded, `cnn` routing,
12-CPU routing, unroutable detection, no job with two artifacts, unique outputs per job,
priority ordering, preemption with a stale old token, drain/resume, and the backoff
delay. The demo runs real worker **processes**; the killed one calls `os._exit(9)` after
taking the lease. The `cnn` and 12-CPU jobs show two dispatch records because their
first worker's 2 s lease expired mid-run and the job was retried; each still produced
exactly one artifact.

### Honest limitations

1. **One machine, not one network.** SQLite is verified across processes on one host.
   The Redis adapter is verified against a live server on localhost, but not across
   physical hosts, partitions or failover, and no PostgreSQL adapter exists.
2. **Capabilities are self-advertised.** A registered worker decides what it claims; the
   control plane trusts that document. There is no hardware attestation, and an
   overclaiming worker produces a failed job, not a routed-around one.
3. **Fairness is bounded, not weighted.** Priority plus optional aging orders work and a
   per-worker in-flight cap stops one worker hoarding claims, but there is no
   tenant-weighted fair queue, quota or reservation.
4. **Preemption is cooperative.** Revoking the token and requeueing is immediate, but a
   worker already inside a kernel only stops at its next renew/report; no kernel is
   interrupted.
5. **Pull, now with long polling.** There is still no server push; `wait_seconds`
   shortens the idle latency without a persistent connection per worker.
6. **SQLite leases are same-host wall clocks.** The Redis path uses the server's clock,
   but a SQLite database must not be shared across hosts (WAL needs shared memory), so
   cross-host clock skew does not apply there by construction.
7. **Cache eviction is explicit.** `WorkerDaemon.prune_exchange(..., dry_run=True)`
   removes old exchange entries on request and keeps referenced ids, but there is no
   background GC or object-store lifecycle policy.
8. **Artifact transfer is per-transport.** Object-store transport is used when injected
   explicitly, otherwise the shared exchange directory.
9. **Native layouts are advertised only when a built kernel library resolves** — a
   worker without one never claims representation work.

## N5 — implemented CUDA device boundary

N5 is the only phase allowed to claim GPU execution. It claims it only when kernels
were compiled, loaded through `libcuda` and launched on a device. The authoring host
has no driver or device, so the honest summary is: **17 real kernels, compiled and
PTX-assembled; device execution not verified here; no speed claim.**

See [N5_CUDA.md](N5_CUDA.md) for the full design, kernel table and limitations.

### What is implemented

* `uzen/runtimes/cuda/kernels.cu`: float32 linear/matmul, layernorm, rmsnorm,
  softmax, GELU (tanh/exact), SiLU, ReLU, tanh, sigmoid, same-shape add/sub/mul,
  scale, clip, embedding, last-axis mean/sum and masked attention with the CPU
  runtime's causal offset, symmetric window and ALiBi semantics.
* `uzen/runtimes/cuda/nvrtc.py`: compiles the checked-in source with NVRTC when no
  toolkit exists, or with `nvcc` when it does. The cache key is the source digest
  plus architecture; the report names the compiler and says whether it was cached.
* `uzen/runtimes/cuda/driver.py`: ctypes binding to the CUDA driver API
  (`cuInit`, device attributes, context, module load, function lookup, allocation,
  copies, launch, synchronize), with typed driver errors.
* `uzen/runtimes/cuda/allocator.py`: size-class pool, no double free, live/peak/
  reserved bytes and H2D/D2H accounting.
* `uzen/runtimes/cuda/executor.py`: NIR dispatch, resident weights keyed by package
  fingerprint, lifetime-based activation release, host-side layout operators that are
  reported separately, and output/capture download with finite checks.
* `uzen/runtimes/cuda/runtime.py`: capabilities, `execution_verified` only after a
  real run, and an explicit fallback policy — `fallback="error"` refuses without a
  device, `fallback="cpu"` runs NumPy and labels the result `numpy-fallback`.

### Build and verify

```bash
python -m uzen.runtimes.cuda.build --json
python -m pytest -o addopts="" -q tests/test_cuda_runtime.py
python examples/cuda_device_demo.py --root /tmp/uzen-cuda-demo
```

On this host the suite reports **11 passed, 4 skipped** and the demo ends
`RESULT: SKIPPED (no CUDA device; kernels compiled to PTX only)`. The four skips are
the GPU parity tests (MLP, transformer with attention capture, weight residency,
attention mask semantics), not fakes. The compile tests run and `ptxas` assembles the
PTX, so kernel syntax and PTX validity are checked without a device.

### Honest limitations

1. No GPU execution, timing or memory measurement is claimed from this host.
2. CUDA only; there is no HIP/ROCm backend.
3. float32 only; no tensor cores, half precision or TF32.
4. Naive GEMM and attention kernels; no cuBLAS/FlashAttention comparison is implied.
5. Elementwise `add/sub/mul` require identical shapes; broadcasting is refused.
6. Layout operators round-trip through host memory and are reported as layout work.
7. One context, one device, default stream; no CUDA graphs or multi-GPU.
8. Dynamic shared-memory scores cap attention near 12k keys on older architectures.
9. `quantized_linear` and N3 int4/int8/sparse layouts are refused on this path.
10. Toolkit-free compilation needs the NVRTC wheel; execution still needs the vendor
    driver. A driver older than the PTX target fails with `runtime.cuda_kernel`.

## N6 — implemented campaign engine

N6 builds research campaigns on the deterministic engines. Candidate generation is
seeded and autonomous, but every candidate still passes the same validation and
inference path; no candidate is trained or optimized by gradients.

* `uzen/core/campaign/spec.py` defines immutable `CampaignSpec`, `Axis` and
  `Objective` records. Architecture axes must name real family-spec fields; weight
  axes are restricted to `scale`, `sparsity` and `quantize`.
* `uzen/core/campaign/space.py` implements deterministic grid, random and evolutionary
  proposal generation. Completed candidate digests are excluded and proposals are a
  pure function of the seed, round and ranked trials.
* `uzen/core/campaign/evaluate.py` builds the architecture or transformed package,
  executes a real NumPy forward pass and measures parameters, bytes, node count,
  output RMS/diversity, sparsity, int8 round-trip error and (for weight campaigns)
  distance/activation agreement from a base package.
* `uzen/core/campaign/store.py` persists campaigns, trials and events in SQLite. Trial
  keys are `(campaign_id, candidate_digest)`, so a retry is idempotent. `max_rounds`
  produces a paused campaign that can be resumed at the next batch boundary.
* `uzen/core/campaign/corpus.py` writes successful trials to content-addressed JSON
  records and rebuilds/re-executes them for verification. Timestamps and latency are
  excluded from the record digest; environment versions are stored separately.
* `uzen/core/campaign/engine.py` schedules batches (bounded thread pool), applies
  constraints, scores objectives, computes the Pareto front, early-stops on patience
  and persists every outcome, including `invalid`, `constraint` and `error` trials.

The CLI exposes `campaign run|show|list|events|verify|replay`; the SDK exposes
`run_campaign`, `campaign`, `campaigns`, `campaign_events`,
`verify_campaign_corpus` and `replay_benchmark`; REST exposes `/v1/campaigns` and
`/v1/benchmarks/{digest}`; MCP exposes five campaign tools. See
[N6_CAMPAIGNS.md](N6_CAMPAIGNS.md) for the full schema, guarantees and limitations.

### N6 honest limits

1. The scheduler is in-process; there is no campaign distribution over N4 workers.
2. Proxy metrics are not task quality. No dataset, loss or accuracy score exists.
3. Resume is batch-granular: a crash inside a batch does not preserve its unwritten
   trials, though completed trials are never duplicated.
4. Verification tolerates low-bit BLAS differences within `1e-9` relative; it never
   claims bitwise identity across machines.
5. Weight campaigns cover scale, magnitude sparsity and symmetric int8 round-trip
   simulation; they do not perform learned pruning or quantization-aware optimization.
6. The REST run endpoint is synchronous and is not a background campaign daemon.

## N7 — distributed campaign trials (post-P6 extension)

N7 keeps N6's planners, scoring, SQLite campaign store and corpus unchanged and moves
only trial evaluation onto the N4 queue. A trial is a read-only `inspection` job with
`options.task_kind=campaign_trial`; it carries the campaign spec and candidate, is
validated by recomputing the candidate digest, and returns metrics without creating an
artifact. `DistributedTrialExecutor` submits every proposal with a stable idempotency
key, waits for workers, and records the same `Trial` rows a local run records.

Workers use the normal atomic claim and token-guarded `JobService.run` path. A timeout
pauses the campaign with `stop_reason=worker_timeout`; resuming resubmits the same
idempotency keys and reuses jobs that settled in the meantime. The coordinator remains
single-host SQLite: this is distributed execution, not a distributed campaign database.
See [N7_DISTRIBUTED_CAMPAIGNS.md](N7_DISTRIBUTED_CAMPAIGNS.md) for the job schema,
worker loop, demo gates and limits.

## N8 — campaign control and isolation (post-P6 extension)

N8 makes a campaign run itself an N4 job (`task_kind=campaign_run`) so the REST request
can return a durable coordinator job instead of blocking. `CampaignEngine.cancel` sets a
durable cancellation flag, cancels the linked coordinator and trial jobs, and
`CampaignStore.record_trial` serializes the cancel check with the insert so no trial is
appended after cancellation returns. A late worker report is settled as `CANCELLED`
instead of publishing a result. Job rows now carry `job_kind`; the claim query accepts a
kind filter, `CampaignWorker` defaults to `campaign_trial`, and a coordinator worker must
opt into `campaign_run`.

```bash
.venv/bin/python -m pytest -o addopts="" -q tests/test_campaign_control.py
.venv/bin/python examples/campaign_control_demo.py --root /tmp/uzen-n8-demo
```

Measured in this workspace: 10 N8-focused tests pass, the 9 N7 tests stay green, the
real-process demo has 10/10 gates PASS, and the full suite is 553 passed / 37 skipped
(37 skips are the documented opt-in/hardware gates plus 4 CUDA compilation skips because
NVRTC is not installed in this restored environment). See
[N8_CAMPAIGN_CONTROL.md](N8_CAMPAIGN_CONTROL.md) for the job contract, cancellation
barrier, interfaces and limits.

## N9 — campaign run durability (post-P6 extension)

N9 closes the duplicate-admission and long-lease gaps documented by N8. A background
submission now reserves the campaign's coordinator slot atomically with
`CampaignStore.reserve_run` under `BEGIN IMMEDIATE`, then creates the job under that
reserved id; concurrent submitters return the same job instead of each creating one.
`JobService.run(..., lease_seconds=N)` renews a leased job every `N/3` seconds with the
dispatch token, and `CampaignWorker` passes its lease into that path. A campaign that
runs longer than its initial lease therefore stays on attempt 1 instead of being reaped
and executed twice.

```bash
.venv/bin/python -m pytest -o addopts="" -q tests/test_campaign_durability.py
.venv/bin/python examples/campaign_durability_demo.py --root /tmp/uzen-n9-demo
```

Measured in this workspace: 4 focused N9 tests pass, the 10 N8 and 9 N7 tests stay
green, the full suite is 557 passed / 37 skipped, and the four-process demo reports 9/9
gates PASS with one admitted job, 64 trials, and a ~0.4-second run under a 0.15-second
lease. Admission remains single-host SQLite and cancellation remains cooperative; see
[N9_CAMPAIGN_DURABILITY.md](N9_CAMPAIGN_DURABILITY.md) for the protocol and limits.
