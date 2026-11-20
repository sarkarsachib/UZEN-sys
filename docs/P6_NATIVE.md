# P6 — Native computational substrate

This document records UZEN's own phase plan for the master roadmap's **Phase 6:
Advanced computational substrate**, and exactly what is implemented. The master
specification lists Rust integration, a native compiler, C++/CUDA kernels,
additional runtime backends, advanced parameter representations and large-scale
autonomous research. Those are too large to pretend to finish in one step, so
they are split into six verifiable phases.

No phase below is claimed because a directory or interface name exists. A phase is
`implemented` only when its kernels/engine execute real computation and the
listed gate passes.

## Phase plan

| Phase | Deliverable | Status |
| --- | --- | --- |
| **N1** | Native CPU arithmetic: C++17 kernels, ctypes ABI, NIR executor, native runtime adapter, parity tests, working demo | **implemented** |
| **N2** | Native graph compiler: lower NIR to a native execution plan, fuse linear+activation and residual+norm, prepack weights, add conv/pool/RNN/SSM/GCN/MoE/int8 kernels, single-pass attention, liveness memory planner | **implemented** |
| **N3** | Advanced parameter representations: block-wise int4/int8 with scales, low-rank/sparse native layouts, packed transposed weights, representation-aware execution | **implemented** |
| **N4** | Distributed execution: durable queue adapter (Redis/Valkey or PostgreSQL), capability-aware native worker scheduling, leases, idempotent dispatch, remote artifact fetch | planned; local SQLite leases and S3-compatible transport already exist |
| **N5** | GPU/native boundary: real CUDA/HIP kernels, device allocator, transfer-aware dispatch, CPU fallback policy that is explicit rather than silent | planned; **no GPU acceleration is claimed now** |
| **N6** | Large-scale autonomous research: native candidate-evaluation batches, experiment scheduler, architecture/parameter search campaigns, reproducible benchmark corpus | planned |

N4 is the next phase. N4–N6 build on N2's plan and N3's layouts and are not started.

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

## N4–N6 boundaries

* **N4** adds distributed scheduling around the existing job/worker contracts; it
  must not weaken lease/idempotency semantics.
* **N5** is the only phase allowed to claim GPU execution, and only with real
  kernels, device transfer tests and measurements.
* **N6** builds research campaigns on the deterministic engines; candidate
  generation may be autonomous, but every candidate still passes the same
  STRICT_PLUS, validation and provenance path.
