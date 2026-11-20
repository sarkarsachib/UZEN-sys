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
| **N2** | Native graph compiler: lower NIR to a native execution plan, fuse elementwise chains, prepack weights, add conv/RNN/SSM/MoE/quantized kernels, memory planner | planned |
| **N3** | Advanced parameter representations: block-wise int4/int8 with scales, low-rank/sparse native layouts, packed transposed weights, representation-aware execution | planned |
| **N4** | Distributed execution: durable queue adapter (Redis/Valkey or PostgreSQL), capability-aware native worker scheduling, leases, idempotent dispatch, remote artifact fetch | planned; local SQLite leases and S3-compatible transport already exist |
| **N5** | GPU/native boundary: real CUDA/HIP kernels, device allocator, transfer-aware dispatch, CPU fallback policy that is explicit rather than silent | planned; **no GPU acceleration is claimed now** |
| **N6** | Large-scale autonomous research: native candidate-evaluation batches, experiment scheduler, architecture/parameter search campaigns, reproducible benchmark corpus | planned |

N2 is the next phase. N3–N6 depend on N2's execution plan and are not started.

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
2. **No native graph compiler yet.** Nodes call C functions one by one; layout
   operators use NumPy; there is no fusion, prepacking or memory planner.
3. **Naive GEMM.** Correctness-first triple loop with OpenMP. It is not competitive
   with OpenBLAS at medium/large sizes.
4. **Attention is correct but not optimized.** Scores are recomputed for the
   softmax pass; there is no flash-attention tiling, KV cache or fused kernel.
5. **Coverage is partial.** Conv, recurrent, SSM, GNN, MoE, sampling and int8
   `quantized_linear` remain unsupported in this backend; use NumPy or torch.
6. **OpenMP is optional.** The build retries without `-fopenmp` if the compiler
   rejects it; the capability report does not claim a thread count.
7. **No CUDA.** `capabilities()["extra"]["cuda"]` is always `False`. A GPU claim
   requires real kernels and measurement, which do not exist yet.
8. **No mobile/Windows validation.** The library is built and tested on Linux
   x86_64; macOS uses `.dylib` naming, but was not exercised here.

## N2 design (not implemented)

N2 will introduce a native execution-plan compiler rather than changing the
existing NIR semantics:

* lower a validated NIR graph into a linear native plan with explicit buffers;
* fuse `linear` + bias + activation and residual + norm chains;
* prepack weights into the layout the kernels consume (transposed/blocked);
* allocate and reuse buffers with a liveness planner;
* add native conv2d/pool2d, recurrent, SSM, GNN and sparse-MoE kernels;
* implement int8 `quantized_linear` with per-channel scales and a typed
  dequantize boundary;
* add blocked GEMM and a tiled attention kernel;
* compare every compiled plan against the NumPy reference and the N1 direct path.

The acceptance gate will be numerical parity plus a measured improvement over N1
on defined shapes, not a claim based on compilation alone.

## N3–N6 boundaries

* **N3** changes tensor representation and native storage, not the NIR language.
* **N4** adds distributed scheduling around the existing job/worker contracts; it
  must not weaken lease/idempotency semantics.
* **N5** is the only phase allowed to claim GPU execution, and only with real
  kernels, device transfer tests and measurements.
* **N6** builds research campaigns on the deterministic engines; candidate
  generation may be autonomous, but every candidate still passes the same
  STRICT_PLUS, validation and provenance path.
