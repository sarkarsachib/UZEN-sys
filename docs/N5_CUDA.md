# N5 — CUDA device boundary

N5 is the first phase allowed to claim GPU execution. It does so only when real
kernels were compiled, loaded through the CUDA driver and launched on a device.
This document separates three things that are easy to blur:

| Question | Answer in this build |
| --- | --- |
| Do CUDA kernels exist? | **Yes.** `uzen/runtimes/cuda/kernels.cu` contains 17 float32 kernels. |
| Do they compile? | **Verified here.** NVRTC 12.4 compiles the source to 51,782 bytes of PTX; `ptxas` 12.4 assembles it to a 52,032-byte cubin. |
| Have they executed on a GPU? | **Not in the authoring environment.** It has no `libcuda` and no device, so no kernel was launched and no speed is claimed. GPU-only tests are present and skip until run on a machine with a device. |

The N5 device runtime is therefore *complete as code and compile-verified*, but not
*execution-verified* on this host. Nothing below reports a GPU timing.

## Path from NIR to the device

```text
Graph + WeightPackage
        │
        ▼
resolve_bindings → per-node dispatch
        │                         │
        │ arithmetic              │ layout (reshape, rope, split_qkv, …)
        ▼                         ▼
CudaExecutor                  NumPy on host (reported)
        │
        ▼
CUDA driver API (ctypes: libcuda)
        │  cuModuleLoadData(PTX) → cuLaunchKernel
        ▼
kernels.cu  ── compiled by NVRTC (or nvcc) ──► PTX
```

The runtime is a fourth executor next to NumPy, torch-CPU and native C++. It uses
the same NIR graph, binding rules and operator parameters, so a graph that runs on
the native CPU runtime can be handed to the CUDA runtime unchanged when it stays
inside the device operator set.

## Kernels

| Operator | Kernel | Notes |
| --- | --- | --- |
| `linear` | `uzen_linear_f32` | 16×16 tiled GEMM, optional bias, weight `(out, in)`, any batch rank flattened to rows |
| `matmul` | `uzen_matmul_f32` | 16×16 tiled, rank-2 only |
| `layernorm` | `uzen_layernorm_f32` | one block per row, optional weight/bias |
| `rmsnorm` | `uzen_rmsnorm_f32` | one block per row, optional weight |
| `softmax` | `uzen_softmax_f32` | last axis, max-shifted |
| `gelu`, `gelu_exact`, `silu`, `relu`, `tanh`, `sigmoid` | one kernel each | float32, `erff`/`tanhf` device math |
| `add`, `sub`, `mul` | `uzen_binary_f32` | **identical shapes only**; broadcasting is refused, not emulated |
| `scale`, `clip` | one kernel each | elementwise |
| `embedding` | `uzen_embedding_f32` | int64 indices, float32 table |
| `mean`, `sum` | `uzen_reduce_last_f32` | last axis, optional keepdims |
| `attention` | `uzen_attention_f32` | causal, KV-cache offset, symmetric window, ALiBi, optional scale |

Layout operators (`reshape`, `transpose`, `slice`, `flatten`, `concat`, `split_qkv`,
`merge_heads`, `rope`, `repeat_kv`, `pos_embedding`, `identity`, `dropout`, `select`)
download, move bytes with NumPy and upload again. They are listed in
`CudaExecution.layout_ops`, and their transfers appear in the allocator stats — a
run never presents them as device work.

Attention matches the CPU reference semantics exactly:

* causal masking uses `qpos = query + (kv_seq - q_seq)`, so single-token decoding
  attends to every cached key up to itself;
* `window=w` keeps keys with `abs(qpos - key) <= w` (symmetric, like the CPU kernel);
* ALiBi subtracts `slope[head] * distance`, where distance is `qpos - key` when
  causal and `abs(qpos - key)` otherwise;
* a fully masked row produces zero, not NaN.

## Compilation

The source is checked in; nothing is generated from a template at run time.

```bash
# NVRTC (default when nvcc is absent): no CUDA toolkit required.
python -m uzen.runtimes.cuda.build --json

# Pick an architecture, force a rebuild, or write a specific PTX file.
python -m uzen.runtimes.cuda.build --arch compute_80 --force
python -m uzen.runtimes.cuda.build --output /tmp/uzen_cuda.ptx
```

* `UZEN_CUDA_ARCH` selects the PTX target (default `compute_60`).
* `UZEN_CUDA_CACHE` overrides the cache directory (default `~/.cache/uzen/cuda`).
* `UZEN_NVRTC_LIBRARY` / `UZEN_CUDA_DRIVER` point at a non-standard `libnvrtc` /
  `libcuda`.
* `UZEN_CUDA_COMPILER` selects a specific `nvcc`.
* The cache key is the SHA-256 of the source plus the architecture, so editing a
  kernel always produces a new artifact. The build report says whether it was a
  cache hit and which compiler ran.

Installing the pip wheel is enough for NVRTC:

```bash
pip install 'uzen[cuda]'
# equivalent to:
pip install nvidia-cuda-nvrtc-cu12
```

GPU execution still requires the vendor driver (`libcuda.so.1`); the NVRTC wheel
does not contain one and this project does not ship or emulate a driver.

## Runtime use

```python
import numpy as np
from uzen.runtimes.cuda import CudaRuntime

with CudaRuntime(fallback="error") as runtime:       # refuses without a device
    execution = runtime.execute_graph(graph, inputs, package, capture=["block_3"])
    print(execution.backend)                          # "cuda"
    print(execution.summary())                        # launches, transfers, peak bytes
    logits = execution.output("logits")
```

* Weights are uploaded once per `(package fingerprint, node, parameter)` and stay
  resident; a different package rebuilds the executor and its cache.
* Activations use a size-class pool. Freed blocks are reused and the report includes
  `allocations`, `reused_blocks`, `peak_live_bytes`, `host_to_device_bytes` and
  `device_to_host_bytes`.
* `capabilities().extra["execution_verified"]` is `False` until a graph actually
  ran in that process. `kernels` lists the compiled kernel names.

### Fallback policy

| Construction | No device | Device present |
| --- | --- | --- |
| `CudaRuntime()` / `fallback="error"` | raises `runtime.unavailable` with the probe reason | runs CUDA |
| `CudaRuntime(fallback="cpu")` | runs the NumPy reference; `backend="numpy-fallback"`, `device="cpu"`, reason recorded | runs CUDA |

The CPU path is never entered silently, and a fallback execution does not increment
the device execution counter or set `execution_verified`.

## Verification

Run on the authoring host (no GPU):

```text
pytest tests/test_cuda_runtime.py      11 passed, 4 skipped
demo /tmp/uzen-cuda-demo               PASS kernels compiled
                                        PASS ptxas assembly
                                        PASS strict refusal
                                        PASS fallback parity
                                        PASS fallback is labelled
                                        RESULT: SKIPPED (no CUDA device)
```

The four skipped tests are the real GPU tests: MLP parity, transformer-decoder
parity with attention capture, weight residency across two runs, and attention
mask/ALiBi/window parity. They are not replaced by mocks or CPU approximations.
The compile tests do run everywhere: they assert all 17 kernel names are present in
the PTX and assemble it with `ptxas` for `sm_60`.

To verify execution on a GPU machine:

```bash
pip install 'uzen[cuda]'
python -m pytest -o addopts="" -q tests/test_cuda_runtime.py
python examples/cuda_device_demo.py --root /tmp/uzen-cuda-demo
```

The demo then runs an MLP and a 2-layer transformer decoder on the device, compares
both against NumPy, captures an attention tensor, checks weight residency and writes
`report.json` with real transfer and peak-memory numbers.

## Honest limitations

1. **No execution measurement here.** The host has no driver or device. No latency,
   throughput or speedup number appears in this document or the demo output.
2. **No HIP/ROCm backend.** N5's CUDA boundary is implemented; a HIP translation is
   not. `CudaRuntime` does not pretend to support ROCm.
3. **float32 only.** No half/bfloat16, tensor cores, TF32 or mixed precision.
4. **Naive kernels.** The GEMM is a shared-memory tile, not a tuned cuBLAS-class
   implementation; attention is a per-query block scan, not FlashAttention. Do not
   expect competitive performance.
5. **Symmetric-shape elementwise only.** Broadcasting `add/sub/mul` is refused with
   `UnsupportedOperator`; the NumPy/native runtimes still support it.
6. **Layout stays on the host.** Reshape/RoPE/split/merge and friends round-trip
   through host memory and are reported; that can dominate small graphs.
7. **No CUDA graphs, streams or multi-GPU.** One context, one device, the default
   stream. There is no peer-to-peer or model-parallel sharding.
8. **Attention shared-memory ceiling.** Scores use dynamic shared memory; a key
   sequence above roughly 12k exceeds the 48 KiB default on older architectures
   unless the launch opts into a larger carve-out.
9. **No quantized device path.** `quantized_linear` and N3's int4/int8/sparse
   representations are refused on the CUDA executor rather than dequantized on the
   fly.
10. **NVRTC is a Python-wheel dependency for toolkit-free builds.** Without NVRTC or
    nvcc, compilation fails with `runtime.unavailable` naming what is missing.
11. **Driver/device differences are real.** PTX is JIT-compiled by the installed
    driver. A driver older than the PTX target rejects the module; the runtime
    surfaces that as `runtime.cuda_kernel`, not a silent CPU run.

## Code map

| File | Role |
| --- | --- |
| `uzen/runtimes/cuda/kernels.cu` | the 17 CUDA kernels |
| `uzen/runtimes/cuda/nvrtc.py` | NVRTC/nvcc compile, digest cache, kernel list |
| `uzen/runtimes/cuda/build.py` | `python -m uzen.runtimes.cuda.build` |
| `uzen/runtimes/cuda/driver.py` | ctypes driver API, context/module/launch |
| `uzen/runtimes/cuda/allocator.py` | pooled device memory and transfer stats |
| `uzen/runtimes/cuda/executor.py` | NIR → kernels, host layout ops, accounting |
| `uzen/runtimes/cuda/runtime.py` | capabilities, fallback policy, weight residency |
| `tests/test_cuda_runtime.py` | compile/PTX, policy and GPU-gated parity tests |
| `examples/cuda_device_demo.py` | executable report with honest skip status |
