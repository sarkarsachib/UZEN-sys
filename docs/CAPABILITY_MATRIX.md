# Capability Matrix

Generated from the live operation registry (`GET /v1/operations`, `uzen ops list`).
Maturity levels: `implemented` (executable and tested), `prototype` (executable with
documented limits), `planned` (designed, not executable), `unsupported`.

No operation in this build is `planned` except where noted at the bottom; anything
absent from the registry is rejected with `registry.not_registered`.

## Arithmetic (`family: arithmetic`)

| Code | Params | Effect |
| --- | --- | --- |
| `w.arithmetic.linear_interpolate` | `t` (req), `select`, `partial`, `report_per_tensor` | param_transform |
| `w.arithmetic.slerp` | `t` (req), `preserve_norm` | param_transform |
| `w.arithmetic.extrapolate` | `factor` (req) | param_transform |
| `w.arithmetic.delta` | `select`, `partial`, `report_per_tensor` | param_transform |
| `w.arithmetic.scaled_add` | `scale` (req) | param_transform |
| `w.arithmetic.hadamard` | — | param_transform |
| `w.arithmetic.scale` | `factor` (req) | param_transform |
| `w.arithmetic.add_scalar` | `value` (req) | param_transform |
| `w.arithmetic.negate` | — | param_transform |

Assumptions: identical parameter names and shapes (or explicit `partial=True`),
compatible dtypes, finite values. Weight-only: structure digest is preserved.

## Merging (`family: merging`)

| Code | Params | Assumptions |
| --- | --- | --- |
| `w.merge.soup` | `weights`, `density` n/a | same initialization lineage, identical shapes |
| `w.merge.task_arithmetic` | `weights` (first input is the base) | shared base model |
| `w.merge.ties` | `density`, `weights` | shared base; sign conflicts are trimmed by magnitude |
| `w.merge.dare` | `density`, `weights` | shared base; seeded random drop + rescale |
| `w.merge.disjoint` | `weights` | disjoint support across inputs |

Merging is only defensible when parameterizations are compatible. Use NPA alignment
first when in doubt; the registry does not silently align for you.

## Decomposition / representation (`family: decomposition`, `representation`, `structural`)

| Code | Params | Notes |
| --- | --- | --- |
| `w.decomp.svd_truncate` | `rank` or `energy` | real SVD; reports per-tensor energy captured |
| `w.decomp.factorize` | `rank` | produces `.factor_a` / `.factor_b` tensors (rank-r reshape) |
| `w.decomp.orthogonal_transform` | `method` (qr/hadamard), `side`, `scale` | preserves singular values |
| `w.decomp.permute` | `axis`, `permutation` | deterministic permutation |
| `w.decomp.sparsify` | `fraction` (req), `mode` | magnitude or random pruning |

## Structural (`family: structural`)

| Code | Params | Notes |
| --- | --- | --- |
| `w.structural.tie` | `names` (req), `strategy` (mean/first/last) | parameter sharing |
| `w.structural.slice` | `name`, `axis`, `start/stop/step` | changes shape of one tensor |
| `w.structural.extract` | `layers` (req), `keep` | layer/component extraction |
| `w.structural.prune_heads` | `name`, `n_heads`, `heads` | attention-head pruning |
| `w.structural.transplant` | `mapping` (req) | component transplantation between packages |

Tensors whose shapes change are reported; callers must re-validate the target graph.

## Generation (`family: generation`)

| Code | Params | Origin labels |
| --- | --- | --- |
| `w.gen.structured` | `distribution` (orthogonal/normal/uniform/xavier/zeros/ones), `std` | `randomly_initialized` / `analytically_derived` |
| `w.gen.compose_donors` | `conflict` (error/first/last/mean) | `inherited` / `transformed` |
| `w.gen.constraint_matched` | — | `analytically_derived` (moments matched to a template) |
| `w.gen.from_factors` | `suffix_a`, `suffix_b` | `analytically_derived` (rank reconstruction) |
| `w.gen.low_rank_regeneration` | `rank` (req) | `transformed` from a donor |

Generation never claims learned intelligence: random tensors are labelled
`randomly_initialized`, analytical ones `analytically_derived`.

## Reconstruction (`family: reconstruction`)

| Code | Params | Notes |
| --- | --- | --- |
| `w.recon.fill_missing` | `allow_random_fill`, `distribution` | donor-first; fails if nothing can fill and random fill is disallowed |
| `w.recon.low_rank_fill` | `rank` | reconstruct missing tensors from a low-rank donor basis |
| `w.recon.rename` | `mapping` (req) | rebind names without changing values |

## Alignment (`family: alignment`)

| Code | Params | Method |
| --- | --- | --- |
| `w.align.identity` | `threshold`, `allow_partial` | shape/name correspondence only |
| `w.align.procrustes` | `threshold` | closed-form orthogonal Procrustes (SVD) |
| `w.align.linear_map` | `threshold` | ridge least-squares map |
| `w.align.permutation` | `threshold` | Hungarian assignment on cosine cost |
| `w.align.basis` | `threshold` | whitening/colouring basis change |
| `w.align.auto` | `threshold` | evaluate candidates, pick the best defensible one |

Below threshold → `NoValidAlignment` (`npa.no_valid_alignment`). Reports always state
`semantic_equivalence_claimed: false`.

For activation-space comparison use `ParameterAligner(method="activation", graph=..., activation_inputs=...)`
(linear CKA + neuron matching per node).

## Search (`family: search`)

| Code | Algorithm | Dimensional limit |
| --- | --- | --- |
| `w.search.cma_es` | CMA-ES | any low-dimensional space |
| `w.search.evolution_strategy` | (μ, λ) ES | any |
| `w.search.bayesian_optimization` | GP + EI | any |
| `w.search.coordinate_descent` | pattern search | any |
| `w.search.simulated_annealing` | Metropolis | any |
| `w.search.beam_search` | grid beam | ≤ 4 dims (explicit error above) |
| `w.search.nsga2` | NSGA-II | multi-objective: requires `extra_objectives`, returns the Pareto front |

Search operates on low-dimensional transformation parameters, never on billions of
individual weights. Budget exhaustion raises `budget.exhausted`. NSGA-II is kept in a
separate `MULTI_OBJECTIVE_ALGORITHMS` registry, so scalar planning never selects a
multi-objective optimizer by accident and the scalar algorithms keep their
single-objective contract.

## Architecture (NASE / NIR)

| Capability | Status |
| --- | --- |
| Families (24, catalogue ordered simple→advanced): `linear_residual`, `gated_mlp`, `mlp`, `conv_residual`, `cnn`, `recurrent` (RNN/LSTM/GRU), `autoencoder`, `siamese`, `retrieval`, `set_transformer`, `vae`, `state_space`, `gnn`, `moe`, `diffusion`, `multimodal`, `hybrid`, `transformer_encoder`, `albert`, `bert`, `vit`, `perceiver`, `seq2seq`, `transformer_decoder` | implemented |
| NIR operators (43): linear, embedding, pos_embedding, layernorm, rmsnorm, attention, split_qkv, merge_heads, `rope`, `repeat_kv`, conv2d, pool2d, rnn, lstm, gru, reshape, transpose, concat, slice, flatten, `select`, `moe`, `ssm`, `gcn`, `sample_normal`, `add_noise`, `clip`, reductions, activations including exact `gelu_exact` | implemented |
| Stored reconstructed-model execution (`tokens` + optional `positions`) | implemented for GPT-2/Llama/OPT/GPT-NeoX/GPT-J/Gemma and the explicit Llama-like profile graphs; finite output validation, optional in-memory K/V cache for RoPE decoder graphs |
| Tokenizer-backed text inference | implemented for imported GPT-2/Llama/OPT/GPT-NeoX and explicit profile artifacts with `tokenizer.json`; stored by SHA-256 and encoded through the optional real `tokenizers` runtime |
| Autoregressive generation | greedy, seeded temperature/top-k/top-p sampling, and beam search by cumulative log-probability; cached single-token decoding for RoPE decoder graphs, full recomputation elsewhere; explicit EOS or context-length stopping |
| Weight-only modification with stored-graph preservation | implemented; parent graph metadata is copied to the modified artifact |
| Source/candidate model output comparison | implemented; same-graph enforcement, recorded tolerances and per-output errors |
| Per-layer activation capture | implemented; `capture="layers"` returns `l<N>_res2` block outputs, explicit NIR node ids are also accepted |
| Logit parity for HF families | implemented for GPT-2/Llama/OPT/GPT-NeoX/GPT-J/Gemma and live-verified Mistral/Qwen2/OLMo; other Llama-like profiles are layout- and execution-tested but not claimed as live parity |
| Graph validation (refs, acyclicity, binding, shapes, dtypes, declared port order) | implemented |
| Composition (sequential/parallel/residual), operator substitution, bypass, truncation | implemented |
| Explicit graph cycles | unsupported by design: recurrent/stateful structure lives inside `rnn`/`lstm`/`gru`/`ssm`, and a cycle is diagnosed rather than silently reinterpreted |
| Sparse-dispatch MoE (per-expert gather/scatter of routed tokens) | implemented; unused experts are never evaluated, with equivalence and NaN-isolation tests |
| Prometheus `/metrics` (counters/gauges/histograms) + JSON logging | implemented; dependency-free registry and scraper endpoint |
| Embedded ONNX runtime | planned |
| Native CPU kernels (C++17 + ctypes) | implemented for float32 linear/matmul, layernorm/rmsnorm, softmax, GELU (tanh/exact)/SiLU/ReLU/tanh/sigmoid, add/sub/mul/scale/clip, embedding, last-axis mean/sum and masked attention; optional OpenMP; no CUDA claim |

## Runtimes

| Runtime | Status |
| --- | --- |
| NumPy reference (safetensors, sharded safetensors index, npz, GGUF including dequantization; atomic writes; graph execution) | implemented |
| Native CPU runtime (`uzen.runtimes.native`) | implemented: a direct per-node executor and a compiled plan path (fusion, prepacked blocked GEMM, liveness buffer planner). Build-on-demand with the local C++ compiler, cached by source digest; all seven supported architecture families are parity-tested against NumPy. Layout operators use NumPy and are reported separately. Unsupported operators/dtypes raise instead of falling back. Seeded sampling operators and multi-axis reductions are refused. No CUDA |
| Native compiled plan (`compile_native_plan` / `execute_plan`) | implemented: linear+bias+activation and residual+norm fusion, 64x64 weight prepacking, slot reuse by liveness, shape-polymorphic binding, measured 0.63x the direct executor's time on a 2-layer transformer (batch 4, seq 16, d_model 64); not a GPU claim |
| PyTorch adapter (weights-only state-dict IO, bf16 widening, `inference_mode` forward) | implemented and verified against `torch 2.15.0+cpu` (CPU only; CUDA availability is reported, never claimed); otherwise typed `runtime.unavailable` |
| Torch-CPU NIR executor (all 44 operators) | implemented; parity-tested against the NumPy reference across all 24 architecture families, including direct `rope` and `repeat_kv` kernels and a torch-forced Llama graph. |
| GGUF v3 container + GGML type registry (F32/F16; Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K/Q8_K codecs; remaining K-quant/IQ metadata) | implemented: Q4_K/Q5_K/Q6_K/Q8_K are ported from upstream and C-reference validated; other K-quant/IQ tensors are inspectable without decoding and refused for reads by type name/id |
| Weight-only int8 execution | implemented for 2-D float weights consumed by `linear` nodes: symmetric int8 `[-127, 127]` values with per-output-channel float32 scales, float activations, an explicit per-tensor report, and NumPy/Torch kernels; non-linear and shared weights are excluded explicitly |
| ONNX interchange for the NIR subset (export/import) | implemented when the `onnx` extra is installed; opset 20 export, standard `Gelu(approximate="tanh")` maps to NIR `gelu`, the Erf GELU form is refused, validated with `onnx.checker`, no embedded executor |
| CUDA-native kernels | planned — UZEN never claims GPU acceleration it does not have |

## Formats

| Format | Status |
| --- | --- |
| `.safetensors` (own codec, strict header validation) | implemented |
| Sharded safetensors (`model.safetensors.index.json`) | implemented with exact weight-map checks; validated on real `distilgpt2` (4 shards, 82 tensors) and `openai-community/gpt2` (5 shards, 160 tensors), where single-file and sharded imports produce identical digests, logits and tokens, and a missing declared shard is refused |
| `.npz` | implemented (NumPy runtime) |
| `.pt/.pth/.bin/.ckpt` | PyTorch adapter with `weights_only=True`; pickle objects refused |
| `.pkl` and other pickle formats | **unsupported by design** (`format.unsupported`) |
| `.gguf` (v3 container) | implemented: F32/F16 exact; Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K/Q8_K read (dequantized); Q8_0 write; registry-driven inspection; requantization with error report; remaining K-quant/IQ refused by code |
| `.onnx` | implemented for the exportable NIR subset (requires `onnx`, opset 20); unmapped operators and the Erf GELU form are refused |

## Interfaces

| Interface | Status |
| --- | --- |
| Python SDK (local + remote HTTP) | implemented |
| CLI (argparse) | implemented |
| REST API (FastAPI, OpenAPI) | implemented (optional extra) |
| MCP stdio server (13 tools) | implemented |
| Worker protocol + daemon (local and remote) | implemented |
| Object-store artifact transport | implemented for S3-compatible endpoints: local SigV4 signing, streamed single and multipart uploads, SHA-256 verification before publication, parent-lineage closure and stored-tokenizer sidecars; validated against a real S3-compatible server and by an independently signature-checking test server |
| Redis/durable queue adapter | planned (SQLite-backed lease queue implemented) |

## Known limitations (explicit)

1. **No distributed scheduler.** The job service claims jobs with SQLite leases; a
   multi-node deployment needs a shared database and an object store.
2. **Remote artifact transfer** between control plane and worker supports a shared
   filesystem, inline tensors and S3-compatible object storage. The worker daemon
   does not fetch artifacts implicitly; callers invoke
   `ObjectStoreArtifactTransport` explicitly (see `examples/object_store_transport_demo.py`).
3. **LLM planning** requires an OpenAI-compatible endpoint; without one it raises rather
   than pretending an LLM was consulted.
4. **Activation alignment** requires a runnable NIR graph and real inputs; it will not
   fabricate activations for unsupported architectures.
5. **Architecture families beyond the five implemented** are reported `planned`; UZEN
   will not synthesize a graph it cannot execute.
6. **The native CPU backend is float32-only and correctness-first.** The direct
   executor uses a naive GEMM and NumPy for layout ops and covers
   [P6_NATIVE.md](P6_NATIVE.md)'s documented operator set; the compiled plan adds
   conv/pool, recurrent, SSM, GCN, MoE, int8 `quantized_linear`, a prepacked GEMM
   and single-pass attention. Seeded `sample_normal`/`add_noise` are refused. It is
   not a CUDA backend.

## P0 substrate hardening

| Control | Status |
| --- | --- |
| Enforced `WeightPackage` immutability | implemented: byte-backed read-only tensor buffers, frozen mappings/attributes, fingerprints re-checked at store, registry and executor boundaries |
| Artifact integrity on load/export | implemented: manifest checksum, payload digest, dtype/shape inventory, lineage parent checksums and policy records are verified before a package is returned |
| Numerical safety coverage | implemented tests for NaN/Infinity, zero-rank scalars, empty tensors, subnormal underflow, float16 extremes and 1M-element dimensions |
| Resource protection | implemented: cooperative operation, per-tensor, total-byte, timeout and cancellation limits for NIR execution; artifact byte budgets |
