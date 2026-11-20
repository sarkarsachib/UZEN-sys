# UZEN — Universal Neural Weight Engineering & Architecture Research System

UZEN is a **headless computational substrate for engineering neural networks** without
training. It treats weights and architectures as programmable mathematical objects:
inspect, understand, research, design, generate, transform, compose, reconstruct,
evaluate, verify, export.

It is not a chatbot, an agent wrapper, a training platform, a model host, or a SaaS
frontend. There is no mandatory UI. Every capability is reachable through the Python
SDK, CLI, REST API and MCP server.

```
STRICT_PLUS: no training · no fine-tuning · no backpropagation · no gradient descent
```

---

## Status — honest capability report

| Area | State |
| --- | --- |
| STRICT_PLUS policy engine + hash-chained audit log | **implemented** |
| NWS tensor substrate, parameter roles, content-addressed artifact store | **implemented** |
| Enforced immutable WeightPackage (read-only buffers, frozen mappings, fingerprint checks) | **implemented** |
| Artifact integrity verification on load/export + hash-chained provenance | **implemented** |
| Cooperative execution and artifact resource budgets | **implemented** |
| Safetensors codec (from scratch, header validation, BJNT-safe BF16) | **implemented** |
| NIR typed dataflow graph (symbolic dims, control-flow `select`, MoE/SSM/GNN/sampling kernels) | **implemented** |
| Weight operation registry (45 registered operations) | **implemented** |
| Arithmetic, merging (incl. TIES/DARE), decomposition (SVD/orthogonal/sparsify) | **implemented** |
| NPA alignment: identity, Procrustes, linear map, permutation (Hungarian), basis, activation/CKA | **implemented** |
| NASE synthesis: 24 families ordered simple→advanced (residual/gated MLP, CNN, RNN, autoencoder, Siamese, retrieval, set transformer, VAE, SSM, GNN, MoE, diffusion, multimodal, hybrid, transformer encoder, ALBERT, BERT, ViT, perceiver, seq2seq, transformer decoder), compose, substitute, bypass, truncate | **implemented** |
| WComp compiler + WPL language (real lexer, recursive-descent parser, typed pipeline) | **implemented** |
| VEL verification: structural, numerical, functional, regression, performance, provenance | **implemented** |
| ANE planning, candidate generation, 6 scalar derivative-free algorithms + NSGA-II Pareto search, diagnosis | **implemented** |
| WGE / WME / WAR / MAR engines | **implemented** (built on NWPC/NASE) |
| WIE inspection, statistics, role inference, family hypothesis | **implemented** |
| Hugging Face / safetensors import plus weights-only PyTorch state-dict and single-file ONNX artifact import (sharded checkpoints, optional Hub download, explicit provenance and compatibility reports) | **implemented** |
| HF-to-NIR reconstruction with tensor/config consistency, graph validation, finite forward execution, and reference-logit parity | **implemented for explicit GPT-2, Llama, OPT, GPT-NeoX, GPT-J and Llama-like profile adapters (Gemma, Mistral sliding-window, Qwen2, Yi, Aquila, Baichuan, InternLM2, OLMo, MiniCPM, Orion); Llama covers RoPE, GQA-aware KV repetition, RMSNorm and SwiGLU; OPT covers the +2 offset position table, biased projections and ReLU; GPT-NeoX covers partial RoPE over interleaved fused QKV and exact/tanh GELU; GPT-J covers parallel residual blocks and interleaved partial RoPE; Gemma covers zero-centered RMSNorm, embedding scale and GeGLU; OLMo adds non-affine LayerNorm and QKV clipping; Qwen2 adds Q/K/V biases. Profiles that need ALiBi or fused non-Llama layouts are reported `unsupported` with their reason** |
| Execute stored reconstructed model graphs and compare weight-modified artifacts on identical inputs | **implemented** (stored GPT-2/Llama/OPT/GPT-NeoX and explicit profile graphs; per-output tolerances and error metrics) |
| Weight-only int8 execution | **implemented** (symmetric per-output-channel int8 linear weights, float activations, explicit quantization report with per-tensor scales and reconstruction errors; selectable through SDK/REST/CLI/MCP execution and generation, or persisted as reusable artifact variants with source-versus-quantized drift comparison and configurable aggregate failure thresholds; embeddings, norms and biases remain floating point) |
| Per-layer activation capture and Hugging Face block-output parity | **implemented** (`capture="layers"` exposes `l<N>_res2` outputs; transformer forward hooks compare every layer, not just logits) |
| Tokenizer-backed text inference for imported GPT-2/Llama/OPT/GPT-NeoX and explicit profile artifacts | **implemented** (`tokenizer.json` stored by SHA-256, or assembled from legacy `vocab.json` + `merges.txt`; real `tokenizers` encoding, optional `uzen[tokenizer]` extra) |
| Autoregressive text generation | **implemented** (greedy, seeded top-k/top-p sampling, beam search; cached single-token decoding for RoPE decoder graphs; EOS and context-length stopping) |
| REST API (FastAPI), Python SDK, CLI, MCP stdio server, worker daemon | **implemented** |
| Worker authentication: versioned HMAC envelopes, signing-key rotation, durable SQLite replay guard, scoped credentials | **implemented** |
| PyTorch runtime adapter (safe state-dict IO, inference-mode forward) | **implemented and tested with CPU-only torch 2.15; CUDA is never claimed** |
| Torch-CPU NIR execution (all 44 operators, parity-tested against NumPy) | **implemented** |
| GGUF v3 adapter + GGML type registry (F32/F16, Q4/Q5/Q8_0/Q4_K/Q5_K/Q6_K/Q8_K decode, Q8_0 encode, payload-free inspection) | **implemented** |
| ONNX interchange (export/import of the supported NIR subset) | **implemented** (requires the `onnx` extra; opset 20, `Gelu(approximate="tanh")` round trip, validated with `onnx.checker`; the Erf GELU form is refused rather than rewritten) |
| Sparse-dispatch MoE (only routed tokens are computed per expert) | **implemented** |
| Prometheus `/metrics` + structured JSON logging (no extra dependencies) | **implemented** |
| Native CPU kernels (C++17, ctypes, optional OpenMP) | **implemented** for linear/matmul, norms, softmax, activations, elementwise arithmetic, embedding, reductions and masked attention; float32 only, with explicit typed refusal outside that set |
| Native NIR executor + runtime adapter | **implemented** for MLP and transformer-decoder graphs, parity-tested against NumPy; no silent fallback |
| Native graph compiler (fusion, prepacked blocked GEMM, liveness buffer planner) | **implemented**; plans fuse linear+bias+activation and residual+norm, prepack linear weights and reuse dead buffer slots, with parity tests per family |
| Native conv/pool, RNN/LSTM/GRU, SSM, GCN, sparse-MoE and int8 quantized-linear kernels | **implemented** in the compiled plan path; seeded sampling operators are refused |
| Native parameter representations: block int8, block int4, low-rank factors, CSR sparse and transposed layouts | **implemented** as per-weight layouts chosen at compile time, each with packed-byte, bits/weight, reconstruction-error and digest provenance; lossy layouts report their measured error instead of claiming to be exact |
| Embedded ONNX runtime / GGUF quantized tensor types | **planned** (typed refusal, no simulation) |
| Rust integration / CUDA kernels | **planned** — UZEN never claims GPU acceleration it does not have |
| S3-compatible object-store artifact transport | **implemented** (stdlib SigV4 client, streamed single/multipart uploads, SHA-256 and tensor-fingerprint verification, parent lineage and tokenizer sidecars; validated against a real S3-compatible server) |
| LLM-assisted planning | **implemented when an OpenAI-compatible endpoint is configured** |

Anything not implemented reports `planned`/`unsupported` and raises a typed error.
Random initialization is always labelled `randomly_initialized`. No operation claims
CUDA acceleration that does not exist.

### Computational-substrate phase plan

The master roadmap's Phase 6 is split into six phases with explicit gates. N1–N3
are implemented; the rest are designs, not partially working claims.

| Phase | Scope | State |
| --- | --- | --- |
| **N1** | Native C++ CPU kernels, ctypes ABI, NIR executor, runtime adapter, parity tests, demo | **implemented** |
| **N2** | Native graph compiler, fusion, prepacking, blocked GEMM, conv/RNN/SSM/GCN/MoE/int8 kernels, single-pass attention, memory planner | **implemented** |
| **N3** | Advanced parameter representations: block int8/int4 with group scales, low-rank factors, CSR sparse and transposed layouts, representation-aware execution | **implemented** |
| **N4** | Durable distributed queue/scheduler and capability-aware native workers | planned |
| **N5** | Real CUDA/HIP kernels and a device boundary; no GPU claim before kernels exist | planned |
| **N6** | Large-scale autonomous research campaigns on the deterministic engines | planned |

See [docs/P6_NATIVE.md](docs/P6_NATIVE.md) for coverage, build instructions, verified
measurements and limitations.

---

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e .            # core: numpy + pydantic
.venv/bin/pip install -e '.[api,dev]' # REST API + test tooling
.venv/bin/pip install -e '.[onnx]'    # optional: ONNX interchange
.venv/bin/pip install -e '.[tokenizer]' # optional: text inference from tokenizer.json
.venv/bin/pip install -e '.[hf]'      # optional: Hugging Face Hub download
.venv/bin/pip install -e '.[hf-reference]'  # optional: Hugging Face logits parity (also install torch)
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu  # optional CPU-only adapter
```

The engine itself needs only `numpy` and `pydantic`. FastAPI/uvicorn are optional
extras used solely by the control-plane service.

## Quick start (SDK)

```python
from uzen import Uzen

client = Uzen(root="./uzen-workspace")

# 1. Build a real architecture (tiny transformer decoder) and its parameters.
model = client.generate(
    family="transformer_decoder",
    spec=dict(d_model=64, n_heads=4, n_layers=2, d_ff=128, vocab_size=128, max_seq_len=32),
    seed=0,
)
print(model["artifact_id"], model["verification"]["functional"]["ok"])

# 2. Weight-only modification: linear interpolation between two artifacts (t=0.5).
merged = client.modify(model["artifact_id"], "interpolate", others=[model["artifact_id"]], params={"t": 0.5})
print(merged["output_artifact_id"])

# 3. Verify provenance + integrity.
print(client.verify_artifact(merged["output_artifact_id"])["ok"])
```

## CLI

```bash
uzen families
uzen ops list --executable-only
uzen policy show
uzen inspect model.safetensors
uzen import model.safetensors --family transformer_decoder
uzen import model.pt --family transformer_decoder
uzen import model.onnx
uzen import ./tiny-hf-model --revision main
uzen import --repo-id org/model-name --revision main
uzen modify <artifact_id> interpolate --other <artifact_id> --param t=0.5
uzen forward <artifact_id> --tokens '[[0,1,2,3]]'
uzen forward <artifact_id> --tokens '[[0,1,2,3]]' --capture layers --values
uzen forward <artifact_id> --text 'hello world' --capture layers
uzen generate-text <artifact_id> --text 'hello world' --max-new-tokens 8
uzen quantize <source_id> --scheme int8
uzen compare-models <source_id> <candidate_id> --tokens '[[0,1,2,3]]'
uzen compare-quantized <source_id> <quantized_id> --tokens '[[0,1,2,3]]'
uzen evaluate <artifact_id> --family mlp --spec '{"input_dim":8,"hidden_dims":[16],"output_dim":4}'
uzen serve --host 0.0.0.0 --port 8080
uzen mcp --root ./uzen-workspace
uzen worker run --root ./uzen-worker
```

## Real-model demo: int8 release candidate

This is a working end-to-end use case, not a synthetic fixture: import the real
`distilgpt2` checkpoint, create a persisted int8 variant, compare source-versus-
variant logits, generate from both, export the variant, re-import it and verify
the reloaded artifact.

```bash
.venv/bin/pip install -e '.[hf,safetensors,tokenizer]'
.venv/bin/python examples/quantized_release_demo.py
```

Set `HF_HUB_DISABLE_XET=1` if the Hub's Xet endpoint is unavailable. The command
writes the export under `<root>/exports` and a machine-readable report under
`<root>/reports/quantized_release.json`. It exits non-zero if the drift or
integrity gate fails. The verified run used `--root /tmp/uzen-demo-distilgpt2`:

```text
source artifact: fe1896b4...  (77 tensors)
variant artifact: 372df735...
package size: 482,039,808 -> 239,212,612 bytes (2.02x smaller)
drift gate: PASS (max_abs=1.94438, mean_abs=0.512857, elements=255125)
source:  ' J.J.K. Rowling.'
variant: ' J.J.K. Rowling.'
generated token sequence match: True
export: .../exports/distilgpt2-int8-372df7355959.safetensors (239,259,484 bytes)
reloaded artifact: 55991488... (logits=[1, 5, 50257], shape_match=True, integrity=True)
report: .../reports/quantized_release.json
```

The generated continuations are reported separately from the logits gate because
a variant can pass a logit threshold and still diverge on longer generations.

## Real-model demo: import and execution matrix

This demo imports real pretrained checkpoints, reconstructs and validates their
NIR graphs, executes a real tokenizer-backed forward pass, generates text, and
writes an audit report. It covers prefixed and prefixless GPT-2 layouts, OPT's
weights-only `.bin` fallback and legacy BPE tokenizer, GPT-NeoX, and a Llama-family
checkpoint.

```bash
HF_HUB_DISABLE_XET=1 .venv/bin/python examples/real_model_import_demo.py \
  --models distilgpt2 openai-community/gpt2 facebook/opt-125m \
           EleutherAI/pythia-14m HuggingFaceTB/SmolLM-135M \
  --root /tmp/uzen-import-all --max-new-tokens 2
```

Verified output:

```text
distilgpt2: ok=True overall=validated reconstruction=validated execution=passed
openai-community/gpt2: ok=True overall=validated reconstruction=validated execution=passed
facebook/opt-125m: ok=True overall=validated reconstruction=validated execution=passed
EleutherAI/pythia-14m: ok=True overall=validated reconstruction=validated execution=passed
HuggingFaceTB/SmolLM-135M: ok=True overall=validated reconstruction=validated execution=passed
forward: logits=[1, 5, 50257]   # GPT-2 / DistilGPT-2
forward: logits=[1, 5, 50272]   # OPT-125M
forward: logits=[1, 5, 50304]   # Pythia-14M
forward: logits=[1, 5, 49152]   # SmolLM-135M
generation: ' J.' / ' John.' / ' Tanya' / ':\n' / ' Alex.'
report: /tmp/uzen-import-all/reports/real_model_import.json
gate: PASS
```

GPT-2 causal-mask buffers (`attn.bias`) are recorded as explicitly ignored
non-parameter tensors; all other source tensors must map to the canonical graph
or the import remains `partial`. OPT is published without safetensors, so UZEN
loads its single `pytorch_model.bin` with `weights_only=True` and assembles a
byte-level BPE tokenizer from `vocab.json` + `merges.txt`; both fallbacks are
recorded in the report. The demo accepts local checkpoint directories as well as
Hub ids and exits non-zero if any requested checkpoint fails.

## Real-model demo: sharded checkpoint and ONNX interchange

This demo turns a real published checkpoint into a genuinely sharded layout —
balanced shards plus a real `model.safetensors.index.json`, written by UZEN's own
safetensors codec — imports both the single-file and sharded layouts, executes
them, and compares digests, logits and generated tokens. It then exports GPT-2
layer-0's real MLP block to a single-file ONNX model, validates it with
`onnx.checker`, imports it back into NIR and compares both executions.

```bash
HF_HUB_DISABLE_XET=1 .venv/bin/python examples/sharded_onnx_import_demo.py \
  --model distilgpt2 --root /tmp/uzen-sharded-onnx-demo --shards 4
```

Verified output (`distilgpt2`, 4 shards, 82 indexed tensors):

```text
wrote 4 shards (82 tensors indexed, 6080 byte index)
[single_file] artifact=fe1896b4… overall=validated execution=passed sharded=False
[sharded]     artifact=fe1896b4… overall=validated execution=passed sharded=True files=4
[compare] digest_equal=True logits_equal=True generation_equal=True
[control] removed model-00001-of-00004.safetensors; refused=True
[onnx] ops=['MatMul', 'Add', 'Gelu', 'MatMul', 'Add'] bytes=18890262 checker=passed max_abs_error=0.000e+00
[onnx] stored artifact=ce7badae… format=onnx graph_nodes=3 integrity=True
gate: PASS
```

The same gate passes on `openai-community/gpt2` with `--shards 5` (160 indexed
tensors, generation `" John. I'm"`, stored ONNX artifact `2fc6dcf1…`).
`onnxruntime` is not installed in this environment, so the ONNX model is
validated by `onnx.checker` and executed through UZEN's NIR runtime, not through
an external ONNX runtime. Reports are written to
`<root>/reports/sharded_onnx_import.json`.

## Real-model demo: S3-compatible artifact transport

A worker without a shared filesystem must receive the exact artifact the control
plane produced. This demo imports `distilgpt2` into a content-addressed store,
pushes the artifact and its stored tokenizer to an S3-compatible bucket, fetches
it into an empty store, verifies every SHA-256 and the tensor fingerprint, then
executes the fetched copy and compares logits and greedy generation.

```bash
UZEN_S3_ENDPOINT=http://127.0.0.1:9000 UZEN_S3_BUCKET=uzen-artifacts \
UZEN_S3_ACCESS_KEY=... UZEN_S3_SECRET_KEY=... \
HF_HUB_DISABLE_XET=1 .venv/bin/python examples/object_store_transport_demo.py \
  --model distilgpt2 --root /tmp/uzen-object-store-demo
```

Verified against a local S3-compatible server (`rclone serve s3`, 483 MB artifact):

```text
source artifact: fe1896b4187d… tensors=77 overall=validated
push: artifacts=1 objects=4 skipped=0 bytes=483484366 seconds=6.983
fetch: fetched=1 present=0 bytes=483484366 verified=True seconds=18.685
forward: shape=[1, 5, 50257] max_abs_error=0.000e+00
generation: ' J.' (equal=True)
gates: all six passed
report: /tmp/uzen-object-store-demo/reports/object_store_transport.json
gate: PASS
```

The six gates cover import validation, digest-matching remote objects, verified
fetch, identical logits, identical generated tokens and refusal of an unknown
artifact id. The transport signs every request with AWS Signature Version 4,
uses single `PutObject` below the threshold and a real multipart upload above
it, and retries transient failures by re-signing each attempt. `onnxruntime`-style
optional runtimes are not involved: the client uses the standard library only.

## REST control plane

`create_app()` exposes a versioned OpenAPI 3.1 surface (`/docs` for the schema):
jobs, models, weights, artifacts, architectures, operations, workers, policy.
Set `UZEN_API_TOKEN` to require bearer auth; worker routes additionally verify each
worker's registered secret hash.

```bash
curl -s localhost:8080/v1/operations | jq '.count'
curl -s -X POST localhost:8080/v1/jobs -H 'content-type: application/json' \
  -d '{"task_scope":"weight_modification","objective":"scale a model",
       "inputs":[{"name":"base","artifact_id":"<id>"}],
       "operations":[{"op":"w.arithmetic.scale","params":{"factor":0.5}}]}'
```

## MCP

`uzen mcp` speaks JSON-RPC 2.0 over stdio (newline or `Content-Length` framing) and
exposes deliberate tools — `uzen.inspect_model`, `uzen.modify_weights`,
`uzen.generate_weights`, `uzen.evaluate_candidate`, `uzen.inspect_provenance`, and
others. Each tool enforces task scope and STRICT_PLUS; an AI client cannot widen scope
or request a prohibited effect. A `tools/call` with `params._meta.progressToken` emits
real JSON-RPC `notifications/progress` messages during import, incrementing per stage.

## The two engineering domains

Weights and architectures are **separate** engineering domains:

* **Weight engineering** (`weight_generation`, `weight_modification`,
  `weight_reconstruction`, `weight_optimization`, `weight_architecture_research`)
  changes parameters only. The structure digest of the output equals the source's.
* **Model/architecture engineering** (`model_creation`, `model_modification`,
  `model_reconstruction`, `model_architecture_research`, `architecture_synthesis`)
  changes graphs. The policy engine refuses architecture effects inside a weight-scope
  job and parameter effects inside an architecture-scope job.

A job's authorized scope is fixed at submission and preserved through planning,
registry dispatch, execution and artifact validation.

## Repository layout

```text
uzen/
├── core/
│   ├── nws/      # tensors, roles, manifests, families, artifact store
│   ├── nir/      # typed graph, operator registry/kernels, executor
│   ├── wpl/      # lexer + recursive-descent parser
│   ├── wie/      # weight intelligence / inspection
│   ├── nwpc/     # operation registry, operation context
│   ├── npa/      # neural parameter alignment
│   ├── nase/     # architecture synthesis
│   ├── wcomp/    # compiler, pipeline, execution plan
│   ├── vel/      # verification & evaluation lab
│   └── ane/      # autonomous neural engineer, search, planning, LLM bridge
├── engines/      # wge, wme, war, mar
├── operations/   # arithmetic, merging, decomposition, structural, generation,
│                 # reconstruction, alignment, search
├── runtimes/     # numpy reference, PyTorch adapter, native C++ CPU runtime
├── workers/      # protocol messages, worker daemon
├── interfaces/   # REST API, MCP server, CLI, Python SDK
├── infrastructure/ # storage (safetensors, SQLite), security, provenance
├── policies/     # strict_plus
└── jobs/         # job service (submit/run/cancel/events)
```

## Documentation

* [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — module contracts and data flow.
* [docs/STRICT_PLUS.md](docs/STRICT_PLUS.md) — policy semantics and enforcement points.
* [docs/CAPABILITY_MATRIX.md](docs/CAPABILITY_MATRIX.md) — every operation, maturity and assumptions.
* [docs/API.md](docs/API.md) — REST/SDK/CLI/MCP contracts and error codes.
* [docs/P1_IMPORT.md](docs/P1_IMPORT.md) — Hugging Face/safetensors import and compatibility report.
* [docs/P2_MODEL_EXECUTION.md](docs/P2_MODEL_EXECUTION.md) — stored-model execution and modified-model comparison.
* [docs/P6_NATIVE.md](docs/P6_NATIVE.md) — native CPU kernels, phase plan and measured limits.

## Tests

```bash
.venv/bin/python -m pytest -q
```

Live Hugging Face validation is opt-in: set the `UZEN_RUN_LIVE_HF` repository
variable to `1`, or dispatch the CI workflow with `live_hf`, to run
`UZEN_LIVE_HF=1 pytest tests/test_live_hf_import.py` against the tiny GPT-2,
Llama, OPT, GPT-NeoX, Qwen2 and OLMo fixtures.

The suite exercises the 15 mandatory acceptance tests against real implementations:
real tensor arithmetic, real SVD, real alignment refusals, real policy rejection, real
job retries, real worker authentication and real graph binding validation. There are
no mocked computations.

## License

MIT. See [LICENSE](LICENSE).

## Explicit non-goals

* Turning UZEN into a web application, dashboard or SaaS.
* TypeScript as the neural computation language.
* Claiming a module is implemented because a route or database table exists.
* Producing "improved" models from random weights without a mathematical justification.
* Overwriting source weights, or emitting artifacts without provenance and validation.
