# P1 — Hugging Face / safetensors import and GPT-2/Llama/OPT/GPT-NeoX reconstruction

P1 now covers two separate stages:

1. **Import** weights and source metadata safely.
2. **Reconstruct** a validated NIR graph for an explicitly supported GPT-2,
   Llama, OPT or GPT-NeoX adapter, or report an unsupported architecture without
   guessing.

Import and reconstruction are reported independently. A complete tensor payload
does not imply a reconstructed graph, and a graph does not imply numerical
equivalence to the source framework.

## Supported sources

| Source | Support |
| --- | --- |
| Local Hugging Face directory (`config.json` + `model.safetensors`) | implemented |
| Sharded safetensors (`model.safetensors.index.json`) | implemented, with missing/extra/duplicate shard and tensor checks |
| Single `.safetensors` file | implemented, with optional adjacent `config.json`, tokenizer and model card |
| Multiple shards without an index | implemented with a warning; files are combined deterministically |
| Hugging Face repository id (`org/model`) | implemented through the optional `huggingface_hub` dependency (`uzen[hf]`); no remote code is executed; if the repository has no safetensors payload, a single `pytorch_model.bin` is loaded through `weights_only=True` |
| `.pt`, `.pth`, `.bin`, `.ckpt` | weights-only state dicts imported through the PyTorch adapter (`torch.load(..., weights_only=True)`); arbitrary pickled objects are refused; no graph reconstruction |
| `.onnx` | single-file ONNX models lifted into NIR with graph and weights stored; requires the `onnx` extra; external-data sidecars are refused so the recorded source hash covers every tensor |
| `.npz` / `.gguf` | not handled by this HF importer; the legacy weights-only artifact importer still supports them (no architecture reconstruction) |

## Unified artifact import

`client.import_weights(path)`, `uzen import PATH` and
`POST /v1/models/import` use the same content-addressed artifact pipeline:

* directories and `.safetensors` files go through the HF/safetensors importer;
* `.pt/.pth/.bin/.ckpt` state dicts go through the PyTorch adapter with
  `weights_only=True`; no pickle object is unpickled;
* single-file `.onnx` models are lifted into NIR and stored with their graph and
  weights;
* `.npz` and `.gguf` remain weights-only imports through the legacy adapter.

Every imported artifact records `metadata.import_report` with the source path,
format, source SHA-256, byte size, adapter, tensor count and whether a graph was
reconstructed. The PyTorch and ONNX import operations include the format and hash
in the artifact identity, so identical tensors imported from those different
formats do not collapse into one provenance record. ONNX graph metadata is stored
under `metadata.graph`; PyTorch state dicts do not carry a graph and are stored
as weights only.

For a Hub repository with no safetensors files, UZEN retries the download for a
single `pytorch_model.bin` and loads it with `torch.load(..., weights_only=True)`.
The fallback is recorded in the import warnings and in the file list; arbitrary
pickled objects are still refused. It requires PyTorch to be installed. This is
what makes the real `facebook/opt-125m` checkpoint usable while preserving the
no-remote-code rule.

Legacy byte-level BPE tokenizers are assembled from `vocab.json` + `merges.txt`
during import, including special-token ids from `tokenizer_config.json` and
`special_tokens_map.json`. The assembled document is stored by SHA-256 exactly
like a native `tokenizer.json`; no tokenizer code is executed. `tokenizer.model`
(SentencePiece) artifacts still have no text-inference adapter and are refused
explicitly.

## Compatibility report

A successful GPT-2 or Llama import reports:

```json
{
  "weights": "complete",
  "config": "parsed",
  "tokenizer": "present",
  "architecture": "recognized",
  "reconstruction": "validated",
  "execution": "passed",
  "overall": "validated"
}
```

For an architecture without an adapter, for example MPT:

```json
{
  "reconstruction": "unsupported",
  "execution": "not_run",
  "overall": "ready_for_reconstruction"
}
```

The reconstruction values are:

* `validated` — every required graph parameter was mapped, graph validation
  passed, and a forward pass produced finite logits with the expected shape.
* `partial` — the graph executes, but unexpected source tensors or other
  explicitly listed details remain unresolved. This is never hidden.
* `unsupported` — no adapter exists for the architecture, or required tensors /
  shapes cannot be mapped. No graph is guessed.
* `failed` — the adapter exists but graph validation or execution failed.
* `not_attempted` — configuration was missing/invalid, so reconstruction was not
  possible.

`overall` is `validated`, `validated_with_unresolved`, `ready_for_reconstruction`,
`weights_only`, `inspect_only`, or `unsupported`. The report also includes
`understanding.graph_reconstructed`, `understanding.execution_validated`, source
file hashes, repository id and requested revision, license evidence, tokenizer
details, role counts, a per-tensor inventory, a tensor mapping, and stage-by-stage
`progress`. Import failures carry the same report in the typed error's details.

## GPT-2 reconstruction

The adapter is intentionally narrow and explicit:

* `transformer.wte.weight` → `embed.weight`; `transformer.wpe.weight` →
  `pos_embed.weight`.
* GPT-2 Conv1D weights are transposed into NIR's `(out_features, in_features)`
  convention. The mapping is recorded per tensor.
* `transformer.h.N.attn.c_attn` is fused QKV and maps directly to the NIR
  `qkv`/`split_qkv` path.
* `ln_1`, `ln_2`, `c_proj`, `c_fc`, and `ln_f` map to the corresponding
  pre-LayerNorm transformer nodes.
* The LM head is tied to `wte` when the checkpoint/config says it is tied.
* `gelu_new` maps to the NIR tanh-approximation GELU; the report notes the
  correspondence.
* Training-only dropout settings are reported and omitted from inference.
* OpenAI's prefixless safetensors layout (`wte.weight`, `h.N...`) is normalized
  to the `transformer.` layout during reconstruction. The normalization is
  recorded in the reconstruction warnings and the tensor mapping.
* The 12 causal-mask buffers named `h.N.attn.bias` (or `attn.masked_bias`) are
  recorded as `ignored` mappings. They are not weights: causality is applied by
  the NIR attention operator. Other unexpected tensors remain `extra` and make
  the result `partial`.
* Every expected graph parameter must be present with the expected shape.
  Missing parameters are reported, never filled with invented values.
* The stored artifact uses the canonical NIR tensor names. The original imported
  package remains available on `ModelImportResult.package` for provenance and
  comparison; `artifact_package` is the canonical graph-bound package.

The repository's live check downloads
`hf-internal-testing/tiny-random-GPT2` (public, 454 KB of real safetensors) and
requires graph validation plus a finite `(1, 4, 1000)` logits forward pass:

```bash
UZEN_LIVE_HF=1 .venv/bin/python -m pytest tests/test_live_hf_import.py -q
```

This verifies tensor/config consistency and execution. A separate live check
compares NIR logits with `transformers.GPT2LMHeadModel` on fixed tokens and
requires `atol=1e-5`, `rtol=1e-5`. The measured max absolute error for
`hf-internal-testing/tiny-random-GPT2` is **2.09e-7** and the measured mean
absolute error is **2.65e-8**. The parity document records both shapes, the exact
tolerances, and the errors rather than treating a successful forward pass as
numerical parity.

### Validated real checkpoints

The following real Hub checkpoints were imported, reconstructed, executed with a
real tokenizer, and integrity-verified by
`examples/real_model_import_demo.py`:

| Checkpoint | Family | Source tensors | Canonical params | Ignored buffers | Result |
| --- | --- | ---: | ---: | ---: | --- |
| `distilgpt2` | GPT-2 | 77 | 77 | 6 | `validated`; forward `[1, 5, 50257]`; generation `' J.'` |
| `openai-community/gpt2` | GPT-2 | 149 | 149 | 12 | `validated`; forward `[1, 5, 50257]`; generation `' John.'` |
| `facebook/opt-125m` | OPT | 197 | 197 | 0 | `validated`; forward `[1, 5, 50272]`; generation `' Tanya'`; weights-only `.bin` fallback and legacy BPE tokenizer |
| `EleutherAI/pythia-14m` | GPT-NeoX | 76 | 76 | 0 | `validated`; forward `[1, 5, 50304]`; generation `':\n'` |
| `HuggingFaceTB/SmolLM-135M` | Llama | 273 | 273 | 0 | `validated`; forward `[1, 5, 49152]`; generation `' Alex.'` |

These generations use `--max-new-tokens 2` and greedy decoding; they are evidence
that the graph and tokenizer execute, not a quality benchmark. Reproduce the full
matrix with:

```bash
HF_HUB_DISABLE_XET=1 .venv/bin/python examples/real_model_import_demo.py \
  --models distilgpt2 openai-community/gpt2 facebook/opt-125m \
           EleutherAI/pythia-14m HuggingFaceTB/SmolLM-135M \
  --root /tmp/uzen-import-all --max-new-tokens 2
```

The report is written to `<root>/reports/real_model_import.json`. The demo also
accepts local checkpoint directories in `--models`. A prefixless synthetic
regression fixture runs in `tests/test_gpt2_reconstruction.py`, so this layout
is covered without network access.

### Validated sharded checkpoints and ONNX interchange

`examples/sharded_onnx_import_demo.py` validates the two import paths that the
matrix above does not cover. Real published `model.safetensors` weights are
re-split into balanced shards with UZEN's own safetensors writer and a real
`model.safetensors.index.json` (no genuinely sharded Hub repository of this size
is published; a source that already ships an index is imported as-is). Both
layouts are then imported, validated, executed and compared:

| Checkpoint | Shards | Indexed tensors | Index bytes | Result |
| --- | ---: | ---: | ---: | --- |
| `distilgpt2` | 4 | 82 | 6,080 | both layouts `validated`/`passed`; logits `[1, 5, 50257]`; generation `' J.J.'`; identical digests, logits and generated tokens |
| `openai-community/gpt2` | 5 | 160 | 9,928 | both layouts `validated`/`passed`; logits `[1, 5, 50257]`; generation `" John. I'm"`; identical digests, logits and generated tokens |

Single-file and sharded imports of the same tensors address one
content-addressed artifact (`fe1896b4…` for `distilgpt2`, `e4f734e1…` for
`openai-community/gpt2`), so the comparison uses each layout's own
`ModelImporter` output. The measured `max_abs_logit_error` is `0.0` on both
checkpoints. Removing a declared shard makes the import fail with
`[artifact.integrity] missing shard '…' declared in weight_map` and no package is
stored.

The second half of the demo exports GPT-2 layer-0's real MLP block
(`mlp_up` → `gelu` → `mlp_down`) with the pretrained weights to a single-file
ONNX model, validates it with `onnx.checker`, imports it back into NIR and
executes both graphs on the same token-derived activations:

* exported ops: `MatMul`, `Add`, `Gelu`, `MatMul`, `Add`; 5 ONNX nodes behind 4
  initializers, 18,890,262 bytes, opset `ai.onnx:20`;
* the re-imported NIR graph executes to `mlp_down:0` with
  `max_abs_error_vs_nir = 0.0` and `mean_abs_error_vs_nir = 0.0`;
* storing the file through the unified artifact import yields a validated `onnx`
  artifact (`ce7badae…` for `distilgpt2`, `2fc6dcf1…` for `openai-community/gpt2`)
  with the graph reconstructed (3 NIR nodes) and integrity verified.

The GELU exchange is deliberately symmetric: the exporter emits the standard
`Gelu(approximate="tanh")` operator instead of a Pow/Mul/Tanh chain, and the
importer maps exactly that form to NIR `gelu`. The Erf formulation
(`approximate="none"`) is refused rather than silently rewritten, so an imported
graph never claims a function it did not preserve. `onnxruntime` was not
installed in the verification environment; the ONNX file is therefore validated
by `onnx.checker` and executed through UZEN's NIR runtime, and the report records
the external-runtime check as `skipped` rather than passed.

```bash
HF_HUB_DISABLE_XET=1 .venv/bin/python examples/sharded_onnx_import_demo.py \
  --model distilgpt2 --root /tmp/uzen-sharded-onnx-demo --shards 4
HF_HUB_DISABLE_XET=1 .venv/bin/python examples/sharded_onnx_import_demo.py \
  --model openai-community/gpt2 --root /tmp/uzen-sharded-onnx-demo-gpt2 --shards 5
```

The report is written to `<root>/reports/sharded_onnx_import.json`. The command
exits non-zero if any gate fails, `--shards` and `--max-new-tokens` are
configurable, and a local checkpoint directory is accepted in place of a Hub id.

## Llama reconstruction

Llama uses a separate explicit graph and mapping; it is not approximated by the
GPT-2 transformer:

* Q, K, V and O are separate projections, contrary to GPT-2's fused Conv1D QKV.
* Q/K receive real rotary position embeddings (`rope`) using `positions` as a
graph input and the configured `rope_theta`. The graph is rotate-half, matching
  the Llama convention.
* `num_key_value_heads` may be smaller than `num_attention_heads`. K/V are
  repeated with the real `repeat_kv` operator before causal attention; the
  mapping preserves the smaller KV projection shapes rather than inventing
  expanded source tensors.
* Pre-attention and post-attention normalization use RMSNorm; the MLP is
  SwiGLU (`silu(gate(x)) * up(x)` followed by `down`).
* The final RMSNorm and LM head are explicit. Tied and untied LM heads are
  handled separately.
* `rope_scaling`, attention/MLP biases, non-SiLU activations and
  `pretraining_tp != 1` are rejected as unsupported instead of approximated.

A live check imports `hf-internal-testing/tiny-random-LlamaForCausalLM`
(21 tensors), reconstructs 53 nodes, validates them, and executes finite
`(1, 4, 32000)` logits. Its logits also match `transformers.LlamaForCausalLM`
at `atol=1e-5`, with a measured max absolute error of **8.94e-8**. The reusable
`compare_llama_logits()` helper accepts explicit `positions`, passes them to
both runtimes, and records the tokens, positions, shapes, tolerances and errors.
This is a validation of the explicit tiny fixture, not a claim of universal
Llama support.

## MCP progress

`uzen.import_model` accepts `params._meta.progressToken`. When the stdio server
is run, every import stage emits a real JSON-RPC
`notifications/progress` message before the final tool result, with an
incrementing `progress` value and a `message` containing the stage and status.
The report still contains its own `progress` array, so a client that ignores
notifications loses no diagnostic information.

## Guarantees and limits

* Source weights are never overwritten; imported arrays are copied into the
  immutable `WeightPackage` and the artifact store records provenance.
* Shard filenames are resolved inside the model directory; path traversal is
  rejected. Weight maps must match shard contents exactly.
* BF16 source tensors retain their source dtype label while being widened for the
  numerical substrate.
* Tokenizer files are parsed as data only. No tokenizer code, custom model code or
  pickle is executed. A valid `tokenizer.json` can later be loaded by the optional
  `tokenizers` runtime for text inference; other tokenizer formats are reported
  without being approximated. Legacy BPE documents that omit `model.type` but
  contain `vocab` and `merges` (including the `openai-community/gpt2` and
  `distilgpt2` tokenizer files) are accepted as BPE.
* PyTorch checkpoints are loaded only with `weights_only=True`; arbitrary pickled
  objects are refused with `format.unsupported` rather than unpickled.
* ONNX models with external tensor data are refused. Their sidecar files would not
  be covered by the main-file SHA-256, so incomplete provenance is not recorded as
  if it were complete.
* A license is reported when found in model-card frontmatter or `config.json`;
  otherwise it is explicitly `unknown`. Importing a checkpoint does not change or
  infer its license.
* Reconstruction currently supports explicit GPT-2, Llama, OPT, GPT-NeoX and
  GPT-J adapters plus the Llama-like profile adapters for Gemma (zero-centered
  RMSNorm and embedding scale), Mistral (sliding-window attention), Qwen2, Yi,
  Aquila, Baichuan, InternLM2, OLMo, MiniCPM and Orion. GPT-J uses a dedicated
  parallel-residual graph with interleaved partial RoPE. Reconstruction still
  requires the exact tensor names and config fields for the selected adapter;
  profiles are not guessed. Import profiles for StableLM (biased LayerNorm and
  GELU), GPT-Neo, Falcon, Bloom, MPT and Phi are recognized but report a
  concrete `unsupported` reason. OPT variants with `word_embed_proj_dim !=
  hidden_size`, post-LN OPT, scaled embeddings or non-ReLU activations, GPT-NeoX
  variants with scaled RoPE or unsupported activations, learned architecture
  variants, quantized formats, and arbitrary custom layers remain unsupported
  rather than approximated.
* Reference-logit parity is checked only in the gated live tests against tiny
  public fixtures. Llama, GPT-2, OPT, GPT-NeoX, Gemma, GPT-J, Qwen2, OLMo and Mistral have live parity;
  the other profile adapters are layout- and execution-tested but are not a
  guarantee for every checkpoint.
* Text inference requires a stored tokenizer document and the optional
  `uzen[tokenizer]` runtime. `vocab.json` + `merges.txt` legacy BPE is assembled
  at import; `tokenizer.model`-only SentencePiece artifacts are imported and
  reported, but text inference refuses them explicitly.
* The Milestone A pipeline (import, analyze, modify, export, reload, verify and
  compare real forward passes) is exercised end to end by the gated live tests
  for the supported adapters. It is not a claim of universal model support.
