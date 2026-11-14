# P2 — Reconstructed model execution and modification validation

P1 imports and reconstructs GPT-2/Llama checkpoints. P2 makes the resulting
artifact a usable model object rather than an import report:

1. **Execute** the validated graph stored in an imported artifact.
2. **Modify** its immutable weight package through the existing registered
   operations.
3. **Compare** the source and modified artifacts on identical inputs with
   recorded tolerances.

No training, optimization or gradient update is involved.

## Stored graph execution

An imported model artifact stores its validated NIR graph in the manifest under
`metadata.graph`. `execute_stored_model()` restores that graph, binds the
artifact package, supplies token and (for Llama) position inputs, validates the
graph against the package, and runs a real forward pass.

```python
client = Uzen(root="./uzen-workspace")
imported = client.import_model("./tiny-model")
result = client.forward_model(
    imported["artifact_id"],
    tokens=[[0, 1, 2, 3]],
    include_values=False,
)
```

The result contains the graph name, input/output shapes and dtypes, the
operation count, and a `captured` map. `include_values=True` additionally returns
the output and captured arrays. An artifact without a stored graph is refused
with `operation.invalid_parameters`; no architecture is guessed.

## Per-layer activation capture

Pass `capture=` to `forward_model()` or `forward_text()` to capture NIR node
outputs during the same forward pass:

```python
execution = client.forward_model(
    imported["artifact_id"],
    tokens=[[0, 1, 2, 3]],
    capture="layers",
    include_values=True,
)
execution["captured"]["l0_res2"]["shape"]   # [1, 4, d_model]
execution["captured"]["l0_res2"]["values"]  # present with include_values=True
```

`capture="layers"` resolves the reconstructed transformer block outputs named
`l<N>_res2`, in numeric layer order. An explicit list of NIR node ids is also
accepted. Unknown nodes raise `operation.invalid_parameters` rather than
returning an empty capture. The same option is available through the REST
forward routes (`"capture": ["layers"]`), the CLI (`--capture layers`), and the
MCP forward tools.

## Layer-by-layer Hugging Face parity

The optional parity helpers compare every reconstructed block output with the
corresponding Hugging Face transformer block, not just the final logits:

```python
from uzen.core.wie.parity import compare_gpt2_activations, compare_llama_activations

report = compare_gpt2_activations(graph, package, local_checkpoint, tokens=[[0, 1, 2, 3]])
report["passed"]
report["max_abs_error"]
report["layers"][0]["node"]          # l0_res2
report["layers"][0]["max_abs_error"]
```

GPT-2 uses `transformers.GPT2LMHeadModel.transformer.h` and Llama uses
`transformers.LlamaForCausalLM.model.layers`. Forward hooks capture each block's
real output; the comparison does not depend on the version-specific
`hidden_states` tuple layout. Each layer records its node id, shape, tolerance
result, and absolute/relative errors. The gated live suite compares both tiny
Hub fixtures layer by layer.

## Tokenizer-backed text inference

An import that finds a valid `tokenizer.json` stores it in a content-addressed
sidecar under `<workspace>/tokenizers/<sha256>/tokenizer.json` and records the
SHA-256, model type and vocabulary size in the artifact manifest. The sidecar is
independent of the original model directory, so inference still works after the
source directory is moved or removed. Imports with different tokenizer documents
receive different artifact ids even when their tensors are identical.

```python
text_result = client.forward_text(
    imported["artifact_id"],
    text="hello world",
    include_values=True,
)
text_result["token_ids"]
text_result["tokens"]
text_result["decoded_text"]
text_result["outputs"]["logits"]
```

Text is encoded by the real Hugging Face `tokenizers` runtime, not a toy
substitute. Install the optional extra with `pip install 'uzen[tokenizer]'`.
If it is absent, text inference raises `runtime.unavailable`; token-ID execution
continues to work. GPT-2 byte-level BPE and Llama SentencePiece-style BPE are
handled by the same `tokenizer.json` path, including special tokens and the
tokenizer's own decoder. Weight-only modifications copy the tokenizer reference,
so modified artifacts keep text inference.

The REST endpoint is `POST /v1/models/{artifact_id}/forward-text`, the CLI form is
`uzen forward ARTIFACT_ID --text 'hello world'`, and MCP exposes
`uzen.forward_text`.

## Modifying a real model

Weight-only modifications preserve the parent graph metadata and provenance
when the modified package is stored. This makes this sequence valid:

```python
modified = client.modify(
    imported["artifact_id"],
    "interpolate",
    params={"t": 0.5},
    others=[imported["artifact_id"]],
)
comparison = client.compare_models(
    imported["artifact_id"],
    modified["artifact_id"],
    tokens=[[0, 1, 2, 3]],
    atol=1e-5,
    rtol=1e-5,
)
```

`compare_models` first verifies that both artifacts carry the same stored NIR
graph. It then executes both packages on exactly the same tokens/positions and
reports, per output:

* `passed`
* shape and dtype
* `max_abs_error`
* `mean_abs_error`
* `max_rel_error`
* `num_elements`

The top-level result contains `passed`, `tolerances`, source/candidate artifact
IDs, and the input shapes. A shape or output-name mismatch is reported, never
silently coerced.

Parent IDs are deduplicated, so interpolating a model with itself records the
source once in lineage.

## Interfaces

REST:

* `POST /v1/models/{artifact_id}/forward`
* `POST /v1/models/compare`

Python SDK:

* `forward_model(..., capture="layers")`
* `compare_models(...)`

CLI:

```bash
uzen forward ARTIFACT_ID --tokens '[[0,1,2,3]]'
uzen compare-models SOURCE CANDIDATE --tokens '[[0,1,2,3]]'
```

MCP:

* `uzen.forward_model`
* `uzen.compare_models`

## Limits

* Execution is for causal-LM graphs with `tokens` and optional `positions`
  inputs, which are the graphs produced by the current GPT-2/Llama adapters.
* Text inference requires `tokenizer.json` plus the optional `tokenizers` runtime;
  `vocab.json`/`merges.txt`-only and `tokenizer.model`-only imports are refused.
* There is no KV cache; each forward pass recomputes the supplied sequence.
* Text inference runs a forward pass and returns logits; autoregressive text
  generation/sampling is not implemented.
* A modification that changes parameter structure is not comparable as the same
  model and is refused by `compare_models` rather than coerced.
* Numerical closeness is evidence for the stated inputs and tolerances, not a
  claim of semantic equivalence.
* Layer parity requires a locally available Hugging Face checkpoint plus
  `transformers` and `torch`; it is validated only for the explicit GPT-2/Llama
  adapters.
