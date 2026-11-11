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

The result contains the graph name, input/output shapes and dtypes, and the
operation count. `include_values=True` additionally returns the output arrays.
An artifact without a stored graph is refused with
`operation.invalid_parameters`; no architecture is guessed.

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

* `forward_model(...)`
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
* There is no tokenizer execution yet; callers supply token IDs.
* There is no KV cache; each forward pass recomputes the supplied sequence.
* A modification that changes parameter structure is not comparable as the same
  model and is refused by `compare_models` rather than coerced.
* Numerical closeness is evidence for the stated inputs and tolerances, not a
  claim of semantic equivalence.
