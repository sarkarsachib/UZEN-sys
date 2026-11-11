# P1 — Hugging Face / safetensors import

P1 imports checkpoint **weights and metadata**. It does not reconstruct an NIR
graph, does not execute a forward pass and does not claim model equivalence.

## Supported sources

| Source | Support |
| --- | --- |
| Local Hugging Face directory (`config.json` + `model.safetensors`) | implemented |
| Sharded safetensors (`model.safetensors.index.json`) | implemented, with missing/extra/duplicate shard and tensor checks |
| Single `.safetensors` file | implemented, with optional adjacent `config.json`, tokenizer and model card |
| Multiple shards without an index | implemented with a warning; files are combined deterministically |
| Hugging Face repository id (`org/model`) | implemented through the optional `huggingface_hub` dependency (`uzen[hf]`); no remote code is executed |
| `.npz` / `.gguf` | not handled by this importer; the existing legacy artifact importer still supports them |
| PyTorch pickle checkpoints | unsupported by design; pickles are never loaded |

## Compatibility report

Every result separates independent claims:

```json
{
  "weights": "complete",
  "config": "parsed",
  "tokenizer": "present",
  "architecture": "recognized",
  "reconstruction": "not_implemented",
  "execution": "not_run",
  "overall": "ready_for_reconstruction"
}
```

`overall` is only a classification of the import stage:

* `ready_for_reconstruction` — weights are complete, configuration was parsed,
  the architecture family was recognized and a tokenizer vocabulary file exists.
  This does **not** mean a graph was reconstructed.
* `weights_only` — the tensor payload imported, but configuration, tokenizer or
  architecture recognition is incomplete.
* `inspect_only` — metadata is available but the weight payload is not complete.
* `unsupported` — a hard import problem (for example a missing declared shard)
  prevented an artifact from being published.

The report also includes `understanding.graph_reconstructed: false` and
`understanding.execution_validated: false`, source file hashes, repository id and
requested revision, license evidence, tokenizer details, role counts and a
per-tensor inventory. Import failures carry the same report in the typed error's
details.

## Guarantees and limits

* Source weights are never overwritten; imported arrays are copied into the
  immutable `WeightPackage` and the artifact store records provenance.
* Shard filenames are resolved inside the model directory; path traversal is
  rejected. Weight maps must match shard contents exactly.
* BF16 source tensors retain their source dtype label while being widened for the
  numerical substrate.
* Tokenizer files are parsed as data only. No tokenizer code, custom model code or
  pickle is executed.
* A license is reported when found in model-card frontmatter or `config.json`;
  otherwise it is explicitly `unknown`. Importing a checkpoint does not change or
  infer its license.
* Graph reconstruction, tokenizer execution and pretrained forward-pass
  validation remain future work. Milestone A is not satisfied by P1.
