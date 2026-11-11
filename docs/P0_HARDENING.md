# P0 engineering hardening

This document records the Phase 1/2 work from the implementation roadmap.

## Baseline

Before the hardening changes, the restored environment (core, API, workers, ONNX,
safetensors and CPU torch extras installed) ran:

```text
.venv/bin/python -m pytest -q
206 passed
```

No skips remained once the optional extras were installed. `torch 2.15.0+cpu`
reported `cuda=False`; no CUDA capability is claimed. After the hardening tests
were added, the suite reported **237 passed**.

## Enforced weight immutability

`WeightPackage` now enforces what the documentation previously promised:

* input arrays are copied into read-only, byte-backed buffers; write assignment
  and `setflags(write=True)` both fail;
* `tensors`, `metas`, `family_spec` and `metadata` are read-only mappings;
* attributes cannot be reassigned after construction;
* a value fingerprint is captured at construction and `assert_unmodified()` is
  checked by `ArtifactStore.put/get/export`, `Registry.invoke` and NIR `execute`;
* transformations must return a new `WeightPackage` (`with_tensor`, `without`,
  `select`, `copy`), never mutate the source.

Any code that previously assigned into `pkg.metas` or `pkg.tensors` now builds a
new package. That includes architecture initialization and NPA alignment.

## Artifact integrity

Reloading and exporting now verify, before returning bytes:

* manifest checksum and declared artifact id;
* tensor payload digest against the manifest summary;
* embedded safetensors metadata;
* declared tensor names, shapes and dtypes, including undeclared tensors;
* finite values;
* lineage operation and parent-manifest checksums;
* policy records for denied or prohibited effects.

A tampered manifest, payload or parent fails with `ArtifactIntegrityError` instead
of producing a trusted package.

## Resource protection

* `ExecutionLimits` provides operation, per-tensor byte, total working-set byte,
  timeout and cancellation-event budgets. Checks are cooperative at node
  boundaries; a running NumPy kernel cannot be preempted.
* `ArtifactStore` accepts `max_artifact_bytes` or `UZEN_MAX_ARTIFACT_BYTES` and
  refuses oversized packages before publication.
* Artifact publication remains atomic and immutable.

## Numerical safety

Added tests for NaN/Infinity, zero-rank scalars, empty tensors, subnormal
underflow, float16 extremes, 1,000,000-element dimensions and shape mismatches.

## Packaging, licensing and CI

* `pyproject.toml` now declares MIT, matching the repository license; `LICENSE`
  contains the MIT text.
* `.github/workflows/ci.yml` runs the core matrix (3.10–3.12), a focused Ruff
gate (undefined names / syntax / invalid comparisons), optional-runtime tests,
  syntax/import checks and a wheel build/install.
* `docs/STRICT_PLUS_AUDIT.md` maps every mutation route to its enforcement point
  and tests.

## Not done in this phase

At the time of this P0 work, P1–P7 remained. P1's Hugging Face/safetensors import
was subsequently implemented (see `docs/P1_IMPORT.md`); architecture reconstruction,
pretrained forward-pass validation, end-to-end real-model weight modification,
WGE/WME/NPA expansion, native/CUDA/distributed runtimes and full API/SDK/MCP
coverage remain future work. The roadmap's Milestone A is not yet satisfied.
