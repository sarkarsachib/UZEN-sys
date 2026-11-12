"""Execute stored reconstructed model artifacts.

An imported model artifact carries the validated NIR graph in its manifest.  This
module is the narrow execution seam for that graph: it reconstructs the graph,
binds the immutable package, supplies causal-LM token/position inputs, and runs a
real forward pass.  It never invents an architecture when the graph metadata is
missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from uzen.core.nir.executor import execute
from uzen.core.nir.graph import Graph
from uzen.core.nws.tensors import WeightPackage
from uzen.core.wie.tokenization import encode_text, tokenizer_spec_from_manifest
from uzen.errors import GraphValidationError, InvalidOperationParameters


@dataclass(frozen=True)
class ModelExecution:
    artifact_id: str
    graph_name: str
    inputs: dict[str, np.ndarray]
    outputs: dict[str, np.ndarray]
    num_ops: int

    def output(self, name: str = "logits") -> np.ndarray:
        try:
            return self.outputs[name]
        except KeyError as exc:
            raise InvalidOperationParameters(
                f"model execution has no output '{name}'; available: {sorted(self.outputs)}"
            ) from exc

    def summary(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "graph_name": self.graph_name,
            "inputs": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in self.inputs.items()
            },
            "outputs": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in self.outputs.items()
            },
            "num_ops": self.num_ops,
        }


@dataclass(frozen=True)
class ModelTextExecution:
    text: str
    decoded_text: str
    token_ids: tuple[int, ...]
    tokens: tuple[str, ...]
    execution: ModelExecution


def graph_from_manifest(manifest: Mapping[str, Any]) -> Graph:
    """Return the stored graph or fail explicitly; never synthesize a substitute."""
    metadata = manifest.get("metadata") or {}
    graph_doc = metadata.get("graph") if isinstance(metadata, Mapping) else None
    if not graph_doc:
        raise InvalidOperationParameters(
            "artifact has no stored NIR graph; import/reconstruct a supported model "
            "before executing it"
        )
    try:
        return Graph.from_dict(graph_doc)
    except Exception as exc:
        raise GraphValidationError(f"stored NIR graph could not be restored: {exc}") from exc


def _causal_inputs(
    graph: Graph,
    tokens: Sequence[Sequence[int]],
    positions: Sequence[Sequence[int]] | Sequence[int] | None,
) -> dict[str, np.ndarray]:
    token_array = np.asarray(tokens, dtype=np.int64)
    if token_array.ndim != 2:
        raise InvalidOperationParameters("tokens must have shape (batch, sequence)")
    if token_array.size == 0:
        raise InvalidOperationParameters("tokens must not be empty")
    if "tokens" not in graph.inputs:
        raise InvalidOperationParameters(
            f"stored graph has no 'tokens' input; inputs are {sorted(graph.inputs)}"
        )
    inputs: dict[str, np.ndarray] = {"tokens": token_array}
    if "positions" in graph.inputs:
        batch, seq = token_array.shape
        if positions is None:
            position_array = np.tile(np.arange(seq, dtype=np.int64), (batch, 1))
        else:
            position_array = np.asarray(positions, dtype=np.int64)
            if position_array.ndim == 1:
                position_array = position_array.reshape(1, -1)
            if position_array.ndim != 2:
                raise InvalidOperationParameters("positions must be rank 1 or 2")
            if position_array.shape[0] == 1 and batch > 1:
                position_array = np.repeat(position_array, batch, axis=0)
            if position_array.shape != (batch, seq):
                raise InvalidOperationParameters(
                    f"positions must have shape {(batch, seq)}, got {position_array.shape}"
                )
        inputs["positions"] = position_array
    extra = sorted(set(graph.inputs) - set(inputs))
    if extra:
        raise InvalidOperationParameters(
            "stored graph requires inputs this causal-LM runner does not provide: "
            + ", ".join(extra)
        )
    return inputs


def execute_stored_model(
    package: WeightPackage,
    manifest: Mapping[str, Any],
    *,
    artifact_id: str,
    tokens: Sequence[Sequence[int]],
    positions: Sequence[Sequence[int]] | Sequence[int] | None = None,
) -> ModelExecution:
    graph = graph_from_manifest(manifest)
    inputs = _causal_inputs(graph, tokens, positions)
    validation = graph.validate(package)
    if not validation.ok:
        raise GraphValidationError(
            "stored graph does not validate against the artifact package",
            errors=list(validation.errors),
        )
    result = execute(graph, inputs, package, validate_types=True)
    return ModelExecution(
        artifact_id=artifact_id,
        graph_name=graph.name,
        inputs=inputs,
        outputs=dict(result.outputs),
        num_ops=result.num_ops,
    )


def execute_stored_text(
    package: WeightPackage,
    manifest: Mapping[str, Any],
    *,
    root: str | Path,
    artifact_id: str,
    text: str,
    positions: Sequence[Sequence[int]] | Sequence[int] | None = None,
) -> ModelTextExecution:
    """Tokenize text with the artifact's stored tokenizer, then execute its graph."""
    spec = tokenizer_spec_from_manifest(manifest)
    encoded = encode_text(root, spec, text)
    execution = execute_stored_model(
        package,
        manifest,
        artifact_id=artifact_id,
        tokens=[encoded.token_ids],
        positions=positions,
    )
    return ModelTextExecution(
        text=encoded.text,
        decoded_text=encoded.decoded_text,
        token_ids=encoded.token_ids,
        tokens=encoded.tokens,
        execution=execution,
    )


def compare_executions(
    source: ModelExecution,
    candidate: ModelExecution,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    """Compare two executions on identical inputs with explicit tolerances."""
    if atol < 0 or rtol < 0:
        raise InvalidOperationParameters("atol and rtol must be non-negative")
    source_names = set(source.outputs)
    candidate_names = set(candidate.outputs)
    if source_names != candidate_names:
        return {
            "passed": False,
            "error": "output names differ",
            "source_outputs": sorted(source_names),
            "candidate_outputs": sorted(candidate_names),
            "tolerances": {"atol": atol, "rtol": rtol},
        }

    outputs: dict[str, dict[str, Any]] = {}
    all_passed = True
    for name in sorted(source_names):
        actual = source.outputs[name]
        other = candidate.outputs[name]
        if actual.shape != other.shape:
            outputs[name] = {
                "passed": False,
                "error": "shape mismatch",
                "source_shape": list(actual.shape),
                "candidate_shape": list(other.shape),
            }
            all_passed = False
            continue
        difference = np.abs(actual.astype(np.float64) - other.astype(np.float64))
        relative = difference / np.maximum(np.abs(actual.astype(np.float64)), 1e-8)
        passed = bool(np.allclose(actual, other, atol=atol, rtol=rtol))
        outputs[name] = {
            "passed": passed,
            "shape": list(actual.shape),
            "dtype": str(actual.dtype),
            "max_abs_error": float(difference.max()) if difference.size else 0.0,
            "mean_abs_error": float(difference.mean()) if difference.size else 0.0,
            "max_rel_error": float(relative.max()) if relative.size else 0.0,
            "num_elements": int(difference.size),
        }
        all_passed = all_passed and passed
    return {
        "passed": all_passed,
        "tolerances": {"atol": atol, "rtol": rtol},
        "source_artifact_id": source.artifact_id,
        "candidate_artifact_id": candidate.artifact_id,
        "inputs": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in source.inputs.items()
        },
        "outputs": outputs,
    }
