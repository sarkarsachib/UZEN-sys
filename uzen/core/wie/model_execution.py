"""Execute stored reconstructed model artifacts.

An imported model artifact carries the validated NIR graph in its manifest.  This
module is the narrow execution seam for that graph: it reconstructs the graph,
binds the immutable package, supplies causal-LM token/position inputs, and runs a
real forward pass.  It never invents an architecture when the graph metadata is
missing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from uzen.core.nir.executor import execute
from uzen.core.nir.graph import Graph
from uzen.core.nws.tensors import WeightPackage
from uzen.core.wie.tokenization import (
    encode_text,
    encode_with_tokenizer,
    infer_eos_token_id,
    load_tokenizer,
    tokenizer_spec_from_manifest,
)
from uzen.errors import GraphValidationError, InvalidOperationParameters, TensorInvalid


@dataclass(frozen=True)
class ModelExecution:
    artifact_id: str
    graph_name: str
    inputs: dict[str, np.ndarray]
    outputs: dict[str, np.ndarray]
    num_ops: int
    intermediates: dict[str, np.ndarray] = field(default_factory=dict)

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
            "captured": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in self.intermediates.items()
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


@dataclass(frozen=True)
class ModelGeneration:
    artifact_id: str
    graph_name: str
    prompt: str
    generated_text: str
    full_text: str
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    full_token_ids: tuple[int, ...]
    generated_tokens: tuple[str, ...]
    steps: int
    stopped_reason: str
    eos_token_id: int | None
    max_new_tokens: int
    max_sequence_length: int | None

    def summary(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "graph_name": self.graph_name,
            "prompt": self.prompt,
            "generated_text": self.generated_text,
            "full_text": self.full_text,
            "prompt_token_ids": list(self.prompt_token_ids),
            "generated_token_ids": list(self.generated_token_ids),
            "full_token_ids": list(self.full_token_ids),
            "generated_tokens": list(self.generated_tokens),
            "steps": self.steps,
            "stopped_reason": self.stopped_reason,
            "eos_token_id": self.eos_token_id,
            "max_new_tokens": self.max_new_tokens,
            "max_sequence_length": self.max_sequence_length,
        }


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


_LAYER_NODE = re.compile(r"^l(\d+)_res2$")


def layer_node_ids(graph: Graph) -> list[str]:
    """Return the `l<N>_res2` block outputs in numeric layer order."""
    layers: list[tuple[int, str]] = []
    for node_id, node in graph.nodes.items():
        match = _LAYER_NODE.match(node_id)
        if match is not None and node.op == "add":
            layers.append((int(match.group(1)), node_id))
    return [node_id for _, node_id in sorted(layers)]


def _resolve_capture(graph: Graph, capture: Sequence[str] | str | None) -> list[str]:
    if capture is None:
        return []
    if isinstance(capture, str):
        requested = layer_node_ids(graph) if capture == "layers" else [capture]
    elif isinstance(capture, Sequence):
        requested = list(capture)
        if requested == ["layers"]:
            requested = layer_node_ids(graph)
    else:
        raise InvalidOperationParameters(
            "capture must be 'layers', a node id, or a sequence of NIR node ids"
        )
    if not requested:
        raise InvalidOperationParameters(
            "capture requested no nodes; use 'layers' or explicit NIR node ids"
        )
    if any(not isinstance(node_id, str) for node_id in requested):
        raise InvalidOperationParameters("capture node ids must be strings")
    missing = [node_id for node_id in dict.fromkeys(requested) if node_id not in graph.nodes]
    if missing:
        raise InvalidOperationParameters(
            "capture requested unknown NIR node ids: " + ", ".join(missing)
        )
    return list(dict.fromkeys(requested))


def execute_stored_model(
    package: WeightPackage,
    manifest: Mapping[str, Any],
    *,
    artifact_id: str,
    tokens: Sequence[Sequence[int]],
    positions: Sequence[Sequence[int]] | Sequence[int] | None = None,
    capture: Sequence[str] | str | None = None,
) -> ModelExecution:
    graph = graph_from_manifest(manifest)
    inputs = _causal_inputs(graph, tokens, positions)
    capture_ids = _resolve_capture(graph, capture)
    validation = graph.validate(package)
    if not validation.ok:
        raise GraphValidationError(
            "stored graph does not validate against the artifact package",
            errors=list(validation.errors),
        )
    result = execute(
        graph, inputs, package, capture=capture_ids, validate_types=True
    )
    return ModelExecution(
        artifact_id=artifact_id,
        graph_name=graph.name,
        inputs=inputs,
        outputs=dict(result.outputs),
        intermediates=dict(result.intermediates),
        num_ops=result.num_ops,
    )


def _max_sequence_length(graph: Graph, package: WeightPackage) -> int | None:
    spec = graph.metadata.get("spec")
    if isinstance(spec, Mapping):
        value = spec.get("max_seq_len")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    for node in graph.nodes.values():
        if node.op != "pos_embedding":
            continue
        for name in node.param_names():
            if name in package:
                shape = package.shape_of(name)
                if shape:
                    return int(shape[0])
    return None


def _position_row(
    positions: Sequence[Sequence[int]] | Sequence[int] | None,
    sequence_length: int,
) -> list[int] | None:
    if positions is None:
        return None
    array = np.asarray(positions, dtype=np.int64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape != (1, sequence_length):
        raise InvalidOperationParameters(
            f"generation positions must have shape (1, {sequence_length}), got {array.shape}"
        )
    return [int(value) for value in array[0]]


def greedy_generate(
    package: WeightPackage,
    manifest: Mapping[str, Any],
    *,
    root: str | Path,
    artifact_id: str,
    text: str,
    max_new_tokens: int = 16,
    eos_token_id: int | None = None,
    positions: Sequence[Sequence[int]] | Sequence[int] | None = None,
) -> ModelGeneration:
    """Generate tokens greedily from repeated real forward passes.

    The graph is executed once per generated token. This is deliberately simple:
    there is no KV cache and no sampling. Generation stops on EOS, the requested
    token budget, or the graph's declared maximum sequence length.
    """
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise InvalidOperationParameters("max_new_tokens must be a positive integer")
    if eos_token_id is not None and (
        isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int) or eos_token_id < 0
    ):
        raise InvalidOperationParameters("eos_token_id must be a non-negative integer")

    graph = graph_from_manifest(manifest)
    spec = tokenizer_spec_from_manifest(manifest)
    tokenizer = load_tokenizer(root, spec)
    encoded = encode_with_tokenizer(tokenizer, text)
    prompt_ids = tuple(encoded.token_ids)
    ids = list(prompt_ids)
    position_row = _position_row(positions, len(prompt_ids))
    resolved_eos = eos_token_id if eos_token_id is not None else infer_eos_token_id(tokenizer)
    max_sequence_length = _max_sequence_length(graph, package)
    if max_sequence_length is not None and len(ids) > max_sequence_length:
        raise InvalidOperationParameters(
            f"prompt has {len(ids)} tokens but the model supports at most "
            f"{max_sequence_length} positions"
        )

    generated: list[int] = []
    stopped_reason = "max_new_tokens"
    while len(generated) < max_new_tokens:
        if max_sequence_length is not None and len(ids) >= max_sequence_length:
            stopped_reason = "max_sequence_length"
            break
        if (
            position_row is not None
            and max_sequence_length is not None
            and position_row[-1] + 1 >= max_sequence_length
        ):
            stopped_reason = "max_sequence_length"
            break
        execution = execute_stored_model(
            package,
            manifest,
            artifact_id=artifact_id,
            tokens=[ids],
            positions=[position_row] if position_row is not None else None,
        )
        logits = execution.output("logits")
        if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != len(ids):
            raise InvalidOperationParameters(
                "logits must have shape (1, sequence, vocabulary), got "
                f"{list(logits.shape)}"
            )
        last_logits = logits[0, -1, :]
        if not np.isfinite(last_logits).all():
            raise TensorInvalid("greedy generation received non-finite logits")
        next_token = int(np.argmax(last_logits))
        ids.append(next_token)
        generated.append(next_token)
        if position_row is not None:
            position_row.append(position_row[-1] + 1)
        if resolved_eos is not None and next_token == resolved_eos:
            stopped_reason = "eos_token"
            break

    generated_tokens = tuple(str(tokenizer.id_to_token(token) or "") for token in generated)
    return ModelGeneration(
        artifact_id=artifact_id,
        graph_name=graph.name,
        prompt=encoded.text,
        generated_text=tokenizer.decode(generated, skip_special_tokens=True),
        full_text=tokenizer.decode(ids, skip_special_tokens=True),
        prompt_token_ids=prompt_ids,
        generated_token_ids=tuple(generated),
        full_token_ids=tuple(ids),
        generated_tokens=generated_tokens,
        steps=len(generated),
        stopped_reason=stopped_reason,
        eos_token_id=resolved_eos,
        max_new_tokens=max_new_tokens,
        max_sequence_length=max_sequence_length,
    )


def execute_stored_text(
    package: WeightPackage,
    manifest: Mapping[str, Any],
    *,
    root: str | Path,
    artifact_id: str,
    text: str,
    positions: Sequence[Sequence[int]] | Sequence[int] | None = None,
    capture: Sequence[str] | str | None = None,
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
        capture=capture,
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
