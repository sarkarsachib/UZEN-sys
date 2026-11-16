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

from uzen.core.nir.executor import KVCache, execute
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
    quantization: dict[str, Any] | None = None

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
            "quantization": self.quantization,
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
    strategy: str = "greedy"
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    seed: int | None = None
    num_beams: int = 1
    quantization: dict[str, Any] | None = None

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
            "strategy": self.strategy,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "seed": self.seed,
            "num_beams": self.num_beams,
            "quantization": self.quantization,
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


def _stored_quantization(graph: Graph) -> dict[str, Any] | None:
    report = graph.metadata.get("quantization")
    return dict(report) if isinstance(report, Mapping) else None


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
    quantize: str | None = None,
    cache: KVCache | None = None,
) -> ModelExecution:
    graph = graph_from_manifest(manifest)
    inputs = _causal_inputs(graph, tokens, positions)
    capture_ids = _resolve_capture(graph, capture)
    stored_quantization = _stored_quantization(graph)
    quantization: dict[str, Any] | None = stored_quantization
    execution_graph = graph
    execution_package = package
    if quantize is not None:
        if quantize != "int8":
            raise InvalidOperationParameters("quantize must be 'int8' when provided")
        if stored_quantization is None:
            from uzen.core.wie.quantization import quantize_int8

            quantized = quantize_int8(graph, package)
            execution_graph = quantized.graph
            execution_package = quantized.package
            quantization = quantized.report.to_dict()
    validation = execution_graph.validate(execution_package)
    if not validation.ok:
        raise GraphValidationError(
            "stored graph does not validate against the artifact package",
            errors=list(validation.errors),
        )
    result = execute(
        execution_graph,
        inputs,
        execution_package,
        capture=capture_ids,
        validate_types=True,
        cache=cache,
    )
    return ModelExecution(
        artifact_id=artifact_id,
        graph_name=execution_graph.name,
        inputs=inputs,
        outputs=dict(result.outputs),
        intermediates=dict(result.intermediates),
        num_ops=result.num_ops,
        quantization=quantization,
    )


def _supports_kv_cache(graph: Graph) -> bool:
    """Cache only decoder-only graphs whose positions are carried by RoPE or inputs."""
    saw_attention = False
    for node in graph.nodes.values():
        if node.op == "pos_embedding":
            return False
        if node.op == "attention":
            saw_attention = True
            if not bool(node.params.get("causal", True)):
                return False
    return saw_attention


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


_GENERATION_STRATEGIES = frozenset({"greedy", "sample", "beam"})


def _validate_generation_options(
    *,
    strategy: str,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    seed: int | None,
    num_beams: int,
) -> None:
    if strategy not in _GENERATION_STRATEGIES:
        raise InvalidOperationParameters(
            f"generation strategy must be one of {sorted(_GENERATION_STRATEGIES)}"
        )
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not np.isfinite(temperature)
        or temperature < 0
    ):
        raise InvalidOperationParameters("temperature must be a finite non-negative number")
    if strategy == "sample" and temperature <= 0:
        raise InvalidOperationParameters("sampling temperature must be greater than zero")
    if top_k is not None and (
        isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1
    ):
        raise InvalidOperationParameters("top_k must be a positive integer")
    if top_p is not None and (
        isinstance(top_p, bool)
        or not isinstance(top_p, (int, float))
        or not np.isfinite(top_p)
        or top_p <= 0
        or top_p > 1
    ):
        raise InvalidOperationParameters("top_p must be in (0, 1]")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
    ):
        raise InvalidOperationParameters("seed must be a non-negative integer")
    if isinstance(num_beams, bool) or not isinstance(num_beams, int) or num_beams < 1:
        raise InvalidOperationParameters("num_beams must be a positive integer")
    if strategy != "beam" and num_beams != 1:
        raise InvalidOperationParameters("num_beams greater than one requires strategy='beam'")


def _select_next_token(
    logits: np.ndarray,
    *,
    strategy: str,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    rng: np.random.Generator,
) -> int:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise InvalidOperationParameters("next-token logits must be a non-empty vector")
    if not np.isfinite(values).all():
        raise TensorInvalid("generation received non-finite logits")
    if strategy != "sample":
        return int(np.argmax(values))

    scaled = values / float(temperature)
    if top_k is not None:
        k = min(int(top_k), scaled.size)
        cutoff = np.partition(scaled, scaled.size - k)[scaled.size - k]
        scaled = np.where(scaled < cutoff, -np.inf, scaled)
    shifted = scaled - np.max(scaled)
    probs = np.exp(shifted)
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0:
        raise TensorInvalid("sampling probabilities are not finite")
    probs /= total
    if top_p is not None:
        order = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[order])
        keep_count = int(np.searchsorted(cumulative, float(top_p), side="left")) + 1
        keep = order[: min(keep_count, order.size)]
        filtered = np.zeros_like(probs)
        filtered[keep] = probs[keep]
        filtered_total = float(filtered.sum())
        if not np.isfinite(filtered_total) or filtered_total <= 0:
            raise TensorInvalid("top-p sampling left no probability mass")
        probs = filtered / filtered_total
    return int(rng.choice(probs.size, p=probs))


def _beam_generate(
    package: WeightPackage,
    manifest: Mapping[str, Any],
    *,
    artifact_id: str,
    prompt_ids: Sequence[int],
    base_positions: Sequence[int] | None,
    max_new_tokens: int,
    max_sequence_length: int | None,
    eos_token_id: int | None,
    num_beams: int,
) -> tuple[list[int], list[int], str]:
    """Score complete beam hypotheses by cumulative log-probability."""

    @dataclass
    class Beam:
        ids: list[int]
        positions: list[int] | None
        generated: list[int]
        score: float
        finished: bool = False
        reason: str | None = None

    beams = [
        Beam(
            ids=list(prompt_ids),
            positions=list(base_positions) if base_positions is not None else None,
            generated=[],
            score=0.0,
        )
    ]
    for _ in range(max_new_tokens):
        candidates: list[Beam] = []
        for beam in beams:
            if beam.finished:
                candidates.append(beam)
                continue
            if max_sequence_length is not None and (
                len(beam.ids) >= max_sequence_length
                or (
                    beam.positions is not None
                    and beam.positions[-1] + 1 >= max_sequence_length
                )
            ):
                beam.finished = True
                beam.reason = "max_sequence_length"
                candidates.append(beam)
                continue
            execution = execute_stored_model(
                package,
                manifest,
                artifact_id=artifact_id,
                tokens=[beam.ids],
                positions=[beam.positions] if beam.positions is not None else None,
            )
            logits = execution.output("logits")
            if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != len(beam.ids):
                raise InvalidOperationParameters(
                    "beam logits must have shape (1, sequence, vocabulary), got "
                    f"{list(logits.shape)}"
                )
            last = np.asarray(logits[0, -1, :], dtype=np.float64)
            if not np.isfinite(last).all():
                raise TensorInvalid("beam generation received non-finite logits")
            shifted = last - np.max(last)
            log_probs = shifted - np.log(np.exp(shifted).sum())
            width = min(num_beams, log_probs.size)
            for token_index in np.argsort(log_probs)[-width:][::-1]:
                token = int(token_index)
                new_positions = None
                if beam.positions is not None:
                    new_positions = beam.positions + [beam.positions[-1] + 1]
                finished = eos_token_id is not None and token == eos_token_id
                candidates.append(
                    Beam(
                        ids=beam.ids + [token],
                        positions=new_positions,
                        generated=beam.generated + [token],
                        score=beam.score + float(log_probs[token_index]),
                        finished=finished,
                        reason="eos_token" if finished else None,
                    )
                )
        if not candidates:
            break
        candidates.sort(key=lambda item: (-item.score, tuple(item.generated)))
        beams = candidates[:num_beams]
        if all(item.finished for item in beams):
            break

    best = max(beams, key=lambda item: (item.score, tuple(item.generated)))
    if best.reason is not None:
        stopped_reason = best.reason
    elif len(best.generated) >= max_new_tokens:
        stopped_reason = "max_new_tokens"
    elif max_sequence_length is not None and len(best.ids) >= max_sequence_length:
        stopped_reason = "max_sequence_length"
    else:
        stopped_reason = "max_new_tokens"
    return list(best.ids), list(best.generated), stopped_reason


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
    strategy: str = "greedy",
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    num_beams: int = 1,
    quantize: str | None = None,
) -> ModelGeneration:
    """Generate tokens from real forward passes.

    Decoder-only graphs without learned positional tables use an in-memory K/V
    cache for greedy and sampled decoding: one prefill pass plus one single-token
    decode per generated token. Other graphs fall back to full-sequence
    recomputation. Beam search scores complete hypotheses by cumulative
    log-probability. Generation stops on EOS, the requested token budget, or the
    graph's declared maximum sequence length.
    """
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise InvalidOperationParameters("max_new_tokens must be a positive integer")
    if eos_token_id is not None and (
        isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int) or eos_token_id < 0
    ):
        raise InvalidOperationParameters("eos_token_id must be a non-negative integer")
    _validate_generation_options(
        strategy=strategy,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
        num_beams=num_beams,
    )

    graph = graph_from_manifest(manifest)
    stored_quantization = _stored_quantization(graph)
    quantization: dict[str, Any] | None = stored_quantization
    if quantize is not None:
        if quantize != "int8":
            raise InvalidOperationParameters("quantize must be 'int8' when provided")
        if stored_quantization is None:
            from uzen.core.wie.quantization import quantize_int8

            quantized = quantize_int8(graph, package)
            graph = quantized.graph
            package = quantized.package
            metadata = dict(manifest.get("metadata") or {})
            metadata["graph"] = graph.to_dict()
            manifest = {**dict(manifest), "metadata": metadata}
            quantization = quantized.report.to_dict()
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
    rng = np.random.default_rng(seed)

    if strategy == "beam" and num_beams > 1:
        ids, generated, stopped_reason = _beam_generate(
            package,
            manifest,
            artifact_id=artifact_id,
            prompt_ids=prompt_ids,
            base_positions=position_row,
            max_new_tokens=max_new_tokens,
            max_sequence_length=max_sequence_length,
            eos_token_id=resolved_eos,
            num_beams=num_beams,
        )

    cache = KVCache() if strategy != "beam" and _supports_kv_cache(graph) else None
    logits: np.ndarray | None = None

    if strategy == "beam" and num_beams > 1:
        cache = None
    elif cache is not None:
        prefill = execute_stored_model(
            package,
            manifest,
            artifact_id=artifact_id,
            tokens=[ids],
            positions=[position_row] if position_row is not None else None,
            cache=cache,
        )
        logits = prefill.output("logits")
        if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != len(ids):
            raise InvalidOperationParameters(
                "prefill logits must have shape (1, sequence, vocabulary), got "
                f"{list(logits.shape)}"
            )

    while not (strategy == "beam" and num_beams > 1) and len(generated) < max_new_tokens:
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
        if logits is None:
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
        next_token = _select_next_token(
            last_logits,
            strategy=strategy,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            rng=rng,
        )
        ids.append(next_token)
        generated.append(next_token)
        if position_row is not None:
            position_row.append(position_row[-1] + 1)
        if resolved_eos is not None and next_token == resolved_eos:
            stopped_reason = "eos_token"
            break
        if len(generated) >= max_new_tokens:
            break
        if cache is not None:
            # Feed the newly appended token through the model once; the cache
            # already holds the prompt's and earlier tokens' K/V tensors.
            decode_position = position_row[-1] if position_row is not None else len(ids) - 1
            decode = execute_stored_model(
                package,
                manifest,
                artifact_id=artifact_id,
                tokens=[[next_token]],
                positions=[[decode_position]],
                cache=cache,
            )
            logits = decode.output("logits")
            if logits.ndim != 3 or logits.shape != (1, 1, int(logits.shape[-1])):
                raise InvalidOperationParameters(
                    "cached decode logits must have shape (1, 1, vocabulary), got "
                    f"{list(logits.shape)}"
                )
        else:
            # Force a full-sequence recomputation for the next token.
            logits = None

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
        strategy=strategy,
        temperature=float(temperature),
        top_k=top_k,
        top_p=top_p,
        seed=seed,
        num_beams=num_beams,
        quantization=quantization,
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
    quantize: str | None = None,
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
        quantize=quantize,
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
    max_abs_error_threshold: float | None = None,
    mean_abs_error_threshold: float | None = None,
    max_rel_error_threshold: float | None = None,
) -> dict[str, Any]:
    """Compare two executions on identical inputs with explicit tolerances."""
    if atol < 0 or rtol < 0:
        raise InvalidOperationParameters("atol and rtol must be non-negative")
    thresholds = {
        "max_abs_error": max_abs_error_threshold,
        "mean_abs_error": mean_abs_error_threshold,
        "max_rel_error": max_rel_error_threshold,
    }
    for name, value in thresholds.items():
        if value is not None and value < 0:
            raise InvalidOperationParameters(f"{name} threshold must be non-negative")
    source_names = set(source.outputs)
    candidate_names = set(candidate.outputs)
    if source_names != candidate_names:
        return {
            "passed": False,
            "error": "output names differ",
            "source_outputs": sorted(source_names),
            "candidate_outputs": sorted(candidate_names),
            "tolerances": {"atol": atol, "rtol": rtol},
            "drift": {
                "complete": False,
                "max_abs_error": None,
                "mean_abs_error": None,
                "max_rel_error": None,
                "num_outputs": 0,
                "num_elements": 0,
                "allclose_passed": False,
                "failed_outputs": [],
                "thresholds": thresholds,
                "failures": ["output names differ"],
                "passed": False,
            },
        }

    outputs: dict[str, dict[str, Any]] = {}
    all_passed = True
    complete = True
    total_difference = 0.0
    total_elements = 0
    max_abs_error = 0.0
    max_rel_error = 0.0
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
            complete = False
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
        if difference.size:
            total_difference += float(difference.sum())
            total_elements += int(difference.size)
            max_abs_error = max(max_abs_error, float(difference.max()))
            max_rel_error = max(max_rel_error, float(relative.max()))
        all_passed = all_passed and passed

    failures: list[str] = []
    if not complete:
        if any(value is not None for value in thresholds.values()):
            failures.append("aggregate thresholds unavailable because outputs are incomplete")
        aggregate = {
            "max_abs_error": None,
            "mean_abs_error": None,
            "max_rel_error": None,
        }
    else:
        aggregate = {
            "max_abs_error": max_abs_error,
            "mean_abs_error": (
                total_difference / total_elements if total_elements else 0.0
            ),
            "max_rel_error": max_rel_error,
        }
        for name, threshold in thresholds.items():
            if threshold is not None and aggregate[name] > threshold:
                failures.append(
                    f"{name} {aggregate[name]:.6g} exceeds threshold {threshold:.6g}"
                )
    drift = {
        "complete": complete,
        **aggregate,
        "num_outputs": len(outputs),
        "num_elements": total_elements if complete else None,
        "allclose_passed": all_passed,
        "failed_outputs": [
            name for name, output in outputs.items() if not output.get("passed")
        ],
        "thresholds": thresholds,
        "failures": failures,
        "passed": complete and not failures and all_passed,
    }
    return {
        "passed": all_passed and drift["passed"],
        "tolerances": {"atol": atol, "rtol": rtol},
        "drift": drift,
        "source_artifact_id": source.artifact_id,
        "candidate_artifact_id": candidate.artifact_id,
        "inputs": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in source.inputs.items()
        },
        "outputs": outputs,
    }
