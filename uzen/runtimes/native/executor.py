"""Native execution of NIR graphs through the C++ kernel library.

This is a third executor alongside the NumPy reference and the torch-CPU
executor. It runs every supported operator through real native kernels; layout
operators (reshape/transpose/split/merge/rope indexing) use NumPy because they
move bytes rather than perform arithmetic, and they are reported separately.

There is no silent fallback. ``backend="native"`` refuses an operator, dtype or
layout it does not implement with :class:`UnsupportedOperator`; callers can choose
the NumPy or torch executor explicitly when they need wider coverage.

Current native coverage: linear, matmul (rank 2), layernorm, rmsnorm, softmax
(last axis), GELU (tanh and exact), SiLU, ReLU, tanh, sigmoid, add/sub/mul with
broadcasting, scale, clip, embedding, mean/sum over the last axis, attention
with causal/window/ALiBi masks, and the layout operators needed by the supported
architecture families.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from uzen.core.nir.executor import resolve_bindings
from uzen.core.nir.graph import Graph
from uzen.core.nir.ops import get_operator, port_order
from uzen.core.nir.types import TensorType, replace_symbols
from uzen.core.nws.tensors import WeightPackage
from uzen.errors import (
    InvalidOperationParameters,
    ShapeMismatch,
    TensorInvalid,
    UnsupportedOperator,
)

from .library import NativeLibrary

GRAPH_INPUT = "input:"

_NATIVE_UNARY = frozenset({"gelu", "gelu_exact", "silu", "relu", "tanh", "sigmoid"})
_NATIVE_BINARY = frozenset({"add", "sub", "mul"})
_LAYOUT_OPS = frozenset(
    {
        "reshape",
        "transpose",
        "slice",
        "flatten",
        "concat",
        "split_qkv",
        "merge_heads",
        "rope",
        "repeat_kv",
        "pos_embedding",
        "identity",
        "dropout",
        "select",
    }
)


@dataclass
class NativeExecution:
    outputs: dict[str, np.ndarray]
    intermediates: dict[str, np.ndarray] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    num_ops: int = 0
    native_ops: list[str] = field(default_factory=list)
    layout_ops: list[str] = field(default_factory=list)

    def output(self, name: str) -> np.ndarray:
        try:
            return self.outputs[name]
        except KeyError as exc:
            raise TensorInvalid(
                f"graph produced no output named '{name}'; available: "
                f"{sorted(self.outputs)}"
            ) from exc

    def summary(self) -> dict[str, Any]:
        return {
            "backend": "native-cpp",
            "outputs": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in self.outputs.items()
            },
            "num_ops": self.num_ops,
            "native_ops": list(self.native_ops),
            "layout_ops": list(self.layout_ops),
            "captured_intermediates": sorted(self.intermediates),
        }


def _key(ref: Any) -> str:
    if ref.node.startswith(GRAPH_INPUT):
        return ref.node
    return f"{ref.node}:{ref.port}"


def _resolve_params(
    params: Mapping[str, Any], symbols: Mapping[str, int]
) -> dict[str, Any]:
    out = dict(params)
    shape = out.get("shape")
    if isinstance(shape, (list, tuple)):
        out["shape"] = [
            int(symbols[dim]) if isinstance(dim, str) and dim in symbols else dim
            for dim in shape
        ]
    return out


def _name(value: Any, default: str = "weight") -> str:
    return str(value if value is not None else default).replace("param:", "")


def _bound(
    bound: Mapping[str, np.ndarray], name: str, *, node: str, operator: str
) -> np.ndarray:
    try:
        return bound[name]
    except KeyError as exc:
        raise ShapeMismatch(
            f"node '{node}' ({operator}) has no bound tensor '{name}'"
        ) from exc


def _axes(value: Any, rank: int) -> tuple[int, ...]:
    axes = (value,) if isinstance(value, int) else tuple(value)
    normalized: list[int] = []
    for axis in axes:
        if axis < -rank or axis >= rank:
            raise InvalidOperationParameters(
                f"axis {axis} out of range for rank {rank}"
            )
        normalized.append(axis % rank)
    return tuple(normalized)


def _rope(
    x: np.ndarray, positions: np.ndarray, params: Mapping[str, Any]
) -> np.ndarray:
    if positions.dtype.kind not in "iu":
        raise InvalidOperationParameters("rope positions must be integer")
    batch, _heads, sequence, dimensions = x.shape
    if positions.ndim == 1:
        pos = positions.reshape(1, -1)
    else:
        pos = positions
    if pos.shape[0] not in (1, batch) or pos.shape[1] != sequence:
        raise ShapeMismatch(
            f"rope positions {positions.shape} do not match "
            f"(batch={batch}, seq={sequence})"
        )
    raw = params.get("rotary_dim")
    rotary_dim = int(raw) if raw is not None else dimensions
    if rotary_dim <= 0 or rotary_dim % 2 or rotary_dim > dimensions:
        raise InvalidOperationParameters(
            f"rope rotary_dim={rotary_dim} must be positive, even and <= {dimensions}"
        )
    theta = float(params.get("theta", 10000.0))
    if theta <= 0:
        raise InvalidOperationParameters("rope theta must be positive")
    half = rotary_dim // 2
    inverse = 1.0 / (
        theta ** (np.arange(0, rotary_dim, 2, dtype=np.float64) / rotary_dim)
    )
    interleaved = bool(params.get("interleaved", False))
    frequencies = (
        np.repeat(inverse, 2)
        if interleaved
        else np.concatenate([inverse, inverse], axis=0)
    )
    angles = pos.astype(np.float64)[:, None, :, None] * frequencies[
        None, None, None, :
    ]
    cosines = np.cos(angles)
    sines = np.sin(angles)
    rotated_input = x[..., :rotary_dim]
    if interleaved:
        pairs = rotated_input.reshape(rotated_input.shape[:-1] + (half, 2))
        rotated = np.stack([-pairs[..., 1], pairs[..., 0]], axis=-1).reshape(
            rotated_input.shape
        )
    else:
        first = rotated_input[..., :half]
        second = rotated_input[..., half:]
        rotated = np.concatenate([-second, first], axis=-1)
    output = (
        rotated_input.astype(np.float64) * cosines
        + rotated.astype(np.float64) * sines
    )
    if rotary_dim != dimensions:
        output = np.concatenate([output, x[..., rotary_dim:]], axis=-1)
    return output.astype(np.float32, copy=False)


def _split_qkv(
    x: np.ndarray, params: Mapping[str, Any]
) -> list[np.ndarray]:
    if x.ndim != 3:
        raise InvalidOperationParameters(
            f"split_qkv expects (batch, seq, 3*dim), got {x.shape}"
        )
    heads = int(params["n_heads"])
    layout = str(params.get("layout", "grouped"))
    batch, sequence, last = x.shape
    if heads <= 0 or last % (3 * heads):
        raise InvalidOperationParameters(
            f"split_qkv: last dim {last} is not divisible by 3*n_heads ({3 * heads})"
        )
    head_dim = last // (3 * heads)
    if layout == "grouped":
        reshaped = np.transpose(
            x.reshape(batch, sequence, 3, heads, head_dim), (0, 2, 3, 1, 4)
        )
        return [reshaped[:, 0], reshaped[:, 1], reshaped[:, 2]]
    if layout == "interleaved":
        reshaped = np.transpose(
            x.reshape(batch, sequence, heads, 3, head_dim), (0, 2, 3, 1, 4)
        )
        return [reshaped[:, :, 0], reshaped[:, :, 1], reshaped[:, :, 2]]
    raise InvalidOperationParameters(
        "split_qkv layout must be 'grouped' or 'interleaved'"
    )


def _dispatch(
    node: Any,
    inputs: list[np.ndarray],
    bound: Mapping[str, np.ndarray],
    params: Mapping[str, Any],
    symbols: Mapping[str, int],
    library: NativeLibrary,
) -> tuple[list[np.ndarray], str]:
    op = node.op

    if op == "linear":
        weight = _bound(
            bound, _name(params.get("weight"), "weight"), node=node.id, operator=op
        )
        bias_name = params.get("bias")
        bias = (
            _bound(bound, _name(bias_name), node=node.id, operator=op)
            if bias_name
            else None
        )
        return [library.linear(inputs[0], weight, bias)], "native"

    if op in _NATIVE_UNARY:
        return [library.unary(op, inputs[0])], "native"

    if op in _NATIVE_BINARY:
        return [library.binary(op, inputs[0], inputs[1])], "native"

    if op == "scale":
        return [
            library.scale(inputs[0], float(params.get("factor", 1.0)))
        ], "native"

    if op == "clip":
        return [
            library.clip(
                inputs[0],
                float(params.get("min", -1.0)),
                float(params.get("max", 1.0)),
            )
        ], "native"

    if op == "layernorm":
        weight_name = params.get("weight")
        bias_name = params.get("bias")
        return [
            library.layernorm(
                inputs[0],
                weight=(
                    _bound(bound, _name(weight_name), node=node.id, operator=op)
                    if weight_name
                    else None
                ),
                bias=(
                    _bound(bound, _name(bias_name), node=node.id, operator=op)
                    if bias_name
                    else None
                ),
                eps=float(params.get("eps", 1e-5)),
            )
        ], "native"

    if op == "rmsnorm":
        weight_name = params.get("weight")
        return [
            library.rmsnorm(
                inputs[0],
                weight=(
                    _bound(bound, _name(weight_name), node=node.id, operator=op)
                    if weight_name
                    else None
                ),
                eps=float(params.get("eps", 1e-6)),
            )
        ], "native"

    if op == "softmax":
        axes = _axes(params.get("axis", -1), inputs[0].ndim)
        if axes != (inputs[0].ndim - 1,):
            raise UnsupportedOperator(
                f"native softmax supports only the last axis, got axes {axes} "
                f"for node '{node.id}'",
                node=node.id,
                operator=op,
            )
        return [library.softmax(inputs[0])], "native"

    if op == "matmul":
        if inputs[0].ndim != 2 or inputs[1].ndim != 2:
            raise UnsupportedOperator(
                f"native matmul supports rank-2 operands only (node '{node.id}')",
                node=node.id,
                operator=op,
            )
        return [library.matmul(inputs[0], inputs[1])], "native"

    if op == "embedding":
        weight = _bound(
            bound, _name(params.get("weight"), "weight"), node=node.id, operator=op
        )
        return [library.embedding(inputs[0], weight)], "native"

    if op in ("mean", "sum"):
        axes = _axes(params.get("axis", -1), inputs[0].ndim)
        if axes != (inputs[0].ndim - 1,):
            raise UnsupportedOperator(
                f"native {op} supports only the last axis, got axes {axes} for "
                f"node '{node.id}'",
                node=node.id,
                operator=op,
            )
        return [
            library.reduce_last(
                inputs[0], mode=op, keepdims=bool(params.get("keepdims", False))
            )
        ], "native"

    if op == "attention":
        if len(inputs) != 3:
            raise InvalidOperationParameters("attention requires (q, k, v)")
        slopes = params.get("alibi_slopes")
        if slopes is not None and len(slopes) == 0:
            slopes = None
        return [
            library.attention(
                inputs[0],
                inputs[1],
                inputs[2],
                scale=params.get("scale", "auto"),
                causal=bool(params.get("causal", True)),
                alibi=(
                    np.asarray(slopes, dtype=np.float32)
                    if slopes is not None
                    else None
                ),
                window=(
                    int(params["window"]) if params.get("window") is not None else None
                ),
            )
        ], "native"

    if op == "pos_embedding":
        x = inputs[0]
        weight = _bound(
            bound, _name(params.get("weight"), "weight"), node=node.id, operator=op
        )
        if x.ndim < 3:
            raise InvalidOperationParameters(
                f"pos_embedding expects (batch, seq, dim), got {x.shape}"
            )
        sequence = x.shape[1]
        offset = int(params.get("offset", 0))
        if offset < 0 or offset + sequence > weight.shape[0]:
            raise ShapeMismatch(
                f"pos_embedding offset={offset}, seq={sequence} exceeds table "
                f"size {weight.shape[0]}"
            )
        table = weight[offset : offset + sequence].reshape(
            (1, sequence) + tuple(x.shape[2:])
        )
        return [x + table], "layout"

    if op == "split_qkv":
        return _split_qkv(inputs[0], params), "layout"

    if op == "merge_heads":
        x = inputs[0]
        batch, heads, sequence, head_dim = x.shape
        return [
            np.transpose(x, (0, 2, 1, 3)).reshape(
                batch, sequence, heads * head_dim
            )
        ], "layout"

    if op == "rope":
        if len(inputs) != 2:
            raise InvalidOperationParameters("rope requires (x, positions)")
        return [_rope(inputs[0], inputs[1], params)], "layout"

    if op == "repeat_kv":
        x = inputs[0]
        heads = int(params["n_heads"])
        if x.ndim != 4 or heads % x.shape[1]:
            raise InvalidOperationParameters(
                f"repeat_kv requires n_heads divisible by kv_heads, got "
                f"n_heads={heads}, kv_heads={x.shape[1] if x.ndim == 4 else '?'}"
            )
        return [np.repeat(x, heads // x.shape[1], axis=1)], "layout"

    if op == "reshape":
        target = params.get("shape")
        if not isinstance(target, (list, tuple)):
            raise InvalidOperationParameters("reshape requires a 'shape' list")
        shape: list[int] = []
        for dimension in target:
            if isinstance(dimension, int):
                shape.append(dimension)
            elif isinstance(dimension, str) and dimension in symbols:
                shape.append(int(symbols[dimension]))
            else:
                raise InvalidOperationParameters(
                    f"native reshape cannot resolve dimension {dimension!r}"
                )
        return [inputs[0].reshape(tuple(shape))], "layout"

    if op == "transpose":
        permutation = params.get("perm")
        if permutation is None:
            permutation = list(reversed(range(inputs[0].ndim)))
        return [
            np.transpose(inputs[0], tuple(int(value) for value in permutation))
        ], "layout"

    if op == "slice":
        x = inputs[0]
        axis = int(params.get("axis", 0)) % x.ndim
        index = [slice(None)] * x.ndim
        index[axis] = slice(
            params.get("start"), params.get("stop"), params.get("step")
        )
        return [x[tuple(index)]], "layout"

    if op == "flatten":
        x = inputs[0]
        start = int(params.get("start_dim", 1))
        return [x.reshape(tuple(x.shape[:start]) + (-1,))], "layout"

    if op == "concat":
        return [
            np.concatenate(
                inputs, axis=int(params.get("axis", -1))
            )
        ], "layout"

    if op == "identity" or op == "dropout":
        return [inputs[0]], "layout"

    if op == "select":
        return [np.where(inputs[0] != 0, inputs[1], inputs[2])], "layout"

    if op == "quantized_linear":
        raise UnsupportedOperator(
            "the native executor does not implement int8 quantized_linear yet; "
            "use the NumPy executor or dequantize the artifact explicitly",
            node=node.id,
            operator=op,
        )

    raise UnsupportedOperator(
        f"native executor does not implement NIR operator '{op}' (node "
        f"'{node.id}'); supported native arithmetic: linear, matmul, layernorm, "
        f"rmsnorm, softmax, {', '.join(sorted(_NATIVE_UNARY))}, "
        f"{', '.join(sorted(_NATIVE_BINARY))}, scale, clip, embedding, "
        "mean/sum(last axis), attention; layout: "
        f"{', '.join(sorted(_LAYOUT_OPS))}",
        node=node.id,
        operator=op,
    )


def execute_native(
    graph: Graph,
    inputs: Mapping[str, np.ndarray],
    package: WeightPackage,
    *,
    capture: Sequence[str] | None = None,
    check_finite: bool = True,
    validate_types: bool = True,
    library: NativeLibrary | None = None,
) -> NativeExecution:
    """Run a graph through the native C++ kernels, or raise UnsupportedOperator."""
    library = library or NativeLibrary()
    package.assert_unmodified()
    bound = resolve_bindings(graph, package)

    missing = [name for name in graph.inputs if name not in inputs]
    if missing:
        raise ShapeMismatch(f"missing graph inputs: {missing}")
    extra = [name for name in inputs if name not in graph.inputs]
    if extra:
        raise ShapeMismatch(f"unexpected graph inputs: {extra}")

    env: dict[str, np.ndarray] = {}
    symbols: dict[str, int] = {}
    for name, spec in graph.inputs.items():
        array = np.asarray(inputs[name])
        if array.ndim != len(spec.shape):
            raise ShapeMismatch(
                f"input '{name}' has rank {array.ndim}, declared rank "
                f"{len(spec.shape)}"
            )
        for dimension, actual in zip(spec.shape, array.shape):
            if isinstance(dimension, int):
                if int(dimension) != int(actual):
                    raise ShapeMismatch(
                        f"input '{name}' dimension {actual} != declared {dimension}"
                    )
            elif isinstance(dimension, str):
                previous = symbols.get(dimension)
                if previous is not None and previous != int(actual):
                    raise ShapeMismatch(
                        f"symbol '{dimension}' bound inconsistently: "
                        f"{previous} vs {actual}"
                    )
                symbols[dimension] = int(actual)
            else:
                raise ShapeMismatch(
                    f"input '{name}' declares an unknown dimension; use concrete "
                    "dims or symbols"
                )
        env[f"{GRAPH_INPUT}{name}"] = array

    order = graph.topological_order()
    captures = set(capture or ())
    captured: dict[str, np.ndarray] = {}
    native_ops: list[str] = []
    layout_ops: list[str] = []

    for node_id in order:
        node = graph.nodes[node_id]
        op = get_operator(node.op)
        in_arrays: list[np.ndarray] = []
        for port in port_order(op, node.inputs):
            key = _key(node.inputs[port])
            if key not in env:
                raise ShapeMismatch(
                    f"node '{node_id}': missing value for input port '{port}'"
                )
            in_arrays.append(env[key])
        params = _resolve_params(node.params, symbols)
        outputs, kind = _dispatch(
            node, in_arrays, bound.get(node_id, {}), params, symbols, library
        )
        (native_ops if kind == "native" else layout_ops).append(node.op)

        if len(outputs) != max(1, len(node.outputs)):
            in_types = [
                TensorType.of(
                    tuple(env[_key(node.inputs[port])].shape),
                    str(env[_key(node.inputs[port])].dtype),
                )
                for port in port_order(op, node.inputs)
            ]
            node.outputs = tuple(
                op.infer_types(in_types, node.params, {})
            )
        for index, output in enumerate(outputs):
            array = np.asarray(output)
            if validate_types and index < len(node.outputs):
                declared = node.outputs[index]
                try:
                    expected = replace_symbols(declared.shape, symbols)
                except KeyError:
                    expected = None
                if expected is not None and tuple(expected) != tuple(array.shape):
                    raise ShapeMismatch(
                        f"node '{node_id}' ({node.op}) produced shape "
                        f"{array.shape}, declared {declared.shape}"
                    )
            if (
                check_finite
                and array.dtype.kind in "fc"
                and not np.isfinite(array).all()
            ):
                raise TensorInvalid(
                    f"node '{node_id}' ({node.op}) produced non-finite values"
                )
            env[f"{node_id}:{index}"] = array
        if node_id in captures:
            captured[node_id] = np.array(outputs[0], copy=True)

    package.assert_unmodified()
    output_arrays: dict[str, np.ndarray] = {}
    for name, ref in graph.outputs.items():
        key = _key(ref)
        if key not in env:
            raise ShapeMismatch(
                f"graph output '{name}' references missing value {ref}"
            )
        output_arrays[name] = env[key]
    return NativeExecution(
        outputs=output_arrays,
        intermediates=captured,
        order=order,
        num_ops=len(order),
        native_ops=native_ops,
        layout_ops=layout_ops,
    )
