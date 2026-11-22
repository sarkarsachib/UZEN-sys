"""Embedded ONNX executor: run ONNX models without ``onnxruntime``.

The interchange module (:mod:`uzen.runtimes.onnx_runtime`) converts between NIR and
ONNX using the optional ``onnx`` package. This module goes further: it executes a
serialized ONNX model with UZEN's own NumPy kernels, decoding the protobuf wire
format itself. There is no dependency on ``onnx``, ``onnxruntime`` or ``protobuf`` at
run time, which is what makes it usable in a minimal deployment.

Supported operators (each one implemented against the ONNX specification, with no
"best effort" rewriting):

    Add, Sub, Mul, Div, Pow, Sqrt, MatMul, Gemm, Relu, Tanh, Sigmoid, Gelu,
    Softmax, Identity, Reshape, Transpose, Flatten, Concat, Gather, Where,
    ReduceMean, ReduceSum, Constant

Anything else raises :class:`~uzen.errors.UnsupportedOperator` naming the node and
op type, and malformed bytes raise :class:`~uzen.errors.UnsupportedFormat`; the
executor never guesses. All compute is float32/int64 NumPy — this is a CPU reference
executor, not a claim of acceleration.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from uzen.errors import InvalidOperationParameters, ShapeMismatch, UnsupportedFormat, UnsupportedOperator

_ONNX_DTYPES: dict[int, np.dtype] = {
    1: np.dtype(np.float32),
    2: np.dtype(np.uint8),
    3: np.dtype(np.int8),
    4: np.dtype(np.uint16),
    5: np.dtype(np.int16),
    6: np.dtype(np.int32),
    7: np.dtype(np.int64),
    9: np.dtype(np.bool_),
    10: np.dtype(np.float16),
    11: np.dtype(np.float64),
}

_TENSOR_FIELDS = {
    "dims": 1,
    "data_type": 2,
    "float_data": 4,
    "int32_data": 5,
    "string_data": 6,
    "int64_data": 7,
    "name": 8,
    "raw_data": 9,
    "double_data": 10,
}

SUPPORTED_OPS = frozenset(
    {
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Pow",
        "Sqrt",
        "MatMul",
        "Gemm",
        "Relu",
        "Tanh",
        "Sigmoid",
        "Gelu",
        "Softmax",
        "Identity",
        "Reshape",
        "Transpose",
        "Flatten",
        "Concat",
        "Gather",
        "Where",
        "ReduceMean",
        "ReduceSum",
        "Constant",
    }
)


# --------------------------------------------------------------------------- #
# protobuf wire format
# --------------------------------------------------------------------------- #


def _varint(buffer: bytes, position: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        if position >= len(buffer):
            raise UnsupportedFormat("truncated protobuf varint")
        byte = buffer[position]
        position += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, position
        shift += 7
        if shift > 70:
            raise UnsupportedFormat("protobuf varint is too long")


def _fields(buffer: bytes) -> list[tuple[int, int, Any]]:
    position = 0
    fields: list[tuple[int, int, Any]] = []
    while position < len(buffer):
        key, position = _varint(buffer, position)
        number, wire = key >> 3, key & 0x07
        if wire == 0:
            value, position = _varint(buffer, position)
        elif wire == 1:
            value = buffer[position : position + 8]
            position += 8
        elif wire == 2:
            length, position = _varint(buffer, position)
            value = buffer[position : position + length]
            position += length
            if len(value) != length:
                raise UnsupportedFormat("truncated protobuf length-delimited field")
        elif wire == 5:
            value = buffer[position : position + 4]
            position += 4
        else:
            raise UnsupportedFormat(f"unsupported protobuf wire type {wire}")
        fields.append((number, wire, value))
    return fields


def _packed_ints(value: Any) -> list[int]:
    if isinstance(value, int):
        return [int(value)]
    numbers: list[int] = []
    position = 0
    while position < len(value):
        number, position = _varint(value, position)
        numbers.append(int(number))
    return numbers


def _packed_floats(value: bytes) -> list[float]:
    if len(value) % 4:
        raise UnsupportedFormat("packed float field is not a multiple of 4 bytes")
    return list(struct.unpack(f"<{len(value) // 4}f", value))


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _parse_tensor(buffer: bytes) -> dict[str, Any]:
    dims: list[int] = []
    dtype_id = 0
    raw = b""
    name = ""
    data: dict[str, Any] = {}
    for number, wire, value in _fields(buffer):
        if number == 1:
            dims.extend(_packed_ints(value) if wire == 2 else [int(value)])
        elif number == 2:
            dtype_id = int(value)
        elif number == 4:
            data.setdefault("float_data", []).extend(
                _packed_floats(value) if wire == 2 else [struct.unpack("<f", value)[0]]
            )
        elif number == 5:
            data.setdefault("int32_data", []).extend(
                _packed_ints(value) if wire == 2 else [int(value)]
            )
        elif number == 6:
            data.setdefault("string_data", []).append(bytes(value))
        elif number == 7:
            data.setdefault("int64_data", []).extend(
                _packed_ints(value) if wire == 2 else [int(value)]
            )
        elif number == 8:
            name = _text(value)
        elif number == 9:
            raw = bytes(value)
        elif number == 10:
            data.setdefault("double_data", []).extend(
                [_packed_floats(value)] if wire == 2 else [struct.unpack("<d", value)[0]]
            )
    dtype = _ONNX_DTYPES.get(dtype_id)
    if dtype is None:
        raise UnsupportedFormat(f"ONNX tensor '{name}' uses unsupported data_type {dtype_id}")
    count = int(np.prod(dims, dtype=np.int64)) if dims else 1
    if raw:
        array = np.frombuffer(raw, dtype=dtype, count=count).copy()
    else:
        payload = (
            data.get("float_data")
            or data.get("int64_data")
            or data.get("int32_data")
            or data.get("double_data")
            or data.get("string_data")
        )
        if payload is None:
            array = np.zeros(count, dtype=dtype)
        else:
            array = np.asarray(payload, dtype=dtype)
    return {"name": name, "dims": dims, "dtype": dtype, "array": array.reshape(dims) if dims else array.reshape(())}


def _parse_attribute(buffer: bytes) -> tuple[str, Any]:
    name = ""
    attribute_type = 0
    values: dict[str, Any] = {}
    for number, wire, value in _fields(buffer):
        if number == 1:
            name = _text(value)
        elif number == 2:
            values["f"] = struct.unpack("<f", value)[0]
        elif number == 3:
            values["i"] = int(value)
        elif number == 4:
            values["s"] = bytes(value)
        elif number == 5:
            values["t"] = _parse_tensor(bytes(value))
        elif number == 7:
            values["floats"] = _packed_floats(value) if wire == 2 else [struct.unpack("<f", value)[0]]
        elif number == 8:
            values["ints"] = _packed_ints(value) if wire == 2 else [int(value)]
        elif number == 9:
            values.setdefault("strings", []).append(bytes(value))
        elif number == 20:
            attribute_type = int(value)
    if attribute_type == 1:
        return name, float(values.get("f", 0.0))
    if attribute_type == 2:
        return name, int(values.get("i", 0))
    if attribute_type == 3:
        return name, _text(values.get("s", b""))
    if attribute_type == 4:
        return name, values["t"]
    if attribute_type == 6:
        return name, [float(item) for item in values.get("floats", [])]
    if attribute_type == 7:
        return name, [int(item) for item in values.get("ints", [])]
    if attribute_type == 8:
        return name, [_text(item) for item in values.get("strings", [])]
    if "i" in values:
        return name, int(values["i"])
    if "f" in values:
        return name, float(values["f"])
    if "s" in values:
        return name, _text(values["s"])
    return name, None


@dataclass
class OnnxNode:
    op_type: str
    inputs: list[str]
    outputs: list[str]
    name: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class OnnxGraph:
    name: str
    nodes: list[OnnxNode]
    initializers: dict[str, np.ndarray]
    inputs: list[str]
    outputs: list[str]

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "nodes": len(self.nodes),
            "initializers": len(self.initializers),
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "operators": sorted({node.op_type for node in self.nodes}),
        }


def _parse_value_info(buffer: bytes) -> str:
    name = ""
    for number, _wire, value in _fields(buffer):
        if number == 1:
            name = _text(value)
    return name


def parse_onnx_graph(buffer: bytes) -> OnnxGraph:
    """Decode a serialized ONNX ``ModelProto`` into a plain graph description."""
    if not isinstance(buffer, (bytes, bytearray)) or not buffer:
        raise UnsupportedFormat("empty ONNX model bytes")
    graph_bytes: bytes | None = None
    for number, _wire, value in _fields(bytes(buffer)):
        if number == 7:
            graph_bytes = bytes(value)
    if graph_bytes is None:
        raise UnsupportedFormat("ONNX model has no graph")
    nodes: list[OnnxNode] = []
    initializers: dict[str, np.ndarray] = {}
    inputs: list[str] = []
    outputs: list[str] = []
    name = ""
    for number, _wire, value in _fields(graph_bytes):
        if number == 1:
            node_inputs: list[str] = []
            node_outputs: list[str] = []
            node_name = ""
            op_type = ""
            attributes: dict[str, Any] = {}
            for field_number, _field_wire, field_value in _fields(bytes(value)):
                if field_number == 1:
                    node_inputs.append(_text(field_value))
                elif field_number == 2:
                    node_outputs.append(_text(field_value))
                elif field_number == 3:
                    node_name = _text(field_value)
                elif field_number == 4:
                    op_type = _text(field_value)
                elif field_number == 5:
                    attribute_name, attribute_value = _parse_attribute(bytes(field_value))
                    attributes[attribute_name] = attribute_value
            nodes.append(
                OnnxNode(
                    op_type=op_type,
                    inputs=[item for item in node_inputs if item],
                    outputs=[item for item in node_outputs if item],
                    name=node_name,
                    attributes=attributes,
                )
            )
        elif number == 2:
            name = _text(value)
        elif number == 5:
            tensor = _parse_tensor(bytes(value))
            initializers[tensor["name"]] = tensor["array"]
        elif number == 11:
            inputs.append(_parse_value_info(bytes(value)))
        elif number == 12:
            outputs.append(_parse_value_info(bytes(value)))
    inputs = [item for item in inputs if item and item not in initializers]
    return OnnxGraph(
        name=name,
        nodes=nodes,
        initializers=initializers,
        inputs=inputs,
        outputs=[item for item in outputs if item],
    )


# --------------------------------------------------------------------------- #
# executor
# --------------------------------------------------------------------------- #


def _gelu_tanh(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


class OnnxEmbeddedExecutor:
    """Execute a parsed ONNX graph with NumPy kernels."""

    backend = "onnx-embedded"

    def __init__(self, source: str | Path | bytes | bytearray | OnnxGraph) -> None:
        if isinstance(source, OnnxGraph):
            self.graph = source
        elif isinstance(source, (bytes, bytearray)):
            self.graph = parse_onnx_graph(bytes(source))
        else:
            self.graph = parse_onnx_graph(Path(source).read_bytes())
        declared = set(self.graph.initializers) | set(self.graph.inputs) | set(self.graph.outputs)
        for node in self.graph.nodes:
            declared.update(node.outputs)
        for node in self.graph.nodes:
            if node.op_type not in SUPPORTED_OPS:
                raise UnsupportedOperator(
                    f"the embedded ONNX executor does not implement '{node.op_type}' "
                    f"(node '{node.name}'); supported: {', '.join(sorted(SUPPORTED_OPS))}",
                    node=node.name or None,
                    operator=node.op_type,
                )
            for name in node.inputs:
                if name and name not in declared:
                    raise InvalidOperationParameters(
                        f"ONNX node '{node.name}' reads '{name}', which no node produces"
                    )
        self._executions = 0

    # ------------------------------------------------------------------ #

    def execute(
        self,
        inputs: Mapping[str, np.ndarray],
        *,
        check_finite: bool = True,
    ) -> dict[str, np.ndarray]:
        missing = [name for name in self.graph.inputs if name not in inputs]
        if missing:
            raise ShapeMismatch(f"missing ONNX graph inputs: {missing}")
        extra = [name for name in inputs if name not in self.graph.inputs]
        if extra:
            raise ShapeMismatch(f"unexpected ONNX graph inputs: {extra}")
        env: dict[str, np.ndarray] = {
            name: np.asarray(array) for name, array in self.graph.initializers.items()
        }
        for name in self.graph.inputs:
            env[name] = np.asarray(inputs[name])
        for node in self.graph.nodes:
            values = [env[name] for name in node.inputs if name]
            result = self._dispatch(node, values, env)
            if not node.outputs:
                continue
            env[node.outputs[0]] = result
        produced: dict[str, np.ndarray] = {}
        for name in self.graph.outputs:
            if name not in env:
                raise ShapeMismatch(f"ONNX output '{name}' was not produced")
            value = env[name]
            if check_finite and value.dtype.kind == "f" and not np.isfinite(value).all():
                raise InvalidOperationParameters(f"ONNX output '{name}' contains non-finite values")
            produced[name] = value
        self._executions += 1
        return produced

    def _dispatch(self, node: OnnxNode, values: list[np.ndarray], env: Mapping[str, np.ndarray]) -> np.ndarray:
        op = node.op_type
        attrs = node.attributes
        if op == "Constant":
            for key in ("value", "value_float", "value_int", "value_floats", "value_ints"):
                if key in attrs:
                    value = attrs[key]
                    if isinstance(value, dict) and "array" in value:
                        return np.asarray(value["array"])
                    return np.asarray(value)
            raise UnsupportedFormat(f"ONNX Constant node '{node.name}' has no supported value")
        if op in ("Add", "Sub", "Mul", "Div", "Pow"):
            left, right = values[0], values[1]
            operation = {
                "Add": np.add,
                "Sub": np.subtract,
                "Mul": np.multiply,
                "Div": np.divide,
                "Pow": np.power,
            }[op]
            return operation(left, right)
        if op == "Sqrt":
            return np.sqrt(values[0])
        if op == "MatMul":
            return np.matmul(values[0], values[1])
        if op == "Gemm":
            a, b = values[0], values[1]
            if int(attrs.get("transA", 0)):
                a = a.T
            if int(attrs.get("transB", 0)):
                b = b.T
            result = float(attrs.get("alpha", 1.0)) * np.matmul(a, b)
            if len(values) > 2:
                result = result + float(attrs.get("beta", 1.0)) * values[2]
            return result
        if op == "Relu":
            return np.maximum(values[0], 0)
        if op == "Tanh":
            return np.tanh(values[0])
        if op == "Sigmoid":
            return 1.0 / (1.0 + np.exp(-values[0]))
        if op == "Gelu":
            approximate = str(attrs.get("approximate", "none"))
            if approximate != "tanh":
                raise UnsupportedOperator(
                    "the embedded executor implements the tanh approximation of Gelu; "
                    f"approximate='{approximate}' is refused rather than rewritten"
                )
            return _gelu_tanh(values[0])
        if op == "Softmax":
            axis = int(attrs.get("axis", -1))
            shifted = values[0] - values[0].max(axis=axis, keepdims=True)
            exponentiated = np.exp(shifted)
            return exponentiated / exponentiated.sum(axis=axis, keepdims=True)
        if op == "Identity":
            return values[0]
        if op == "Reshape":
            shape = [int(item) for item in np.asarray(values[1]).reshape(-1)]
            resolved: list[int] = []
            source = values[0].shape
            for index, dimension in enumerate(shape):
                if dimension == 0:
                    resolved.append(int(source[index]))
                elif dimension == -1:
                    known = int(np.prod([value for value in resolved if value], dtype=np.int64))
                    resolved.append(int(source_total(values[0]) // max(known, 1)))
                else:
                    resolved.append(dimension)
            return values[0].reshape(tuple(resolved))
        if op == "Transpose":
            perm = attrs.get("perm")
            if not perm:
                perm = list(reversed(range(values[0].ndim)))
            return np.transpose(values[0], [int(item) for item in perm])
        if op == "Flatten":
            axis = int(attrs.get("axis", 1))
            source = values[0]
            leading = int(np.prod(source.shape[:axis], dtype=np.int64)) if axis else 1
            return source.reshape((leading, -1))
        if op == "Concat":
            axis = int(attrs.get("axis", 0))
            return np.concatenate(values, axis=axis)
        if op == "Gather":
            axis = int(attrs.get("axis", 0))
            return np.take(values[0], np.asarray(values[1]).astype(np.int64), axis=axis)
        if op == "Where":
            return np.where(values[0] != 0, values[1], values[2])
        if op in ("ReduceMean", "ReduceSum"):
            axes = attrs.get("axes")
            axis: Any = [int(item) for item in axes] if axes else None
            keepdims = bool(int(attrs.get("keepdims", 1)))
            function = np.mean if op == "ReduceMean" else np.sum
            return function(values[0], axis=axis, keepdims=keepdims)
        raise UnsupportedOperator(  # pragma: no cover - constructor pre-validates
            f"the embedded ONNX executor does not implement '{op}'", operator=op
        )

    def describe(self) -> dict[str, Any]:
        return {
            **self.graph.describe(),
            "backend": self.backend,
            "executions": self._executions,
            "supported_operators": sorted(SUPPORTED_OPS),
        }


def source_total(array: np.ndarray) -> int:
    return int(np.prod(array.shape, dtype=np.int64))


def execute_onnx_model(
    source: str | Path | bytes | bytearray | OnnxGraph,
    inputs: Mapping[str, np.ndarray],
    *,
    check_finite: bool = True,
) -> dict[str, np.ndarray]:
    """Convenience wrapper: parse and execute in one call."""
    return OnnxEmbeddedExecutor(source).execute(inputs, check_finite=check_finite)


class EmbeddedOnnxRuntime:
    """Runtime adapter with the same shape as the other execution backends."""

    name = "onnx-embedded"

    def __init__(self, model: str | Path | bytes | bytearray | OnnxGraph) -> None:
        self.executor = OnnxEmbeddedExecutor(model)

    def execute(self, inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        return self.executor.execute(inputs)

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "device": "cpu",
            "executor": "uzen.numpy",
            "accelerated": False,
            "operators": sorted(SUPPORTED_OPS),
        }


__all__ = [
    "SUPPORTED_OPS",
    "EmbeddedOnnxRuntime",
    "OnnxEmbeddedExecutor",
    "OnnxGraph",
    "OnnxNode",
    "execute_onnx_model",
    "parse_onnx_graph",
]
