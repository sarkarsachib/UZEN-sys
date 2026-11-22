"""NIR graph execution on a CUDA device.

Arithmetic runs in the kernels from ``kernels.cu``; layout operators (reshape,
transpose, split/merge, RoPE indexing, concatenation, selection) move bytes and use
NumPy on the host, exactly as the native CPU runtime does. They are reported
separately (:attr:`CudaExecution.layout_ops`) so a run never hides a host round
trip.

There is no silent fallback inside the executor. An operator, rank, dtype or
broadcast shape the device path does not implement raises
:class:`UnsupportedOperator`; choosing the NumPy runtime instead is an explicit
policy decision made by :class:`CudaRuntime`.

Weights are uploaded once per ``(package fingerprint, node, parameter)`` and kept
resident for the life of the runtime. Activations are pooled and released when
their last consumer has run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from uzen.core.nir.executor import resolve_bindings
from uzen.core.nir.graph import Graph
from uzen.core.nir.ops import get_operator, port_order
from uzen.core.nir.types import replace_symbols
from uzen.core.nws.tensors import WeightPackage
from uzen.errors import (
    InvalidOperationParameters,
    ShapeMismatch,
    TensorInvalid,
    UnsupportedOperator,
)
from uzen.runtimes.native.executor import (
    GRAPH_INPUT,
    _axes,
    _key,
    _name,
    _resolve_params,
    _rope,
    _split_qkv,
)

from .allocator import DeviceAllocator
from .driver import Arg, CudaContext, CudaModule, DeviceBuffer, f32, i32, null_ptr, ptr
from .nvrtc import KERNELS

UNARY_KERNELS = {
    "gelu": "uzen_gelu_tanh_f32",
    "gelu_exact": "uzen_gelu_exact_f32",
    "silu": "uzen_silu_f32",
    "relu": "uzen_relu_f32",
    "tanh": "uzen_tanh_f32",
    "sigmoid": "uzen_sigmoid_f32",
}
BINARY_KERNELS = {"add": 0, "sub": 1, "mul": 2}
LAYOUT_OPS = frozenset(
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
DEVICE_OPS = frozenset(
    {
        "linear",
        "matmul",
        "layernorm",
        "rmsnorm",
        "batchnorm",
        "softmax",
        "embedding",
        "mean",
        "sum",
        "attention",
        "scale",
        "clip",
        *UNARY_KERNELS,
        *BINARY_KERNELS,
    }
)


@dataclass
class DeviceTensor:
    buffer: DeviceBuffer
    shape: tuple[int, ...]
    dtype: np.dtype

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize


@dataclass
class CudaExecution:
    outputs: dict[str, np.ndarray]
    intermediates: dict[str, np.ndarray] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    num_ops: int = 0
    device_ops: list[str] = field(default_factory=list)
    layout_ops: list[str] = field(default_factory=list)
    kernels_launched: int = 0
    backend: str = "cuda"
    device: str = "cuda:0"
    reason: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)

    def output(self, name: str) -> np.ndarray:
        try:
            return self.outputs[name]
        except KeyError as exc:
            raise TensorInvalid(
                f"graph produced no output named '{name}'; available: "
                f"{sorted(self.outputs)}"
            ) from exc

    def summary(self) -> dict[str, Any]:
        payload = {
            "backend": self.backend,
            "device": self.device,
            "outputs": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in self.outputs.items()
            },
            "num_ops": self.num_ops,
            "device_ops": list(self.device_ops),
            "layout_ops": list(self.layout_ops),
            "kernels_launched": self.kernels_launched,
            "captured_intermediates": sorted(self.intermediates),
            "memory": dict(self.stats),
        }
        if self.reason:
            payload["reason"] = self.reason
        return payload


def _shape(value: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(item) for item in value)


def _count(shape: Sequence[int]) -> int:
    return int(np.prod(tuple(shape), dtype=np.int64))


class CudaExecutor:
    """Launch kernels for one graph against one context/allocator."""

    def __init__(
        self,
        context: CudaContext,
        module: CudaModule,
        allocator: DeviceAllocator,
        *,
        weight_cache: dict[tuple[str, str, str], DeviceBuffer],
        package_fingerprint: str,
        device_label: str = "cuda:0",
    ) -> None:
        self.context = context
        self.module = module
        self.allocator = allocator
        self.weight_cache = weight_cache
        self.package_fingerprint = package_fingerprint
        self.device_label = device_label
        self._functions = {name: module.function(name) for name in KERNELS}
        self._launches = 0

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _new(self, shape: Sequence[int], dtype: Any = np.float32) -> DeviceTensor:
        shape_tuple = _shape(shape)
        nbytes = _count(shape_tuple) * np.dtype(dtype).itemsize
        if nbytes == 0:
            # cuMemAlloc rejects zero bytes; use one byte and never launch.
            return DeviceTensor(self.allocator.allocate(1), shape_tuple, np.dtype(dtype))
        return DeviceTensor(self.allocator.allocate(nbytes), shape_tuple, np.dtype(dtype))

    def _upload(self, array: Any, dtype: Any | None = None) -> DeviceTensor:
        value = np.ascontiguousarray(array if dtype is None else np.asarray(array, dtype=dtype))
        if value.dtype not in (np.float32, np.int64):
            raise UnsupportedOperator(
                f"the CUDA executor uploads float32 and int64 only, got {value.dtype}"
            )
        return DeviceTensor(
            self.allocator.upload(value), tuple(value.shape), value.dtype
        )

    def _download(self, tensor: DeviceTensor) -> np.ndarray:
        return self.allocator.download_array(tensor.buffer, tensor.shape, tensor.dtype)

    def _weight(self, node_id: str, name: str, value: np.ndarray) -> DeviceTensor:
        array = np.ascontiguousarray(value, dtype=np.float32)
        key = (self.package_fingerprint, node_id, name)
        buffer = self.weight_cache.get(key)
        if buffer is None:
            buffer = self.allocator.upload(array)
            self.weight_cache[key] = buffer
        return DeviceTensor(buffer, tuple(array.shape), np.dtype(np.float32))

    def _launch(
        self,
        kernel: str,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        arguments: Sequence[Arg],
        *,
        shared_bytes: int = 0,
    ) -> None:
        self._functions[kernel].launch(
            grid, block, arguments, shared_bytes=shared_bytes
        )
        self._launches += 1

    def _elementwise(
        self, kernel: str, x: DeviceTensor, *, extra: Sequence[Arg] = ()
    ) -> DeviceTensor:
        output = self._new(x.shape)
        count = _count(x.shape)
        if count:
            self._launch(
                kernel,
                ((count + 255) // 256, 1, 1),
                (256, 1, 1),
                [ptr(x.buffer), ptr(output.buffer), i32(count), *extra],
            )
        return output

    def _release(self, key: str, env: dict[str, DeviceTensor], protected: set[str]) -> None:
        if key in protected:
            return
        tensor = env.pop(key, None)
        if tensor is not None:
            self.allocator.free(tensor.buffer)

    # ------------------------------------------------------------------ #
    # execution
    # ------------------------------------------------------------------ #

    def execute(
        self,
        graph: Graph,
        inputs: Mapping[str, np.ndarray],
        package: WeightPackage,
        *,
        capture: Sequence[str] | None = None,
        check_finite: bool = True,
        validate_types: bool = True,
    ) -> CudaExecution:
        self.context.make_current()
        self._launches = 0
        package.assert_unmodified()
        bound = resolve_bindings(graph, package)

        missing = [name for name in graph.inputs if name not in inputs]
        if missing:
            raise ShapeMismatch(f"missing graph inputs: {missing}")
        extra_inputs = [name for name in inputs if name not in graph.inputs]
        if extra_inputs:
            raise ShapeMismatch(f"unexpected graph inputs: {extra_inputs}")

        env: dict[str, DeviceTensor] = {}
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
            if array.dtype not in (np.float32, np.int64):
                raise UnsupportedOperator(
                    f"the CUDA executor accepts float32 and int64 graph inputs; "
                    f"input '{name}' is {array.dtype}"
                )
            env[f"{GRAPH_INPUT}{name}"] = self._upload(array)

        order = graph.topological_order()
        captures = set(capture or ())
        captured_tensors: dict[str, DeviceTensor] = {}
        device_ops: list[str] = []
        layout_ops: list[str] = []
        protected = {_key(ref) for ref in graph.outputs.values()}
        protected.update(f"{node_id}:0" for node_id in captures)

        uses: dict[str, int] = {}
        for node_id in order:
            node = graph.nodes[node_id]
            operator = get_operator(node.op)
            for port in port_order(operator, node.inputs):
                uses[_key(node.inputs[port])] = uses.get(_key(node.inputs[port]), 0) + 1
        for key in protected:
            uses[key] = uses.get(key, 0) + 1

        stats_before = self.allocator.snapshot()
        for node_id in order:
            node = graph.nodes[node_id]
            operator = get_operator(node.op)
            keys: list[str] = []
            arrays: list[DeviceTensor] = []
            for port in port_order(operator, node.inputs):
                key = _key(node.inputs[port])
                if key not in env:
                    raise ShapeMismatch(
                        f"node '{node_id}': missing value for input port '{port}'"
                    )
                keys.append(key)
                arrays.append(env[key])
            params = _resolve_params(node.params, symbols)
            outputs, kind = self._dispatch(
                node, arrays, bound.get(node_id, {}), params, symbols
            )
            (device_ops if kind == "device" else layout_ops).append(node.op)

            declared = None
            if validate_types and node.outputs:
                try:
                    declared = tuple(
                        int(item) for item in replace_symbols(node.outputs[0].shape, symbols)
                    )
                except (KeyError, TypeError):
                    declared = None
            if declared is not None and tuple(outputs[0].shape) != declared:
                raise ShapeMismatch(
                    f"node '{node_id}' ({node.op}) produced shape "
                    f"{outputs[0].shape}, declared {declared}"
                )

            for index, output in enumerate(outputs):
                env[f"{node_id}:{index}"] = output
            if node_id in captures:
                captured_tensors[node_id] = outputs[0]
            for key in keys:
                uses[key] -= 1
                if uses[key] <= 0:
                    self._release(key, env, protected)

        self.context.synchronize()
        package.assert_unmodified()

        output_arrays: dict[str, np.ndarray] = {}
        for name, ref in graph.outputs.items():
            key = _key(ref)
            if key not in env:
                raise ShapeMismatch(
                    f"graph output '{name}' references missing value {ref}"
                )
            value = self._download(env[key])
            if check_finite and not np.isfinite(value).all():
                raise TensorInvalid(
                    f"graph output '{name}' contains non-finite values"
                )
            output_arrays[name] = value

        captured: dict[str, np.ndarray] = {}
        for node_id, tensor in captured_tensors.items():
            value = self._download(tensor)
            if check_finite and not np.isfinite(value).all():
                raise TensorInvalid(
                    f"captured node '{node_id}' produced non-finite values"
                )
            captured[node_id] = value

        for key in list(env):
            self.allocator.free(env[key].buffer)
            env.pop(key, None)

        stats_after = self.allocator.snapshot()
        stats = {
            key: stats_after[key]
            for key in (
                "host_to_device_bytes",
                "device_to_host_bytes",
                "peak_live_bytes",
                "reserved_bytes",
                "allocations",
                "reused_blocks",
                "uploads",
                "downloads",
            )
        }
        stats["execution_host_to_device_bytes"] = (
            stats_after["host_to_device_bytes"] - stats_before["host_to_device_bytes"]
        )
        stats["execution_device_to_host_bytes"] = (
            stats_after["device_to_host_bytes"] - stats_before["device_to_host_bytes"]
        )
        return CudaExecution(
            outputs=output_arrays,
            intermediates=captured,
            order=order,
            num_ops=len(order),
            device_ops=device_ops,
            layout_ops=layout_ops,
            kernels_launched=self._launches,
            backend="cuda",
            device=self.device_label,
            stats=stats,
        )

    # ------------------------------------------------------------------ #
    # dispatch
    # ------------------------------------------------------------------ #

    def _dispatch(
        self,
        node: Any,
        inputs: list[DeviceTensor],
        bound: Mapping[str, np.ndarray],
        params: Mapping[str, Any],
        symbols: Mapping[str, int],
    ) -> tuple[list[DeviceTensor], str]:
        op = node.op
        x = inputs[0] if inputs else None

        if op == "linear":
            weight = self._bound_weight(node, params, bound, "weight", "weight")
            bias = (
                self._bound_weight(node, params, bound, "bias", "bias")
                if params.get("bias")
                else None
            )
            return [self._linear(x, weight, bias)], "device"

        if op == "matmul":
            return [self._matmul(inputs[0], inputs[1])], "device"

        if op == "layernorm":
            weight = (
                self._bound_weight(node, params, bound, "weight", "weight")
                if params.get("weight")
                else None
            )
            bias = (
                self._bound_weight(node, params, bound, "bias", "bias")
                if params.get("bias")
                else None
            )
            return [
                self._norm(
                    "uzen_layernorm_f32",
                    x,
                    weight,
                    bias,
                    float(params.get("eps", 1e-5)),
                )
            ], "device"

        if op == "rmsnorm":
            weight = (
                self._bound_weight(node, params, bound, "weight", "weight")
                if params.get("weight")
                else None
            )
            return [
                self._norm(
                    "uzen_rmsnorm_f32", x, weight, None, float(params.get("eps", 1e-6))
                )
            ], "device"

        if op == "batchnorm":
            mean = self._bound_weight(
                node, params, bound, "running_mean", "running_mean"
            )
            variance = self._bound_weight(
                node, params, bound, "running_var", "running_var"
            )
            weight = (
                self._bound_weight(node, params, bound, "weight", "weight")
                if params.get("weight")
                else None
            )
            bias = (
                self._bound_weight(node, params, bound, "bias", "bias")
                if params.get("bias")
                else None
            )
            if params.get("axis", -1) not in (-1, x.ndim - 1):
                raise UnsupportedOperator(
                    "CUDA batchnorm applies its statistics along the last axis; "
                    f"node '{node.id}' requested axis {params.get('axis')}",
                    node=node.id,
                    operator=op,
                )
            return [
                self._batchnorm(x, weight, bias, mean, variance, float(params.get("eps", 1e-5)))
            ], "device"

        if op == "softmax":
            axes = _axes(params.get("axis", -1), x.ndim)
            if axes != (x.ndim - 1,):
                raise UnsupportedOperator(
                    f"CUDA softmax supports only the last axis, got axes {axes} "
                    f"for node '{node.id}'",
                    node=node.id,
                    operator=op,
                )
            return [self._softmax(x)], "device"

        if op in UNARY_KERNELS:
            return [self._elementwise(UNARY_KERNELS[op], x)], "device"

        if op in BINARY_KERNELS:
            if inputs[0].shape != inputs[1].shape:
                raise UnsupportedOperator(
                    f"CUDA {op} requires identical shapes; got {inputs[0].shape} and "
                    f"{inputs[1].shape} for node '{node.id}' (broadcasting is not "
                    "implemented on the device path)",
                    node=node.id,
                    operator=op,
                )
            output = self._new(x.shape)
            count = _count(x.shape)
            if count:
                self._launch(
                    "uzen_binary_f32",
                    ((count + 255) // 256, 1, 1),
                    (256, 1, 1),
                    [
                        ptr(inputs[0].buffer),
                        ptr(inputs[1].buffer),
                        ptr(output.buffer),
                        i32(count),
                        i32(BINARY_KERNELS[op]),
                    ],
                )
            return [output], "device"

        if op == "scale":
            return [
                self._elementwise(
                    "uzen_scale_f32", x, extra=[f32(float(params.get("factor", 1.0)))]
                )
            ], "device"

        if op == "clip":
            return [
                self._elementwise(
                    "uzen_clip_f32",
                    x,
                    extra=[
                        f32(float(params.get("min", -1.0))),
                        f32(float(params.get("max", 1.0))),
                    ],
                )
            ], "device"

        if op == "embedding":
            weight = self._bound_weight(node, params, bound, "weight", "weight")
            return [self._embedding(x, weight)], "device"

        if op in ("mean", "sum"):
            axes = _axes(params.get("axis", -1), x.ndim)
            if axes != (x.ndim - 1,):
                raise UnsupportedOperator(
                    f"CUDA {op} supports only the last axis, got axes {axes} for "
                    f"node '{node.id}'",
                    node=node.id,
                    operator=op,
                )
            return [
                self._reduce_last(x, mode=op, keepdims=bool(params.get("keepdims", False)))
            ], "device"

        if op == "attention":
            if len(inputs) != 3:
                raise InvalidOperationParameters("attention requires (q, k, v)")
            slopes = params.get("alibi_slopes")
            if slopes is not None and len(slopes) == 0:
                slopes = None
            return [
                self._attention(
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    scale=params.get("scale", "auto"),
                    causal=bool(params.get("causal", True)),
                    slopes=np.asarray(slopes, dtype=np.float32) if slopes is not None else None,
                    window=(
                        int(params["window"]) if params.get("window") is not None else None
                    ),
                )
            ], "device"

        if op in LAYOUT_OPS:
            return self._layout(node, inputs, bound, params, symbols), "layout"

        if op == "quantized_linear":
            raise UnsupportedOperator(
                "the CUDA executor does not implement int8 quantized_linear; "
                "dequantize explicitly and use float32, or run the NumPy/CPU "
                "runtime",
                node=node.id,
                operator=op,
            )

        raise UnsupportedOperator(
            f"the CUDA executor does not implement NIR operator '{op}' (node "
            f"'{node.id}'); device arithmetic: {', '.join(sorted(DEVICE_OPS))}; "
            f"host layout: {', '.join(sorted(LAYOUT_OPS))}",
            node=node.id,
            operator=op,
        )

    # ------------------------------------------------------------------ #
    # device kernels
    # ------------------------------------------------------------------ #

    def _bound_weight(
        self,
        node: Any,
        params: Mapping[str, Any],
        bound: Mapping[str, np.ndarray],
        parameter: str,
        default: str,
    ) -> DeviceTensor:
        name = _name(params.get(parameter), default)
        try:
            array = bound[name]
        except KeyError as exc:
            raise ShapeMismatch(
                f"node '{node.id}' ({node.op}) has no bound tensor '{name}'"
            ) from exc
        return self._weight(node.id, name, array)

    def _linear(
        self, x: DeviceTensor, weight: DeviceTensor, bias: DeviceTensor | None
    ) -> DeviceTensor:
        if x.ndim < 1:
            raise ShapeMismatch("CUDA linear requires an input of rank >= 1")
        if len(weight.shape) != 2 or x.shape[-1:] != weight.shape[1:]:
            raise ShapeMismatch(
                f"CUDA linear requires weight (out, in) matching the input's last "
                f"dimension; got input {x.shape}, weight {weight.shape}"
            )
        out_features = weight.shape[0]
        in_features = weight.shape[1]
        rows = _count(x.shape[:-1]) if x.ndim > 1 else 1
        if bias is not None and bias.shape != (out_features,):
            raise ShapeMismatch(
                f"CUDA linear bias {bias.shape} != ({out_features},)"
            )
        output_shape = tuple(x.shape[:-1]) + (out_features,)
        if rows and out_features and not in_features:
            return self._upload(np.zeros(output_shape, dtype=np.float32))
        output = self._new(output_shape)
        if rows and out_features:
            self._launch(
                "uzen_linear_f32",
                ((out_features + 15) // 16, (rows + 15) // 16, 1),
                (16, 16, 1),
                [
                    ptr(x.buffer),
                    ptr(weight.buffer),
                    ptr(bias.buffer) if bias is not None else null_ptr(),
                    ptr(output.buffer),
                    i32(rows),
                    i32(in_features),
                    i32(out_features),
                ],
            )
        return output

    def _matmul(self, left: DeviceTensor, right: DeviceTensor) -> DeviceTensor:
        if len(left.shape) != 2 or len(right.shape) != 2:
            raise UnsupportedOperator(
                "CUDA matmul supports rank-2 operands only; use linear for batched "
                "projections"
            )
        if left.shape[1] != right.shape[0]:
            raise ShapeMismatch(
                f"CUDA matmul inner dimensions disagree: {left.shape} x {right.shape}"
            )
        output_shape = (left.shape[0], right.shape[1])
        if left.shape[0] and right.shape[1] and not left.shape[1]:
            return self._upload(np.zeros(output_shape, dtype=np.float32))
        output = self._new(output_shape)
        if left.shape[0] and right.shape[1] and left.shape[1]:
            self._launch(
                "uzen_matmul_f32",
                ((right.shape[1] + 15) // 16, (left.shape[0] + 15) // 16, 1),
                (16, 16, 1),
                [
                    ptr(left.buffer),
                    ptr(right.buffer),
                    ptr(output.buffer),
                    i32(left.shape[0]),
                    i32(left.shape[1]),
                    i32(right.shape[1]),
                ],
            )
        return output

    def _norm(
        self,
        kernel: str,
        x: DeviceTensor,
        weight: DeviceTensor | None,
        bias: DeviceTensor | None,
        eps: float,
    ) -> DeviceTensor:
        if x.ndim < 1:
            raise ShapeMismatch(f"CUDA {kernel} requires rank >= 1")
        cols = x.shape[-1]
        rows = _count(x.shape[:-1]) if x.ndim > 1 else 1
        for label, parameter in (("weight", weight), ("bias", bias)):
            if parameter is not None and parameter.shape != (cols,):
                raise ShapeMismatch(
                    f"CUDA {label} {parameter.shape} != ({cols},) for {kernel}"
                )
        output = self._new(x.shape)
        if rows and cols:
            arguments: list[Arg] = [
                ptr(x.buffer),
                ptr(weight.buffer) if weight is not None else null_ptr(),
            ]
            if kernel != "uzen_rmsnorm_f32":
                arguments.append(ptr(bias.buffer) if bias is not None else null_ptr())
            arguments.extend(
                [ptr(output.buffer), i32(rows), i32(cols), f32(eps)]
            )
            self._launch(kernel, (rows, 1, 1), (256, 1, 1), arguments)
        return output

    def _batchnorm(
        self,
        x: DeviceTensor,
        weight: DeviceTensor | None,
        bias: DeviceTensor | None,
        running_mean: DeviceTensor,
        running_var: DeviceTensor,
        eps: float,
    ) -> DeviceTensor:
        if x.ndim < 2:
            raise ShapeMismatch("CUDA batchnorm requires a rank >= 2 input")
        cols = x.shape[-1]
        rows = _count(x.shape[:-1])
        for label, parameter in (
            ("weight", weight),
            ("bias", bias),
            ("running_mean", running_mean),
            ("running_var", running_var),
        ):
            if parameter is not None and parameter.shape != (cols,):
                raise ShapeMismatch(
                    f"CUDA batchnorm {label} {parameter.shape} != ({cols},)"
                )
        output = self._new(x.shape)
        if rows and cols:
            self._launch(
                "uzen_batchnorm_f32",
                (rows, 1, 1),
                (256, 1, 1),
                [
                    ptr(x.buffer),
                    ptr(weight.buffer) if weight is not None else null_ptr(),
                    ptr(bias.buffer) if bias is not None else null_ptr(),
                    ptr(running_mean.buffer),
                    ptr(running_var.buffer),
                    ptr(output.buffer),
                    i32(rows),
                    i32(cols),
                    f32(eps),
                ],
            )
        return output

    def _softmax(self, x: DeviceTensor) -> DeviceTensor:
        if x.ndim < 1:
            raise ShapeMismatch("CUDA softmax requires rank >= 1")
        cols = x.shape[-1]
        rows = _count(x.shape[:-1]) if x.ndim > 1 else 1
        output = self._new(x.shape)
        if rows and cols:
            self._launch(
                "uzen_softmax_f32",
                (rows, 1, 1),
                (256, 1, 1),
                [ptr(x.buffer), ptr(output.buffer), i32(rows), i32(cols)],
            )
        return output

    def _embedding(self, indices: DeviceTensor, weight: DeviceTensor) -> DeviceTensor:
        if indices.dtype.kind not in "iu":
            raise InvalidOperationParameters(
                f"CUDA embedding requires integer indices, got {indices.dtype}"
            )
        if len(weight.shape) != 2:
            raise InvalidOperationParameters("CUDA embedding weight must be 2-D")
        if _count(indices.shape):
            host_indices = self._download(indices)
            minimum = int(host_indices.min())
            maximum = int(host_indices.max())
            if minimum < 0 or maximum >= weight.shape[0]:
                raise InvalidOperationParameters(
                    f"CUDA embedding index out of range [0, {weight.shape[0]}): "
                    f"min={minimum}, max={maximum}"
                )
        tokens = _count(indices.shape)
        output = self._new(tuple(indices.shape) + (weight.shape[1],))
        if tokens and weight.shape[1]:
            self._launch(
                "uzen_embedding_f32",
                ((tokens * weight.shape[1] + 255) // 256, 1, 1),
                (256, 1, 1),
                [
                    ptr(indices.buffer),
                    ptr(weight.buffer),
                    ptr(output.buffer),
                    i32(tokens),
                    i32(weight.shape[1]),
                ],
            )
        return output

    def _reduce_last(
        self, x: DeviceTensor, *, mode: str, keepdims: bool
    ) -> DeviceTensor:
        if x.ndim < 1:
            raise ShapeMismatch(f"CUDA {mode} requires rank >= 1")
        cols = x.shape[-1]
        rows = _count(x.shape[:-1]) if x.ndim > 1 else 1
        shape = tuple(x.shape[:-1]) + ((1,) if keepdims else ())
        output = self._new(shape)
        if rows and cols:
            self._launch(
                "uzen_reduce_last_f32",
                (rows, 1, 1),
                (256, 1, 1),
                [
                    ptr(x.buffer),
                    ptr(output.buffer),
                    i32(rows),
                    i32(cols),
                    i32(0 if mode == "mean" else 1),
                ],
            )
        return output

    def _attention(
        self,
        q: DeviceTensor,
        k: DeviceTensor,
        v: DeviceTensor,
        *,
        scale: Any,
        causal: bool,
        slopes: np.ndarray | None,
        window: int | None,
    ) -> DeviceTensor:
        if not (len(q.shape) == len(k.shape) == len(v.shape) == 4):
            raise UnsupportedOperator(
                "CUDA attention requires (batch, heads, seq, head_dim) tensors"
            )
        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
            raise ShapeMismatch("CUDA attention batch dimensions disagree")
        if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
            raise ShapeMismatch("CUDA attention head dimensions disagree")
        if q.shape[3] != k.shape[3] or q.shape[3] != v.shape[3]:
            raise ShapeMismatch("CUDA attention head_dim dimensions disagree")
        if k.shape[2] != v.shape[2]:
            raise ShapeMismatch("CUDA attention k/v sequence lengths disagree")
        if q.shape[2] > k.shape[2]:
            raise ShapeMismatch(
                "CUDA attention q sequence cannot exceed the k/v sequence"
            )
        if slopes is not None and slopes.shape != (q.shape[1],):
            raise ShapeMismatch(
                f"CUDA attention alibi {slopes.shape} != ({q.shape[1]},)"
            )
        head_dim = q.shape[3]
        if scale is None or scale == "auto":
            scale_value = 1.0 / float(np.sqrt(head_dim))
        else:
            scale_value = float(scale)
        slope_tensor = self._upload(slopes) if slopes is not None else None
        output = self._new(q.shape)
        if _count(q.shape[:3]) and k.shape[2]:
            self._launch(
                "uzen_attention_f32",
                (q.shape[2], q.shape[1], q.shape[0]),
                (128, 1, 1),
                [
                    ptr(q.buffer),
                    ptr(k.buffer),
                    ptr(v.buffer),
                    ptr(output.buffer),
                    i32(q.shape[0]),
                    i32(q.shape[1]),
                    i32(q.shape[2]),
                    i32(k.shape[2]),
                    i32(head_dim),
                    f32(scale_value),
                    i32(1 if causal else 0),
                    i32(window if window is not None else -1),
                    ptr(slope_tensor.buffer) if slope_tensor is not None else null_ptr(),
                ],
                shared_bytes=int(k.shape[2]) * 4,
            )
        if slope_tensor is not None:
            self.allocator.free(slope_tensor.buffer)
        return output

    # ------------------------------------------------------------------ #
    # layout operators (host byte movement, reported as such)
    # ------------------------------------------------------------------ #

    def _layout(
        self,
        node: Any,
        inputs: list[DeviceTensor],
        bound: Mapping[str, np.ndarray],
        params: Mapping[str, Any],
        symbols: Mapping[str, int],
    ) -> list[DeviceTensor]:
        arrays = [self._download(tensor) for tensor in inputs]
        op = node.op
        x = arrays[0] if arrays else None

        if op == "pos_embedding":
            weight = bound[_name(params.get("weight"), "weight")]
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
            results = [x + table]

        elif op == "split_qkv":
            results = _split_qkv(x, params)

        elif op == "merge_heads":
            batch, heads, sequence, head_dim = x.shape
            results = [
                np.transpose(x, (0, 2, 1, 3)).reshape(
                    batch, sequence, heads * head_dim
                )
            ]

        elif op == "rope":
            if len(arrays) != 2:
                raise InvalidOperationParameters("rope requires (x, positions)")
            results = [_rope(x, arrays[1], params)]

        elif op == "repeat_kv":
            heads = int(params["n_heads"])
            if x.ndim != 4 or heads % x.shape[1]:
                raise InvalidOperationParameters(
                    f"repeat_kv requires n_heads divisible by kv_heads, got "
                    f"n_heads={heads}, kv_heads={x.shape[1] if x.ndim == 4 else '?'}"
                )
            results = [np.repeat(x, heads // x.shape[1], axis=1)]

        elif op == "reshape":
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
                        f"CUDA reshape cannot resolve dimension {dimension!r}"
                    )
            results = [x.reshape(tuple(shape))]

        elif op == "transpose":
            permutation = params.get("perm")
            if permutation is None:
                permutation = list(reversed(range(x.ndim)))
            results = [np.transpose(x, tuple(int(value) for value in permutation))]

        elif op == "slice":
            axis = int(params.get("axis", 0)) % x.ndim
            index = [slice(None)] * x.ndim
            index[axis] = slice(
                params.get("start"), params.get("stop"), params.get("step")
            )
            results = [x[tuple(index)]]

        elif op == "flatten":
            start = int(params.get("start_dim", 1))
            results = [x.reshape(tuple(x.shape[:start]) + (-1,))]

        elif op == "concat":
            results = [np.concatenate(arrays, axis=int(params.get("axis", -1)))]

        elif op in ("identity", "dropout"):
            results = [x]

        elif op == "select":
            results = [np.where(arrays[0] != 0, arrays[1], arrays[2])]

        else:  # pragma: no cover - guarded by LAYOUT_OPS
            raise UnsupportedOperator(
                f"the CUDA runtime does not implement layout operator '{op}'"
            )

        uploaded: list[DeviceTensor] = []
        for result in results:
            array = np.asarray(result)
            if array.dtype.kind == "f":
                array = array.astype(np.float32, copy=False)
            uploaded.append(self._upload(array))
        return uploaded
