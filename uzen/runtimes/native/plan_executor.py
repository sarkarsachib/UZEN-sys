"""Execution of compiled native plans.

The executor walks a :class:`~uzen.runtimes.native.compiler.NativePlan`, reading
its inputs from graph inputs or planned buffers and writing each result into the
planned buffer slot. Arithmetic goes through the compiled kernels; layout
operators (byte movement only) reuse the direct executor's NumPy handlers, which
are reported separately in the execution summary.

There is no silent fallback: an operator the plan does not contain cannot appear
here, and a value that is not float32 is refused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from uzen.core.nir.executor import resolve_bindings
from uzen.core.nir.graph import Graph
from uzen.core.nws.tensors import WeightPackage
from uzen.errors import (
    InvalidOperationParameters,
    ShapeMismatch,
    TensorInvalid,
)

from . import executor as direct
from .compiler import BoundPlan, NativePlan, PlanStep, compile_native_plan
from .library import ACTIVATION_CODES, NativeLibrary

GRAPH_INPUT = "input:"


@dataclass
class NativePlanExecution:
    """Outputs plus the evidence of what the compiled plan actually did."""

    outputs: dict[str, np.ndarray]
    intermediates: dict[str, np.ndarray] = field(default_factory=dict)
    native_ops: list[str] = field(default_factory=list)
    layout_ops: list[str] = field(default_factory=list)
    fused_steps: int = 0
    steps: int = 0
    prepacked: int = 0
    memory: dict[str, Any] = field(default_factory=dict)
    plan: dict[str, Any] = field(default_factory=dict)
    representations: dict[str, Any] = field(default_factory=dict)

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
            "compiled": True,
            "steps": self.steps,
            "fused_steps": self.fused_steps,
            "native_ops": list(self.native_ops),
            "layout_ops": list(self.layout_ops),
            "prepacked_weights": self.prepacked,
            "representations": dict(self.representations),
            "representation_bytes": int(
                sum(
                    int(item.get("packed_bytes", 0))
                    for item in self.representations.values()
                )
            ),
            "memory": dict(self.memory),
            "plan": dict(self.plan),
            "outputs": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in self.outputs.items()
            },
            "captured_intermediates": sorted(self.intermediates),
        }


@dataclass
class _Context:
    package: WeightPackage
    bound: dict[str, dict[str, np.ndarray]]
    symbols: dict[str, int]
    library: NativeLibrary
    packed: Mapping[str, Any]
    graph: Graph

    def tensor(self, name: str | None, *, step: PlanStep) -> np.ndarray | None:
        if not name:
            return None
        if name not in self.package:
            raise ShapeMismatch(
                f"plan step {step.index} ({step.op}) needs bound tensor '{name}', "
                f"which is missing from package '{self.package.name}'"
            )
        return self.package.get(name)


def _weights(*names: str | None, context: _Context, step: PlanStep) -> list[Any]:
    resolved = [context.tensor(name, step=step) for name in names]
    if any(value is None for value in resolved):
        raise InvalidOperationParameters(
            f"plan step {step.index} ({step.op}) has an unbound parameter reference"
        )
    return resolved


def _run_step(
    step: PlanStep, arrays: list[np.ndarray], context: _Context
) -> list[np.ndarray]:
    op = step.op
    attrs = step.attrs
    library = context.library

    if op == "gemm_packed":
        name = attrs.get("weight")
        if not name or name not in context.packed:
            raise ShapeMismatch(
                f"plan step {step.index} needs prepacked weight '{name}', which was "
                "not prepared"
            )
        prepared = context.packed[name]
        bias = context.tensor(attrs.get("bias"), step=step)
        activation = str(attrs.get("activation", "none"))
        kind = str(getattr(prepared, "kind", "dense_blocked"))
        if kind == "dense_blocked":
            return [
                library.gemm_packed(
                    arrays[0], prepared.packed, bias=bias, activation=activation
                )
            ]
        if kind == "transposed":
            return [
                library.gemm_transposed(
                    arrays[0], prepared.packed, bias=bias, activation=activation
                )
            ]
        if kind == "int8_block":
            return [
                library.gemm_int8_block(
                    arrays[0], prepared.packed, bias=bias, activation=activation
                )
            ]
        if kind == "int4_block":
            return [
                library.gemm_int4_block(
                    arrays[0], prepared.packed, bias=bias, activation=activation
                )
            ]
        if kind == "low_rank":
            if prepared.factors is None:
                raise InvalidOperationParameters(
                    f"plan step {step.index} needs low-rank factors for '{name}'"
                )
            u, v = prepared.factors
            return [
                library.gemm_low_rank(
                    arrays[0], v, u, bias=bias, activation=activation
                )
            ]
        if kind == "sparse_csr":
            if prepared.sparse is None:
                raise InvalidOperationParameters(
                    f"plan step {step.index} needs a CSR payload for '{name}'"
                )
            values, indices, row_ptr = prepared.sparse
            return [
                library.sparse_gemm(
                    arrays[0],
                    values,
                    indices,
                    row_ptr,
                    prepared.in_features,
                    prepared.out_features,
                    bias=bias,
                    activation=activation,
                )
            ]
        raise InvalidOperationParameters(
            f"plan step {step.index} uses unknown weight representation '{kind}' "
            f"for '{name}'"
        )

    if op == "add_norm":
        weight = context.tensor(attrs.get("weight"), step=step)
        bias = context.tensor(attrs.get("bias"), step=step)
        kind = str(attrs["kind"])
        store_residual = bool(attrs.get("store_residual", False))
        normalized, residual = library.add_norm(
            arrays[0],
            arrays[1],
            weight=weight,
            bias=bias,
            eps=float(attrs.get("eps", 1e-5)),
            kind=kind,
            residual=store_residual,
        )
        if store_residual:
            if residual is None:  # pragma: no cover - defensive
                raise TensorInvalid("add_norm did not produce the requested residual")
            return [residual, normalized]
        return [normalized]

    if op == "layernorm":
        return [
            library.layernorm(
                arrays[0],
                weight=context.tensor(attrs.get("weight"), step=step),
                bias=context.tensor(attrs.get("bias"), step=step),
                eps=float(attrs.get("eps", 1e-5)),
            )
        ]

    if op == "rmsnorm":
        return [
            library.rmsnorm(
                arrays[0],
                weight=context.tensor(attrs.get("weight"), step=step),
                eps=float(attrs.get("eps", 1e-6)),
            )
        ]

    if op == "unary":
        return [library.unary(str(attrs["name"]), arrays[0])]

    if op == "binary":
        return [library.binary(str(attrs["name"]), arrays[0], arrays[1])]

    if op == "scale":
        return [library.scale(arrays[0], float(attrs.get("factor", 1.0)))]

    if op == "clip":
        return [
            library.clip(
                arrays[0],
                float(attrs.get("min", -1.0)),
                float(attrs.get("max", 1.0)),
            )
        ]

    if op == "softmax":
        return [library.softmax(arrays[0])]

    if op == "matmul":
        return [library.matmul(arrays[0], arrays[1])]

    if op == "embedding":
        weight = context.tensor(attrs.get("weight"), step=step)
        if weight is None:
            raise InvalidOperationParameters(
                f"plan step {step.index} (embedding) has no bound weight"
            )
        return [library.embedding(arrays[0], weight)]

    if op == "reduce":
        return [
            library.reduce_axis(
                arrays[0],
                int(attrs.get("axis", -1)),
                mode=str(attrs.get("mode", "mean")),
                keepdims=bool(attrs.get("keepdims", False)),
            )
        ]

    if op == "attention":
        slopes = attrs.get("alibi_slopes")
        return [
            library.attention_flash(
                arrays[0],
                arrays[1],
                arrays[2],
                scale=attrs.get("scale", "auto"),
                causal=bool(attrs.get("causal", True)),
                alibi=(
                    np.asarray(slopes, dtype=np.float32) if slopes else None
                ),
                window=(
                    int(attrs["window"]) if attrs.get("window") is not None else None
                ),
            )
        ]

    if op == "conv2d":
        weight = context.tensor(attrs.get("weight"), step=step)
        if weight is None:
            raise InvalidOperationParameters(
                f"plan step {step.index} (conv2d) has no bound weight"
            )
        return [
            library.conv2d(
                arrays[0],
                weight,
                context.tensor(attrs.get("bias"), step=step),
                stride=attrs.get("stride", 1),
                padding=attrs.get("padding", 0),
            )
        ]

    if op == "maxpool2d":
        return [
            library.maxpool2d(
                arrays[0],
                kernel=attrs.get("kernel", 2),
                stride=attrs.get("stride"),
                padding=attrs.get("padding", 0),
            )
        ]

    if op == "recurrent":
        weight_ih, weight_hh = _weights(
            attrs.get("weight_ih"),
            attrs.get("weight_hh"),
            context=context,
            step=step,
        )
        return [
            library.recurrent(
                arrays[0],
                weight_ih,
                weight_hh,
                context.tensor(attrs.get("bias"), step=step),
                kind=str(attrs["kind"]),
            )
        ]

    if op == "ssm":
        a, b, c = _weights(
            attrs.get("A"), attrs.get("B"), attrs.get("C"), context=context, step=step
        )
        return [
            library.ssm(arrays[0], a, b, c, context.tensor(attrs.get("D"), step=step))
        ]

    if op == "gcn":
        weight = context.tensor(attrs.get("weight"), step=step)
        if weight is None:
            raise InvalidOperationParameters(
                f"plan step {step.index} (gcn) has no bound weight"
            )
        return [
            library.gcn(
                arrays[0],
                arrays[1],
                weight,
                context.tensor(attrs.get("bias"), step=step),
            )
        ]

    if op == "moe":
        router, up_weight, down_weight = _weights(
            attrs.get("router"),
            attrs.get("up_weight"),
            attrs.get("down_weight"),
            context=context,
            step=step,
        )
        return [
            library.moe(
                arrays[0],
                router,
                up_weight,
                down_weight,
                up_bias=context.tensor(attrs.get("up_bias"), step=step),
                down_bias=context.tensor(attrs.get("down_bias"), step=step),
                top_k=int(attrs.get("top_k", 2)),
                activation=str(attrs.get("activation", "gelu")),
            )
        ]

    if op == "quantized_linear":
        weight, scale = _weights(
            attrs.get("weight"), attrs.get("scale"), context=context, step=step
        )
        return [
            library.quantized_linear(
                arrays[0],
                weight,
                scale,
                context.tensor(attrs.get("bias"), step=step),
            )
        ]

    if op == "layout":
        if step.node_id is None:
            raise InvalidOperationParameters(
                f"plan step {step.index} (layout) has no source node"
            )
        node = context.graph.nodes[step.node_id]
        params = direct._resolve_params(node.params, context.symbols)
        outputs, _ = direct._dispatch(
            node,
            arrays,
            context.bound.get(node.id, {}),
            params,
            context.symbols,
            library,
        )
        return list(outputs)

    raise InvalidOperationParameters(
        f"plan step {step.index} uses unknown plan operator '{op}'"
    )


def _step_arrays(
    step: PlanStep,
    values: Mapping[str, np.ndarray],
    bound: BoundPlan,
) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for ref in step.inputs:
        if ref.kind == "input":
            try:
                arrays.append(values[ref.label])
            except KeyError as exc:
                raise ShapeMismatch(
                    f"plan step {step.index} ({step.op}) is missing input '{ref}'"
                ) from exc
        else:
            arrays.append(bound.value(ref))
    return arrays


def execute_plan(
    plan: NativePlan,
    inputs: Mapping[str, np.ndarray],
    package: WeightPackage,
    *,
    capture: Sequence[str] | None = None,
    check_finite: bool = True,
    library: NativeLibrary | None = None,
) -> NativePlanExecution:
    """Run a compiled plan through the native kernels."""
    library = library or NativeLibrary()
    package.assert_unmodified()
    bound_plan = plan.bind(inputs, package)
    values = {f"{GRAPH_INPUT}{name}": np.asarray(value) for name, value in inputs.items()}

    captured_names = set(capture or ())
    for name in captured_names:
        if name not in plan.graph.nodes:
            raise InvalidOperationParameters(
                f"capture target '{name}' is not a node in graph '{plan.graph.name}'"
            )
        if not any(
            target.kind == "node" and target.name == name
            for step in plan.steps
            for target in step.targets
        ):
            raise InvalidOperationParameters(
                f"capture target '{name}' was folded into a fused plan step; compile "
                "with fuse=False to capture this intermediate"
            )

    packed = plan.prepare_weights(package, library)
    representation_reports = {
        name: weight.describe() for name, weight in packed.items()
    }
    context = _Context(
        package=package,
        bound=resolve_bindings(plan.graph, package),
        symbols=dict(bound_plan.symbols),
        library=library,
        packed=packed,
        graph=plan.graph,
    )

    captured: dict[str, np.ndarray] = {}
    native_ops: list[str] = []
    layout_ops: list[str] = []

    for step in plan.steps:
        arrays = _step_arrays(step, values, bound_plan)
        results = _run_step(step, arrays, context)
        if len(results) != len(step.targets):
            raise ShapeMismatch(
                f"plan step {step.index} ({step.op}) produced {len(results)} values "
                f"for {len(step.targets)} planned targets"
            )
        for target, result in zip(step.targets, results):
            array = np.asarray(result)
            if array.dtype != np.float32:
                raise TensorInvalid(
                    f"plan step {step.index} ({step.op}) produced {array.dtype}; the "
                    "native plan is float32-only"
                )
            buffer = bound_plan.memory.buffers[target.label]
            if tuple(array.shape) != buffer.shape:
                raise ShapeMismatch(
                    f"plan step {step.index} ({step.op}) produced shape {array.shape}, "
                    f"planned {buffer.shape} for '{target}'"
                )
            if check_finite and array.size and not np.isfinite(array).all():
                raise TensorInvalid(
                    f"plan step {step.index} ({step.op}) produced non-finite values"
                )
            slot = bound_plan.slots[buffer.slot]
            np.copyto(slot[: buffer.elements], array.reshape(-1), casting="no")
        (native_ops if step.kind == "native" else layout_ops).append(step.op)
        if captured_names:
            for target in step.targets:
                if target.kind == "node" and target.name in captured_names:
                    captured[target.name] = np.array(bound_plan.value(target), copy=True)

    outputs: dict[str, np.ndarray] = {}
    for name, ref in plan.outputs.items():
        if ref.kind == "input":
            outputs[name] = np.array(values[ref.label], copy=True)
        else:
            outputs[name] = np.array(bound_plan.value(ref), copy=True)

    package.assert_unmodified()
    return NativePlanExecution(
        outputs=outputs,
        intermediates=captured,
        native_ops=native_ops,
        layout_ops=layout_ops,
        fused_steps=plan.fused_steps(),
        steps=plan.num_steps,
        prepacked=len(packed),
        memory=bound_plan.memory.describe(),
        plan=plan.describe(),
        representations=representation_reports,
    )


def execute_compiled(
    graph: Graph,
    inputs: Mapping[str, np.ndarray],
    package: WeightPackage,
    *,
    capture: Sequence[str] | None = None,
    fuse: bool = True,
    representations: Mapping[str, Any] | None = None,
    library: NativeLibrary | None = None,
) -> NativePlanExecution:
    """Compile ``graph`` and run it in one call."""
    plan = compile_native_plan(
        graph, package, fuse=fuse, representations=representations
    )
    return execute_plan(
        plan, inputs, package, capture=capture, library=library
    )


__all__ = [
    "NativePlanExecution",
    "compile_native_plan",
    "execute_compiled",
    "execute_plan",
]
