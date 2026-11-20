"""Native execution-plan compiler for NIR graphs (phase N2).

``compile_native_plan`` lowers a validated NIR graph into a linear plan of native
steps over explicitly allocated buffers. Compared with the direct executor
(``executor.execute_native``, which calls one C function per node) the compiler:

* lowers every node onto the compiled kernel set (blocked prepacked GEMM,
  fused residual-add+norm, conv/pool, recurrent, state-space, GCN, sparse MoE,
  int8 quantized linear, single-pass attention);
* fuses ``linear`` + bias + activation into one kernel call, and a residual
  ``add`` feeding exactly one ``layernorm``/``rmsnorm`` into one pass;
* prepacks linear weights into the blocked layout the GEMM consumes;
* plans buffers by liveness so an intermediate can reuse the slot of a buffer
  that is already dead, and reports the peak footprint.

Coverage is explicit: an operator the plan cannot execute raises
``UnsupportedOperator`` at compile time. Nothing falls back silently at run time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from uzen.core.nir.graph import Graph
from uzen.core.nir.ops import get_operator, port_order
from uzen.core.nir.types import TensorType
from uzen.core.nws.tensors import WeightPackage
from uzen.errors import (
    GraphValidationError,
    InvalidOperationParameters,
    ShapeMismatch,
    UnsupportedOperator,
)

GRAPH_INPUT = "input:"

ACTIVATION_CODES = {
    "none": 0,
    "relu": 1,
    "gelu": 2,
    "silu": 3,
    "tanh": 4,
    "sigmoid": 5,
    "gelu_exact": 6,
}
FUSABLE_ACTIVATIONS = frozenset(ACTIVATION_CODES) - {"none"}
NORM_OPS = frozenset({"layernorm", "rmsnorm"})
BINARY_OPS = frozenset({"add", "sub", "mul"})
RECURRENT_OPS = frozenset({"rnn", "lstm", "gru"})
REDUCE_OPS = frozenset({"mean", "sum"})
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

#: Plan-level operator vocabulary. Anything outside this set is refused.
PLAN_OPS = frozenset(
    {
        "gemm_packed",
        "matmul",
        "layernorm",
        "rmsnorm",
        "add_norm",
        "softmax",
        "unary",
        "binary",
        "scale",
        "clip",
        "embedding",
        "reduce",
        "attention",
        "conv2d",
        "maxpool2d",
        "recurrent",
        "ssm",
        "gcn",
        "moe",
        "quantized_linear",
        "layout",
    }
)

_WEIGHT_KEYS = (
    "weight",
    "bias",
    "scale",
    "weight_ih",
    "weight_hh",
    "A",
    "B",
    "C",
    "D",
    "router",
    "up_weight",
    "down_weight",
    "up_bias",
    "down_bias",
)


def param_name(value: Any) -> str | None:
    """Turn a ``param:<name>`` reference into the tensor name, or None."""
    if value is None:
        return None
    return str(value).replace("param:", "")


@dataclass(frozen=True)
class ValueRef:
    """A planned value: a graph input or one port of a node's output."""

    kind: str  # "input" | "node"
    name: str
    port: int = 0

    @property
    def label(self) -> str:
        return f"input:{self.name}" if self.kind == "input" else f"{self.name}:{self.port}"

    def __str__(self) -> str:
        return self.label

    @classmethod
    def of(cls, ref: Any) -> "ValueRef":
        if ref.node.startswith(GRAPH_INPUT):
            return cls("input", ref.node[len(GRAPH_INPUT):], 0)
        return cls("node", ref.node, int(ref.port))


@dataclass(frozen=True)
class PlanStep:
    """One compiled step: read ``inputs``, write ``targets``."""

    index: int
    op: str
    inputs: tuple[ValueRef, ...]
    targets: tuple[ValueRef, ...]
    attrs: Mapping[str, Any] = field(default_factory=dict)
    nodes: tuple[str, ...] = ()
    kind: str = "native"
    node_id: str | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "op": self.op,
            "kind": self.kind,
            "inputs": [str(ref) for ref in self.inputs],
            "targets": [str(ref) for ref in self.targets],
            "nodes": list(self.nodes),
            "attrs": dict(self.attrs),
        }


@dataclass(frozen=True)
class Buffer:
    key: str
    slot: int
    shape: tuple[int, ...]
    elements: int
    reused: bool = False


@dataclass
class MemoryPlan:
    """Result of the liveness allocation pass."""

    buffers: dict[str, Buffer]
    slot_elements: list[int]
    peak_elements: int

    @property
    def peak_bytes(self) -> int:
        return self.peak_elements * 4

    @property
    def allocated_bytes(self) -> int:
        return sum(self.slot_elements) * 4

    @property
    def reuse_count(self) -> int:
        return sum(1 for buffer in self.buffers.values() if buffer.reused)

    def describe(self) -> dict[str, Any]:
        return {
            "buffers": len(self.buffers),
            "slots": len(self.slot_elements),
            "reused_buffers": self.reuse_count,
            "peak_elements": self.peak_elements,
            "peak_bytes": self.peak_bytes,
            "allocated_bytes": self.allocated_bytes,
            "slot_elements": list(self.slot_elements),
        }


@dataclass
class BoundPlan:
    """A plan bound to concrete input shapes: buffers exist, shapes resolved."""

    signature: tuple[Any, ...]
    symbols: dict[str, int]
    shapes: dict[str, tuple[int, ...]]
    memory: MemoryPlan
    slots: list[np.ndarray]

    def value(self, ref: ValueRef) -> np.ndarray:
        try:
            buffer = self.memory.buffers[ref.label]
        except KeyError as exc:
            raise InvalidOperationParameters(
                f"plan value '{ref}' has no allocated buffer (it was fused away or "
                "never written)"
            ) from exc
        return self.slots[buffer.slot][: buffer.elements].reshape(buffer.shape)

    def describe(self) -> dict[str, Any]:
        return {
            "signature": list(self.signature[0]) if self.signature else [],
            "symbols": dict(self.symbols),
            "memory": self.memory.describe(),
        }


@dataclass
class NativePlan:
    """A lowered, shape-polymorphic native execution plan."""

    graph: Graph
    steps: list[PlanStep]
    inputs: dict[str, ValueRef]
    outputs: dict[str, ValueRef]
    fusions: list[dict[str, Any]]
    prepack_names: tuple[str, ...]
    symbolic_shapes: dict[str, tuple[Any, ...]]
    _bound: dict[tuple[Any, ...], BoundPlan] = field(default_factory=dict, repr=False)
    _packed: dict[str, Any] = field(default_factory=dict, repr=False)
    _packed_key: str | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ #
    # inspection
    # ------------------------------------------------------------------ #

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    def operators(self) -> list[str]:
        return sorted({step.op for step in self.steps})

    @property
    def packed_weights(self) -> dict[str, Any]:
        """The prepacked weights prepared for the most recent package."""
        return dict(self._packed)

    @property
    def bound_variants(self) -> int:
        """Number of input-shape variants bound so far."""
        return len(self._bound)

    def fused_steps(self) -> int:
        return sum(1 for step in self.steps if len(step.nodes) > 1)

    def describe(self) -> dict[str, Any]:
        histogram: dict[str, int] = {}
        for step in self.steps:
            histogram[step.op] = histogram.get(step.op, 0) + 1
        return {
            "graph": self.graph.name,
            "steps": self.num_steps,
            "operators": histogram,
            "fused_steps": self.fused_steps(),
            "fusions": list(self.fusions),
            "prepacked_weights": list(self.prepack_names),
            "bound_variants": len(self._bound),
            "float32_only": True,
        }

    # ------------------------------------------------------------------ #
    # binding and weights
    # ------------------------------------------------------------------ #

    def bind(
        self, inputs: Mapping[str, np.ndarray], package: WeightPackage
    ) -> BoundPlan:
        """Bind concrete input shapes, resolving symbols and allocating buffers."""
        signature = tuple(
            sorted((name, tuple(int(d) for d in np.shape(value))) for name, value in inputs.items())
        )
        key = (signature, package.fingerprint)
        cached = self._bound.get(key)
        if cached is not None:
            return cached
        symbols = bind_symbols(self.graph, inputs)
        shapes = resolve_shapes(self.graph, symbols, package)
        memory = plan_memory(self.steps, shapes, self.outputs)
        slots = [np.empty(size, dtype=np.float32) for size in memory.slot_elements]
        bound = BoundPlan(
            signature=key,
            symbols=dict(symbols),
            shapes=shapes,
            memory=memory,
            slots=slots,
        )
        # Keep a small cache of shape variants; a plan is reused across batches.
        if len(self._bound) >= 4:
            self._bound.pop(next(iter(self._bound)))
        self._bound[key] = bound
        return bound

    def prepare_weights(self, package: WeightPackage, library: Any) -> dict[str, Any]:
        """Prepack linear weights once per package fingerprint."""
        key = package.fingerprint
        if self._packed_key == key and self._packed:
            return self._packed
        packed = {
            name: library.pack_weight(package.get(name)) for name in self.prepack_names
        }
        self._packed = packed
        self._packed_key = key
        return packed

    def release_weights(self) -> None:
        self._packed = {}
        self._packed_key = None

    def release_buffers(self) -> None:
        self._bound.clear()


# --------------------------------------------------------------------------- #
# shape resolution
# --------------------------------------------------------------------------- #


def bind_symbols(
    graph: Graph, inputs: Mapping[str, np.ndarray]
) -> dict[str, int]:
    """Validate graph inputs against their declaration and bind symbols."""
    missing = [name for name in graph.inputs if name not in inputs]
    if missing:
        raise ShapeMismatch(f"missing graph inputs: {missing}")
    extra = [name for name in inputs if name not in graph.inputs]
    if extra:
        raise ShapeMismatch(f"unexpected graph inputs: {extra}")
    symbols: dict[str, int] = {}
    for name, spec in graph.inputs.items():
        array = np.asarray(inputs[name])
        if array.ndim != len(spec.shape):
            raise ShapeMismatch(
                f"input '{name}' has rank {array.ndim}, declared rank {len(spec.shape)}"
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
                        f"symbol '{dimension}' bound inconsistently: {previous} vs {actual}"
                    )
                symbols[dimension] = int(actual)
            else:
                raise ShapeMismatch(
                    f"input '{name}' declares an unknown dimension; use concrete dims "
                    "or symbols"
                )
    return symbols


def _concrete(dimension: Any, symbols: Mapping[str, int], where: str) -> int | None:
    """Resolve a dimension, returning None when the operator left it unknown."""
    if isinstance(dimension, int) and not isinstance(dimension, bool):
        return int(dimension)
    if isinstance(dimension, str):
        if dimension not in symbols:
            raise InvalidOperationParameters(
                f"dimension symbol '{dimension}' at '{where}' is not bound by the "
                "graph inputs"
            )
        return int(symbols[dimension])
    return None


def resolve_shapes(
    graph: Graph, symbols: Mapping[str, int], package: WeightPackage
) -> dict[str, tuple[int | None, ...]]:
    """Concrete shape for every input and node output port.

    Symbolic inference can leave a dimension unknown (a symbolic ``slice`` writes
    ``None``). A second pass re-infers those nodes from their now-concrete input
    shapes, so a plan can allocate a buffer for them.
    """
    bound_types = {
        name: TensorType.of(meta.shape, meta.dtype) for name, meta in package.metas.items()
    }
    types, errors = graph.infer_types(bound=bound_types)
    if errors:
        raise GraphValidationError(
            "native plan cannot resolve shapes", errors=errors[:10]
        )
    shapes: dict[str, tuple[int | None, ...]] = {}
    for name, spec in graph.inputs.items():
        shapes[f"input:{name}"] = tuple(
            _concrete(dimension, symbols, f"input:{name}") for dimension in spec.shape
        )
    for node_id, outputs in types.items():
        for port, tensor in enumerate(outputs):
            shapes[f"{node_id}:{port}"] = tuple(
                _concrete(dimension, symbols, f"{node_id}:{port}")
                for dimension in tensor.shape
            )

    order = graph.topological_order()
    for node_id in order:
        node = graph.nodes[node_id]
        ports = list(range(len(types.get(node_id, []))))
        if not ports or not any(
            any(dimension is None for dimension in shapes[f"{node_id}:{port}"])
            for port in ports
        ):
            continue
        operator = get_operator(node.op)
        refs = [
            ValueRef.of(node.inputs[port])
            for port in port_order(operator, node.inputs)
        ]
        in_types: list[TensorType] = []
        for ref in refs:
            shape = shapes.get(ref.label)
            if shape is None or any(dimension is None for dimension in shape):
                in_types = []
                break
            in_types.append(TensorType.of(shape, "float32"))
        if not in_types:
            continue
        try:
            outs = operator.infer_types(in_types, node.params, bound_types)
        except Exception:  # an operator that still cannot infer is reported later
            continue
        for port, tensor in enumerate(outs):
            if port >= len(ports):
                continue
            candidate = tuple(
                _concrete(dimension, symbols, f"{node_id}:{port}")
                for dimension in tensor.shape
            )
            if all(isinstance(dimension, int) for dimension in candidate):
                shapes[f"{node_id}:{port}"] = candidate
    return shapes


def _elements(shape: Sequence[int]) -> int:
    total = 1
    for dimension in shape:
        total *= int(dimension)
    return int(total)


def plan_memory(
    steps: Sequence[PlanStep],
    shapes: Mapping[str, tuple[int, ...]],
    outputs: Mapping[str, ValueRef],
) -> MemoryPlan:
    """Allocate buffers by liveness, reusing slots of identical size.

    A target is allocated before the step's dying inputs are released, so a kernel
    never reads from the buffer it is writing to.
    """
    last_use: dict[str, int] = {}
    for index, step in enumerate(steps):
        for ref in step.inputs:
            if ref.kind == "node":
                last_use[ref.label] = max(last_use.get(ref.label, -1), index)
    protected = {ref.label for ref in outputs.values() if ref.kind == "node"}

    free: dict[int, list[int]] = {}
    buffers: dict[str, Buffer] = {}
    slot_elements: list[int] = []
    live_slots: set[int] = set()
    peak = 0

    for index, step in enumerate(steps):
        for target in step.targets:
            shape = shapes.get(target.label)
            if shape is None:
                raise ShapeMismatch(
                    f"plan step {index} ({step.op}) targets '{target}', which has no "
                    "resolved shape"
                )
            elements = _elements(shape)
            pool = free.get(elements)
            reused = bool(pool)
            if pool:
                slot = pool.pop()
            else:
                slot = len(slot_elements)
                slot_elements.append(elements)
            buffers[target.label] = Buffer(target.label, slot, shape, elements, reused)
            live_slots.add(slot)
            peak = max(peak, sum(slot_elements[value] for value in live_slots))
        for ref in step.inputs:
            if ref.kind != "node" or last_use.get(ref.label) != index:
                continue
            if ref.label in protected:
                continue
            buffer = buffers.get(ref.label)
            if buffer is None or buffer.slot not in live_slots:
                continue
            live_slots.discard(buffer.slot)
            free.setdefault(buffer.elements, []).append(buffer.slot)

    return MemoryPlan(buffers, slot_elements, peak)


# --------------------------------------------------------------------------- #
# lowering
# --------------------------------------------------------------------------- #


def _symbolic_shapes(
    graph: Graph, package: WeightPackage | None
) -> dict[str, tuple[Any, ...]]:
    if package is not None:
        bound_types = {
            name: TensorType.of(meta.shape, meta.dtype)
            for name, meta in package.metas.items()
        }
        types, errors = graph.infer_types(bound=bound_types)
        if errors:
            raise GraphValidationError(
                "native plan compilation failed shape inference", errors=errors[:10]
            )
    else:
        missing = [nid for nid, node in graph.nodes.items() if not node.outputs]
        if missing:
            raise GraphValidationError(
                "compile_native_plan needs inferred node output types; validate the "
                "graph against a package first (or pass package=...)",
                errors=[f"node '{nid}' has no inferred outputs" for nid in missing[:10]],
            )
        types = {nid: list(node.outputs) for nid, node in graph.nodes.items()}
    shapes: dict[str, tuple[Any, ...]] = {
        f"input:{name}": tuple(spec.shape) for name, spec in graph.inputs.items()
    }
    for node_id, outputs in types.items():
        for port, tensor in enumerate(outputs):
            shapes[f"{node_id}:{port}"] = tuple(tensor.shape)
    return shapes


def _activation_for(
    graph: Graph,
    node_id: str,
    consumers: Mapping[str, list[str]],
    output_nodes: set[str],
    shapes: Mapping[str, tuple[Any, ...]],
) -> str | None:
    if graph.nodes[node_id].op != "linear" or node_id in output_nodes:
        return None
    users = consumers.get(node_id, [])
    if len(users) != 1:
        return None
    child = graph.nodes[users[0]]
    if child.op not in FUSABLE_ACTIVATIONS or child.id in output_nodes:
        return None
    if shapes.get(f"{node_id}:0") != shapes.get(f"{child.id}:0"):
        return None
    return child.id


def _fusable_norm(
    graph: Graph,
    node_id: str,
    consumers: Mapping[str, list[str]],
    shapes: Mapping[str, tuple[Any, ...]],
) -> str | None:
    node = graph.nodes[node_id]
    if node.op not in NORM_OPS or len(node.inputs) != 1:
        return None
    ref = next(iter(node.inputs.values()))
    if ref.node.startswith(GRAPH_INPUT):
        return None
    parent = graph.nodes.get(ref.node)
    if parent is None or parent.op != "add" or len(parent.inputs) != 2:
        return None
    # A residual add may feed the normalization *and* other consumers; the fused
    # step materializes the sum, so later readers still see it.
    if node_id not in consumers.get(parent.id, []):
        return None
    result_shape = shapes.get(f"{parent.id}:0")
    for port in port_order(get_operator("add"), parent.inputs):
        source = ValueRef.of(parent.inputs[port]).label
        if shapes.get(source) != result_shape:
            return None
    return parent.id


def _inputs_of(node: Any) -> list[ValueRef]:
    operator = get_operator(node.op)
    return [
        ValueRef.of(node.inputs[port]) for port in port_order(operator, node.inputs)
    ]


def compile_native_plan(
    graph: Graph,
    package: WeightPackage | None = None,
    *,
    fuse: bool = True,
) -> NativePlan:
    """Lower a NIR graph into a native execution plan.

    Raises :class:`UnsupportedOperator` for any operator the plan cannot run and
    :class:`GraphValidationError` when node shapes are not inferable.
    """
    order = graph.topological_order()
    shapes = _symbolic_shapes(graph, package)

    consumers: dict[str, list[str]] = {node_id: [] for node_id in graph.nodes}
    for node_id, node in graph.nodes.items():
        for ref in node.inputs.values():
            if ref.node in consumers:
                consumers[ref.node].append(node_id)
    output_nodes = {
        ref.node for ref in graph.outputs.values() if not ref.node.startswith(GRAPH_INPUT)
    }

    fused_activation: dict[str, str] = {}
    fused_norm: dict[str, str] = {}
    skipped: set[str] = set()
    if fuse:
        for node_id in graph.nodes:
            child = _activation_for(graph, node_id, consumers, output_nodes, shapes)
            if child is not None:
                fused_activation[node_id] = child
                skipped.add(child)
        for node_id in graph.nodes:
            parent = _fusable_norm(graph, node_id, consumers, shapes)
            if parent is not None:
                fused_norm[node_id] = parent
                skipped.add(parent)

    steps: list[PlanStep] = []
    fusions: list[dict[str, Any]] = []
    prepack: set[str] = set()

    def emit(
        op: str,
        inputs: list[ValueRef],
        targets: list[ValueRef],
        attrs: dict[str, Any] | None = None,
        *,
        nodes: tuple[str, ...],
        kind: str = "native",
        node_id: str | None = None,
    ) -> None:
        steps.append(
            PlanStep(
                index=len(steps),
                op=op,
                inputs=tuple(inputs),
                targets=tuple(targets),
                attrs=dict(attrs or {}),
                nodes=nodes,
                kind=kind,
                node_id=node_id,
            )
        )

    for node_id in order:
        if node_id in skipped:
            continue
        node = graph.nodes[node_id]
        op = node.op
        inputs = _inputs_of(node)
        target = ValueRef("node", node_id, 0)

        if node_id in fused_activation:
            activation_node = fused_activation[node_id]
            weight = param_name(node.params.get("weight", "weight"))
            if weight:
                prepack.add(weight)
            emit(
                "gemm_packed",
                inputs,
                [ValueRef("node", activation_node, 0)],
                {
                    "weight": weight,
                    "bias": param_name(node.params.get("bias")),
                    "activation": graph.nodes[activation_node].op,
                },
                nodes=(node_id, activation_node),
            )
            fusions.append(
                {
                    "kind": "linear_activation",
                    "nodes": [node_id, activation_node],
                    "step": len(steps) - 1,
                }
            )
            continue

        if op == "linear":
            weight = param_name(node.params.get("weight", "weight"))
            if weight:
                prepack.add(weight)
            emit(
                "gemm_packed",
                inputs,
                [target],
                {
                    "weight": weight,
                    "bias": param_name(node.params.get("bias")),
                    "activation": "none",
                },
                nodes=(node_id,),
            )
            continue

        if node_id in fused_norm:
            add_id = fused_norm[node_id]
            parent = graph.nodes[add_id]
            add_inputs = _inputs_of(parent)
            # Store the residual when anything besides this normalization needs it.
            needs_residual = (
                add_id in output_nodes or len(consumers.get(add_id, [])) > 1
            )
            targets = [ValueRef("node", node_id, 0)]
            if needs_residual:
                targets.insert(0, ValueRef("node", add_id, 0))
            emit(
                "add_norm",
                add_inputs,
                targets,
                {
                    "kind": "layernorm" if op == "layernorm" else "rmsnorm",
                    "weight": param_name(node.params.get("weight")),
                    "bias": param_name(node.params.get("bias")),
                    "eps": float(node.params.get("eps", 1e-5 if op == "layernorm" else 1e-6)),
                    "store_residual": needs_residual,
                },
                nodes=(add_id, node_id),
            )
            fusions.append(
                {
                    "kind": "residual_norm",
                    "nodes": [add_id, node_id],
                    "step": len(steps) - 1,
                }
            )
            continue

        if op in FUSABLE_ACTIVATIONS:
            emit("unary", inputs, [target], {"name": op}, nodes=(node_id,))
        elif op in BINARY_OPS:
            emit("binary", inputs, [target], {"name": op}, nodes=(node_id,))
        elif op == "scale":
            emit(
                "scale",
                inputs,
                [target],
                {"factor": float(node.params.get("factor", 1.0))},
                nodes=(node_id,),
            )
        elif op == "clip":
            emit(
                "clip",
                inputs,
                [target],
                {
                    "min": float(node.params.get("min", -1.0)),
                    "max": float(node.params.get("max", 1.0)),
                },
                nodes=(node_id,),
            )
        elif op in NORM_OPS:
            emit(
                op,
                inputs,
                [target],
                {
                    "weight": param_name(node.params.get("weight")),
                    "bias": param_name(node.params.get("bias")),
                    "eps": float(node.params.get("eps", 1e-5 if op == "layernorm" else 1e-6)),
                },
                nodes=(node_id,),
            )
        elif op == "softmax":
            _require_last_axis(node, shapes, "softmax")
            emit("softmax", inputs, [target], {}, nodes=(node_id,))
        elif op == "matmul":
            for ref in inputs:
                rank = len(shapes.get(ref.label, ()))
                if rank != 2:
                    raise UnsupportedOperator(
                        f"the native plan implements rank-2 matmul only; node "
                        f"'{node_id}' has an operand of rank {rank}",
                        node=node_id,
                        operator=op,
                    )
            emit("matmul", inputs, [target], {}, nodes=(node_id,))
        elif op == "embedding":
            emit(
                "embedding",
                inputs,
                [target],
                {"weight": param_name(node.params.get("weight", "weight"))},
                nodes=(node_id,),
            )
        elif op in REDUCE_OPS:
            emit(
                "reduce",
                inputs,
                [target],
                {
                    "mode": op,
                    "axis": _single_axis(node, shapes, op),
                    "keepdims": bool(node.params.get("keepdims", False)),
                },
                nodes=(node_id,),
            )
        elif op == "attention":
            slopes = node.params.get("alibi_slopes")
            emit(
                "attention",
                inputs,
                [target],
                {
                    "scale": node.params.get("scale", "auto"),
                    "causal": bool(node.params.get("causal", True)),
                    "alibi_slopes": list(slopes) if slopes else None,
                    "window": node.params.get("window"),
                },
                nodes=(node_id,),
            )
        elif op == "conv2d":
            emit(
                "conv2d",
                inputs,
                [target],
                {
                    "weight": param_name(node.params.get("weight", "weight")),
                    "bias": param_name(node.params.get("bias")),
                    "stride": node.params.get("stride", 1),
                    "padding": node.params.get("padding", 0),
                },
                nodes=(node_id,),
            )
        elif op == "maxpool2d":
            emit(
                "maxpool2d",
                inputs,
                [target],
                {
                    "kernel": node.params.get("kernel", 2),
                    "stride": node.params.get("stride", node.params.get("kernel", 2)),
                    "padding": node.params.get("padding", 0),
                },
                nodes=(node_id,),
            )
        elif op in RECURRENT_OPS:
            emit(
                "recurrent",
                inputs,
                [target],
                {
                    "kind": op,
                    "weight_ih": param_name(node.params.get("weight_ih", "weight_ih")),
                    "weight_hh": param_name(node.params.get("weight_hh", "weight_hh")),
                    "bias": param_name(node.params.get("bias")),
                },
                nodes=(node_id,),
            )
        elif op == "ssm":
            emit(
                "ssm",
                inputs,
                [target],
                {
                    "A": param_name(node.params.get("A", "A")),
                    "B": param_name(node.params.get("B", "B")),
                    "C": param_name(node.params.get("C", "C")),
                    "D": param_name(node.params.get("D")),
                },
                nodes=(node_id,),
            )
        elif op == "gcn":
            emit(
                "gcn",
                inputs,
                [target],
                {
                    "weight": param_name(node.params.get("weight", "weight")),
                    "bias": param_name(node.params.get("bias")),
                },
                nodes=(node_id,),
            )
        elif op == "moe":
            emit(
                "moe",
                inputs,
                [target],
                {
                    "router": param_name(node.params.get("router", "router")),
                    "up_weight": param_name(node.params.get("up_weight", "up_weight")),
                    "down_weight": param_name(node.params.get("down_weight", "down_weight")),
                    "up_bias": param_name(node.params.get("up_bias")),
                    "down_bias": param_name(node.params.get("down_bias")),
                    "top_k": int(node.params.get("top_k", 2)),
                    "activation": str(node.params.get("activation", "gelu")),
                },
                nodes=(node_id,),
            )
        elif op == "quantized_linear":
            emit(
                "quantized_linear",
                inputs,
                [target],
                {
                    "weight": param_name(node.params.get("weight", "weight")),
                    "scale": param_name(node.params.get("scale", "scale")),
                    "bias": param_name(node.params.get("bias")),
                },
                nodes=(node_id,),
            )
        elif op in LAYOUT_OPS:
            targets = [ValueRef("node", node_id, port) for port in range(len(node.outputs))]
            emit(
                "layout",
                inputs,
                targets,
                {"n_outputs": len(node.outputs)},
                nodes=(node_id,),
                kind="layout",
                node_id=node_id,
            )
        else:
            raise UnsupportedOperator(
                f"the native plan compiler does not implement NIR operator '{op}' "
                f"(node '{node_id}'); supported operators: "
                f"{', '.join(sorted(PLAN_OPS))} covering "
                f"{', '.join(sorted(FUSABLE_ACTIVATIONS | BINARY_OPS | REDUCE_OPS | RECURRENT_OPS | LAYOUT_OPS | NORM_OPS))}",
                node=node_id,
                operator=op,
            )

    plan = NativePlan(
        graph=graph,
        steps=steps,
        inputs={name: ValueRef("input", name) for name in graph.inputs},
        outputs={name: ValueRef.of(ref) for name, ref in graph.outputs.items()},
        fusions=fusions,
        prepack_names=tuple(sorted(prepack)),
        symbolic_shapes=shapes,
    )
    # A plan whose steps produce no buffer at all is still valid (a graph that
    # only forwards an input), but a plan must materialize every declared output.
    materialized = {target.label for step in steps for target in step.targets}
    for name, ref in plan.outputs.items():
        if ref.kind == "node" and ref.label not in materialized:
            raise InvalidOperationParameters(
                f"graph output '{name}' references '{ref}', which the plan does not "
                "materialize"
            )
    return plan


def _single_axis(node: Any, shapes: Mapping[str, tuple[Any, ...]], op: str) -> int:
    inputs = _inputs_of(node)
    rank = len(shapes.get(inputs[0].label, ()))
    axis = node.params.get("axis", -1)
    if isinstance(axis, (list, tuple)):
        if len(axis) != 1:
            raise UnsupportedOperator(
                f"the native plan reduces one axis at a time; node '{node.id}' asks "
                f"for axes {list(axis)}",
                node=node.id,
                operator=op,
            )
        axis = axis[0]
    if not isinstance(axis, int) or isinstance(axis, bool) or not rank:
        raise UnsupportedOperator(
            f"the native plan needs a concrete integer axis for {op}; node "
            f"'{node.id}' has {node.params.get('axis')!r}",
            node=node.id,
            operator=op,
        )
    if axis < -rank or axis >= rank:
        raise UnsupportedOperator(
            f"axis {axis} out of range for rank {rank} in node '{node.id}'",
            node=node.id,
            operator=op,
        )
    return axis


def _require_last_axis(node: Any, shapes: Mapping[str, tuple[Any, ...]], op: str) -> None:
    inputs = _inputs_of(node)
    rank = len(shapes.get(inputs[0].label, ()))
    axis = node.params.get("axis", -1)
    normalized = axis % rank if isinstance(axis, int) and rank else axis
    if normalized != rank - 1:
        raise UnsupportedOperator(
            f"the native plan implements {op} on the last axis only; node "
            f"'{node.id}' uses axis {axis}",
            node=node.id,
            operator=op,
        )
