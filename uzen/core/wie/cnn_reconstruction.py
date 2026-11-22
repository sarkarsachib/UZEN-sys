"""Reconstruct an explicit sequential CNN checkpoint into an executable NIR graph.

This is the first non-causal model ingestion adapter. It does not guess an
architecture from tensor names: the caller supplies the layer specification and an
optional canonical-to-source tensor map. The adapter builds exactly those layers,
maps every parameter, validates the graph, and executes a real forward pass.

Supported layer kinds are ``conv2d``, ``batchnorm``, ``relu``, ``gelu``, ``silu``,
``tanh``, ``maxpool2d``, ``flatten``, ``dropout`` (inference identity) and
``linear``. A convolutional layer may carry an inline ``pool`` descriptor. This is
deliberately an explicit adapter, not a general model parser.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from uzen.core.nir.executor import execute
from uzen.core.nir.graph import Graph
from uzen.core.nir.types import Ref, TensorType
from uzen.core.nws.families import ArchitectureDefinition
from uzen.core.nws.tensors import WeightPackage
from uzen.core.wie.reconstruction import ReconstructionReport, ReconstructionResult
from uzen.errors import InvalidOperationParameters

_ACTIVATIONS = {"relu", "gelu", "gelu_exact", "silu", "tanh"}


def _pair(value: Any, name: str) -> tuple[int, int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return (int(value), int(value))
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(
        isinstance(v, int) and not isinstance(v, bool) for v in value
    ):
        return (int(value[0]), int(value[1]))
    raise InvalidOperationParameters(f"CNN {name} must be an int or [height, width]")


def _shape_after_conv(
    height: int, width: int, kernel: tuple[int, int], stride: tuple[int, int], padding: tuple[int, int]
) -> tuple[int, int]:
    out_h = (height + 2 * padding[0] - kernel[0]) // stride[0] + 1
    out_w = (width + 2 * padding[1] - kernel[1]) // stride[1] + 1
    if out_h <= 0 or out_w <= 0:
        raise InvalidOperationParameters("CNN convolution produces a non-positive feature map")
    return out_h, out_w


def _validate_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    channels = int(spec.get("in_channels", 1))
    height = int(spec.get("height", 28))
    width = int(spec.get("width", 28))
    if min(channels, height, width) <= 0:
        raise InvalidOperationParameters("CNN input channels, height and width must be positive")
    layers = spec.get("layers")
    if not isinstance(layers, list) or not layers:
        raise InvalidOperationParameters("CNN spec requires a non-empty layers list")
    return {
        "name": str(spec.get("name") or "cnn"),
        "in_channels": channels,
        "height": height,
        "width": width,
        "layers": [dict(layer) for layer in layers],
    }


def reconstruct_cnn_model(
    arrays: Mapping[str, np.ndarray],
    spec: Mapping[str, Any],
    *,
    source_names: Mapping[str, str] | None = None,
    source_dtypes: Mapping[str, str] | None = None,
    name: str | None = None,
    source_hash: str | None = None,
    on_progress: Any | None = None,
) -> ReconstructionResult:
    """Build, validate and execute an explicit sequential CNN graph."""
    progress: list[dict[str, Any]] = []
    warnings: list[str] = []
    reasons: list[str] = []
    missing: list[str] = []
    extra: list[str] = []
    shape_errors: list[str] = []

    def step(stage: str, status: str, message: str) -> None:
        progress.append({"stage": stage, "status": status, "message": message})
        if on_progress is not None:
            on_progress(stage, status, message)

    try:
        clean = _validate_spec(spec)
    except (TypeError, ValueError, InvalidOperationParameters) as exc:
        reason = f"invalid CNN specification: {exc}"
        step("config", "unsupported", reason)
        return ReconstructionResult(
            report=ReconstructionReport(
                status="unsupported",
                architecture="cnn",
                family="cnn",
                spec={},
                graph=None,
                graph_summary=None,
                tensor_mapping=[],
                missing=[],
                extra=sorted(arrays),
                shape_errors=[],
                unresolved=[reason],
                warnings=warnings,
                validation=None,
                execution=None,
                source_tensor_count=len(arrays),
                mapped_tensor_count=0,
                progress=progress,
                reasons=[reason],
            )
        )

    channels = clean["in_channels"]
    height = clean["height"]
    width = clean["width"]
    graph = Graph(name=clean["name"], metadata={"family": "cnn", "spec": clean})
    graph.add_input("image", TensorType.of(("B", channels, height, width)))
    parameter_shapes: dict[str, tuple[int, ...]] = {}
    canonical: dict[str, np.ndarray] = {}
    mapping: list[dict[str, Any]] = []
    used: set[str] = set()
    current = Ref("input:image")
    num_classes = 0
    layer_index = 0

    def add_parameter(
        target: str,
        array: np.ndarray | None,
        source: str | None,
        *,
        note: str = "",
    ) -> None:
        expected = parameter_shapes[target]
        if array is None:
            missing.append(target)
            mapping.append(
                {
                    "target": target,
                    "source": source,
                    "transform": "identity",
                    "status": "missing",
                    "expected_shape": list(expected),
                    "note": note,
                }
            )
            return
        shape = tuple(int(v) for v in array.shape)
        if shape != expected:
            shape_errors.append(
                f"{target}: source {source} shape {list(shape)} != expected {list(expected)}"
            )
            mapping.append(
                {
                    "target": target,
                    "source": source,
                    "transform": "identity",
                    "status": "shape_error",
                    "source_shape": list(shape),
                    "expected_shape": list(expected),
                    "note": note,
                }
            )
            return
        canonical[target] = np.ascontiguousarray(array)
        used.add(str(source))
        mapping.append(
            {
                "target": target,
                "source": source,
                "transform": "identity",
                "status": "mapped",
                "source_shape": list(shape),
                "expected_shape": list(expected),
                "note": note,
            }
        )

    def source_for(target: str) -> str:
        return str((source_names or {}).get(target, target))

    def source_array(target: str) -> np.ndarray | None:
        source = source_for(target)
        return arrays.get(source)

    def add_node(kind: str, params: dict[str, Any] | None = None) -> Ref:
        nonlocal current
        name = f"{kind}_{layer_index}"
        current = graph.add_node(name, kind, {"x": current}, params or {})
        return current

    for raw in clean["layers"]:
        if not isinstance(raw, dict):
            raise InvalidOperationParameters(f"CNN layer #{layer_index} must be an object")
        kind = str(raw.get("kind") or "").lower()
        layer_name = str(raw.get("name") or f"{kind}_{layer_index}")
        if kind == "conv2d":
            out_channels = int(raw.get("out_channels") or 0)
            kernel = _pair(raw.get("kernel", 3), "kernel")
            stride = _pair(raw.get("stride", 1), "stride")
            padding = _pair(raw.get("padding", 0), "padding")
            if out_channels <= 0:
                raise InvalidOperationParameters(f"CNN layer '{layer_name}' needs out_channels > 0")
            w_name = f"{layer_name}.weight"
            b_name = f"{layer_name}.bias"
            parameter_shapes[w_name] = (out_channels, channels, kernel[0], kernel[1])
            parameter_shapes[b_name] = (out_channels,)
            add_parameter(w_name, source_array(w_name), source_for(w_name))
            add_parameter(b_name, source_array(b_name), source_for(b_name))
            current = graph.add_node(
                f"{layer_name}_conv",
                "conv2d",
                {"x": current},
                {
                    "weight": f"param:{w_name}",
                    "bias": f"param:{b_name}",
                    "kernel": list(kernel),
                    "stride": list(stride),
                    "padding": list(padding),
                },
            )
            channels = out_channels
            height, width = _shape_after_conv(height, width, kernel, stride, padding)
            layer_index += 1
            if raw.get("batch_norm"):
                bn = f"{layer_name}.bn"
                for suffix, shape in (
                    ("weight", (channels,)),
                    ("bias", (channels,)),
                    ("running_mean", (channels,)),
                    ("running_var", (channels,)),
                ):
                    target = f"{bn}.{suffix}"
                    parameter_shapes[target] = shape
                    add_parameter(target, source_array(target), source_for(target))
                current = graph.add_node(
                    f"{layer_name}_bn",
                    "batchnorm",
                    {"x": current},
                    {
                        "weight": f"param:{bn}.weight",
                        "bias": f"param:{bn}.bias",
                        "running_mean": f"param:{bn}.running_mean",
                        "running_var": f"param:{bn}.running_var",
                        "eps": float(raw.get("eps", 1e-5)),
                        "axis": 1,
                    },
                )
            activation = str(raw.get("activation") or "")
            if activation:
                if activation not in _ACTIVATIONS:
                    raise InvalidOperationParameters(f"unsupported CNN activation '{activation}'")
                current = graph.add_node(
                    f"{layer_name}_{activation}", activation, {"x": current}
                )
            pool = raw.get("pool")
            if pool:
                if not isinstance(pool, dict):
                    raise InvalidOperationParameters(f"CNN layer '{layer_name}' pool must be an object")
                kernel = _pair(pool.get("kernel", 2), "kernel")
                stride = _pair(pool.get("stride", kernel), "stride")
                padding = _pair(pool.get("padding", 0), "padding")
                current = graph.add_node(
                    f"{layer_name}_pool",
                    "maxpool2d",
                    {"x": current},
                    {"kernel": list(kernel), "stride": list(stride), "padding": list(padding)},
                )
                height, width = _shape_after_conv(height, width, kernel, stride, padding)
                layer_index += 1
        elif kind == "batchnorm":
            bn = layer_name
            for suffix, shape in (
                ("weight", (channels,)),
                ("bias", (channels,)),
                ("running_mean", (channels,)),
                ("running_var", (channels,)),
            ):
                target = f"{bn}.{suffix}"
                parameter_shapes[target] = shape
                add_parameter(target, source_array(target), source_for(target))
            current = graph.add_node(
                f"{bn}_bn",
                "batchnorm",
                {"x": current},
                {
                    "weight": f"param:{bn}.weight",
                    "bias": f"param:{bn}.bias",
                    "running_mean": f"param:{bn}.running_mean",
                    "running_var": f"param:{bn}.running_var",
                    "eps": float(raw.get("eps", 1e-5)),
                    "axis": 1,
                },
            )
            layer_index += 1
        elif kind in _ACTIVATIONS:
            current = add_node(kind)
            layer_index += 1
        elif kind == "maxpool2d":
            kernel = _pair(raw.get("kernel", 2), "kernel")
            stride = _pair(raw.get("stride", kernel), "stride")
            padding = _pair(raw.get("padding", 0), "padding")
            current = graph.add_node(
                f"{layer_name}_pool",
                "maxpool2d",
                {"x": current},
                {"kernel": list(kernel), "stride": list(stride), "padding": list(padding)},
            )
            height, width = _shape_after_conv(height, width, kernel, stride, padding)
            layer_index += 1
        elif kind == "flatten":
            current = graph.add_node(f"{layer_name}_flatten", "flatten", {"x": current}, {"start_dim": 1})
            layer_index += 1
        elif kind == "dropout":
            current = graph.add_node(f"{layer_name}_dropout", "dropout", {"x": current}, {})
            layer_index += 1
        elif kind == "linear":
            out_features = int(raw.get("out_features") or 0)
            if out_features <= 0:
                raise InvalidOperationParameters(f"CNN layer '{layer_name}' needs out_features > 0")
            in_features = channels * height * width
            w_name = f"{layer_name}.weight"
            b_name = f"{layer_name}.bias"
            parameter_shapes[w_name] = (out_features, in_features)
            parameter_shapes[b_name] = (out_features,)
            add_parameter(w_name, source_array(w_name), source_for(w_name))
            add_parameter(b_name, source_array(b_name), source_for(b_name))
            current = graph.add_node(
                f"{layer_name}_linear",
                "linear",
                {"x": current},
                {"weight": f"param:{w_name}", "bias": f"param:{b_name}"},
            )
            channels = out_features
            height = 1
            width = 1
            num_classes = out_features
            activation = str(raw.get("activation") or "")
            if activation:
                if activation not in _ACTIVATIONS:
                    raise InvalidOperationParameters(f"unsupported CNN activation '{activation}'")
                current = graph.add_node(f"{layer_name}_{activation}", activation, {"x": current})
            layer_index += 1
        else:
            raise InvalidOperationParameters(f"unsupported CNN layer kind '{kind}'")

    if num_classes <= 0:
        raise InvalidOperationParameters("CNN spec must end in a linear classification layer")
    graph.set_output("logits", current)
    definition = ArchitectureDefinition(
        family="cnn",
        spec=clean,
        graph=graph,
        parameter_shapes=parameter_shapes,
        default_inputs={"image": ("B", clean["in_channels"], clean["height"], clean["width"])},
    )

    extra = sorted(set(arrays) - used)
    ignored_metadata = [source for source in extra if source.endswith(".num_batches_tracked")]
    if ignored_metadata:
        warnings.append(
            "inference metadata tensors are preserved as source metadata, not graph parameters: "
            + ", ".join(ignored_metadata)
        )
    extra = [source for source in extra if source not in ignored_metadata]
    for source in extra:
        reasons.append(f"source tensor '{source}' is not represented by the CNN graph")
    step(
        "tensor_mapping",
        "complete" if not missing and not shape_errors else "failed",
        f"mapped {len(canonical)}/{len(parameter_shapes)} tensors; {len(missing)} missing, "
        f"{len(shape_errors)} shape errors, {len(extra)} unmapped",
    )

    report_base = dict(
        architecture="cnn",
        family="cnn",
        spec=clean,
        tensor_mapping=mapping,
        missing=sorted(set(missing)),
        extra=extra,
        shape_errors=shape_errors,
        warnings=warnings,
        source_tensor_count=len(arrays),
        mapped_tensor_count=len(canonical),
        progress=progress,
    )
    if missing or shape_errors:
        reasons.append("all CNN parameters must be present with exact shapes; none are invented")
        return ReconstructionResult(
            report=ReconstructionReport(
                status="unsupported",
                graph=None,
                graph_summary=None,
                unresolved=reasons,
                validation=None,
                execution=None,
                reasons=reasons,
                **report_base,
            )
        )

    package = WeightPackage.from_arrays(
        canonical,
        name=name or clean["name"],
        family="cnn",
        family_spec=clean,
        metadata={
            "reconstruction": {
                "architecture": "cnn",
                "adapter": "explicit_sequential_cnn",
                "source_hash": source_hash,
                "tensor_mapping": mapping,
                "graph": graph.to_dict(),
            },
            "source_hash": source_hash,
        },
        source_dtypes={
            target: source_dtypes[item["source"]]
            if source_dtypes and item["source"] in source_dtypes
            else "float32"
            for target, item in ((m["target"], m) for m in mapping)
            if target is not None and item.get("status") == "mapped"
        },
    )
    validation = graph.validate(package)
    validation_doc = {
        "ok": validation.ok,
        "errors": list(validation.errors),
        "warnings": list(validation.warnings),
        "num_nodes": validation.num_nodes,
        "num_params": validation.num_params,
        "param_names": list(validation.param_names),
        "order": list(validation.order),
    }
    step(
        "graph_validation",
        "complete" if validation.ok else "failed",
        f"graph validation {'passed' if validation.ok else 'failed'} with {validation.num_nodes} nodes",
    )
    if not validation.ok:
        reasons.append("CNN graph validation failed; no artifact should be published")
        return ReconstructionResult(
            report=ReconstructionReport(
                status="failed",
                graph=graph.to_dict(),
                graph_summary=graph.summary(),
                unresolved=reasons,
                validation=validation_doc,
                execution=None,
                reasons=reasons,
                **report_base,
            ),
            package=None,
            definition=definition,
        )

    rng = np.random.default_rng(0)
    inputs = {
        "image": rng.standard_normal(
            (1, clean["in_channels"], clean["height"], clean["width"])
        ).astype(np.float32)
    }
    try:
        result = execute(graph, inputs, package, check_finite=True, validate_types=True)
        logits = result.output("logits")
        expected = (1, num_classes)
        if tuple(logits.shape) != expected:
            execution = {
                "status": "failed",
                "message": f"logits shape {tuple(logits.shape)} != expected {expected}",
            }
        else:
            execution = {
                "status": "passed",
                "message": "forward pass produced finite class logits with the expected shape",
                "outputs": {name: list(value.shape) for name, value in result.outputs.items()},
                "num_ops": result.num_ops,
            }
    except Exception as exc:
        execution = {"status": "failed", "message": str(exc)}
    step("forward_execution", execution["status"], execution.get("message", ""))

    status = "validated" if execution["status"] == "passed" and not extra else "partial"
    if execution["status"] != "passed":
        status = "failed"
        reasons.append("CNN forward execution failed; the graph is not marked validated")
    elif extra:
        reasons.append("CNN graph executes, but some source tensors remain unresolved")
    else:
        reasons.append("all CNN tensors mapped, graph validated, and forward execution produced finite logits")
    report = ReconstructionReport(
        status=status,
        graph=graph.to_dict(),
        graph_summary=graph.summary(),
        unresolved=reasons,
        validation=validation_doc,
        execution=execution,
        reasons=reasons,
        **report_base,
    )
    if status == "failed":
        return ReconstructionResult(report=report, package=None, definition=definition)
    return ReconstructionResult(report=report, package=package, definition=definition)
