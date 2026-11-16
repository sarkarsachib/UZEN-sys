"""Weight-only int8 execution for reconstructed NIR graphs.

The transform is intentionally narrow and auditable: only 2-D float weights
consumed by ``linear`` nodes are quantized.  Embeddings, normalization weights,
biases and any parameter shared with a non-linear node stay in their original
dtype.  Quantization is symmetric per output channel with int8 values in
``[-127, 127]`` and a scale stored alongside the integer tensor.  Activations
remain floating point; the ``quantized_linear`` kernel dequantizes weights as it
executes.

Every decision and every measured reconstruction error is returned in
:class:`QuantizationReport`; this is not an implicit or lossless conversion.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from uzen.core.nir.executor import ExecutionResult, execute
from uzen.core.nir.graph import Graph
from uzen.core.nws.tensors import WeightPackage
from uzen.errors import GraphValidationError, InvalidOperationParameters, TensorInvalid

QUANTIZATION_SCHEME = "int8_symmetric_per_output_channel"
QUANTIZATION_BITS = 8
QUANTIZATION_AXIS = 1
QUANTIZATION_ZERO_POINT = 0


@dataclass(frozen=True)
class QuantizedTensorReport:
    """One tensor's quantization record and measured reconstruction error."""

    name: str
    shape: tuple[int, ...]
    source_dtype: str
    quantized_dtype: str
    scheme: str
    axis: int
    zero_point: int
    scale_count: int
    scale_min: float
    scale_max: float
    scale_sha256: str
    original_bytes: int
    quantized_bytes: int
    bits_per_weight: float
    max_abs_error: float
    mean_abs_error: float
    max_relative_error: float
    saturated_values: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "source_dtype": self.source_dtype,
            "quantized_dtype": self.quantized_dtype,
            "scheme": self.scheme,
            "axis": self.axis,
            "zero_point": self.zero_point,
            "scale_count": self.scale_count,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
            "scale_sha256": self.scale_sha256,
            "original_bytes": self.original_bytes,
            "quantized_bytes": self.quantized_bytes,
            "bits_per_weight": self.bits_per_weight,
            "max_abs_error": self.max_abs_error,
            "mean_abs_error": self.mean_abs_error,
            "max_relative_error": self.max_relative_error,
            "saturated_values": self.saturated_values,
        }


@dataclass(frozen=True)
class QuantizationReport:
    """Explicit record of what was quantized, what was excluded and the error."""

    graph_name: str
    source_package: str
    scheme: str = QUANTIZATION_SCHEME
    bits: int = QUANTIZATION_BITS
    axis: int = QUANTIZATION_AXIS
    zero_point: int = QUANTIZATION_ZERO_POINT
    tensors: tuple[QuantizedTensorReport, ...] = ()
    excluded: tuple[dict[str, Any], ...] = ()
    num_linear_nodes: int = 0
    num_rewritten_nodes: int = 0
    original_bytes: int = 0
    quantized_bytes: int = 0
    compression_ratio: float = 1.0
    package_original_bytes: int = 0
    package_quantized_bytes: int = 0
    package_compression_ratio: float = 1.0
    max_abs_error: float = 0.0
    mean_abs_error: float = 0.0
    graph_valid: bool = False
    validation_errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "uzen.quantization/1",
            "graph_name": self.graph_name,
            "source_package": self.source_package,
            "scheme": self.scheme,
            "bits": self.bits,
            "axis": self.axis,
            "zero_point": self.zero_point,
            "tensors": [item.to_dict() for item in self.tensors],
            "excluded": [dict(item) for item in self.excluded],
            "num_linear_nodes": self.num_linear_nodes,
            "num_rewritten_nodes": self.num_rewritten_nodes,
            "original_bytes": self.original_bytes,
            "quantized_bytes": self.quantized_bytes,
            "compression_ratio": self.compression_ratio,
            "package_original_bytes": self.package_original_bytes,
            "package_quantized_bytes": self.package_quantized_bytes,
            "package_compression_ratio": self.package_compression_ratio,
            "max_abs_error": self.max_abs_error,
            "mean_abs_error": self.mean_abs_error,
            "graph_valid": self.graph_valid,
            "validation_errors": list(self.validation_errors),
        }


@dataclass
class QuantizedModel:
    """A graph and package whose linear weights are int8."""

    graph: Graph
    package: WeightPackage
    report: QuantizationReport

    def execute(
        self,
        inputs: Mapping[str, np.ndarray],
        *,
        capture: Sequence[str] | None = None,
        check_finite: bool = True,
        validate_types: bool = True,
    ) -> ExecutionResult:
        return execute(
            self.graph,
            inputs,
            self.package,
            capture=capture,
            check_finite=check_finite,
            validate_types=validate_types,
        )


def _param_name(value: Any) -> str | None:
    if not isinstance(value, str) or not value.startswith("param:"):
        return None
    name = value[len("param:") :]
    return name or None


def _scale_name(weight_name: str) -> str:
    return f"{weight_name}.int8_scale"


def _quantize_tensor(name: str, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, QuantizedTensorReport]:
    source = np.asarray(weight)
    if source.ndim != 2:
        raise InvalidOperationParameters(f"quantized linear weight '{name}' must be 2-D")
    if source.dtype.kind not in "fc":
        raise InvalidOperationParameters(
            f"quantized linear weight '{name}' must be floating point, got {source.dtype}"
        )
    if not np.isfinite(source).all():
        raise TensorInvalid(f"quantized linear weight '{name}' contains non-finite values")
    values = source.astype(np.float64, copy=False)
    absmax = np.max(np.abs(values), axis=1)
    scales = np.where(absmax > 0, absmax / 127.0, 1.0).astype(np.float32)
    if scales.ndim != 1 or scales.size != values.shape[0]:
        raise InvalidOperationParameters(f"quantization scales for '{name}' have the wrong shape")
    scaled = values / scales.astype(np.float64)[:, None]
    saturated = int(np.count_nonzero(np.abs(scaled) > 127.0))
    quantized = np.clip(np.rint(scaled), -127, 127).astype(np.int8)
    dequantized = quantized.astype(np.float64) * scales.astype(np.float64)[:, None]
    difference = np.abs(values - dequantized)
    denominator = np.maximum(np.abs(values), 1e-12)
    scale_bytes = np.ascontiguousarray(scales).tobytes()
    report = QuantizedTensorReport(
        name=name,
        shape=tuple(int(v) for v in source.shape),
        source_dtype=str(source.dtype),
        quantized_dtype="int8",
        scheme=QUANTIZATION_SCHEME,
        axis=QUANTIZATION_AXIS,
        zero_point=QUANTIZATION_ZERO_POINT,
        scale_count=int(scales.size),
        scale_min=float(scales.min()),
        scale_max=float(scales.max()),
        scale_sha256=hashlib.sha256(scale_bytes).hexdigest(),
        original_bytes=int(source.nbytes),
        quantized_bytes=int(quantized.nbytes + scales.nbytes),
        bits_per_weight=float((quantized.nbytes + scales.nbytes) * 8 / max(1, source.size)),
        max_abs_error=float(difference.max()) if difference.size else 0.0,
        mean_abs_error=float(difference.mean()) if difference.size else 0.0,
        max_relative_error=float((difference / denominator).max()) if difference.size else 0.0,
        saturated_values=saturated,
    )
    return quantized, scales, report


def quantize_int8(graph: Graph, package: WeightPackage) -> QuantizedModel:
    """Return a derived graph/package with symmetric int8 linear weights.

    The source graph and package are never modified.  If a candidate weight is
    shared with a non-linear operator, or is not a 2-D float tensor, its linear
    node is left unchanged and the reason is recorded in the report instead of
    silently widening or approximating the layout.
    """
    package.assert_unmodified()
    owners: dict[str, set[str]] = {}
    for node in graph.nodes.values():
        for parameter in node.param_names():
            owners.setdefault(parameter, set()).add(node.op)
    candidate_weights: dict[str, str] = {}
    excluded: list[dict[str, Any]] = []
    linear_nodes = 0
    for node in graph.nodes.values():
        if node.op != "linear":
            continue
        linear_nodes += 1
        weight_name = _param_name(node.params.get("weight"))
        if weight_name is None:
            excluded.append(
                {
                    "node": node.id,
                    "reason": "linear node does not use a direct param: weight reference",
                }
            )
            continue
        if weight_name not in package:
            excluded.append(
                {"weight": weight_name, "node": node.id, "reason": "weight is absent from the package"}
            )
            continue
        non_linear_owners = set(owners.get(weight_name, set())) - {"linear"}
        if non_linear_owners:
            excluded.append(
                {
                    "weight": weight_name,
                    "reason": "weight is also consumed by non-linear operators: "
                    + ", ".join(sorted(non_linear_owners)),
                }
            )
            continue
        candidate_weights.setdefault(weight_name, node.id)

    canonical: dict[str, np.ndarray] = {name: np.asarray(array) for name, array in package.tensors.items()}
    tensor_reports: list[QuantizedTensorReport] = []
    quantized_names: set[str] = set()
    for weight_name in sorted(candidate_weights):
        try:
            quantized, scales, tensor_report = _quantize_tensor(
                weight_name, np.asarray(package.get(weight_name))
            )
        except (InvalidOperationParameters, TensorInvalid) as exc:
            excluded.append({"weight": weight_name, "reason": str(exc)})
            continue
        scale_name = _scale_name(weight_name)
        if scale_name in canonical:
            raise InvalidOperationParameters(
                f"cannot quantize '{weight_name}': auxiliary name '{scale_name}' already exists"
            )
        canonical[weight_name] = quantized
        canonical[scale_name] = scales
        quantized_names.add(weight_name)
        tensor_reports.append(tensor_report)

    new_graph = Graph.from_dict(graph.to_dict())
    rewritten = 0
    for node in new_graph.nodes.values():
        if node.op != "linear":
            continue
        weight_name = _param_name(node.params.get("weight"))
        if weight_name not in quantized_names:
            continue
        node.op = "quantized_linear"
        node.params["scale"] = f"param:{_scale_name(weight_name)}"
        node.params["quantization"] = QUANTIZATION_SCHEME
        node.note = (node.note + " " if node.note else "") + "weight-only int8 execution"
        rewritten += 1

    original_bytes = sum(item.original_bytes for item in tensor_reports)
    quantized_bytes = sum(item.quantized_bytes for item in tensor_reports)
    package_original_bytes = int(package.nbytes())
    package_quantized_bytes = int(sum(np.asarray(value).nbytes for value in canonical.values()))
    total_elements = sum(int(np.prod(item.shape)) for item in tensor_reports)
    weighted_error = sum(
        item.mean_abs_error * int(np.prod(item.shape)) for item in tensor_reports
    )
    base_package = WeightPackage.from_arrays(
        canonical,
        name=f"{package.name}.int8",
        family=package.family,
        family_spec=dict(package.family_spec),
        metadata=dict(package.metadata),
        source_artifact_id=package.source_artifact_id,
    )
    validation = new_graph.validate(base_package)
    if not validation.ok:
        raise GraphValidationError(
            "int8-quantized graph failed validation",
            errors=list(validation.errors),
        )
    report = QuantizationReport(
        graph_name=new_graph.name,
        source_package=package.name,
        tensors=tuple(tensor_reports),
        excluded=tuple(excluded),
        num_linear_nodes=linear_nodes,
        num_rewritten_nodes=rewritten,
        original_bytes=original_bytes,
        quantized_bytes=quantized_bytes,
        compression_ratio=(
            float(original_bytes / quantized_bytes) if quantized_bytes else 1.0
        ),
        package_original_bytes=package_original_bytes,
        package_quantized_bytes=package_quantized_bytes,
        package_compression_ratio=(
            float(package_original_bytes / package_quantized_bytes)
            if package_quantized_bytes
            else 1.0
        ),
        max_abs_error=max((item.max_abs_error for item in tensor_reports), default=0.0),
        mean_abs_error=(float(weighted_error / total_elements) if total_elements else 0.0),
        graph_valid=True,
        validation_errors=tuple(validation.errors),
    )
    new_graph.metadata["quantization"] = report.to_dict()
    final_package = WeightPackage.from_arrays(
        canonical,
        name=base_package.name,
        family=package.family,
        family_spec=dict(package.family_spec),
        metadata={**dict(package.metadata), "quantization": report.to_dict()},
        source_artifact_id=package.source_artifact_id,
    )
    return QuantizedModel(graph=new_graph, package=final_package, report=report)
