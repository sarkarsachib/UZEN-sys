"""Real, inference-only candidate evaluation.

Every metric comes from a concrete forward pass or from the package bytes. There is
no surrogate model, no training loop and no score inferred from a filename. The
campaign engine records ``invalid`` / ``error`` trials rather than hiding them.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from uzen.core.campaign.models import Candidate
from uzen.core.campaign.spec import ARCHITECTURE_MODE, WEIGHT_MODE, CampaignSpec
from uzen.core.campaign.transforms import apply_transforms
from uzen.core.nir.executor import execute
from uzen.core.nws.families import ArchitectureDefinition, build_architecture
from uzen.core.nws.tensors import WeightPackage
from uzen.util.canonical import canonical_bytes


def _relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("metric operands must have identical shapes")
    denominator = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b) / max(denominator, 1e-12))


def _int8_roundtrip_error(package: WeightPackage) -> float:
    total_sq = 0.0
    total_ref = 0.0
    for name in package.names():
        array = np.asarray(package.get(name), dtype=np.float32)
        if array.ndim < 2 or array.size == 0:
            continue
        maximum = float(np.max(np.abs(array)))
        if maximum == 0.0:
            continue
        scale = maximum / 127.0
        quantized = np.clip(np.rint(array / scale), -127, 127).astype(np.int8)
        restored = quantized.astype(np.float64) * scale
        difference = restored - array.astype(np.float64)
        total_sq += float(np.sum(difference * difference))
        total_ref += float(np.sum(array.astype(np.float64) ** 2))
    return float(np.sqrt(total_sq / max(total_ref, 1e-12)))


def _output_metrics(outputs: dict[str, np.ndarray], batch: int) -> dict[str, float]:
    arrays = [np.asarray(value, dtype=np.float64) for value in outputs.values()]
    if not arrays:
        return {"output_rms": 0.0, "output_diversity": 0.0, "output_bytes": 0}
    squares = sum(float(np.sum(array * array)) for array in arrays)
    count = sum(array.size for array in arrays)
    rms = float(np.sqrt(squares / max(count, 1)))
    diversities: list[float] = []
    for array in arrays:
        if array.ndim == 0 or array.shape[0] != batch or batch < 2:
            diversities.append(0.0)
            continue
        rows = array.reshape(batch, -1)
        norms = np.linalg.norm(rows, axis=1)
        distances: list[float] = []
        for left in range(batch):
            for right in range(left + 1, batch):
                denominator = max(0.5 * (float(norms[left]) + float(norms[right])), 1e-12)
                distances.append(float(np.linalg.norm(rows[left] - rows[right]) / denominator))
        diversities.append(float(np.mean(distances)) if distances else 0.0)
    return {
        "output_rms": rms,
        "output_diversity": float(np.mean(diversities)),
        "output_bytes": int(sum(array.nbytes for array in outputs.values())),
    }


def _package_metrics(package: WeightPackage) -> dict[str, float]:
    arrays = [np.asarray(package.get(name)) for name in package.names()]
    total = sum(int(array.size) for array in arrays)
    zeros = sum(int(np.count_nonzero(array == 0.0)) for array in arrays)
    return {
        "parameters": int(package.numel()),
        "parameter_bytes": int(package.nbytes()),
        "weight_sparsity": float(zeros / total) if total else 0.0,
        "quantization_rmse": _int8_roundtrip_error(package),
    }


class Evaluator:
    """Build and execute one candidate against the NumPy reference runtime."""

    def __init__(self, spec: CampaignSpec) -> None:
        self.spec = spec
        self._base_definition: ArchitectureDefinition | None = None
        self._base_package: WeightPackage | None = None
        self._base_inputs: dict[str, np.ndarray] | None = None
        self._base_outputs: dict[str, np.ndarray] | None = None
        if spec.mode == WEIGHT_MODE:
            definition = build_architecture(spec.family, dict(spec.base_spec))
            package = definition.build_package(seed=int(spec.evaluation["init_seed"]))
            inputs = definition.sample_inputs(
                batch=int(spec.evaluation["batch"]),
                seq=int(spec.evaluation["seq"]),
                seed=int(spec.evaluation["input_seed"]),
            )
            outputs = execute(definition.graph, inputs, package, check_finite=True).outputs
            self._base_definition = definition
            self._base_package = package
            self._base_inputs = inputs
            self._base_outputs = outputs
        elif spec.mode != ARCHITECTURE_MODE:  # pragma: no cover - spec validation
            raise ValueError(f"unsupported campaign mode '{spec.mode}'")

    def evaluate(self, candidate: Candidate) -> tuple[dict[str, Any], str, str]:
        if self.spec.mode == ARCHITECTURE_MODE:
            definition = build_architecture(self.spec.family, dict(candidate.spec))
            package = definition.build_package(seed=int(self.spec.evaluation["init_seed"]))
            inputs = definition.sample_inputs(
                batch=int(self.spec.evaluation["batch"]),
                seq=int(self.spec.evaluation["seq"]),
                seed=int(self.spec.evaluation["input_seed"]),
            )
            base_outputs = None
        else:
            assert self._base_definition is not None
            assert self._base_package is not None
            assert self._base_inputs is not None
            definition = self._base_definition
            package, _diagnostics = apply_transforms(self._base_package, candidate.transforms)
            inputs = {name: np.array(value, copy=True) for name, value in self._base_inputs.items()}
            base_outputs = self._base_outputs

        result = execute(definition.graph, inputs, package, check_finite=True)
        metrics: dict[str, Any] = {
            **_package_metrics(package),
            "nodes": int(len(definition.graph.nodes)),
            **_output_metrics(result.outputs, int(self.spec.evaluation["batch"])),
        }
        if base_outputs is not None:
            assert self._base_package is not None
            metrics["distance_from_base"] = self._relative_package_distance(package)
            agreements: list[float] = []
            for name, base in base_outputs.items():
                if name in result.outputs:
                    distance = _relative_l2(result.outputs[name], base)
                    agreements.append(1.0 - distance)
            metrics["activation_agreement"] = (
                float(np.mean(agreements)) if agreements else 0.0
            )
        graph_digest = hashlib.sha256(canonical_bytes(definition.graph.to_dict())).hexdigest()
        return metrics, package.fingerprint(), graph_digest

    def _relative_package_distance(self, package: WeightPackage) -> float:
        assert self._base_package is not None
        names = [name for name in package.names() if name in self._base_package]
        numerator = 0.0
        denominator = 0.0
        for name in names:
            current = np.asarray(package.get(name), dtype=np.float64).reshape(-1)
            base = np.asarray(self._base_package.get(name), dtype=np.float64).reshape(-1)
            numerator += float(np.sum((current - base) ** 2))
            denominator += float(np.sum(base**2))
        return float(np.sqrt(numerator / max(denominator, 1e-12)))


def evaluate_candidate(spec: CampaignSpec, candidate: Candidate) -> dict[str, Any]:
    """Convenience one-shot evaluation used by corpus replay and tests."""
    metrics, fingerprint, graph_digest = Evaluator(spec).evaluate(candidate)
    return {
        "status": "ok",
        "metrics": metrics,
        "package_fingerprint": fingerprint,
        "graph_digest": graph_digest,
    }
