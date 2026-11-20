"""Inference-only weight transformations used by parameter campaigns.

These are the same kind of deterministic transformations as N3/N4: scaling,
magnitude pruning and a symmetric int8 round-trip simulation. They never train,
optimize or update weights. A transformed package is a new immutable package, and
the source package is never modified.
"""

from __future__ import annotations

import numpy as np

from uzen.core.nws.tensors import Origin, WeightPackage
from uzen.errors import InvalidOperationParameters


def _float_tensors(package: WeightPackage) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name in package.names():
        array = package.get(name)
        if np.issubdtype(array.dtype, np.floating):
            arrays[name] = np.asarray(array, dtype=np.float32)
    return arrays


def _prune_tensor(array: np.ndarray, fraction: float) -> np.ndarray:
    if fraction <= 0 or array.size == 0:
        return array
    flat = np.abs(array.reshape(-1))
    order = np.argsort(flat, kind="stable")
    keep = int(round(array.size * (1.0 - fraction)))
    keep = max(0, min(array.size, keep))
    pruned = np.array(array, copy=True)
    flat_pruned = pruned.reshape(-1)
    flat_pruned[order[: array.size - keep]] = 0.0
    return pruned


def _quantize_tensor(array: np.ndarray) -> tuple[np.ndarray, float]:
    if array.size == 0:
        return array, 0.0
    maximum = float(np.max(np.abs(array)))
    if maximum == 0.0:
        return np.zeros_like(array), 0.0
    scale = maximum / 127.0
    quantized = np.clip(np.rint(array / scale), -127, 127).astype(np.int8)
    restored = (quantized.astype(np.float32) * scale).astype(np.float32)
    denominator = float(np.linalg.norm(array.astype(np.float64)))
    error = float(np.linalg.norm((restored - array).astype(np.float64)) / max(denominator, 1e-12))
    return restored, error


def apply_transforms(
    package: WeightPackage,
    transforms: dict[str, object],
) -> tuple[WeightPackage, dict[str, float]]:
    """Apply scale, sparsity and optional int8 simulation.

    Only matrices (ndim >= 2) are pruned or quantized; norm/bias vectors keep full
    precision so a transform cannot masquerade as a broken layer. Scaling applies
    to every float tensor. The returned diagnostics are measured, not estimated.
    """
    unknown = set(transforms) - {"scale", "sparsity", "quantize"}
    if unknown:
        raise InvalidOperationParameters(f"unknown weight transforms: {sorted(unknown)}")
    scale = float(transforms.get("scale", 1.0))
    sparsity = float(transforms.get("sparsity", 0.0))
    quantize = str(transforms.get("quantize", "none"))
    if not np.isfinite(scale) or scale <= 0:
        raise InvalidOperationParameters("weight scale must be finite and positive")
    if not 0.0 <= sparsity < 1.0:
        raise InvalidOperationParameters("weight sparsity must be in [0, 1)")
    if quantize not in ("none", "int8"):
        raise InvalidOperationParameters("weight quantize must be none|int8")

    arrays = _float_tensors(package)
    diagnostics: dict[str, float] = {
        "scale": scale,
        "requested_sparsity": sparsity,
        "quantization_rmse": 0.0,
    }
    total_values = 0
    zero_values = 0
    quantized_errors: list[float] = []
    for name, source in arrays.items():
        array = np.array(source, copy=True)
        if scale != 1.0:
            array = (array * scale).astype(np.float32)
        if array.ndim >= 2 and sparsity > 0:
            array = _prune_tensor(array, sparsity)
        if array.ndim >= 2 and quantize == "int8":
            array, error = _quantize_tensor(array)
            quantized_errors.append(error)
        arrays[name] = array.astype(np.float32, copy=False)
        total_values += array.size
        zero_values += int(np.count_nonzero(array == 0.0))
    diagnostics["realized_sparsity"] = (
        float(zero_values / total_values) if total_values else 0.0
    )
    diagnostics["quantization_rmse"] = (
        float(np.mean(quantized_errors)) if quantized_errors else 0.0
    )
    transformed = WeightPackage.from_arrays(
        arrays,
        name=f"{package.name}:campaign",
        family=package.family,
        origin=Origin.TRANSFORMED,
        source_artifact_id=package.source_artifact_id,
        family_spec=dict(package.family_spec),
        metadata={
            **dict(package.metadata),
            "campaign_transform": {
                "scale": scale,
                "sparsity": sparsity,
                "quantize": quantize,
            },
            "initialization": "transformed; NOT learned",
        },
        ops=("campaign.transform",),
    )
    return transformed, diagnostics
