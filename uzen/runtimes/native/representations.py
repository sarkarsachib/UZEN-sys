"""Native parameter representations for compiled plans (phase N3).

A compiled plan's GEMM steps can consume the same linear weight through different
native layouts. The NIR graph is unchanged: only how the parameter is stored and
which kernel reads it changes.

| Kind | Storage | Kernel |
| --- | --- | --- |
| ``dense_blocked`` | 64x64 float32 panels (N2's prepack) | ``uzen_gemm_packed_f32`` |
| ``transposed`` | float32 (in, out) contiguous | ``uzen_gemm_transposed_f32`` |
| ``int8_block`` | int8 panels + one float32 scale per (row, 64-value group) | ``uzen_gemm_int8_block_f32`` |
| ``int4_block`` | 4-bit two's-complement pairs + per-group scales | ``uzen_gemm_int4_block_f32`` |
| ``low_rank`` | float32 factors u (out, r) and v (r, in) | ``uzen_gemm_low_rank_f32`` |
| ``sparse_csr`` | float32 values + int32 indices, CSR rows over outputs | ``uzen_sparse_gemm_f32`` |

Every lossy representation is an explicit transform with a measured report: the
number of bits per weight, the packed byte count, the reconstruction error against
the source tensor and a digest of the packed bytes. Nothing is chosen silently and
no approximation is reported as exact.

No parameter is trained or updated here. Quantization, magnitude pruning and SVD
are one-shot representation transforms on fixed weights; activations stay float32.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from uzen.errors import InvalidOperationParameters, TensorInvalid

from .library import (
    BLOCK_K,
    BLOCK_N,
    INT4_BLOCK_RANGE,
    INT8_BLOCK_RANGE,
    NativeLibrary,
    PackedBlockWeight,
    PackedTransposedWeight,
    PackedWeight,
)

#: Values per quantized scale group (matches the compiled kernel's kBlockK).
GROUP_SIZE = BLOCK_K
INT8_MAX = INT8_BLOCK_RANGE
INT4_MAX = INT4_BLOCK_RANGE

REPRESENTATION_KINDS = (
    "dense_blocked",
    "transposed",
    "int8_block",
    "int4_block",
    "low_rank",
    "sparse_csr",
)

REPRESENTATION_NOTES = {
    "dense_blocked": "blocked float32 panels, exact",
    "transposed": "in_features-major float32, exact",
    "int8_block": "group-wise symmetric int8 with per-group scales",
    "int4_block": "group-wise symmetric int4 (two nibbles per byte)",
    "low_rank": "rank-r SVD factors, approximate",
    "sparse_csr": "magnitude-pruned CSR rows over output features, approximate unless threshold is 0",
}

_OPTIONS: dict[str, frozenset[str]] = {
    "dense_blocked": frozenset({"kind"}),
    "transposed": frozenset({"kind"}),
    "int8_block": frozenset({"kind"}),
    "int4_block": frozenset({"kind"}),
    "low_rank": frozenset({"kind", "rank"}),
    "sparse_csr": frozenset({"kind", "threshold"}),
}


@dataclass(frozen=True)
class RepresentationReport:
    """What a parameter representation costs and how far it moved the numbers."""

    name: str
    kind: str
    shape: tuple[int, int]
    source_dtype: str
    packed_bytes: int
    source_bytes: int
    max_abs_error: float = 0.0
    mean_abs_error: float = 0.0
    max_relative_error: float = 0.0
    group_size: int = 0
    groups: int = 0
    zero_point: int = 0
    saturated_values: int = 0
    scale_min: float = 0.0
    scale_max: float = 0.0
    scale_sha256: str = ""
    rank: int = 0
    spectral_energy: float = 0.0
    nonzeros: int = 0
    sparsity: float = 0.0
    threshold: float = 0.0
    digest: str = ""

    @property
    def elements(self) -> int:
        return int(self.shape[0]) * int(self.shape[1])

    @property
    def bits_per_weight(self) -> float:
        return self.packed_bytes * 8 / self.elements if self.elements else 0.0

    @property
    def compression_ratio(self) -> float:
        return self.source_bytes / self.packed_bytes if self.packed_bytes else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "shape": list(self.shape),
            "source_dtype": self.source_dtype,
            "bits_per_weight": round(self.bits_per_weight, 4),
            "packed_bytes": self.packed_bytes,
            "source_bytes": self.source_bytes,
            "compression_ratio": round(self.compression_ratio, 4),
            "max_abs_error": self.max_abs_error,
            "mean_abs_error": self.mean_abs_error,
            "max_relative_error": self.max_relative_error,
            "group_size": self.group_size,
            "groups": self.groups,
            "zero_point": self.zero_point,
            "saturated_values": self.saturated_values,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
            "scale_sha256": self.scale_sha256,
            "rank": self.rank,
            "spectral_energy": self.spectral_energy,
            "nonzeros": self.nonzeros,
            "sparsity": self.sparsity,
            "threshold": self.threshold,
            "digest": self.digest,
        }


@dataclass(frozen=True)
class NativeWeight:
    """A prepared linear weight in one native representation."""

    name: str
    kind: str
    in_features: int
    out_features: int
    report: RepresentationReport
    packed: PackedWeight | PackedBlockWeight | PackedTransposedWeight | None = None
    factors: tuple[np.ndarray, np.ndarray] | None = None
    sparse: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    @property
    def byte_size(self) -> int:
        return self.report.packed_bytes

    def describe(self) -> dict[str, Any]:
        """The report plus the packed layout's own description.

        Exact layouts keep the ``padding_overhead`` / ``padded_elements`` /
        ``block_in`` / ``block_out`` fields that N2's ``PackedWeight.describe()``
        exposed, so callers written against the dense prepack keep working.
        """
        described = self.report.to_dict()
        if self.packed is not None:
            for key, value in self.packed.describe().items():
                described.setdefault(key, value)
        else:
            described.setdefault("padding_overhead", 1.0)
        return described

    def to_dense(self) -> np.ndarray:
        """Reconstruct the float32 matrix this representation stands for."""
        if self.kind == "dense_blocked":
            if not isinstance(self.packed, PackedWeight):
                raise InvalidOperationParameters(
                    f"'{self.name}' has no blocked dense payload"
                )
            return _unpack_blocked(self.packed.array, self.in_features, self.out_features)
        if self.kind == "transposed":
            if not isinstance(self.packed, PackedTransposedWeight):
                raise InvalidOperationParameters(
                    f"'{self.name}' has no transposed dense payload"
                )
            return np.ascontiguousarray(
                self.packed.array.reshape(self.in_features, self.out_features).T,
                dtype=np.float32,
            )
        if self.kind in ("int8_block", "int4_block"):
            return _dequantize_block(self)
        if self.kind == "low_rank":
            if self.factors is None:
                raise InvalidOperationParameters(
                    f"'{self.name}' has no low-rank factors"
                )
            u, v = self.factors
            return np.ascontiguousarray(u @ v, dtype=np.float32)
        if self.kind == "sparse_csr":
            if self.sparse is None:
                raise InvalidOperationParameters(f"'{self.name}' has no CSR payload")
            values, indices, row_ptr = self.sparse
            dense = np.zeros((self.out_features, self.in_features), dtype=np.float32)
            for o in range(self.out_features):
                start, stop = int(row_ptr[o]), int(row_ptr[o + 1])
                dense[o, indices[start:stop]] = values[start:stop]
            return dense
        raise InvalidOperationParameters(
            f"'{self.name}' uses unknown representation '{self.kind}'"
        )


# --------------------------------------------------------------------------- #
# option normalisation
# --------------------------------------------------------------------------- #


def normalize_representations(
    spec: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Validate a ``weight name -> kind/options`` mapping.

    Accepted forms per weight: a kind string, or a mapping with ``kind`` plus the
    options that kind needs (``rank`` for ``low_rank``, ``threshold`` for
    ``sparse_csr``). Unknown kinds and unknown option keys are refused rather
    than ignored.
    """
    if spec is None:
        return {}
    if not isinstance(spec, Mapping):
        raise InvalidOperationParameters(
            "representations must be a mapping of weight name to kind or options"
        )
    normalized: dict[str, dict[str, Any]] = {}
    for name, value in spec.items():
        if not isinstance(name, str) or not name:
            raise InvalidOperationParameters(
                f"representation keys must be non-empty weight names, got {name!r}"
            )
        if isinstance(value, str):
            options: dict[str, Any] = {"kind": value}
        elif isinstance(value, Mapping):
            options = dict(value)
            if "kind" not in options:
                raise InvalidOperationParameters(
                    f"representation for '{name}' has no 'kind'"
                )
        else:
            raise InvalidOperationParameters(
                f"representation for '{name}' must be a kind string or mapping, "
                f"got {type(value).__name__}"
            )
        kind = options["kind"]
        if kind not in REPRESENTATION_KINDS:
            raise InvalidOperationParameters(
                f"unknown representation '{kind}' for '{name}'; available: "
                f"{', '.join(REPRESENTATION_KINDS)}"
            )
        allowed = _OPTIONS[kind]
        unknown = set(options) - allowed
        if unknown:
            raise InvalidOperationParameters(
                f"representation '{kind}' does not accept option(s) "
                f"{sorted(unknown)}; accepted: {sorted(allowed)}"
            )
        if kind == "low_rank" and "rank" not in options:
            raise InvalidOperationParameters(
                f"low_rank representation for '{name}' needs an explicit 'rank'"
            )
        if kind == "sparse_csr" and "threshold" not in options:
            raise InvalidOperationParameters(
                f"sparse_csr representation for '{name}' needs an explicit "
                "'threshold' (0.0 keeps exactly the nonzero weights)"
            )
        if kind == "low_rank":
            rank = options["rank"]
            if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
                raise InvalidOperationParameters(
                    f"low_rank rank for '{name}' must be a positive integer, got {rank!r}"
                )
        if kind == "sparse_csr":
            threshold = options["threshold"]
            if (
                not isinstance(threshold, (int, float))
                or isinstance(threshold, bool)
                or not np.isfinite(threshold)
                or threshold < 0
            ):
                raise InvalidOperationParameters(
                    f"sparse_csr threshold for '{name}' must be a finite "
                    f"non-negative number, got {threshold!r}"
                )
            options["threshold"] = float(threshold)
        normalized[name] = options
    return normalized


def representation_key(spec: Mapping[str, Mapping[str, Any]]) -> str:
    """Stable cache key for a normalised representation mapping."""
    payload = json.dumps(spec, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _digest(
    kind: str, shape: tuple[int, int], arrays: tuple[np.ndarray, ...], extra: Any = None
) -> str:
    digest = hashlib.sha256()
    digest.update(f"{kind}|{shape[0]}x{shape[1]}".encode("utf-8"))
    if extra is not None:
        digest.update(json.dumps(extra, sort_keys=True).encode("utf-8"))
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _errors(source: np.ndarray, approximation: np.ndarray) -> tuple[float, float, float]:
    difference = np.abs(source - approximation)
    denominator = np.maximum(np.abs(source), 1e-12)
    return (
        float(difference.max()) if difference.size else 0.0,
        float(difference.mean()) if difference.size else 0.0,
        float((difference / denominator).max()) if difference.size else 0.0,
    )


def _source(name: str, weight: Any, *, exact: bool) -> np.ndarray:
    array = np.asarray(weight)
    if array.ndim != 2:
        raise InvalidOperationParameters(
            f"native representation for '{name}' requires a 2-D weight, got rank "
            f"{array.ndim}"
        )
    if array.dtype.kind not in "fc":
        raise InvalidOperationParameters(
            f"native representation for '{name}' requires a floating-point "
            f"weight, got {array.dtype}"
        )
    if array.size == 0:
        raise InvalidOperationParameters(
            f"native representation for '{name}' requires a non-empty weight"
        )
    if not np.isfinite(array).all():
        raise TensorInvalid(
            f"native representation for '{name}' got non-finite weights"
        )
    if exact and array.dtype != np.float32:
        raise InvalidOperationParameters(
            f"exact representation '{name}' requires float32 weights, got "
            f"{array.dtype}; use a quantized or factorized representation for "
            "reduced precision"
        )
    return np.ascontiguousarray(array)


def _quantize_groups(
    values: np.ndarray, maximum: int
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Symmetric group-wise quantization; returns (q, scales, groups, saturated)."""
    out_features, in_features = values.shape
    groups = -(-in_features // GROUP_SIZE)
    padded = np.zeros((out_features, groups * GROUP_SIZE), dtype=np.float64)
    padded[:, :in_features] = values
    blocks = padded.reshape(out_features, groups, GROUP_SIZE)
    absmax = np.max(np.abs(blocks), axis=2)
    scales = np.where(absmax > 0, absmax / float(maximum), 1.0)
    scaled = blocks / scales[:, :, None]
    saturated = int(np.count_nonzero(np.abs(scaled) > float(maximum) + 1e-9))
    quantized = np.clip(np.rint(scaled), -maximum, maximum)
    return quantized, scales, groups, saturated


def _unpack_blocked(packed: np.ndarray, in_features: int, out_features: int) -> np.ndarray:
    """Inverse of the C++ blocked prepack, for verification only."""
    blocks_out = -(-out_features // BLOCK_N)
    blocks_in = -(-in_features // BLOCK_K)
    panels = packed.reshape(blocks_out, blocks_in, BLOCK_K, BLOCK_N)
    dense = panels.transpose(0, 3, 1, 2).reshape(
        blocks_out * BLOCK_N, blocks_in * BLOCK_K
    )
    return np.ascontiguousarray(
        dense[:out_features, :in_features], dtype=np.float32
    )


def _dequantize_block(weight: NativeWeight) -> np.ndarray:
    packed = weight.packed
    if not isinstance(packed, PackedBlockWeight):
        raise InvalidOperationParameters(f"'{weight.name}' has no packed block payload")
    blocks_out = -(-weight.out_features // BLOCK_N)
    blocks_in = -(-weight.in_features // GROUP_SIZE)
    scales = packed.scales.reshape(blocks_out, blocks_in, BLOCK_N)
    if weight.kind == "int8_block":
        panels = packed.weight.reshape(blocks_out, blocks_in, GROUP_SIZE, BLOCK_N)
        values = panels.astype(np.float64)
    else:
        packed_bytes = packed.weight.reshape(blocks_out, blocks_in, GROUP_SIZE // 2, BLOCK_N)
        low = (packed_bytes & 0x0F).astype(np.int16)
        high = (packed_bytes >> 4).astype(np.int16)
        low = np.where(low >= 8, low - 16, low)
        high = np.where(high >= 8, high - 16, high)
        interleaved = np.stack([low, high], axis=3)
        values = interleaved.reshape(blocks_out, blocks_in, GROUP_SIZE, BLOCK_N).astype(
            np.float64
        )
    dense = values.transpose(0, 3, 1, 2).reshape(
        blocks_out * BLOCK_N, blocks_in * GROUP_SIZE
    )
    scale_grid = scales.transpose(0, 2, 1).reshape(blocks_out * BLOCK_N, blocks_in)
    scale_grid = np.repeat(scale_grid, GROUP_SIZE, axis=1)
    dequantized = dense * scale_grid
    return np.ascontiguousarray(
        dequantized[: weight.out_features, : weight.in_features], dtype=np.float32
    )


def _sparse_payload(
    values: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out_features, in_features = values.shape
    mask = np.abs(values) > threshold
    counts = mask.sum(axis=1)
    row_ptr = np.zeros(out_features + 1, dtype=np.int32)
    row_ptr[1:] = np.cumsum(counts).astype(np.int32)
    nonzeros = int(counts.sum())
    indices = np.empty(nonzeros, dtype=np.int32)
    kept = np.empty(nonzeros, dtype=np.float32)
    cursor = 0
    for o in range(out_features):
        row_indices = np.nonzero(mask[o])[0].astype(np.int32)
        count = int(row_indices.size)
        indices[cursor : cursor + count] = row_indices
        kept[cursor : cursor + count] = values[o, row_indices].astype(np.float32)
        cursor += count
    return kept, indices, row_ptr


# --------------------------------------------------------------------------- #
# preparation
# --------------------------------------------------------------------------- #


def prepare_representation(
    name: str,
    weight: Any,
    kind: str = "dense_blocked",
    *,
    library: NativeLibrary,
    rank: int | None = None,
    threshold: float | None = None,
) -> NativeWeight:
    """Prepare ``weight`` in one native representation with a measured report."""
    if kind not in REPRESENTATION_KINDS:
        raise InvalidOperationParameters(
            f"unknown representation '{kind}'; available: "
            f"{', '.join(REPRESENTATION_KINDS)}"
        )
    exact = kind in ("dense_blocked", "transposed")
    array = _source(name, weight, exact=exact)
    out_features, in_features = (int(value) for value in array.shape)
    values = array.astype(np.float64, copy=False)
    source_bytes = int(array.nbytes)
    shape = (out_features, in_features)

    if kind == "dense_blocked":
        packed = library.pack_weight(array)
        dense_source = _unpack_blocked(packed.array, in_features, out_features)
        max_abs, mean_abs, max_rel = _errors(
            values, dense_source.astype(np.float64)
        )
        report = RepresentationReport(
            name=name,
            kind=kind,
            shape=shape,
            source_dtype=str(array.dtype),
            packed_bytes=int(packed.array.nbytes),
            source_bytes=source_bytes,
            max_abs_error=max_abs,
            mean_abs_error=mean_abs,
            max_relative_error=max_rel,
            digest=_digest(kind, shape, (packed.array,)),
        )
        return NativeWeight(
            name=name,
            kind=kind,
            in_features=in_features,
            out_features=out_features,
            report=report,
            packed=packed,
        )

    if kind == "transposed":
        packed = library.pack_weight_transposed(array)
        max_abs, mean_abs, max_rel = _errors(
            values, packed.array.reshape(in_features, out_features).T.astype(np.float64)
        )
        report = RepresentationReport(
            name=name,
            kind=kind,
            shape=shape,
            source_dtype=str(array.dtype),
            packed_bytes=int(packed.array.nbytes),
            source_bytes=source_bytes,
            max_abs_error=max_abs,
            mean_abs_error=mean_abs,
            max_relative_error=max_rel,
            digest=_digest(kind, shape, (packed.array,)),
        )
        return NativeWeight(
            name=name,
            kind=kind,
            in_features=in_features,
            out_features=out_features,
            report=report,
            packed=packed,
        )

    if kind in ("int8_block", "int4_block"):
        maximum = INT8_MAX if kind == "int8_block" else INT4_MAX
        quantized, scales, groups, saturated = _quantize_groups(values, maximum)
        scales32 = np.ascontiguousarray(scales, dtype=np.float32)
        # Drop the group padding again: the packing kernels take (out, in).
        quantized_2d = quantized.reshape(out_features, groups * GROUP_SIZE)[
            :, :in_features
        ]
        if kind == "int8_block":
            packed = library.pack_weight_int8_block(
                quantized_2d.astype(np.int8), scales32
            )
        else:
            nibbles = np.where(quantized_2d < 0, quantized_2d + 16, quantized_2d).astype(
                np.uint8
            )
            packed = library.pack_weight_int4_block(nibbles, scales32)
        dequantized = quantized * scales[:, :, None]
        max_abs, mean_abs, max_rel = _errors(
            values, dequantized.reshape(out_features, groups * GROUP_SIZE)[:, :in_features]
        )
        scales_bytes = np.ascontiguousarray(scales32).tobytes()
        report = RepresentationReport(
            name=name,
            kind=kind,
            shape=shape,
            source_dtype=str(array.dtype),
            packed_bytes=int(packed.weight.nbytes + packed.scales.nbytes),
            source_bytes=source_bytes,
            max_abs_error=max_abs,
            mean_abs_error=mean_abs,
            max_relative_error=max_rel,
            group_size=GROUP_SIZE,
            groups=groups,
            zero_point=0,
            saturated_values=saturated,
            scale_min=float(scales32.min()),
            scale_max=float(scales32.max()),
            scale_sha256=hashlib.sha256(scales_bytes).hexdigest(),
            digest=_digest(kind, shape, (packed.weight, packed.scales)),
        )
        return NativeWeight(
            name=name,
            kind=kind,
            in_features=in_features,
            out_features=out_features,
            report=report,
            packed=packed,
        )

    if kind == "low_rank":
        if rank is None:
            raise InvalidOperationParameters(
                f"low_rank representation for '{name}' needs an explicit rank"
            )
        if rank <= 0:
            raise InvalidOperationParameters(
                f"low_rank rank for '{name}' must be positive, got {rank}"
            )
        if rank > min(out_features, in_features):
            raise InvalidOperationParameters(
                f"low_rank rank {rank} for '{name}' exceeds the maximum "
                f"{min(out_features, in_features)} for shape {shape}"
            )
        left, singular, right = np.linalg.svd(values, full_matrices=False)
        u = np.ascontiguousarray(left[:, :rank] * singular[:rank], dtype=np.float32)
        v = np.ascontiguousarray(right[:rank], dtype=np.float32)
        approximation = u.astype(np.float64) @ v.astype(np.float64)
        max_abs, mean_abs, max_rel = _errors(values, approximation)
        total_energy = float(np.sum(singular**2))
        captured = float(np.sum(singular[:rank] ** 2))
        report = RepresentationReport(
            name=name,
            kind=kind,
            shape=shape,
            source_dtype=str(array.dtype),
            packed_bytes=int(u.nbytes + v.nbytes),
            source_bytes=source_bytes,
            max_abs_error=max_abs,
            mean_abs_error=mean_abs,
            max_relative_error=max_rel,
            rank=int(rank),
            spectral_energy=(captured / total_energy if total_energy > 0 else 0.0),
            digest=_digest(kind, shape, (u, v), extra={"rank": int(rank)}),
        )
        return NativeWeight(
            name=name,
            kind=kind,
            in_features=in_features,
            out_features=out_features,
            report=report,
            factors=(u, v),
        )

    if kind == "sparse_csr":
        if threshold is None:
            raise InvalidOperationParameters(
                f"sparse_csr representation for '{name}' needs an explicit threshold"
            )
        keep, indices, row_ptr = _sparse_payload(values, float(threshold))
        dense = np.zeros(shape, dtype=np.float32)
        for o in range(out_features):
            start, stop = int(row_ptr[o]), int(row_ptr[o + 1])
            dense[o, indices[start:stop]] = keep[start:stop]
        max_abs, mean_abs, max_rel = _errors(values, dense.astype(np.float64))
        nonzeros = int(keep.size)
        report = RepresentationReport(
            name=name,
            kind=kind,
            shape=shape,
            source_dtype=str(array.dtype),
            packed_bytes=int(keep.nbytes + indices.nbytes + row_ptr.nbytes),
            source_bytes=source_bytes,
            max_abs_error=max_abs,
            mean_abs_error=mean_abs,
            max_relative_error=max_rel,
            nonzeros=nonzeros,
            sparsity=1.0 - nonzeros / float(shape[0] * shape[1]),
            threshold=float(threshold),
            digest=_digest(
                kind, shape, (keep, indices, row_ptr), extra={"threshold": float(threshold)}
            ),
        )
        return NativeWeight(
            name=name,
            kind=kind,
            in_features=in_features,
            out_features=out_features,
            report=report,
            sparse=(keep, indices, row_ptr),
        )

    raise InvalidOperationParameters(  # pragma: no cover - guarded above
        f"unknown representation '{kind}'"
    )


def prepare_weight(
    name: str,
    weight: Any,
    options: Mapping[str, Any],
    library: NativeLibrary,
) -> NativeWeight:
    """Prepare a weight from normalised representation ``options``."""
    return prepare_representation(
        name,
        weight,
        str(options.get("kind", "dense_blocked")),
        library=library,
        rank=options.get("rank"),
        threshold=(
            float(options["threshold"]) if "threshold" in options else None
        ),
    )


__all__ = [
    "GROUP_SIZE",
    "INT4_MAX",
    "INT8_MAX",
    "NativeWeight",
    "REPRESENTATION_KINDS",
    "REPRESENTATION_NOTES",
    "RepresentationReport",
    "normalize_representations",
    "prepare_representation",
    "prepare_weight",
    "representation_key",
]
