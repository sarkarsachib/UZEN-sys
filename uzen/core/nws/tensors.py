"""Immutable neural parameter packages.

A :class:`WeightPackage` is the in-memory carrier of a complete parameter set:
named NumPy tensors plus per-tensor metadata (role, origin, provenance). The
substrate deliberately does not assume a transformer, a specific naming scheme,
or even that tensors form a known architecture — families register their own
binding plans elsewhere.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from uzen.core.nws.dtypes import dtype_name, ensure_array
from uzen.core.nws.roles import ParameterRole, infer_role
from uzen.errors import ArtifactIntegrityError, ShapeMismatch, TensorInvalid
from uzen.policies.strict_plus.effects import Origin
from uzen.util.canonical import ContentHasher, canonical_bytes, sha256_bytes
from uzen.util.linalg import tensor_statistics


@dataclass(frozen=True)
class TensorMeta:
    name: str
    shape: tuple[int, ...]
    dtype: str
    role: ParameterRole = ParameterRole.UNCLASSIFIED
    origin: Origin = Origin.INHERITED
    source_artifact_id: str | None = None
    source_dtype: str | None = None
    ops: tuple[str, ...] = ()
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "role": self.role.value,
            "origin": self.origin.value,
            "ops": list(self.ops),
        }
        if self.source_artifact_id:
            out["source_artifact_id"] = self.source_artifact_id
        if self.source_dtype and self.source_dtype != self.dtype:
            out["source_dtype"] = self.source_dtype
        if self.note:
            out["note"] = self.note
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TensorMeta":
        return cls(
            name=str(data["name"]),
            shape=tuple(int(d) for d in data["shape"]),
            dtype=str(data["dtype"]),
            role=ParameterRole(data.get("role", "unclassified")),
            origin=Origin(data.get("origin", "inherited")),
            source_artifact_id=data.get("source_artifact_id"),
            source_dtype=data.get("source_dtype"),
            ops=tuple(data.get("ops", ())),
            note=str(data.get("note", "")),
        )


def _immutable_array(array: np.ndarray) -> np.ndarray:
    """Copy an array into a read-only buffer that cannot be re-enabled for writing.

    A plain ``setflags(write=False)`` is only advisory: the owner of the array can
    set it back to writeable. Backing the array with ``bytes`` makes NumPy reject
    ``setflags(write=True)`` because the buffer itself is immutable.
    """
    arr = array if array.ndim == 0 else np.ascontiguousarray(array)
    raw = arr.tobytes(order="C") if arr.size else b""
    frozen = np.frombuffer(raw, dtype=arr.dtype)
    return frozen.reshape(arr.shape)


def _tensor_fingerprint(tensors: Mapping[str, np.ndarray]) -> str:
    from uzen.infrastructure.storage.safetensors_io import safetensors_tensor_digest

    return safetensors_tensor_digest(tensors)


@dataclass
class WeightPackage:
    """A named, self-describing, immutable set of parameter tensors.

    Immutability is enforced, not merely documented:

    * tensor buffers are copied into read-only byte-backed arrays, so a caller
      cannot write through an array it supplied (and cannot re-enable writes on
      the package's arrays);
    * ``tensors``/``metas``/``family_spec``/``metadata`` are read-only mappings;
    * attribute assignment after construction is rejected;
    * the value fingerprint is captured at construction and re-checked at
      artifact and execution boundaries by :meth:`assert_unmodified`.

    Derive new packages with :meth:`with_tensor`, :meth:`without`,
    :meth:`select` or :meth:`copy` — never by mutating an existing one.
    """

    tensors: Mapping[str, np.ndarray]
    metas: Mapping[str, TensorMeta]
    name: str = "weights"
    family: str = "generic"
    family_spec: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source_artifact_id: str | None = None
    _fingerprint: str = field(default="", init=False, repr=False, compare=False)
    _sealed: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.tensors, Mapping):
            raise TypeError("tensors must be a mapping of name -> numpy array")
        if not isinstance(self.metas, Mapping):
            raise TypeError("metas must be a mapping of name -> TensorMeta")
        tensors: dict[str, np.ndarray] = {}
        metas: dict[str, TensorMeta] = {}
        for key, value in self.tensors.items():
            arr = _immutable_array(ensure_array(value, name=key))
            tensors[key] = arr
            meta = self.metas.get(key)
            if meta is None:
                meta = TensorMeta(
                    name=key,
                    shape=tuple(int(d) for d in arr.shape),
                    dtype=dtype_name(arr.dtype),
                    role=infer_role(key, arr.shape),
                    origin=Origin.INHERITED,
                    source_artifact_id=self.source_artifact_id,
                )
            elif not isinstance(meta, TensorMeta):
                raise TypeError(f"metadata for '{key}' must be a TensorMeta")
            metas[key] = meta
        extra = set(self.metas) - set(tensors)
        if extra:
            raise TensorInvalid(f"metadata without tensor data: {sorted(extra)}")
        object.__setattr__(self, "tensors", MappingProxyType(tensors))
        object.__setattr__(self, "metas", MappingProxyType(metas))
        object.__setattr__(self, "family_spec", MappingProxyType(dict(self.family_spec)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "_fingerprint", _tensor_fingerprint(tensors))
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError(
                f"WeightPackage is immutable; cannot assign '{name}'. "
                "Derive a new package with with_tensor(), without(), select() or copy()."
            )
        object.__setattr__(self, name, value)

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_arrays(
        cls,
        arrays: Mapping[str, np.ndarray],
        *,
        name: str = "weights",
        family: str = "generic",
        origin: Origin = Origin.INHERITED,
        source_artifact_id: str | None = None,
        family_spec: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        ops: Sequence[str] = (),
        source_dtypes: Mapping[str, str] | None = None,
    ) -> "WeightPackage":
        tensors: dict[str, np.ndarray] = {}
        metas: dict[str, TensorMeta] = {}
        declared_sources = dict(source_dtypes or {})
        for key, value in arrays.items():
            arr = ensure_array(value, name=key)
            tensors[key] = arr
            source_dtype = declared_sources.get(key)
            metas[key] = TensorMeta(
                name=key,
                shape=tuple(int(d) for d in arr.shape),
                dtype=dtype_name(arr.dtype),
                role=infer_role(key, arr.shape),
                origin=origin,
                source_artifact_id=source_artifact_id,
                source_dtype=source_dtype,
                ops=tuple(ops),
                note=(
                    "bfloat16 widened losslessly to float32"
                    if source_dtype == "bfloat16"
                    else ""
                ),
            )
        return cls(
            tensors=tensors,
            metas=metas,
            name=name,
            family=family,
            family_spec=dict(family_spec or {}),
            metadata=dict(metadata or {}),
            source_artifact_id=source_artifact_id,
        )

    def copy(self) -> "WeightPackage":
        return WeightPackage(
            tensors={k: np.array(v, copy=True) for k, v in self.tensors.items()},
            metas={k: replace(m) for k, m in self.metas.items()},
            name=self.name,
            family=self.family,
            family_spec=dict(self.family_spec),
            metadata=dict(self.metadata),
            source_artifact_id=self.source_artifact_id,
        )

    def with_tensor(
        self,
        name: str,
        array: np.ndarray,
        *,
        origin: Origin | None = None,
        ops: Sequence[str] = (),
        note: str = "",
    ) -> "WeightPackage":
        """Return a new package with one tensor replaced (original untouched)."""
        arr = ensure_array(array, name=name)
        tensors = dict(self.tensors)
        metas = dict(self.metas)
        tensors[name] = arr
        prior = metas.get(name)
        metas[name] = TensorMeta(
            name=name,
            shape=tuple(int(d) for d in arr.shape),
            dtype=dtype_name(arr.dtype),
            role=prior.role if prior else infer_role(name, arr.shape),
            origin=origin or (prior.origin if prior else Origin.TRANSFORMED),
            source_artifact_id=self.source_artifact_id,
            ops=tuple(ops) or (prior.ops if prior else ()),
            note=note or (prior.note if prior else ""),
        )
        return WeightPackage(
            tensors=tensors,
            metas=metas,
            name=self.name,
            family=self.family,
            family_spec=dict(self.family_spec),
            metadata=dict(self.metadata),
            source_artifact_id=self.source_artifact_id,
        )

    def without(self, names: Iterable[str]) -> "WeightPackage":
        drop = set(names)
        return WeightPackage(
            tensors={k: v for k, v in self.tensors.items() if k not in drop},
            metas={k: v for k, v in self.metas.items() if k not in drop},
            name=self.name,
            family=self.family,
            family_spec=dict(self.family_spec),
            metadata=dict(self.metadata),
            source_artifact_id=self.source_artifact_id,
        )

    # ------------------------------------------------------------------ #
    # access
    # ------------------------------------------------------------------ #

    def names(self) -> list[str]:
        return sorted(self.tensors)

    def __len__(self) -> int:
        return len(self.tensors)

    def __contains__(self, name: object) -> bool:
        return name in self.tensors

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def get(self, name: str) -> np.ndarray:
        try:
            return self.tensors[name]
        except KeyError as exc:
            raise TensorInvalid(f"tensor '{name}' not present in package '{self.name}'") from exc

    def select(self, *patterns: str, regex: bool = False) -> "WeightPackage":
        """Select tensors by glob (default) or regular-expression patterns."""
        matched: list[str] = []
        for name in self.names():
            for pat in patterns:
                if regex:
                    if re.search(pat, name):
                        matched.append(name)
                        break
                elif fnmatch.fnmatch(name, pat):
                    matched.append(name)
                    break
        keep = set(matched)
        return WeightPackage(
            tensors={k: v for k, v in self.tensors.items() if k in keep},
            metas={k: v for k, v in self.metas.items() if k in keep},
            name=f"{self.name}[selected]",
            family=self.family,
            family_spec=dict(self.family_spec),
            metadata={**self.metadata, "selection": list(patterns)},
            source_artifact_id=self.source_artifact_id,
        )

    def by_role(self, *roles: ParameterRole) -> list[str]:
        wanted = set(roles)
        return [n for n in self.names() if self.metas[n].role in wanted]

    def shape_of(self, name: str) -> tuple[int, ...]:
        return self.metas[name].shape

    def numel(self) -> int:
        return int(sum(v.size for v in self.tensors.values()))

    def nbytes(self) -> int:
        return int(sum(v.nbytes for v in self.tensors.values()))

    def dtypes(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for meta in self.metas.values():
            counts[meta.dtype] = counts.get(meta.dtype, 0) + 1
        return counts

    # ------------------------------------------------------------------ #
    # integrity
    # ------------------------------------------------------------------ #

    def issues(self, *, require_finite: bool = True) -> list[str]:
        problems: list[str] = []
        for name in self.names():
            arr = self.tensors[name]
            meta = self.metas[name]
            if tuple(arr.shape) != meta.shape:
                problems.append(f"{name}: array shape {arr.shape} != metadata {meta.shape}")
            if dtype_name(arr.dtype) != meta.dtype and not (
                meta.source_dtype == "bfloat16" and meta.dtype == "float32"
            ):
                problems.append(f"{name}: array dtype {arr.dtype} != metadata {meta.dtype}")
            if require_finite and arr.dtype.kind in "fc" and not np.isfinite(arr).all():
                problems.append(f"{name}: contains non-finite values")
        return problems

    def validate(self, *, require_finite: bool = True) -> None:
        problems = self.issues(require_finite=require_finite)
        if problems:
            raise TensorInvalid("package failed validation", problems=problems[:50])

    def fingerprint(self) -> str:
        return self._fingerprint or _tensor_fingerprint(self.tensors)

    def assert_unmodified(self) -> None:
        """Re-hash the live tensors and fail if they differ from construction time."""
        actual = _tensor_fingerprint(self.tensors)
        if actual != self._fingerprint:
            raise ArtifactIntegrityError(
                "weight package changed after construction; source weights must remain "
                "immutable and transformations must produce a new package",
                expected_fingerprint=self._fingerprint,
                actual_fingerprint=actual,
            )

    def structure_digest(self) -> str:
        """Digest of names/shapes/dtypes only (architecture-level identity)."""
        h = ContentHasher()
        for name in self.names():
            meta = self.metas[name]
            h.field(f"t:{name}", f"{meta.dtype}|{list(meta.shape)}")
        return h.hexdigest()

    # ------------------------------------------------------------------ #
    # reporting / serialization
    # ------------------------------------------------------------------ #

    def summary(self, *, stats: bool = False) -> dict[str, Any]:
        roles: dict[str, int] = {}
        origins: dict[str, int] = {}
        for meta in self.metas.values():
            roles[meta.role.value] = roles.get(meta.role.value, 0) + 1
            origins[meta.origin.value] = origins.get(meta.origin.value, 0) + 1
        out: dict[str, Any] = {
            "name": self.name,
            "family": self.family,
            "num_tensors": len(self.tensors),
            "numel": self.numel(),
            "nbytes": self.nbytes(),
            "dtypes": self.dtypes(),
            "roles": roles,
            "origins": origins,
            "fingerprint": self.fingerprint(),
            "structure_digest": self.structure_digest(),
            "source_artifact_id": self.source_artifact_id,
        }
        if stats:
            out["tensors"] = {
                name: {**tensor_statistics(self.tensors[name]), "role": self.metas[name].role.value}
                for name in self.names()
            }
        return out

    def tensors_meta_list(self) -> list[dict[str, Any]]:
        return [self.metas[n].to_dict() for n in self.names()]

    def assert_shape(self, name: str, expected: Sequence[int]) -> None:
        actual = self.shape_of(name) if name in self.metas else tuple(self.tensors[name].shape)
        if tuple(expected) != tuple(actual):
            raise ShapeMismatch(
                f"tensor '{name}' has shape {actual}, expected {tuple(expected)}"
            )


def package_digest(pkg: WeightPackage) -> str:
    return sha256_bytes(canonical_bytes(pkg.fingerprint()))
