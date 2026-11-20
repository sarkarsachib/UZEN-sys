"""NumPy reference runtime.

This is the always-available execution backend. It loads and saves safetensors,
NPZ, GGUF and sharded-safetensors checkpoints, executes NIR graphs through the
typed operator kernels, and never enables autograd (there is none).

Loading is strict by design:

* a sharded checkpoint is read through ``model.safetensors.index.json``; every
  declared shard must exist, every indexed tensor must be present in the shard
  that declares it, and no shard may smuggle in an unindexed tensor;
* shard paths are resolved inside the checkpoint directory (path traversal is
  refused);
* duplicate tensor names across unindexed shards are refused instead of silently
  picking one;
* GGUF tensors are dequantized by the existing codec registry, and types without
  a decoder raise a typed ``UnsupportedFormat`` instead of returning garbage.

Saving is atomic: buffers are written to a sibling temporary file and renamed only
after the payload is complete.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from uzen.core.nir.executor import execute
from uzen.core.nir.graph import Graph
from uzen.core.nws.tensors import WeightPackage
from uzen.errors import ArtifactIntegrityError, UnsupportedFormat
from uzen.infrastructure.storage.gguf_io import read_gguf, write_gguf
from uzen.infrastructure.storage.safetensors_io import read_safetensors, write_safetensors
from uzen.runtimes.base import RuntimeCapabilities


class NumpyRuntime:
    name = "numpy"

    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            name=self.name,
            backend="numpy",
            device="cpu",
            dtypes=("float16", "float32", "float64", "int32", "int64", "uint8", "bool"),
            supports_autograd=False,
            supports_safetensors=True,
            supports_torch_checkpoints=False,
            supports_onnx=False,
            supports_gguf=True,
            extra={
                "numpy_version": np.__version__,
                "execution": "typed NIR interpreter",
                "load_formats": ["safetensors", "safetensors-sharded", "npz", "gguf"],
                "save_formats": ["safetensors", "npz", "gguf"],
                "sharded_safetensors": True,
                "gguf_dequantization": True,
                "atomic_writes": True,
                "note": "reference runtime; deterministic closed-form kernels only",
            },
        )

    # ------------------------------------------------------------------ #
    # loading
    # ------------------------------------------------------------------ #

    def load(self, path: str | os.PathLike[str]) -> WeightPackage:
        p = Path(path)
        if p.is_dir():
            return self._load_directory(p)
        suffix = p.suffix.lower()
        if suffix == ".safetensors" or suffix == "":
            arrays, header = read_safetensors(p)
            return WeightPackage.from_arrays(
                arrays,
                name=p.stem,
                family=header.metadata.get("family", "imported"),
                metadata={
                    "source_path": str(p),
                    "source_format": "safetensors",
                    "source_tensor_count": len(arrays),
                },
            )
        if suffix == ".npz":
            data = np.load(p, allow_pickle=False)
            arrays = {k: np.asarray(data[k]) for k in data.files}
            return WeightPackage.from_arrays(
                arrays,
                name=p.stem,
                family="imported",
                metadata={
                    "source_path": str(p),
                    "source_format": "npz",
                    "source_tensor_count": len(arrays),
                },
            )
        if suffix == ".gguf":
            arrays, gguf_metadata = read_gguf(p)
            family = str(gguf_metadata.get("general.architecture") or "imported")
            return WeightPackage.from_arrays(
                arrays,
                name=p.stem,
                family=family,
                metadata={
                    "source_path": str(p),
                    "source_format": "gguf",
                    "source_tensor_count": len(arrays),
                    "gguf_metadata": dict(gguf_metadata),
                },
            )
        raise UnsupportedFormat(
            f"the NumPy runtime cannot read '{p.name}'; supported: .safetensors, "
            "sharded .safetensors directory, .npz, .gguf"
        )

    def _load_directory(self, root: Path) -> WeightPackage:
        index_path = root / "model.safetensors.index.json"
        if index_path.exists():
            arrays, details = self._read_sharded(root, index_path)
            return WeightPackage.from_arrays(
                arrays,
                name=root.name,
                family="imported",
                metadata={
                    "source_path": str(root),
                    "source_format": "safetensors-sharded",
                    **details,
                },
            )
        single = root / "model.safetensors"
        if single.exists():
            arrays, header = read_safetensors(single)
            return WeightPackage.from_arrays(
                arrays,
                name=root.name,
                family=header.metadata.get("family", "imported"),
                metadata={
                    "source_path": str(root),
                    "source_format": "safetensors",
                    "source_tensor_count": len(arrays),
                },
            )
        shards = sorted(root.glob("*.safetensors"))
        if not shards:
            raise UnsupportedFormat(
                f"directory '{root}' contains no safetensors payload or index"
            )
        arrays, details = self._read_unindexed(shards)
        return WeightPackage.from_arrays(
            arrays,
            name=root.name,
            family="imported",
            metadata={
                "source_path": str(root),
                "source_format": "safetensors-sharded",
                **details,
            },
        )

    def _read_sharded(
        self, root: Path, index_path: Path
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        try:
            document = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError(
                f"cannot read shard index '{index_path}': {exc}"
            ) from exc
        weight_map = document.get("weight_map") if isinstance(document, Mapping) else None
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise ArtifactIntegrityError(
                "shard index must contain a non-empty 'weight_map' object"
            )
        root_resolved = root.resolve()
        by_shard: dict[str, set[str]] = {}
        for tensor_name, shard_name in weight_map.items():
            if not isinstance(tensor_name, str) or not isinstance(shard_name, str):
                raise ArtifactIntegrityError(
                    "weight_map entries must map tensor names to shard filenames"
                )
            shard_path = (root / shard_name).resolve()
            if root_resolved not in shard_path.parents:
                raise ArtifactIntegrityError(
                    f"shard '{shard_name}' resolves outside the checkpoint directory"
                )
            by_shard.setdefault(shard_name, set()).add(tensor_name)
        arrays: dict[str, np.ndarray] = {}
        for shard_name in sorted(by_shard):
            shard_path = root / shard_name
            if not shard_path.exists():
                raise ArtifactIntegrityError(
                    f"missing shard '{shard_name}' declared in the weight map"
                )
            shard_arrays, _ = read_safetensors(shard_path)
            expected = by_shard[shard_name]
            missing = expected - set(shard_arrays)
            extra = set(shard_arrays) - expected
            if missing:
                raise ArtifactIntegrityError(
                    f"shard '{shard_name}' is missing indexed tensors: {sorted(missing)[:8]}"
                )
            if extra:
                raise ArtifactIntegrityError(
                    f"shard '{shard_name}' contains unindexed tensors: {sorted(extra)[:8]}"
                )
            arrays.update(shard_arrays)
        if set(arrays) != set(weight_map):
            raise ArtifactIntegrityError(
                "loaded tensors do not match the index weight_map"
            )
        return arrays, {
            "index": index_path.name,
            "shards": len(by_shard),
            "weight_map_entries": len(weight_map),
            "source_tensor_count": len(arrays),
        }

    def _read_unindexed(
        self, shards: Sequence[Path]
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        arrays: dict[str, np.ndarray] = {}
        for shard in shards:
            shard_arrays, _ = read_safetensors(shard)
            shared = set(arrays) & set(shard_arrays)
            if shared:
                raise ArtifactIntegrityError(
                    f"tensor names occur in more than one unindexed shard: "
                    f"{sorted(shared)[:8]}"
                )
            arrays.update(shard_arrays)
        return arrays, {
            "shards": len(shards),
            "shard_files": [shard.name for shard in shards],
            "weight_map_entries": len(arrays),
            "source_tensor_count": len(arrays),
        }

    # ------------------------------------------------------------------ #
    # saving
    # ------------------------------------------------------------------ #

    def save(
        self,
        package: WeightPackage,
        path: str | os.PathLike[str],
        *,
        quantize: str | None = None,
    ) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        suffix = p.suffix.lower()
        if suffix == ".npz":
            temporary = p.with_name(p.name + ".tmp")
            try:
                with temporary.open("wb") as handle:
                    np.savez(handle, **package.tensors)
                os.replace(temporary, p)
            finally:
                temporary.unlink(missing_ok=True)
            return str(p)
        if suffix == ".gguf":
            write_gguf(
                p,
                package.tensors,
                {
                    "general.architecture": package.family,
                    "uzen.family": package.family,
                    "uzen.name": package.name,
                },
                quantize=quantize,
            )
            return str(p)
        write_safetensors(
            p,
            package.tensors,
            metadata={
                "family": package.family,
                "name": package.name,
                "format": "uzen",
            },
            atomic=True,
        )
        return str(p)

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
    ) -> dict[str, np.ndarray]:
        result = execute(graph, inputs, package, capture=capture)
        return dict(result.outputs)

    def execute_with_intermediates(
        self,
        graph: Graph,
        inputs: Mapping[str, np.ndarray],
        package: WeightPackage,
        *,
        capture: Sequence[str],
    ) -> dict[str, np.ndarray]:
        return execute(graph, inputs, package, capture=capture).intermediates

    def describe(self) -> dict[str, Any]:
        return self.capabilities().to_dict()
