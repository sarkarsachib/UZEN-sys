"""Advanced NumPy runtime: GGUF, sharded safetensors, atomic writes, strict refusal."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from uzen.core.nws.tensors import WeightPackage
from uzen.errors import ArtifactIntegrityError, UnsupportedFormat
from uzen.infrastructure.storage.gguf_io import read_gguf, write_gguf
from uzen.infrastructure.storage.safetensors_io import write_safetensors
from uzen.runtimes.numpy_runtime import NumpyRuntime


def _package() -> WeightPackage:
    return WeightPackage.from_arrays(
        {
            "w": np.arange(12, dtype=np.float32).reshape(3, 4),
            "b": np.linspace(-1.0, 1.0, 4, dtype=np.float32),
        },
        name="tiny",
        family="mlp",
    )


def test_capabilities_advertise_the_extended_formats():
    caps = NumpyRuntime().capabilities()
    assert caps.supports_gguf is True
    assert caps.supports_safetensors is True
    assert caps.supports_autograd is False
    assert "gguf" in caps.extra["load_formats"]
    assert caps.extra["sharded_safetensors"] is True


def test_gguf_round_trip_through_the_runtime(tmp_path: Path):
    runtime = NumpyRuntime()
    package = _package()
    path = runtime.save(package, tmp_path / "tiny.gguf")
    assert Path(path).suffix == ".gguf"

    loaded = runtime.load(path)
    assert loaded.family == "mlp"
    assert loaded.metadata["source_format"] == "gguf"
    for name in package.names():
        np.testing.assert_array_equal(loaded.tensors[name], package.tensors[name])


def test_sharded_index_load_matches_the_source_package(tmp_path: Path):
    package = _package()
    shard_dir = tmp_path / "sharded"
    shard_dir.mkdir()
    names = list(package.names())
    write_safetensors(shard_dir / "model-00001-of-00002.safetensors", {names[0]: package.tensors[names[0]]})
    write_safetensors(shard_dir / "model-00002-of-00002.safetensors", {names[1]: package.tensors[names[1]]})
    (shard_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": int(sum(package.tensors[n].nbytes for n in names))},
                "weight_map": {
                    names[0]: "model-00001-of-00002.safetensors",
                    names[1]: "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = NumpyRuntime().load(shard_dir)
    assert loaded.metadata["source_format"] == "safetensors-sharded"
    assert loaded.metadata["shards"] == 2
    for name in names:
        np.testing.assert_array_equal(loaded.tensors[name], package.tensors[name])


def _write_shard(path: Path, arrays: dict[str, np.ndarray]) -> None:
    write_safetensors(path, arrays)


def test_sharded_index_refuses_a_missing_declared_shard(tmp_path: Path):
    root = tmp_path / "missing"
    root.mkdir()
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "missing.safetensors"}}), encoding="utf-8"
    )
    with pytest.raises(ArtifactIntegrityError, match="missing shard"):
        NumpyRuntime().load(root)


def test_sharded_index_refuses_path_traversal(tmp_path: Path):
    root = tmp_path / "traversal"
    root.mkdir()
    outside = tmp_path / "outside.safetensors"
    _write_shard(outside, {"w": np.zeros((2, 2), dtype=np.float32)})
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "../outside.safetensors"}}), encoding="utf-8"
    )
    with pytest.raises(ArtifactIntegrityError, match="outside"):
        NumpyRuntime().load(root)


def test_sharded_index_refuses_missing_and_extra_tensors(tmp_path: Path):
    root = tmp_path / "mismatch"
    root.mkdir()
    _write_shard(root / "shard.safetensors", {"w": np.zeros((2, 2), np.float32), "extra": np.ones(1, np.float32)})
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "shard.safetensors", "b": "shard.safetensors"}}),
        encoding="utf-8",
    )
    with pytest.raises(ArtifactIntegrityError, match="missing indexed tensors"):
        NumpyRuntime().load(root)


def test_unindexed_shards_refuse_duplicate_tensor_names(tmp_path: Path):
    root = tmp_path / "duplicates"
    root.mkdir()
    _write_shard(root / "a.safetensors", {"w": np.zeros((2, 2), np.float32)})
    _write_shard(root / "b.safetensors", {"w": np.ones((2, 2), np.float32)})
    with pytest.raises(ArtifactIntegrityError, match="more than one unindexed shard"):
        NumpyRuntime().load(root)


def test_single_model_file_inside_a_directory_is_loaded(tmp_path: Path):
    root = tmp_path / "single"
    root.mkdir()
    package = _package()
    write_safetensors(root / "model.safetensors", package.tensors, metadata={"family": "mlp"})
    loaded = NumpyRuntime().load(root)
    assert loaded.family == "mlp"
    for name in package.names():
        np.testing.assert_array_equal(loaded.tensors[name], package.tensors[name])


def test_npz_save_is_atomic_and_round_trips(tmp_path: Path):
    runtime = NumpyRuntime()
    package = _package()
    path = tmp_path / "weights.npz"
    runtime.save(package, path)
    assert path.exists()
    assert not (tmp_path / "weights.npz.tmp").exists()
    loaded = runtime.load(path)
    for name in package.names():
        np.testing.assert_array_equal(loaded.tensors[name], package.tensors[name])


def test_runtime_refuses_unsupported_files(tmp_path: Path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"not a checkpoint")
    with pytest.raises(UnsupportedFormat):
        NumpyRuntime().load(path)


def test_gguf_export_refuses_dtypes_it_cannot_represent(tmp_path: Path):
    package = WeightPackage.from_arrays(
        {"w": np.zeros((2, 2), dtype=np.float64)}, name="double", family="mlp"
    )
    with pytest.raises(UnsupportedFormat):
        NumpyRuntime().save(package, tmp_path / "double.gguf")
