"""Enforced immutability and integrity-boundary tests for NWS."""

from __future__ import annotations

import json

import numpy as np
import pytest

from uzen.core.nir.executor import execute
from uzen.core.nir.graph import Graph
from uzen.core.nir.types import Ref, TensorType
from uzen.core.nws.store import ArtifactStore
from uzen.core.nws.tensors import WeightPackage
from uzen.errors import ArtifactIntegrityError


def _package() -> WeightPackage:
    return WeightPackage.from_arrays(
        {"w": np.arange(4, dtype=np.float32).reshape(2, 2)}, name="immutable"
    )


def test_package_owns_copies_and_arrays_are_read_only():
    source = np.arange(4, dtype=np.float32).reshape(2, 2)
    package = WeightPackage.from_arrays({"w": source})

    source[0, 0] = 99.0
    assert package.get("w")[0, 0] == 0.0

    with pytest.raises(ValueError):
        package.get("w")[0, 0] = 1.0
    with pytest.raises(ValueError):
        package.get("w").setflags(write=True)


def test_package_mappings_and_attributes_are_frozen():
    package = _package()

    with pytest.raises(TypeError):
        package.tensors["other"] = np.zeros(1)  # type: ignore[index]
    with pytest.raises(TypeError):
        package.metas["other"] = package.metas["w"]  # type: ignore[index]
    with pytest.raises(TypeError):
        package.metadata["x"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        package.family_spec["x"] = 1  # type: ignore[index]
    with pytest.raises(AttributeError):
        package.name = "changed"


def test_with_tensor_returns_new_package_and_leaves_source_untouched():
    package = _package()
    before = package.fingerprint()
    derived = package.with_tensor("w", np.full((2, 2), 7.0, dtype=np.float32))

    assert package.get("w")[0, 0] == 0.0
    assert derived.get("w")[0, 0] == 7.0
    assert package.fingerprint() == before
    assert derived.fingerprint() != before
    package.assert_unmodified()
    derived.assert_unmodified()


def test_fingerprint_check_detects_bypassed_mutation():
    package = _package()
    mutated = np.array(package.get("w"), copy=True)
    mutated[0, 0] = 123.0
    object.__setattr__(package, "tensors", {"w": mutated})

    with pytest.raises(ArtifactIntegrityError):
        package.assert_unmodified()


def test_store_put_rejects_tampered_package(tmp_path):
    package = _package()
    object.__setattr__(package, "tensors", {"w": np.zeros((2, 2), dtype=np.float32)})
    store = ArtifactStore(tmp_path / "artifacts")

    with pytest.raises(ArtifactIntegrityError):
        store.put(package, operation="init")


def test_store_get_rejects_payload_tampering(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    record = store.put(_package(), operation="init")
    payload = record.path / "tensors.safetensors"
    raw = bytearray(payload.read_bytes())
    raw[-1] ^= 0xFF
    payload.write_bytes(bytes(raw))

    with pytest.raises(ArtifactIntegrityError):
        store.get(record.artifact_id)


def test_store_get_rejects_manifest_tampering(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    record = store.put(_package(), operation="init")
    manifest_path = record.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["metadata"]["tampered"] = True
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ArtifactIntegrityError):
        store.get(record.artifact_id)
    result = store.verify(record.artifact_id)
    assert result["ok"] is False
    assert result["manifest_checksum_ok"] is False


def test_store_export_refuses_tampered_artifact(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    record = store.put(_package(), operation="init")
    payload = record.path / "tensors.safetensors"
    raw = bytearray(payload.read_bytes())
    raw[-1] ^= 0xFF
    payload.write_bytes(bytes(raw))

    with pytest.raises(ArtifactIntegrityError):
        store.export(record.artifact_id, tmp_path / "export.safetensors")


def test_executor_rejects_tampered_parameter_package():
    package = _package()
    object.__setattr__(package, "tensors", {"w": np.zeros((2, 2), dtype=np.float32)})
    graph = Graph(name="identity", inputs={"x": TensorType.of((2,))})
    graph.add_node("id", "identity", {"x": Ref("input:x")})
    graph.set_output("y", Ref("id"))

    with pytest.raises(ArtifactIntegrityError):
        execute(graph, {"x": np.zeros(2, dtype=np.float32)}, package)
