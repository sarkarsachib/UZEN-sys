"""Weight-only int8 execution and quantization-report coverage."""

from __future__ import annotations

import json

import numpy as np
import pytest

from uzen.core.nir.executor import execute
from uzen.core.nir.graph import Graph
from uzen.core.nir.types import Ref, TensorType
from uzen.core.nws.tensors import WeightPackage
from uzen.core.wie.importer import ModelImporter
from uzen.core.wie.quantization import quantize_int8
from uzen.errors import InvalidOperationParameters
from uzen.interfaces.python_sdk.client import Uzen
from uzen.runtimes.torch_executor import execute_torch
from tests.test_llama_reconstruction import _write_llama


def _mlp_graph() -> tuple[Graph, WeightPackage]:
    rng = np.random.default_rng(41)
    graph = Graph(name="int8-mlp", inputs={"x": TensorType.of((1, 4))})
    graph.add_node(
        "fc1",
        "linear",
        {"x": Ref("input:x")},
        {"weight": "param:fc1.weight", "bias": "param:fc1.bias"},
    )
    graph.add_node("act", "relu", {"x": Ref("fc1")})
    graph.add_node(
        "fc2",
        "linear",
        {"x": Ref("act")},
        {"weight": "param:fc2.weight"},
    )
    graph.set_output("y", Ref("fc2"))
    package = WeightPackage.from_arrays(
        {
            "fc1.weight": rng.standard_normal((6, 4)).astype(np.float32),
            "fc1.bias": rng.standard_normal((6,)).astype(np.float32),
            "fc2.weight": rng.standard_normal((3, 6)).astype(np.float32),
        },
        name="int8-source",
        family="mlp",
    )
    return graph, package


def test_int8_quantization_makes_a_valid_quantized_graph_and_report():
    graph, package = _mlp_graph()
    source_weight = package.get("fc1.weight").copy()
    result = quantize_int8(graph, package)

    assert result.report.graph_valid is True
    assert result.report.num_linear_nodes == 2
    assert result.report.num_rewritten_nodes == 2
    assert len(result.report.tensors) == 2
    assert result.report.original_bytes == 72 + 96
    assert result.report.quantized_bytes < result.report.original_bytes
    assert result.report.compression_ratio > 1
    assert result.report.package_original_bytes > result.report.original_bytes
    assert result.report.package_quantized_bytes < result.report.package_original_bytes
    assert result.report.package_compression_ratio > 1
    assert result.report.max_abs_error > 0
    assert result.report.mean_abs_error > 0
    assert result.package.get("fc1.weight").dtype == np.int8
    assert result.package.get("fc1.weight.int8_scale").dtype == np.float32
    assert result.package.get("fc1.bias").dtype == np.float32
    assert result.graph.nodes["fc1"].op == "quantized_linear"
    assert result.graph.nodes["fc2"].op == "quantized_linear"
    assert result.graph.nodes["fc1"].params["quantization"] == "int8_symmetric_per_output_channel"
    assert result.graph.metadata["quantization"]["schema"] == "uzen.quantization/1"
    assert result.package.metadata["quantization"]["num_rewritten_nodes"] == 2

    # The source package is immutable and remains float32.
    assert package.get("fc1.weight").dtype == np.float32
    np.testing.assert_array_equal(package.get("fc1.weight"), source_weight)

    item = next(item for item in result.report.tensors if item.name == "fc1.weight")
    assert item.scale_count == 6
    assert item.scale_min > 0
    assert item.scale_max >= item.scale_min
    assert len(item.scale_sha256) == 64
    assert item.zero_point == 0
    assert item.axis == 1
    assert item.bits_per_weight > 8  # includes the per-output-channel scales
    json.dumps(result.report.to_dict())


def test_int8_execution_tracks_the_float_graph_and_is_deterministic():
    graph, package = _mlp_graph()
    result = quantize_int8(graph, package)
    inputs = {"x": np.array([[0.2, -0.4, 0.7, 1.1]], dtype=np.float32)}

    reference = execute(graph, inputs, package).output("y")
    first = result.execute(inputs).output("y")
    second = result.execute(inputs).output("y")
    np.testing.assert_array_equal(first, second)
    assert first.shape == reference.shape
    error = float(np.max(np.abs(first - reference)))
    assert error < 0.1
    assert error <= result.report.max_abs_error + 0.05


def test_int8_torch_execution_matches_numpy_quantized_execution():
    graph, package = _mlp_graph()
    result = quantize_int8(graph, package)
    inputs = {"x": np.array([[0.2, -0.4, 0.7, 1.1]], dtype=np.float32)}
    numpy_output = result.execute(inputs).output("y")
    torch_output = execute_torch(result.graph, inputs, result.package).outputs["y"]
    np.testing.assert_allclose(torch_output, numpy_output, rtol=1e-6, atol=1e-7)


def test_int8_zero_rows_get_a_finite_scale_and_no_error():
    graph, package = _mlp_graph()
    arrays = dict(package.tensors)
    weights = np.array(arrays["fc1.weight"], copy=True)
    weights[2] = 0.0
    arrays["fc1.weight"] = weights
    package = WeightPackage.from_arrays(arrays, name="zero-row")
    result = quantize_int8(graph, package)
    scales = result.package.get("fc1.weight.int8_scale")
    assert np.isfinite(scales).all()
    assert (scales > 0).all()
    assert np.all(result.package.get("fc1.weight")[2] == 0)
    item = next(item for item in result.report.tensors if item.name == "fc1.weight")
    assert item.scale_min > 0


def test_int8_quantized_llama_execution_is_near_the_float_reference(tmp_path):
    root, _arrays = _write_llama(tmp_path / "int8-llama")
    imported = ModelImporter().import_source(root)
    assert imported.reconstruction is not None
    graph = imported.reconstruction.definition.graph
    package = imported.artifact_package
    assert package is not None
    quantized = quantize_int8(graph, package)
    assert quantized.report.num_rewritten_nodes > 0
    assert quantized.report.compression_ratio > 1

    inputs = {
        "tokens": np.array([[0, 1, 2, 3]], dtype=np.int64),
        "positions": np.arange(4, dtype=np.int64).reshape(1, 4),
    }
    reference = execute(graph, inputs, package).output("logits")
    actual = quantized.execute(inputs).output("logits")
    assert np.isfinite(actual).all()
    assert float(np.max(np.abs(actual - reference))) < 0.2


def test_int8_forward_is_exposed_through_sdk_and_rest(tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from uzen.interfaces.api.app import create_app

    root, _arrays = _write_llama(tmp_path / "int8-api-llama")
    client = Uzen(root=tmp_path / "int8-api-workspace")
    artifact_id = client.import_model(root)["artifact_id"]
    tokens = [[0, 1, 2, 3]]
    positions = [[0, 1, 2, 3]]

    plain = client.forward_model(
        artifact_id, tokens=tokens, positions=positions, include_values=True
    )
    assert plain["quantization"] is None

    quantized = client.forward_model(
        artifact_id,
        tokens=tokens,
        positions=positions,
        include_values=True,
        quantize="int8",
    )
    report = quantized["quantization"]
    assert report["schema"] == "uzen.quantization/1"
    assert report["num_rewritten_nodes"] > 0
    assert report["graph_valid"] is True
    assert report["tensors"]
    assert quantized["outputs"]["logits"]["shape"] == [1, 4, 32]
    plain_values = np.asarray(plain["outputs"]["logits"]["values"])
    quantized_values = np.asarray(quantized["outputs"]["logits"]["values"])
    assert quantized_values.shape == plain_values.shape
    assert float(np.max(np.abs(quantized_values - plain_values))) < 0.2

    service_execution = client.service.forward_model(
        artifact_id, tokens=tokens, positions=positions, quantize="int8"
    )
    assert service_execution.quantization == report

    api = fastapi_testclient.TestClient(create_app(service=client.service))
    response = api.post(
        f"/v1/models/{artifact_id}/forward",
        json={
            "tokens": tokens,
            "positions": positions,
            "include_values": True,
            "quantize": "int8",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["quantization"]["schema"] == "uzen.quantization/1"
    assert body["quantization"]["num_rewritten_nodes"] == report["num_rewritten_nodes"]
    np.testing.assert_allclose(
        np.asarray(body["outputs"]["logits"]["values"]), quantized_values, rtol=1e-7, atol=1e-7
    )

    with pytest.raises(InvalidOperationParameters, match="quantize must be 'int8'"):
        client.forward_model(artifact_id, tokens=tokens, quantize="int4")


def test_int8_variant_is_persisted_and_reusable(tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from uzen.interfaces.api.app import create_app

    root, _arrays = _write_llama(tmp_path / "int8-variant-llama")
    client = Uzen(root=tmp_path / "int8-variant-workspace")
    source_id = client.import_model(root)["artifact_id"]

    variant = client.quantize_model(source_id)
    variant_id = variant["artifact_id"]
    assert variant_id != source_id
    assert variant["parent_artifact_id"] == source_id
    assert variant["quantization"]["schema"] == "uzen.quantization/1"
    assert variant["quantization"]["num_rewritten_nodes"] > 0

    manifest = client.artifact(variant_id)
    assert manifest["lineage"]["parents"] == [source_id]
    assert manifest["lineage"]["operation"] == "quantize.int8"
    assert manifest["metadata"]["quantization"]["schema"] == "uzen.quantization/1"
    assert manifest["metadata"]["quantized_from"] == source_id
    assert any(
        node["op"] == "quantized_linear"
        for node in manifest["metadata"]["graph"]["nodes"]
    )

    repeated = client.quantize_model(source_id)
    assert repeated["artifact_id"] == variant_id
    with pytest.raises(InvalidOperationParameters, match="already contains a quantized graph"):
        client.quantize_model(variant_id)
    with pytest.raises(InvalidOperationParameters, match="scheme must be 'int8'"):
        client.quantize_model(source_id, scheme="int4")

    tokens = [[0, 1, 2, 3]]
    positions = [[0, 1, 2, 3]]
    source_execution = client.forward_model(
        source_id, tokens=tokens, positions=positions, include_values=True
    )
    variant_execution = client.forward_model(
        variant_id, tokens=tokens, positions=positions, include_values=True
    )
    assert variant_execution["quantization"] == variant["quantization"]
    assert variant_execution["outputs"]["logits"]["shape"] == [1, 4, 32]
    source_values = np.asarray(source_execution["outputs"]["logits"]["values"])
    variant_values = np.asarray(variant_execution["outputs"]["logits"]["values"])
    assert float(np.max(np.abs(variant_values - source_values))) < 0.2

    reopened = Uzen(root=client.service.root)
    reopened_execution = reopened.forward_model(
        variant_id, tokens=tokens, positions=positions, include_values=True
    )
    assert reopened_execution["quantization"] == variant["quantization"]
    np.testing.assert_allclose(
        np.asarray(reopened_execution["outputs"]["logits"]["values"]),
        variant_values,
    )

    api = fastapi_testclient.TestClient(create_app(service=client.service))
    created = api.post(f"/v1/models/{source_id}/quantize", json={})
    assert created.status_code == 200, created.text
    assert created.json()["artifact_id"] == variant_id
    forwarded = api.post(
        f"/v1/models/{variant_id}/forward",
        json={"tokens": tokens, "positions": positions, "include_values": True},
    )
    assert forwarded.status_code == 200, forwarded.text
    assert forwarded.json()["quantization"] == variant["quantization"]


def test_quantized_comparison_reports_output_drift(tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from uzen.interfaces.api.app import create_app

    root, _arrays = _write_llama(tmp_path / "int8-drift-llama")
    client = Uzen(root=tmp_path / "int8-drift-workspace")
    source_id = client.import_model(root)["artifact_id"]
    variant = client.quantize_model(source_id)
    variant_id = variant["artifact_id"]
    tokens = [[0, 1, 2, 3]]
    positions = [[0, 1, 2, 3]]

    comparison = client.compare_quantized_model(
        source_id,
        variant_id,
        tokens=tokens,
        positions=positions,
        atol=0.2,
        rtol=0.2,
    )
    assert comparison["source_vs_quantized"] is True
    assert comparison["quantization"] == variant["quantization"]
    assert comparison["passed"] is True
    logits = comparison["outputs"]["logits"]
    assert logits["shape"] == [1, 4, 32]
    assert logits["max_abs_error"] > 0
    assert logits["max_abs_error"] < 0.2
    assert logits["num_elements"] == 128
    drift = comparison["drift"]
    numeric_outputs = [
        output
        for output in comparison["outputs"].values()
        if "max_abs_error" in output
    ]
    assert drift["complete"] is True
    assert drift["allclose_passed"] is True
    assert drift["failed_outputs"] == []
    assert drift["passed"] is True
    assert drift["max_abs_error"] == max(
        output["max_abs_error"] for output in numeric_outputs
    )
    assert drift["mean_abs_error"] == pytest.approx(
        sum(
            output["mean_abs_error"] * output["num_elements"]
            for output in numeric_outputs
        )
        / sum(output["num_elements"] for output in numeric_outputs)
    )
    assert drift["max_rel_error"] == max(
        output["max_rel_error"] for output in numeric_outputs
    )
    assert drift["num_outputs"] == len(comparison["outputs"])
    assert drift["num_elements"] == sum(
        output["num_elements"] for output in numeric_outputs
    )
    assert drift["thresholds"] == {
        "max_abs_error": None,
        "mean_abs_error": None,
        "max_rel_error": None,
    }
    assert drift["failures"] == []

    thresholded = client.compare_quantized_model(
        source_id,
        variant_id,
        tokens=tokens,
        positions=positions,
        atol=0.2,
        rtol=0.2,
        max_abs_error_threshold=0.001,
    )
    assert thresholded["passed"] is False
    assert thresholded["outputs"]["logits"]["passed"] is True
    assert thresholded["drift"]["thresholds"]["max_abs_error"] == 0.001
    assert thresholded["drift"]["allclose_passed"] is True
    assert thresholded["drift"]["passed"] is False
    assert thresholded["drift"]["failures"]

    with pytest.raises(InvalidOperationParameters, match="threshold must be non-negative"):
        client.compare_quantized_model(
            source_id,
            variant_id,
            tokens=tokens,
            mean_abs_error_threshold=-1,
        )

    api = fastapi_testclient.TestClient(create_app(service=client.service))
    response = api.post(
        "/v1/models/compare-quantized",
        json={
            "source_artifact_id": source_id,
            "quantized_artifact_id": variant_id,
            "tokens": tokens,
            "positions": positions,
            "atol": 0.2,
            "rtol": 0.2,
            "max_abs_error_threshold": 0.2,
            "mean_abs_error_threshold": 0.2,
            "max_rel_error_threshold": 1e9,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["source_vs_quantized"] is True
    assert body["quantization"]["schema"] == "uzen.quantization/1"
    assert body["outputs"]["logits"]["max_abs_error"] == logits["max_abs_error"]
    assert body["drift"]["thresholds"] == {
        "max_abs_error": 0.2,
        "mean_abs_error": 0.2,
        "max_rel_error": 1e9,
    }
    assert body["drift"]["passed"] is True

    with pytest.raises(InvalidOperationParameters, match="not a persisted quantized"):
        client.compare_quantized_model(source_id, source_id, tokens=tokens)


def test_int8_excludes_weights_shared_with_non_linear_operators():
    graph = Graph(
        name="shared-weight",
        inputs={"tokens": TensorType.of((1, 3), "int64")},
    )
    graph.add_node(
        "embed",
        "embedding",
        {"x": Ref("input:tokens")},
        {"weight": "param:shared.weight"},
    )
    graph.add_node(
        "project",
        "linear",
        {"x": Ref("embed")},
        {"weight": "param:shared.weight"},
    )
    graph.set_output("y", Ref("project"))
    package = WeightPackage.from_arrays(
        {"shared.weight": np.arange(16, dtype=np.float32).reshape(4, 4) / 10.0},
        name="shared",
    )

    result = quantize_int8(graph, package)
    assert result.report.num_linear_nodes == 1
    assert result.report.num_rewritten_nodes == 0
    assert result.report.tensors == ()
    assert result.report.excluded
    assert "embedding" in result.report.excluded[0]["reason"]
    assert result.graph.nodes["project"].op == "linear"
    assert result.package.get("shared.weight").dtype == np.float32
