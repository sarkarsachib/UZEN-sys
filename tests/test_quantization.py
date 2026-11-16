"""Weight-only int8 execution and quantization-report coverage."""

from __future__ import annotations

import json

import numpy as np

from uzen.core.nir.executor import execute
from uzen.core.nir.graph import Graph
from uzen.core.nir.types import Ref, TensorType
from uzen.core.nws.tensors import WeightPackage
from uzen.core.wie.importer import ModelImporter
from uzen.core.wie.quantization import quantize_int8
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
