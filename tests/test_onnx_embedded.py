"""P13: embedded ONNX executor (own protobuf decoding, no onnxruntime)."""

from __future__ import annotations

import numpy as np
import pytest

from uzen.core.nir.executor import execute as numpy_execute
from uzen.core.nws.families import build_architecture
from uzen.errors import ShapeMismatch, UnsupportedFormat, UnsupportedOperator
from uzen.runtimes.onnx_embedded import (
    SUPPORTED_OPS,
    EmbeddedOnnxRuntime,
    OnnxEmbeddedExecutor,
    execute_onnx_model,
    parse_onnx_graph,
)

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402


def _tensor(name: str, array: np.ndarray):
    return numpy_helper.from_array(np.asarray(array), name=name)


def _linear_model() -> bytes:
    weight = np.arange(12, dtype=np.float32).reshape(4, 3) / 10.0
    bias = np.linspace(-1.0, 1.0, 4).astype(np.float32)
    nodes = [
        helper.make_node("MatMul", ["x", "weight_t"], ["mm"], name="mm"),
        helper.make_node("Add", ["mm", "bias"], ["y"], name="add"),
    ]
    graph = helper.make_graph(
        nodes,
        "linear",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])],
        initializer=[_tensor("weight_t", weight.T), _tensor("bias", bias)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 8
    return model.SerializeToString()


def test_parse_reports_structure_without_onnx_runtime() -> None:
    graph = parse_onnx_graph(_linear_model())
    assert graph.name == "linear"
    assert graph.inputs == ["x"] and graph.outputs == ["y"]
    assert set(graph.initializers) == {"weight_t", "bias"}
    described = OnnxEmbeddedExecutor(graph).describe()
    assert described["operators"] == ["Add", "MatMul"]
    assert described["backend"] == "onnx-embedded"


def test_linear_model_executes_with_numpy_parity() -> None:
    weight = np.arange(12, dtype=np.float32).reshape(4, 3) / 10.0
    bias = np.linspace(-1.0, 1.0, 4).astype(np.float32)
    x = np.arange(6, dtype=np.float32).reshape(2, 3) / 3.0
    produced = execute_onnx_model(_linear_model(), {"x": x})
    np.testing.assert_allclose(produced["y"], x @ weight.T + bias, rtol=1e-6, atol=1e-6)
    assert produced["y"].shape == (2, 4)


def test_exported_uzen_graph_runs_in_the_embedded_executor() -> None:
    from uzen.runtimes.onnx_runtime import export_onnx

    definition = build_architecture("mlp", dict(input_dim=8, hidden_dims=[16], output_dim=4))
    package = definition.build_package(seed=11)
    model = export_onnx(definition.graph, package)
    serialized = model.SerializeToString()
    inputs = definition.sample_inputs(batch=2, seed=0)
    reference = numpy_execute(definition.graph, inputs, package)
    produced = execute_onnx_model(serialized, inputs)
    # The ONNX exporter names graph outputs after the producing node (``node:port``),
    # so compare in declaration order rather than by NIR output name.
    assert len(produced) == len(reference.outputs)
    for value, expected in zip(produced.values(), reference.outputs.values()):
        np.testing.assert_allclose(value, expected, rtol=1e-5, atol=1e-6)


def test_softmax_axis_and_elementwise_ops() -> None:
    nodes = [
        helper.make_node("Softmax", ["x"], ["s"], axis=1),
        helper.make_node("Mul", ["s", "x"], ["m"]),
        helper.make_node("Sqrt", ["m"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "ops",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    x = np.abs(np.arange(6, dtype=np.float32).reshape(2, 3)) + 1.0
    produced = execute_onnx_model(model.SerializeToString(), {"x": x})
    shifted = np.exp(x - x.max(axis=1, keepdims=True))
    expected = np.sqrt(shifted / shifted.sum(axis=1, keepdims=True) * x)
    np.testing.assert_allclose(produced["y"], expected, rtol=1e-6, atol=1e-6)


def test_unsupported_operator_is_named_not_guessed() -> None:
    nodes = [helper.make_node("Conv", ["x", "w"], ["y"])]
    graph = helper.make_graph(
        nodes,
        "conv",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 2])],
        initializer=[_tensor("w", np.zeros((1, 1, 3, 3), dtype=np.float32))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    with pytest.raises(UnsupportedOperator) as excinfo:
        OnnxEmbeddedExecutor(model.SerializeToString())
    assert "Conv" in str(excinfo.value)
    assert "Conv" not in SUPPORTED_OPS


def test_malformed_bytes_are_refused() -> None:
    with pytest.raises(UnsupportedFormat):
        parse_onnx_graph(b"")
    with pytest.raises(UnsupportedFormat):
        parse_onnx_graph(b"\x08\x01")  # valid varint, but no graph field
    with pytest.raises(UnsupportedFormat):
        parse_onnx_graph(b"\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff")


def test_input_validation() -> None:
    executor = OnnxEmbeddedExecutor(_linear_model())
    with pytest.raises(ShapeMismatch):
        executor.execute({})
    with pytest.raises(ShapeMismatch):
        executor.execute({"x": np.zeros((2, 3), np.float32), "other": np.zeros(1, np.float32)})


def test_runtime_adapter_reports_cpu_and_lists_operators() -> None:
    runtime = EmbeddedOnnxRuntime(_linear_model())
    capabilities = runtime.capabilities()
    assert capabilities["backend"] == "onnx-embedded"
    assert capabilities["device"] == "cpu" and capabilities["accelerated"] is False
    produced = runtime.execute({"x": np.ones((1, 3), dtype=np.float32)})
    assert produced["y"].shape == (1, 4)
