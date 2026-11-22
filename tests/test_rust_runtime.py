"""P13: Rust inference kernels — real build, real FFI calls, NumPy parity."""

from __future__ import annotations

import numpy as np
import pytest

from uzen.core.nir.executor import execute as numpy_execute
from uzen.core.nir.graph import Graph
from uzen.core.nir.types import Ref, TensorType
from uzen.core.nws.families import build_architecture
from uzen.core.nws.tensors import WeightPackage
from uzen.errors import RuntimeUnavailable, ShapeMismatch, UnsupportedOperator
from uzen.runtimes.rust import RustKernels, RustRuntime, build_library, find_cargo, library_path

needs_toolchain = pytest.mark.skipif(
    find_cargo() is None and library_path() is None,
    reason="needs a Rust toolchain (or a previously built library)",
)


@pytest.fixture(scope="module")
def build_report() -> dict:
    if find_cargo() is None:
        pytest.skip("cargo is not installed")
    return build_library(release=True, force=False)


@pytest.fixture(scope="module")
def kernels(build_report: dict) -> RustKernels:
    return RustKernels(build_report["library"])


def test_build_produces_a_loadable_library(build_report: dict, kernels: RustKernels) -> None:
    assert build_report["profile"] == "release"
    assert kernels.path.exists()
    assert kernels.version == 1
    described = kernels.describe()
    assert described["backend"] == "rust"
    assert "linear" in described["operators"]


def test_missing_library_and_toolchain_raise_instead_of_faking() -> None:
    with pytest.raises(RuntimeUnavailable):
        RustKernels("/nonexistent/libuzen_rust.so", build=False)


@needs_toolchain
def test_linear_matches_numpy(kernels: RustKernels) -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 5)).astype(np.float32)
    w = rng.standard_normal((4, 5)).astype(np.float32)
    b = rng.standard_normal(4).astype(np.float32)
    np.testing.assert_allclose(kernels.linear(x, w, b), x @ w.T + b, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(kernels.linear(x, w), x @ w.T, rtol=1e-6, atol=1e-6)


@needs_toolchain
def test_norms_match_the_mathematical_definition(kernels: RustKernels) -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4, 7)).astype(np.float32)
    w = rng.standard_normal(7).astype(np.float32)
    b = rng.standard_normal(7).astype(np.float32)
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    expected = (x - mean) / np.sqrt(var + 1e-5) * w + b
    np.testing.assert_allclose(kernels.layernorm(x, w, b, eps=1e-5), expected, rtol=1e-5, atol=1e-5)
    expected_rms = x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + 1e-6) * w
    np.testing.assert_allclose(kernels.rmsnorm(x, w, eps=1e-6), expected_rms, rtol=1e-5, atol=1e-5)


@needs_toolchain
def test_softmax_is_stable_and_normalized(kernels: RustKernels) -> None:
    x = np.array([[1000.0, 1001.0, 999.0], [-1.0, 0.0, 1.0]], dtype=np.float32)
    out = kernels.softmax(x)
    np.testing.assert_allclose(out.sum(axis=-1), np.ones(2), rtol=1e-6)
    shifted = np.exp(x - x.max(axis=-1, keepdims=True))
    expected = shifted / shifted.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-7)


@needs_toolchain
def test_elementwise_kernels_match_numpy(kernels: RustKernels) -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 6)).astype(np.float32)
    y = rng.standard_normal((2, 6)).astype(np.float32)
    expected_gelu = 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    np.testing.assert_allclose(kernels.gelu(x), expected_gelu, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(kernels.silu(x), x / (1.0 + np.exp(-x)), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(kernels.scale(x, 2.5), x * 2.5, rtol=1e-7)
    np.testing.assert_allclose(kernels.add(x, y), x + y, rtol=1e-7)
    np.testing.assert_allclose(kernels.mul(x, y), x * y, rtol=1e-7)


@needs_toolchain
def test_shape_validation_happens_before_calling_into_rust(kernels: RustKernels) -> None:
    x = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ShapeMismatch):
        kernels.linear(x, np.zeros((4, 5), dtype=np.float32))
    with pytest.raises(ShapeMismatch):
        kernels.add(x, np.zeros((3, 2), dtype=np.float32))
    with pytest.raises(ShapeMismatch):
        kernels.layernorm(np.zeros(3, dtype=np.float32))


@needs_toolchain
def test_runtime_executes_an_mlp_graph_with_numpy_parity(kernels: RustKernels) -> None:
    definition = build_architecture("mlp", dict(input_dim=8, hidden_dims=[16, 12], output_dim=4))
    package = definition.build_package(seed=7)
    inputs = definition.sample_inputs(batch=2, seed=0)
    reference = numpy_execute(definition.graph, inputs, package)
    runtime = RustRuntime(kernels)
    produced = runtime.execute(definition.graph, package, inputs)
    assert set(produced) == set(reference.outputs)
    for name, expected in reference.outputs.items():
        np.testing.assert_allclose(produced[name], expected, rtol=1e-5, atol=1e-6)


@needs_toolchain
def test_runtime_refuses_an_operator_it_does_not_implement(kernels: RustKernels) -> None:
    graph = Graph(name="matmul_probe")
    graph.add_input("a", TensorType.of([2, 3], "float32"))
    graph.add_input("b", TensorType.of([3, 2], "float32"))
    graph.add_node("mm", "matmul", {"a": Ref("input:a"), "b": Ref("input:b")})
    graph.set_output("y", Ref("mm:0"))
    package = WeightPackage.from_arrays({}, name="empty", family="generic")
    inputs = {
        "a": np.zeros((2, 3), dtype=np.float32),
        "b": np.zeros((3, 2), dtype=np.float32),
    }
    with pytest.raises(UnsupportedOperator) as excinfo:
        RustRuntime(kernels).execute(graph, package, inputs)
    assert "matmul" in str(excinfo.value)
