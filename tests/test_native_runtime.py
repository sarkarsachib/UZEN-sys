"""Native C++ kernels, ctypes bridge and native NIR execution."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from uzen.core.nir.executor import execute as numpy_execute
from uzen.core.nws.families import build_architecture
from uzen.core.wie.quantization import quantize_int8
from uzen.errors import InvalidOperationParameters, UnsupportedOperator
from uzen.runtimes.native import (
    NativeLibrary,
    NativeRuntime,
    build_native_library,
    execute_native,
)

pytestmark = pytest.mark.skipif(
    shutil.which("g++") is None and shutil.which("c++") is None,
    reason="a C++ compiler is required for the native runtime tests",
)


@pytest.fixture(scope="session")
def native_library(tmp_path_factory: pytest.TempPathFactory) -> NativeLibrary:
    output = tmp_path_factory.mktemp("native") / "libuzen_native.so"
    build_native_library(output, force=True)
    return NativeLibrary(output)


def _reference_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    scale: float | str | None = None,
    causal: bool = True,
    alibi: np.ndarray | None = None,
    window: int | None = None,
) -> np.ndarray:
    q64, k64, v64 = (array.astype(np.float64) for array in (q, k, v))
    scale_value = 1.0 / np.sqrt(q.shape[-1]) if scale in (None, "auto") else float(scale)
    scores = np.einsum("bhtd,bhsd->bhts", q64, k64) * scale_value
    t, s = scores.shape[-2], scores.shape[-1]
    qpos, kpos = np.arange(s - t, s)[:, None], np.arange(s)[None, :]
    if alibi is not None:
        distance = (qpos - kpos) if causal else np.abs(qpos - kpos)
        scores = scores - np.asarray(alibi)[None, :, None, None] * distance[
            None, None, :, :
        ]
    mask = np.zeros((t, s), dtype=bool)
    if causal:
        mask |= kpos > qpos
    if window is not None:
        mask |= np.abs(qpos - kpos) > window
    scores = np.where(mask[None, None, :, :], -np.inf, scores)
    maximum = np.max(scores, axis=-1, keepdims=True)
    maximum = np.where(np.isfinite(maximum), maximum, 0.0)
    exponentials = np.exp(scores - maximum)
    denominator = exponentials.sum(axis=-1, keepdims=True)
    denominator = np.where(denominator == 0, 1.0, denominator)
    return np.einsum(
        "bhts,bhsd->bhtd", exponentials / denominator, v64
    ).astype(np.float32)


def test_build_produces_a_loadable_library(native_library: NativeLibrary, tmp_path: Path):
    assert native_library.path.is_file()
    rebuilt = build_native_library(tmp_path / "rebuilt.so", force=True)
    assert rebuilt.is_file()
    NativeLibrary(rebuilt)


def test_linear_and_matmul_match_numpy(native_library: NativeLibrary):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((7, 5), dtype=np.float32)
    weight = rng.standard_normal((3, 5), dtype=np.float32)
    bias = rng.standard_normal(3, dtype=np.float32)
    np.testing.assert_allclose(
        native_library.linear(x, weight, bias),
        x @ weight.T + bias,
        rtol=1e-5,
        atol=1e-6,
    )
    left = rng.standard_normal((17, 13), dtype=np.float32)
    right = rng.standard_normal((13, 11), dtype=np.float32)
    np.testing.assert_allclose(
        native_library.matmul(left, right), left @ right, rtol=1e-4, atol=1e-5
    )
    batched = rng.standard_normal((2, 4, 5), dtype=np.float32)
    np.testing.assert_allclose(
        native_library.linear(batched, weight),
        batched @ weight.T,
        rtol=1e-5,
        atol=1e-6,
    )


def test_normalization_and_softmax_match_numpy(native_library: NativeLibrary):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((9, 7), dtype=np.float32)
    weight = rng.standard_normal(7, dtype=np.float32)
    bias = rng.standard_normal(7, dtype=np.float32)

    expected_layer = (
        (x - x.mean(axis=-1, keepdims=True))
        / np.sqrt(x.var(axis=-1, keepdims=True) + 1e-5)
        * weight
        + bias
    ).astype(np.float32)
    np.testing.assert_allclose(
        native_library.layernorm(x, weight=weight, bias=bias),
        expected_layer,
        rtol=1e-5,
        atol=1e-6,
    )

    expected_rms = (
        x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + 1e-6) * weight
    ).astype(np.float32)
    np.testing.assert_allclose(
        native_library.rmsnorm(x, weight=weight),
        expected_rms,
        rtol=1e-5,
        atol=1e-6,
    )

    x64 = x.astype(np.float64)
    exponentials = np.exp(x64 - x64.max(axis=-1, keepdims=True))
    expected_softmax = (exponentials / exponentials.sum(axis=-1, keepdims=True)).astype(
        np.float32
    )
    np.testing.assert_allclose(
        native_library.softmax(x), expected_softmax, rtol=1e-5, atol=1e-6
    )


def test_activations_and_elementwise_match_numpy(native_library: NativeLibrary):
    rng = np.random.default_rng(2)
    x = rng.standard_normal((5, 6), dtype=np.float32)
    y = rng.standard_normal((5, 6), dtype=np.float32)
    gelu = 0.5 * x * (
        1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3))
    )
    exact = 0.5 * x * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))
    expected = {
        "gelu": gelu,
        "gelu_exact": exact,
        "silu": x / (1.0 + np.exp(-x)),
        "relu": np.maximum(x, 0.0),
        "tanh": np.tanh(x),
        "sigmoid": 1.0 / (1.0 + np.exp(-x)),
    }
    for name, reference in expected.items():
        np.testing.assert_allclose(
            native_library.unary(name, x),
            reference.astype(np.float32),
            rtol=1e-5,
            atol=1e-6,
        )

    np.testing.assert_allclose(
        native_library.binary("add", x, y), x + y, rtol=0, atol=0
    )
    np.testing.assert_allclose(
        native_library.binary("mul", x, y), x * y, rtol=0, atol=0
    )
    np.testing.assert_allclose(
        native_library.scale(x, 0.25), x * np.float32(0.25), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        native_library.clip(x, -0.5, 0.5), np.clip(x, -0.5, 0.5), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"causal": False},
        {"window": 1},
        {"alibi": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)},
        {
            "causal": False,
            "window": 2,
            "alibi": np.array([0.1, 0.2], dtype=np.float32),
            "scale": 0.3,
        },
    ],
)
def test_attention_masks_match_the_reference(
    native_library: NativeLibrary, kwargs: dict
):
    rng = np.random.default_rng(3)
    heads = len(kwargs.get("alibi", [0, 0]))
    q = rng.standard_normal((2, heads, 3, 8), dtype=np.float32)
    k = rng.standard_normal((2, heads, 5, 8), dtype=np.float32)
    v = rng.standard_normal((2, heads, 5, 8), dtype=np.float32)
    if "causal" not in kwargs:
        kwargs = {"causal": True, **kwargs}
    np.testing.assert_allclose(
        native_library.attention(q, k, v, **kwargs),
        _reference_attention(q, k, v, **kwargs),
        rtol=1e-5,
        atol=1e-6,
    )


def test_embedding_and_reductions_match_numpy(native_library: NativeLibrary):
    rng = np.random.default_rng(4)
    weight = rng.standard_normal((10, 4), dtype=np.float32)
    indices = np.array([[1, 2], [9, 0]], dtype=np.int64)
    np.testing.assert_array_equal(
        native_library.embedding(indices, weight), weight[indices]
    )
    x = rng.standard_normal((6, 5), dtype=np.float32)
    np.testing.assert_allclose(
        native_library.reduce_last(x, mode="mean"), x.mean(axis=-1), rtol=1e-5
    )
    np.testing.assert_allclose(
        native_library.reduce_last(x, mode="sum"), x.sum(axis=-1), rtol=1e-5
    )


def test_native_mlp_matches_the_numpy_executor(native_library: NativeLibrary):
    definition = build_architecture(
        "mlp", dict(input_dim=8, hidden_dims=[16, 12], output_dim=4)
    )
    package = definition.build_package(seed=3)
    inputs = definition.sample_inputs(batch=2, seed=0)
    reference = numpy_execute(definition.graph, inputs, package)
    execution = execute_native(definition.graph, inputs, package, library=native_library)
    for name, expected in reference.outputs.items():
        np.testing.assert_allclose(
            execution.outputs[name], expected, rtol=1e-4, atol=1e-5
        )
    assert "linear" in execution.native_ops


def test_native_transformer_matches_the_numpy_executor(native_library: NativeLibrary):
    definition = build_architecture(
        "transformer_decoder",
        dict(
            d_model=32,
            n_heads=4,
            n_layers=2,
            d_ff=64,
            vocab_size=64,
            max_seq_len=16,
        ),
    )
    package = definition.build_package(seed=5)
    inputs = definition.sample_inputs(batch=1, seq=6, seed=0)
    reference = numpy_execute(definition.graph, inputs, package)
    execution = execute_native(definition.graph, inputs, package, library=native_library)
    for name, expected in reference.outputs.items():
        np.testing.assert_allclose(
            execution.outputs[name], expected, rtol=1e-4, atol=1e-5
        )
    assert {"embedding", "attention", "layernorm", "gelu"} <= set(
        execution.native_ops
    )
    assert {"split_qkv", "merge_heads", "pos_embedding"} <= set(
        execution.layout_ops
    )


def test_native_capture_matches_the_numpy_executor(native_library: NativeLibrary):
    definition = build_architecture(
        "transformer_decoder",
        dict(
            d_model=16,
            n_heads=2,
            n_layers=1,
            d_ff=32,
            vocab_size=32,
            max_seq_len=8,
        ),
    )
    package = definition.build_package(seed=1)
    inputs = definition.sample_inputs(batch=1, seq=4, seed=0)
    attention_node = next(
        node_id
        for node_id, node in definition.graph.nodes.items()
        if node.op == "attention"
    )
    reference = numpy_execute(
        definition.graph, inputs, package, capture=[attention_node]
    )
    execution = execute_native(
        definition.graph,
        inputs,
        package,
        capture=[attention_node],
        library=native_library,
    )
    np.testing.assert_allclose(
        execution.intermediates[attention_node],
        reference.intermediates[attention_node],
        rtol=1e-4,
        atol=1e-5,
    )


def test_native_executor_refuses_quantized_linear(native_library: NativeLibrary):
    definition = build_architecture(
        "transformer_decoder",
        dict(
            d_model=16,
            n_heads=2,
            n_layers=1,
            d_ff=32,
            vocab_size=32,
            max_seq_len=8,
        ),
    )
    package = definition.build_package(seed=2)
    quantized = quantize_int8(definition.graph, package)
    assert quantized.report.num_rewritten_nodes > 0
    inputs = definition.sample_inputs(batch=1, seq=3, seed=0)
    with pytest.raises(UnsupportedOperator, match="quantized_linear"):
        execute_native(
            quantized.graph, inputs, quantized.package, library=native_library
        )


def test_native_executor_rejects_non_float32_arithmetic(native_library: NativeLibrary):
    definition = build_architecture(
        "mlp", dict(input_dim=4, hidden_dims=[8], output_dim=2)
    )
    package = definition.build_package(seed=0)
    inputs = definition.sample_inputs(batch=1, seed=0)
    name = next(iter(inputs))
    with pytest.raises(InvalidOperationParameters, match="float32"):
        execute_native(
            definition.graph,
            {name: np.asarray(inputs[name], dtype=np.float64)},
            package,
            library=native_library,
        )


def test_native_runtime_capabilities_are_honest(native_library: NativeLibrary):
    runtime = NativeRuntime(library_path=native_library.path)
    capabilities = runtime.capabilities()
    assert capabilities.backend == "native-cpp"
    assert capabilities.dtypes == ("float32",)
    assert capabilities.supports_autograd is False
    assert capabilities.supports_gguf is True
    assert capabilities.extra["cuda"] is False
    described = runtime.describe()
    assert described["kernel_library"]["path"] == str(native_library.path)
    assert "linear" in described["kernel_library"]["kernels"]
