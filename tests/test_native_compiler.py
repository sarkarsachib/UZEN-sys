"""Native graph compiler: prepared kernels, fusions, memory plan, plan execution."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from uzen.core.nir.executor import execute as numpy_execute
from uzen.core.nir.ops import (
    _conv2d_run,
    _gcn_run,
    _gru_run,
    _lstm_run,
    _moe_run,
    _pool2d_run,
    _rnn_run,
    _ssm_run,
)
from uzen.core.nws.families import build_architecture
from uzen.core.wie.quantization import quantize_int8
from uzen.errors import InvalidOperationParameters, UnsupportedOperator
from uzen.runtimes.native import (
    NativeLibrary,
    NativeRuntime,
    build_native_library,
    compile_native_plan,
    execute_native,
    execute_plan,
)

pytestmark = pytest.mark.skipif(
    shutil.which("g++") is None and shutil.which("c++") is None,
    reason="a C++ compiler is required for the native compiler tests",
)

TRANSFORMER_SPEC = dict(
    d_model=32, n_heads=4, n_layers=2, d_ff=64, vocab_size=64, max_seq_len=16
)


@pytest.fixture(scope="session")
def native_library(tmp_path_factory: pytest.TempPathFactory) -> NativeLibrary:
    output = tmp_path_factory.mktemp("native-compiler") / "libuzen_native.so"
    build_native_library(output, force=True)
    return NativeLibrary(output)


def _assert_close(actual: np.ndarray, expected: np.ndarray, *, atol: float = 1e-6):
    assert actual.shape == expected.shape
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=atol)


# --------------------------------------------------------------------------- #
# prepared kernels
# --------------------------------------------------------------------------- #


def test_packed_gemm_matches_numpy(native_library: NativeLibrary):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((6, 40), dtype=np.float32)
    weight = rng.standard_normal((24, 40), dtype=np.float32)
    bias = rng.standard_normal(24, dtype=np.float32)
    packed = native_library.pack_weight(weight)
    assert packed.out_features == 24 and packed.in_features == 40
    assert packed.padded_elements >= weight.size
    expected = x @ weight.T + bias
    _assert_close(native_library.gemm_packed(x, packed, bias=bias), expected)
    gelu = 0.5 * expected * (
        1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (expected + 0.044715 * expected**3))
    )
    _assert_close(
        native_library.gemm_packed(x, packed, bias=bias, activation="gelu"), gelu
    )
    _assert_close(
        native_library.gemm_packed(x, packed, activation="relu"),
        np.maximum(x @ weight.T, 0.0),
    )
    # Tall/skinny and wide shapes exercise partial blocks in both directions.
    for shape in ((1, 5), (9, 5)):
        small = rng.standard_normal(shape, dtype=np.float32)
        w = rng.standard_normal((3, 5), dtype=np.float32)
        _assert_close(
            native_library.gemm_packed(small, native_library.pack_weight(w)),
            small @ w.T,
        )


def test_fused_add_norm_matches_the_reference(native_library: NativeLibrary):
    rng = np.random.default_rng(1)
    a = rng.standard_normal((4, 12), dtype=np.float32)
    b = rng.standard_normal((4, 12), dtype=np.float32)
    weight = rng.standard_normal(12, dtype=np.float32)
    bias = rng.standard_normal(12, dtype=np.float32)
    total = (a + b).astype(np.float64)
    expected_layer = (
        (total - total.mean(axis=-1, keepdims=True))
        / np.sqrt(total.var(axis=-1, keepdims=True) + 1e-5)
        * weight
        + bias
    ).astype(np.float32)
    output, residual = native_library.add_norm(
        a, b, weight=weight, bias=bias, residual=True
    )
    _assert_close(output, expected_layer)
    _assert_close(residual, a + b)
    expected_rms = (
        total / np.sqrt((total * total).mean(axis=-1, keepdims=True) + 1e-6) * weight
    ).astype(np.float32)
    output, residual = native_library.add_norm(
        a, b, weight=weight, eps=1e-6, kind="rmsnorm"
    )
    _assert_close(output, expected_rms)
    assert residual is None
    with pytest.raises(InvalidOperationParameters, match="equal shapes"):
        native_library.add_norm(a, b[:1], weight=weight)


def test_conv2d_and_maxpool2d_match_the_numpy_operators(native_library: NativeLibrary):
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 3, 7, 7), dtype=np.float32)
    weight = rng.standard_normal((4, 3, 3, 3), dtype=np.float32)
    bias = rng.standard_normal(4, dtype=np.float32)
    params = {"weight": "param:w", "bias": "param:b", "stride": 2, "padding": 1}
    expected = _conv2d_run([x], params, {"w": weight, "b": bias})[0]
    _assert_close(
        native_library.conv2d(x, weight, bias, stride=2, padding=1), expected
    )
    expected_pool = _pool2d_run([x], {"kernel": 2}, {})[0]
    _assert_close(native_library.maxpool2d(x, kernel=2), expected_pool)
    expected_pool_pad = _pool2d_run([x], {"kernel": 3, "stride": 2, "padding": 1}, {})[0]
    _assert_close(
        native_library.maxpool2d(x, kernel=3, stride=2, padding=1), expected_pool_pad
    )


@pytest.mark.parametrize(
    ("kind", "run"),
    [("rnn", _rnn_run), ("lstm", _lstm_run), ("gru", _gru_run)],
)
def test_recurrent_kernels_match_the_numpy_operators(
    native_library: NativeLibrary, kind: str, run
):
    rng = np.random.default_rng(3)
    gates = {"rnn": 1, "lstm": 4, "gru": 3}[kind]
    features, hidden, batch, steps = 5, 4, 3, 6
    x = rng.standard_normal((batch, steps, features), dtype=np.float32)
    weight_ih = rng.standard_normal((gates * hidden, features), dtype=np.float32)
    weight_hh = rng.standard_normal((gates * hidden, hidden), dtype=np.float32)
    bias = rng.standard_normal(gates * hidden, dtype=np.float32)
    bound = {"wih": weight_ih, "whh": weight_hh, "b": bias}
    params = {
        "weight_ih": "param:wih",
        "weight_hh": "param:whh",
        "bias": "param:b",
    }
    expected = run([x], params, bound)[0]
    _assert_close(
        native_library.recurrent(x, weight_ih, weight_hh, bias, kind=kind), expected
    )
    # Running the same input twice is deterministic and independent of state.
    again = native_library.recurrent(x, weight_ih, weight_hh, bias, kind=kind)
    np.testing.assert_array_equal(again, native_library.recurrent(
        x, weight_ih, weight_hh, bias, kind=kind
    ))


def test_ssm_gcn_and_moe_match_the_numpy_operators(native_library: NativeLibrary):
    rng = np.random.default_rng(4)

    x = rng.standard_normal((2, 4, 3), dtype=np.float32)
    a = rng.standard_normal((3, 2), dtype=np.float32) * 0.5
    b = rng.standard_normal((3, 2), dtype=np.float32)
    c = rng.standard_normal((3, 2), dtype=np.float32)
    d = rng.standard_normal(3, dtype=np.float32)
    expected = _ssm_run(
        [x], {"A": "param:A", "B": "param:B", "C": "param:C", "D": "param:D"},
        {"A": a, "B": b, "C": c, "D": d},
    )[0]
    _assert_close(native_library.ssm(x, a, b, c, d), expected)

    nodes, in_features, out_features = 5, 3, 4
    features = rng.standard_normal((2, nodes, in_features), dtype=np.float32)
    adjacency = (rng.random((2, nodes, nodes)) > 0.5).astype(np.float32)
    weight = rng.standard_normal((out_features, in_features), dtype=np.float32)
    bias = rng.standard_normal(out_features, dtype=np.float32)
    expected_gcn = _gcn_run(
        [features, adjacency],
        {"weight": "param:w", "bias": "param:b"},
        {"w": weight, "b": bias},
    )[0]
    _assert_close(
        native_library.gcn(features, adjacency, weight, bias), expected_gcn
    )

    tokens, width, experts, hidden = 6, 4, 3, 5
    moe_x = rng.standard_normal((tokens, width), dtype=np.float32)
    router = rng.standard_normal((experts, width), dtype=np.float32)
    up_weight = rng.standard_normal((experts, hidden, width), dtype=np.float32)
    down_weight = rng.standard_normal((experts, width, hidden), dtype=np.float32)
    up_bias = rng.standard_normal((experts, hidden), dtype=np.float32)
    down_bias = rng.standard_normal((experts, width), dtype=np.float32)
    params = {
        "router": "param:router",
        "up_weight": "param:up",
        "down_weight": "param:down",
        "up_bias": "param:upb",
        "down_bias": "param:downb",
        "top_k": 2,
        "activation": "gelu",
    }
    bound = {
        "router": router,
        "up": up_weight,
        "down": down_weight,
        "upb": up_bias,
        "downb": down_bias,
    }
    expected_moe = _moe_run([moe_x], params, bound)[0]
    _assert_close(
        native_library.moe(
            moe_x,
            router,
            up_weight,
            down_weight,
            up_bias=up_bias,
            down_bias=down_bias,
            top_k=2,
        ),
        expected_moe,
    )


def test_int8_quantized_linear_matches_the_reference(native_library: NativeLibrary):
    rng = np.random.default_rng(5)
    x = rng.standard_normal((3, 6), dtype=np.float32)
    codes = rng.integers(-127, 128, size=(4, 6), dtype=np.int8)
    scale = np.abs(rng.standard_normal(4)).astype(np.float32) + 0.01
    bias = rng.standard_normal(4, dtype=np.float32)
    expected = (
        x.astype(np.float64) @ (codes.astype(np.float64) * scale[:, None]).T + bias
    ).astype(np.float32)
    _assert_close(
        native_library.quantized_linear(x, codes, scale, bias), expected
    )
    with pytest.raises(InvalidOperationParameters, match="int8"):
        native_library.quantized_linear(x, codes.astype(np.float32), scale)


def test_flash_attention_matches_the_reference(native_library: NativeLibrary):
    rng = np.random.default_rng(6)
    q = rng.standard_normal((2, 4, 3, 8), dtype=np.float32)
    k = rng.standard_normal((2, 4, 9, 8), dtype=np.float32)
    v = rng.standard_normal((2, 4, 9, 8), dtype=np.float32)
    for kwargs in ({}, {"causal": False}, {"window": 2}, {"alibi": np.linspace(0.1, 0.4, 4).astype(np.float32)}):
        reference = native_library.attention(q, k, v, **kwargs)
        _assert_close(native_library.attention_flash(q, k, v, **kwargs), reference)


# --------------------------------------------------------------------------- #
# compilation and plan execution
# --------------------------------------------------------------------------- #


def _definition(family: str, spec: dict, seed: int = 7):
    definition = build_architecture(family, spec)
    return definition, definition.build_package(seed=seed)


def test_plan_fuses_linear_activation_and_residual_norm(native_library: NativeLibrary):
    definition, package = _definition("transformer_decoder", TRANSFORMER_SPEC)
    plan = compile_native_plan(definition.graph, package)
    kinds = {fusion["kind"] for fusion in plan.fusions}
    assert kinds == {"linear_activation", "residual_norm"}
    assert plan.fused_steps() >= 2
    assert plan.num_steps < len(definition.graph.nodes)
    assert plan.prepack_names, "linear weights must be prepacked"
    described = plan.describe()
    assert described["steps"] == plan.num_steps
    assert "gemm_packed" in described["operators"]
    # The fused step really is one kernel call for linear+bias+activation.
    gemm = next(step for step in plan.steps if step.op == "gemm_packed"
                and step.attrs["activation"] != "none")
    assert len(gemm.nodes) == 2


def test_plan_execution_matches_the_numpy_executor(native_library: NativeLibrary):
    definition, package = _definition("mlp", dict(input_dim=8, hidden_dims=[16, 12], output_dim=4))
    inputs = definition.sample_inputs(batch=3, seed=0)
    reference = numpy_execute(definition.graph, inputs, package)
    plan = compile_native_plan(definition.graph, package)
    execution = execute_plan(plan, inputs, package, library=native_library)
    for name, expected in reference.outputs.items():
        _assert_close(execution.outputs[name], expected)
    assert execution.fused_steps >= 1
    assert "gemm_packed" in execution.native_ops


@pytest.mark.parametrize(
    ("family", "spec"),
    [
        ("transformer_decoder", TRANSFORMER_SPEC),
        (
            "cnn",
            dict(
                in_channels=3,
                height=16,
                width=16,
                channels=[8, 12],
                kernel=3,
                padding=1,
                num_classes=5,
                pool=2,
            ),
        ),
        (
            "recurrent",
            dict(vocab_size=32, d_model=16, hidden_size=12, cell="lstm", num_classes=4),
        ),
        (
            "moe",
            dict(vocab_size=32, d_model=16, n_experts=4, top_k=2, d_ff=24, num_classes=3),
        ),
        ("state_space", dict(vocab_size=32, d_model=16, state_dim=4, num_classes=3)),
        ("gnn", dict(in_features=6, hidden_dim=10, n_layers=2, nodes=5, num_classes=3)),
        (
            "hybrid",
            dict(
                in_channels=3,
                height=16,
                width=16,
                channels=[8],
                kernel=3,
                padding=1,
                hidden_size=12,
                num_classes=4,
                pool=2,
            ),
        ),
    ],
)
def test_plans_cover_every_supported_family(
    native_library: NativeLibrary, family: str, spec: dict
):
    definition, package = _definition(family, spec)
    inputs = definition.sample_inputs(batch=2, seed=0)
    reference = numpy_execute(definition.graph, inputs, package)
    plan = compile_native_plan(definition.graph, package)
    execution = execute_plan(plan, inputs, package, library=native_library)
    for name, expected in reference.outputs.items():
        _assert_close(execution.outputs[name], expected)
    # Where the direct executor covers the graph too, both native paths must agree.
    try:
        direct = execute_native(
            definition.graph, inputs, package, library=native_library
        )
    except UnsupportedOperator:
        # The compiled plan is the wider-coverage path (conv, recurrent, MoE, ...).
        assert {"conv2d", "recurrent", "moe", "ssm", "gcn"} & set(
            step.op for step in plan.steps
        )
    else:
        for name, expected in direct.outputs.items():
            _assert_close(execution.outputs[name], expected)


def test_memory_plan_reuses_slots_without_corrupting_results(
    native_library: NativeLibrary,
):
    definition, package = _definition("transformer_decoder", TRANSFORMER_SPEC)
    inputs = definition.sample_inputs(batch=2, seq=6, seed=0)
    fused = compile_native_plan(definition.graph, package)
    unfused = compile_native_plan(definition.graph, package, fuse=False)
    fused_result = execute_plan(fused, inputs, package, library=native_library)
    unfused_result = execute_plan(unfused, inputs, package, library=native_library)
    for name, expected in unfused_result.outputs.items():
        _assert_close(fused_result.outputs[name], expected)
    memory = fused_result.memory
    assert memory["reused_buffers"] > 0
    assert memory["peak_bytes"] <= memory["allocated_bytes"]
    assert fused.num_steps < unfused.num_steps


def test_plan_is_shape_polymorphic(native_library: NativeLibrary):
    definition, package = _definition("transformer_decoder", TRANSFORMER_SPEC)
    plan = compile_native_plan(definition.graph, package)
    for batch, seq in ((1, 3), (2, 6)):
        inputs = definition.sample_inputs(batch=batch, seq=seq, seed=1)
        reference = numpy_execute(definition.graph, inputs, package)
        execution = execute_plan(plan, inputs, package, library=native_library)
        for name, expected in reference.outputs.items():
            _assert_close(execution.outputs[name], expected)
    assert plan.bound_variants == 2


def test_plan_capture_and_fusion_interaction(native_library: NativeLibrary):
    definition, package = _definition("transformer_decoder", TRANSFORMER_SPEC)
    inputs = definition.sample_inputs(batch=1, seq=4, seed=0)
    fused = compile_native_plan(definition.graph, package)
    linear = next(
        fusion["nodes"][0]
        for fusion in fused.fusions
        if fusion["kind"] == "linear_activation"
    )
    with pytest.raises(InvalidOperationParameters, match="fused"):
        execute_plan(fused, inputs, package, capture=[linear], library=native_library)
    unfused = compile_native_plan(definition.graph, package, fuse=False)
    reference = numpy_execute(definition.graph, inputs, package, capture=[linear])
    execution = execute_plan(
        unfused, inputs, package, capture=[linear], library=native_library
    )
    _assert_close(execution.intermediates[linear], reference.intermediates[linear])


def test_compiled_plan_executes_int8_quantized_linear(native_library: NativeLibrary):
    definition, package = _definition("transformer_decoder", TRANSFORMER_SPEC)
    quantized = quantize_int8(definition.graph, package)
    assert quantized.report.num_rewritten_nodes > 0
    inputs = definition.sample_inputs(batch=1, seq=5, seed=0)
    reference = numpy_execute(quantized.graph, inputs, quantized.package)
    plan = compile_native_plan(quantized.graph, quantized.package)
    execution = execute_plan(
        plan, inputs, quantized.package, library=native_library
    )
    for name, expected in reference.outputs.items():
        _assert_close(execution.outputs[name], expected, atol=1e-5)
    assert "quantized_linear" in execution.native_ops
    with pytest.raises(UnsupportedOperator, match="quantized_linear"):
        execute_native(
            quantized.graph, inputs, quantized.package, library=native_library
        )


def test_plan_refuses_stochastic_operators(native_library: NativeLibrary):
    definition, package = _definition("vae", dict(input_dim=8, latent_dim=4))
    with pytest.raises(UnsupportedOperator, match="sample_normal"):
        compile_native_plan(definition.graph, package)


def test_plan_rejects_non_float32_inputs(native_library: NativeLibrary):
    definition, package = _definition("mlp", dict(input_dim=4, hidden_dims=[8], output_dim=2))
    plan = compile_native_plan(definition.graph, package)
    inputs = definition.sample_inputs(batch=1, seed=0)
    name = next(iter(inputs))
    converted = {key: np.asarray(value, dtype=np.float64) for key, value in inputs.items()}
    with pytest.raises(InvalidOperationParameters, match="float32"):
        execute_plan(plan, converted, package, library=native_library)
    assert name


def test_runtime_compiles_and_reports_plan_capabilities(native_library: NativeLibrary):
    runtime = NativeRuntime(library_path=native_library.path)
    definition, package = _definition("mlp", dict(input_dim=6, hidden_dims=[10], output_dim=3))
    inputs = definition.sample_inputs(batch=2, seed=0)
    reference = numpy_execute(definition.graph, inputs, package)
    plan = runtime.compile(definition.graph, package)
    execution = runtime.execute_compiled_graph(plan, inputs, package)
    for name, expected in reference.outputs.items():
        _assert_close(execution.outputs[name], expected)
    outputs = runtime.execute_compiled(definition.graph, inputs, package)
    for name, expected in reference.outputs.items():
        _assert_close(outputs[name], expected)
    capabilities = runtime.capabilities()
    assert capabilities.extra["compiled_plan"] is True
    assert "gemm_packed" in capabilities.extra["plan_ops"]
    kernels = runtime.describe()["kernel_library"]["kernels"]
    assert {"gemm_packed", "conv2d", "moe", "quantized_linear"} <= set(kernels)
