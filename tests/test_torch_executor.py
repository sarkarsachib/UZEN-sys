"""Torch-CPU NIR execution: parity with the NumPy reference across every family."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tests.test_llama_reconstruction import _write_llama  # noqa: E402
from tests.test_neox_reconstruction import _write_neox  # noqa: E402
from uzen.core.nir.executor import execute  # noqa: E402
from uzen.core.nir.graph import Graph  # noqa: E402
from uzen.core.nir.types import Ref, TensorType  # noqa: E402
from uzen.core.nws.families import build_architecture  # noqa: E402
from uzen.core.nws.tensors import WeightPackage  # noqa: E402
from uzen.core.wie.importer import ModelImporter  # noqa: E402
from uzen.errors import InvalidOperationParameters  # noqa: E402
from uzen.runtimes.pytorch_runtime import PyTorchRuntime  # noqa: E402
from uzen.runtimes.torch_executor import execute_torch, is_available  # noqa: E402

FAMILY_SPECS = {
    "linear_residual": dict(input_dim=6, hidden_dim=10, out_dim=4),
    "gated_mlp": dict(input_dim=6, hidden_dim=10, out_dim=4),
    "conv_residual": dict(channels=2, height=8, width=8, num_classes=3),
    "transformer_encoder": dict(vocab_size=16, d_model=8, n_heads=2, n_layers=2, d_ff=16, max_seq_len=8, num_classes=3),
    "albert": dict(vocab_size=16, d_model=8, n_heads=2, n_layers=3, d_ff=16, max_seq_len=8, num_classes=3),
    "bert": dict(vocab_size=16, d_model=8, n_heads=2, n_layers=2, d_ff=16, max_seq_len=8, num_classes=3),
    "vit": dict(in_channels=1, height=8, width=8, patch=2, d_model=8, n_heads=2, n_layers=1, d_ff=16, num_classes=3),
    "set_transformer": dict(vocab_size=16, d_model=8, n_heads=2, n_layers=1, d_ff=16, max_seq_len=8, num_classes=3),
    "siamese": dict(input_dim=6, hidden_dim=10, num_classes=2),
    "retrieval": dict(vocab_size=16, d_model=8, hidden_dim=16, embed_dim=4, max_seq_len=8),
    "perceiver": dict(vocab_size=16, d_model=8, n_heads=2, latent_count=3, d_ff=16, max_seq_len=8, num_classes=3),
    "seq2seq": dict(vocab_size=16, d_model=8, n_heads=2, n_layers=1, d_ff=16, max_seq_len=8),
    "transformer_decoder": dict(
        d_model=16, n_heads=4, n_layers=2, d_ff=32, vocab_size=24, max_seq_len=12
    ),
    "mlp": dict(input_dim=8, hidden_dims=[16, 12], output_dim=4),
    "cnn": dict(in_channels=1, height=8, width=8, channels=[4], num_classes=3),
    "recurrent": dict(cell="gru", vocab_size=10, d_model=4, hidden_size=8, num_classes=3),
    "autoencoder": dict(input_dim=10, latent_dim=3),
    "moe": dict(vocab_size=16, d_model=8, n_experts=4, top_k=2, d_ff=16, num_classes=3),
    "state_space": dict(vocab_size=16, d_model=8, state_dim=4, num_classes=3),
    "gnn": dict(in_features=5, hidden_dim=6, num_classes=3, n_layers=2),
    "vae": dict(input_dim=6, latent_dim=3, hidden_dim=8),
    "diffusion": dict(input_dim=6, hidden_dim=12, alpha_bar=0.6, noise_seed=4),
    "multimodal": dict(
        in_channels=1, height=8, width=8, vocab_size=12, d_model=6, num_classes=3
    ),
    "hybrid": dict(
        in_channels=1, height=8, width=8, channels=[3], hidden_size=5, num_classes=3
    ),
}


@pytest.mark.parametrize("family,spec", FAMILY_SPECS.items())
def test_torch_cpu_matches_numpy_for_every_family(family, spec):
    definition = build_architecture(family, spec)
    package = definition.build_package(seed=7)
    inputs = definition.sample_inputs(batch=2, seq=5, seed=7)
    expected = execute(definition.graph, inputs, package).outputs

    got = execute_torch(definition.graph, inputs, package)
    assert got.num_ops == len(definition.graph.nodes)
    assert set(got.outputs) == set(expected)
    for name, value in expected.items():
        np.testing.assert_allclose(
            got.outputs[name], value, rtol=1e-5, atol=1e-6, err_msg=f"{family}:{name}"
        )


def test_torch_windowed_alibi_attention_matches_numpy():
    rng = np.random.default_rng(21)
    values = {
        "q": rng.standard_normal((1, 2, 4, 2)).astype(np.float32),
        "k": rng.standard_normal((1, 2, 4, 2)).astype(np.float32),
        "v": rng.standard_normal((1, 2, 4, 2)).astype(np.float32),
    }
    g = Graph(
        name="windowed-alibi",
        inputs={
            "q": TensorType.of((1, 2, 4, 2)),
            "k": TensorType.of((1, 2, 4, 2)),
            "v": TensorType.of((1, 2, 4, 2)),
        },
    )
    g.add_node(
        "attn",
        "attention",
        {"q": Ref("input:q"), "k": Ref("input:k"), "v": Ref("input:v")},
        {"causal": True, "window": 2, "alibi_slopes": [0.25, 0.5], "scale": "auto"},
    )
    g.set_output("y", Ref("attn"))
    package = WeightPackage.from_arrays({})
    expected = execute(g, values, package).outputs["y"]
    got = execute_torch(g, values, package).outputs["y"]
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-7)


def test_torch_clip_matches_numpy():
    g = Graph(name="clip", inputs={"x": TensorType.of((2, 3, 4))})
    g.add_node("clip", "clip", {"x": Ref("input:x")}, {"min": -0.5, "max": 0.75})
    g.set_output("y", Ref("clip"))
    x = np.linspace(-2.0, 2.0, 24, dtype=np.float32).reshape(2, 3, 4)
    package = WeightPackage.from_arrays({})
    expected = execute(g, {"x": x}, package).outputs["y"]
    got = execute_torch(g, {"x": x}, package).outputs["y"]
    np.testing.assert_allclose(got, expected, rtol=0, atol=0)


def test_runtime_prefers_torch_and_can_force_the_reference():
    definition = build_architecture("mlp", dict(input_dim=4, hidden_dims=[6], output_dim=2))
    package = definition.build_package(seed=1)
    inputs = definition.sample_inputs(batch=2, seed=1)
    runtime = PyTorchRuntime(device="cpu")

    torch_outputs = runtime.execute(definition.graph, inputs, package)
    assert runtime.last_backend == "torch"
    assert runtime.last_fallback_reason is None

    numpy_outputs = runtime.execute(definition.graph, inputs, package, backend="numpy")
    assert runtime.last_backend == "numpy"
    for name in torch_outputs:
        np.testing.assert_allclose(torch_outputs[name], numpy_outputs[name], rtol=1e-6, atol=1e-7)


def test_torch_backend_can_be_requested_explicitly():
    definition = build_architecture("transformer_decoder", FAMILY_SPECS["transformer_decoder"])
    package = definition.build_package(seed=2)
    inputs = definition.sample_inputs(batch=2, seq=5, seed=2)
    runtime = PyTorchRuntime(device="cpu")

    got = runtime.execute(definition.graph, inputs, package, backend="torch")
    assert runtime.last_backend == "torch"
    expected = execute(definition.graph, inputs, package).outputs
    for name in expected:
        np.testing.assert_allclose(got[name], expected[name], rtol=1e-5, atol=1e-6)


def test_torch_gelu_exact_matches_numpy():
    graph = Graph(name="exact-gelu", inputs={"x": TensorType.of((1, 5))})
    graph.add_node("act", "gelu_exact", {"x": Ref("input:x")})
    graph.set_output("y", Ref("act"))
    package = WeightPackage.from_arrays({})
    x = np.array([[-3.0, -1.0, 0.0, 1.0, 3.0]], dtype=np.float32)
    expected = execute(graph, {"x": x}, package).outputs["y"]
    got = execute_torch(graph, {"x": x}, package).outputs["y"]
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-7)


def test_torch_neox_graph_matches_numpy(tmp_path):
    root, _arrays = _write_neox(tmp_path / "neox-torch")
    result = ModelImporter().import_source(root)
    assert result.reconstruction is not None
    graph = result.reconstruction.definition.graph
    package = result.artifact_package
    tokens = np.array([[0, 1, 2, 3]], dtype=np.int64)
    positions = np.array([[0, 1, 2, 3]], dtype=np.int64)
    inputs = {"tokens": tokens, "positions": positions}

    expected = execute(graph, inputs, package).outputs
    got = execute_torch(graph, inputs, package).outputs
    for name in expected:
        np.testing.assert_allclose(
            got[name], expected[name], rtol=1e-5, atol=1e-6, err_msg=name
        )


def test_torch_pos_embedding_honours_the_offset_parameter():
    graph = Graph(name="offset", inputs={"tokens": TensorType.of((1, 3), "int64")})
    graph.add_node(
        "embed", "embedding", {"x": Ref("input:tokens")}, {"weight": "param:embed"}
    )
    graph.add_node(
        "pos",
        "pos_embedding",
        {"x": Ref("embed")},
        {"weight": "param:pos", "offset": 2},
    )
    graph.set_output("y", Ref("pos"))
    embed = np.arange(20, dtype=np.float32).reshape(5, 4)
    pos = np.arange(24, dtype=np.float32).reshape(6, 4)
    package = WeightPackage.from_arrays({"embed": embed, "pos": pos})
    inputs = {"tokens": np.array([[0, 2, 4]], dtype=np.int64)}

    expected = embed[inputs["tokens"]] + pos[2:5]
    np.testing.assert_array_equal(execute(graph, inputs, package).outputs["y"], expected)
    np.testing.assert_allclose(
        execute_torch(graph, inputs, package).outputs["y"], expected, rtol=0, atol=0
    )


def test_capture_returns_intermediates():
    definition = build_architecture("mlp", dict(input_dim=4, hidden_dims=[6], output_dim=2))
    package = definition.build_package(seed=3)
    inputs = definition.sample_inputs(batch=3, seed=3)
    first = definition.graph.topological_order()[0]

    result = execute_torch(definition.graph, inputs, package, capture=[first])
    assert set(result.intermediates) == {first}
    assert result.intermediates[first].shape[0] == 3


def test_invalid_backend_is_rejected():
    definition = build_architecture("mlp", dict(input_dim=3, hidden_dims=[4], output_dim=2))
    package = definition.build_package(seed=4)
    inputs = definition.sample_inputs(batch=2, seed=4)
    with pytest.raises(InvalidOperationParameters):
        PyTorchRuntime(device="cpu").execute(
            definition.graph, inputs, package, backend="metal"
        )


def test_torch_rope_matches_numpy_reference():
    rng = np.random.default_rng(21)
    x = rng.standard_normal((2, 3, 4, 6)).astype(np.float32)
    positions = np.array([[0, 2, 3, 5]], dtype=np.int64)
    graph = Graph(
        name="rope-parity",
        inputs={"x": TensorType.of((2, 3, 4, 6)), "positions": TensorType.of((1, 4), "int64")},
    )
    graph.add_node(
        "rope",
        "rope",
        {"x": Ref("input:x"), "positions": Ref("input:positions")},
        {"theta": 10000.0},
    )
    graph.set_output("y", Ref("rope"))
    package = WeightPackage.from_arrays({})
    inputs = {"x": x, "positions": positions}

    expected = execute(graph, inputs, package).outputs["y"]
    got = execute_torch(graph, inputs, package).outputs["y"]
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_torch_interleaved_partial_rope_matches_numpy_reference():
    rng = np.random.default_rng(23)
    x = rng.standard_normal((2, 3, 4, 8)).astype(np.float32)
    positions = np.array([[0, 2, 3, 5]], dtype=np.int64)
    graph = Graph(
        name="interleaved-rope-parity",
        inputs={"x": TensorType.of((2, 3, 4, 8)), "positions": TensorType.of((1, 4), "int64")},
    )
    graph.add_node(
        "rope",
        "rope",
        {"x": Ref("input:x"), "positions": Ref("input:positions")},
        {"theta": 10000.0, "rotary_dim": 4, "interleaved": True},
    )
    graph.set_output("y", Ref("rope"))
    package = WeightPackage.from_arrays({})
    inputs = {"x": x, "positions": positions}

    expected = execute(graph, inputs, package).outputs["y"]
    got = execute_torch(graph, inputs, package).outputs["y"]
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-7)


def test_torch_repeat_kv_matches_numpy_reference():
    rng = np.random.default_rng(22)
    x = rng.standard_normal((2, 2, 4, 5)).astype(np.float32)
    graph = Graph(name="repeat-kv-parity", inputs={"x": TensorType.of((2, 2, 4, 5))})
    graph.add_node("repeat_kv", "repeat_kv", {"x": Ref("input:x")}, {"n_heads": 4})
    graph.set_output("y", Ref("repeat_kv"))
    package = WeightPackage.from_arrays({})

    expected = execute(graph, {"x": x}, package).outputs["y"]
    got = execute_torch(graph, {"x": x}, package).outputs["y"]
    np.testing.assert_array_equal(got, expected)


def test_torch_cpu_llama_graph_uses_native_kernels_and_matches_numpy(tmp_path):
    root, _arrays = _write_llama(tmp_path / "torch-llama")
    result = ModelImporter().import_source(root)
    assert result.reconstruction is not None
    definition = result.reconstruction.definition
    inputs = {
        "tokens": np.array([[0, 1, 2, 3]], dtype=np.int64),
        "positions": np.arange(4, dtype=np.int64).reshape(1, 4),
    }
    expected = execute(definition.graph, inputs, result.artifact_package).outputs
    runtime = PyTorchRuntime(device="cpu")
    got = runtime.execute(
        definition.graph, inputs, result.artifact_package, backend="torch"
    )

    assert runtime.last_backend == "torch"
    assert runtime.last_fallback_reason is None
    for name, value in expected.items():
        np.testing.assert_allclose(got[name], value, rtol=1e-5, atol=1e-6, err_msg=name)


def test_executor_reports_availability():
    assert is_available() is True
