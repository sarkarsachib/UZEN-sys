"""GPT-NeoX reconstruction without network access.

The fixture exercises the two details that distinguish GPT-NeoX from the other
adapters: the per-head-interleaved fused QKV layout and a partial rotary width.
It also carries the attention/MLP biases used by the safetensors fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uzen.core.wie.importer import ModelImporter
from uzen.infrastructure.storage.safetensors_io import write_safetensors


def _write_neox(
    root: Path,
    *,
    parallel_residual: bool = True,
    hidden_act: str = "gelu_fast",
    attention_bias: bool = True,
) -> tuple[Path, dict[str, np.ndarray]]:
    root.mkdir(parents=True, exist_ok=True)
    hidden = 16
    n_heads = 4
    n_layers = 2
    ffn = 8
    vocab = 32
    max_seq = 8
    rng = np.random.default_rng(23)
    arrays: dict[str, np.ndarray] = {
        "gpt_neox.embed_in.weight": rng.normal(
            scale=0.02, size=(vocab, hidden)
        ).astype(np.float32),
        "embed_out.weight": rng.normal(scale=0.02, size=(vocab, hidden)).astype(
            np.float32
        ),
        "gpt_neox.final_layer_norm.weight": np.ones((hidden,), dtype=np.float32),
        "gpt_neox.final_layer_norm.bias": np.zeros((hidden,), dtype=np.float32),
    }
    for layer in range(n_layers):
        p = f"gpt_neox.layers.{layer}"
        arrays.update(
            {
                f"{p}.input_layernorm.weight": np.ones((hidden,), dtype=np.float32),
                f"{p}.input_layernorm.bias": np.zeros((hidden,), dtype=np.float32),
                f"{p}.post_attention_layernorm.weight": np.ones(
                    (hidden,), dtype=np.float32
                ),
                f"{p}.post_attention_layernorm.bias": np.zeros(
                    (hidden,), dtype=np.float32
                ),
                f"{p}.attention.query_key_value.weight": rng.normal(
                    scale=0.02, size=(3 * hidden, hidden)
                ).astype(np.float32),
                f"{p}.attention.dense.weight": rng.normal(
                    scale=0.02, size=(hidden, hidden)
                ).astype(np.float32),
                f"{p}.mlp.dense_h_to_4h.weight": rng.normal(
                    scale=0.02, size=(ffn, hidden)
                ).astype(np.float32),
                f"{p}.mlp.dense_h_to_4h.bias": rng.normal(
                    scale=0.02, size=(ffn,)
                ).astype(np.float32),
                f"{p}.mlp.dense_4h_to_h.weight": rng.normal(
                    scale=0.02, size=(hidden, ffn)
                ).astype(np.float32),
                f"{p}.mlp.dense_4h_to_h.bias": rng.normal(
                    scale=0.02, size=(hidden,)
                ).astype(np.float32),
            }
        )
        if attention_bias:
            arrays[f"{p}.attention.query_key_value.bias"] = rng.normal(
                scale=0.02, size=(3 * hidden,)
            ).astype(np.float32)
            arrays[f"{p}.attention.dense.bias"] = rng.normal(
                scale=0.02, size=(hidden,)
            ).astype(np.float32)
    write_safetensors(root / "model.safetensors", arrays)
    config = {
        "architectures": ["GPTNeoXForCausalLM"],
        "model_type": "gpt_neox",
        "hidden_size": hidden,
        "num_attention_heads": n_heads,
        "num_hidden_layers": n_layers,
        "intermediate_size": ffn,
        "vocab_size": vocab,
        "max_position_embeddings": max_seq,
        "hidden_act": hidden_act,
        "layer_norm_eps": 1e-5,
        "partial_rotary_factor": 0.5,
        "rope_theta": 10000.0,
        "rope_scaling": None,
        "use_parallel_residual": parallel_residual,
        "attention_bias": attention_bias,
        "tie_word_embeddings": False,
        "attention_dropout": 0.1,
        "hidden_dropout": 0.1,
        "type_vocab_size": 8,
    }
    (root / "config.json").write_text(json.dumps(config))
    return root, arrays


def test_neox_reconstruction_validates_and_executes(tmp_path):
    root, _arrays = _write_neox(tmp_path / "neox")
    result = ModelImporter().import_source(root)

    assert result.ok
    assert result.report.compatibility.reconstruction == "validated"
    assert result.report.compatibility.execution == "passed"
    assert result.report.compatibility.overall == "validated"
    assert result.report.reconstruction["missing"] == []
    assert result.report.reconstruction["shape_errors"] == []
    assert result.report.reconstruction["extra"] == []
    assert result.report.reconstruction["execution"]["outputs"]["logits"]["shape"] == [
        1,
        4,
        32,
    ]

    spec = result.report.reconstruction["spec"]
    assert spec["rotary_dim"] == 2
    assert spec["parallel_residual"] is True
    assert spec["attention_bias"] is True
    assert spec["activation"] == "gelu"
    assert spec["tie_embeddings"] is False

    graph = result.report.reconstruction["graph"]
    assert graph["name"] == "gpt_neox_L2"
    nodes = {node["id"]: node for node in graph["nodes"]}
    assert nodes["l0_split"]["params"] == {
        "n_heads": 4,
        "layout": "interleaved",
    }
    assert nodes["l0_q_rope"]["params"] == {
        "theta": 10000.0,
        "rotary_dim": 2,
    }
    assert sum(node["op"] == "rope" for node in graph["nodes"]) == 4
    assert sum(node["op"] == "gelu" for node in graph["nodes"]) == 2

    mapping = {m["target"]: m for m in result.report.reconstruction["tensor_mapping"]}
    assert mapping["layers.0.qkv.weight"]["source_shape"] == [48, 16]
    assert mapping["layers.0.qkv.bias"]["source_shape"] == [48]
    assert mapping["layers.0.qkv.weight"]["note"] == "per-head-interleaved QKV layout"
    assert mapping["lm_head.weight"]["source"] == "embed_out.weight"


def test_neox_sequential_residual_variant_is_explicit(tmp_path):
    root, _arrays = _write_neox(
        tmp_path / "neox-sequential",
        parallel_residual=False,
    )
    result = ModelImporter().import_source(root)

    assert result.ok
    assert result.report.compatibility.reconstruction == "validated"
    assert result.report.reconstruction["spec"]["parallel_residual"] is False
    graph = result.report.reconstruction["graph"]
    res1 = next(node for node in graph["nodes"] if node["id"] == "l0_res1")
    ln2 = next(node for node in graph["nodes"] if node["id"] == "l0_ln2")
    assert res1["inputs"]["a"].startswith("input:") or res1["inputs"]["a"] == "embed"
    assert ln2["inputs"]["x"] == "l0_res1"


def test_neox_rejects_exact_gelu_instead_of_silently_approximating(tmp_path):
    root, _arrays = _write_neox(
        tmp_path / "neox-exact-gelu",
        hidden_act="gelu",
    )
    result = ModelImporter().import_source(root)

    assert result.ok
    assert result.report.compatibility.reconstruction == "unsupported"
    assert result.report.compatibility.execution == "not_run"
    assert result.report.reconstruction["graph"] is None
    assert result.reconstruction is not None
    assert result.reconstruction.package is None
    assert any(
        "tanh" in reason for reason in result.report.compatibility.reasons
    )
