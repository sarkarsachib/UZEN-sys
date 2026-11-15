"""Llama/RoPE/GQA reconstruction without network access.

The fixture deliberately uses fewer KV heads than query heads so the
``repeat_kv`` path is exercised, not merely present in an unused branch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uzen.core.nir.executor import KVCache, execute
from uzen.core.wie.importer import ModelImporter
from uzen.infrastructure.storage.safetensors_io import write_safetensors


def _write_llama(root: Path, *, rope_scaling: dict | None = None) -> tuple[Path, dict[str, np.ndarray]]:
    root.mkdir(parents=True, exist_ok=True)
    hidden = 16
    n_heads = 4
    n_kv_heads = 2
    head_dim = 4
    n_layers = 1
    intermediate = 32
    vocab = 32
    max_seq = 8
    rng = np.random.default_rng(11)
    arrays: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": rng.normal(scale=0.02, size=(vocab, hidden)).astype(np.float32),
        "model.norm.weight": np.ones((hidden,), dtype=np.float32),
        "lm_head.weight": rng.normal(scale=0.02, size=(vocab, hidden)).astype(np.float32),
    }
    for layer in range(n_layers):
        p = f"model.layers.{layer}"
        arrays.update(
            {
                f"{p}.input_layernorm.weight": np.ones((hidden,), dtype=np.float32),
                f"{p}.post_attention_layernorm.weight": np.ones((hidden,), dtype=np.float32),
                f"{p}.self_attn.q_proj.weight": rng.normal(
                    scale=0.02, size=(n_heads * head_dim, hidden)
                ).astype(np.float32),
                f"{p}.self_attn.k_proj.weight": rng.normal(
                    scale=0.02, size=(n_kv_heads * head_dim, hidden)
                ).astype(np.float32),
                f"{p}.self_attn.v_proj.weight": rng.normal(
                    scale=0.02, size=(n_kv_heads * head_dim, hidden)
                ).astype(np.float32),
                f"{p}.self_attn.o_proj.weight": rng.normal(
                    scale=0.02, size=(hidden, n_heads * head_dim)
                ).astype(np.float32),
                f"{p}.mlp.gate_proj.weight": rng.normal(
                    scale=0.02, size=(intermediate, hidden)
                ).astype(np.float32),
                f"{p}.mlp.up_proj.weight": rng.normal(
                    scale=0.02, size=(intermediate, hidden)
                ).astype(np.float32),
                f"{p}.mlp.down_proj.weight": rng.normal(
                    scale=0.02, size=(hidden, intermediate)
                ).astype(np.float32),
            }
        )
    write_safetensors(root / "model.safetensors", arrays)
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": hidden,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv_heads,
        "head_dim": head_dim,
        "num_hidden_layers": n_layers,
        "intermediate_size": intermediate,
        "vocab_size": vocab,
        "max_position_embeddings": max_seq,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "hidden_act": "silu",
        "attention_bias": False,
        "mlp_bias": False,
        "tie_word_embeddings": False,
        "rope_scaling": rope_scaling,
    }
    (root / "config.json").write_text(json.dumps(config))
    return root, arrays


def test_llama_kv_cache_decode_matches_full_forward(tmp_path):
    root, _arrays = _write_llama(tmp_path / "llama-kv-cache")
    result = ModelImporter().import_source(root)
    assert result.reconstruction is not None
    graph = result.reconstruction.definition.graph
    package = result.artifact_package
    tokens = np.array([[0, 1, 2, 3, 4]], dtype=np.int64)
    positions = np.array([[0, 1, 2, 3, 4]], dtype=np.int64)

    full = execute(graph, {"tokens": tokens, "positions": positions}, package).output("logits")
    cache = KVCache()
    execute(
        graph,
        {"tokens": tokens[:, :-1], "positions": positions[:, :-1]},
        package,
        cache=cache,
    )
    decode = execute(
        graph,
        {"tokens": tokens[:, -1:], "positions": positions[:, -1:]},
        package,
        cache=cache,
    ).output("logits")
    assert len(cache) == 1
    np.testing.assert_allclose(decode[0, 0], full[0, -1], rtol=1e-6, atol=1e-7)


def test_llama_gqa_reconstruction_validates_and_executes(tmp_path):
    root, _arrays = _write_llama(tmp_path / "llama-gqa")
    result = ModelImporter().import_source(root)

    assert result.ok
    assert result.report.compatibility.reconstruction == "validated"
    assert result.report.compatibility.execution == "passed"
    assert result.report.compatibility.overall == "validated"
    assert result.report.reconstruction["missing"] == []
    assert result.report.reconstruction["shape_errors"] == []
    assert result.report.reconstruction["extra"] == []
    assert result.report.reconstruction["spec"]["n_kv_heads"] == 2
    assert result.report.reconstruction["spec"]["rope_theta"] == 10000.0
    assert result.report.reconstruction["execution"]["outputs"]["logits"]["shape"] == [1, 4, 32]

    graph = result.report.reconstruction["graph"]
    assert graph["name"] == "llama_L1"
    operators = [node["op"] for node in graph["nodes"]]
    assert "rope" in operators
    assert "repeat_kv" in operators
    assert "silu" in operators

    mapping = {m["target"]: m for m in result.report.reconstruction["tensor_mapping"]}
    assert mapping["layers.0.k_proj.weight"]["source_shape"] == [8, 16]
    assert mapping["layers.0.q_proj.weight"]["source_shape"] == [16, 16]
    assert mapping["layers.0.k_proj.weight"]["transform"] == "identity"


def test_llama_rejects_rope_scaling_instead_of_guessing(tmp_path):
    root, _arrays = _write_llama(
        tmp_path / "llama-rope-scaling",
        rope_scaling={"type": "linear", "factor": 2.0},
    )
    result = ModelImporter().import_source(root)

    assert result.ok
    assert result.report.compatibility.reconstruction == "unsupported"
    assert result.report.compatibility.execution == "not_run"
    assert result.report.reconstruction["graph"] is None
    assert result.reconstruction is not None
    assert result.reconstruction.package is None
    assert any("rope_scaling" in reason for reason in result.report.compatibility.reasons)
