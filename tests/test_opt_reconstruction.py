"""OPT reconstruction without network access.

The fixture uses OPT's real tensor layout: a position table with the +2 offset,
biased LayerNorm/Q/K/V/O/MLP tensors, and a tied LM head that is not stored as
its own tensor in the checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from uzen.core.wie.importer import ModelImporter
from uzen.infrastructure.storage.safetensors_io import write_safetensors


def _write_opt(
    root: Path,
    *,
    word_embed_proj_dim: int | None = None,
    do_layer_norm_before: bool = True,
) -> tuple[Path, dict[str, np.ndarray]]:
    root.mkdir(parents=True, exist_ok=True)
    hidden = 16
    n_heads = 4
    n_layers = 2
    ffn = 8
    vocab = 32
    max_seq = 8
    rng = np.random.default_rng(17)
    arrays: dict[str, np.ndarray] = {
        "model.decoder.embed_tokens.weight": rng.normal(
            scale=0.02, size=(vocab, hidden)
        ).astype(np.float32),
        "model.decoder.embed_positions.weight": rng.normal(
            scale=0.02, size=(max_seq + 2, hidden)
        ).astype(np.float32),
        "model.decoder.final_layer_norm.weight": np.ones((hidden,), dtype=np.float32),
        "model.decoder.final_layer_norm.bias": np.zeros((hidden,), dtype=np.float32),
    }
    for layer in range(n_layers):
        p = f"model.decoder.layers.{layer}"
        arrays.update(
            {
                f"{p}.self_attn_layer_norm.weight": np.ones(
                    (hidden,), dtype=np.float32
                ),
                f"{p}.self_attn_layer_norm.bias": np.zeros(
                    (hidden,), dtype=np.float32
                ),
                f"{p}.self_attn.q_proj.weight": rng.normal(
                    scale=0.02, size=(hidden, hidden)
                ).astype(np.float32),
                f"{p}.self_attn.q_proj.bias": rng.normal(
                    scale=0.02, size=(hidden,)
                ).astype(np.float32),
                f"{p}.self_attn.k_proj.weight": rng.normal(
                    scale=0.02, size=(hidden, hidden)
                ).astype(np.float32),
                f"{p}.self_attn.k_proj.bias": rng.normal(
                    scale=0.02, size=(hidden,)
                ).astype(np.float32),
                f"{p}.self_attn.v_proj.weight": rng.normal(
                    scale=0.02, size=(hidden, hidden)
                ).astype(np.float32),
                f"{p}.self_attn.v_proj.bias": rng.normal(
                    scale=0.02, size=(hidden,)
                ).astype(np.float32),
                f"{p}.self_attn.out_proj.weight": rng.normal(
                    scale=0.02, size=(hidden, hidden)
                ).astype(np.float32),
                f"{p}.self_attn.out_proj.bias": rng.normal(
                    scale=0.02, size=(hidden,)
                ).astype(np.float32),
                f"{p}.final_layer_norm.weight": np.ones(
                    (hidden,), dtype=np.float32
                ),
                f"{p}.final_layer_norm.bias": np.zeros(
                    (hidden,), dtype=np.float32
                ),
                f"{p}.fc1.weight": rng.normal(scale=0.02, size=(ffn, hidden)).astype(
                    np.float32
                ),
                f"{p}.fc1.bias": rng.normal(scale=0.02, size=(ffn,)).astype(
                    np.float32
                ),
                f"{p}.fc2.weight": rng.normal(scale=0.02, size=(hidden, ffn)).astype(
                    np.float32
                ),
                f"{p}.fc2.bias": rng.normal(scale=0.02, size=(hidden,)).astype(
                    np.float32
                ),
            }
        )
    write_safetensors(root / "model.safetensors", arrays)
    config = {
        "architectures": ["OPTForCausalLM"],
        "model_type": "opt",
        "hidden_size": hidden,
        "num_attention_heads": n_heads,
        "num_hidden_layers": n_layers,
        "ffn_dim": ffn,
        "vocab_size": vocab,
        "max_position_embeddings": max_seq,
        "activation_function": "relu",
        "layer_norm_eps": 1e-5,
        "word_embed_proj_dim": word_embed_proj_dim or hidden,
        "do_layer_norm_before": do_layer_norm_before,
        "_remove_final_layer_norm": False,
        "attention_dropout": 0.1,
        "tie_word_embeddings": True,
    }
    (root / "config.json").write_text(json.dumps(config))
    return root, arrays


def test_opt_pytorch_bin_fallback_uses_weights_only_loader(tmp_path):
    torch = pytest.importorskip("torch")
    root, arrays = _write_opt(tmp_path / "opt-bin")
    (root / "model.safetensors").unlink()
    torch.save(
        {name: torch.from_numpy(value) for name, value in arrays.items()},
        root / "pytorch_model.bin",
    )

    result = ModelImporter().import_source(root)

    assert result.ok
    assert result.report.compatibility.overall == "validated"
    assert result.report.compatibility.execution == "passed"
    assert result.report.reconstruction["execution"]["outputs"]["logits"]["shape"] == [
        1,
        4,
        32,
    ]
    assert any(
        "pytorch_model.bin" in warning and "weights_only=True" in warning
        for warning in result.report.warnings
    )
    assert {item["path"] for item in result.report.files} == {
        "config.json",
        "pytorch_model.bin",
    }


def test_opt_reconstruction_validates_executes_and_ties_lm_head(tmp_path):
    root, _arrays = _write_opt(tmp_path / "opt")
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
    assert spec["position_offset"] == 2
    assert spec["pos_table_size"] == 10
    assert spec["d_ff"] == 8
    assert spec["tie_embeddings"] is True

    graph = result.report.reconstruction["graph"]
    assert graph["name"] == "opt_L2"
    assert any(
        node["op"] == "pos_embedding" and node["params"]["offset"] == 2
        for node in graph["nodes"]
    )
    assert sum(node["op"] == "relu" for node in graph["nodes"]) == 2

    assert result.artifact_package is not None
    assert "lm_head.weight" not in result.package
    np.testing.assert_array_equal(
        result.artifact_package.get("lm_head.weight"),
        result.artifact_package.get("embed.weight"),
    )
    mapping = {m["target"]: m for m in result.report.reconstruction["tensor_mapping"]}
    assert mapping["layers.0.q_proj.weight"]["transform"] == "identity"
    assert mapping["lm_head.weight"]["transform"] == "tied_embedding"
    assert mapping["pos_embed.weight"]["source_shape"] == [10, 16]


def test_opt_rejects_word_embedding_projection_instead_of_guessing(tmp_path):
    root, _arrays = _write_opt(
        tmp_path / "opt-projected",
        word_embed_proj_dim=32,
    )
    result = ModelImporter().import_source(root)

    assert result.ok
    assert result.report.compatibility.reconstruction == "unsupported"
    assert result.report.compatibility.execution == "not_run"
    assert result.report.reconstruction["graph"] is None
    assert result.reconstruction is not None
    assert result.reconstruction.package is None
    assert any(
        "word_embed_proj_dim" in reason for reason in result.report.compatibility.reasons
    )


def test_opt_rejects_post_layer_norm_variant_instead_of_guessing(tmp_path):
    root, _arrays = _write_opt(
        tmp_path / "opt-post-ln",
        do_layer_norm_before=False,
    )
    result = ModelImporter().import_source(root)

    assert result.ok
    assert result.report.compatibility.reconstruction == "unsupported"
    assert any(
        "do_layer_norm_before" in reason
        for reason in result.report.compatibility.reasons
    )
