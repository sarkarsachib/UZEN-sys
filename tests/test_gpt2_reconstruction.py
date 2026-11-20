"""GPT-2 architecture reconstruction without network access."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uzen.core.wie.importer import ModelImporter
from uzen.infrastructure.storage.safetensors_io import write_safetensors
from uzen.jobs.service import JobService


def _write_gpt2(root: Path) -> tuple[Path, dict[str, np.ndarray]]:
    root.mkdir(parents=True, exist_ok=True)
    d_model, n_heads, n_layers, vocab, max_seq, d_ff = 8, 2, 1, 16, 8, 12
    rng = np.random.default_rng(7)
    arrays: dict[str, np.ndarray] = {
        "transformer.wte.weight": rng.normal(scale=0.02, size=(vocab, d_model)).astype(np.float32),
        "transformer.wpe.weight": rng.normal(scale=0.02, size=(max_seq, d_model)).astype(np.float32),
        "transformer.ln_f.weight": np.ones((d_model,), dtype=np.float32),
        "transformer.ln_f.bias": np.zeros((d_model,), dtype=np.float32),
    }
    for layer in range(n_layers):
        p = f"transformer.h.{layer}"
        arrays.update(
            {
                f"{p}.ln_1.weight": np.ones((d_model,), dtype=np.float32),
                f"{p}.ln_1.bias": np.zeros((d_model,), dtype=np.float32),
                f"{p}.attn.c_attn.weight": rng.normal(scale=0.02, size=(d_model, 3 * d_model)).astype(np.float32),
                f"{p}.attn.c_attn.bias": np.zeros((3 * d_model,), dtype=np.float32),
                f"{p}.attn.c_proj.weight": rng.normal(scale=0.02, size=(d_model, d_model)).astype(np.float32),
                f"{p}.attn.c_proj.bias": np.zeros((d_model,), dtype=np.float32),
                f"{p}.ln_2.weight": np.ones((d_model,), dtype=np.float32),
                f"{p}.ln_2.bias": np.zeros((d_model,), dtype=np.float32),
                f"{p}.mlp.c_fc.weight": rng.normal(scale=0.02, size=(d_model, d_ff)).astype(np.float32),
                f"{p}.mlp.c_fc.bias": np.zeros((d_ff,), dtype=np.float32),
                f"{p}.mlp.c_proj.weight": rng.normal(scale=0.02, size=(d_ff, d_model)).astype(np.float32),
                f"{p}.mlp.c_proj.bias": np.zeros((d_model,), dtype=np.float32),
            }
        )
    write_safetensors(root / "model.safetensors", arrays)
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["GPT2LMHeadModel"],
                "model_type": "gpt2",
                "n_embd": d_model,
                "n_head": n_heads,
                "n_layer": n_layers,
                "vocab_size": vocab,
                "n_positions": max_seq,
                "n_inner": d_ff,
                "activation_function": "gelu_new",
                "layer_norm_epsilon": 1e-5,
                "tie_word_embeddings": True,
            }
        )
    )
    (root / "tokenizer.json").write_text(
        json.dumps({"model": {"type": "BPE", "vocab": {str(i): i for i in range(vocab)}}})
    )
    return root, arrays


def test_gpt2_reconstruction_validates_graph_and_executes(tmp_path):
    root, arrays = _write_gpt2(tmp_path / "gpt2")
    result = ModelImporter().import_source(root)

    assert result.ok
    assert result.reconstruction is not None
    assert result.report.compatibility.reconstruction == "validated"
    assert result.report.compatibility.execution == "passed"
    assert result.report.compatibility.overall == "validated"
    assert result.report.understanding["graph_reconstructed"] is True
    assert result.report.understanding["execution_validated"] is True
    assert result.report.reconstruction["missing"] == []
    assert result.report.reconstruction["shape_errors"] == []
    assert result.report.reconstruction["extra"] == []
    assert result.report.progress
    assert result.artifact_package is not None

    canonical = result.artifact_package
    mapping = {m["target"]: m for m in result.report.reconstruction["tensor_mapping"]}
    assert mapping["layers.0.qkv.weight"]["transform"] == "transpose"
    assert np.array_equal(
        canonical.get("layers.0.qkv.weight"),
        arrays["transformer.h.0.attn.c_attn.weight"].T,
    )
    assert np.array_equal(
        canonical.get("layers.0.mlp_down.weight"),
        arrays["transformer.h.0.mlp.c_proj.weight"].T,
    )
    execution = result.report.reconstruction["execution"]
    assert execution["status"] == "passed"
    assert execution["outputs"]["logits"]["shape"] == [1, 4, 16]


def test_gpt2_prefixless_openai_safetensors_layout_is_normalized(tmp_path):
    root, arrays = _write_gpt2(tmp_path / "gpt2-prefixless")
    prefixless = {
        key.removeprefix("transformer."): value for key, value in arrays.items()
    }
    prefixless["h.0.attn.bias"] = np.zeros((1, 1, 8, 8), dtype=np.float32)
    write_safetensors(root / "model.safetensors", prefixless)

    result = ModelImporter().import_source(root)

    assert result.ok
    assert result.report.compatibility.reconstruction == "validated"
    assert result.report.compatibility.execution == "passed"
    assert any(
        "prefixless GPT-2" in warning
        for warning in result.report.reconstruction["warnings"]
    )
    mapping = {m["target"]: m for m in result.report.reconstruction["tensor_mapping"]}
    assert mapping["embed.weight"]["source"] == "transformer.wte.weight"
    assert mapping["layers.0.qkv.weight"]["source"] == (
        "transformer.h.0.attn.c_attn.weight"
    )
    ignored = [
        item
        for item in result.report.reconstruction["tensor_mapping"]
        if item["status"] == "ignored"
    ]
    assert [item["source"] for item in ignored] == ["transformer.h.0.attn.bias"]
    assert ignored[0]["target"] is None
    assert result.report.reconstruction["extra"] == []


def test_gpt2_service_stores_graph_and_canonical_package(tmp_path):
    root, _arrays = _write_gpt2(tmp_path / "gpt2-service")
    service = JobService(tmp_path / "workspace")
    record, report = service.import_model(root)

    assert report["compatibility"]["overall"] == "validated"
    manifest = service.artifact_manifest(record.artifact_id)
    assert manifest["metadata"]["graph"]["name"] == "transformer_decoder_L1"
    package = service.get_package(record.artifact_id)
    assert "layers.0.qkv.weight" in package
    assert "transformer.h.0.attn.c_attn.weight" not in package
    assert service.verify_artifact(record.artifact_id)["ok"] is True
