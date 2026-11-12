"""Live Hugging Face Hub validation.

Run explicitly with::

    UZEN_LIVE_HF=1 .venv/bin/python -m pytest tests/test_live_hf_import.py -q

The tests download tiny public GPT-2 and Llama checkpoints, import the actual
files, reconstruct their graphs, and require graph validation plus a finite
forward pass. GPT-2 and Llama logits are also compared with the Hugging Face
reference implementations; a successful download alone is not success.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from uzen.core.wie.importer import ModelImporter
from uzen.core.wie.parity import compare_gpt2_logits, compare_llama_logits

pytestmark = pytest.mark.skipif(
    os.environ.get("UZEN_LIVE_HF") != "1",
    reason="set UZEN_LIVE_HF=1 to run live Hugging Face Hub validation",
)


def test_live_tiny_gpt2_import_reconstruction_and_execution():
    pytest.importorskip("huggingface_hub")
    repo_id = "hf-internal-testing/tiny-random-GPT2"
    result = ModelImporter().import_source(repo_id, revision="main")

    assert result.ok
    report = result.report
    assert report.repo_id == repo_id
    assert report.revision == "main"
    assert report.source_kind == "huggingface_repo"
    assert report.source_hash
    assert report.config["model_type"] == "gpt2"
    assert len(report.tensors) == 64
    assert any(file["path"] == "config.json" for file in report.files)
    assert any(file["path"] == "tokenizer.json" for file in report.files)

    assert report.compatibility.weights == "complete"
    assert report.compatibility.reconstruction == "validated"
    assert report.compatibility.execution == "passed"
    assert report.compatibility.overall == "validated"
    assert report.understanding["graph_reconstructed"] is True
    assert report.understanding["execution_validated"] is True
    assert report.reconstruction["missing"] == []
    assert report.reconstruction["shape_errors"] == []
    assert report.reconstruction["extra"] == []
    assert report.reconstruction["validation"]["ok"] is True
    assert report.reconstruction["execution"]["status"] == "passed"
    assert report.reconstruction["execution"]["outputs"]["logits"]["shape"] == [1, 4, 1000]
    assert report.reconstruction["spec"] == {
        "d_model": 32,
        "n_heads": 4,
        "n_layers": 5,
        "d_ff": 128,
        "vocab_size": 1000,
        "max_seq_len": 512,
        "norm": "layernorm",
        "activation": "gelu",
        "tie_embeddings": True,
    }
    assert any(step["stage"] == "forward_execution" and step["status"] == "passed" for step in report.progress)

    assert result.package is not None
    assert result.artifact_package is not None
    source = result.package
    canonical = result.artifact_package
    assert "transformer.h.0.attn.c_attn.weight" in source
    assert "layers.0.qkv.weight" in canonical
    assert np.array_equal(
        canonical.get("layers.0.qkv.weight"),
        source.get("transformer.h.0.attn.c_attn.weight").T,
    )
    assert np.array_equal(canonical.get("embed.weight"), source.get("transformer.wte.weight"))
    assert np.array_equal(canonical.get("lm_head.weight"), source.get("transformer.wte.weight"))
    assert canonical.shape_of("embed.weight") == (report.config["vocab_size"], report.config["n_embd"])
    assert canonical.shape_of("layers.4.mlp_up.weight") == (
        report.reconstruction["spec"]["d_ff"],
        report.config["n_embd"],
    )


def test_live_tiny_gpt2_matches_huggingface_logits():
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    result = ModelImporter().import_source("hf-internal-testing/tiny-random-GPT2", revision="main")
    assert result.ok
    parity = compare_gpt2_logits(
        result.reconstruction.definition.graph,
        result.artifact_package,
        result.report.source,
        tokens=[[0, 1, 2, 3]],
        atol=1e-5,
        rtol=1e-5,
    )
    assert parity["tolerances"] == {"atol": 1e-5, "rtol": 1e-5}
    assert parity["passed"], parity
    assert parity["max_abs_error"] <= 1e-5
    assert parity["actual_shape"] == parity["reference_shape"] == [1, 4, 1000]


def test_live_tiny_gpt2_tokenizer_text_inference(tmp_path):
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("tokenizers")
    from uzen.interfaces.python_sdk.client import Uzen

    client = Uzen(root=tmp_path / "gpt2-text-workspace")
    imported = client.import_model("hf-internal-testing/tiny-random-GPT2", revision="main")
    artifact_id = imported["artifact_id"]
    assert client.artifact(artifact_id)["metadata"]["tokenizer"]["sha256"]

    result = client.forward_text(artifact_id, text="Hello world", include_values=True)
    token_ids = result["token_ids"][0]
    assert result["tokens"][0]
    assert result["decoded_text"]
    assert result["outputs"]["logits"]["shape"] == [1, len(token_ids), 1000]
    assert np.isfinite(np.asarray(result["outputs"]["logits"]["values"])).all()


def test_live_tiny_llama_import_reconstruction_and_execution():
    pytest.importorskip("huggingface_hub")
    repo_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    result = ModelImporter().import_source(repo_id, revision="main")

    assert result.ok
    report = result.report
    assert report.repo_id == repo_id
    assert report.source_kind == "huggingface_repo"
    assert report.config["model_type"] == "llama"
    assert len(report.tensors) == 21
    assert report.compatibility.reconstruction == "validated"
    assert report.compatibility.execution == "passed"
    assert report.compatibility.overall == "validated"
    assert report.reconstruction["missing"] == []
    assert report.reconstruction["shape_errors"] == []
    assert report.reconstruction["extra"] == []
    assert report.reconstruction["validation"]["ok"] is True
    assert report.reconstruction["execution"]["status"] == "passed"
    assert report.reconstruction["execution"]["outputs"]["logits"]["shape"] == [1, 4, 32000]
    assert report.reconstruction["spec"]["n_kv_heads"] == 4
    assert report.reconstruction["spec"]["rope_theta"] == 10000.0
    assert result.artifact_package is not None
    assert "layers.0.q_proj.weight" in result.artifact_package
    assert "model.layers.0.self_attn.q_proj.weight" in result.package


def test_live_tiny_llama_tokenizer_text_inference(tmp_path):
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("tokenizers")
    from uzen.interfaces.python_sdk.client import Uzen

    client = Uzen(root=tmp_path / "llama-text-workspace")
    imported = client.import_model(
        "hf-internal-testing/tiny-random-LlamaForCausalLM", revision="main"
    )
    artifact_id = imported["artifact_id"]
    assert client.artifact(artifact_id)["metadata"]["tokenizer"]["sha256"]

    result = client.forward_text(artifact_id, text="Hello world", include_values=True)
    token_ids = result["token_ids"][0]
    assert result["tokens"][0]
    assert result["decoded_text"]
    assert result["outputs"]["logits"]["shape"] == [1, len(token_ids), 32000]
    assert np.isfinite(np.asarray(result["outputs"]["logits"]["values"])).all()


def test_live_tiny_llama_matches_huggingface_logits():
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("transformers")
    pytest.importorskip("torch")

    result = ModelImporter().import_source(
        "hf-internal-testing/tiny-random-LlamaForCausalLM", revision="main"
    )
    parity = compare_llama_logits(
        result.reconstruction.definition.graph,
        result.artifact_package,
        result.report.source,
        tokens=[[0, 1, 2, 3]],
        positions=[[0, 1, 2, 3]],
        atol=1e-5,
        rtol=1e-5,
    )
    assert parity["tolerances"] == {"atol": 1e-5, "rtol": 1e-5}
    assert parity["positions"] == [[0, 1, 2, 3]]
    assert parity["passed"], parity
    assert parity["max_abs_error"] <= 1e-5
    assert parity["actual_shape"] == parity["reference_shape"] == [1, 4, 32000]
