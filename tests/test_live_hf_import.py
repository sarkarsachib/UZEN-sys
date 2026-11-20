"""Live Hugging Face Hub validation.

Run explicitly with::

    UZEN_LIVE_HF=1 .venv/bin/python -m pytest tests/test_live_hf_import.py -q

The tests download tiny public GPT-2, Llama, OPT and GPT-NeoX checkpoints,
import the actual files, reconstruct their graphs, and require graph validation
plus a finite forward pass. Every family's logits are also compared with its
Hugging Face reference implementation; a successful download alone is not
success.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from uzen.core.wie.importer import ModelImporter
from uzen.core.wie.parity import (
    compare_gpt2_activations,
    compare_gpt2_logits,
    compare_llama_activations,
    compare_llama_logits,
    compare_neox_activations,
    compare_neox_logits,
    compare_opt_activations,
    compare_opt_logits,
)

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


def test_live_tiny_gpt2_matches_huggingface_hidden_states():
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    result = ModelImporter().import_source("hf-internal-testing/tiny-random-GPT2", revision="main")
    assert result.ok
    parity = compare_gpt2_activations(
        result.reconstruction.definition.graph,
        result.artifact_package,
        result.report.source,
        tokens=[[0, 1, 2, 3]],
        atol=1e-5,
        rtol=1e-5,
    )
    assert parity["num_layers"] == 5
    assert parity["passed"], parity
    assert parity["max_abs_error"] <= 1e-5
    assert all(layer["passed"] for layer in parity["layers"])
    assert [layer["node"] for layer in parity["layers"]] == [
        f"l{i}_res2" for i in range(5)
    ]


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


def test_live_tiny_gpt2_greedy_generation(tmp_path):
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("tokenizers")
    from uzen.interfaces.python_sdk.client import Uzen

    client = Uzen(root=tmp_path / "gpt2-generation-workspace")
    imported = client.import_model("hf-internal-testing/tiny-random-GPT2", revision="main")
    result = client.generate_text(
        imported["artifact_id"], text="Hello", max_new_tokens=2
    )
    assert 1 <= result["steps"] <= 2
    assert result["stopped_reason"] in {"eos_token", "max_new_tokens"}
    assert len(result["full_token_ids"]) == (
        len(result["prompt_token_ids"]) + result["steps"]
    )
    assert isinstance(result["generated_text"], str)


def test_live_tiny_gpt2_modify_export_reload_roundtrip(tmp_path):
    """Modify a real checkpoint, export it, reload it and compare real outputs.

    The safetensors export embeds the stored NIR graph, so the reloaded artifact
    executes from its own manifest without the parent artifact.
    """
    pytest.importorskip("huggingface_hub")
    from uzen.interfaces.python_sdk.client import Uzen

    client = Uzen(root=tmp_path / "roundtrip-workspace")
    imported = client.import_model("hf-internal-testing/tiny-random-GPT2", revision="main")
    source_id = imported["artifact_id"]
    source_execution = client.service.forward_model(source_id, tokens=[[0, 1, 2, 3]])
    source_fingerprint = client.package(source_id).fingerprint()

    modified = client.modify(source_id, "scale", params={"factor": 0.5})
    modified_id = modified["artifact_id"]
    assert modified["architecture_preserved"] is True
    modified_execution = client.service.forward_model(
        modified_id, tokens=[[0, 1, 2, 3]], capture="layers"
    )

    expected_layers = sorted(modified_execution.intermediates)
    assert expected_layers == [f"l{i}_res2" for i in range(5)]
    assert not np.array_equal(
        modified_execution.outputs["logits"], source_execution.outputs["logits"]
    )

    export_path = client.service.store.export(
        modified_id, tmp_path / "modified-gpt2.safetensors"
    )
    assert export_path.exists() and export_path.stat().st_size > 0

    reloaded = client.import_weights(export_path)
    reloaded_id = reloaded["artifact_id"]
    reloaded_manifest = client.artifact(reloaded_id)
    assert reloaded_manifest["metadata"]["graph"]["name"] == modified_execution.graph_name
    assert (
        reloaded_manifest["metadata"]["import_report"]["metadata"]["graph_restored_from_metadata"]
        is True
    )

    reloaded_package = client.package(reloaded_id)
    modified_package = client.package(modified_id)
    assert reloaded_package.fingerprint() == modified_package.fingerprint()
    for name in reloaded_package.names():
        np.testing.assert_array_equal(
            reloaded_package.tensors[name], modified_package.tensors[name]
        )

    reloaded_execution = client.service.forward_model(
        reloaded_id, tokens=[[0, 1, 2, 3]], capture="layers"
    )
    np.testing.assert_array_equal(
        reloaded_execution.outputs["logits"], modified_execution.outputs["logits"]
    )
    assert sorted(reloaded_execution.intermediates) == expected_layers
    for name in expected_layers:
        np.testing.assert_array_equal(
            reloaded_execution.intermediates[name], modified_execution.intermediates[name]
        )

    assert client.package(source_id).fingerprint() == source_fingerprint
    assert client.verify_artifact(modified_id)["ok"] is True
    assert client.verify_artifact(reloaded_id)["ok"] is True
    lineage = client.artifact(modified_id)["lineage"]
    assert lineage["parents"] == [source_id]
    assert lineage["operation"] == "w.arithmetic.scale"


def test_live_tiny_neox_import_reconstruction_and_execution():
    pytest.importorskip("huggingface_hub")
    repo_id = "hyper-accel/tiny-random-gpt_neox"
    result = ModelImporter().import_source(repo_id, revision="main")

    assert result.ok
    report = result.report
    assert report.repo_id == repo_id
    assert report.source_kind == "huggingface_repo"
    assert report.config["model_type"] == "gpt_neox"
    assert len(report.tensors) == 28
    assert report.compatibility.reconstruction == "validated"
    assert report.compatibility.execution == "passed"
    assert report.compatibility.overall == "validated"
    assert report.reconstruction["missing"] == []
    assert report.reconstruction["shape_errors"] == []
    assert report.reconstruction["extra"] == []
    assert report.reconstruction["validation"]["ok"] is True
    assert report.reconstruction["execution"]["status"] == "passed"
    assert report.reconstruction["execution"]["outputs"]["logits"]["shape"] == [
        1,
        4,
        50432,
    ]
    assert report.reconstruction["spec"] == {
        "architecture": "gpt_neox",
        "d_model": 384,
        "n_heads": 4,
        "head_dim": 96,
        "n_layers": 2,
        "d_ff": 1536,
        "vocab_size": 50432,
        "max_seq_len": 2048,
        "rotary_dim": 24,
        "rope_theta": 10000.0,
        "layer_norm_eps": 1e-5,
        "activation": "gelu",
        "parallel_residual": True,
        "attention_bias": True,
        "tie_embeddings": False,
    }
    assert result.artifact_package is not None
    assert "gpt_neox.layers.0.attention.query_key_value.weight" in result.package
    assert "layers.0.qkv.weight" in result.artifact_package
    mapping = {m["target"]: m for m in report.reconstruction["tensor_mapping"]}
    assert mapping["layers.0.qkv.weight"]["source_shape"] == [1152, 384]
    assert mapping["layers.0.qkv.bias"]["source_shape"] == [1152]
    assert mapping["lm_head.weight"]["source"] == "embed_out.weight"


def test_live_tiny_neox_matches_huggingface_logits():
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("transformers")
    pytest.importorskip("torch")

    result = ModelImporter().import_source(
        "hyper-accel/tiny-random-gpt_neox", revision="main"
    )
    parity = compare_neox_logits(
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
    assert parity["actual_shape"] == parity["reference_shape"] == [1, 4, 50432]


def test_live_tiny_neox_matches_huggingface_hidden_states():
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("transformers")
    pytest.importorskip("torch")

    result = ModelImporter().import_source(
        "hyper-accel/tiny-random-gpt_neox", revision="main"
    )
    parity = compare_neox_activations(
        result.reconstruction.definition.graph,
        result.artifact_package,
        result.report.source,
        tokens=[[0, 1, 2, 3]],
        positions=[[0, 1, 2, 3]],
        atol=1e-5,
        rtol=1e-5,
    )
    assert parity["num_layers"] == 2
    assert parity["positions"] == [[0, 1, 2, 3]]
    assert parity["passed"], parity
    assert parity["max_abs_error"] <= 1e-5
    assert parity["layers"][0]["node"] == "l0_res2"


def test_live_tiny_qwen2_biases_match_huggingface(tmp_path):
    """Verify a real Qwen2 checkpoint with Q/K/V biases, not just the synthetic spec."""
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM

    local = tmp_path / "qwen2"
    snapshot_download(
        repo_id="Jiqing/tiny-random-qwen2",
        local_dir=local,
        allow_patterns=["*.json", "*.safetensors"],
    )
    result = ModelImporter().import_source(local)
    assert result.ok
    assert result.report.compatibility.reconstruction == "validated"
    assert result.report.reconstruction["spec"]["profile"] == "qwen2"
    assert any(
        node.get("params", {}).get("bias") == "param:layers.0.q_proj.bias"
        for node in result.report.reconstruction["graph"]["nodes"]
    )

    parity = compare_llama_logits(
        result.reconstruction.definition.graph,
        result.artifact_package,
        local,
        tokens=[[0, 1, 2, 3]],
        positions=[[0, 1, 2, 3]],
        atol=1e-5,
        rtol=1e-5,
        model_cls=AutoModelForCausalLM,
    )
    assert parity["passed"], parity
    assert parity["max_abs_error"] <= 1e-5


def test_live_tiny_olmo_non_affine_norms_match_huggingface(tmp_path):
    """Verify OLMo's non-parametric norms through the shared profile adapter."""
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM

    local = tmp_path / "olmo"
    snapshot_download(
        repo_id="hf-internal-testing/tiny-random-OlmoForCausalLM",
        local_dir=local,
        allow_patterns=["*.json", "*.safetensors"],
    )
    result = ModelImporter().import_source(local)
    assert result.ok
    assert result.report.compatibility.reconstruction == "validated"
    assert result.report.reconstruction["spec"]["profile"] == "olmo"
    assert "norm.weight" not in result.artifact_package.tensors

    parity = compare_llama_logits(
        result.reconstruction.definition.graph,
        result.artifact_package,
        local,
        tokens=[[0, 1, 2, 3]],
        positions=[[0, 1, 2, 3]],
        atol=1e-5,
        rtol=1e-5,
        model_cls=AutoModelForCausalLM,
        model_kwargs={"torch_dtype": __import__("torch").float64},
    )
    assert parity["passed"], parity
    assert parity["max_abs_error"] <= 1e-5


def test_live_tiny_gemma_matches_huggingface(tmp_path):
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from transformers import GemmaForCausalLM

    repo_id = "fxmarty/tiny-random-GemmaForCausalLM"
    model = GemmaForCausalLM.from_pretrained(repo_id).eval()
    local = tmp_path / "gemma"
    model.save_pretrained(local, safe_serialization=True)

    result = ModelImporter().import_source(local)
    assert result.ok
    assert result.report.compatibility.reconstruction == "validated"
    assert result.report.reconstruction["spec"]["profile"] == "gemma"
    assert result.report.reconstruction["spec"]["zero_centered_norm"] is True
    assert any(
        node["op"] == "scale"
        and node["params"].get("factor") == (result.report.config["hidden_size"] ** 0.5)
        for node in result.report.reconstruction["graph"]["nodes"]
    )
    assert any(
        item.get("transform") == "add_one"
        for item in result.report.reconstruction["tensor_mapping"]
    )

    parity = compare_llama_logits(
        result.reconstruction.definition.graph,
        result.artifact_package,
        local,
        tokens=[[0, 1, 2, 3]],
        positions=[[0, 1, 2, 3]],
        atol=1e-5,
        rtol=1e-5,
        model_cls=GemmaForCausalLM,
    )
    assert parity["passed"], parity
    assert parity["max_abs_error"] <= 1e-5


def test_live_tiny_gptj_matches_huggingface(tmp_path):
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from transformers import GPTJForCausalLM

    repo_id = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    model = GPTJForCausalLM.from_pretrained(repo_id).eval()
    local = tmp_path / "gptj"
    model.save_pretrained(local, safe_serialization=True)

    result = ModelImporter().import_source(local)
    assert result.ok
    assert result.report.compatibility.reconstruction == "validated"
    assert result.report.reconstruction["spec"]["profile"] == "gptj"
    assert result.report.reconstruction["spec"]["rotary_dim"] == 4
    assert result.report.reconstruction["spec"]["activation"] == "gelu"
    assert any(
        node["op"] == "rope" and node["params"].get("interleaved") is True
        for node in result.report.reconstruction["graph"]["nodes"]
    )
    assert not any(
        node["id"] == "l0_ln2"
        for node in result.report.reconstruction["graph"]["nodes"]
    )

    parity = compare_llama_logits(
        result.reconstruction.definition.graph,
        result.artifact_package,
        local,
        tokens=[[0, 1, 2, 3]],
        positions=[[0, 1, 2, 3]],
        atol=1e-5,
        rtol=1e-5,
        model_cls=GPTJForCausalLM,
    )
    assert parity["passed"], parity
    assert parity["max_abs_error"] <= 1e-5


def test_live_tiny_mistral_sliding_window_matches_huggingface(tmp_path):
    """Verify the windowed Mistral path against a real checkpoint."""
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from transformers import AutoModelForCausalLM, MistralForCausalLM

    repo_id = "hf-internal-testing/tiny-random-MistralForCausalLM"
    model = MistralForCausalLM.from_pretrained(repo_id).eval()
    local = tmp_path / "mistral"
    model.save_pretrained(local, safe_serialization=True)

    result = ModelImporter().import_source(local)
    assert result.ok
    assert result.report.compatibility.reconstruction == "validated"
    assert result.report.reconstruction["spec"]["profile"] == "mistral"
    window = result.report.reconstruction["spec"]["sliding_window"]
    assert window > 0
    assert any(
        node["op"] == "attention" and node.get("params", {}).get("window") == window
        for node in result.report.reconstruction["graph"]["nodes"]
    )

    parity = compare_llama_logits(
        result.reconstruction.definition.graph,
        result.artifact_package,
        local,
        tokens=[[0, 1, 2, 3]],
        positions=[[0, 1, 2, 3]],
        atol=1e-5,
        rtol=1e-5,
        model_cls=AutoModelForCausalLM,
    )
    assert parity["passed"], parity
    assert parity["max_abs_error"] <= 1e-5


def test_live_tiny_neox_exact_gelu_matches_huggingface(tmp_path):
    """Verify the exact-GELU path against a real NeoX checkpoint.

    The upstream tiny NeoX fixture is distributed as a pickle, which the importer
    deliberately refuses. This test converts that trusted fixture to safetensors
    once, then imports and compares the converted directory.
    """
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from transformers import GPTNeoXForCausalLM

    repo_id = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
    model = GPTNeoXForCausalLM.from_pretrained(repo_id).eval()
    local = tmp_path / "exact-gelu-neox"
    model.save_pretrained(local, safe_serialization=True)

    result = ModelImporter().import_source(local)
    assert result.ok
    assert result.report.config["hidden_act"] == "gelu"
    assert result.report.compatibility.reconstruction == "validated"
    assert result.report.compatibility.execution == "passed"
    assert result.report.reconstruction["spec"]["activation"] == "gelu_exact"
    assert any(
        node["op"] == "gelu_exact"
        for node in result.report.reconstruction["graph"]["nodes"]
    )

    parity = compare_neox_logits(
        result.reconstruction.definition.graph,
        result.artifact_package,
        local,
        tokens=[[0, 1, 2, 3]],
        positions=[[0, 1, 2, 3]],
        atol=1e-5,
        rtol=1e-5,
    )
    assert parity["passed"], parity
    assert parity["max_abs_error"] <= 1e-5


def test_live_tiny_neox_modify_export_reload_roundtrip(tmp_path):
    """Scale a real GPT-NeoX checkpoint, export it, reload it and compare outputs."""
    pytest.importorskip("huggingface_hub")
    from uzen.interfaces.python_sdk.client import Uzen

    tokens = [[0, 1, 2, 3]]
    positions = [[0, 1, 2, 3]]
    client = Uzen(root=tmp_path / "neox-roundtrip-workspace")
    imported = client.import_model(
        "hyper-accel/tiny-random-gpt_neox", revision="main"
    )
    source_id = imported["artifact_id"]
    source_execution = client.service.forward_model(
        source_id, tokens=tokens, positions=positions
    )
    source_fingerprint = client.package(source_id).fingerprint()

    modified = client.modify(source_id, "scale", params={"factor": 0.5})
    modified_id = modified["artifact_id"]
    assert modified["architecture_preserved"] is True
    modified_execution = client.service.forward_model(
        modified_id, tokens=tokens, positions=positions, capture="layers"
    )

    expected_layers = sorted(modified_execution.intermediates)
    assert expected_layers == ["l0_res2", "l1_res2"]
    assert not np.array_equal(
        modified_execution.outputs["logits"], source_execution.outputs["logits"]
    )

    export_path = client.service.store.export(
        modified_id, tmp_path / "modified-neox.safetensors"
    )
    assert export_path.exists() and export_path.stat().st_size > 0

    reloaded = client.import_weights(export_path)
    reloaded_id = reloaded["artifact_id"]
    reloaded_manifest = client.artifact(reloaded_id)
    assert reloaded_manifest["metadata"]["graph"]["name"] == modified_execution.graph_name
    assert (
        reloaded_manifest["metadata"]["import_report"]["metadata"]["graph_restored_from_metadata"]
        is True
    )

    reloaded_package = client.package(reloaded_id)
    modified_package = client.package(modified_id)
    assert reloaded_package.fingerprint() == modified_package.fingerprint()
    for name in reloaded_package.names():
        np.testing.assert_array_equal(
            reloaded_package.tensors[name], modified_package.tensors[name]
        )

    reloaded_execution = client.service.forward_model(
        reloaded_id, tokens=tokens, positions=positions, capture="layers"
    )
    np.testing.assert_array_equal(
        reloaded_execution.outputs["logits"], modified_execution.outputs["logits"]
    )
    assert sorted(reloaded_execution.intermediates) == expected_layers
    for name in expected_layers:
        np.testing.assert_array_equal(
            reloaded_execution.intermediates[name], modified_execution.intermediates[name]
        )

    assert client.package(source_id).fingerprint() == source_fingerprint
    assert client.verify_artifact(modified_id)["ok"] is True
    assert client.verify_artifact(reloaded_id)["ok"] is True
    lineage = client.artifact(modified_id)["lineage"]
    assert lineage["parents"] == [source_id]
    assert lineage["operation"] == "w.arithmetic.scale"


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


def test_live_tiny_llama_greedy_generation(tmp_path):
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("tokenizers")
    from uzen.interfaces.python_sdk.client import Uzen

    client = Uzen(root=tmp_path / "llama-generation-workspace")
    imported = client.import_model(
        "hf-internal-testing/tiny-random-LlamaForCausalLM", revision="main"
    )
    result = client.generate_text(
        imported["artifact_id"], text="Hello", max_new_tokens=2
    )
    assert 1 <= result["steps"] <= 2
    assert result["stopped_reason"] in {"eos_token", "max_new_tokens"}
    assert len(result["full_token_ids"]) == (
        len(result["prompt_token_ids"]) + result["steps"]
    )
    assert isinstance(result["generated_text"], str)

    # The cached path must select exactly the same tokens as full recomputation.
    from uzen.core.wie import model_execution

    original = model_execution._supports_kv_cache
    model_execution._supports_kv_cache = lambda graph: False
    try:
        full = client.generate_text(
            imported["artifact_id"], text="Hello", max_new_tokens=2
        )
    finally:
        model_execution._supports_kv_cache = original
    assert full["generated_token_ids"] == result["generated_token_ids"]


def test_live_tiny_llama_modify_export_reload_roundtrip(tmp_path):
    """Scale a real Llama checkpoint, export it, reload it and compare outputs.

    Llama's graph takes explicit positions. The safetensors export embeds the
    stored NIR graph, so the reloaded artifact executes from its own manifest.
    """
    pytest.importorskip("huggingface_hub")
    from uzen.interfaces.python_sdk.client import Uzen

    tokens = [[0, 1, 2, 3]]
    positions = [[0, 1, 2, 3]]
    client = Uzen(root=tmp_path / "llama-roundtrip-workspace")
    imported = client.import_model(
        "hf-internal-testing/tiny-random-LlamaForCausalLM", revision="main"
    )
    source_id = imported["artifact_id"]
    source_execution = client.service.forward_model(
        source_id, tokens=tokens, positions=positions
    )
    source_fingerprint = client.package(source_id).fingerprint()

    modified = client.modify(source_id, "scale", params={"factor": 0.5})
    modified_id = modified["artifact_id"]
    assert modified["architecture_preserved"] is True
    modified_execution = client.service.forward_model(
        modified_id, tokens=tokens, positions=positions, capture="layers"
    )

    expected_layers = sorted(modified_execution.intermediates)
    spec = imported["report"]["reconstruction"]["spec"]
    assert len(expected_layers) == spec["n_layers"]
    assert all(name.startswith("l") and name.endswith("_res2") for name in expected_layers)
    assert not np.array_equal(
        modified_execution.outputs["logits"], source_execution.outputs["logits"]
    )

    export_path = client.service.store.export(
        modified_id, tmp_path / "modified-llama.safetensors"
    )
    assert export_path.exists() and export_path.stat().st_size > 0

    reloaded = client.import_weights(export_path)
    reloaded_id = reloaded["artifact_id"]
    reloaded_manifest = client.artifact(reloaded_id)
    assert reloaded_manifest["metadata"]["graph"]["name"] == modified_execution.graph_name
    assert (
        reloaded_manifest["metadata"]["import_report"]["metadata"]["graph_restored_from_metadata"]
        is True
    )

    reloaded_package = client.package(reloaded_id)
    modified_package = client.package(modified_id)
    assert reloaded_package.fingerprint() == modified_package.fingerprint()
    for name in reloaded_package.names():
        np.testing.assert_array_equal(
            reloaded_package.tensors[name], modified_package.tensors[name]
        )

    reloaded_execution = client.service.forward_model(
        reloaded_id,
        tokens=tokens,
        positions=positions,
        capture="layers",
    )
    np.testing.assert_array_equal(
        reloaded_execution.outputs["logits"], modified_execution.outputs["logits"]
    )
    assert sorted(reloaded_execution.intermediates) == expected_layers
    for name in expected_layers:
        np.testing.assert_array_equal(
            reloaded_execution.intermediates[name], modified_execution.intermediates[name]
        )

    assert client.package(source_id).fingerprint() == source_fingerprint
    assert client.verify_artifact(modified_id)["ok"] is True
    assert client.verify_artifact(reloaded_id)["ok"] is True
    lineage = client.artifact(modified_id)["lineage"]
    assert lineage["parents"] == [source_id]
    assert lineage["operation"] == "w.arithmetic.scale"


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


def test_live_tiny_llama_matches_huggingface_hidden_states():
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("transformers")
    pytest.importorskip("torch")

    result = ModelImporter().import_source(
        "hf-internal-testing/tiny-random-LlamaForCausalLM", revision="main"
    )
    parity = compare_llama_activations(
        result.reconstruction.definition.graph,
        result.artifact_package,
        result.report.source,
        tokens=[[0, 1, 2, 3]],
        positions=[[0, 1, 2, 3]],
        atol=1e-5,
        rtol=1e-5,
    )
    assert parity["num_layers"] >= 1
    assert parity["num_layers"] == len(parity["layers"])
    assert parity["positions"] == [[0, 1, 2, 3]]
    assert parity["passed"], parity
    assert parity["max_abs_error"] <= 1e-5
    assert parity["layers"][0]["node"] == "l0_res2"


def test_live_tiny_opt_import_reconstruction_and_execution():
    pytest.importorskip("huggingface_hub")
    repo_id = "hf-internal-testing/tiny-random-OPTForCausalLM"
    result = ModelImporter().import_source(repo_id, revision="main")

    assert result.ok
    report = result.report
    assert report.repo_id == repo_id
    assert report.source_kind == "huggingface_repo"
    assert report.config["model_type"] == "opt"
    assert len(report.tensors) == 84
    assert report.compatibility.reconstruction == "validated"
    assert report.compatibility.execution == "passed"
    assert report.compatibility.overall == "validated"
    assert report.reconstruction["missing"] == []
    assert report.reconstruction["shape_errors"] == []
    assert report.reconstruction["extra"] == []
    assert report.reconstruction["validation"]["ok"] is True
    assert report.reconstruction["execution"]["status"] == "passed"
    assert report.reconstruction["execution"]["outputs"]["logits"]["shape"] == [
        1,
        4,
        50265,
    ]
    exact_spec = {
        "architecture": "opt",
        "d_model": 16,
        "n_heads": 4,
        "head_dim": 4,
        "n_layers": 5,
        "d_ff": 4,
        "vocab_size": 50265,
        "max_seq_len": 100,
        "pos_table_size": 102,
        "position_offset": 2,
        "norm": "layernorm",
        "layer_norm_eps": 1e-5,
        "activation": "relu",
        "tie_embeddings": True,
    }
    assert report.reconstruction["spec"] == exact_spec
    assert result.artifact_package is not None
    assert "model.decoder.layers.0.self_attn.q_proj.weight" in result.package
    assert "layers.0.q_proj.weight" in result.artifact_package
    np.testing.assert_array_equal(
        result.artifact_package.get("lm_head.weight"),
        result.artifact_package.get("embed.weight"),
    )
    lm_mapping = next(
        item
        for item in report.reconstruction["tensor_mapping"]
        if item["target"] == "lm_head.weight"
    )
    assert lm_mapping["transform"] == "tied_embedding"


def test_live_tiny_opt_tokenizer_text_inference_and_generation(tmp_path):
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("tokenizers")
    from uzen.interfaces.python_sdk.client import Uzen

    client = Uzen(root=tmp_path / "opt-text-workspace")
    imported = client.import_model(
        "hf-internal-testing/tiny-random-OPTForCausalLM", revision="main"
    )
    artifact_id = imported["artifact_id"]
    assert client.artifact(artifact_id)["metadata"]["tokenizer"]["sha256"]

    result = client.forward_text(artifact_id, text="Hello world", include_values=True)
    token_ids = result["token_ids"][0]
    assert result["tokens"][0]
    assert result["outputs"]["logits"]["shape"] == [1, len(token_ids), 50265]
    assert np.isfinite(np.asarray(result["outputs"]["logits"]["values"])).all()

    generated = client.generate_text(artifact_id, text="Hello", max_new_tokens=2)
    assert 1 <= generated["steps"] <= 2
    assert generated["stopped_reason"] in {"eos_token", "max_new_tokens"}
    assert len(generated["full_token_ids"]) == (
        len(generated["prompt_token_ids"]) + generated["steps"]
    )
    assert isinstance(generated["generated_text"], str)


def test_live_tiny_opt_matches_huggingface_logits():
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("transformers")
    pytest.importorskip("torch")

    result = ModelImporter().import_source(
        "hf-internal-testing/tiny-random-OPTForCausalLM", revision="main"
    )
    parity = compare_opt_logits(
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
    assert parity["actual_shape"] == parity["reference_shape"] == [1, 4, 50265]


def test_live_tiny_opt_matches_huggingface_hidden_states():
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("transformers")
    pytest.importorskip("torch")

    result = ModelImporter().import_source(
        "hf-internal-testing/tiny-random-OPTForCausalLM", revision="main"
    )
    parity = compare_opt_activations(
        result.reconstruction.definition.graph,
        result.artifact_package,
        result.report.source,
        tokens=[[0, 1, 2, 3]],
        atol=1e-5,
        rtol=1e-5,
    )
    assert parity["num_layers"] == 5
    assert parity["passed"], parity
    assert parity["max_abs_error"] <= 1e-5
    assert [layer["node"] for layer in parity["layers"]] == [
        f"l{i}_res2" for i in range(5)
    ]


def test_live_tiny_opt_modify_export_reload_roundtrip(tmp_path):
    """Scale a real OPT checkpoint, export it, reload it and compare outputs."""
    pytest.importorskip("huggingface_hub")
    from uzen.interfaces.python_sdk.client import Uzen

    tokens = [[0, 1, 2, 3]]
    client = Uzen(root=tmp_path / "opt-roundtrip-workspace")
    imported = client.import_model(
        "hf-internal-testing/tiny-random-OPTForCausalLM", revision="main"
    )
    source_id = imported["artifact_id"]
    source_execution = client.service.forward_model(source_id, tokens=tokens)
    source_fingerprint = client.package(source_id).fingerprint()

    modified = client.modify(source_id, "scale", params={"factor": 0.5})
    modified_id = modified["artifact_id"]
    assert modified["architecture_preserved"] is True
    modified_execution = client.service.forward_model(
        modified_id, tokens=tokens, capture="layers"
    )

    expected_layers = sorted(modified_execution.intermediates)
    assert expected_layers == [f"l{i}_res2" for i in range(5)]
    assert not np.array_equal(
        modified_execution.outputs["logits"], source_execution.outputs["logits"]
    )

    export_path = client.service.store.export(
        modified_id, tmp_path / "modified-opt.safetensors"
    )
    assert export_path.exists() and export_path.stat().st_size > 0

    reloaded = client.import_weights(export_path)
    reloaded_id = reloaded["artifact_id"]
    reloaded_manifest = client.artifact(reloaded_id)
    assert reloaded_manifest["metadata"]["graph"]["name"] == modified_execution.graph_name
    assert (
        reloaded_manifest["metadata"]["import_report"]["metadata"]["graph_restored_from_metadata"]
        is True
    )

    reloaded_package = client.package(reloaded_id)
    modified_package = client.package(modified_id)
    assert reloaded_package.fingerprint() == modified_package.fingerprint()
    for name in reloaded_package.names():
        np.testing.assert_array_equal(
            reloaded_package.tensors[name], modified_package.tensors[name]
        )

    reloaded_execution = client.service.forward_model(
        reloaded_id, tokens=tokens, capture="layers"
    )
    np.testing.assert_array_equal(
        reloaded_execution.outputs["logits"], modified_execution.outputs["logits"]
    )
    assert sorted(reloaded_execution.intermediates) == expected_layers
    for name in expected_layers:
        np.testing.assert_array_equal(
            reloaded_execution.intermediates[name], modified_execution.intermediates[name]
        )

    assert client.package(source_id).fingerprint() == source_fingerprint
    assert client.verify_artifact(modified_id)["ok"] is True
    assert client.verify_artifact(reloaded_id)["ok"] is True
    lineage = client.artifact(modified_id)["lineage"]
    assert lineage["parents"] == [source_id]
    assert lineage["operation"] == "w.arithmetic.scale"
