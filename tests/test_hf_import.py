"""Hugging Face and safetensors import-pipeline tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from uzen.core.wie.importer import ModelImporter
from uzen.errors import RuntimeUnavailable
from uzen.infrastructure.storage.safetensors_io import write_safetensors
from uzen.interfaces.python_sdk.client import Uzen
from uzen.jobs.service import JobService

ARRAYS = {
    "model.embed_tokens.weight": np.arange(16 * 8, dtype=np.float32).reshape(16, 8),
    "model.layers.0.q_proj.weight": np.eye(8, dtype=np.float32),
    "model.layers.0.mlp_up.weight": np.ones((16, 8), dtype=np.float32),
    "model.norm.weight": np.ones(8, dtype=np.float32),
    "lm_head.weight": np.zeros((16, 8), dtype=np.float32),
}

CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 8,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "intermediate_size": 16,
    "vocab_size": 3,
    "max_position_embeddings": 32,
}


def _write_hf(root: Path, *, sharded: bool = False, config: dict | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(CONFIG if config is None else config))
    (root / "tokenizer.json").write_text(
        json.dumps({"model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "c": 2}}, "added_tokens": []})
    )
    (root / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 32, "padding_side": "left"})
    )
    (root / "README.md").write_text(
        "---\nlicense: apache-2.0\nlibrary_name: transformers\n---\n# tiny\n"
    )
    if sharded:
        first = {k: v for k, v in list(ARRAYS.items())[:2]}
        second = {k: v for k, v in list(ARRAYS.items())[2:]}
        write_safetensors(root / "model-00001-of-00002.safetensors", first)
        write_safetensors(root / "model-00002-of-00002.safetensors", second)
        weight_map = {
            name: "model-00001-of-00002.safetensors" if name in first else "model-00002-of-00002.safetensors"
            for name in ARRAYS
        }
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map})
        )
    else:
        write_safetensors(root / "model.safetensors", ARRAYS)
    return root


def test_hf_directory_import_reports_complete_weights_but_no_reconstruction(tmp_path):
    root = _write_hf(tmp_path / "tiny-hf")
    result = ModelImporter().import_source(root)

    assert result.ok
    assert result.package is not None
    report = result.report
    assert report.source_kind == "huggingface_directory"
    assert report.family == "transformer_decoder"
    assert report.compatibility.weights == "complete"
    assert report.compatibility.config == "parsed"
    assert report.compatibility.tokenizer == "present"
    assert report.compatibility.architecture == "recognized"
    assert report.compatibility.reconstruction == "not_implemented"
    assert report.compatibility.execution == "not_run"
    assert report.compatibility.overall == "ready_for_reconstruction"
    assert report.license["declared"] == "apache-2.0"
    assert report.tokenizer["vocab_size"] == 3
    assert report.source_hash
    assert len(report.tensors) == len(ARRAYS)
    assert report.role_counts.get("unclassified", 0) == 0
    assert any(t["role"] == "embedding" for t in report.tensors)
    assert any("not implemented" in reason for reason in report.compatibility.reasons)
    assert report.understanding["weights_loaded"] is True
    assert report.understanding["graph_reconstructed"] is False
    assert report.understanding["execution_validated"] is False


def test_sharded_index_import_combines_all_shards(tmp_path):
    root = _write_hf(tmp_path / "tiny-sharded", sharded=True)
    result = ModelImporter().import_source(root)

    assert result.ok
    assert result.report.source_kind == "huggingface_directory"
    assert result.report.metadata["sharded"] is True
    assert len(result.report.tensors) == len(ARRAYS)
    assert {f["role"] for f in result.report.files} >= {"shard_index", "weights"}


def test_missing_shard_is_an_explicit_failure(tmp_path):
    root = _write_hf(tmp_path / "broken", sharded=True)
    (root / "model-00002-of-00002.safetensors").unlink()
    result = ModelImporter().import_source(root)

    assert result.ok is False
    assert result.package is None
    assert result.report.compatibility.weights == "failed"
    assert result.report.compatibility.overall == "unsupported"
    assert any("missing shard" in problem for problem in result.report.problems)


def test_single_safetensors_without_config_is_weights_only(tmp_path):
    root = tmp_path / "bare"
    root.mkdir()
    write_safetensors(root / "model.safetensors", ARRAYS)
    result = ModelImporter().import_source(root / "model.safetensors")

    assert result.ok
    assert result.report.source_kind == "safetensors_file"
    assert result.package is not None and result.package.name == "model"
    assert result.report.compatibility.config == "missing"
    assert result.report.compatibility.architecture == "inferred"
    assert result.report.compatibility.overall == "weights_only"
    assert any("config.json is missing" in warning for warning in result.report.warnings)


def test_invalid_tokenizer_is_not_treated_as_present(tmp_path):
    root = _write_hf(tmp_path / "bad-tokenizer")
    (root / "tokenizer.json").write_text("{not-json")
    result = ModelImporter().import_source(root)

    assert result.ok
    assert result.report.compatibility.tokenizer == "invalid"
    assert result.report.compatibility.overall == "weights_only"
    assert "tokenizer_json_error" in result.report.tokenizer


def test_invalid_config_is_reported_without_discarding_weights(tmp_path):
    root = _write_hf(tmp_path / "bad-config")
    (root / "config.json").write_text("{not-json")
    result = ModelImporter().import_source(root)

    assert result.ok
    assert result.package is not None
    assert result.report.compatibility.config == "invalid"
    assert result.report.compatibility.overall == "weights_only"
    assert any("config.json is invalid" in warning for warning in result.report.warnings)


def test_repo_import_uses_injected_hub_and_records_revision(tmp_path):
    root = _write_hf(tmp_path / "hub-model")

    class FakeHub:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def snapshot_download(self, **kwargs):
            self.calls.append(kwargs)
            return str(root)

    hub = FakeHub()
    result = ModelImporter(hub=hub).import_source("acme/tiny", revision="v1")
    assert result.ok
    assert result.report.source_kind == "huggingface_repo"
    assert result.report.repo_id == "acme/tiny"
    assert result.report.revision == "v1"
    assert hub.calls[0]["repo_id"] == "acme/tiny"
    assert hub.calls[0]["revision"] == "v1"


def test_repo_import_without_hub_extra_is_a_typed_refusal(monkeypatch):
    def unavailable():
        raise RuntimeUnavailable("install uzen[hf]")

    monkeypatch.setattr("uzen.core.wie.importer._load_huggingface_hub", unavailable)
    with pytest.raises(RuntimeUnavailable):
        ModelImporter().import_source("acme/missing", revision="main")


def test_service_import_model_stores_report_and_artifact(tmp_path):
    root = _write_hf(tmp_path / "service-model")
    service = JobService(tmp_path / "workspace")
    record, report = service.import_model(root)

    assert record.num_tensors == len(ARRAYS)
    assert report["compatibility"]["weights"] == "complete"
    loaded = service.get_package(record.artifact_id)
    assert loaded.family == "transformer_decoder"
    manifest = service.artifact_manifest(record.artifact_id)
    assert manifest["metadata"]["import_report"]["source_hash"] == report["source_hash"]
    assert service.verify_artifact(record.artifact_id)["ok"] is True


def test_sdk_import_model_exposes_report(tmp_path):
    root = _write_hf(tmp_path / "sdk-model")
    client = Uzen(root=tmp_path / "sdk")
    imported = client.import_model(root)
    assert imported["artifact_id"]
    assert imported["compatibility"]["weights"] == "complete"
    assert imported["compatibility"]["reconstruction"] == "not_implemented"
    assert imported["report"]["understanding"]["graph_reconstructed"] is False


def test_api_import_directory_returns_compatibility_report(tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from uzen.interfaces.api.app import create_app

    root = _write_hf(tmp_path / "api-model")
    service = JobService(tmp_path / "api")
    api = fastapi_testclient.TestClient(create_app(service=service))
    response = api.post("/v1/models/import", json={"source": str(root)})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["compatibility"]["weights"] == "complete"
    assert body["compatibility"]["execution"] == "not_run"
    assert body["report"]["source_kind"] == "huggingface_directory"
    assert body["report"]["understanding"]["graph_reconstructed"] is False
