"""Tokenizer-backed text inference for imported GPT-2 and Llama artifacts."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tests.test_gpt2_reconstruction import _write_gpt2
from tests.test_llama_reconstruction import _write_llama
from uzen.core.wie.tokenization import tokenizer_spec_from_manifest
from uzen.errors import InvalidOperationParameters
from uzen.interfaces.cli.main import main as cli_main
from uzen.interfaces.mcp.server import MCPServer
from uzen.interfaces.python_sdk.client import Uzen

pytest.importorskip("tokenizers")
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402


def _write_text_tokenizer(root, vocab_size: int, *, metaspace: bool) -> None:
    """Create a real BPE tokenizer.json appropriate to the model family."""
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    if metaspace:
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
            replacement="▁", prepend_scheme="never"
        )
        tokenizer.decoder = decoders.Metaspace(
            replacement="▁", prepend_scheme="never"
        )
    else:
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>"],
        show_progress=False,
        min_frequency=1,
    )
    tokenizer.train_from_iterator(
        [
            "hello world",
            "hello there",
            "the world is wide",
            "a small model",
            "hello hello world",
        ],
        trainer=trainer,
    )
    tokenizer.save(str(root / "tokenizer.json"))


def _gpt2_text_client(tmp_path):
    root, _arrays = _write_gpt2(tmp_path / "gpt2-text")
    _write_text_tokenizer(root, 16, metaspace=False)
    client = Uzen(root=tmp_path / "gpt2-text-workspace")
    imported = client.import_model(root)
    return client, imported["artifact_id"]


def _llama_text_client(tmp_path):
    root, _arrays = _write_llama(tmp_path / "llama-text")
    _write_text_tokenizer(root, 32, metaspace=True)
    client = Uzen(root=tmp_path / "llama-text-workspace")
    imported = client.import_model(root)
    return client, imported["artifact_id"]


def test_gpt2_text_inference_uses_stored_tokenizer(tmp_path):
    client, artifact_id = _gpt2_text_client(tmp_path)
    manifest = client.artifact(artifact_id)
    spec = manifest["metadata"]["tokenizer"]
    assert spec["kind"] == "huggingface-tokenizer-json"
    assert spec["model_type"] == "BPE"
    assert len(spec["sha256"]) == 64
    assert manifest["lineage"]["operations"][0]["op"] == "tokenizer.attach"

    result = client.forward_text(
        artifact_id, text="hello world", include_values=True
    )
    token_ids = result["token_ids"][0]
    assert result["text"] == "hello world"
    assert result["tokens"][0]
    assert result["decoded_text"]
    assert len(token_ids) == len(result["tokens"][0])
    assert result["outputs"]["logits"]["shape"] == [1, len(token_ids), 16]

    direct = client.forward_model(
        artifact_id, tokens=[token_ids], include_values=True
    )
    np.testing.assert_array_equal(
        np.asarray(result["outputs"]["logits"]["values"]),
        np.asarray(direct["outputs"]["logits"]["values"]),
    )


def test_modified_model_keeps_text_tokenizer(tmp_path):
    client, source_id = _gpt2_text_client(tmp_path)
    modified = client.modify(
        source_id,
        "interpolate",
        params={"t": 0.5},
        others=[source_id],
    )
    candidate_id = modified["artifact_id"]
    manifest = client.artifact(candidate_id)
    assert manifest["metadata"]["tokenizer"]["sha256"] == client.artifact(source_id)["metadata"]["tokenizer"]["sha256"]

    source = client.forward_text(source_id, text="hello world")
    candidate = client.forward_text(candidate_id, text="hello world")
    assert source["token_ids"] == candidate["token_ids"]
    assert candidate["outputs"]["logits"]["shape"] == source["outputs"]["logits"]["shape"]


def test_llama_text_inference_with_positions(tmp_path):
    client, artifact_id = _llama_text_client(tmp_path)
    result = client.forward_text(
        artifact_id,
        text="hello world",
        positions=[[0, 1]],
        include_values=True,
    )
    ids = result["token_ids"][0]
    assert result["outputs"]["logits"]["shape"] == [1, len(ids), 32]
    assert np.isfinite(np.asarray(result["outputs"]["logits"]["values"])).all()


def test_text_inference_refuses_missing_tokenizer():
    with pytest.raises(InvalidOperationParameters, match="no stored tokenizer"):
        tokenizer_spec_from_manifest({"metadata": {"graph": {}}})


def test_rest_mcp_and_cli_text_forward(tmp_path, capsys):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from uzen.interfaces.api.app import create_app

    client, artifact_id = _gpt2_text_client(tmp_path)

    api = fastapi_testclient.TestClient(create_app(service=client.service))
    response = api.post(
        f"/v1/models/{artifact_id}/forward-text",
        json={"text": "hello world", "include_values": False},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["token_ids"][0]
    assert payload["outputs"]["logits"]["shape"][0] == 1

    server = MCPServer(root=client.service.root)
    listed = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert "uzen.forward_text" in {tool["name"] for tool in listed["result"]["tools"]}
    called = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "uzen.forward_text",
                "arguments": {"artifact_id": artifact_id, "text": "hello world"},
            },
        }
    )
    mcp_payload = json.loads(called["result"]["content"][0]["text"])
    assert mcp_payload["token_ids"][0] == payload["token_ids"][0]

    status = cli_main(
        [
            "--root",
            str(client.service.root),
            "forward",
            artifact_id,
            "--text",
            "hello world",
        ]
    )
    assert status == 0
    cli_payload = json.loads(capsys.readouterr().out)
    assert cli_payload["token_ids"][0] == payload["token_ids"][0]
