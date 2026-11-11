"""P2: execute reconstructed model artifacts and validate weight modifications."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tests.test_gpt2_reconstruction import _write_gpt2
from tests.test_llama_reconstruction import _write_llama
from uzen.errors import InvalidOperationParameters
from uzen.infrastructure.storage.safetensors_io import write_safetensors
from uzen.interfaces.cli.main import main as cli_main
from uzen.interfaces.mcp.server import MCPServer
from uzen.interfaces.python_sdk.client import Uzen


def _gpt2_client(tmp_path):
    root, _arrays = _write_gpt2(tmp_path / "gpt2-model")
    client = Uzen(root=tmp_path / "gpt2-workspace")
    imported = client.import_model(root)
    return client, imported["artifact_id"], root


def test_forward_reconstructed_model_returns_values(tmp_path):
    client, artifact_id, _root = _gpt2_client(tmp_path)
    execution = client.forward_model(artifact_id, tokens=[[0, 1, 2, 3]], include_values=True)

    assert execution["artifact_id"] == artifact_id
    assert execution["graph_name"] == "transformer_decoder_L1"
    assert execution["outputs"]["logits"]["shape"] == [1, 4, 16]
    assert np.asarray(execution["outputs"]["logits"]["values"]).shape == (1, 4, 16)


def test_weight_modification_preserves_graph_and_can_be_compared(tmp_path):
    client, source_id, _root = _gpt2_client(tmp_path)
    modified = client.modify(
        source_id,
        "interpolate",
        params={"t": 0.5},
        others=[source_id],
    )
    candidate_id = modified["artifact_id"]
    manifest = client.artifact(candidate_id)
    assert manifest["metadata"]["graph"]["name"] == "transformer_decoder_L1"
    assert manifest["lineage"]["parents"] == [source_id]

    comparison = client.compare_models(
        source_id,
        candidate_id,
        tokens=[[0, 1, 2, 3]],
        atol=1e-6,
        rtol=1e-6,
    )
    assert comparison["passed"], comparison
    assert comparison["tolerances"] == {"atol": 1e-6, "rtol": 1e-6}
    assert comparison["outputs"]["logits"]["max_abs_error"] <= 1e-6

    changed = client.modify(source_id, "scale", params={"factor": 0.5})
    changed_comparison = client.compare_models(
        source_id,
        changed["artifact_id"],
        tokens=[[0, 1, 2, 3]],
    )
    assert changed_comparison["passed"] is False
    assert changed_comparison["outputs"]["logits"]["max_abs_error"] > 0


def test_llama_graph_executes_with_explicit_positions(tmp_path):
    root, _arrays = _write_llama(tmp_path / "llama-model")
    client = Uzen(root=tmp_path / "llama-workspace")
    imported = client.import_model(root)
    execution = client.forward_model(
        imported["artifact_id"],
        tokens=[[0, 1, 2, 3]],
        positions=[[0, 1, 2, 3]],
        include_values=True,
    )
    assert execution["graph_name"] == "llama_L1"
    assert execution["outputs"]["logits"]["shape"] == [1, 4, 32]
    assert np.isfinite(np.asarray(execution["outputs"]["logits"]["values"])).all()


def test_forward_refuses_artifact_without_stored_graph(tmp_path):
    path = tmp_path / "plain.safetensors"
    write_safetensors(path, {"x": np.zeros((4, 4), dtype=np.float32)})
    client = Uzen(root=tmp_path / "plain-workspace")
    artifact_id = client.import_weights(path)["artifact_id"]
    with pytest.raises(InvalidOperationParameters, match="no stored NIR graph"):
        client.forward_model(artifact_id, tokens=[[0, 1, 2, 3]])


def test_rest_forward_and_compare_routes(tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from uzen.interfaces.api.app import create_app

    client, source_id, _root = _gpt2_client(tmp_path)
    api = fastapi_testclient.TestClient(create_app(service=client.service))

    forward = api.post(
        f"/v1/models/{source_id}/forward",
        json={"tokens": [[0, 1, 2, 3]], "include_values": False},
    )
    assert forward.status_code == 200, forward.text
    assert forward.json()["outputs"]["logits"]["shape"] == [1, 4, 16]

    candidate_id = client.modify(
        source_id, "interpolate", params={"t": 0.5}, others=[source_id]
    )["artifact_id"]
    compared = api.post(
        "/v1/models/compare",
        json={
            "source_artifact_id": source_id,
            "candidate_artifact_id": candidate_id,
            "tokens": [[0, 1, 2, 3]],
        },
    )
    assert compared.status_code == 200, compared.text
    assert compared.json()["passed"] is True


def test_cli_forward_runs_real_model(tmp_path, capsys):
    client, source_id, _root = _gpt2_client(tmp_path)
    status = cli_main(
        [
            "--root",
            str(client.service.root),
            "forward",
            source_id,
            "--tokens",
            "[[0,1,2,3]]",
        ]
    )
    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["outputs"]["logits"]["shape"] == [1, 4, 16]


def test_mcp_forward_and_compare_tools(tmp_path):
    client, source_id, _root = _gpt2_client(tmp_path)
    candidate_id = client.modify(
        source_id, "interpolate", params={"t": 0.5}, others=[source_id]
    )["artifact_id"]
    server = MCPServer(root=client.service.root)

    listed = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    names = {tool["name"] for tool in listed["result"]["tools"]}
    assert {"uzen.forward_model", "uzen.compare_models"} <= names

    forwarded = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "uzen.forward_model",
                "arguments": {"artifact_id": source_id, "tokens": [[0, 1, 2, 3]]},
            },
        }
    )
    payload = json.loads(forwarded["result"]["content"][0]["text"])
    assert payload["outputs"]["logits"]["shape"] == [1, 4, 16]

    compared = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "uzen.compare_models",
                "arguments": {
                    "source_artifact_id": source_id,
                    "candidate_artifact_id": candidate_id,
                    "tokens": [[0, 1, 2, 3]],
                },
            },
        }
    )
    comparison = json.loads(compared["result"]["content"][0]["text"])
    assert comparison["passed"] is True
