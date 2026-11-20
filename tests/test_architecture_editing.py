"""Architecture composition/surgery through the shared service and interfaces."""

from __future__ import annotations

import json

import pytest

from uzen.errors import GraphValidationError
from uzen.interfaces.python_sdk.client import Uzen


MLP_A = {"input_dim": 8, "hidden_dims": [16], "output_dim": 4}
MLP_B = {"input_dim": 4, "hidden_dims": [8], "output_dim": 2}
TRANSFORMER = {
    "d_model": 16,
    "n_heads": 2,
    "n_layers": 2,
    "d_ff": 32,
    "vocab_size": 64,
    "max_seq_len": 8,
}


def test_compose_persists_validated_artifact_with_lineage(tmp_path):
    client = Uzen(root=tmp_path)
    a = client.generate(family="mlp", spec=MLP_A, seed=1)
    b = client.generate(family="mlp", spec=MLP_B, seed=2)
    result = client.compose_architectures([a["artifact_id"], b["artifact_id"]])
    assert result["artifact_id"]
    assert result["forward_outputs"] == {"out": [2, 2]}
    assert result["components"] == [a["artifact_id"], b["artifact_id"]]
    manifest = client.artifact(result["artifact_id"])
    assert manifest["metadata"]["architecture"]["family"] == "composed"
    lineage = client.lineage(result["artifact_id"])
    assert {a["artifact_id"], b["artifact_id"]}.issubset(set(lineage["ancestors"]))


def test_compose_rejects_shape_mismatch(tmp_path):
    client = Uzen(root=tmp_path)
    a = client.generate(family="mlp", spec=MLP_A, seed=1)
    # a produces 4 features; this expects 8
    bad = client.generate(
        family="mlp", spec={"input_dim": 8, "hidden_dims": [8], "output_dim": 2}, seed=2
    )
    with pytest.raises(GraphValidationError):
        client.compose_architectures([a["artifact_id"], bad["artifact_id"]])


def test_bypass_activation_preserves_surviving_weights(tmp_path):
    client = Uzen(root=tmp_path)
    source = client.generate(family="mlp", spec={"input_dim": 8, "hidden_dims": [16, 16], "output_dim": 4}, seed=0)
    result = client.edit_architecture(
        source["artifact_id"], "bypass", node_id="act0", port="x"
    )
    assert result["artifact_id"]
    assert result["invalidated_parameters"] == []
    assert result["forward_outputs"]["logits"] == [2, 4]
    parent = client.artifact(source["artifact_id"])
    child = client.artifact(result["artifact_id"])
    assert child["metadata"]["graph"]["nodes"]
    assert child["metadata"]["architecture"]["family"] == "mlp"
    assert parent["artifact_id"] != child["artifact_id"]


def test_truncate_restores_family_spec_and_removes_tail(tmp_path):
    client = Uzen(root=tmp_path)
    source = client.generate(family="transformer_decoder", spec=TRANSFORMER, seed=0)
    result = client.edit_architecture(source["artifact_id"], "truncate", keep_layers=1)
    assert result["artifact_id"]
    assert "layers.1.attn_out.weight" in result["invalidated_parameters"]
    manifest = client.artifact(result["artifact_id"])
    assert manifest["metadata"]["architecture"]["spec"]["n_layers"] == 1


def test_substitute_operator_without_new_parameters(tmp_path):
    client = Uzen(root=tmp_path)
    source = client.generate(family="mlp", spec=MLP_A, seed=0)
    result = client.edit_architecture(
        source["artifact_id"], "substitute", node_id="act0", new_op="gelu"
    )
    assert result["artifact_id"]
    assert result["invalidated_parameters"] == []
    assert any("gelu" in change for change in result["changes"])


def test_rest_compose_and_edit(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from uzen.interfaces.api.app import create_app

    client = Uzen(root=tmp_path)
    a = client.generate(family="mlp", spec=MLP_A, seed=1)
    b = client.generate(family="mlp", spec=MLP_B, seed=2)
    rest = TestClient(create_app(root=tmp_path))
    composed = rest.post(
        "/v1/architectures/compose",
        json={"artifact_ids": [a["artifact_id"], b["artifact_id"]]},
    )
    assert composed.status_code == 200, composed.text
    edited = rest.post(
        f"/v1/architectures/{a['artifact_id']}/edit",
        json={"operation": "bypass", "node_id": "act0", "port": "x"},
    )
    assert edited.status_code == 200, edited.text
    assert edited.json()["artifact_id"]


def test_mcp_exposes_architecture_and_campaign_tools(tmp_path):
    from uzen.interfaces.mcp.server import MCPServer

    server = MCPServer(root=tmp_path)
    for name in (
        "uzen.compose_architectures",
        "uzen.modify_architecture",
        "uzen.run_campaign",
        "uzen.get_campaign",
        "uzen.verify_campaign_corpus",
        "uzen.replay_benchmark",
    ):
        assert name in server.tools
    spec = {
        "name": "mcp-campaign",
        "family": "transformer_decoder",
        "base_spec": {
            "d_model": 8,
            "n_heads": 2,
            "n_layers": 1,
            "d_ff": 16,
            "vocab_size": 32,
            "max_seq_len": 8,
        },
        "axes": [{"name": "d_model", "kind": "int", "low": 8, "high": 16, "step": 8}],
        "objectives": [{"name": "parameters", "direction": "min"}],
        "budget": 2,
        "batch_size": 2,
        "strategy": "grid",
        "evaluation": {"batch": 2, "seq": 3},
    }
    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "uzen.run_campaign", "arguments": {"spec": spec}},
        }
    )
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["status"] == "completed"
    assert payload["trials_evaluated"] == 2
