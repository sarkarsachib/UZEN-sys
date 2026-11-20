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


def test_export_embeds_graph_and_reloaded_artifact_executes(tmp_path):
    client, source_id, _root = _gpt2_client(tmp_path)
    modified = client.modify(source_id, "scale", params={"factor": 0.5})
    modified_id = modified["artifact_id"]
    expected = client.service.forward_model(
        modified_id, tokens=[[0, 1, 2, 3]], capture="layers"
    )

    export_path = client.service.store.export(
        modified_id, tmp_path / "modified.safetensors"
    )
    reloaded = client.import_weights(export_path)
    reloaded_id = reloaded["artifact_id"]
    manifest = client.artifact(reloaded_id)
    assert manifest["metadata"]["graph"]["name"] == expected.graph_name
    assert (
        manifest["metadata"]["import_report"]["metadata"]["graph_restored_from_metadata"]
        is True
    )

    actual = client.service.forward_model(
        reloaded_id, tokens=[[0, 1, 2, 3]], capture="layers"
    )
    np.testing.assert_array_equal(actual.outputs["logits"], expected.outputs["logits"])
    np.testing.assert_array_equal(
        actual.intermediates["l0_res2"], expected.intermediates["l0_res2"]
    )


def test_forward_captures_transformer_layer_outputs(tmp_path):
    client, artifact_id, _root = _gpt2_client(tmp_path)
    execution = client.forward_model(
        artifact_id,
        tokens=[[0, 1, 2, 3]],
        capture="layers",
        include_values=True,
    )

    assert list(execution["captured"]) == ["l0_res2"]
    assert execution["captured"]["l0_res2"]["shape"] == [1, 4, 8]
    values = np.asarray(execution["captured"]["l0_res2"]["values"])
    assert values.shape == (1, 4, 8)
    assert np.isfinite(values).all()


def test_forward_rejects_unknown_capture_node(tmp_path):
    client, artifact_id, _root = _gpt2_client(tmp_path)
    with pytest.raises(InvalidOperationParameters, match="unknown NIR node"):
        client.forward_model(
            artifact_id, tokens=[[0, 1, 2, 3]], capture=["not_a_node"]
        )


def test_rest_forward_supports_layer_capture(tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from uzen.interfaces.api.app import create_app

    client, artifact_id, _root = _gpt2_client(tmp_path)
    api = fastapi_testclient.TestClient(create_app(service=client.service))
    response = api.post(
        f"/v1/models/{artifact_id}/forward",
        json={
            "tokens": [[0, 1, 2, 3]],
            "capture": ["layers"],
            "include_values": True,
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["captured"]["l0_res2"]["shape"] == [1, 4, 8]
    assert np.asarray(body["captured"]["l0_res2"]["values"]).shape == (1, 4, 8)


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


def test_cli_compare_models_supports_aggregate_thresholds(tmp_path, capsys):
    client, source_id, _root = _gpt2_client(tmp_path)
    changed_id = client.modify(
        source_id, "scale", params={"factor": 0.5}
    )["artifact_id"]

    status = cli_main(
        [
            "--root",
            str(client.service.root),
            "compare-models",
            source_id,
            changed_id,
            "--tokens",
            "[[0,1,2,3]]",
            "--atol",
            "1.0",
            "--rtol",
            "1.0",
            "--max-abs-error-threshold",
            "0.001",
            "--mean-abs-error-threshold",
            "0.001",
            "--max-rel-error-threshold",
            "0.001",
        ]
    )
    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["outputs"]["logits"]["passed"] is True
    assert payload["drift"]["thresholds"] == {
        "max_abs_error": 0.001,
        "mean_abs_error": 0.001,
        "max_rel_error": 0.001,
    }
    assert payload["drift"]["passed"] is False
    assert payload["passed"] is False


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
            "--capture",
            "layers",
            "--values",
        ]
    )
    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["outputs"]["logits"]["shape"] == [1, 4, 16]
    assert payload["captured"]["l0_res2"]["shape"] == [1, 4, 8]
    assert payload["captured"]["l0_res2"]["values"]


def test_cli_and_mcp_persist_quantized_variant(tmp_path, capsys):
    client, source_id, _root = _gpt2_client(tmp_path)
    status = cli_main(
        [
            "--root",
            str(client.service.root),
            "quantize",
            source_id,
            "--scheme",
            "int8",
        ]
    )
    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["parent_artifact_id"] == source_id
    assert payload["quantization"]["schema"] == "uzen.quantization/1"

    server = MCPServer(root=client.service.root)
    listed = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert "uzen.quantize_model" in {tool["name"] for tool in listed["result"]["tools"]}
    called = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "uzen.quantize_model",
                "arguments": {"artifact_id": source_id, "scheme": "int8"},
            },
        }
    )
    mcp_payload = json.loads(called["result"]["content"][0]["text"])
    assert mcp_payload["artifact_id"] == payload["artifact_id"]
    assert mcp_payload["quantization"]["schema"] == "uzen.quantization/1"


def test_cli_and_mcp_forward_support_int8(tmp_path, capsys):
    client, source_id, _root = _gpt2_client(tmp_path)
    status = cli_main(
        [
            "--root",
            str(client.service.root),
            "forward",
            source_id,
            "--tokens",
            "[[0,1,2,3]]",
            "--quantize",
            "int8",
        ]
    )
    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["quantization"]["schema"] == "uzen.quantization/1"
    assert payload["quantization"]["num_rewritten_nodes"] > 0

    server = MCPServer(root=client.service.root)
    listed = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    tools = {tool["name"]: tool for tool in listed["result"]["tools"]}
    assert tools["uzen.forward_model"]["inputSchema"]["properties"]["quantize"]["enum"] == ["int8"]

    called = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "uzen.forward_model",
                "arguments": {
                    "artifact_id": source_id,
                    "tokens": [[0, 1, 2, 3]],
                    "quantize": "int8",
                },
            },
        }
    )
    mcp_payload = json.loads(called["result"]["content"][0]["text"])
    assert mcp_payload["quantization"]["schema"] == "uzen.quantization/1"


def test_cli_and_mcp_compare_quantized_variant(tmp_path, capsys):
    client, source_id, _root = _gpt2_client(tmp_path)
    quantized = client.quantize_model(source_id)
    quantized_id = quantized["artifact_id"]

    status = cli_main(
        [
            "--root",
            str(client.service.root),
            "compare-quantized",
            source_id,
            quantized_id,
            "--tokens",
            "[[0,1,2,3]]",
            "--atol",
            "1.0",
            "--rtol",
            "1.0",
            "--max-abs-error-threshold",
            "1.0",
        ]
    )
    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_vs_quantized"] is True
    assert payload["quantization"] == quantized["quantization"]
    assert payload["drift"]["thresholds"]["max_abs_error"] == 1.0
    assert payload["drift"]["passed"] is True

    server = MCPServer(root=client.service.root)
    listed = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    tools = {tool["name"]: tool for tool in listed["result"]["tools"]}
    assert "uzen.compare_quantized_model" in tools
    assert (
        tools["uzen.compare_quantized_model"]["inputSchema"]["properties"][
            "max_abs_error_threshold"
        ]["minimum"]
        == 0
    )

    called = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "uzen.compare_quantized_model",
                "arguments": {
                    "source_artifact_id": source_id,
                    "quantized_artifact_id": quantized_id,
                    "tokens": [[0, 1, 2, 3]],
                    "atol": 1.0,
                    "rtol": 1.0,
                },
            },
        }
    )
    mcp_payload = json.loads(called["result"]["content"][0]["text"])
    assert mcp_payload["source_vs_quantized"] is True
    assert mcp_payload["quantization"]["schema"] == "uzen.quantization/1"


def test_mcp_forward_and_compare_tools(tmp_path):
    client, source_id, _root = _gpt2_client(tmp_path)
    candidate_id = client.modify(
        source_id, "interpolate", params={"t": 0.5}, others=[source_id]
    )["artifact_id"]
    server = MCPServer(root=client.service.root)

    listed = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    tools = {tool["name"]: tool for tool in listed["result"]["tools"]}
    assert {"uzen.forward_model", "uzen.compare_models"} <= set(tools)
    assert (
        tools["uzen.compare_models"]["inputSchema"]["properties"][
            "max_abs_error_threshold"
        ]["minimum"]
        == 0
    )

    forwarded = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "uzen.forward_model",
                "arguments": {
                    "artifact_id": source_id,
                    "tokens": [[0, 1, 2, 3]],
                    "capture": ["layers"],
                    "include_values": True,
                },
            },
        }
    )
    payload = json.loads(forwarded["result"]["content"][0]["text"])
    assert payload["outputs"]["logits"]["shape"] == [1, 4, 16]
    assert payload["captured"]["l0_res2"]["shape"] == [1, 4, 8]

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
                    "max_abs_error_threshold": 1.0,
                    "mean_abs_error_threshold": 1.0,
                    "max_rel_error_threshold": 1e9,
                },
            },
        }
    )
    comparison = json.loads(compared["result"]["content"][0]["text"])
    assert comparison["passed"] is True
    assert comparison["drift"]["thresholds"] == {
        "max_abs_error": 1.0,
        "mean_abs_error": 1.0,
        "max_rel_error": 1e9,
    }
    assert comparison["drift"]["passed"] is True
