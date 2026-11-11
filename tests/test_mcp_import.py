"""MCP model-import tool tests."""

from __future__ import annotations

import json

from tests.test_gpt2_reconstruction import _write_gpt2
from uzen.interfaces.mcp.server import MCPServer


def test_mcp_import_model_returns_report_and_progress(tmp_path):
    root, _arrays = _write_gpt2(tmp_path / "mcp-gpt2")
    server = MCPServer(root=tmp_path / "mcp")

    listed = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    names = {tool["name"] for tool in listed["result"]["tools"]}
    assert "uzen.import_model" in names
    schema = next(t for t in listed["result"]["tools"] if t["name"] == "uzen.import_model")
    assert "source" in schema["inputSchema"]["properties"]
    assert "repo_id" in schema["inputSchema"]["properties"]

    progress_events: list[dict] = []
    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "uzen.import_model",
                "arguments": {"source": str(root)},
                "_meta": {"progressToken": "import-1"},
            },
        },
        emit=progress_events.append,
    )
    assert "result" in response, response
    assert progress_events
    assert all(event["method"] == "notifications/progress" for event in progress_events)
    assert all(event["params"]["progressToken"] == "import-1" for event in progress_events)
    assert [event["params"]["progress"] for event in progress_events] == list(
        range(1, len(progress_events) + 1)
    )
    assert any("tensor_mapping" in event["params"]["message"] for event in progress_events)
    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["compatibility"]["overall"] == "validated"
    assert payload["report"]["progress"]
    assert payload["report"]["reconstruction"]["status"] == "validated"
    assert payload["report"]["reconstruction"]["graph"]["name"] == "transformer_decoder_L1"


def test_mcp_import_model_returns_typed_error(tmp_path):
    server = MCPServer(root=tmp_path / "mcp-errors")
    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "uzen.import_model", "arguments": {}},
        }
    )
    assert response["result"]["isError"] is True
    error = json.loads(response["result"]["content"][0]["text"])
    assert error["code"]
    assert "source" in error["message"]
