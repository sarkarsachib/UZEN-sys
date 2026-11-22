"""End-to-end LLM planner wiring through a real local OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from uzen.core.ane.llm import LLMConfig, LLMPlanner
from uzen.core.ane.planner import JobRequest
from uzen.core.nws.families import build_architecture
from uzen.errors import UzenError
from uzen.jobs.service import JobService


class _Handler(BaseHTTPRequestHandler):
    proposal: dict = {}

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        body = json.dumps(
            {
                "choices": [
                    {"message": {"content": json.dumps(self.proposal)}}
                ]
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        return


@pytest.fixture()
def endpoint(monkeypatch):
    proposal = {
        "rationale": ["scale is the requested deterministic weight operation"],
        "stages": [
            {
                "name": "scaled",
                "op": "w.arithmetic.scale",
                "args": ["base"],
                "params": {"factor": 0.25},
            }
        ],
        "search_space": [],
    }
    _Handler.proposal = proposal
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    monkeypatch.setenv("UZEN_LLM_BASE_URL", base)
    monkeypatch.setenv("UZEN_LLM_API_KEY", "test-key-not-logged")
    monkeypatch.setenv("UZEN_LLM_MODEL", "test-model")
    try:
        yield proposal
    finally:
        server.shutdown()
        server.server_close()


def test_llm_planner_is_used_by_a_real_job(tmp_path, endpoint):
    service = JobService(tmp_path / "ws")
    definition = build_architecture("mlp", {"input_dim": 4, "hidden_dims": [8], "output_dim": 3})
    package = definition.build_package(seed=3)
    record = service.store.put(package, operation="init")
    request = {
        "task_scope": "weight_modification",
        "objective": "scale the input weights by one quarter",
        "inputs": [{"name": "base", "artifact_id": record.artifact_id}],
        "options": {"planner": "llm"},
        "evaluation_plan": {"metric": "finite"},
        "budget": {"candidates": 1, "evaluations": 1},
    }
    job = service.run(service.submit(request)["id"])
    assert job["state"] == "SUCCEEDED", job
    assert job["plan"]["rationale"][0].startswith("LLM-proposed plan")
    produced = service.get_package(job["result"]["output_artifact_id"])
    assert produced.tensors["head.weight"].shape == package.tensors["head.weight"].shape


def test_llm_planner_does_not_silently_fallback_when_unconfigured(monkeypatch):
    for name in ("UZEN_LLM_BASE_URL", "UZEN_LLM_API_KEY", "UZEN_LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    request = JobRequest.from_dict(
        {"task_scope": "weight_modification", "objective": "scale", "inputs": [{"name": "base"}]}
    )
    with pytest.raises(UzenError, match="no LLM endpoint is configured"):
        LLMPlanner(config=None).propose(request, ["base"])


def test_llm_planner_config_reads_environment(monkeypatch):
    monkeypatch.setenv("UZEN_LLM_BASE_URL", "http://localhost:1234/v1/")
    monkeypatch.setenv("UZEN_LLM_API_KEY", "secret")
    monkeypatch.setenv("UZEN_LLM_MODEL", "model")
    config = LLMConfig.from_env()
    assert config is not None
    assert config.base_url == "http://localhost:1234/v1"
