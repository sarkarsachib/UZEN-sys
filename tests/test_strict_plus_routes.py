"""Cross-route STRICT_PLUS enforcement audit tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from uzen.core.ane.planner import JobRequest
from uzen.core.nwpc.context import OperationContext
from uzen.core.nwpc.registry import default_registry
from uzen.core.wpl.compiler import compile_program
from uzen.errors import (
    InvalidOperationParameters,
    OperationNotRegistered,
    PolicyViolation,
    ProgramInvalid,
    RuntimeUnavailable,
)
from uzen.interfaces.cli.main import main as cli_main
from uzen.interfaces.python_sdk.client import Uzen
from uzen.jobs.service import JobService
from uzen.policies.strict_plus.effects import TaskScope
from uzen.policies.strict_plus.engine import AuditLog, StrictPlusEngine


def test_cli_submit_rejects_a_prohibited_operation(tmp_path, capsys):
    """The CLI has no separate mutation path; its refusal must come from the service."""
    request = {
        "task_scope": "weight_modification",
        "objective": "try to train",
        "operations": [{"op": "w.train.finetune", "params": {}}],
    }
    code = cli_main(
        ["--root", str(tmp_path / "cli-workspace"), "job", "submit", "--json", json.dumps(request)]
    )
    assert code == 2
    captured = capsys.readouterr()
    assert '"error"' in captured.err


def test_cli_submit_rejects_an_operation_outside_the_scope(tmp_path, capsys):
    request = {
        "task_scope": "inspection",
        "objective": "try to mutate during inspection",
        "operations": [{"op": "w.arithmetic.scale", "params": {"factor": 2.0}}],
    }
    code = cli_main(
        ["--root", str(tmp_path / "cli-workspace-2"), "job", "submit", "--json", json.dumps(request)]
    )
    assert code == 2
    assert "error" in capsys.readouterr().err


def test_registry_route_rejects_training_operation(mlp_package):
    with pytest.raises(OperationNotRegistered):
        default_registry().invoke(
            "w.train.finetune",
            [mlp_package],
            {},
            scope=TaskScope.WEIGHT_MODIFICATION,
        )


def test_nwpc_context_rejects_grad_tracking_write():
    torch = pytest.importorskip("torch")
    ctx = OperationContext(
        scope=TaskScope.WEIGHT_MODIFICATION,
        seed=0,
        engine=StrictPlusEngine(audit=AuditLog()),
        operation="audit.test",
    )
    with pytest.raises(PolicyViolation):
        ctx.emit(torch.tensor([1.0], requires_grad=True), name="w")


def test_wpl_route_rejects_transform_outside_scope():
    program = """
    program "audit" {
      input a: package
      step s = scale(a)
      output o = s
    }
    """
    with pytest.raises(ProgramInvalid) as exc:
        compile_program(program, scope=TaskScope.INSPECTION)
    assert "not authorized" in str(exc.value)


def test_worker_capabilities_declare_strict_plus():
    from uzen.workers.protocol.messages import detect_capabilities

    assert detect_capabilities()["strict_plus"] is True


def test_runtime_route_refuses_grad_tracking_tensor():
    torch = pytest.importorskip("torch")
    from uzen.runtimes.pytorch_runtime import PyTorchRuntime, torch_available

    if not torch_available():
        pytest.skip("torch is not installed")
    with pytest.raises(RuntimeUnavailable):
        PyTorchRuntime().tensor_to_numpy(torch.tensor([1.0], requires_grad=True))


def _training_request(artifact_id: str = "0" * 64) -> dict:
    return {
        "task_scope": "weight_modification",
        "objective": "fine-tune with gradients",
        "inputs": [{"name": "base", "artifact_id": artifact_id}],
        "operations": [{"op": "w.train.finetune", "params": {"lr": 0.1}}],
    }


def _generated_artifact(tmp_path) -> tuple[Uzen, str]:
    client = Uzen(root=tmp_path / "generated")
    generated = client.generate(
        family="mlp",
        spec={"input_dim": 4, "hidden_dims": [8], "output_dim": 2},
        seed=0,
    )
    return client, generated["artifact_id"]


def test_api_job_route_rejects_prohibited_operation(tmp_path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from uzen.interfaces.api.app import create_app

    service = JobService(tmp_path / "api-workspace")
    api = fastapi_testclient.TestClient(create_app(service=service))
    response = api.post("/v1/jobs", json=_training_request())

    assert response.status_code == 404, response.text
    assert response.json()["detail"]["code"] == "operation.not_registered"
    assert service.list() == []


def test_sdk_routes_reject_prohibited_operation(tmp_path):
    client, artifact_id = _generated_artifact(tmp_path)

    with pytest.raises(OperationNotRegistered):
        client.submit_job(_training_request(artifact_id))
    with pytest.raises(InvalidOperationParameters, match="unknown modification method"):
        client.modify(artifact_id, "finetune")


def test_cli_modify_route_reports_prohibited_operation(tmp_path, capsys):
    client, artifact_id = _generated_artifact(tmp_path)
    status = cli_main(
        ["--root", str(client.service.root), "modify", artifact_id, "finetune"]
    )

    assert status == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["error"]["code"] == "operation.invalid_parameters"


def test_ane_planner_rejects_prohibited_operation(tmp_path):
    service = JobService(tmp_path / "ane-workspace")
    request = JobRequest.from_dict(_training_request())
    with pytest.raises(OperationNotRegistered):
        service.ane.plan(request)


def test_worker_never_claims_prohibited_operation(tmp_path):
    from uzen.workers.daemon import WorkerDaemon

    daemon = WorkerDaemon(root=tmp_path / "worker-workspace", worker_id="w-audit")
    with pytest.raises(OperationNotRegistered):
        daemon.service.submit(_training_request())
    assert daemon.run_once()["executed"] is None
