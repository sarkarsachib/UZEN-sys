"""Cross-route STRICT_PLUS enforcement audit tests."""

from __future__ import annotations

import numpy as np
import pytest

from uzen.core.nwpc.context import OperationContext
from uzen.core.nwpc.registry import default_registry
from uzen.core.wpl.compiler import compile_program
from uzen.errors import (
    OperationNotRegistered,
    PolicyViolation,
    ProgramInvalid,
    RuntimeUnavailable,
)
from uzen.policies.strict_plus.effects import TaskScope
from uzen.policies.strict_plus.engine import AuditLog, StrictPlusEngine


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
