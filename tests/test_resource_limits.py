"""Resource-protection tests: operation, tensor, total-byte and artifact budgets."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from uzen.core.nir.executor import ExecutionLimits, execute
from uzen.core.nir.graph import Graph
from uzen.core.nir.types import Ref, TensorType
from uzen.core.nws.store import ArtifactStore
from uzen.core.nws.tensors import WeightPackage
from uzen.errors import BudgetExhausted


def _graph() -> Graph:
    graph = Graph(name="identity", inputs={"x": TensorType.of((2,))})
    graph.add_node("id", "identity", {"x": Ref("input:x")})
    graph.set_output("y", Ref("id"))
    return graph


def test_operation_budget_is_enforced():
    with pytest.raises(BudgetExhausted):
        execute(
            _graph(),
            {"x": np.zeros(2, dtype=np.float32)},
            limits=ExecutionLimits(max_operations=0),
        )


def test_per_tensor_budget_is_enforced():
    with pytest.raises(BudgetExhausted):
        execute(
            _graph(),
            {"x": np.zeros(2, dtype=np.float32)},
            limits=ExecutionLimits(max_tensor_bytes=1),
        )


def test_total_working_set_budget_is_enforced():
    with pytest.raises(BudgetExhausted):
        execute(
            _graph(),
            {"x": np.zeros(2, dtype=np.float32)},
            limits=ExecutionLimits(max_total_bytes=4),
        )


def test_cancellation_is_checked_at_node_boundaries():
    event = threading.Event()
    event.set()
    with pytest.raises(BudgetExhausted):
        execute(
            _graph(),
            {"x": np.zeros(2, dtype=np.float32)},
            limits=ExecutionLimits(cancel_event=event),
        )


def test_timeout_is_enforced():
    with pytest.raises(BudgetExhausted):
        execute(
            _graph(),
            {"x": np.zeros(2, dtype=np.float32)},
            limits=ExecutionLimits(timeout_seconds=1e-9),
        )


def test_artifact_byte_budget_rejects_before_publication(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts", max_artifact_bytes=4)
    package = WeightPackage.from_arrays({"w": np.zeros((2, 2), dtype=np.float32)})

    with pytest.raises(BudgetExhausted):
        store.put(package, operation="init")
    assert store.list() == []


def test_artifact_env_budget_is_honoured(tmp_path, monkeypatch):
    monkeypatch.setenv("UZEN_MAX_ARTIFACT_BYTES", "4")
    store = ArtifactStore(tmp_path / "artifacts")
    package = WeightPackage.from_arrays({"w": np.zeros((2, 2), dtype=np.float32)})

    with pytest.raises(BudgetExhausted):
        store.put(package, operation="init")
