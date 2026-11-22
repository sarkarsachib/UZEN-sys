"""Explicit CNN reconstruction parity against a real PyTorch model."""

from __future__ import annotations

import numpy as np
import pytest

from uzen.core.wie.cnn_reconstruction import reconstruct_cnn_model


torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402


class _Cnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.fc1 = nn.Linear(4 * 4 * 4, 5)

    def forward(self, x):  # type: ignore[no-untyped-def]
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.nn.functional.max_pool2d(x, 2)
        return self.fc1(torch.flatten(x, 1))


def _fixture() -> tuple[_Cnn, dict[str, np.ndarray], dict]:
    torch.manual_seed(7)
    model = _Cnn().eval()
    with torch.no_grad():
        model.bn1.running_mean.copy_(torch.tensor([0.1, -0.2, 0.3, 0.0]))
        model.bn1.running_var.copy_(torch.tensor([1.1, 0.9, 1.4, 1.0]))
    arrays = {name: value.detach().cpu().numpy().astype(np.float32) for name, value in model.state_dict().items()}
    spec = {
        "in_channels": 1,
        "height": 8,
        "width": 8,
        "layers": [
            {
                "name": "conv1",
                "kind": "conv2d",
                "out_channels": 4,
                "kernel": 3,
                "padding": 1,
                "batch_norm": True,
                "activation": "relu",
                "pool": {"kernel": 2},
            },
            {"name": "flatten", "kind": "flatten"},
            {"name": "fc1", "kind": "linear", "out_features": 5},
        ],
    }
    source_names = {
        "conv1.weight": "conv1.weight",
        "conv1.bias": "conv1.bias",
        "conv1.bn.weight": "bn1.weight",
        "conv1.bn.bias": "bn1.bias",
        "conv1.bn.running_mean": "bn1.running_mean",
        "conv1.bn.running_var": "bn1.running_var",
        "fc1.weight": "fc1.weight",
        "fc1.bias": "fc1.bias",
    }
    return model, arrays, {"spec": spec, "source_names": source_names}


def test_cnn_reconstruction_matches_torch_forward() -> None:
    model, arrays, kwargs = _fixture()
    result = reconstruct_cnn_model(arrays, kwargs["spec"], source_names=kwargs["source_names"])
    assert result.report.status == "validated", result.report.to_dict()
    assert result.usable and result.package is not None and result.definition is not None

    rng = np.random.default_rng(11)
    x = rng.standard_normal((2, 1, 8, 8)).astype(np.float32)
    from uzen.core.nir.executor import execute

    got = execute(result.definition.graph, {"image": x}, result.package).output("logits")
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


def test_cnn_reconstruction_does_not_invent_missing_parameters() -> None:
    _, arrays, kwargs = _fixture()
    arrays.pop("bn1.bias")
    result = reconstruct_cnn_model(arrays, kwargs["spec"], source_names=kwargs["source_names"])
    assert result.report.status == "unsupported"
    assert "conv1.bn.bias" in result.report.missing
    assert result.package is None


def test_cnn_reconstruction_rejects_shape_mismatch() -> None:
    _, arrays, kwargs = _fixture()
    arrays["fc1.weight"] = arrays["fc1.weight"][:, :3]
    result = reconstruct_cnn_model(arrays, kwargs["spec"], source_names=kwargs["source_names"])
    assert result.report.status == "unsupported"
    assert result.report.shape_errors
