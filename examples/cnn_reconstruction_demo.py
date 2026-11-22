"""Reconstruct a real PyTorch CNN state dict and verify forward parity.

Run: ``.venv/bin/python examples/cnn_reconstruction_demo.py``
"""

from __future__ import annotations

import numpy as np

from uzen.core.nir.executor import execute
from uzen.core.wie.cnn_reconstruction import reconstruct_cnn_model

try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover - optional runtime
    raise SystemExit(f"PyTorch is required for this demo: {exc}")


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.fc1 = nn.Linear(4 * 4 * 4, 5)

    def forward(self, x):  # type: ignore[no-untyped-def]
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.nn.functional.max_pool2d(x, 2)
        return self.fc1(torch.flatten(x, 1))


def main() -> int:
    torch.manual_seed(5)
    model = Net().eval()
    with torch.no_grad():
        model.bn1.running_mean.copy_(torch.tensor([0.1, -0.2, 0.3, 0.0]))
        model.bn1.running_var.copy_(torch.tensor([1.1, 0.9, 1.4, 1.0]))
    arrays = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
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
    result = reconstruct_cnn_model(arrays, spec, source_names=source_names)
    print(f"status: {result.report.status}")
    print(f"mapped: {result.report.mapped_tensor_count}/{len(arrays)} source tensors")
    print(f"graph: {result.report.validation['num_nodes']} nodes, {result.report.validation['num_params']} parameters")
    if not result.usable or result.definition is None or result.package is None:
        print("reconstruction failed:")
        for reason in result.report.reasons:
            print(f"  - {reason}")
        return 1
    rng = np.random.default_rng(11)
    x = rng.standard_normal((2, 1, 8, 8)).astype(np.float32)
    got = execute(result.definition.graph, {"image": x}, result.package).output("logits")
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).numpy()
    error = float(np.max(np.abs(got - expected)))
    print(f"forward parity max_abs_error: {error:.3e}")
    print("PASS" if error < 1e-5 else "FAIL")
    return 0 if error < 1e-5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
