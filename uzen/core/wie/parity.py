"""Reference-logit parity checks for reconstructed Hugging Face models.

This module deliberately keeps ``transformers`` and ``torch`` optional.  It is
used by the live validation tests, not by the core import path, so a normal
UZEN installation does not need a deep-learning runtime just to inspect and
map weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from uzen.core.nir.executor import execute
from uzen.core.nir.graph import Graph
from uzen.core.nws.tensors import WeightPackage
from uzen.core.wie.model_execution import layer_node_ids
from uzen.errors import RuntimeUnavailable


def _comparison_report(
    actual: np.ndarray,
    reference: np.ndarray,
    *,
    reference_name: str,
    source_path: Path,
    tokens: np.ndarray,
    atol: float,
    rtol: float,
    positions: np.ndarray | None = None,
) -> dict[str, Any]:
    tolerance = {"atol": atol, "rtol": rtol}
    report: dict[str, Any] = {
        "reference": reference_name,
        "source": str(source_path),
        "tolerances": tolerance,
        "tokens": tokens.tolist(),
        "actual_shape": list(actual.shape),
        "reference_shape": list(reference.shape),
    }
    if positions is not None:
        report["positions"] = positions.tolist()
    if actual.shape != reference.shape:
        report.update({"passed": False, "allclose": False, "error": "shape mismatch"})
        return report

    difference = np.abs(actual - reference)
    relative = difference / np.maximum(np.abs(reference), 1e-8)
    allclose = bool(np.allclose(actual, reference, atol=atol, rtol=rtol))
    report.update(
        {
            "passed": allclose,
            "allclose": allclose,
            "max_abs_error": float(difference.max()),
            "mean_abs_error": float(difference.mean()),
            "max_rel_error": float(relative.max()),
            "num_elements": int(difference.size),
        }
    )
    return report


def compare_gpt2_logits(
    graph: Graph,
    package: WeightPackage,
    source: str | Path,
    *,
    tokens: Sequence[Sequence[int]] | None = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    """Compare UZEN's reconstructed GPT-2 logits against Hugging Face.

    The source must be a local checkpoint directory.  No Hub request is made:
    live tests download the fixture explicitly and then pass its snapshot path.
    Both arrays are evaluated in float32 under inference-only, no-gradient
    execution.  The returned document records the exact tolerances and errors
    so a caller never has to infer that parity was checked from a success flag
    alone.
    """

    if atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be non-negative")
    try:
        import torch
        from transformers import GPT2LMHeadModel
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeUnavailable(
            "GPT-2 parity requires optional packages: install transformers and torch"
        ) from exc

    token_array = (
        np.asarray(tokens, dtype=np.int64)
        if tokens is not None
        else np.array([[0, 1, 2, 3]], dtype=np.int64)
    )
    if token_array.ndim != 2:
        raise ValueError("tokens must have shape (batch, sequence)")
    if token_array.size == 0:
        raise ValueError("tokens must not be empty")
    if int(token_array.min()) < 0:
        raise ValueError("tokens must be non-negative")

    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"local reference checkpoint not found: {source_path}")

    model = GPT2LMHeadModel.from_pretrained(str(source_path), local_files_only=True).eval()
    with torch.no_grad():
        reference = model(input_ids=torch.as_tensor(token_array)).logits.detach().cpu().numpy()
    actual = execute(graph, {"tokens": token_array}, package).output("logits")
    return _comparison_report(
        actual,
        reference,
        reference_name="transformers.GPT2LMHeadModel",
        source_path=source_path,
        tokens=token_array,
        atol=atol,
        rtol=rtol,
    )


def _activation_errors(
    actual: np.ndarray,
    reference: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if actual.shape != reference.shape:
        return {
            "passed": False,
            "error": "shape mismatch",
            "actual_shape": list(actual.shape),
            "reference_shape": list(reference.shape),
        }
    actual64 = actual.astype(np.float64)
    reference64 = reference.astype(np.float64)
    difference = np.abs(actual64 - reference64)
    relative = difference / np.maximum(np.abs(reference64), 1e-8)
    passed = bool(np.allclose(actual, reference, atol=atol, rtol=rtol))
    return {
        "passed": passed,
        "actual_shape": list(actual.shape),
        "reference_shape": list(reference.shape),
        "max_abs_error": float(difference.max()) if difference.size else 0.0,
        "mean_abs_error": float(difference.mean()) if difference.size else 0.0,
        "max_rel_error": float(relative.max()) if relative.size else 0.0,
        "num_elements": int(difference.size),
    }


def _activation_report(
    layers: list[dict[str, Any]],
    *,
    reference_name: str,
    source_path: Path,
    tokens: np.ndarray,
    atol: float,
    rtol: float,
    positions: np.ndarray | None = None,
) -> dict[str, Any]:
    numeric = [layer["max_abs_error"] for layer in layers if "max_abs_error" in layer]
    report: dict[str, Any] = {
        "reference": reference_name,
        "source": str(source_path),
        "tolerances": {"atol": atol, "rtol": rtol},
        "tokens": tokens.tolist(),
        "num_layers": len(layers),
        "layers": layers,
        "passed": bool(layers) and all(layer["passed"] for layer in layers),
        "max_abs_error": max(numeric) if numeric else None,
    }
    if positions is not None:
        report["positions"] = positions.tolist()
    return report


def _block_outputs(
    blocks: Sequence[Any],
    call: Any,
) -> list[np.ndarray]:
    """Run a Hugging Face model and capture each transformer block's output."""
    captured: list[np.ndarray] = []
    handles = []

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        value = output[0] if isinstance(output, (tuple, list)) else output
        captured.append(value.detach().cpu().numpy())

    for block in blocks:
        handles.append(block.register_forward_hook(hook))
    try:
        call()
    finally:
        for handle in handles:
            handle.remove()
    return captured


def _compare_block_outputs(
    captured: Mapping[str, np.ndarray],
    block_outputs: Sequence[np.ndarray],
    node_ids: Sequence[str],
    *,
    atol: float,
    rtol: float,
) -> list[dict[str, Any]]:
    if len(block_outputs) != len(node_ids):
        raise ValueError(
            "Hugging Face block count differs from reconstructed layers: "
            f"{len(block_outputs)} blocks for {len(node_ids)} layers"
        )
    layers: list[dict[str, Any]] = []
    for index, node_id in enumerate(node_ids):
        reference_array = block_outputs[index]
        actual = captured.get(node_id)
        layer: dict[str, Any] = {"layer": index, "node": node_id}
        if actual is None:
            layer.update({"passed": False, "error": "missing captured activation"})
        else:
            layer.update(_activation_errors(actual, reference_array, atol=atol, rtol=rtol))
        layers.append(layer)
    return layers


def compare_gpt2_activations(
    graph: Graph,
    package: WeightPackage,
    source: str | Path,
    *,
    tokens: Sequence[Sequence[int]] | None = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    """Compare each reconstructed GPT-2 block output with Hugging Face hidden states.

    The NIR graph exposes block outputs as ``l<N>_res2``. They are compared with
    ``hidden_states[N + 1]`` from ``GPT2LMHeadModel(..., output_hidden_states=True)``.
    This checks internal activations, not only final logits.
    """
    if atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be non-negative")
    try:
        import torch
        from transformers import GPT2LMHeadModel
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeUnavailable(
            "GPT-2 activation parity requires optional packages: install transformers and torch"
        ) from exc

    token_array = (
        np.asarray(tokens, dtype=np.int64)
        if tokens is not None
        else np.array([[0, 1, 2, 3]], dtype=np.int64)
    )
    if token_array.ndim != 2:
        raise ValueError("tokens must have shape (batch, sequence)")
    if token_array.size == 0:
        raise ValueError("tokens must not be empty")
    if int(token_array.min()) < 0:
        raise ValueError("tokens must be non-negative")

    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"local reference checkpoint not found: {source_path}")
    node_ids = layer_node_ids(graph)
    if not node_ids:
        raise ValueError("graph has no reconstructed transformer layer outputs (l<N>_res2)")

    model = GPT2LMHeadModel.from_pretrained(str(source_path), local_files_only=True).eval()
    blocks = getattr(getattr(model, "transformer", None), "h", None)
    if blocks is None:
        raise ValueError("Hugging Face GPT-2 model does not expose transformer blocks")
    with torch.no_grad():
        block_outputs = _block_outputs(
            blocks, lambda: model(input_ids=torch.as_tensor(token_array))
        )
    captured = execute(
        graph, {"tokens": token_array}, package, capture=node_ids
    ).intermediates
    return _activation_report(
        _compare_block_outputs(captured, block_outputs, node_ids, atol=atol, rtol=rtol),
        reference_name="transformers.GPT2LMHeadModel",
        source_path=source_path,
        tokens=token_array,
        atol=atol,
        rtol=rtol,
    )


def compare_llama_logits(
    graph: Graph,
    package: WeightPackage,
    source: str | Path,
    *,
    tokens: Sequence[Sequence[int]] | None = None,
    positions: Sequence[Sequence[int]] | Sequence[int] | None = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    """Compare reconstructed Llama logits against ``transformers.LlamaForCausalLM``.

    ``positions`` is optional; when omitted it is ``arange(sequence_length)``
    repeated for each batch item.  When supplied, the same positions are passed
    to both UZEN and the Hugging Face reference, so RoPE and position handling
    are compared rather than silently diverging.  The result records the exact
    tokens, positions, tolerances, shapes, and measured absolute/relative errors.
    """

    if atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be non-negative")
    try:
        import torch
        from transformers import LlamaForCausalLM
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeUnavailable(
            "Llama parity requires optional packages: install transformers and torch"
        ) from exc

    token_array = (
        np.asarray(tokens, dtype=np.int64)
        if tokens is not None
        else np.array([[0, 1, 2, 3]], dtype=np.int64)
    )
    if token_array.ndim != 2:
        raise ValueError("tokens must have shape (batch, sequence)")
    if token_array.size == 0:
        raise ValueError("tokens must not be empty")
    if int(token_array.min()) < 0:
        raise ValueError("tokens must be non-negative")

    batch, seq = token_array.shape
    if positions is None:
        position_array = np.tile(np.arange(seq, dtype=np.int64), (batch, 1))
    else:
        position_array = np.asarray(positions, dtype=np.int64)
        if position_array.ndim == 1:
            position_array = position_array.reshape(1, -1)
        if position_array.ndim != 2:
            raise ValueError("positions must be rank 1 or 2")
        if position_array.shape[0] == 1 and batch > 1:
            position_array = np.repeat(position_array, batch, axis=0)
        if position_array.shape != (batch, seq):
            raise ValueError(
                f"positions must have shape {(batch, seq)}, got {position_array.shape}"
            )
        if int(position_array.min()) < 0:
            raise ValueError("positions must be non-negative")

    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"local reference checkpoint not found: {source_path}")

    model = LlamaForCausalLM.from_pretrained(str(source_path), local_files_only=True).eval()
    with torch.no_grad():
        reference = model(
            input_ids=torch.as_tensor(token_array),
            position_ids=torch.as_tensor(position_array),
        ).logits.detach().cpu().numpy()
    actual = execute(
        graph, {"tokens": token_array, "positions": position_array}, package
    ).output("logits")
    return _comparison_report(
        actual,
        reference,
        reference_name="transformers.LlamaForCausalLM",
        source_path=source_path,
        tokens=token_array,
        positions=position_array,
        atol=atol,
        rtol=rtol,
    )


def compare_llama_activations(
    graph: Graph,
    package: WeightPackage,
    source: str | Path,
    *,
    tokens: Sequence[Sequence[int]] | None = None,
    positions: Sequence[Sequence[int]] | Sequence[int] | None = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    """Compare each reconstructed Llama block output with Hugging Face hidden states."""
    if atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be non-negative")
    try:
        import torch
        from transformers import LlamaForCausalLM
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeUnavailable(
            "Llama activation parity requires optional packages: install transformers and torch"
        ) from exc

    token_array = (
        np.asarray(tokens, dtype=np.int64)
        if tokens is not None
        else np.array([[0, 1, 2, 3]], dtype=np.int64)
    )
    if token_array.ndim != 2:
        raise ValueError("tokens must have shape (batch, sequence)")
    if token_array.size == 0:
        raise ValueError("tokens must not be empty")
    if int(token_array.min()) < 0:
        raise ValueError("tokens must be non-negative")

    batch, seq = token_array.shape
    if positions is None:
        position_array = np.tile(np.arange(seq, dtype=np.int64), (batch, 1))
    else:
        position_array = np.asarray(positions, dtype=np.int64)
        if position_array.ndim == 1:
            position_array = position_array.reshape(1, -1)
        if position_array.ndim != 2:
            raise ValueError("positions must be rank 1 or 2")
        if position_array.shape[0] == 1 and batch > 1:
            position_array = np.repeat(position_array, batch, axis=0)
        if position_array.shape != (batch, seq):
            raise ValueError(
                f"positions must have shape {(batch, seq)}, got {position_array.shape}"
            )
        if int(position_array.min()) < 0:
            raise ValueError("positions must be non-negative")

    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"local reference checkpoint not found: {source_path}")
    node_ids = layer_node_ids(graph)
    if not node_ids:
        raise ValueError("graph has no reconstructed transformer layer outputs (l<N>_res2)")

    model = LlamaForCausalLM.from_pretrained(str(source_path), local_files_only=True).eval()
    blocks = getattr(getattr(model, "model", None), "layers", None)
    if blocks is None:
        raise ValueError("Hugging Face Llama model does not expose decoder layers")
    with torch.no_grad():
        block_outputs = _block_outputs(
            blocks,
            lambda: model(
                input_ids=torch.as_tensor(token_array),
                position_ids=torch.as_tensor(position_array),
            ),
        )
    captured = execute(
        graph,
        {"tokens": token_array, "positions": position_array},
        package,
        capture=node_ids,
    ).intermediates
    return _activation_report(
        _compare_block_outputs(captured, block_outputs, node_ids, atol=atol, rtol=rtol),
        reference_name="transformers.LlamaForCausalLM",
        source_path=source_path,
        tokens=token_array,
        positions=position_array,
        atol=atol,
        rtol=rtol,
    )
