"""Reference-logit parity checks for reconstructed Hugging Face models.

This module deliberately keeps ``transformers`` and ``torch`` optional.  It is
used by the live validation tests, not by the core import path, so a normal
UZEN installation does not need a deep-learning runtime just to inspect and
map weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from uzen.core.nir.executor import execute
from uzen.core.nir.graph import Graph
from uzen.core.nws.tensors import WeightPackage
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
