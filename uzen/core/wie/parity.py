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
from uzen.errors import RuntimeUnavailable


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
    if int(token_array.min()) < 0:
        raise ValueError("tokens must be non-negative")

    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"local reference checkpoint not found: {source_path}")

    model = GPT2LMHeadModel.from_pretrained(str(source_path), local_files_only=True).eval()
    with torch.no_grad():
        reference = model(input_ids=torch.as_tensor(token_array)).logits.detach().cpu().numpy()
    actual = execute(graph, {"tokens": token_array}, package).output("logits")

    if actual.shape != reference.shape:
        return {
            "passed": False,
            "reference": "transformers.GPT2LMHeadModel",
            "source": str(source_path),
            "tolerances": {"atol": atol, "rtol": rtol},
            "tokens": token_array.tolist(),
            "actual_shape": list(actual.shape),
            "reference_shape": list(reference.shape),
            "error": "shape mismatch",
        }

    difference = np.abs(actual - reference)
    relative = difference / np.maximum(np.abs(reference), 1e-8)
    result = {
        "passed": bool(np.allclose(actual, reference, atol=atol, rtol=rtol)),
        "reference": "transformers.GPT2LMHeadModel",
        "source": str(source_path),
        "tolerances": {"atol": atol, "rtol": rtol},
        "tokens": token_array.tolist(),
        "actual_shape": list(actual.shape),
        "reference_shape": list(reference.shape),
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "max_rel_error": float(relative.max()),
        "num_elements": int(difference.size),
        "allclose": bool(np.allclose(actual, reference, atol=atol, rtol=rtol)),
    }
    return result
