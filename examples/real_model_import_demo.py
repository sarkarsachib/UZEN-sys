"""Real-model use case: import, reconstruct, execute and audit checkpoints.

This demo imports real pretrained checkpoints, validates their reconstructed
NIR graphs, runs a real forward pass and a short greedy generation, verifies
the persisted artifact, and writes a support-matrix report.

Examples::

    HF_HUB_DISABLE_XET=1 .venv/bin/python examples/real_model_import_demo.py \
        --models distilgpt2 openai-community/gpt2 --root /tmp/uzen-import-demo

A local Hugging Face directory can be supplied in place of a Hub repository id::

    .venv/bin/python examples/real_model_import_demo.py --models /path/to/model

The demo requires the optional inference extras::

    .venv/bin/pip install -e '.[hf,safetensors,tokenizer]'

It exits non-zero when any requested checkpoint does not reach a fully
validated reconstruction with a passing forward pass and intact artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from uzen import Uzen
from uzen.errors import UzenError


def _print(message: str) -> None:
    print(message, flush=True)


def _source_and_repo(value: str) -> tuple[str, str | None]:
    path = Path(value).expanduser()
    if path.exists():
        return str(path), None
    return value, value


def _ignored_buffers(reconstruction: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "source": item.get("source"),
            "dtype": item.get("dtype"),
            "shape": item.get("source_shape"),
            "note": item.get("note"),
        }
        for item in reconstruction.get("tensor_mapping", [])
        if item.get("status") == "ignored"
    ]


def _run_one(
    client: Uzen,
    value: str,
    *,
    prompt: str,
    max_new_tokens: int,
    skip_generation: bool,
) -> dict[str, Any]:
    source, repo_id = _source_and_repo(value)
    _print(f"\n=== {value} ===")
    started = time.monotonic()
    imported = client.import_model(source, repo_id=repo_id)
    artifact_id = imported["artifact_id"]
    report = imported["report"]
    reconstruction = report.get("reconstruction") or {}
    compatibility = imported["compatibility"]
    manifest = client.artifact(artifact_id)

    row: dict[str, Any] = {
        "model": value,
        "source": source,
        "artifact_id": artifact_id,
        "family": imported["family"],
        "num_tensors": imported["num_tensors"],
        "compatibility": compatibility,
        "overall": compatibility.get("overall"),
        "execution": compatibility.get("execution"),
        "reconstruction": {
            "status": reconstruction.get("status"),
            "architecture": reconstruction.get("architecture"),
            "mapped_tensor_count": reconstruction.get("mapped_tensor_count"),
            "ignored_buffers": _ignored_buffers(reconstruction),
            "warnings": reconstruction.get("warnings", []),
        },
        "license": report.get("license"),
        "tokenizer": report.get("tokenizer"),
        "import_seconds": round(time.monotonic() - started, 3),
    }
    _print(
        f"artifact={artifact_id} tensors={imported['num_tensors']} "
        f"family={imported['family']} overall={row['overall']}"
    )
    _print(
        f"reconstruction={row['reconstruction']['status']} "
        f"mapped={row['reconstruction']['mapped_tensor_count']} "
        f"ignored_buffers={len(row['reconstruction']['ignored_buffers'])}"
    )

    verify = client.verify_artifact(artifact_id)
    row["integrity_ok"] = bool(verify.get("ok"))
    _print(f"artifact integrity: {row['integrity_ok']}")

    if not skip_generation:
        if reconstruction.get("graph") is None:
            row["forward"] = {"status": "skipped", "reason": "no reconstructed graph"}
            row["generation"] = {"status": "skipped", "reason": "no reconstructed graph"}
        else:
            execution_started = time.monotonic()
            forward = client.forward_text(artifact_id, text=prompt)
            row["forward"] = {
                "text": forward.get("text"),
                "token_ids": forward.get("token_ids"),
                "logits_shape": forward.get("outputs", {}).get("logits", {}).get("shape"),
                "seconds": round(time.monotonic() - execution_started, 3),
            }
            _print(
                f"forward: tokens={row['forward']['token_ids']} "
                f"logits={row['forward']['logits_shape']} "
                f"({row['forward']['seconds']}s)"
            )
            generation_started = time.monotonic()
            generated = client.generate_text(
                artifact_id,
                text=prompt,
                max_new_tokens=max_new_tokens,
                strategy="greedy",
            )
            row["generation"] = {
                "generated_text": generated.get("generated_text"),
                "generated_token_ids": generated.get("generated_token_ids"),
                "seconds": round(time.monotonic() - generation_started, 3),
            }
            _print(f"generation: {row['generation']['generated_text']!r}")
    else:
        row["forward"] = {"status": "skipped", "reason": "--skip-generation"}
        row["generation"] = {"status": "skipped", "reason": "--skip-generation"}

    row["ok"] = bool(
        row["overall"] == "validated"
        and row["execution"] == "passed"
        and row["integrity_ok"]
        and row["forward"].get("logits_shape")
    )
    return row


def run_demo(args: argparse.Namespace) -> int:
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    client = Uzen(root=root)
    rows: list[dict[str, Any]] = []
    for value in args.models:
        try:
            rows.append(
                _run_one(
                    client,
                    value,
                    prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    skip_generation=args.skip_generation,
                )
            )
        except (UzenError, OSError, ValueError) as exc:
            rows.append(
                {
                    "model": value,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            _print(f"FAILED {value}: {type(exc).__name__}: {exc}")

    passed = bool(rows) and all(row["ok"] for row in rows)
    report = {
        "demo": "real_model_import",
        "use_case": "validate real pretrained checkpoints for headless neural engineering",
        "ok": passed,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "models": rows,
    }
    report_path = root / "reports" / "real_model_import.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    _print("\n=== support matrix ===")
    for row in rows:
        _print(
            f"{row['model']}: ok={row['ok']} overall={row.get('overall')} "
            f"reconstruction={row.get('reconstruction', {}).get('status')} "
            f"execution={row.get('execution')}"
        )
    _print(f"report: {report_path}")
    _print(f"gate: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Import real checkpoints, reconstruct and execute their NIR graphs, "
            "and write a support-matrix report."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["distilgpt2", "openai-community/gpt2"],
        help="Hub repository ids or local checkpoint directories",
    )
    parser.add_argument("--root", default="./uzen-real-model-import-demo")
    parser.add_argument("--prompt", default="Hello, my name is")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="import and verify only; skip forward passes and generation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_demo(args)
    except Exception as exc:  # surface a concise failure without hiding the type
        print(
            json.dumps(
                {"demo": "real_model_import", "ok": False, "error": f"{type(exc).__name__}: {exc}"},
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
