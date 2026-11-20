"""Real-model use case: build and verify an int8 release candidate.

This demo imports a real, pretrained Hugging Face causal language model,
persists a weight-only int8 variant, compares source-versus-variant drift,
generates from both models, exports the variant, re-imports the export, and
checks its integrity.

The default model is DistilGPT-2 (a real pretrained checkpoint). The demo needs
the optional Hugging Face, safetensors and tokenizer extras::

    .venv/bin/pip install -e '.[hf,safetensors,tokenizer]'
    .venv/bin/python examples/quantized_release_demo.py

If your network cannot use Hugging Face Xet downloads, set::

    HF_HUB_DISABLE_XET=1 .venv/bin/python examples/quantized_release_demo.py

The demo writes a JSON release report under ``<root>/reports`` and exits non-zero
when the configured quality/integrity gate fails. It does not claim that int8 is
semantically equivalent to the source model; generated continuations are shown
separately because they can diverge even when logits pass a threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from uzen import Uzen
from uzen.core.wie.tokenization import encode_text, tokenizer_spec_from_manifest


def _print(message: str) -> None:
    print(message, flush=True)


def _set_error(message: str) -> None:
    print(
        json.dumps(
            {"demo": "quantized_release", "ok": False, "error": message},
            indent=2,
        ),
        file=sys.stderr,
    )


def _package_summary(package: Any) -> dict[str, Any]:
    dtypes = sorted({str(array.dtype) for array in package.tensors.values()})
    return {
        "num_tensors": len(package),
        "numel": int(package.numel()),
        "bytes": int(package.nbytes()),
        "dtypes": dtypes,
    }


def _quantization_summary(report: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "schema",
        "scheme",
        "bits",
        "axis",
        "num_rewritten_nodes",
        "compression_ratio",
        "max_abs_error",
        "mean_abs_error",
        "graph_valid",
    )
    return {key: report.get(key) for key in keys}


def run_demo(args: argparse.Namespace) -> int:
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    client = Uzen(root=root)

    _print(f"[1/7] importing real checkpoint: {args.repo_id}")
    started = time.monotonic()
    imported = client.import_model(args.repo_id, repo_id=args.repo_id)
    source_id = imported["artifact_id"]
    compatibility = imported["compatibility"]
    manifest = client.artifact(source_id)
    if compatibility.get("execution") != "passed":
        _set_error(
            "the checkpoint was imported but its reconstructed graph did not execute: "
            f"{compatibility}"
        )
        return 2
    tokenizer_spec = tokenizer_spec_from_manifest(manifest)
    encoded = encode_text(client.service.root, tokenizer_spec, args.prompt)
    tokens = [list(encoded.token_ids)]
    positions = [list(range(len(encoded.token_ids)))]
    _print(
        f"      source artifact: {source_id} "
        f"({imported['num_tensors']} tensors, {time.monotonic() - started:.1f}s)"
    )
    _print(f"      compatibility: {compatibility['overall']}")
    _print(f"      prompt tokens: {tokens[0]}")

    _print("[2/7] persisting weight-only int8 variant")
    started = time.monotonic()
    variant = client.quantize_model(source_id)
    variant_id = variant["artifact_id"]
    _print(
        f"      variant artifact: {variant_id} "
        f"({time.monotonic() - started:.1f}s)"
    )
    source_package = client.package(source_id)
    variant_package = client.package(variant_id)
    source_size = _package_summary(source_package)
    variant_size = _package_summary(variant_package)
    _print(
        f"      package size: {source_size['bytes']:,} -> "
        f"{variant_size['bytes']:,} bytes "
        f"({source_size['bytes'] / variant_size['bytes']:.2f}x smaller)"
    )

    _print("[3/7] comparing source and variant logits")
    started = time.monotonic()
    comparison = client.compare_quantized_model(
        source_id,
        variant_id,
        tokens=tokens,
        positions=positions,
        atol=args.atol,
        rtol=args.rtol,
        max_abs_error_threshold=args.max_abs_error_threshold,
    )
    drift = comparison["drift"]
    _print(
        f"      drift gate: {'PASS' if comparison['passed'] else 'FAIL'} "
        f"(max_abs={drift['max_abs_error']:.6g}, "
        f"mean_abs={drift['mean_abs_error']:.6g}, "
        f"elements={drift['num_elements']}, {time.monotonic() - started:.1f}s)"
    )
    if drift["failures"]:
        for failure in drift["failures"]:
            _print(f"      threshold failure: {failure}")

    source_generation: dict[str, Any] | None = None
    variant_generation: dict[str, Any] | None = None
    generated_sequence_match: bool | None = None
    if args.skip_generation:
        _print("[4/7] generation skipped (--skip-generation)")
    else:
        _print("[4/7] generating from source and variant")
        started = time.monotonic()
        source_generation = client.generate_text(
            source_id,
            text=args.prompt,
            max_new_tokens=args.max_new_tokens,
            strategy="greedy",
        )
        variant_generation = client.generate_text(
            variant_id,
            text=args.prompt,
            max_new_tokens=args.max_new_tokens,
            strategy="greedy",
        )
        generated_sequence_match = (
            source_generation["generated_token_ids"]
            == variant_generation["generated_token_ids"]
        )
        _print(f"      source:  {source_generation['generated_text']!r}")
        _print(f"      variant: {variant_generation['generated_text']!r}")
        _print(f"      generated token sequence match: {generated_sequence_match}")
        _print(f"      generation time: {time.monotonic() - started:.1f}s")

    _print("[5/7] exporting the int8 artifact")
    safe_repo = args.repo_id.replace("/", "--")
    export_path = root / "exports" / f"{safe_repo}-int8-{variant_id[:12]}.safetensors"
    started = time.monotonic()
    exported = client.service.store.export(variant_id, export_path)
    _print(f"      export: {exported} ({exported.stat().st_size:,} bytes, {time.monotonic() - started:.1f}s)")

    _print("[6/7] re-importing the exported artifact and executing it")
    started = time.monotonic()
    reloaded = client.import_weights(exported)
    reloaded_id = reloaded["artifact_id"]
    reloaded_execution = client.forward_model(
        reloaded_id,
        tokens=tokens,
        positions=positions,
    )
    reloaded_verify = client.verify_artifact(reloaded_id)
    variant_verify = client.verify_artifact(variant_id)
    reloaded_shape = reloaded_execution["outputs"]["logits"]["shape"]
    source_shape = comparison["outputs"]["logits"]["shape"]
    shape_match = reloaded_shape == source_shape
    _print(
        f"      reloaded artifact: {reloaded_id} "
        f"(logits={reloaded_shape}, shape_match={shape_match}, "
        f"integrity={reloaded_verify['ok']}, {time.monotonic() - started:.1f}s)"
    )

    gate_passed = bool(
        comparison["passed"]
        and variant_verify["ok"]
        and reloaded_verify["ok"]
        and shape_match
    )
    report = {
        "demo": "quantized_release",
        "use_case": "int8 release candidate for a pretrained causal language model",
        "ok": gate_passed,
        "repo_id": args.repo_id,
        "prompt": args.prompt,
        "source_artifact_id": source_id,
        "variant_artifact_id": variant_id,
        "reloaded_artifact_id": reloaded_id,
        "source_package": source_size,
        "variant_package": variant_size,
        "compression": {
            "source_bytes": source_size["bytes"],
            "variant_bytes": variant_size["bytes"],
            "ratio": source_size["bytes"] / variant_size["bytes"],
            "quantization_report": _quantization_summary(variant["quantization"]),
        },
        "comparison": {
            "passed": comparison["passed"],
            "tolerances": comparison["tolerances"],
            "drift": drift,
            "outputs": comparison["outputs"],
        },
        "generation": {
            "source": source_generation,
            "variant": variant_generation,
            "generated_sequence_match": generated_sequence_match,
        },
        "integrity": {
            "variant": variant_verify,
            "reloaded": reloaded_verify,
        },
        "export": {
            "path": str(exported),
            "bytes": exported.stat().st_size,
            "reloaded_shape": reloaded_shape,
            "source_shape": source_shape,
        },
    }
    report_path = root / "reports" / "quantized_release.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")

    _print("[7/7] release report")
    _print(f"      report: {report_path}")
    _print(
        json.dumps(
            {
                "ok": gate_passed,
                "source_artifact_id": source_id,
                "variant_artifact_id": variant_id,
                "reloaded_artifact_id": reloaded_id,
                "compression_ratio": report["compression"]["ratio"],
                "drift_gate_passed": comparison["passed"],
                "generated_sequence_match": generated_sequence_match,
                "export": str(exported),
                "report": str(report_path),
            },
            indent=2,
        )
    )
    if not gate_passed:
        _set_error("release gate failed; inspect the JSON report for details")
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import a real model, build an int8 variant and verify a release candidate."
    )
    parser.add_argument("--root", default="./uzen-quantized-release-demo")
    parser.add_argument("--repo-id", default="distilgpt2")
    parser.add_argument("--prompt", default="Hello, my name is")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--atol", type=float, default=0.5)
    parser.add_argument("--rtol", type=float, default=0.5)
    parser.add_argument("--max-abs-error-threshold", type=float, default=2.0)
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="skip the two CPU text-generation passes",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_demo(args)
    except Exception as exc:  # surface a concise demo failure without hiding the type
        _set_error(f"{type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
