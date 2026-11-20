"""Real-model use case: sharded safetensors import and ONNX interchange.

Part 1 imports one real pretrained checkpoint twice: as its published
single-file ``model.safetensors`` and as a real sharded checkpoint
(``model-0000i-of-0000N.safetensors`` + ``model.safetensors.index.json``)
written from the same tensors with UZEN's own safetensors writer. The two
artifacts must agree on tensor digest, logits and generated tokens, and a
deliberately incomplete shard set must be refused with the importer's own
integrity error rather than silently loaded.

Part 2 exports the reconstructed layer-0 MLP block of that same real checkpoint
to a single-file ``.onnx`` model, validates it with ``onnx.checker``, imports it
back through the public artifact pipeline, and compares its execution with the
NIR graph it came from.

Usage::

    HF_HUB_DISABLE_XET=1 .venv/bin/python examples/sharded_onnx_import_demo.py \\
        --model distilgpt2 --root /tmp/uzen-sharded-onnx-demo

A local Hugging Face checkpoint directory that already contains
``model.safetensors`` or ``model.safetensors.index.json`` may be passed instead::

    .venv/bin/python examples/sharded_onnx_import_demo.py --model /path/to/model

The demo needs the ``hf``, ``safetensors``, ``tokenizer`` and ``onnx`` extras.
It exits non-zero when any gate fails.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from uzen import Uzen
from uzen.core.nir.executor import execute
from uzen.core.nir.graph import Graph
from uzen.core.nir.types import Ref, TensorType
from uzen.core.nws.tensors import WeightPackage
from uzen.errors import UzenError
from uzen.infrastructure.storage.safetensors_io import (
    read_header,
    read_safetensors,
    safetensors_tensor_digest,
    write_safetensors,
)

ALLOW_PATTERNS = [
    "config.json",
    "*.safetensors",
    "*.json",
    "*.txt",
    "*.model",
    "*.jinja",
    "README.md",
]
METADATA_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "chat_template.jinja",
    "README.md",
]
TOKEN_EMBEDDING = "embed.weight"
ONNX_MAX_ABS_ERROR = 1e-5


def _print(message: str) -> None:
    print(message, flush=True)


def _resolve_source(value: str) -> Path:
    """Return a local directory for a Hub repository id or a local path."""
    path = Path(value).expanduser()
    if path.exists():
        return path.resolve()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        import huggingface_hub  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            f"cannot download '{value}': the optional huggingface_hub dependency is "
            f"missing ({type(exc).__name__}: {exc}); install the 'hf' extra or pass a "
            f"local checkpoint directory"
        ) from exc
    snapshot = huggingface_hub.snapshot_download(repo_id=value, allow_patterns=ALLOW_PATTERNS)
    return Path(snapshot)


def _copy_metadata(source: Path, destination: Path) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in METADATA_FILES:
        candidate = source / name
        if not candidate.is_file():
            continue
        # Hub snapshots store files as relative symlinks into a blob cache, so the
        # real target is linked/copied rather than the link itself.
        real = candidate.resolve(strict=True)
        target = destination / name
        try:
            os.link(real, target)
        except OSError:
            shutil.copy2(real, target)
        copied.append(name)
    return copied


def _link(source: Path, destination: Path) -> None:
    destination.unlink(missing_ok=True)
    real = source.resolve(strict=True)
    try:
        os.link(real, destination)
    except OSError:
        shutil.copy2(real, destination)


def _write_shards(
    single_file: Path,
    destination: Path,
    shard_count: int,
) -> dict[str, Any]:
    """Split a real safetensors file into shards plus a real weight index."""
    arrays, header = read_safetensors(single_file)
    if shard_count > len(arrays):
        raise SystemExit(
            f"cannot split {len(arrays)} tensors into {shard_count} shards; "
            f"use --shards <= {len(arrays)}"
        )
    names = sorted(arrays)
    buckets: list[list[str]] = [[] for _ in range(shard_count)]
    loads = [0] * shard_count
    for name in sorted(names, key=lambda n: (-arrays[n].nbytes, n)):
        index = loads.index(min(loads))
        buckets[index].append(name)
        loads[index] += int(arrays[name].nbytes)

    destination.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}
    shard_rows: list[dict[str, Any]] = []
    for position, bucket in enumerate(buckets, start=1):
        filename = f"model-{position:05d}-of-{shard_count:05d}.safetensors"
        payload = {name: arrays[name] for name in sorted(bucket)}
        path = write_safetensors(destination / filename, payload)
        for name in payload:
            weight_map[name] = filename
        shard_rows.append(
            {
                "file": filename,
                "tensors": len(payload),
                "bytes": path.stat().st_size,
                "sha256_prefix": (read_header(path, compute_sha=True).sha256 or "")[:16],
            }
        )
    total_size = int(sum(int(arrays[name].nbytes) for name in names))
    index_document = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    index_path = destination / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index_document, indent=2, sort_keys=True) + "\n")
    return {
        "shard_count": shard_count,
        "shards": shard_rows,
        "index_bytes": index_path.stat().st_size,
        "indexed_tensors": len(weight_map),
        "total_size": total_size,
        "source_digest": safetensors_tensor_digest(arrays),
        "source_tensor_count": len(arrays),
        "source_dtypes": sorted({entry.dtype_code for entry in header.entries.values()}),
    }


def _logits(payload: dict[str, Any]) -> np.ndarray:
    outputs = payload.get("outputs") or {}
    for name, value in outputs.items():
        if "logits" in name.lower() and "values" in value:
            return np.asarray(value["values"], dtype=np.float64)
    raise SystemExit(f"execution payload exposes no logits values: {sorted(outputs)}")


def _direct_import(path: Path) -> dict[str, Any]:
    """Import a layout directly and return what the importer itself produced.

    The artifact store is content-addressed, so importing the same canonical
    tensors from two layouts reuses one artifact id. Comparing the *importer's*
    output for each layout keeps the sharded-merge check independent of that
    deduplication.
    """
    from uzen.core.wie.importer import ModelImporter

    result = ModelImporter().import_source(path)
    package = result.artifact_package or result.package
    files = result.report.files
    return {
        "ok": bool(result.report.ok) and package is not None,
        "tensor_count": len(package) if package is not None else 0,
        "tensor_digest": (
            safetensors_tensor_digest(package.tensors) if package is not None else None
        ),
        "sharded_flag": bool((result.report.metadata or {}).get("sharded")),
        "weight_files": [item["path"] for item in files if item.get("role") == "weights"],
        "overall": result.report.compatibility.overall,
        "reconstruction": result.report.compatibility.reconstruction,
        "execution": result.report.compatibility.execution,
    }


def _import_and_execute(
    client: Uzen,
    path: Path,
    *,
    label: str,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    started = time.monotonic()
    direct = _direct_import(path)
    imported = client.import_model(path)
    artifact_id = imported["artifact_id"]
    report = imported["report"]
    compatibility = imported["compatibility"]
    verify = client.verify_artifact(artifact_id)
    forward = client.forward_text(artifact_id, text=prompt, include_values=True)
    generation = client.generate_text(
        artifact_id, text=prompt, max_new_tokens=max_new_tokens, strategy="greedy"
    )
    files = report.get("files") or []
    row: dict[str, Any] = {
        "label": label,
        "source": str(path),
        "artifact_id": artifact_id,
        "overall": compatibility.get("overall"),
        "reconstruction": compatibility.get("reconstruction"),
        "execution": compatibility.get("execution"),
        "num_tensors": imported["num_tensors"],
        "sharded_flag": bool((report.get("metadata") or {}).get("sharded")),
        "file_roles": sorted({str(item.get("role")) for item in files}),
        "weight_files": [item["path"] for item in files if item.get("role") == "weights"],
        "integrity_ok": bool(verify.get("ok")),
        "token_ids": forward.get("token_ids"),
        "logits": _logits(forward),
        "generated_text": generation.get("generated_text"),
        "generated_token_ids": generation.get("generated_token_ids"),
        "direct_import": direct,
        "seconds": round(time.monotonic() - started, 3),
    }
    row["logits_shape"] = list(row["logits"].shape)
    _print(
        f"[{label}] artifact={artifact_id} tensors={row['num_tensors']} "
        f"overall={row['overall']} execution={row['execution']} "
        f"sharded={row['sharded_flag']} files={len(row['weight_files'])}"
    )
    _print(
        f"[{label}] logits={row['logits_shape']} generation={row['generated_text']!r} "
        f"integrity={row['integrity_ok']}"
    )
    return row


def _compare(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    difference = np.abs(left["logits"] - right["logits"])
    left_import = left["direct_import"]
    right_import = right["direct_import"]
    return {
        "tensor_digest_equal": left_import["tensor_digest"] == right_import["tensor_digest"],
        "tensor_count_equal": left_import["tensor_count"] == right_import["tensor_count"],
        "logits_shape_equal": left["logits_shape"] == right["logits_shape"],
        "same_token_ids": left["token_ids"] == right["token_ids"],
        "same_generated_tokens": left["generated_token_ids"] == right["generated_token_ids"],
        "same_artifact_id": left["artifact_id"] == right["artifact_id"],
        "max_abs_logit_error": float(difference.max()) if difference.size else 0.0,
        "mean_abs_logit_error": float(difference.mean()) if difference.size else 0.0,
    }


def _missing_shard_control(
    sharded_dir: Path,
    work: Path,
    indexed_tensors: int,
) -> dict[str, Any]:
    """Remove one declared shard and require the importer to refuse the source."""
    from uzen.core.wie.importer import ModelImporter

    broken = work / "sharded-missing-shard"
    if broken.exists():
        shutil.rmtree(broken)
    broken.mkdir(parents=True)
    for item in sharded_dir.iterdir():
        if item.is_file():
            _link(item, broken / item.name)
    shard_files = sorted(broken.glob("model-*-of-*.safetensors"))
    if not shard_files:
        raise SystemExit("no shard files to remove for the negative control")
    removed = shard_files[0].name
    (broken / removed).unlink()

    result = ModelImporter().import_source(broken)
    problems = list(result.report.problems)
    refused = (
        not result.report.ok
        and result.package is None
        and any("missing shard" in problem for problem in problems)
    )
    _print(f"[control] removed {removed}; refused={refused} problems={problems[:1]}")
    return {
        "removed_shard": removed,
        "refused": refused,
        "report_ok": bool(result.report.ok),
        "package_stored": result.package is not None,
        "problems": problems,
        "declared_tensors": indexed_tensors,
    }


def _onnx_roundtrip(
    client: Uzen,
    artifact_id: str,
    *,
    work: Path,
    prompt: str,
) -> dict[str, Any]:
    """Export a real GPT-2 layer-0 MLP block to ONNX and import it back."""
    from uzen.runtimes.onnx_runtime import export_onnx, import_onnx, is_available

    if not is_available():
        raise SystemExit("the 'onnx' extra is required for the ONNX interchange demo")

    import onnx  # noqa: PLC0415 - optional dependency, imported after the check

    package = client.package(artifact_id)
    tensors = package.tensors
    required = (
        TOKEN_EMBEDDING,
        "layers.0.mlp_up.weight",
        "layers.0.mlp_up.bias",
        "layers.0.mlp_down.weight",
        "layers.0.mlp_down.bias",
    )
    missing = [name for name in required if name not in tensors]
    if missing:
        raise SystemExit(
            "the stored artifact does not expose the canonical GPT-2 MLP tensors "
            f"this demo exports: missing {missing}"
        )

    forward = client.forward_text(artifact_id, text=prompt)
    token_ids = forward.get("token_ids") or [[]]
    ids = np.asarray(token_ids[0], dtype=np.int64)
    embedding = np.asarray(tensors[TOKEN_EMBEDDING], dtype=np.float32)
    x = np.asarray(embedding[ids], dtype=np.float32)
    hidden = int(x.shape[-1])

    graph = Graph(name="gpt2_layer0_mlp")
    source = graph.add_input("x", TensorType.of(("T", hidden), "float32"))
    up = graph.add_node(
        "mlp_up",
        "linear",
        {"x": source},
        {"weight": "param:mlp_up.weight", "bias": "param:mlp_up.bias"},
    )
    activation = graph.add_node("gelu", "gelu", {"x": up})
    down = graph.add_node(
        "mlp_down",
        "linear",
        {"x": activation},
        {"weight": "param:mlp_down.weight", "bias": "param:mlp_down.bias"},
    )
    graph.set_output("mlp_out", down)
    weights = WeightPackage.from_arrays(
        {
            "mlp_up.weight": tensors["layers.0.mlp_up.weight"],
            "mlp_up.bias": tensors["layers.0.mlp_up.bias"],
            "mlp_down.weight": tensors["layers.0.mlp_down.weight"],
            "mlp_down.bias": tensors["layers.0.mlp_down.bias"],
        },
        name="gpt2-layer0-mlp",
        family="mlp",
    )
    graph.require_valid(weights)
    reference = execute(graph, {"x": x}, weights).outputs["mlp_out"]

    onnx_path = work / "gpt2_layer0_mlp.onnx"
    model = export_onnx(graph, weights, save_path=onnx_path)
    onnx.checker.check_model(model)
    loaded = onnx.load(str(onnx_path))
    op_types = [node.op_type for node in loaded.graph.node]

    onnxruntime_result: dict[str, Any] = {"status": "not_installed"}
    try:
        import onnxruntime as ort  # noqa: PLC0415 - optional dependency

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        values = session.run(None, {"x": x})
        runtime_error = float(np.abs(np.asarray(values[0]) - reference).max())
        onnxruntime_result = {
            "status": "passed",
            "version": ort.__version__,
            "max_abs_error_vs_nir": runtime_error,
        }
    except ImportError:
        onnxruntime_result = {
            "status": "skipped",
            "reason": "onnxruntime is not installed; the ONNX file is validated with onnx.checker",
        }
    except Exception as exc:  # pragma: no cover - depends on the local ORT build
        onnxruntime_result = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }

    imported_graph, imported_weights, onnx_metadata = import_onnx(onnx_path)
    imported_graph.require_valid(imported_weights)
    output_name = next(iter(imported_graph.outputs))
    restored = execute(imported_graph, {"x": x}, imported_weights).outputs[output_name]
    difference = np.abs(np.asarray(restored, dtype=np.float64) - np.asarray(reference, dtype=np.float64))

    stored = client.import_weights(onnx_path)
    stored_report = stored.get("import_report") or {}
    stored_verify = client.verify_artifact(stored["artifact_id"])
    stored_manifest = client.artifact(stored["artifact_id"])
    stored_graph = (stored_manifest.get("metadata") or {}).get("graph") or {}

    row: dict[str, Any] = {
        "source_artifact": artifact_id,
        "block": "layers.0.mlp (mlp_up -> gelu -> mlp_down)",
        "input_tokens": ids.tolist(),
        "input_shape": list(x.shape),
        "onnx_path": str(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "opset": [f"{o.domain or 'ai.onnx'}:{o.version}" for o in loaded.opset_import],
        "onnx_ops": op_types,
        "checker": "passed",
        "onnxruntime": onnxruntime_result,
        "reference_shape": list(np.asarray(reference).shape),
        "imported_num_nodes": int(onnx_metadata.get("num_nodes", 0)),
        "imported_num_initializers": int(onnx_metadata.get("num_initializers", 0)),
        "reimported_output": output_name,
        "max_abs_error_vs_nir": float(difference.max()) if difference.size else 0.0,
        "mean_abs_error_vs_nir": float(difference.mean()) if difference.size else 0.0,
        "artifact_id": stored["artifact_id"],
        "artifact_format": stored.get("format"),
        "artifact_graph_reconstructed": bool(stored_report.get("graph_reconstructed")),
        "artifact_num_nodes": stored_report.get("num_nodes"),
        "artifact_integrity_ok": bool(stored_verify.get("ok")),
        "artifact_graph_nodes": len(stored_graph.get("nodes") or []),
    }
    row["ok"] = bool(
        row["max_abs_error_vs_nir"] <= ONNX_MAX_ABS_ERROR
        and row["artifact_format"] == "onnx"
        and row["artifact_graph_reconstructed"]
        and row["artifact_integrity_ok"]
        and row["artifact_graph_nodes"] > 0
        and onnxruntime_result.get("status") != "failed"
    )
    _print(
        f"[onnx] ops={op_types} bytes={row['onnx_bytes']} "
        f"checker={row['checker']} max_abs_error={row['max_abs_error_vs_nir']:.3e}"
    )
    _print(
        f"[onnx] stored artifact={row['artifact_id']} format={row['artifact_format']} "
        f"graph_nodes={row['artifact_graph_nodes']} integrity={row['artifact_integrity_ok']}"
    )
    return row


def run_demo(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    work = root / "work"
    work.mkdir(parents=True, exist_ok=True)
    client = Uzen(root=root)

    source = _resolve_source(args.model)
    _print(f"source: {source}")
    single_file = source / "model.safetensors"
    index_file = source / "model.safetensors.index.json"
    if not single_file.exists() and not index_file.exists():
        raise SystemExit(
            f"{source} contains neither model.safetensors nor "
            "model.safetensors.index.json; this demo needs a real safetensors checkpoint"
        )

    report: dict[str, Any] = {
        "demo": "sharded_onnx_import",
        "use_case": (
            "import a real checkpoint from a sharded safetensors layout and exchange a "
            "real-weight subgraph with ONNX"
        ),
        "model": args.model,
        "source": str(source),
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
    }

    if index_file.exists():
        _print("source is already sharded; importing it through the weight index")
        sharded_dir = source
        sharding: dict[str, Any] = {"source_layout": "published_shards"}
        single_row = None
    else:
        single_dir = work / "single"
        if single_dir.exists():
            shutil.rmtree(single_dir)
        metadata = _copy_metadata(source, single_dir)
        _link(single_file, single_dir / "model.safetensors")
        sharded_dir = work / "sharded"
        if sharded_dir.exists():
            shutil.rmtree(sharded_dir)
        _copy_metadata(source, sharded_dir)
        started = time.monotonic()
        sharding = _write_shards(single_file, sharded_dir, args.shards)
        sharding["seconds"] = round(time.monotonic() - started, 3)
        sharding["source_layout"] = "split_locally"
        sharding["metadata_files"] = metadata
        _print(
            f"wrote {sharding['shard_count']} shards "
            f"({sharding['indexed_tensors']} tensors indexed, "
            f"{sharding['index_bytes']} byte index) in {sharding['seconds']}s"
        )

    single_row = None
    if (work / "single" / "model.safetensors").exists():
        single_row = _import_and_execute(
            client,
            work / "single",
            label="single_file",
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
        )
    sharded_row = _import_and_execute(
        client,
        sharded_dir,
        label="sharded",
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )

    comparison: dict[str, Any] = {}
    if single_row is not None:
        comparison = _compare(single_row, sharded_row)
        comparison["ok"] = bool(
            comparison["tensor_digest_equal"]
            and comparison["tensor_count_equal"]
            and comparison["logits_shape_equal"]
            and comparison["same_token_ids"]
            and comparison["same_generated_tokens"]
            and comparison["max_abs_logit_error"] == 0.0
        )
        _print(
            f"[compare] digest_equal={comparison['tensor_digest_equal']} "
            f"logits_equal={comparison['max_abs_logit_error'] == 0.0} "
            f"generation_equal={comparison['same_generated_tokens']}"
        )

    control = _missing_shard_control(sharded_dir, work, sharding.get("indexed_tensors", 0))
    onnx_row = _onnx_roundtrip(
        client,
        sharded_row["artifact_id"],
        work=work,
        prompt=args.prompt,
    )

    sharded_direct = sharded_row["direct_import"]
    sharded_ok = bool(
        sharded_row["overall"] == "validated"
        and sharded_row["execution"] == "passed"
        and sharded_row["integrity_ok"]
        and sharded_row["sharded_flag"] is True
        and len(sharded_row["weight_files"]) >= 2
        and sharded_direct["ok"]
        and sharded_direct["sharded_flag"] is True
        and len(sharded_direct["weight_files"]) >= 2
        and bool(sharded_direct["tensor_digest"])
        and sharded_row["token_ids"]
    )
    single_ok = True if single_row is None else bool(
        single_row["overall"] == "validated"
        and single_row["execution"] == "passed"
        and single_row["integrity_ok"]
        and single_row["direct_import"]["ok"]
        and single_row["direct_import"]["sharded_flag"] is False
        and len(single_row["direct_import"]["weight_files"]) == 1
    )
    passed = bool(
        sharded_ok
        and single_ok
        and (comparison.get("ok") if single_row is not None else True)
        and control["refused"]
        and onnx_row["ok"]
    )

    for row in (single_row, sharded_row):
        if row is not None:
            row.pop("logits", None)

    report.update(
        {
            "ok": passed,
            "sharding": sharding,
            "single_file_import": single_row,
            "sharded_import": sharded_row,
            "comparison": comparison,
            "missing_shard_control": control,
            "onnx_interchange": onnx_row,
        }
    )
    report_path = root / "reports" / "sharded_onnx_import.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")

    _print("\n=== gates ===")
    _print(f"sharded import validated and executed: {sharded_ok}")
    if single_row is not None:
        _print(f"single-file import validated and executed: {single_ok}")
        _print(f"sharded == single-file (digest/logits/generation): {comparison.get('ok')}")
        if comparison.get("same_artifact_id"):
            _print(
                "note: both layouts address the same content-addressed artifact "
                "(identical canonical tensors); the comparison above uses each "
                "layout's own importer output"
            )
    _print(f"incomplete shard set refused: {control['refused']}")
    _print(f"ONNX export/import/execution: {onnx_row['ok']}")
    _print(f"report: {report_path}")
    _print(f"gate: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Import a real checkpoint from both a single-file and a sharded safetensors "
            "layout, compare them, and round-trip a real-weight subgraph through ONNX."
        )
    )
    parser.add_argument("--model", default="distilgpt2", help="Hub repository id or local directory")
    parser.add_argument("--root", default="/tmp/uzen-sharded-onnx-demo")
    parser.add_argument("--shards", type=int, default=4, help="shard count for the local split")
    parser.add_argument("--prompt", default="Hello, my name is")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_demo(args)
    except UzenError as exc:
        print(
            json.dumps(
                {
                    "demo": "sharded_onnx_import",
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "details": getattr(exc, "details", None),
                },
                indent=2,
                default=str,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
