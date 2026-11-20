"""Real-model use case: move a validated artifact through S3 object storage.

A worker without a shared filesystem must be able to fetch the exact artifact the
control plane produced, verify it, execute it, and publish new artifacts back.
This demo proves that path with a real published checkpoint and a real
S3-compatible endpoint:

1. import `distilgpt2` (or `--model`) into a local content-addressed store;
2. push the artifact and its parent lineage to the bucket with SigV4-signed,
   streamed (multipart for large payloads) uploads;
3. re-download it into an empty store, verifying every SHA-256 and the UZEN
   tensor fingerprint before publication;
4. execute the *fetched* artifact's real NIR graph and compare logits and greedy
   generation against the source;
5. confirm that an unknown artifact id is refused instead of invented.

The endpoint can be AWS S3, MinIO, Ceph, Cloudflare R2 or any other
S3-compatible service. Credentials never appear in the report or on stdout.

Usage::

    UZEN_S3_ENDPOINT=http://127.0.0.1:9000 \
    UZEN_S3_BUCKET=uzen-artifacts \
    UZEN_S3_ACCESS_KEY=... UZEN_S3_SECRET_KEY=... \
    HF_HUB_DISABLE_XET=1 .venv/bin/python examples/object_store_transport_demo.py \
        --model distilgpt2 --root /tmp/uzen-object-store-demo

Requires the ``hf``, ``safetensors`` and ``tokenizer`` extras for the model and
no extra dependency for the transport (SigV4 uses the standard library).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np  # noqa: E402 - after the environment default

from uzen import Uzen  # noqa: E402
from uzen.core.nws.store import (  # noqa: E402
    LINEAGE_NAME,
    MANIFEST_NAME,
    TENSORS_NAME,
    ArtifactStore,
)
from uzen.errors import (  # noqa: E402
    ArtifactNotFound,
    ObjectStoreUnavailable,
)
from uzen.infrastructure.storage.object_store import (  # noqa: E402
    ObjectStoreArtifactTransport,
    S3Client,
    S3Config,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="distilgpt2", help="Hub id or local checkpoint directory")
    parser.add_argument("--root", default="/tmp/uzen-object-store-demo")
    parser.add_argument("--prompt", default="Hello, my name is")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--endpoint", default=os.environ.get("UZEN_S3_ENDPOINT"))
    parser.add_argument("--bucket", default=os.environ.get("UZEN_S3_BUCKET"))
    parser.add_argument("--region", default=os.environ.get("UZEN_S3_REGION", "us-east-1"))
    parser.add_argument("--access-key", default=os.environ.get("UZEN_S3_ACCESS_KEY"))
    parser.add_argument("--secret-key", default=os.environ.get("UZEN_S3_SECRET_KEY"))
    parser.add_argument("--session-token", default=os.environ.get("UZEN_S3_SESSION_TOKEN"))
    parser.add_argument("--prefix", default=os.environ.get("UZEN_S3_PREFIX", "uzen/artifacts"))
    parser.add_argument(
        "--path-style",
        choices=("auto", "true", "false"),
        default="auto",
        help="bucket addressing; auto = path style for custom endpoints",
    )
    parser.add_argument("--create-bucket", action="store_true", help="create the bucket if absent")
    parser.add_argument(
        "--no-parents",
        action="store_true",
        help="transport only the requested artifact, not its lineage",
    )
    return parser


def _config(args: argparse.Namespace) -> S3Config:
    missing = [
        name
        for name, value in (
            ("--bucket / UZEN_S3_BUCKET", args.bucket),
            ("--access-key / UZEN_S3_ACCESS_KEY", args.access_key),
            ("--secret-key / UZEN_S3_SECRET_KEY", args.secret_key),
        )
        if not value
    ]
    if missing:
        raise ObjectStoreUnavailable(
            "no S3-compatible endpoint is configured",
            missing=missing,
            hint="pass the flags or set the UZEN_S3_* environment variables",
        )
    path_style = None if args.path_style == "auto" else args.path_style == "true"
    return S3Config(
        bucket=str(args.bucket),
        access_key=str(args.access_key),
        secret_key=str(args.secret_key),
        region=args.region,
        endpoint=args.endpoint or None,
        prefix=args.prefix,
        session_token=args.session_token,
        path_style=path_style,
    )


def _artifact_bytes(store: ArtifactStore, artifact_id: str) -> dict[str, int]:
    directory = store.artifact_dir(artifact_id)
    sizes = {}
    for name in (MANIFEST_NAME, TENSORS_NAME, LINEAGE_NAME):
        path = directory / name
        if path.exists():
            sizes[name] = path.stat().st_size
    return sizes


def _logits(payload: dict[str, Any]) -> np.ndarray:
    outputs = payload.get("outputs") or {}
    if "logits" not in outputs:
        raise SystemExit(f"execution did not produce logits; outputs: {sorted(outputs)}")
    return np.asarray(outputs["logits"]["values"], dtype=np.float64)


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = _config(args)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    source_root = root / "source"
    fetched_root = root / "fetched"
    shutil.rmtree(fetched_root, ignore_errors=True)

    client = Uzen(root=source_root)
    print(f"importing {args.model} ...")
    if Path(args.model).exists():
        imported = client.import_model(Path(args.model))
    else:
        imported = client.import_model(args.model, repo_id=args.model)
    artifact_id = str(imported["artifact_id"])
    compatibility = imported.get("compatibility") or {}
    print(
        f"source artifact: {artifact_id[:12]}… tensors={imported['num_tensors']} "
        f"overall={compatibility.get('overall')}"
    )

    s3 = S3Client(config)
    print(f"bucket: s3://{config.bucket} ({config.endpoint or 'aws'}, prefix={config.prefix})")
    if args.create_bucket:
        print(f"ensure bucket: {s3.ensure_bucket()}")

    transport = ObjectStoreArtifactTransport(s3)
    pushed = transport.push_artifact(
        client.service.store, artifact_id, include_parents=not args.no_parents
    )
    print(
        f"push: artifacts={len(pushed['artifacts'])} objects={len(pushed['uploaded'])} "
        f"skipped={len(pushed['skipped'])} bytes={pushed['bytes_uploaded']} "
        f"seconds={pushed['seconds']}"
    )
    uploaded_digests = {entry["key"]: entry["sha256"] for entry in pushed["uploaded"]}
    remote_checks = []
    for key in [*uploaded_digests, *pushed["skipped"]]:
        remote = s3.head(key)
        expected = uploaded_digests.get(key)
        remote_checks.append(
            {
                "key": key,
                "exists": bool(remote.get("exists")),
                "sha256": remote.get("sha256"),
                "sha256_match": bool(remote.get("sha256"))
                and (expected is None or remote.get("sha256") == expected),
                "bytes": remote.get("bytes"),
            }
        )

    fetched_store = ArtifactStore(fetched_root / "artifacts")
    fetched = ObjectStoreArtifactTransport(s3).fetch_artifact(
        fetched_store, artifact_id, include_parents=not args.no_parents
    )
    print(
        f"fetch: fetched={len(fetched['fetched'])} present={len(fetched['already_present'])} "
        f"bytes={fetched['bytes_downloaded']} verified={fetched['verified']} "
        f"seconds={fetched['seconds']}"
    )

    restored = Uzen(root=fetched_root)
    source_forward = client.forward_text(artifact_id, text=args.prompt, include_values=True)
    fetched_forward = restored.forward_text(artifact_id, text=args.prompt, include_values=True)
    source_logits = _logits(source_forward)
    fetched_logits = _logits(fetched_forward)
    shapes_equal = source_logits.shape == fetched_logits.shape
    max_abs_error = (
        float(np.max(np.abs(source_logits - fetched_logits))) if shapes_equal else None
    )
    mean_abs_error = (
        float(np.mean(np.abs(source_logits - fetched_logits))) if shapes_equal else None
    )
    print(
        f"forward: shape={list(fetched_logits.shape)} "
        f"max_abs_error={max_abs_error:.3e}" if max_abs_error is not None else "forward: shape mismatch"
    )

    source_generation = client.generate_text(
        artifact_id, text=args.prompt, max_new_tokens=args.max_new_tokens
    )
    fetched_generation = restored.generate_text(
        artifact_id, text=args.prompt, max_new_tokens=args.max_new_tokens
    )
    generation_equal = (
        source_generation.get("generated_token_ids")
        == fetched_generation.get("generated_token_ids")
    )
    print(
        f"generation: {fetched_generation.get('generated_text')!r} "
        f"(equal={generation_equal})"
    )

    missing_artifact_refused = False
    missing_detail = None
    try:
        ObjectStoreArtifactTransport(s3).fetch_artifact(fetched_store, "0" * 64)
    except ArtifactNotFound as exc:
        missing_artifact_refused = True
        missing_detail = exc.message
    except Exception as exc:  # a transport error is not proof of a refusal
        missing_detail = f"{type(exc).__name__}: {exc}"

    verified = restored.verify_artifact(artifact_id)
    sizes = _artifact_bytes(client.service.store, artifact_id)
    gates = {
        "artifact imported and validated": compatibility.get("overall") == "validated",
        "all remote objects readable with matching digest": bool(remote_checks)
        and all(check["exists"] and check["sha256_match"] for check in remote_checks),
        "artifact fetched and integrity-verified": bool(fetched["verified"]),
        "fetched graph executes with identical logits": shapes_equal
        and max_abs_error is not None
        and max_abs_error <= 1e-6,
        "fetched tokenizer generates identical tokens": generation_equal,
        "unknown artifact refused": missing_artifact_refused,
    }

    report = {
        "demo": "object_store_transport",
        "use_case": "distribute a validated model artifact through S3-compatible storage and execute the fetched copy",
        "model": args.model,
        "artifact_id": artifact_id,
        "artifact_files": sizes,
        "source_store": str(source_root),
        "fetched_store": str(fetched_root),
        "object_store": config.describe(),
        "push": pushed,
        "remote_checks": remote_checks,
        "fetch": fetched,
        "source_forward": source_forward,
        "fetched_forward": fetched_forward,
        "comparison": {
            "logits_shape": list(source_logits.shape),
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
            "generation_source": source_generation.get("generated_text"),
            "generation_fetched": fetched_generation.get("generated_text"),
            "generation_equal": generation_equal,
        },
        "restored_verification": verified,
        "unknown_artifact_control": {
            "refused": missing_artifact_refused,
            "detail": missing_detail,
        },
        "gates": gates,
        "ok": all(gates.values()),
    }
    report_path = reports / "object_store_transport.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("\n=== gates ===")
    for name, passed in gates.items():
        print(f"{name}: {passed}")
    print(f"\nreport: {report_path}")
    print(f"gate: {'PASS' if report['ok'] else 'FAIL'}")
    return report


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run(args)
    except ObjectStoreUnavailable as exc:
        print(
            "object-store transport is not configured: "
            + ", ".join(exc.details.get("missing", []))
            + "\nSet UZEN_S3_BUCKET, UZEN_S3_ACCESS_KEY, UZEN_S3_SECRET_KEY and "
            "UZEN_S3_ENDPOINT (for MinIO/R2/Ceph), or pass the matching flags.",
            file=sys.stderr,
        )
        return 2
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
