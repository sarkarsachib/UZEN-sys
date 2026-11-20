"""Content-addressed, replayable benchmark corpus.

Each successful trial produces one JSON record keyed by the hash of its full
description (campaign spec, candidate, evaluation settings and measured metrics).
The filename is the digest, so two processes cannot silently overwrite different
results. ``verify`` rebuilds the candidate from the record and executes it again;
it never trusts the stored numbers.

Cross-host floating point can differ in the last bits. Verification uses a tight
relative tolerance and reports the measured deltas rather than claiming bitwise
identity across BLAS implementations.
"""

from __future__ import annotations

import hashlib
import os
import platform
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from uzen.core.campaign.evaluate import Evaluator
from uzen.core.campaign.models import Candidate
from uzen.core.campaign.spec import CampaignSpec
from uzen.errors import ArtifactIntegrityError
from uzen.util.canonical import canonical_bytes, canonical_json, read_json, utc_now_iso

CORPUS_SCHEMA = "uzen.benchmark/1"
ENGINE_VERSION = "uzen.campaign/1"
DEFAULT_TOLERANCE = 1e-9


def _environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _payload(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key not in ("digest", "recorded_at")}


class BenchmarkCorpus:
    def __init__(self, root: str | Path) -> None:
        self.directory = Path(root) / "benchmarks"
        self.directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # read
    # ------------------------------------------------------------------ #

    def digests(self) -> list[str]:
        return sorted(path.stem for path in self.directory.glob("*.json"))

    def merkle(self) -> str:
        return _digest({"schema": CORPUS_SCHEMA, "records": self.digests()})

    def path_for(self, digest: str) -> Path:
        if not isinstance(digest, str) or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ArtifactIntegrityError(f"not a benchmark digest: {digest!r}")
        return self.directory / f"{digest}.json"

    def get(self, digest: str) -> dict[str, Any]:
        path = self.path_for(digest)
        if not path.exists():
            raise ArtifactIntegrityError(f"benchmark record '{digest}' not found")
        record = read_json(path)
        if not isinstance(record, dict):
            raise ArtifactIntegrityError(f"benchmark record '{digest}' is not an object")
        if record.get("digest") != digest:
            raise ArtifactIntegrityError(
                f"benchmark record '{digest}' has a mismatched digest field",
                recorded=record.get("digest"),
            )
        actual = _digest(_payload(record))
        if actual != digest:
            raise ArtifactIntegrityError(
                f"benchmark record '{digest}' content does not hash to its filename",
                computed=actual,
            )
        return record

    def records(self, digests: Iterable[str] | None = None) -> list[dict[str, Any]]:
        selected = list(digests) if digests is not None else self.digests()
        return [self.get(digest) for digest in selected]

    # ------------------------------------------------------------------ #
    # write
    # ------------------------------------------------------------------ #

    def append(
        self,
        spec: CampaignSpec,
        candidate: Candidate,
        metrics: Mapping[str, Any],
        *,
        package_fingerprint: str = "",
        graph_digest: str = "",
        status: str = "ok",
        error: str = "",
    ) -> str:
        payload = {
            "schema": CORPUS_SCHEMA,
            "engine_version": ENGINE_VERSION,
            "campaign_spec": spec.to_dict(),
            "candidate": candidate.to_dict(),
            "evaluation": dict(spec.evaluation or {}),
            "status": status,
            "error": error,
            "metrics": dict(metrics),
            "package_fingerprint": package_fingerprint,
            "graph_digest": graph_digest,
            "environment": _environment(),
        }
        digest = _digest(payload)
        path = self.path_for(digest)
        if path.exists():
            # Idempotent append. Verify the on-disk content still hashes correctly.
            self.get(digest)
            return digest
        record = {**payload, "digest": digest, "recorded_at": utc_now_iso()}
        self._write(path, record)
        return digest

    def _write(self, path: Path, record: Mapping[str, Any]) -> None:
        fd, temp_name = tempfile.mkstemp(
            prefix=".benchmark-", suffix=".json.tmp", dir=self.directory
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(canonical_json(record))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    # ------------------------------------------------------------------ #
    # replay / verification
    # ------------------------------------------------------------------ #

    def replay(self, digest: str) -> dict[str, Any]:
        record = self.get(digest)
        spec = CampaignSpec.from_dict(record["campaign_spec"])
        candidate_payload = record["candidate"]
        candidate = Candidate.build(
            spec.mode,
            spec.family,
            tuple(candidate_payload["values"]),
            candidate_payload["config"],
        )
        try:
            metrics, fingerprint, graph_digest = Evaluator(spec).evaluate(candidate)
            recomputed = {
                "status": "ok",
                "metrics": metrics,
                "package_fingerprint": fingerprint,
                "graph_digest": graph_digest,
            }
        except Exception as exc:  # replay must report failure, not crash the verifier
            recomputed = {
                "status": "error",
                "metrics": {},
                "package_fingerprint": "",
                "graph_digest": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
        differences = compare_records(record, recomputed, tolerance=DEFAULT_TOLERANCE)
        return {
            "digest": digest,
            "record": record,
            "recomputed": recomputed,
            "matches": not differences,
            "differences": differences,
        }

    def verify(
        self,
        digests: Iterable[str] | None = None,
        *,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> dict[str, Any]:
        selected = list(digests) if digests is not None else self.digests()
        mismatches: list[dict[str, Any]] = []
        checked = 0
        for digest in selected:
            checked += 1
            result = self.replay(digest)
            if not result["matches"] or tolerance != DEFAULT_TOLERANCE:
                # ``replay`` already used the default tolerance; a stricter or looser
                # request is recompiled here so the caller's tolerance is honoured.
                differences = compare_records(
                    result["record"], result["recomputed"], tolerance=tolerance
                )
                if differences:
                    mismatches.append(
                        {
                            "digest": digest,
                            "differences": differences,
                        }
                    )
        return {
            "checked": checked,
            "matched": checked - len(mismatches),
            "mismatches": mismatches,
            "merkle": self.merkle(),
            "tolerance": tolerance,
        }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def compare_records(
    recorded: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    for scalar in ("status", "package_fingerprint", "graph_digest"):
        if recorded.get(scalar) != recomputed.get(scalar):
            differences.append(
                {
                    "field": scalar,
                    "recorded": recorded.get(scalar),
                    "recomputed": recomputed.get(scalar),
                }
            )
    recorded_metrics = recorded.get("metrics") or {}
    recomputed_metrics = recomputed.get("metrics") or {}
    for name in sorted(set(recorded_metrics) | set(recomputed_metrics)):
        left = recorded_metrics.get(name)
        right = recomputed_metrics.get(name)
        if _is_number(left) and _is_number(right):
            delta = abs(float(left) - float(right))
            scale = max(1.0, abs(float(left)), abs(float(right)))
            if delta > tolerance * scale:
                differences.append(
                    {
                        "field": f"metrics.{name}",
                        "recorded": left,
                        "recomputed": right,
                        "absolute_delta": delta,
                    }
                )
        elif left != right:
            differences.append(
                {
                    "field": f"metrics.{name}",
                    "recorded": left,
                    "recomputed": right,
                }
            )
    return differences
