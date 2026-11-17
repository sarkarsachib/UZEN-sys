"""Object-store transport tests against a real HTTP S3-compatible endpoint.

The test server implements the S3 REST operations the client uses (PutObject,
GetObject, HeadObject, DeleteObject, CreateBucket and the multipart sequence) and
independently verifies every AWS Signature Version 4 request, so a broken
canonical request or signing key fails the test instead of being accepted by a
permissive fake.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import numpy as np
import pytest

from uzen.core.nws.store import ArtifactStore
from uzen.errors import (
    ArtifactIntegrityError,
    ObjectStoreError,
    ObjectStoreUnavailable,
)
from uzen.infrastructure.storage.object_store import (
    S3Client,
    S3Config,
    ObjectStoreArtifactTransport,
    sign_v4,
)

ACCESS_KEY = "UZENTESTACCESS"
SECRET_KEY = "uzen-test-secret-key"
REGION = "us-east-1"


# --------------------------------------------------------------------------- #
# independent SigV4 verifier (written from the AWS specification)
# --------------------------------------------------------------------------- #


def _expected_signature(
    method: str, path: str, raw_query: str, headers: dict[str, str], payload_hash: str
) -> tuple[str, str]:
    authorization = headers["authorization"]
    signed_headers = authorization.split("SignedHeaders=")[1].split(",")[0]
    canonical_headers = "".join(
        f"{name}:{headers[name].strip()}\n" for name in signed_headers.split(";")
    )
    canonical_request = "\n".join(
        [method, path, raw_query, canonical_headers, signed_headers, payload_hash]
    )
    amz_date = headers["x-amz-date"]
    datestamp = amz_date[:8]
    scope = f"{datestamp}/{REGION}/s3/aws4_request"
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ]
    )
    key = ("AWS4" + SECRET_KEY).encode()
    for part in (datestamp, REGION, "s3", "aws4_request"):
        key = hmac.new(key, part.encode(), hashlib.sha256).digest()
    return hmac.new(key, string_to_sign.encode(), hashlib.sha256).hexdigest(), scope


class _S3State:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.metadata: dict[str, dict[str, str]] = {}
        self.buckets: set[str] = set()
        self.uploads: dict[str, dict[str, bytes]] = {}
        self.fail_next_put = 0
        self.requests: list[tuple[str, str]] = []


def _make_handler(state: _S3State):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # keep pytest output clean
            pass

        # -- helpers -------------------------------------------------- #

        def _split(self):
            parts = urlsplit(self.path)
            path = parts.path
            segments = path.lstrip("/").split("/", 1)
            bucket = unquote(segments[0]) if segments and segments[0] else ""
            key = unquote(segments[1]) if len(segments) > 1 else ""
            if key.endswith("/"):
                key = key[:-1]
            return bucket, key, parts.query, path

        def _read_body(self) -> bytes:
            length = int(self.headers.get("Content-Length", "0") or 0)
            return self.rfile.read(length) if length else b""

        def _authorized(self, body: bytes) -> bool:
            if "Authorization" not in self.headers:
                self._xml_error(403, "AccessDenied", "missing authorization")
                return False
            payload_hash = self.headers.get("x-amz-content-sha256", "")
            expected_hash = hashlib.sha256(body).hexdigest()
            if payload_hash != expected_hash:
                self._xml_error(400, "XAmzContentSHA256Mismatch", "payload hash mismatch")
                return False
            bucket, _, raw_query, path = self._split()
            headers = {name.lower(): value for name, value in self.headers.items()}
            try:
                expected, scope = _expected_signature(
                    self.command, path, raw_query, headers, payload_hash
                )
            except (KeyError, IndexError):
                self._xml_error(403, "AccessDenied", "malformed authorization")
                return False
            if not self.headers["Authorization"].endswith(expected) or scope.split("/")[1] != "us-east-1":
                self._xml_error(403, "SignatureDoesNotMatch", "signature mismatch")
                return False
            return True

        def _xml_error(self, status: int, code: str, message: str) -> None:
            body = (
                f"<?xml version=\"1.0\"?><Error><Code>{code}</Code>"
                f"<Message>{message}</Message></Error>"
            ).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/xml")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _send(self, status: int, body: bytes = b"", headers: dict[str, str] | None = None) -> None:
            self.send_response(status)
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD" and body:
                self.wfile.write(body)

        # -- verbs ---------------------------------------------------- #

        def do_PUT(self) -> None:
            body = self._read_body()
            if not self._authorized(body):
                return
            bucket, key, raw_query, _ = self._split()
            query = parse_qs(raw_query, keep_blank_values=True)
            state.requests.append(("PUT", f"{bucket}/{key}"))
            if state.fail_next_put:
                state.fail_next_put -= 1
                self._xml_error(503, "SlowDown", "injected failure")
                return
            if not key:
                state.buckets.add(bucket)
                self._send(200)
                return
            if bucket not in state.buckets:
                state.buckets.add(bucket)
            if "uploadId" in query:
                upload_id = query["uploadId"][0]
                number = query["partNumber"][0]
                state.uploads.setdefault(upload_id, {})[number] = body
                etag = hashlib.md5(body).hexdigest()
                self._send(200, headers={"ETag": f'"{etag}"'})
                return
            state.objects[f"{bucket}/{key}"] = body
            meta = {
                name.lower(): value
                for name, value in self.headers.items()
                if name.lower().startswith("x-amz-meta-")
            }
            state.metadata[f"{bucket}/{key}"] = meta
            self._send(200, headers={"ETag": f'"{hashlib.md5(body).hexdigest()}"'})

        def do_POST(self) -> None:
            body = self._read_body()
            if not self._authorized(body):
                return
            bucket, key, raw_query, _ = self._split()
            query = parse_qs(raw_query, keep_blank_values=True)
            state.requests.append(("POST", f"{bucket}/{key}"))
            if "uploads" in query:
                upload_id = uuid.uuid4().hex
                state.uploads[upload_id] = {}
                payload = (
                    "<?xml version=\"1.0\"?><InitiateMultipartUploadResult>"
                    f"<Bucket>{bucket}</Bucket><Key>{key}</Key>"
                    f"<UploadId>{upload_id}</UploadId></InitiateMultipartUploadResult>"
                ).encode()
                self._send(200, payload, {"Content-Type": "application/xml"})
                return
            if "uploadId" in query:
                upload_id = query["uploadId"][0]
                parts = state.uploads.pop(upload_id, {})
                ordered = b"".join(parts[number] for number in sorted(parts, key=int))
                state.objects[f"{bucket}/{key}"] = ordered
                payload = (
                    "<?xml version=\"1.0\"?><CompleteMultipartUploadResult>"
                    f"<Bucket>{bucket}</Bucket><Key>{key}</Key>"
                    f"<ETag>\"{hashlib.md5(ordered).hexdigest()}\"</ETag>"
                    "</CompleteMultipartUploadResult>"
                ).encode()
                self._send(200, payload, {"Content-Type": "application/xml"})
                return
            self._xml_error(400, "InvalidRequest", "unsupported POST")

        def do_GET(self) -> None:
            if not self._authorized(b""):
                return
            bucket, key, raw_query, _ = self._split()
            state.requests.append(("GET", f"{bucket}/{key}"))
            data = state.objects.get(f"{bucket}/{key}")
            if data is None:
                self._xml_error(404, "NoSuchKey", "not found")
                return
            headers = {"Content-Type": "application/octet-stream"}
            headers.update(state.metadata.get(f"{bucket}/{key}", {}))
            self._send(200, data, headers)

        def do_HEAD(self) -> None:
            if not self._authorized(b""):
                return
            bucket, key, _, _ = self._split()
            state.requests.append(("HEAD", f"{bucket}/{key}"))
            if not key:
                self._send(200 if bucket in state.buckets else 404)
                return
            data = state.objects.get(f"{bucket}/{key}")
            if data is None:
                self._send(404)
                return
            headers = {"Content-Type": "application/octet-stream"}
            headers.update(state.metadata.get(f"{bucket}/{key}", {}))
            self._send(200, headers=headers)

        def do_DELETE(self) -> None:
            if not self._authorized(b""):
                return
            bucket, key, raw_query, _ = self._split()
            query = parse_qs(raw_query, keep_blank_values=True)
            state.requests.append(("DELETE", f"{bucket}/{key}"))
            if "uploadId" in query:
                state.uploads.pop(query["uploadId"][0], None)
                self._send(204)
                return
            state.objects.pop(f"{bucket}/{key}", None)
            self._send(204)

    return Handler


@pytest.fixture()
def s3_server():
    state = _S3State()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(state))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        yield state, f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _config(endpoint: str, **overrides) -> S3Config:
    values = dict(
        bucket="uzen-artifacts",
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        region=REGION,
        endpoint=endpoint,
        prefix="uzen/artifacts",
        path_style=True,
        part_size=5 * (1 << 20),
        multipart_threshold=5 * (1 << 20),
        max_attempts=3,
    )
    values.update(overrides)
    return S3Config(**values)


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #


def test_signing_is_deterministic_and_covers_payload_and_host():
    from datetime import datetime, timezone

    moment = datetime(2013, 5, 24, 0, 0, 0, tzinfo=timezone.utc)
    args = dict(
        method="GET",
        host="examplebucket.s3.amazonaws.com",
        path="/test.txt",
        query=None,
        headers={"range": "bytes=0-9"},
        payload_hash=hashlib.sha256(b"").hexdigest(),
        region=REGION,
        service="s3",
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        now=moment,
    )
    first = sign_v4(**args)
    second = sign_v4(**args)
    assert first == second
    assert first["x-amz-date"] == "20130524T000000Z"
    assert first["x-amz-content-sha256"] == hashlib.sha256(b"").hexdigest()
    assert "SignedHeaders=host;range;x-amz-content-sha256;x-amz-date" in first["Authorization"]
    assert first["Authorization"].startswith(
        f"AWS4-HMAC-SHA256 Credential={ACCESS_KEY}/20130524/us-east-1/s3/aws4_request"
    )


def test_put_get_head_delete_roundtrip(s3_server, tmp_path):
    state, endpoint = s3_server
    client = S3Client(_config(endpoint))
    source = tmp_path / "payload.bin"
    source.write_bytes(b"uzen-object-store-" * 100)

    client.ensure_bucket()
    result = client.put_file(
        source, "artifacts/demo/tensors.bin", metadata={"artifact-id": "abc"}
    )
    assert result["multipart"] is False
    assert result["bytes"] == source.stat().st_size

    head = client.head("artifacts/demo/tensors.bin")
    assert head["exists"] is True
    assert head["sha256"] == result["sha256"]

    destination = tmp_path / "roundtrip.bin"
    downloaded = client.get_file("artifacts/demo/tensors.bin", destination)
    assert destination.read_bytes() == source.read_bytes()
    assert downloaded["sha256"] == result["sha256"]

    missing = client.head("artifacts/demo/missing.bin")
    assert missing["exists"] is False

    client.delete("artifacts/demo/tensors.bin")
    assert client.head("artifacts/demo/tensors.bin")["exists"] is False


def test_multipart_upload_is_real_and_verified(s3_server, tmp_path):
    state, endpoint = s3_server
    client = S3Client(_config(endpoint))
    client.ensure_bucket()

    payload = np.random.default_rng(7).integers(0, 256, size=11 * 1024 * 1024, dtype=np.uint8)
    source = tmp_path / "large.bin"
    source.write_bytes(payload.tobytes())
    result = client.put_file(source, "artifacts/large.bin")
    assert result["multipart"] is True
    assert result["parts"] == 3
    assert state.objects["uzen-artifacts/artifacts/large.bin"] == source.read_bytes()

    destination = tmp_path / "large-copy.bin"
    client.get_file(
        "artifacts/large.bin", destination, expected_sha256=result["sha256"]
    )
    assert destination.read_bytes() == source.read_bytes()


def test_download_rejects_tampered_object(s3_server, tmp_path):
    state, endpoint = s3_server
    client = S3Client(_config(endpoint))
    client.ensure_bucket()
    source = tmp_path / "payload.bin"
    source.write_bytes(b"original-bytes")
    result = client.put_file(source, "artifacts/tamper.bin")

    state.objects["uzen-artifacts/artifacts/tamper.bin"] = b"tampered-bytes"
    destination = tmp_path / "tampered.bin"
    with pytest.raises(ArtifactIntegrityError):
        client.get_file(
            "artifacts/tamper.bin", destination, expected_sha256=result["sha256"]
        )
    assert not destination.exists()


def test_client_retries_transient_failures(s3_server, tmp_path):
    state, endpoint = s3_server
    client = S3Client(_config(endpoint))
    client.ensure_bucket()
    state.fail_next_put = 1
    source = tmp_path / "retry.bin"
    source.write_bytes(b"retry-me")
    result = client.put_file(source, "artifacts/retry.bin")
    assert result["bytes"] == len(b"retry-me")
    puts = [entry for entry in state.requests if entry[0] == "PUT" and "retry" in entry[1]]
    assert len(puts) == 2


def test_object_store_error_carries_status(s3_server, tmp_path):
    state, endpoint = s3_server
    client = S3Client(_config(endpoint))
    with pytest.raises(ObjectStoreError) as excinfo:
        client.get_bytes("artifacts/does-not-exist.json")
    assert excinfo.value.details.get("status") == 404


def test_config_from_env_refuses_missing_credentials(monkeypatch):
    for name in ("UZEN_S3_BUCKET", "UZEN_S3_ACCESS_KEY", "UZEN_S3_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ObjectStoreUnavailable) as excinfo:
        S3Config.from_env()
    assert "UZEN_S3_BUCKET" in excinfo.value.details["missing"]


def _two_generation_store(tmp_path: Path) -> tuple[ArtifactStore, str, str]:
    from uzen.core.nws.families import build_architecture

    definition = build_architecture(
        "mlp", {"input_dim": 8, "hidden_dims": [16, 12], "output_dim": 4}
    )
    package = definition.build_package(seed=5, name="transport-base")
    store = ArtifactStore(tmp_path / "source-store")
    parent = store.put(package, operation="random_initialization", note="base")
    child = store.put(
        package,
        operation="w.arithmetic.scale",
        parents=[parent.artifact_id],
        operations=[{"op": "w.arithmetic.scale", "params": {"factor": 1.0}}],
        rng_seeds={},
    )
    assert child.artifact_id != parent.artifact_id
    return store, parent.artifact_id, child.artifact_id


def test_transport_pushes_and_fetches_parent_lineage(s3_server, tmp_path):
    state, endpoint = s3_server
    client = S3Client(_config(endpoint))
    client.ensure_bucket()
    source_store, parent_id, child_id = _two_generation_store(tmp_path)

    transport = ObjectStoreArtifactTransport(client)
    pushed = transport.push_artifact(source_store, child_id)
    assert set(pushed["artifacts"]) == {parent_id, child_id}
    assert pushed["bytes_uploaded"] > 0
    assert all(entry["multipart"] is False for entry in pushed["uploaded"])

    target_store = ArtifactStore(tmp_path / "target-store")
    fetched = ObjectStoreArtifactTransport(client).fetch_artifact(target_store, child_id)
    assert set(fetched["fetched"]) == {parent_id, child_id}
    assert fetched["verified"] is True

    original = source_store.verify(child_id)
    transported = target_store.verify(child_id)
    assert transported["ok"] is True
    assert transported["tensor_digest"] == original["tensor_digest"]

    package = target_store.get(child_id)
    assert package.name == source_store.get(child_id).name

    again = ObjectStoreArtifactTransport(client).push_artifact(source_store, child_id)
    assert again["uploaded"] == []
    assert len(again["skipped"]) == len(pushed["uploaded"])


def test_transport_carries_the_stored_tokenizer(s3_server, tmp_path):
    from uzen.core.nws.families import build_architecture
    from uzen.core.wie.tokenization import store_tokenizer, tokenizer_path

    state, endpoint = s3_server
    client = S3Client(_config(endpoint))
    client.ensure_bucket()
    definition = build_architecture(
        "mlp", {"input_dim": 8, "hidden_dims": [16, 12], "output_dim": 4}
    )
    source_store = ArtifactStore(tmp_path / "source-workspace" / "artifacts")
    document = (
        b'{"version":"1.0","model":{"type":"BPE","vocab":{"a":0,"b":1},'
        b'"merges":[]},"added_tokens":[]}'
    )
    spec = store_tokenizer(source_store.root.parent, document)
    record = source_store.put(
        definition.build_package(seed=3),
        operation="tokenizer.attach",
        metadata={"tokenizer": spec},
    )
    transport = ObjectStoreArtifactTransport(client)
    pushed = transport.push_artifact(source_store, record.artifact_id, include_parents=False)
    assert any(entry["file"] == "tokenizer.json" for entry in pushed["uploaded"])

    target_store = ArtifactStore(tmp_path / "target-workspace" / "artifacts")
    ObjectStoreArtifactTransport(client).fetch_artifact(
        target_store, record.artifact_id, include_parents=False
    )
    fetched = tokenizer_path(target_store.root.parent, spec["sha256"])
    assert fetched.is_file()
    assert hashlib.sha256(fetched.read_bytes()).hexdigest() == spec["sha256"]
    assert fetched.read_bytes() == document


@pytest.mark.skipif(
    os.environ.get("UZEN_LIVE_S3") != "1",
    reason="set UZEN_LIVE_S3=1 (and UZEN_S3_* variables) to test a real S3 endpoint",
)
def test_live_s3_artifact_roundtrip(tmp_path):
    base = os.environ.get("UZEN_S3_TEST_PREFIX", "uzen/live-test")
    config = S3Config.from_env(prefix=f"{base}/{uuid.uuid4().hex[:8]}")
    client = S3Client(config)
    source_store, parent_id, child_id = _two_generation_store(tmp_path)
    transport = ObjectStoreArtifactTransport(client)
    pushed = transport.push_artifact(source_store, child_id)
    assert pushed["bytes_uploaded"] > 0

    target_store = ArtifactStore(tmp_path / "live-target-store")
    fetched = ObjectStoreArtifactTransport(client).fetch_artifact(target_store, child_id)
    assert set(fetched["fetched"]) == {parent_id, child_id}
    assert fetched["verified"] is True


def test_transport_refuses_corrupted_remote_payload(s3_server, tmp_path):
    state, endpoint = s3_server
    client = S3Client(_config(endpoint))
    client.ensure_bucket()
    source_store, parent_id, child_id = _two_generation_store(tmp_path)
    transport = ObjectStoreArtifactTransport(client)
    transport.push_artifact(source_store, child_id)

    key = transport.artifact_key(child_id, "tensors.safetensors")
    original = state.objects[f"uzen-artifacts/{key}"]
    state.objects[f"uzen-artifacts/{key}"] = original[:-16] + b"tampered-payload"

    target_store = ArtifactStore(tmp_path / "target-store")
    with pytest.raises(ArtifactIntegrityError):
        ObjectStoreArtifactTransport(client).fetch_artifact(target_store, child_id)
    assert target_store.exists(child_id) is False
