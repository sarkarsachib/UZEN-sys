"""S3-compatible object storage for distributing artifacts between workers.

This module implements the remote artifact transport that the worker daemon needs
when a shared filesystem is not available. It is a real S3 REST client:

* AWS Signature Version 4 is computed locally (HMAC-SHA256), with the payload
  SHA-256 committed in ``x-amz-content-sha256``;
* uploads are streamed from disk (single ``PutObject`` up to the configured
  threshold, then a real ``CreateMultipartUpload`` / ``UploadPart`` /
  ``CompleteMultipartUpload`` sequence);
* downloads are streamed to a temporary file and hashed before publication, so a
  truncated or corrupted object never enters an :class:`~uzen.core.nws.store.ArtifactStore`;
* retries re-sign each attempt (the signature covers ``x-amz-date``) and back off
  on 429/5xx and network errors.

An :class:`ObjectStoreArtifactTransport` pushes and fetches content-addressed
artifacts including their parent lineage, so a fetched artifact verifies with the
same checks as a locally produced one.

Nothing here fakes a result: without configured credentials
:meth:`S3Config.from_env` raises :class:`~uzen.errors.ObjectStoreUnavailable`, and
every transfer is checked against a SHA-256 digest.
"""

from __future__ import annotations

import hashlib
import hmac
import http.client
import json
import os
import shutil
import ssl
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote, urlsplit
from xml.etree import ElementTree

from uzen.core.nws.manifest import PackageManifest
from uzen.core.nws.store import (
    LINEAGE_NAME,
    MANIFEST_NAME,
    TENSORS_NAME,
    ArtifactStore,
)
from uzen.errors import (
    ArtifactIntegrityError,
    ArtifactNotFound,
    ObjectStoreError,
    ObjectStoreUnavailable,
)
from uzen.core.wie.tokenization import TOKENIZER_KIND, tokenizer_path
from uzen.infrastructure.storage.safetensors_io import (
    read_safetensors,
    safetensors_tensor_digest,
)
from uzen.util.canonical import sha256_bytes, sha256_file, utc_now_iso

SIGV4_ALGORITHM = "AWS4-HMAC-SHA256"
EMPTY_SHA256 = sha256_bytes(b"")
MIN_PART_SIZE = 5 * (1 << 20)
MAX_PART_SIZE = 5 * (1 << 30)
DEFAULT_PART_SIZE = 16 * (1 << 20)
DEFAULT_MULTIPART_THRESHOLD = 64 * (1 << 20)
MAX_BYTES_RESPONSE = 32 * (1 << 20)


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class S3Config:
    """Connection settings for an S3-compatible endpoint.

    ``endpoint=None`` addresses AWS S3 (``s3.<region>.amazonaws.com``, virtual-host
    style). A custom endpoint such as MinIO, Ceph, Cloudflare R2 or a local test
    server defaults to path-style addressing. Credentials are never logged;
    :meth:`describe` omits the secret key and session token.
    """

    bucket: str
    access_key: str
    secret_key: str
    region: str = "us-east-1"
    endpoint: str | None = None
    prefix: str = "uzen/artifacts"
    session_token: str | None = None
    path_style: bool | None = None
    part_size: int = DEFAULT_PART_SIZE
    multipart_threshold: int = DEFAULT_MULTIPART_THRESHOLD
    max_attempts: int = 4
    connect_timeout: float = 20.0
    read_timeout: float = 300.0

    def __post_init__(self) -> None:
        if not self.bucket:
            raise ValueError("S3 bucket must not be empty")
        if not self.access_key or not self.secret_key:
            raise ValueError("S3 access key and secret key are required")
        if not MIN_PART_SIZE <= self.part_size <= MAX_PART_SIZE:
            raise ValueError(
                f"part_size must be between {MIN_PART_SIZE} and {MAX_PART_SIZE} bytes"
            )
        if self.multipart_threshold < MIN_PART_SIZE:
            raise ValueError(
                f"multipart_threshold must be at least {MIN_PART_SIZE} bytes"
            )
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.endpoint:
            parts = urlsplit(self.endpoint)
            if parts.scheme not in ("http", "https") or not parts.hostname:
                raise ValueError(f"invalid S3 endpoint: {self.endpoint!r}")

    @property
    def resolved_path_style(self) -> bool:
        if self.path_style is not None:
            return self.path_style
        if self.endpoint is not None:
            return True
        # Virtual-host addressing cannot serve buckets whose names contain dots
        # over TLS, so those fall back to path style.
        return "." in self.bucket

    @classmethod
    def from_env(cls, *, prefix: str | None = None) -> "S3Config":
        """Build a config from ``UZEN_S3_*`` variables.

        Missing credentials raise :class:`ObjectStoreUnavailable` instead of
        silently falling back to an unauthenticated or simulated transport.
        """
        bucket = os.environ.get("UZEN_S3_BUCKET", "")
        access_key = os.environ.get("UZEN_S3_ACCESS_KEY", "")
        secret_key = os.environ.get("UZEN_S3_SECRET_KEY", "")
        missing = [
            name
            for name, value in (
                ("UZEN_S3_BUCKET", bucket),
                ("UZEN_S3_ACCESS_KEY", access_key),
                ("UZEN_S3_SECRET_KEY", secret_key),
            )
            if not value
        ]
        if missing:
            raise ObjectStoreUnavailable(
                "object-store transport is not configured",
                missing=missing,
                hint="set UZEN_S3_BUCKET, UZEN_S3_ACCESS_KEY and UZEN_S3_SECRET_KEY",
            )
        raw_style = os.environ.get("UZEN_S3_PATH_STYLE")
        path_style: bool | None = None
        if raw_style:
            path_style = raw_style.strip().lower() in ("1", "true", "yes", "on")
        return cls(
            bucket=bucket,
            access_key=access_key,
            secret_key=secret_key,
            region=os.environ.get("UZEN_S3_REGION", "us-east-1"),
            endpoint=os.environ.get("UZEN_S3_ENDPOINT") or None,
            prefix=prefix or os.environ.get("UZEN_S3_PREFIX") or "uzen/artifacts",
            session_token=os.environ.get("UZEN_S3_SESSION_TOKEN") or None,
            path_style=path_style,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "bucket": self.bucket,
            "region": self.region,
            "endpoint": self.endpoint or "https://s3.amazonaws.com",
            "prefix": self.prefix,
            "path_style": self.resolved_path_style,
            "part_size": self.part_size,
            "multipart_threshold": self.multipart_threshold,
            "max_attempts": self.max_attempts,
        }


# --------------------------------------------------------------------------- #
# SigV4
# --------------------------------------------------------------------------- #


def _canonical_query(params: Mapping[str, str] | None) -> str:
    if not params:
        return ""
    encoded = []
    for name in sorted(params):
        value = params[name]
        encoded.append(
            f"{quote(str(name), safe='-_.~')}={quote(str(value), safe='-_.~')}"
        )
    return "&".join(encoded)


def _signing_key(secret_key: str, datestamp: str, region: str, service: str) -> bytes:
    key = ("AWS4" + secret_key).encode("utf-8")
    for part in (datestamp, region, service, "aws4_request"):
        key = hmac.new(key, part.encode("utf-8"), hashlib.sha256).digest()
    return key


def sign_v4(
    *,
    method: str,
    host: str,
    path: str,
    query: Mapping[str, str] | None,
    headers: Mapping[str, str],
    payload_hash: str,
    region: str,
    service: str,
    access_key: str,
    secret_key: str,
    session_token: str | None = None,
    now: datetime | None = None,
) -> dict[str, str]:
    """Return the headers to send, including a SigV4 ``Authorization`` value.

    ``path`` must already be URI-encoded exactly as it will be sent (S3 requires a
    single encoding of each path segment). ``now`` exists so the signing logic can
    be checked against published test vectors.
    """
    moment = now or datetime.now(timezone.utc)
    amz_date = moment.strftime("%Y%m%dT%H%M%SZ")
    datestamp = moment.strftime("%Y%m%d")

    canonical: dict[str, str] = {
        name.lower(): " ".join(str(value).split()) for name, value in headers.items()
    }
    canonical["host"] = host
    canonical["x-amz-date"] = amz_date
    canonical["x-amz-content-sha256"] = payload_hash
    if session_token:
        canonical["x-amz-security-token"] = session_token

    names = sorted(canonical)
    canonical_headers = "".join(f"{name}:{canonical[name]}\n" for name in names)
    signed_headers = ";".join(names)
    canonical_request = "\n".join(
        [method, path, _canonical_query(query), canonical_headers, signed_headers, payload_hash]
    )
    scope = f"{datestamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join(
        [SIGV4_ALGORITHM, amz_date, scope, sha256_bytes(canonical_request.encode("utf-8"))]
    )
    signature = hmac.new(
        _signing_key(secret_key, datestamp, region, service),
        string_to_sign.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    out = {name: value for name, value in canonical.items() if name != "host"}
    out["Authorization"] = (
        f"{SIGV4_ALGORITHM} Credential={access_key}/{scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    return out


# --------------------------------------------------------------------------- #
# client
# --------------------------------------------------------------------------- #


@dataclass
class _Response:
    status: int
    headers: dict[str, str]
    body: bytes


class _Retryable(Exception):
    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class S3Client:
    """Minimal but complete S3 REST client for artifact-sized objects."""

    def __init__(self, config: S3Config) -> None:
        self.config = config

    # -- addressing ----------------------------------------------------- #

    def _target(self, key: str) -> tuple[str, int | None, str, str, str]:
        """Return ``(scheme, port, host, host_header, path)`` for an object key."""
        cfg = self.config
        if cfg.endpoint:
            parts = urlsplit(cfg.endpoint)
            scheme = parts.scheme
            host = parts.hostname or ""
            port = parts.port
            base_path = parts.path.rstrip("/")
        else:
            scheme = "https"
            host = f"s3.{cfg.region}.amazonaws.com"
            port = None
            base_path = ""

        default_port = 443 if scheme == "https" else 80
        host_header = host if port in (None, default_port) else f"{host}:{port}"

        encoded = "/".join(quote(segment, safe="-_.~") for segment in key.split("/")) if key else ""
        if cfg.resolved_path_style:
            prefix = f"{base_path}/{cfg.bucket}"
            path = f"{prefix}/{encoded}" if encoded else prefix
        else:
            host = f"{cfg.bucket}.{host}"
            host_header = host if port in (None, default_port) else f"{host}:{port}"
            prefix = base_path
            path = f"{prefix}/{encoded}" if encoded else (prefix or "/")
        if not path:
            path = "/"
        return scheme, port, host, host_header, path

    # -- transport ------------------------------------------------------ #

    def _send_once(
        self,
        method: str,
        key: str,
        *,
        query: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        body: bytes | None = None,
        body_path: Path | None = None,
        body_length: int | None = None,
        payload_hash: str | None = None,
        stream_to: Path | None = None,
    ) -> _Response:
        cfg = self.config
        scheme, port, host, host_header, path = self._target(key)
        if payload_hash is None:
            if body is not None:
                payload_hash = sha256_bytes(body)
            elif body_path is not None:
                raise ValueError("streaming upload requires a precomputed payload hash")
            else:
                payload_hash = EMPTY_SHA256

        signed_headers = sign_v4(
            method=method,
            host=host_header,
            path=path,
            query=query,
            headers=headers or {},
            payload_hash=payload_hash,
            region=cfg.region,
            service="s3",
            access_key=cfg.access_key,
            secret_key=cfg.secret_key,
            session_token=cfg.session_token,
        )

        if scheme == "https":
            connection: http.client.HTTPConnection = http.client.HTTPSConnection(
                host,
                port,
                timeout=cfg.read_timeout,
                context=ssl.create_default_context(),
            )
        else:
            connection = http.client.HTTPConnection(host, port, timeout=cfg.read_timeout)

        target = path
        encoded_query = _canonical_query(query)
        if encoded_query:
            target = f"{path}?{encoded_query}"
        try:
            connection.putrequest(
                method, target, skip_host=True, skip_accept_encoding=True
            )
            connection.putheader("Host", host_header)
            for name, value in signed_headers.items():
                connection.putheader(name, value)
            length = body_length if body_length is not None else (len(body) if body is not None else None)
            if body is not None or body_path is not None:
                connection.putheader("Content-Length", str(length if length is not None else 0))
            connection.endheaders()
            if body is not None:
                connection.send(body)
            elif body_path is not None:
                with open(body_path, "rb") as handle:
                    while True:
                        chunk = handle.read(1 << 20)
                        if not chunk:
                            break
                        connection.send(chunk)
            response = connection.getresponse()
            response_headers = {name.lower(): value for name, value in response.getheaders()}
            if stream_to is not None and 200 <= response.status < 300:
                with open(stream_to, "wb") as handle:
                    while True:
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        handle.write(chunk)
                data = b""
            else:
                data = response.read()
            result = _Response(status=response.status, headers=response_headers, body=data)
        finally:
            connection.close()

        if 200 <= result.status < 300:
            return result
        retryable = result.status in (408, 425, 429, 500, 502, 503, 504)
        error_code, error_message = _s3_error_detail(result.body)
        if retryable:
            raise _Retryable(
                f"S3 {method} returned HTTP {result.status}: {error_message or error_code}",
                status=result.status,
            )
        raise ObjectStoreError(
            f"S3 {method} {key or self.config.bucket} failed with HTTP {result.status}",
            status=result.status,
            code=error_code,
            response=error_message,
            key=key,
        )

    def _request(self, method: str, key: str = "", **kwargs: Any) -> _Response:
        last: Exception | None = None
        for attempt in range(self.config.max_attempts):
            try:
                return self._send_once(method, key, **kwargs)
            except _Retryable as exc:
                last = exc
                if attempt + 1 >= self.config.max_attempts:
                    break
                time.sleep(min(0.25 * (2**attempt), 5.0))
            except (OSError, http.client.HTTPException) as exc:
                last = exc
                if attempt + 1 >= self.config.max_attempts:
                    break
                time.sleep(min(0.25 * (2**attempt), 5.0))
        raise ObjectStoreError(
            f"S3 {method} {key or self.config.bucket} failed after "
            f"{self.config.max_attempts} attempts: {last}",
            attempts=self.config.max_attempts,
            key=key,
        )

    # -- operations ----------------------------------------------------- #

    def create_bucket(self) -> dict[str, Any]:
        try:
            self._request("PUT", "", body=b"")
        except ObjectStoreError as exc:
            if exc.details.get("status") in (200, 409):
                return {"bucket": self.config.bucket, "created": False}
            raise
        return {"bucket": self.config.bucket, "created": True}

    def head(self, key: str) -> dict[str, Any]:
        try:
            response = self._request("HEAD", key)
        except ObjectStoreError as exc:
            if exc.details.get("status") == 404:
                return {"key": key, "exists": False}
            raise
        headers = response.headers
        raw_length = headers.get("content-length")
        return {
            "key": key,
            "exists": True,
            "bytes": int(raw_length) if raw_length is not None else None,
            "sha256": headers.get("x-amz-meta-sha256"),
            "etag": headers.get("etag"),
            "content_type": headers.get("content-type"),
        }

    def get_bytes(self, key: str, *, limit: int = MAX_BYTES_RESPONSE) -> bytes:
        response = self._request("GET", key)
        if len(response.body) > limit:
            raise ObjectStoreError(
                f"object '{key}' exceeds the {limit}-byte in-memory limit", key=key
            )
        return response.body

    def put_file(
        self,
        path: str | os.PathLike[str],
        key: str,
        *,
        metadata: Mapping[str, str] | None = None,
        content_type: str = "application/octet-stream",
    ) -> dict[str, Any]:
        source = Path(path)
        size = source.stat().st_size
        digest = sha256_file(source)
        meta = {"sha256": digest, **{str(k): str(v) for k, v in (metadata or {}).items()}}
        meta_headers = {f"x-amz-meta-{name}": value for name, value in meta.items()}
        if size > self.config.multipart_threshold and size > 0:
            return self._multipart_upload(
                source,
                key,
                size=size,
                digest=digest,
                meta_headers=meta_headers,
                content_type=content_type,
            )
        self._request(
            "PUT",
            key,
            headers={"content-type": content_type, **meta_headers},
            body_path=source,
            body_length=size,
            payload_hash=digest,
        )
        return {
            "key": key,
            "bytes": size,
            "sha256": digest,
            "multipart": False,
            "parts": 1,
        }

    def _multipart_upload(
        self,
        source: Path,
        key: str,
        *,
        size: int,
        digest: str,
        meta_headers: Mapping[str, str],
        content_type: str,
    ) -> dict[str, Any]:
        initiated = self._request(
            "POST",
            key,
            query={"uploads": ""},
            headers={"content-type": content_type, **meta_headers},
        )
        upload_id = _xml_text(initiated.body, "UploadId")
        if not upload_id:
            raise ObjectStoreError(
                f"multipart upload for '{key}' did not return an UploadId", key=key
            )
        parts: list[tuple[int, str]] = []
        sent = 0
        try:
            with open(source, "rb") as handle:
                number = 1
                while True:
                    chunk = handle.read(self.config.part_size)
                    if not chunk:
                        break
                    part_hash = sha256_bytes(chunk)
                    response = self._request(
                        "PUT",
                        key,
                        query={"partNumber": str(number), "uploadId": upload_id},
                        body=chunk,
                        payload_hash=part_hash,
                    )
                    etag = response.headers.get("etag") or f'"{hashlib.md5(chunk).hexdigest()}"'
                    parts.append((number, etag))
                    sent += len(chunk)
                    number += 1
            if not parts:
                raise ObjectStoreError(
                    f"multipart upload for '{key}' produced no parts", key=key
                )
            body = _complete_multipart_body(parts)
            self._request(
                "POST",
                key,
                query={"uploadId": upload_id},
                headers={"content-type": "application/xml"},
                body=body,
                payload_hash=sha256_bytes(body),
            )
        except BaseException:
            try:
                self._request("DELETE", key, query={"uploadId": upload_id})
            except Exception:
                pass
            raise
        return {
            "key": key,
            "bytes": sent,
            "sha256": digest,
            "multipart": True,
            "parts": len(parts),
            "upload_id": upload_id,
        }

    def get_file(
        self,
        key: str,
        destination: str | os.PathLike[str],
        *,
        expected_sha256: str | None = None,
    ) -> dict[str, Any]:
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(
            dir=str(dest.parent), prefix=f".{dest.name}.", suffix=".part"
        )
        os.close(handle)
        temp = Path(temp_name)
        try:
            response = self._request("GET", key, stream_to=temp)
            declared = response.headers.get("x-amz-meta-sha256")
            expected = expected_sha256 or declared
            actual = sha256_file(temp)
            actual_bytes = temp.stat().st_size
            if expected and actual != expected:
                raise ArtifactIntegrityError(
                    f"object '{key}' failed digest verification",
                    key=key,
                    expected_sha256=expected,
                    actual_sha256=actual,
                )
            declared_length = response.headers.get("content-length")
            if declared_length is not None and int(declared_length) != actual_bytes:
                raise ArtifactIntegrityError(
                    f"object '{key}' length does not match Content-Length",
                    key=key,
                    expected_bytes=int(declared_length),
                    actual_bytes=actual_bytes,
                )
            os.replace(temp, dest)
        finally:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass
        return {"key": key, "path": str(dest), "bytes": dest.stat().st_size, "sha256": actual}

    def delete(self, key: str, *, missing_ok: bool = True) -> dict[str, Any]:
        try:
            self._request("DELETE", key)
        except ObjectStoreError as exc:
            if missing_ok and exc.details.get("status") == 404:
                return {"key": key, "deleted": False}
            raise
        return {"key": key, "deleted": True}

    def ensure_bucket(self) -> dict[str, Any]:
        try:
            response = self._request("HEAD", "")
        except ObjectStoreError as exc:
            if exc.details.get("status") != 404:
                raise
            return self.create_bucket()
        return {"bucket": self.config.bucket, "created": False, "status": response.status}


def _s3_error_detail(body: bytes) -> tuple[str | None, str | None]:
    if not body:
        return None, None
    try:
        ElementTree.fromstring(body)
    except ElementTree.ParseError:
        return None, body[:300].decode("utf-8", errors="replace")
    return _xml_text(body, "Code"), _xml_text(body, "Message")


def _xml_text(body: bytes, tag: str) -> str | None:
    try:
        root = ElementTree.fromstring(body)
    except ElementTree.ParseError:
        return None
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] == tag and element.text:
            return element.text
    return None


def _complete_multipart_body(parts: list[tuple[int, str]]) -> bytes:
    pieces = ["<CompleteMultipartUpload>"]
    for number, etag in parts:
        clean = etag.strip('"')
        pieces.append(
            f"<Part><PartNumber>{number}</PartNumber><ETag>&quot;{clean}&quot;</ETag></Part>"
        )
    pieces.append("</CompleteMultipartUpload>")
    return "".join(pieces).encode("utf-8")


# --------------------------------------------------------------------------- #
# artifact transport
# --------------------------------------------------------------------------- #

ARTIFACT_FILES = (MANIFEST_NAME, TENSORS_NAME, LINEAGE_NAME)


class ObjectStoreArtifactTransport:
    """Move content-addressed artifacts (with parents) through an S3 bucket."""

    def __init__(
        self,
        client: S3Client,
        *,
        prefix: str | None = None,
    ) -> None:
        self.client = client
        self.prefix = (prefix if prefix is not None else client.config.prefix).strip("/")

    def artifact_key(self, artifact_id: str, filename: str) -> str:
        return f"{self.prefix}/{artifact_id[:2]}/{artifact_id}/{filename}"

    def tokenizer_key(self, digest: str) -> str:
        return f"{self.prefix}/tokenizers/{digest[:2]}/{digest}/tokenizer.json"

    @staticmethod
    def _tokenizer_digest(manifest: PackageManifest) -> str | None:
        metadata = manifest.metadata if isinstance(manifest.metadata, Mapping) else {}
        spec = metadata.get("tokenizer")
        if not isinstance(spec, Mapping) or spec.get("kind") != TOKENIZER_KIND:
            return None
        digest = str(spec.get("sha256") or "")
        return digest or None

    # -- push ----------------------------------------------------------- #

    def push_artifact(
        self,
        store: ArtifactStore,
        artifact_id: str,
        *,
        include_parents: bool = True,
    ) -> dict[str, Any]:
        order = _artifact_order(store, artifact_id, include_parents=include_parents)
        uploaded: list[dict[str, Any]] = []
        skipped: list[str] = []
        total_bytes = 0
        started = time.perf_counter()
        for current in order:
            directory = store.artifact_dir(current)
            if not (directory / MANIFEST_NAME).exists():
                raise ArtifactNotFound(f"artifact {current} is not present in {store.root}")
            manifest = store.manifest(current)
            tokenizer_digest = self._tokenizer_digest(manifest)
            if tokenizer_digest:
                tokenizer_file = tokenizer_path(store.root.parent, tokenizer_digest)
                if not tokenizer_file.is_file():
                    raise ArtifactIntegrityError(
                        f"artifact {current} declares tokenizer {tokenizer_digest} but the "
                        "workspace does not contain it",
                        artifact_id=current,
                    )
                key = self.tokenizer_key(tokenizer_digest)
                digest = sha256_file(tokenizer_file)
                remote = self.client.head(key)
                if remote.get("exists") and remote.get("sha256") == digest:
                    skipped.append(key)
                else:
                    result = self.client.put_file(
                        tokenizer_file,
                        key,
                        metadata={"sha256": digest, "file": "tokenizer.json"},
                        content_type="application/json",
                    )
                    total_bytes += int(result["bytes"])
                    uploaded.append(
                        {
                            "artifact_id": current,
                            "file": "tokenizer.json",
                            "key": key,
                            "bytes": result["bytes"],
                            "sha256": result["sha256"],
                            "multipart": result["multipart"],
                            "parts": result["parts"],
                        }
                    )
            for filename in ARTIFACT_FILES:
                source = directory / filename
                if not source.exists():
                    continue
                key = self.artifact_key(current, filename)
                digest = sha256_file(source)
                remote = self.client.head(key)
                if remote.get("exists") and remote.get("sha256") == digest:
                    skipped.append(key)
                    continue
                content_type = (
                    "application/json" if filename.endswith(".json") else "application/octet-stream"
                )
                result = self.client.put_file(
                    source,
                    key,
                    metadata={"sha256": digest, "artifact-id": current, "file": filename},
                    content_type=content_type,
                )
                total_bytes += int(result["bytes"])
                uploaded.append(
                    {
                        "artifact_id": current,
                        "file": filename,
                        "key": key,
                        "bytes": result["bytes"],
                        "sha256": result["sha256"],
                        "multipart": result["multipart"],
                        "parts": result["parts"],
                    }
                )
        return {
            "artifact_id": artifact_id,
            "artifacts": order,
            "uploaded": uploaded,
            "skipped": skipped,
            "bytes_uploaded": total_bytes,
            "seconds": round(time.perf_counter() - started, 3),
            "bucket": self.client.config.bucket,
            "prefix": self.prefix,
        }

    # -- fetch ---------------------------------------------------------- #

    def fetch_artifact(
        self,
        store: ArtifactStore,
        artifact_id: str,
        *,
        include_parents: bool = True,
    ) -> dict[str, Any]:
        fetched: list[str] = []
        present: list[str] = []
        total_bytes = 0
        started = time.perf_counter()
        visiting: set[str] = set()

        def download(current: str) -> None:
            nonlocal total_bytes
            if current in visiting:
                raise ArtifactIntegrityError(
                    f"artifact lineage for {artifact_id} contains a cycle at {current}"
                )
            visiting.add(current)
            try:
                if store.exists(current):
                    present.append(current)
                    return
                manifest = self._remote_manifest(current)
                if include_parents:
                    for parent in manifest.lineage.parents:
                        download(parent)
                total_bytes += self._download_one(store, current, manifest)
                fetched.append(current)
            finally:
                visiting.discard(current)

        if not include_parents:
            if store.exists(artifact_id):
                present.append(artifact_id)
            else:
                manifest = self._remote_manifest(artifact_id)
                total_bytes += self._download_one(store, artifact_id, manifest)
                fetched.append(artifact_id)
        else:
            download(artifact_id)

        if include_parents:
            report = store.verify(artifact_id)
            verified = bool(report["ok"])
            tensor_digest = report["tensor_digest"]
            problems = report["problems"]
        else:
            # The lineage was intentionally not transported, so only the
            # self-contained evidence can be checked here.
            verified, tensor_digest, problems = self._self_verified(store, artifact_id)
        if not verified:
            raise ArtifactIntegrityError(
                f"artifact {artifact_id} failed verification after transport",
                problems=problems[:50],
            )
        return {
            "artifact_id": artifact_id,
            "fetched": fetched,
            "already_present": present,
            "bytes_downloaded": total_bytes,
            "seconds": round(time.perf_counter() - started, 3),
            "verified": verified,
            "tensor_digest": tensor_digest,
            "bucket": self.client.config.bucket,
            "prefix": self.prefix,
            "fetched_at": utc_now_iso(),
        }

    @staticmethod
    def _self_verified(
        store: ArtifactStore, artifact_id: str
    ) -> tuple[bool, str | None, list[str]]:
        manifest = store.manifest(artifact_id)
        problems: list[str] = []
        ok, reason = manifest.verify_checksum()
        if not ok:
            problems.append(reason or "manifest checksum failed")
        arrays, _ = read_safetensors(store.artifact_dir(artifact_id) / TENSORS_NAME)
        digest = safetensors_tensor_digest(arrays)
        declared = manifest.summary.get("fingerprint")
        if declared and digest != declared:
            problems.append("tensor payload digest does not match manifest")
        return not problems, digest, problems

    def _remote_manifest(self, artifact_id: str) -> PackageManifest:
        key = self.artifact_key(artifact_id, MANIFEST_NAME)
        try:
            raw = self.client.get_bytes(key)
        except ObjectStoreError as exc:
            if exc.details.get("status") == 404:
                raise ArtifactNotFound(
                    f"artifact {artifact_id} is not present in the object store", key=key
                ) from exc
            raise
        try:
            manifest = PackageManifest.from_dict(json.loads(raw.decode("utf-8")))
        except (ValueError, KeyError) as exc:
            raise ArtifactIntegrityError(
                f"remote manifest for {artifact_id} is not valid JSON", key=key
            ) from exc
        ok, reason = manifest.verify_checksum()
        if not ok:
            raise ArtifactIntegrityError(
                f"remote manifest for {artifact_id} failed verification: {reason}", key=key
            )
        if manifest.artifact_id != artifact_id:
            raise ArtifactIntegrityError(
                f"remote manifest declares artifact {manifest.artifact_id}, expected {artifact_id}",
                key=key,
            )
        return manifest

    def _download_one(
        self,
        store: ArtifactStore,
        artifact_id: str,
        manifest: PackageManifest,
    ) -> int:
        target = store.artifact_dir(artifact_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(dir=str(target.parent), prefix=f".fetch-{artifact_id[:8]}-")
        )
        received = 0
        try:
            for filename in ARTIFACT_FILES:
                key = self.artifact_key(artifact_id, filename)
                if filename == LINEAGE_NAME:
                    remote = self.client.head(key)
                    if not remote.get("exists"):
                        continue
                if filename == TENSORS_NAME and not manifest.summary.get("fingerprint"):
                    raise ArtifactIntegrityError(
                        f"manifest for {artifact_id} has no tensor fingerprint", key=key
                    )
                # The upload recorded the file SHA-256 as object metadata, which
                # ``get_file`` verifies. The tensor fingerprint is checked below
                # against the payload itself, so the manifest cannot vouch for
                # bytes it does not describe.
                result = self.client.get_file(key, staging / filename)
                received += int(result["bytes"])
            tokenizer_digest = self._tokenizer_digest(manifest)
            if tokenizer_digest:
                key = self.tokenizer_key(tokenizer_digest)
                destination = tokenizer_path(store.root.parent, tokenizer_digest)
                try:
                    tokenizer_result = self.client.get_file(
                        key, destination, expected_sha256=tokenizer_digest
                    )
                except ObjectStoreError as exc:
                    if exc.details.get("status") == 404:
                        raise ArtifactIntegrityError(
                            f"artifact {artifact_id} declares tokenizer {tokenizer_digest} "
                            "but the object store does not contain it",
                            key=key,
                        ) from exc
                    raise
                received += int(tokenizer_result["bytes"])
            arrays, _ = read_safetensors(staging / TENSORS_NAME)
            actual = safetensors_tensor_digest(arrays)
            declared = manifest.summary.get("fingerprint")
            if declared and actual != declared:
                raise ArtifactIntegrityError(
                    f"fetched payload for {artifact_id} does not match the manifest fingerprint",
                    expected=declared,
                    actual=actual,
                )
            if target.exists():
                shutil.rmtree(staging, ignore_errors=True)
                return 0
            os.replace(staging, target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return received


def _artifact_order(
    store: ArtifactStore, artifact_id: str, *, include_parents: bool
) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()

    def visit(current: str) -> None:
        if current in seen:
            return
        seen.add(current)
        manifest = store.manifest(current)
        if include_parents:
            for parent in manifest.lineage.parents:
                visit(parent)
        order.append(current)

    visit(artifact_id)
    return order


def client_from_env(*, prefix: str | None = None) -> S3Client:
    return S3Client(S3Config.from_env(prefix=prefix))
