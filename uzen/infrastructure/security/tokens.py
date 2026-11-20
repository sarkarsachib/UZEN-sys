"""Worker authentication and request signing.

Workers authenticate with a shared secret that is *never stored in plaintext* on the
control plane (only its SHA-256 hash is persisted). Every control-plane request is
HMAC-signed over canonical JSON of the envelope, with a timestamp and a nonce:

    signature = HMAC-SHA256(secret, canonical_json({...envelope...}))

The layer is deliberately dependency-free, but it is not a toy:

* **Versioned envelopes.**  Envelopes carry a schema identifier, an optional
  ``key_id`` and an optional ``expires_at``.  Legacy v1 envelopes (no schema) are
  still accepted, so an older worker is not broken by an upgrade.
* **Signing-key rotation.**  :class:`SigningKeyRing` keeps a primary key and any
  number of previous keys.  New envelopes are signed with the primary; both the
  primary and retired keys can verify, and retiring a key removes it immediately.
  Key material is never included in :meth:`SigningKeyRing.describe`.
* **Replay protection that survives a restart.**  :class:`ReplayGuard` is the
  in-process cache; :class:`SqliteReplayGuard` is the durable variant, using an
  ``INSERT OR IGNORE`` under ``BEGIN IMMEDIATE`` so two control-plane processes
  sharing a database cannot both accept the same nonce.
* **Scoped credentials.**  :class:`WorkerCredentialStore` stores hashes, optional
  scopes and an expiry, supports rotation and revocation, and authenticates with
  constant-time comparison.

Verification always enforces a signature match (constant time), timestamp
freshness (a bounded replay window) and nonce uniqueness when a replay guard is
supplied.  This remains transport security for worker↔control-plane traffic, not a
replacement for TLS, which should be terminated in front of the service.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from uzen.errors import WorkerAuthError
from uzen.util.canonical import canonical_bytes, sha256_text

DEFAULT_WINDOW_SECONDS = 300
DEFAULT_KEY_ID = "primary"
ENVELOPE_SCHEMA = "uzen.worker.envelope/2"
LEGACY_ENVELOPE_SCHEMAS = frozenset({"uzen.worker.envelope/1"})
_MAX_NONCE_LENGTH = 256


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #


def generate_worker_secret() -> str:
    """Return a high-entropy URL-safe worker secret."""
    return secrets.token_urlsafe(48)


def hash_secret(secret: str) -> str:
    """Hash a secret for storage; the plaintext is never persisted by UZEN."""
    return sha256_text(secret)


def verify_secret(expected_hash: str, presented: str) -> bool:
    """Constant-time comparison of a presented secret against a stored hash."""
    if not expected_hash or not isinstance(presented, str):
        return False
    return hmac.compare_digest(hash_secret(presented), expected_hash)


def sign(secret: str, payload: Mapping[str, Any]) -> str:
    """HMAC-SHA256 over the canonical form of *payload*."""
    return hmac.new(
        secret.encode("utf-8"), canonical_bytes(dict(payload)), hashlib.sha256
    ).hexdigest()


def verify_signature(secret: str, payload: Mapping[str, Any], signature: str) -> bool:
    """Verify a signature without leaking timing information."""
    if not isinstance(signature, str) or len(signature) != 64:
        return False
    return hmac.compare_digest(sign(secret, payload), signature)


# --------------------------------------------------------------------------- #
# replay guards
# --------------------------------------------------------------------------- #


class ReplayGuard:
    """In-memory nonce cache with expiry (single-process control plane)."""

    def __init__(self, window_seconds: int = DEFAULT_WINDOW_SECONDS) -> None:
        self.window = int(window_seconds)
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()

    def check_and_store(self, nonce: str, *, now: float | None = None) -> bool:
        """Return True exactly once for a nonce inside the replay window."""
        if not isinstance(nonce, str) or not nonce or len(nonce) > _MAX_NONCE_LENGTH:
            return False
        now = now if now is not None else time.time()
        with self._lock:
            expired = [k for k, ts in self._seen.items() if now - ts > self.window]
            for key in expired:
                self._seen.pop(key, None)
            if nonce in self._seen:
                return False
            self._seen[nonce] = now
            return True

    def __len__(self) -> int:
        with self._lock:
            return len(self._seen)


class SqliteReplayGuard:
    """Durable replay guard for one or more control-plane processes.

    A shared ``BEGIN IMMEDIATE`` transaction serializes concurrent inserts, so the
    nonce check and the nonce write cannot interleave between processes.  Entries
    older than the replay window are pruned on every call, and the table is capped
    at ``max_entries`` so a burst of traffic cannot grow it without bound.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
        max_entries: int = 100_000,
    ) -> None:
        if int(max_entries) <= 0:
            raise ValueError("max_entries must be positive")
        self._dsn = str(path)
        self.path = Path(path)
        self.window = int(window_seconds)
        self.max_entries = int(max_entries)
        if self._dsn != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._dsn, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS replay_nonces ("
            "nonce TEXT PRIMARY KEY, seen_at REAL NOT NULL)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_replay_nonces_seen ON replay_nonces(seen_at)"
        )
        self._conn.commit()

    def check_and_store(self, nonce: str, *, now: float | None = None) -> bool:
        if not isinstance(nonce, str) or not nonce or len(nonce) > _MAX_NONCE_LENGTH:
            return False
        now = float(now if now is not None else time.time())
        cutoff = now - self.window
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                self._conn.execute("DELETE FROM replay_nonces WHERE seen_at <= ?", (cutoff,))
                row = self._conn.execute("SELECT COUNT(*) FROM replay_nonces").fetchone()
                count = int(row[0]) if row else 0
                if count >= self.max_entries:
                    excess = count - self.max_entries + 1
                    self._conn.execute(
                        "DELETE FROM replay_nonces WHERE nonce IN ("
                        "SELECT nonce FROM replay_nonces ORDER BY seen_at ASC LIMIT ?)",
                        (excess,),
                    )
                cursor = self._conn.execute(
                    "INSERT OR IGNORE INTO replay_nonces(nonce, seen_at) VALUES (?, ?)",
                    (nonce, now),
                )
                self._conn.commit()
                return cursor.rowcount == 1
            except Exception:
                self._conn.rollback()
                raise

    def prune(self, *, now: float | None = None) -> int:
        """Delete expired nonces and return how many rows were removed."""
        now = float(now if now is not None else time.time())
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM replay_nonces WHERE seen_at <= ?", (now - self.window,)
            )
            self._conn.commit()
            return int(cursor.rowcount)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "SqliteReplayGuard":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# signing keys
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SigningKey:
    """One named HMAC key.  The secret is excluded from repr output."""

    key_id: str
    secret: str = field(repr=False)
    created_at: float = field(default_factory=time.time)

    def describe(self) -> dict[str, Any]:
        return {
            "key_id": self.key_id,
            "created_at": self.created_at,
            "fingerprint": hash_secret(self.secret)[:16],
        }


class SigningKeyRing:
    """Primary signing key plus retired keys retained for verification."""

    def __init__(
        self,
        primary: SigningKey | None = None,
        *,
        previous: Sequence[SigningKey] = (),
    ) -> None:
        primary = primary or SigningKey(DEFAULT_KEY_ID, generate_worker_secret())
        self._lock = threading.RLock()
        self._keys: dict[str, SigningKey] = {primary.key_id: primary}
        for key in previous:
            if key.key_id in self._keys:
                raise ValueError(f"duplicate signing key id '{key.key_id}'")
            self._keys[key.key_id] = key
        self._primary_id = primary.key_id

    @property
    def primary(self) -> SigningKey:
        with self._lock:
            return self._keys[self._primary_id]

    def add(self, key: SigningKey) -> SigningKey:
        """Register a verification key without making it primary."""
        with self._lock:
            if key.key_id in self._keys:
                raise ValueError(f"duplicate signing key id '{key.key_id}'")
            self._keys[key.key_id] = key
            return key

    def rotate(self, *, secret: str | None = None, key_id: str | None = None) -> SigningKey:
        """Make a fresh key primary; the previous key remains verifiable."""
        with self._lock:
            if key_id is None:
                key_id = f"key-{int(time.time() * 1000)}"
            suffix = 1
            candidate = key_id
            while candidate in self._keys:
                suffix += 1
                candidate = f"{key_id}-{suffix}"
            key = SigningKey(candidate, secret or generate_worker_secret())
            self._keys[key.key_id] = key
            self._primary_id = key.key_id
            return key

    def retire(self, key_id: str) -> bool:
        """Remove a non-primary key; returns False if unknown or primary."""
        with self._lock:
            if key_id == self._primary_id or key_id not in self._keys:
                return False
            self._keys.pop(key_id, None)
            return True

    def sign(self, payload: Mapping[str, Any]) -> tuple[str, str]:
        """Return ``(key_id, signature)`` using the current primary key."""
        key = self.primary
        return key.key_id, sign(key.secret, payload)

    def verify(
        self,
        payload: Mapping[str, Any],
        signature: str,
        *,
        key_id: str | None = None,
    ) -> bool:
        """Verify against one named key, or against every retained key."""
        with self._lock:
            if key_id is not None:
                key = self._keys.get(key_id)
                return bool(key and verify_signature(key.secret, payload, signature))
            candidates = list(self._keys.values())
        return any(verify_signature(key.secret, payload, signature) for key in candidates)

    def describe(self) -> dict[str, Any]:
        with self._lock:
            return {
                "primary": self._primary_id,
                "keys": [key.describe() for key in self._keys.values()],
            }


# --------------------------------------------------------------------------- #
# envelopes
# --------------------------------------------------------------------------- #


@dataclass
class SignedEnvelope:
    worker_id: str
    timestamp: float
    nonce: str
    body: dict[str, Any] = field(default_factory=dict)
    schema: str = ENVELOPE_SCHEMA
    key_id: str | None = None
    expires_at: float | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "worker_id": self.worker_id,
            "timestamp": self.timestamp,
            "nonce": self.nonce,
            "body": self.body,
        }
        if self.key_id is not None:
            payload["key_id"] = self.key_id
        if self.expires_at is not None:
            payload["expires_at"] = self.expires_at
        return payload


def build_envelope(
    worker_id: str,
    body: Mapping[str, Any],
    *,
    secret: str | None = None,
    keyring: SigningKeyRing | None = None,
    key_id: str | None = None,
    ttl: float | None = None,
    timestamp: float | None = None,
) -> dict[str, Any]:
    """Build and sign a worker envelope.

    Provide exactly one of ``secret=`` (single-key worker) or ``keyring=``
    (rotating control-plane key set).  ``ttl`` adds an explicit expiry that is
    covered by the signature.
    """
    if (secret is None) == (keyring is None):
        raise ValueError("provide exactly one of secret= or keyring=")
    now = float(timestamp if timestamp is not None else time.time())
    envelope = SignedEnvelope(
        worker_id=worker_id,
        timestamp=now,
        nonce=secrets.token_hex(16),
        body=dict(body),
        expires_at=(now + float(ttl)) if ttl is not None else None,
    ).to_payload()
    if keyring is not None:
        # The key id is part of the signed payload, so it must be present before
        # the signature is computed (otherwise verification would hash a different
        # envelope than the signer did).
        envelope["key_id"] = keyring.primary.key_id
        selected_id, signature = keyring.sign(envelope)
        envelope["key_id"] = selected_id
    else:
        assert secret is not None
        if key_id is not None:
            envelope["key_id"] = key_id
        signature = sign(secret, envelope)
    envelope["signature"] = signature
    return envelope


def verify_envelope(
    envelope: Mapping[str, Any],
    *,
    secret_resolver=None,
    replay_guard: ReplayGuard | SqliteReplayGuard | None = None,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    now: float | None = None,
    keyring: SigningKeyRing | None = None,
) -> dict[str, Any]:
    """Verify a signed envelope and return its body.

    Raises :class:`WorkerAuthError` for malformed, expired, replayed or
    incorrectly signed envelopes.  ``secret_resolver(worker_id)`` is used when no
    keyring is supplied, preserving the original single-key API.
    """
    now = float(now if now is not None else time.time())
    for key in ("worker_id", "timestamp", "nonce", "signature"):
        if key not in envelope:
            raise WorkerAuthError(f"signed envelope is missing '{key}'")
    schema = str(envelope.get("schema") or ENVELOPE_SCHEMA)
    if schema != ENVELOPE_SCHEMA and schema not in LEGACY_ENVELOPE_SCHEMAS:
        raise WorkerAuthError(f"unsupported envelope schema '{schema}'")

    worker_id = str(envelope["worker_id"])
    signature = str(envelope["signature"])
    signed = {k: v for k, v in envelope.items() if k != "signature"}

    key_id = str(envelope["key_id"]) if envelope.get("key_id") else None
    if keyring is not None:
        if not keyring.verify(signed, signature, key_id=key_id):
            raise WorkerAuthError("invalid worker signature", worker_id=worker_id)
    else:
        if secret_resolver is None:
            raise WorkerAuthError("no worker secret resolver configured", worker_id=worker_id)
        secret = secret_resolver(worker_id)
        if not secret:
            raise WorkerAuthError(f"unknown worker '{worker_id}'", worker_id=worker_id)
        if not verify_signature(secret, signed, signature):
            raise WorkerAuthError("invalid worker signature", worker_id=worker_id)

    try:
        issued = float(envelope["timestamp"])
    except (TypeError, ValueError) as exc:
        raise WorkerAuthError("invalid envelope timestamp") from exc

    expires_at = envelope.get("expires_at")
    if expires_at is not None:
        try:
            expires = float(expires_at)
        except (TypeError, ValueError) as exc:
            raise WorkerAuthError("invalid envelope expiry") from exc
        if now > expires:
            raise WorkerAuthError("signed envelope has expired", worker_id=worker_id)

    if abs(now - issued) > window_seconds:
        raise WorkerAuthError(
            f"envelope timestamp outside the {window_seconds}s replay window",
            worker_id=worker_id,
        )

    nonce = envelope["nonce"]
    if not isinstance(nonce, str) or not nonce or len(nonce) > _MAX_NONCE_LENGTH:
        raise WorkerAuthError("invalid envelope nonce", worker_id=worker_id)
    if replay_guard is not None and not replay_guard.check_and_store(nonce, now=now):
        raise WorkerAuthError("replayed nonce", worker_id=worker_id)
    return dict(envelope.get("body") or {})


# --------------------------------------------------------------------------- #
# credential store
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WorkerCredential:
    """A hashed worker credential with scopes and an optional expiry."""

    worker_id: str
    token_hash: str = field(repr=False)
    scopes: frozenset[str] = frozenset({"*"})
    issued_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    generation: int = 1

    def expired(self, *, now: float | None = None) -> bool:
        if self.expires_at is None:
            return False
        return float(now if now is not None else time.time()) > self.expires_at

    def permits(self, scope: str | None) -> bool:
        if scope is None or "*" in self.scopes:
            return True
        return scope in self.scopes

    def describe(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "scopes": sorted(self.scopes),
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "generation": self.generation,
        }


class WorkerCredentialStore:
    """Hash-only credential store with scopes, rotation and revocation."""

    def __init__(self) -> None:
        self._credentials: dict[str, WorkerCredential] = {}
        self._lock = threading.RLock()

    def _issue_locked(
        self,
        worker_id: str,
        secret: str,
        *,
        scopes: Sequence[str],
        ttl: float | None,
    ) -> WorkerCredential:
        now = time.time()
        previous = self._credentials.get(worker_id)
        credential = WorkerCredential(
            worker_id=worker_id,
            token_hash=hash_secret(secret),
            scopes=frozenset(scopes),
            issued_at=now,
            expires_at=(now + float(ttl)) if ttl is not None else None,
            generation=(previous.generation + 1) if previous else 1,
        )
        self._credentials[worker_id] = credential
        return credential

    def register(
        self,
        worker_id: str,
        secret: str,
        *,
        scopes: Sequence[str] = ("*",),
        ttl: float | None = None,
    ) -> str:
        """Store a caller-supplied secret and return its hash (legacy API)."""
        if not worker_id or not secret:
            raise ValueError("worker_id and secret must be non-empty")
        with self._lock:
            credential = self._issue_locked(worker_id, secret, scopes=scopes, ttl=ttl)
            return credential.token_hash

    def issue(
        self,
        worker_id: str,
        *,
        scopes: Sequence[str] = ("*",),
        ttl: float | None = None,
    ) -> str:
        """Generate, store and return a fresh secret."""
        if not worker_id:
            raise ValueError("worker_id must be non-empty")
        secret = generate_worker_secret()
        with self._lock:
            self._issue_locked(worker_id, secret, scopes=scopes, ttl=ttl)
        return secret

    def rotate(self, worker_id: str) -> str:
        """Replace an existing credential, invalidating the previous secret."""
        if not worker_id:
            raise ValueError("worker_id must be non-empty")
        secret = generate_worker_secret()
        with self._lock:
            previous = self._credentials.get(worker_id)
            scopes = previous.scopes if previous else frozenset({"*"})
            ttl = (
                max(0.0, previous.expires_at - previous.issued_at)
                if previous and previous.expires_at is not None
                else None
            )
            self._issue_locked(worker_id, secret, scopes=tuple(scopes), ttl=ttl)
        return secret

    def revoke(self, worker_id: str) -> bool:
        with self._lock:
            return self._credentials.pop(worker_id, None) is not None

    def authenticate(
        self,
        worker_id: str,
        secret: str,
        *,
        required_scope: str | None = None,
        now: float | None = None,
    ) -> bool:
        with self._lock:
            credential = self._credentials.get(worker_id)
        if credential is None or credential.expired(now=now):
            return False
        if not verify_secret(credential.token_hash, secret):
            return False
        return credential.permits(required_scope)

    def credential(self, worker_id: str) -> WorkerCredential | None:
        with self._lock:
            return self._credentials.get(worker_id)

    def token_hash(self, worker_id: str) -> str | None:
        with self._lock:
            credential = self._credentials.get(worker_id)
            return credential.token_hash if credential is not None else None

    def describe(self) -> list[dict[str, Any]]:
        with self._lock:
            return [credential.describe() for credential in self._credentials.values()]

    def __contains__(self, worker_id: object) -> bool:
        with self._lock:
            return worker_id in self._credentials
