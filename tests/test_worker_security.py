"""Advanced worker security: envelopes, key rotation, replay durability, credentials."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from uzen.errors import WorkerAuthError
from uzen.infrastructure.security.tokens import (
    ENVELOPE_SCHEMA,
    SigningKeyRing,
    SqliteReplayGuard,
    WorkerCredentialStore,
    build_envelope,
    generate_worker_secret,
    hash_secret,
    sign,
    verify_envelope,
    verify_secret,
    verify_signature,
)


def test_sign_and_verify_are_canonical_and_constant_time_compatible():
    secret = generate_worker_secret()
    payload = {"worker_id": "w1", "body": {"b": 2, "a": 1}}
    signature = sign(secret, payload)
    assert len(signature) == 64
    assert verify_signature(secret, {"body": {"a": 1, "b": 2}, "worker_id": "w1"}, signature)
    assert not verify_signature(secret, payload, "0" * 64)
    assert not verify_signature(secret, payload, "short")
    assert verify_secret(hash_secret(secret), secret)
    assert not verify_secret(hash_secret(secret), secret + "x")


def test_v2_envelope_round_trip_and_legacy_envelope_are_accepted():
    secret = generate_worker_secret()
    envelope = build_envelope("worker-1", {"job": "abc"}, secret=secret, ttl=30)
    assert envelope["schema"] == ENVELOPE_SCHEMA
    assert envelope["expires_at"] is not None
    assert verify_envelope(envelope, secret_resolver=lambda worker: secret) == {"job": "abc"}

    now = time.time()
    legacy_payload = {
        "worker_id": "worker-1",
        "timestamp": now,
        "nonce": "legacy-nonce",
        "body": {"job": "legacy"},
    }
    legacy = {**legacy_payload, "signature": sign(secret, legacy_payload)}
    assert verify_envelope(
        legacy, secret_resolver=lambda worker: secret, now=now
    ) == {"job": "legacy"}


def test_envelope_rejects_expiry_tampering_and_replay():
    secret = generate_worker_secret()
    now = time.time()
    expired = build_envelope("w", {}, secret=secret, ttl=1, timestamp=now - 10)
    with pytest.raises(WorkerAuthError, match="expired"):
        verify_envelope(expired, secret_resolver=lambda worker: secret, now=now)

    envelope = build_envelope("w", {"value": 1}, secret=secret)
    envelope["body"] = {"value": 2}
    with pytest.raises(WorkerAuthError, match="invalid worker signature"):
        verify_envelope(envelope, secret_resolver=lambda worker: secret)

    current = build_envelope("w", {}, secret=secret)
    guard = SqliteReplayGuard(":memory:")
    verify_envelope(current, secret_resolver=lambda worker: secret, replay_guard=guard)
    with pytest.raises(WorkerAuthError, match="replayed nonce"):
        verify_envelope(current, secret_resolver=lambda worker: secret, replay_guard=guard)
    guard.close()


def test_signing_key_ring_rotation_keeps_old_keys_verifiable():
    ring = SigningKeyRing()
    payload = {"worker_id": "w", "body": {"x": 1}}
    original = ring.primary
    old_key_id, old_signature = ring.sign(payload)

    rotated = ring.rotate()
    assert rotated.key_id != original.key_id
    assert ring.primary.key_id == rotated.key_id
    assert ring.verify(payload, old_signature, key_id=old_key_id)
    assert ring.verify(payload, old_signature)
    assert not ring.verify(payload, old_signature, key_id="missing")

    new_key_id, new_signature = ring.sign(payload)
    assert new_key_id == rotated.key_id
    assert new_signature != old_signature
    assert ring.retire(old_key_id) is True
    assert not ring.verify(payload, old_signature, key_id=old_key_id)
    assert ring.retire(rotated.key_id) is False  # never retire the primary
    assert "secret" not in str(ring.describe()).lower()


def test_keyring_envelope_carries_the_signing_key_id():
    ring = SigningKeyRing()
    envelope = build_envelope("w", {"ok": True}, keyring=ring)
    assert envelope["key_id"] == ring.primary.key_id
    assert verify_envelope(envelope, keyring=ring) == {"ok": True}

    ring.rotate()
    assert verify_envelope(envelope, keyring=ring) == {"ok": True}
    with pytest.raises(WorkerAuthError, match="invalid worker signature"):
        verify_envelope(envelope, keyring=SigningKeyRing())


def test_sqlite_replay_guard_survives_process_restart_and_prunes(tmp_path: Path):
    path = tmp_path / "replay.sqlite3"
    now = 1_000.0
    with SqliteReplayGuard(path, window_seconds=10) as first:
        assert first.check_and_store("nonce-a", now=now) is True
        assert first.check_and_store("nonce-a", now=now) is False
    with SqliteReplayGuard(path, window_seconds=10) as second:
        assert second.check_and_store("nonce-a", now=now) is False
        assert second.check_and_store("nonce-b", now=now + 11) is True
        # nonce-a is older than the window and was pruned, so it is accepted again.
        assert second.check_and_store("nonce-a", now=now + 12) is True
        assert second.prune(now=now + 30) >= 1


def test_worker_credentials_support_scopes_expiry_rotation_and_revocation():
    store = WorkerCredentialStore()
    secret = store.issue("worker-1", scopes=("jobs:read",), ttl=10)
    now = time.time()
    assert store.authenticate("worker-1", secret, required_scope="jobs:read", now=now)
    assert not store.authenticate("worker-1", secret, required_scope="jobs:write", now=now)
    assert not store.authenticate("worker-1", secret, now=now + 11)
    assert not store.authenticate("worker-1", "wrong", now=now)

    rotated = store.rotate("worker-1")
    assert rotated != secret
    assert not store.authenticate("worker-1", secret, now=time.time())
    assert store.authenticate("worker-1", rotated, now=time.time())
    assert store.revoke("worker-1") is True
    assert not store.authenticate("worker-1", rotated)
    assert store.revoke("worker-1") is False


def test_worker_credential_describe_never_exposes_hashes_or_secrets():
    store = WorkerCredentialStore()
    secret = store.issue("worker-1", scopes=("jobs:read",))
    described = store.describe()
    assert described == [
        {
            "worker_id": "worker-1",
            "scopes": ["jobs:read"],
            "issued_at": described[0]["issued_at"],
            "expires_at": None,
            "generation": 1,
        }
    ]
    rendered = str(described)
    assert secret not in rendered
    assert hash_secret(secret) not in rendered
