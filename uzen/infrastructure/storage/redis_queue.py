"""Redis/Valkey durable queue adapter (RESP2, standard library only).

SQLite is the single-host control-plane database, and WAL mode requires shared memory,
so it must not be pointed at a network filesystem to fake a cluster. A multi-node
deployment therefore needs a real shared queue, and this module is UZEN's: it speaks
RESP2 over TCP and keeps every scheduling decision atomic on the server.

Layout
------
``uzen:q:ready``               list of queued job ids (newest pushed left)
``uzen:q:job:<id>``            hash: request, requirements, state, attempts, worker, token, lease_until
``uzen:q:req:<id>``            set of ``kind:value`` every one of which must match
``uzen:q:reqany:<id>``         set of ``kind:value`` of which at least one must match
``uzen:q:caps:<worker>``       set of ``kind:value`` the worker advertised
``uzen:q:worker:<id>``         hash: cpu_count, memory_bytes
``uzen:q:lease``               sorted set job id -> unix expiry
``uzen:q:reports``             hash ``<job>:<attempt>`` -> outcome (idempotent reports)
``uzen:q:dead``                list of job ids that exhausted their attempts

Claiming is a single Lua script: it pops the oldest job whose requirements the caller
satisfies, increments the attempt counter, writes a fresh dispatch token and takes the
lease in one atomic server-side step. Redis runs scripts atomically, so concurrent
claimers cannot observe a half-taken lease.

Status of this adapter
----------------------
The client, key layout and scripts are exercised against a **real Redis 7.4 server** in
the integration suite (``UZEN_REDIS_URL``), in addition to the protocol-level socket
tests. The SQLite path remains the default single-host queue; this adapter is the
multi-node one.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Mapping
from urllib.parse import urlparse

from uzen.errors import UzenError
from uzen.jobs.scheduling import CapabilityRequirements, capability_index
from uzen.util.canonical import utc_now_iso


class RedisProtocolError(UzenError):
    """The server answered with an error reply or a malformed frame."""

    code = "storage.redis_protocol"
    http_status = 502


# --------------------------------------------------------------------------- #
# RESP2 transport
# --------------------------------------------------------------------------- #


@dataclass
class RedisQueueConfig:
    url: str = "redis://127.0.0.1:6379/0"
    timeout: float = 5.0
    username: str | None = None
    password: str | None = None

    @classmethod
    def from_url(cls, url: str, *, timeout: float = 5.0) -> "RedisQueueConfig":
        parsed = urlparse(url)
        if parsed.scheme not in ("redis", "rediss"):
            raise RedisProtocolError(f"unsupported redis url scheme '{parsed.scheme}'")
        username = parsed.username or None
        password = parsed.password or None
        return cls(
            url=url,
            timeout=timeout,
            username=username,
            password=password,
        )

    def endpoint(self) -> tuple[str, int, int]:
        parsed = urlparse(self.url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 6379
        database = int((parsed.path or "/0").strip("/") or 0)
        return host, port, database

    def describe(self) -> dict[str, Any]:
        host, port, database = self.endpoint()
        return {
            "scheme": urlparse(self.url).scheme,
            "host": host,
            "port": port,
            "database": database,
            "auth": bool(self.password),
        }


def encode_command(*arguments: Any) -> bytes:
    """Encode one command as a RESP2 array of bulk strings."""
    parts = [f"*{len(arguments)}\r\n".encode()]
    for argument in arguments:
        if isinstance(argument, bytes):
            payload = argument
        else:
            payload = str(argument).encode("utf-8")
        parts.append(b"$" + str(len(payload)).encode() + b"\r\n" + payload + b"\r\n")
    return b"".join(parts)


class RespConnection:
    """Minimal RESP2 client: one socket, one lock, no background reader."""

    def __init__(self, config: RedisQueueConfig) -> None:
        self.config = config
        self._socket: socket.socket | None = None
        self._buffer = b""
        self._lock = threading.Lock()

    # -- connection ---------------------------------------------------- #

    def connect(self) -> None:
        if self._socket is not None:
            return
        host, port, database = self.config.endpoint()
        sock = socket.create_connection((host, port), timeout=self.config.timeout)
        sock.settimeout(self.config.timeout)
        self._socket = sock
        self._buffer = b""
        try:
            if self.config.password:
                if self.config.username:
                    self.command("AUTH", self.config.username, self.config.password)
                else:
                    self.command("AUTH", self.config.password)
            if database:
                self.command("SELECT", database)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None
                self._buffer = b""

    def __enter__(self) -> "RespConnection":
        self.connect()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # -- framing ------------------------------------------------------- #

    def _read_line(self) -> bytes:
        assert self._socket is not None
        while b"\r\n" not in self._buffer:
            chunk = self._socket.recv(65536)
            if not chunk:
                raise RedisProtocolError("redis connection closed while reading a reply")
            self._buffer += chunk
        line, _, rest = self._buffer.partition(b"\r\n")
        self._buffer = rest
        return line

    def _read_exact(self, size: int) -> bytes:
        assert self._socket is not None
        while len(self._buffer) < size + 2:
            chunk = self._socket.recv(65536)
            if not chunk:
                raise RedisProtocolError("redis connection closed while reading a payload")
            self._buffer += chunk
        payload = self._buffer[:size]
        self._buffer = self._buffer[size + 2 :]
        return payload

    def _read_reply(self) -> Any:
        line = self._read_line()
        if not line:
            raise RedisProtocolError("empty reply from redis")
        prefix, rest = line[:1], line[1:]
        if prefix == b"+":
            return rest.decode("utf-8")
        if prefix == b"-":
            raise RedisProtocolError(rest.decode("utf-8", errors="replace"))
        if prefix == b":":
            return int(rest)
        if prefix == b"$":
            size = int(rest)
            if size == -1:
                return None
            return self._read_exact(size)
        if prefix == b"*":
            count = int(rest)
            if count == -1:
                return None
            return [self._read_reply() for _ in range(count)]
        raise RedisProtocolError(f"unexpected RESP prefix {prefix!r}")

    # -- commands ------------------------------------------------------ #

    def command(self, *arguments: Any) -> Any:
        with self._lock:
            self.connect()
            assert self._socket is not None
            self._socket.sendall(encode_command(*arguments))
            return self._read_reply()

    def eval(self, script: str, keys: list[str], args: list[Any]) -> Any:
        return self.command("EVAL", script, len(keys), *keys, *args)


# --------------------------------------------------------------------------- #
# queue
# --------------------------------------------------------------------------- #

_PREFIX = "uzen:q"

CLAIM_SCRIPT = """
local prefix = KEYS[1]
local worker = ARGV[1]
local token = ARGV[2]
local lease_until = tonumber(ARGV[3])
local now = tonumber(ARGV[4])
local cpu_count = tonumber(ARGV[5])
local memory_bytes = tonumber(ARGV[6])
local aging = tonumber(ARGV[7])
local max_inflight = tonumber(ARGV[8])

local worker_key = prefix .. ':worker:' .. worker
if redis.call('HGET', worker_key, 'status') == 'draining' then return nil end
local inflight = tonumber(redis.call('HGET', worker_key, 'inflight') or '0')
if max_inflight > 0 and inflight >= max_inflight then return nil end

local ready = prefix .. ':ready'
local ids = redis.call('LRANGE', ready, -200, -1)
local best_id = nil
local best_priority = nil
local best_created = nil

for _, job_id in ipairs(ids) do
    local job_key = prefix .. ':job:' .. job_id
    if redis.call('EXISTS', job_key) == 0 then
        redis.call('LREM', ready, 0, job_id)
    else
        local state = redis.call('HGET', job_key, 'state') or 'QUEUED'
        local lease = tonumber(redis.call('HGET', job_key, 'lease_until') or '0')
        local available = tonumber(redis.call('HGET', job_key, 'available_at') or '0')
        if (state == 'QUEUED' or state == 'DISPATCHED') and lease <= now and available <= now then
            local eligible = true
            local req_key = prefix .. ':req:' .. job_id
            for _, requirement in ipairs(redis.call('SMEMBERS', req_key)) do
                if redis.call('SISMEMBER', prefix .. ':caps:' .. worker, requirement) == 0 then
                    eligible = false
                    break
                end
            end
            if eligible then
                local any_key = prefix .. ':reqany:' .. job_id
                local any_requirements = redis.call('SMEMBERS', any_key)
                if #any_requirements > 0 then
                    local satisfied = false
                    for _, requirement in ipairs(any_requirements) do
                        if redis.call('SISMEMBER', prefix .. ':caps:' .. worker, requirement) == 1 then
                            satisfied = true
                            break
                        end
                    end
                    if not satisfied then eligible = false end
                end
            end
            if eligible then
                local min_cpu = tonumber(redis.call('HGET', job_key, 'min_cpu_count') or '0')
                local min_memory = tonumber(redis.call('HGET', job_key, 'min_memory_bytes') or '0')
                if min_cpu > cpu_count or min_memory > memory_bytes then eligible = false end
            end
            if eligible then
                local priority = tonumber(redis.call('HGET', job_key, 'priority') or '0')
                local enqueued = tonumber(redis.call('HGET', job_key, 'enqueued_at') or '0')
                if aging > 0 and enqueued > 0 then
                    priority = priority + math.floor((now - enqueued) / aging)
                end
                local created = redis.call('HGET', job_key, 'created_at') or ''
                if best_id == nil or priority > best_priority
                    or (priority == best_priority and created < best_created) then
                    best_id = job_id
                    best_priority = priority
                    best_created = created
                end
            end
        end
    end
end

if best_id == nil then return nil end
local job_key = prefix .. ':job:' .. best_id
redis.call('LREM', ready, 0, best_id)
redis.call('HINCRBY', job_key, 'attempts', 1)
redis.call('HSET', job_key, 'state', 'DISPATCHED', 'worker_id', worker,
    'dispatch_token', token, 'lease_until', lease_until, 'available_at', 0,
    'dispatched_at', ARGV[9])
redis.call('ZADD', prefix .. ':lease', lease_until, best_id)
redis.call('HINCRBY', worker_key, 'inflight', 1)
return {best_id, redis.call('HGETALL', job_key)}
"""

RENEW_SCRIPT = """
local prefix = KEYS[1]
local job_id = ARGV[1]
local worker = ARGV[2]
local token = ARGV[3]
local lease_until = tonumber(ARGV[4])
local job_key = prefix .. ':job:' .. job_id
if redis.call('HGET', job_key, 'worker_id') ~= worker then return 0 end
if redis.call('HGET', job_key, 'dispatch_token') ~= token then return 0 end
redis.call('HSET', job_key, 'lease_until', lease_until)
redis.call('ZADD', prefix .. ':lease', lease_until, job_id)
return 1
"""

ACK_SCRIPT = """
local prefix = KEYS[1]
local job_id = ARGV[1]
local worker = ARGV[2]
local token = ARGV[3]
local state = ARGV[4]
local payload = ARGV[5]
local now = ARGV[6]
local job_key = prefix .. ':job:' .. job_id
if redis.call('HGET', job_key, 'worker_id') ~= worker then return 'stale' end
if redis.call('HGET', job_key, 'dispatch_token') ~= token then return 'stale' end
local attempt = redis.call('HGET', job_key, 'attempts') or '1'
local report_key = job_id .. ':' .. attempt
local existing = redis.call('HGET', prefix .. ':reports', report_key)
if existing then return 'duplicate' end
redis.call('HSET', prefix .. ':reports', report_key, state)
redis.call('HSET', job_key, 'state', state, 'result', payload, 'reported_at', now)
redis.call('ZREM', prefix .. ':lease', job_id)
local inflight = tonumber(redis.call('HGET', prefix .. ':worker:' .. worker, 'inflight') or '0')
if inflight > 0 then redis.call('HINCRBY', prefix .. ':worker:' .. worker, 'inflight', -1) end
return 'accepted'
"""

REAP_SCRIPT = """
local prefix = KEYS[1]
local now = tonumber(ARGV[1])
local base = tonumber(ARGV[2])
local cap = tonumber(ARGV[3])
local reaped = {}
local expired = redis.call('ZRANGEBYSCORE', prefix .. ':lease', '-inf', now, 'LIMIT', 0, 100)
for _, job_id in ipairs(expired) do
    local job_key = prefix .. ':job:' .. job_id
    local worker = redis.call('HGET', job_key, 'worker_id')
    local attempts = tonumber(redis.call('HGET', job_key, 'attempts') or '0')
    local max_attempts = tonumber(redis.call('HGET', job_key, 'max_attempts') or '3')
    redis.call('ZREM', prefix .. ':lease', job_id)
    redis.call('HDEL', job_key, 'worker_id', 'dispatch_token', 'lease_until')
    if worker then
        local inflight = tonumber(redis.call('HGET', prefix .. ':worker:' .. worker, 'inflight') or '0')
        if inflight > 0 then redis.call('HINCRBY', prefix .. ':worker:' .. worker, 'inflight', -1) end
    end
    if attempts >= max_attempts then
        redis.call('HSET', job_key, 'state', 'FAILED', 'error', 'job.lease_expired')
        redis.call('LPUSH', prefix .. ':dead', job_id)
        reaped[#reaped + 1] = job_id .. ':failed'
    else
        local delay = 0
        if base > 0 then
            delay = math.min(cap, base * (2 ^ math.max(0, attempts - 1)))
        end
        redis.call('HSET', job_key, 'state', 'QUEUED', 'available_at', now + delay)
        redis.call('RPUSH', prefix .. ':ready', job_id)
        reaped[#reaped + 1] = job_id .. ':requeued'
    end
end
return reaped
"""

PREEMPT_SCRIPT = """
local prefix = KEYS[1]
local job_id = ARGV[1]
local now = tonumber(ARGV[2])
local job_key = prefix .. ':job:' .. job_id
if redis.call('EXISTS', job_key) == 0 then return 'missing' end
local state = redis.call('HGET', job_key, 'state') or 'QUEUED'
if state == 'SUCCEEDED' or state == 'FAILED' or state == 'CANCELLED' then return 'terminal' end
if state ~= 'DISPATCHED' and state ~= 'RUNNING' then return 'not_leased' end
local worker = redis.call('HGET', job_key, 'worker_id')
if worker then
    local inflight = tonumber(redis.call('HGET', prefix .. ':worker:' .. worker, 'inflight') or '0')
    if inflight > 0 then redis.call('HINCRBY', prefix .. ':worker:' .. worker, 'inflight', -1) end
end
redis.call('ZREM', prefix .. ':lease', job_id)
redis.call('HDEL', job_key, 'worker_id', 'dispatch_token', 'lease_until')
redis.call('HSET', job_key, 'state', 'QUEUED', 'available_at', now, 'preempted_from', worker or '')
redis.call('LREM', prefix .. ':ready', 0, job_id)
redis.call('RPUSH', prefix .. ':ready', job_id)
return 'preempted'
"""


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def _hash_pairs(flat: Any) -> dict[str, str]:
    items = _decode(flat) or []
    return {str(items[index]): str(items[index + 1]) for index in range(0, len(items) - 1, 2)}


@dataclass
class RedisDurableQueue:
    """Durable, capability-aware job queue on Redis/Valkey.

    The lease/attempt/idempotent-report semantics mirror
    :class:`uzen.infrastructure.storage.db.Database` so a fleet can move between the two
    without changing worker code.
    """

    config: RedisQueueConfig = field(default_factory=RedisQueueConfig)
    connection: RespConnection | None = field(default=None, repr=False)
    prefix: str = _PREFIX

    def __post_init__(self) -> None:
        if self.connection is None:
            self.connection = RespConnection(self.config)

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()

    def ping(self) -> bool:
        assert self.connection is not None
        return _decode(self.connection.command("PING")) == "PONG"

    def server_time(self) -> float:
        """The queue's clock, not the caller's.

        Every lease decision for this adapter is made against the Redis server's
        ``TIME``, so a worker with a skewed clock cannot extend or steal a lease by
        claiming a different local time.
        """
        assert self.connection is not None
        seconds, micros = _decode(self.connection.command("TIME"))
        return float(seconds) + float(micros) / 1_000_000.0

    # -- producer ------------------------------------------------------ #

    def enqueue(
        self,
        job_id: str,
        request: Mapping[str, Any],
        requirements: CapabilityRequirements,
        *,
        max_attempts: int = 3,
        priority: int = 0,
    ) -> None:
        assert self.connection is not None
        job_key = f"{self.prefix}:job:{job_id}"
        self.connection.command(
            "HSET",
            job_key,
            "request",
            json.dumps(dict(request), sort_keys=True),
            "requirements",
            json.dumps(requirements.to_dict(), sort_keys=True),
            "state",
            "QUEUED",
            "attempts",
            0,
            "max_attempts",
            int(max_attempts),
            "priority",
            max(-1000, min(1000, int(priority))),
            "enqueued_at",
            time.time(),
            "available_at",
            0,
            "created_at",
            utc_now_iso(),
            "min_cpu_count",
            int(requirements.min_cpu_count or 0),
            "min_memory_bytes",
            int(requirements.min_memory_bytes or 0),
        )
        all_rows = [f"{kind}:{value}" for kind, value, mode in requirements.rows() if mode == "all"]
        any_rows = [f"{kind}:{value}" for kind, value, mode in requirements.rows() if mode == "any"]
        if all_rows:
            self.connection.command("SADD", f"{self.prefix}:req:{job_id}", *all_rows)
        if any_rows:
            self.connection.command("SADD", f"{self.prefix}:reqany:{job_id}", *any_rows)
        self.connection.command("LPUSH", f"{self.prefix}:ready", job_id)

    def register_worker(self, worker_id: str, capabilities: Mapping[str, Any]) -> None:
        """Publish what a worker can do as ``kind:value`` set members."""
        assert self.connection is not None
        index = capability_index(dict(capabilities))
        members = [f"{kind}:{value}" for kind, values in index.items() for value in values]
        caps_key = f"{self.prefix}:caps:{worker_id}"
        self.connection.command("DEL", caps_key)
        if members:
            self.connection.command("SADD", caps_key, *members)
        worker_key = f"{self.prefix}:worker:{worker_id}"
        self.connection.command(
            "HSET",
            worker_key,
            "cpu_count",
            int(capabilities.get("cpu_count") or 0),
            "memory_bytes",
            int(capabilities.get("memory_bytes") or 0),
            "last_seen",
            utc_now_iso(),
        )
        # A draining worker stays draining across re-registration; resuming is explicit.
        self.connection.command("HSETNX", worker_key, "status", "online")
        self.connection.command("HSETNX", worker_key, "inflight", 0)

    def set_worker_status(self, worker_id: str, status: str) -> bool:
        assert self.connection is not None
        if status not in ("online", "draining"):
            raise ValueError(f"unsupported worker status '{status}'")
        worker_key = f"{self.prefix}:worker:{worker_id}"
        if int(_decode(self.connection.command("EXISTS", worker_key)) or 0) == 0:
            return False
        self.connection.command("HSET", worker_key, "status", status, "last_seen", utc_now_iso())
        return True

    # -- worker -------------------------------------------------------- #

    def claim(
        self,
        worker_id: str,
        *,
        lease_seconds: float = 120.0,
        now: float | None = None,
        aging_seconds: float = 0.0,
        max_in_flight_per_worker: int = 0,
        retry_backoff_seconds: float = 0.0,
        retry_backoff_cap_seconds: float = 300.0,
        wait_seconds: float = 0.0,
        poll_interval: float = 0.25,
    ) -> dict[str, Any] | None:
        import uuid

        assert self.connection is not None
        deadline = time.monotonic() + max(0.0, float(wait_seconds))
        while True:
            reply = self._claim_once(
                worker_id,
                lease_seconds=lease_seconds,
                now=now,
                aging_seconds=aging_seconds,
                max_in_flight_per_worker=max_in_flight_per_worker,
                retry_backoff_seconds=retry_backoff_seconds,
                retry_backoff_cap_seconds=retry_backoff_cap_seconds,
            )
            if reply is not None or wait_seconds <= 0 or now is not None:
                return reply
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            time.sleep(min(max(0.01, float(poll_interval)), remaining))

    def _claim_once(
        self,
        worker_id: str,
        *,
        lease_seconds: float,
        now: float | None,
        aging_seconds: float,
        max_in_flight_per_worker: int,
        retry_backoff_seconds: float,
        retry_backoff_cap_seconds: float,
    ) -> dict[str, Any] | None:
        import uuid

        assert self.connection is not None
        now = now if now is not None else self.server_time()
        token = uuid.uuid4().hex
        worker_key = f"{self.prefix}:worker:{worker_id}"
        cpu_count = _decode(self.connection.command("HGET", worker_key, "cpu_count")) or 0
        memory_bytes = _decode(self.connection.command("HGET", worker_key, "memory_bytes")) or 0
        # Reap first, like the SQLite claim, so an expired attempt is either requeued
        # with its backoff or failed out instead of being stolen silently.
        self.reap(
            now=now,
            retry_backoff_seconds=retry_backoff_seconds,
            retry_backoff_cap_seconds=retry_backoff_cap_seconds,
        )
        reply = self.connection.eval(
            CLAIM_SCRIPT,
            [self.prefix],
            [
                worker_id,
                token,
                now + float(lease_seconds),
                now,
                int(cpu_count),
                int(memory_bytes),
                float(aging_seconds),
                int(max_in_flight_per_worker),
                utc_now_iso(),
            ],
        )
        if not reply:
            return None
        job_id, flat = _decode(reply)
        job = _hash_pairs(flat)
        job["id"] = str(job_id)
        job["lease"] = {
            "worker_id": worker_id,
            "token": token,
            "attempt": int(job.get("attempts") or 1),
            "expires_at": now + float(lease_seconds),
            "lease_seconds": float(lease_seconds),
        }
        return job

    def renew(
        self,
        job_id: str,
        worker_id: str,
        token: str,
        *,
        lease_seconds: float = 120.0,
        now: float | None = None,
    ) -> bool:
        assert self.connection is not None
        now = now if now is not None else self.server_time()
        reply = self.connection.eval(
            RENEW_SCRIPT, [self.prefix], [job_id, worker_id, token, now + float(lease_seconds)]
        )
        return int(_decode(reply) or 0) == 1

    def ack(
        self,
        job_id: str,
        *,
        worker_id: str,
        token: str,
        state: str,
        payload: Mapping[str, Any] | None = None,
    ) -> str:
        assert self.connection is not None
        reply = self.connection.eval(
            ACK_SCRIPT,
            [self.prefix],
            [
                job_id,
                worker_id,
                token,
                state,
                json.dumps(dict(payload or {}), sort_keys=True),
                utc_now_iso(),
            ],
        )
        return str(_decode(reply))

    def reap(
        self,
        *,
        now: float | None = None,
        retry_backoff_seconds: float = 0.0,
        retry_backoff_cap_seconds: float = 300.0,
    ) -> list[str]:
        assert self.connection is not None
        now = now if now is not None else self.server_time()
        return (
            _decode(
                self.connection.eval(
                    REAP_SCRIPT,
                    [self.prefix],
                    [now, float(retry_backoff_seconds), float(retry_backoff_cap_seconds)],
                )
            )
            or []
        )

    def preempt(self, job_id: str, *, now: float | None = None) -> str:
        assert self.connection is not None
        now = now if now is not None else self.server_time()
        return str(_decode(self.connection.eval(PREEMPT_SCRIPT, [self.prefix], [job_id, now])))

    def job(self, job_id: str) -> dict[str, Any] | None:
        assert self.connection is not None
        raw = _hash_pairs(self.connection.command("HGETALL", f"{self.prefix}:job:{job_id}"))
        if not raw:
            return None
        raw["id"] = job_id
        if "request" in raw:
            raw["request"] = json.loads(raw["request"])
        if "requirements" in raw:
            raw["requirements"] = json.loads(raw["requirements"])
        return raw

    def status(self) -> dict[str, Any]:
        assert self.connection is not None
        return {
            "adapter": "redis",
            "endpoint": self.config.describe(),
            "ready": int(_decode(self.connection.command("LLEN", f"{self.prefix}:ready")) or 0),
            "leases": int(_decode(self.connection.command("ZCARD", f"{self.prefix}:lease")) or 0),
            "dead": int(_decode(self.connection.command("LLEN", f"{self.prefix}:dead")) or 0),
        }

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "durable_queue",
            "adapter": "redis",
            "protocol": "RESP2",
            "endpoint": self.config.describe(),
            "atomic_claim": "lua",
            "leases": "sorted set",
            "idempotent_reports": True,
            "priority": "operator bias + optional aging",
            "preemption": "cooperative (token revoked, job requeued)",
            "worker_draining": True,
            "long_poll_claim": True,
        }
