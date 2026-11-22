"""Worker isolation: run untrusted job payloads under enforced limits.

Two sandboxes are provided, and both refuse rather than pretend:

* :class:`DockerSandbox` runs the payload in a container with no network
  (``--network none``), a read-only root filesystem, dropped capabilities,
  ``no-new-privileges``, a non-root user, a CPU quota, a memory limit and a PID
  limit. ``available()`` reports the real reason when the Docker CLI or daemon is
  missing; :meth:`DockerSandbox.run` then raises
  :class:`~uzen.errors.RuntimeUnavailable` instead of silently executing on the host.
* :class:`ProcessSandbox` is the no-container fallback for POSIX hosts. It applies
  ``setrlimit`` CPU, address-space, process-count, file-size and descriptor limits
  before ``exec``, runs in a new session that is killed as a group on timeout, uses a
  fresh temporary working directory, and scrubs the environment (credentials such as
  ``UZEN_*``, ``AWS_*``, ``*_TOKEN`` never reach the child). Network isolation uses a
  network namespace (``unshare -n``) *when the host permits it*; the result's
  ``isolation`` report states exactly which enforcement was active.

Every result carries that report, so a caller can assert what was enforced instead of
trusting a flag. The worker capability document advertises the same facts
(:func:`describe_isolation`), which is how a scheduler can route a job to a host that
can actually isolate it.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from uzen.errors import InvalidOperationParameters, RuntimeUnavailable

try:  # POSIX only; the module still imports on any platform
    import resource
except Exception:  # pragma: no cover - non-POSIX
    resource = None  # type: ignore[assignment]

DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_OUTPUT_BYTES = 1_000_000
_SCRUBBED_PREFIXES = ("UZEN_", "AWS_", "GCP_", "AZURE_", "HF_", "CONVEX_", "OPENAI_")
_SCRUBBED_SUFFIXES = ("_TOKEN", "_SECRET", "_KEY", "_PASSWORD")
_SAFE_ENV_KEYS = ("PATH", "LANG", "LC_ALL", "TZ", "TERM")


@dataclass(frozen=True)
class SandboxSpec:
    """What to run and the limits it must run under."""

    command: Sequence[str]
    image: str = "python:3.12-slim"
    cpu_seconds: int = 30
    memory_bytes: int | None = 512 * 1024 * 1024
    # ``None`` means "the sandbox's own default"; an explicit value always wins.
    pids_limit: int | None = None
    network: bool = False
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    working_dir: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    read_only: bool = True
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES
    user: str | None = None

    def __post_init__(self) -> None:
        command = [str(part) for part in self.command]
        if not command:
            raise InvalidOperationParameters("a sandbox command is required")
        if self.timeout_seconds <= 0:
            raise InvalidOperationParameters("timeout_seconds must be positive")
        if self.cpu_seconds <= 0:
            raise InvalidOperationParameters("cpu_seconds must be positive")
        if self.memory_bytes is not None and self.memory_bytes < 1024 * 1024:
            raise InvalidOperationParameters("memory_bytes must be at least 1 MiB")
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "env", {str(k): str(v) for k, v in self.env.items()})

    def labelled_command(self) -> list[str]:
        """The in-container argv: a script path or Python flags imply the interpreter."""
        first = self.command[0]
        if first.endswith(".py") or first.startswith("-"):
            return ["python", *self.command]
        return list(self.command)


@dataclass
class SandboxResult:
    exit_code: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    killed_reason: str | None = None
    sandbox: str = "process"
    isolation: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def to_dict(self) -> dict[str, Any]:
        return {
            "sandbox": self.sandbox,
            "exit_code": self.exit_code,
            "ok": self.ok,
            "timed_out": self.timed_out,
            "killed_reason": self.killed_reason,
            "duration_seconds": round(self.duration_seconds, 6),
            "stdout": self.stdout,
            "stderr": self.stderr,
            "isolation": dict(self.isolation),
        }


def scrub_environment(extra: Mapping[str, str] | None = None, *, home: str | None = None) -> dict[str, str]:
    """Return the minimal environment a sandboxed child receives."""
    env: dict[str, str] = {}
    for key in _SAFE_ENV_KEYS:
        if key in os.environ:
            env[key] = os.environ[key]
    env.setdefault("PATH", os.defpath)
    env["HOME"] = home or tempfile.gettempdir()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    for key, value in dict(extra or {}).items():
        text = str(key)
        if any(text.startswith(prefix) for prefix in _SCRUBBED_PREFIXES) or text.endswith(
            _SCRUBBED_SUFFIXES
        ):
            raise InvalidOperationParameters(
                f"environment variable '{text}' looks like a credential and is not "
                "passed into a sandbox"
            )
        env[text] = str(value)
    return env


_unshare_state: dict[str, Any] | None = None


def network_namespace_available() -> tuple[bool, str]:
    """Probe (once) whether an unprivileged network namespace can be created."""
    global _unshare_state
    if _unshare_state is not None:
        return bool(_unshare_state["available"]), str(_unshare_state["reason"])
    unshare = shutil.which("unshare")
    if unshare is None:
        _unshare_state = {"available": False, "reason": "unshare binary not installed"}
        return False, "unshare binary not installed"
    try:
        probe = subprocess.run(
            [unshare, "--net", "--", "true"],
            capture_output=True,
            timeout=5,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _unshare_state = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
        return False, str(_unshare_state["reason"])
    if probe.returncode == 0:
        _unshare_state = {"available": True, "reason": "unshare --net succeeded"}
        return True, "unshare --net succeeded"
    detail = (probe.stderr or probe.stdout or "").strip().splitlines()
    reason = detail[-1] if detail else f"unshare exited {probe.returncode}"
    _unshare_state = {"available": False, "reason": reason}
    return False, reason


def _read_stream(stream: Any, limit: int, sink: list[bytes]) -> None:
    total = 0
    try:
        while True:
            chunk = stream.read(8192)
            if not chunk:
                break
            if total < limit:
                sink.append(chunk[: limit - total])
            total += len(chunk)
    finally:
        try:
            stream.close()
        except OSError:
            pass


class ProcessSandbox:
    """POSIX process sandbox using rlimits, a process group and a scrubbed env."""

    name = "process"

    def available(self) -> tuple[bool, str]:
        if resource is None or os.name != "posix":
            return False, "POSIX rlimits are not available on this platform"
        return True, "setrlimit/seccomp-free process isolation available"

    def _limits(self, spec: SandboxSpec) -> Any:
        def apply() -> None:  # runs in the child, before exec
            os.setsid()
            if resource is None:  # pragma: no cover - guarded by available()
                return
            resource.setrlimit(resource.RLIMIT_CPU, (spec.cpu_seconds, spec.cpu_seconds + 1))
            if spec.memory_bytes:
                resource.setrlimit(resource.RLIMIT_AS, (spec.memory_bytes, spec.memory_bytes))
            if spec.pids_limit:
                resource.setrlimit(resource.RLIMIT_NPROC, (spec.pids_limit, spec.pids_limit))
            resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
            resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        return apply

    def run(self, spec: SandboxSpec) -> SandboxResult:
        available, reason = self.available()
        if not available:
            raise RuntimeUnavailable("the process sandbox cannot run here", reason=reason)
        namespace, namespace_reason = (False, "network isolation not requested")
        command = list(spec.command)
        if not spec.network:
            namespace, namespace_reason = network_namespace_available()
            if namespace:
                command = ["unshare", "--net", "--", *command]
        workdir = Path(spec.working_dir) if spec.working_dir else Path(tempfile.mkdtemp(prefix="uzen-sbx-"))
        workdir.mkdir(parents=True, exist_ok=True)
        env = scrub_environment(spec.env, home=str(workdir))
        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        started = time.monotonic()
        try:
            process = subprocess.Popen(
                command,
                cwd=str(workdir),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=self._limits(spec),
                close_fds=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeValidationError(f"command not found: {command[0]}") from exc
        readers = [
            threading.Thread(
                target=_read_stream,
                args=(process.stdout, spec.max_output_bytes, stdout_chunks),
                daemon=True,
            ),
            threading.Thread(
                target=_read_stream,
                args=(process.stderr, spec.max_output_bytes, stderr_chunks),
                daemon=True,
            ),
        ]
        for reader in readers:
            reader.start()
        timed_out = False
        killed_reason: str | None = None
        try:
            exit_code = process.wait(timeout=spec.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            killed_reason = "wall-clock timeout"
            self._kill(process)
            exit_code = process.wait(timeout=10)
        if not timed_out and exit_code is not None and exit_code < 0:
            killed_reason = f"signal {-exit_code}"
        duration = time.monotonic() - started
        for reader in readers:
            reader.join(timeout=2)
        return SandboxResult(
            exit_code=exit_code,
            stdout=b"".join(stdout_chunks).decode("utf-8", "replace"),
            stderr=b"".join(stderr_chunks).decode("utf-8", "replace"),
            duration_seconds=duration,
            timed_out=timed_out,
            killed_reason=killed_reason,
            sandbox=self.name,
            isolation=self.isolation_report(spec, namespace=namespace, reason=namespace_reason),
        )

    def _kill(self, process: subprocess.Popen[bytes]) -> None:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                process.kill()
            except ProcessLookupError:
                pass

    def isolation_report(
        self, spec: SandboxSpec, *, namespace: bool | None = None, reason: str = ""
    ) -> dict[str, Any]:
        if namespace is None:
            namespace, reason = network_namespace_available() if not spec.network else (False, "network allowed by spec")
        return {
            "sandbox": self.name,
            "cpu_seconds": spec.cpu_seconds,
            "memory_bytes": spec.memory_bytes,
            "pids_limit": spec.pids_limit,
            "file_size_bytes": 64 * 1024 * 1024,
            "open_files": 256,
            "core_dumps": "disabled",
            "network": "namespace" if namespace else "none_requested" if spec.network else "not_isolated",
            "network_reason": reason,
            "filesystem": "ephemeral_tempdir",
            "env_scrubbed": True,
            "new_session": True,
        }


class RuntimeValidationError(InvalidOperationParameters):
    """The sandboxed command cannot be started at all."""

    code = "worker.sandbox_invalid"


class DockerSandbox:
    """Container sandbox using the Docker CLI with explicit isolation flags."""

    name = "docker"

    def __init__(
        self,
        *,
        docker: str | None = None,
        cpus: float = 1.0,
        pids_limit: int = 128,
        user: str = "65534:65534",
    ) -> None:
        self.docker = docker or shutil.which("docker") or "docker"
        self.cpus = float(cpus)
        self.pids_limit = int(pids_limit)
        self.user = user

    def available(self) -> tuple[bool, str]:
        if shutil.which(self.docker) is None:
            return False, f"docker CLI '{self.docker}' not found"
        try:
            probe = subprocess.run(
                [self.docker, "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                text=True,
                timeout=8,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"{type(exc).__name__}: {exc}"
        if probe.returncode != 0:
            detail = (probe.stderr or probe.stdout).strip().splitlines()
            return False, detail[-1] if detail else f"docker info exited {probe.returncode}"
        return True, f"docker server {probe.stdout.strip()}"

    def argv(self, spec: SandboxSpec, *, container_name: str | None = None) -> list[str]:
        """The exact ``docker run`` argument vector for a spec."""
        memory = spec.memory_bytes or 512 * 1024 * 1024
        argv = [
            self.docker,
            "run",
            "--rm",
            "--network",
            "bridge" if spec.network else "none",
            "--cpus",
            str(self.cpus),
            "--memory",
            str(int(memory)),
            "--memory-swap",
            str(int(memory)),
            "--pids-limit",
            str(spec.pids_limit or self.pids_limit),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            spec.user or self.user,
            "--workdir",
            "/workspace",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
        ]
        if container_name:
            argv.extend(["--name", container_name])
        if spec.read_only:
            argv.append("--read-only")
        if spec.working_dir:
            argv.extend(["--mount", f"type=bind,src={Path(spec.working_dir).resolve()},dst=/workspace,ro"])
        for key, value in sorted(spec.env.items()):
            argv.extend(["--env", f"{key}={value}"])
        argv.append(spec.image)
        argv.extend(spec.labelled_command())
        return argv

    def run(self, spec: SandboxSpec) -> SandboxResult:
        available, reason = self.available()
        if not available:
            raise RuntimeUnavailable(
                "the Docker sandbox is not available on this host", reason=reason
            )
        started = time.monotonic()
        try:
            completed = subprocess.run(
                self.argv(spec),
                capture_output=True,
                text=True,
                timeout=spec.timeout_seconds + 10,
            )
        except subprocess.TimeoutExpired as exc:
            return SandboxResult(
                exit_code=None,
                stdout=(exc.stdout or b"").decode("utf-8", "replace")
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or ""),
                stderr=(exc.stderr or b"").decode("utf-8", "replace")
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or ""),
                duration_seconds=time.monotonic() - started,
                timed_out=True,
                killed_reason="docker run wall-clock timeout",
                sandbox=self.name,
                isolation=self.isolation_report(spec),
            )
        return SandboxResult(
            exit_code=completed.returncode,
            stdout=completed.stdout[-spec.max_output_bytes :],
            stderr=completed.stderr[-spec.max_output_bytes :],
            duration_seconds=time.monotonic() - started,
            sandbox=self.name,
            isolation=self.isolation_report(spec),
        )

    def isolation_report(self, spec: SandboxSpec) -> dict[str, Any]:
        return {
            "sandbox": self.name,
            "network": "bridge" if spec.network else "none",
            "root_filesystem": "read_only" if spec.read_only else "writable",
            "capabilities": "all_dropped",
            "no_new_privileges": True,
            "user": spec.user or self.user,
            "cpus": self.cpus,
            "memory_bytes": spec.memory_bytes or 512 * 1024 * 1024,
            "pids_limit": spec.pids_limit or self.pids_limit,
            "tmpfs": "/tmp:rw,noexec,nosuid,size=64m",
            "image": spec.image,
        }


def default_sandbox() -> Any:
    """Docker when a daemon answers, the rlimit process sandbox otherwise."""
    docker = DockerSandbox()
    available, _ = docker.available()
    if available:
        return docker
    process = ProcessSandbox()
    available, reason = process.available()
    if available:
        return process
    raise RuntimeUnavailable("no worker sandbox is available", reason=reason)


def describe_isolation() -> dict[str, Any]:
    """The sandbox facts a worker advertises in its capability document."""
    docker = DockerSandbox()
    docker_ok, docker_reason = docker.available()
    process_ok, process_reason = ProcessSandbox().available()
    namespace_ok, namespace_reason = network_namespace_available()
    return {
        "docker": docker_ok,
        "docker_reason": docker_reason,
        "process": process_ok,
        "process_reason": process_reason,
        "network_namespace": namespace_ok,
        "network_namespace_reason": namespace_reason,
        "default": "docker" if docker_ok else ("process" if process_ok else "none"),
    }


__all__ = [
    "DockerSandbox",
    "ProcessSandbox",
    "RuntimeValidationError",
    "SandboxResult",
    "SandboxSpec",
    "default_sandbox",
    "describe_isolation",
    "network_namespace_available",
    "scrub_environment",
]
