"""P12: worker isolation — enforced limits, timeouts and honest reporting."""

from __future__ import annotations

import os
import sys
import time

import pytest

from uzen.errors import InvalidOperationParameters, RuntimeUnavailable
from uzen.workers.sandbox import (
    DockerSandbox,
    ProcessSandbox,
    SandboxSpec,
    default_sandbox,
    describe_isolation,
    network_namespace_available,
    scrub_environment,
)

PY = sys.executable


def test_spec_validation_rejects_empty_and_absurd_limits() -> None:
    with pytest.raises(InvalidOperationParameters):
        SandboxSpec(command=[])
    with pytest.raises(InvalidOperationParameters):
        SandboxSpec(command=[PY], timeout_seconds=0)
    with pytest.raises(InvalidOperationParameters):
        SandboxSpec(command=[PY], cpu_seconds=-1)
    with pytest.raises(InvalidOperationParameters):
        SandboxSpec(command=[PY], memory_bytes=1024)


def test_scrubbed_environment_drops_credentials(monkeypatch) -> None:
    monkeypatch.setenv("UZEN_API_TOKEN", "super-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "super-secret")
    env = scrub_environment({"MY_SETTING": "ok"})
    assert env["MY_SETTING"] == "ok"
    assert "UZEN_API_TOKEN" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    with pytest.raises(InvalidOperationParameters):
        scrub_environment({"HF_TOKEN": "nope"})
    with pytest.raises(InvalidOperationParameters):
        scrub_environment({"OPENAI_API_KEY": "nope"})


def test_process_sandbox_runs_a_command_and_reports_limits() -> None:
    sandbox = ProcessSandbox()
    if not sandbox.available()[0]:
        pytest.skip("POSIX rlimits unavailable")
    result = sandbox.run(
        SandboxSpec(
            command=[PY, "-c", "import sys; print('hello sandbox'); print('warn', file=sys.stderr)"],
            timeout_seconds=30,
            memory_bytes=256 * 1024 * 1024,
        )
    )
    assert result.ok, result.to_dict()
    assert "hello sandbox" in result.stdout
    assert "warn" in result.stderr
    assert result.isolation["cpu_seconds"] == 30
    assert result.isolation["memory_bytes"] == 256 * 1024 * 1024
    assert result.isolation["env_scrubbed"] is True
    assert result.isolation["core_dumps"] == "disabled"


def test_process_sandbox_does_not_leak_parent_credentials(monkeypatch) -> None:
    sandbox = ProcessSandbox()
    if not sandbox.available()[0]:
        pytest.skip("POSIX rlimits unavailable")
    monkeypatch.setenv("UZEN_API_TOKEN", "top-secret")
    result = sandbox.run(
        SandboxSpec(
            command=[PY, "-c", "import os; print(os.environ.get('UZEN_API_TOKEN', 'absent'))"],
            timeout_seconds=30,
        )
    )
    assert result.ok, result.to_dict()
    assert result.stdout.strip() == "absent"


def test_memory_limit_is_enforced() -> None:
    sandbox = ProcessSandbox()
    if not sandbox.available()[0]:
        pytest.skip("POSIX rlimits unavailable")
    code = (
        "x = bytearray(300 * 1024 * 1024)\n"
        "print(len(x))\n"
    )
    result = sandbox.run(
        SandboxSpec(command=[PY, "-c", code], memory_bytes=64 * 1024 * 1024, timeout_seconds=30)
    )
    assert not result.ok
    assert result.exit_code != 0
    assert "MemoryError" in result.stderr or result.exit_code < 0


def test_wall_timeout_kills_the_process_group() -> None:
    sandbox = ProcessSandbox()
    if not sandbox.available()[0]:
        pytest.skip("POSIX rlimits unavailable")
    started = time.monotonic()
    result = sandbox.run(
        SandboxSpec(command=[PY, "-c", "import time; time.sleep(60)"], timeout_seconds=1.0)
    )
    elapsed = time.monotonic() - started
    assert result.timed_out is True
    assert result.killed_reason == "wall-clock timeout"
    assert result.exit_code is not None and result.exit_code != 0
    assert elapsed < 20


def test_cpu_rlimit_stops_a_busy_loop() -> None:
    sandbox = ProcessSandbox()
    if not sandbox.available()[0]:
        pytest.skip("POSIX rlimits unavailable")
    result = sandbox.run(
        SandboxSpec(
            command=[PY, "-c", "while True: pass"],
            cpu_seconds=1,
            timeout_seconds=20,
        )
    )
    assert result.exit_code is not None and result.exit_code != 0
    assert result.killed_reason in (None, "signal -9", "signal -24", "signal -11") or result.exit_code < 0


def test_network_namespace_report_matches_reality() -> None:
    available, reason = network_namespace_available()
    sandbox = ProcessSandbox()
    if not sandbox.available()[0]:
        pytest.skip("POSIX rlimits unavailable")
    result = sandbox.run(
        SandboxSpec(
            command=[
                PY,
                "-c",
                "import socket\ntry:\n socket.create_connection(('1.1.1.1', 80), timeout=2)\n print('connected')\nexcept OSError as exc:\n print('blocked', type(exc).__name__)\n",
            ],
            timeout_seconds=20,
        )
    )
    assert result.isolation["network"] == ("namespace" if available else "not_isolated")
    assert result.isolation["network_reason"]
    if available:
        assert "blocked" in result.stdout
    else:
        # The report is honest about the absence of enforcement instead of claiming it.
        assert "unavailable" in result.isolation["network_reason"].lower() or reason


def test_docker_argv_enforces_isolation() -> None:
    sandbox = DockerSandbox(cpus=0.5, pids_limit=32)
    spec = SandboxSpec(
        command=["-c", "print(1)"],
        image="python:3.12-slim",
        memory_bytes=128 * 1024 * 1024,
        env={"MODE": "test"},
    )
    argv = sandbox.argv(spec)
    joined = " ".join(argv)
    assert argv[:3] == [sandbox.docker, "run", "--rm"]
    assert "--network none" in joined
    assert "--read-only" in argv
    assert "--cap-drop ALL" in joined
    assert "--security-opt no-new-privileges" in joined
    assert "--pids-limit 32" in joined
    assert f"--memory {128 * 1024 * 1024}" in joined
    assert "--user 65534:65534" in joined
    assert "--env MODE=test" in joined
    assert argv[-4:] == ["python:3.12-slim", "python", "-c", "print(1)"]


def test_docker_network_flag_follows_the_spec() -> None:
    sandbox = DockerSandbox()
    assert "--network bridge" in " ".join(sandbox.argv(SandboxSpec(command=[PY], network=True)))
    assert "--network none" in " ".join(sandbox.argv(SandboxSpec(command=[PY], network=False)))


def test_docker_unavailable_raises_instead_of_running_on_the_host() -> None:
    sandbox = DockerSandbox(docker="/nonexistent/docker")
    available, reason = sandbox.available()
    assert available is False and "not found" in reason
    with pytest.raises(RuntimeUnavailable):
        sandbox.run(SandboxSpec(command=[PY, "-c", "print('should not run')"]))


def test_default_sandbox_and_capability_description() -> None:
    sandbox = default_sandbox()
    assert sandbox.name in ("docker", "process")
    report = describe_isolation()
    assert report["default"] in ("docker", "process", "none")
    assert set(report) >= {
        "docker",
        "docker_reason",
        "process",
        "process_reason",
        "network_namespace",
        "network_namespace_reason",
    }
    from uzen.workers.protocol.messages import detect_capabilities

    capabilities = detect_capabilities()
    assert capabilities["sandbox"]["default"] == report["default"]


@pytest.mark.skipif(not os.environ.get("UZEN_TEST_DOCKER"), reason="set UZEN_TEST_DOCKER=1 to run a container")
def test_docker_really_runs_a_container() -> None:
    sandbox = DockerSandbox()
    available, reason = sandbox.available()
    if not available:
        pytest.skip(f"docker unavailable: {reason}")
    result = sandbox.run(
        SandboxSpec(command=["-c", "print('container ok')"], image="python:3.12-slim", timeout_seconds=120)
    )
    assert result.ok, result.to_dict()
    assert "container ok" in result.stdout
    assert result.isolation["network"] == "none"
