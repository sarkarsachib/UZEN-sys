"""Capability model for distributed job scheduling.

A job is dispatched to a worker only when that worker has *proved* it can execute it.
The proof is expressed as a requirement set derived from the job request itself, and
matched against the capability document a worker advertises from its real

    environment (:func:`uzen.workers.protocol.messages.detect_capabilities`).

Two representations of the same decision exist, deliberately:

* :func:`requirements_from_request` / :func:`match_worker` — the Python predicate,
  used to explain *why* a worker is or is not eligible (and reported to operators).
* the row form written into the job database (``job_requirements`` /
  ``worker_capabilities``) — the SQL predicate used by
  :meth:`uzen.infrastructure.storage.db.Database.claim_job`, so the capability test
  happens inside the same atomic statement that takes the lease. That is what makes
  dispatch safe when several control-plane processes share one database.

Requirement kinds
-----------------
``operation``       registered operation code the job will execute (all-of)
``operator``        NIR operator the graph needs (all-of)
``family``          architecture family the artifact/graph belongs to (all-of)
``policy``          policy the worker must enforce (all-of, always ``STRICT_PLUS``)
``device``          execution device (all-of: ``cpu`` or ``cuda``)
``representation``  native weight layout the worker must support (all-of)
``transport``       artifact transport the worker may use (any-of)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

CAPABILITY_KINDS: tuple[str, ...] = (
    "operation",
    "operator",
    "family",
    "policy",
    "device",
    "representation",
    "transport",
)

# Requirement kinds whose values are alternatives rather than a conjunction: a
# worker satisfies ``transport`` if it implements *any* of the listed transports.
ANY_OF_KINDS: frozenset[str] = frozenset({"transport"})

POLICY_REQUIREMENT = "STRICT_PLUS"


def _as_strings(value: Any) -> tuple[str, ...]:
    """Normalize an advertised/required capability value into a sorted tuple."""
    if value is None:
        return ()
    if isinstance(value, str):
        items: Iterable[Any] = (value,)
    elif isinstance(value, Mapping):
        items = (str(k) for k in value)
    elif isinstance(value, Sequence):
        items = value
    else:
        items = (value,)
    seen: dict[str, None] = {}
    for item in items:
        text = str(item).strip()
        if text:
            seen.setdefault(text, None)
    return tuple(sorted(seen))


@dataclass(frozen=True)
class CapabilityRequirements:
    """What a job needs from the worker that claims it."""

    operation: tuple[str, ...] = ()
    operator: tuple[str, ...] = ()
    family: tuple[str, ...] = ()
    policy: tuple[str, ...] = (POLICY_REQUIREMENT,)
    device: tuple[str, ...] = ()
    representation: tuple[str, ...] = ()
    transport: tuple[str, ...] = ()
    min_cpu_count: int = 0
    min_memory_bytes: int | None = None

    def rows(self) -> list[tuple[str, str, str]]:
        """Flatten to ``(kind, value, mode)`` rows for the job database."""
        rows: list[tuple[str, str, str]] = []
        for kind in CAPABILITY_KINDS:
            mode = "any" if kind in ANY_OF_KINDS else "all"
            for value in getattr(self, kind):
                rows.append((kind, value, mode))
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": list(self.operation),
            "operator": list(self.operator),
            "family": list(self.family),
            "policy": list(self.policy),
            "device": list(self.device),
            "representation": list(self.representation),
            "transport": list(self.transport),
            "min_cpu_count": self.min_cpu_count,
            "min_memory_bytes": self.min_memory_bytes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CapabilityRequirements":
        return cls(
            operation=_as_strings(data.get("operation")),
            operator=_as_strings(data.get("operator")),
            family=_as_strings(data.get("family")),
            policy=_as_strings(data.get("policy")) or (POLICY_REQUIREMENT,),
            device=_as_strings(data.get("device")),
            representation=_as_strings(data.get("representation")),
            transport=_as_strings(data.get("transport")),
            min_cpu_count=int(data.get("min_cpu_count") or 0),
            min_memory_bytes=(
                int(data["min_memory_bytes"]) if data.get("min_memory_bytes") is not None else None
            ),
        )

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Mapping[str, Any]],
        *,
        min_cpu_count: int | None = None,
        min_memory_bytes: int | None = None,
    ) -> "CapabilityRequirements":
        """Rebuild a requirement set from its stored (kind, value) rows."""
        grouped: dict[str, list[str]] = {}
        for row in rows:
            grouped.setdefault(str(row["kind"]), []).append(str(row["value"]))
        return cls(
            operation=tuple(sorted(set(grouped.get("operation", ())))),
            operator=tuple(sorted(set(grouped.get("operator", ())))),
            family=tuple(sorted(set(grouped.get("family", ())))),
            policy=tuple(sorted(set(grouped.get("policy", ())))),
            device=tuple(sorted(set(grouped.get("device", ())))),
            representation=tuple(sorted(set(grouped.get("representation", ())))),
            transport=tuple(sorted(set(grouped.get("transport", ())))),
            min_cpu_count=int(min_cpu_count or 0),
            min_memory_bytes=int(min_memory_bytes) if min_memory_bytes else None,
        )

    def is_empty(self) -> bool:
        return not any(getattr(self, kind) for kind in CAPABILITY_KINDS) and not (
            self.min_cpu_count or self.min_memory_bytes
        )


def requirements_from_request(request: Mapping[str, Any]) -> CapabilityRequirements:
    """Derive a worker requirement set from a job request.

    Only values the request actually states become requirements. A request that says
    nothing about the device is schedulable on any device; one that asks for
    ``cuda`` is not silently downgraded to CPU — it stays unroutable until a worker
    advertises CUDA.
    """
    compute = dict(request.get("compute_requirements") or {})
    options = dict(request.get("options") or {})
    plan = dict(request.get("evaluation_plan") or {})

    operations: list[str] = []
    for entry in request.get("operations") or []:
        if isinstance(entry, Mapping):
            code = entry.get("op") or entry.get("code")
            if code:
                operations.append(str(code))

    graphs = {
        "plan": plan.get("graph"),
        "options": options.get("graph"),
    }
    families: list[str] = []
    for key in ("family", "research_family"):
        if options.get(key):
            families.append(str(options[key]))
    for graph in graphs.values():
        if isinstance(graph, Mapping) and graph.get("family"):
            families.append(str(graph["family"]))
    if plan.get("family"):
        families.append(str(plan["family"]))

    device = compute.get("device") or options.get("device") or plan.get("device")
    operators = _as_strings(compute.get("operators") or options.get("operators"))
    representations: list[str] = []
    requested = options.get("representations") or compute.get("representations")
    if isinstance(requested, Mapping):
        for spec in requested.values():
            kind = spec.get("kind") if isinstance(spec, Mapping) else spec
            if kind:
                representations.append(str(kind))
    elif requested:
        representations.extend(_as_strings(requested))
    transports = _as_strings(
        compute.get("artifact_transport") or options.get("artifact_transport")
    )
    min_cpu = compute.get("min_cpu_count") or options.get("min_cpu_count") or 0
    min_memory = compute.get("min_memory_bytes") or options.get("min_memory_bytes")

    return CapabilityRequirements(
        operation=tuple(sorted(set(operations))),
        operator=operators,
        family=tuple(sorted(set(families))),
        device=_as_strings(device),
        representation=tuple(sorted(set(representations))),
        transport=transports,
        min_cpu_count=int(min_cpu),
        min_memory_bytes=int(min_memory) if min_memory is not None else None,
    )


def capability_index(capabilities: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Normalize a worker capability document into the same kind/value space."""
    index: dict[str, tuple[str, ...]] = {
        "operation": _as_strings(capabilities.get("supported_operations")),
        "operator": _as_strings(capabilities.get("supported_operators")),
        "family": _as_strings(capabilities.get("supported_architectures")),
        "representation": _as_strings(capabilities.get("representations")),
        "transport": _as_strings(capabilities.get("artifact_transport")),
    }
    devices: list[str] = []
    reported = _as_strings(capabilities.get("device"))
    if reported:
        devices.extend(reported)
    elif capabilities.get("cuda_available"):
        devices.append("cuda")
    if capabilities.get("cuda_available") and "cuda" not in devices:
        devices.append("cuda")
    if capabilities.get("cpu_count") is not None and "cpu" not in devices:
        devices.append("cpu")
    index["device"] = tuple(sorted(set(devices)))
    index["policy"] = (POLICY_REQUIREMENT,) if capabilities.get("strict_plus", True) else ()
    return index


@dataclass
class WorkerMatch:
    """Structured eligibility decision for one worker against one requirement set."""

    worker_id: str
    eligible: bool
    missing: dict[str, list[str]] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "eligible": self.eligible,
            "missing": {k: list(v) for k, v in self.missing.items()},
            "reasons": list(self.reasons),
        }


def match_worker(
    requirements: CapabilityRequirements,
    capabilities: Mapping[str, Any],
    *,
    worker_id: str = "",
) -> WorkerMatch:
    """Decide whether one advertised capability document satisfies a requirement set."""
    index = capability_index(capabilities)
    missing: dict[str, list[str]] = {}
    reasons: list[str] = []
    for kind in CAPABILITY_KINDS:
        wanted = getattr(requirements, kind)
        if not wanted:
            continue
        available = set(index.get(kind, ()))
        if kind in ANY_OF_KINDS:
            shared = [value for value in wanted if value in available]
            if not shared:
                missing[kind] = list(wanted)
                reasons.append(
                    f"worker supports {sorted(available) or 'no'} {kind}s, job needs one of {list(wanted)}"
                )
            continue
        absent = [value for value in wanted if value not in available]
        if absent:
            missing[kind] = absent
            reasons.append(f"worker does not advertise {kind} {absent}")

    if requirements.min_cpu_count:
        cpu = capabilities.get("cpu_count")
        cpu = int(cpu) if cpu is not None else 0
        if cpu < requirements.min_cpu_count:
            missing["cpu_count"] = [str(requirements.min_cpu_count)]
            reasons.append(f"worker has {cpu} cpus, job needs {requirements.min_cpu_count}")
    if requirements.min_memory_bytes:
        memory = capabilities.get("memory_bytes")
        memory = int(memory) if memory is not None else 0
        if memory < requirements.min_memory_bytes:
            missing["memory_bytes"] = [str(requirements.min_memory_bytes)]
            reasons.append(
                f"worker has {memory} bytes of memory, job needs {requirements.min_memory_bytes}"
            )
    return WorkerMatch(worker_id=worker_id, eligible=not missing, missing=missing, reasons=reasons)


def explain_assignment(
    requirements: CapabilityRequirements,
    workers: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Explain routing for one requirement set across the registered workers.

    Used by the scheduler's status report: an unroutable job must state which workers
    were considered and exactly which capability each one lacked, instead of sitting
    in the queue with no explanation.
    """
    matches = [
        match_worker(
            requirements,
            dict(worker.get("capabilities") or {}),
            worker_id=str(worker.get("worker_id", "")),
        )
        for worker in workers
    ]
    eligible = [m.worker_id for m in matches if m.eligible]
    return {
        "eligible_workers": eligible,
        "routable": bool(eligible),
        "considered": [m.to_dict() for m in matches],
    }
