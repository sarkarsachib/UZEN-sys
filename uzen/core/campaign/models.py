"""Campaign candidate, trial and report records.

These records are the unit of persistence and of the benchmark corpus. Timestamps
are deliberately excluded from candidate digests, corpus digests and the
deterministic report digest so a rerun on another machine compares the actual
research result rather than wall-clock metadata.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping

from uzen.util.canonical import canonical_bytes, utc_now_iso


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


@dataclass(frozen=True)
class Candidate:
    values: tuple[Any, ...]
    config: dict[str, Any]
    digest: str

    @classmethod
    def build(cls, mode: str, family: str, values: tuple[Any, ...], config: Mapping[str, Any]) -> "Candidate":
        clean = dict(config)
        payload = {"mode": mode, "family": family, "config": clean}
        return cls(values=tuple(values), config=clean, digest=_digest(payload))

    @property
    def spec(self) -> dict[str, Any]:
        return dict(self.config.get("spec", {}))

    @property
    def transforms(self) -> dict[str, Any]:
        return dict(self.config.get("transforms", {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "values": list(self.values),
            "config": self.config,
        }


@dataclass
class Trial:
    candidate: Candidate
    round_index: int
    sequence: int
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    package_fingerprint: str = ""
    graph_digest: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    score: float | None = None
    constraint_failures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "candidate": self.candidate.to_dict(),
            "round": self.round_index,
            "sequence": self.sequence,
            "status": self.status,
            "metrics": dict(self.metrics),
            "error": self.error,
            "package_fingerprint": self.package_fingerprint,
            "graph_digest": self.graph_digest,
            "score": self.score,
            "constraint_failures": list(self.constraint_failures),
            "created_at": self.created_at,
        }
        return payload


@dataclass
class CampaignReport:
    campaign_id: str
    name: str
    mode: str
    status: str
    stop_reason: str
    spec_digest: str
    seed: int
    budget: int
    batch_size: int
    strategy: str
    trials: list[Trial]
    pareto: list[dict[str, Any]]
    best: dict[str, Any] | None
    rounds: int
    corpus_digest: str
    created_at: str
    updated_at: str
    objectives: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for trial in self.trials:
            counts[trial.status] = counts.get(trial.status, 0) + 1
        return {
            "campaign_id": self.campaign_id,
            "name": self.name,
            "mode": self.mode,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "spec_digest": self.spec_digest,
            "seed": self.seed,
            "budget": self.budget,
            "batch_size": self.batch_size,
            "strategy": self.strategy,
            "objectives": self.objectives,
            "rounds": self.rounds,
            "trials_evaluated": len(self.trials),
            "status_counts": counts,
            "best": self.best,
            "pareto": self.pareto,
            "trials": [trial.to_dict() for trial in self.trials],
            "corpus_digest": self.corpus_digest,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deterministic_digest": self.deterministic_digest(),
        }

    def deterministic_digest(self) -> str:
        payload = {
            "campaign_id": self.campaign_id,
            "spec_digest": self.spec_digest,
            "seed": self.seed,
            "budget": self.budget,
            "strategy": self.strategy,
            "stop_reason": self.stop_reason,
            "objectives": self.objectives,
            "rounds": self.rounds,
            "trials": [
                {
                    "candidate": trial.candidate.digest,
                    "round": trial.round_index,
                    "sequence": trial.sequence,
                    "status": trial.status,
                    "metrics": trial.metrics,
                    "error": trial.error,
                    "package_fingerprint": trial.package_fingerprint,
                    "graph_digest": trial.graph_digest,
                }
                for trial in self.trials
            ],
            "pareto": self.pareto,
            "best": self.best,
            "corpus_digest": self.corpus_digest,
        }
        return _digest(payload)
