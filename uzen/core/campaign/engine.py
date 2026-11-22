"""Campaign engine: plan, evaluate batches, rank, stop, persist, resume.

The scheduler is intentionally in-process and batch-oriented. It uses a bounded
thread pool for independent NumPy forward passes; durable state lives in SQLite and
the benchmark corpus. It does not claim distributed execution or hardware
acceleration. A campaign can be stopped at any batch boundary and resumed from the
same deterministic seed.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from uzen.core.campaign.corpus import BenchmarkCorpus
from uzen.core.campaign.evaluate import Evaluator
from uzen.core.campaign.models import CampaignReport, Candidate, Trial
from uzen.core.campaign.space import propose
from uzen.core.campaign.spec import CampaignSpec, Objective
from uzen.core.campaign.store import CampaignStore
from uzen.core.nir.graph import GraphValidationError
from uzen.errors import (
    CampaignNotFound,
    CampaignTimeout,
    InvalidOperationParameters,
    ShapeMismatch,
    UzenError,
)
from uzen.util.canonical import utc_now_iso

TrialExecutor = Callable[[CampaignSpec, Sequence[Candidate]], list[dict[str, Any]]]

TERMINAL_STOP_REASONS = ("budget", "patience", "exhausted")
PAUSE_STOP_REASON = "max_rounds"


def _is_eligible(trial: Trial, objectives: Sequence[Objective]) -> bool:
    return trial.status == "ok" and all(
        objective.name in trial.metrics for objective in objectives
    )


def score_trials(trials: Sequence[Trial], objectives: Sequence[Objective]) -> list[Trial]:
    """Assign normalized weighted scores in place and return best-first trials.

    Each objective is min-max normalized over the eligible trials, direction is
    applied, and the weighted sum is divided by the total weight. Scores therefore
    live in [0, 1] and remain comparable across reports. Ties are broken by
    candidate digest, never by insertion order.
    """
    eligible = [trial for trial in trials if _is_eligible(trial, objectives)]
    for trial in trials:
        trial.score = None
    if not eligible:
        return []
    ranges: dict[str, tuple[float, float]] = {}
    for objective in objectives:
        values = [float(trial.metrics[objective.name]) for trial in eligible]
        ranges[objective.name] = (min(values), max(values))
    total_weight = sum(objective.weight for objective in objectives)
    for trial in eligible:
        score = 0.0
        for objective in objectives:
            low, high = ranges[objective.name]
            value = float(trial.metrics[objective.name])
            normalized = 0.0 if high == low else (value - low) / (high - low)
            if objective.direction == "min":
                normalized = 1.0 - normalized
            score += objective.weight * normalized
        trial.score = score / total_weight
    return sorted(
        eligible,
        key=lambda trial: (-(trial.score or 0.0), trial.candidate.digest),
    )


def dominates(left: Mapping[str, Any], right: Mapping[str, Any], objectives: Sequence[Objective]) -> bool:
    at_least_one = False
    for objective in objectives:
        a = float(left[objective.name])
        b = float(right[objective.name])
        if objective.direction == "min":
            if a > b:
                return False
            if a < b:
                at_least_one = True
        else:
            if a < b:
                return False
            if a > b:
                at_least_one = True
    return at_least_one


def pareto_front(
    trials: Sequence[Trial], objectives: Sequence[Objective]
) -> list[Trial]:
    eligible = [trial for trial in trials if _is_eligible(trial, objectives)]
    front: list[Trial] = []
    for candidate in eligible:
        if not any(
            dominates(other.metrics, candidate.metrics, objectives)
            for other in eligible
            if other is not candidate
        ):
            front.append(candidate)
    return sorted(front, key=lambda trial: (-(trial.score or 0.0), trial.candidate.digest))


def _trial_entry(trial: Trial) -> dict[str, Any]:
    return {
        "candidate": trial.candidate.to_dict(),
        "status": trial.status,
        "metrics": dict(trial.metrics),
        "score": trial.score,
        "round": trial.round_index,
        "sequence": trial.sequence,
        "package_fingerprint": trial.package_fingerprint,
        "graph_digest": trial.graph_digest,
        "constraint_failures": list(trial.constraint_failures),
    }


def _constraint_failures(spec: CampaignSpec, metrics: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    for metric, bound in (spec.constraints or {}).items():
        value = metrics.get(metric)
        if value is None:
            failures.append(f"{metric} is missing")
            continue
        number = float(value)
        if "min" in bound and number < float(bound["min"]):
            failures.append(f"{metric}={number} < min {bound['min']}")
        if "max" in bound and number > float(bound["max"]):
            failures.append(f"{metric}={number} > max {bound['max']}")
    return failures


class CampaignEngine:
    def __init__(
        self,
        root: str | Path,
        *,
        store: CampaignStore | None = None,
        corpus: BenchmarkCorpus | None = None,
        trial_executor: TrialExecutor | None = None,
    ) -> None:
        self.root = Path(root)
        self.store = store or CampaignStore(self.root)
        self.corpus = corpus or BenchmarkCorpus(self.store.directory)
        # ``None`` keeps the N6 local thread-pool evaluator. N7 supplies a
        # DistributedTrialExecutor, which submits trials to the N4 queue.
        self.trial_executor = trial_executor

    # ------------------------------------------------------------------ #
    # run / resume
    # ------------------------------------------------------------------ #

    def run(
        self,
        spec: CampaignSpec,
        *,
        resume: bool = True,
        max_rounds: int | None = None,
    ) -> CampaignReport:
        row = self.store.create_or_load(spec, resume=resume)
        campaign_id = str(row["campaign_id"])
        if row["status"] == "completed" and row["stop_reason"] in TERMINAL_STOP_REASONS:
            return self.report(campaign_id)

        self.store.update_campaign(campaign_id, status="running", stop_reason="", rounds=int(row["rounds"]))
        evaluator = Evaluator(spec)
        existing = self.store.trials(campaign_id)
        total = len(existing)
        round_index = max((trial.round_index for trial in existing), default=-1) + 1
        effective_max_rounds = max_rounds if max_rounds is not None else spec.max_rounds

        best_score, last_improvement = self._history(existing, spec.objectives)
        stop_reason = ""
        round_count = int(row["rounds"]) or (round_index if existing else 0)

        while True:
            if total >= spec.budget:
                stop_reason = "budget"
                break
            if effective_max_rounds is not None and round_index >= effective_max_rounds:
                stop_reason = "max_rounds"
                break
            count = min(spec.batch_size, spec.budget - total)
            ranked = score_trials(existing, spec.objectives)
            proposals = propose(
                spec,
                existing,
                round_index=round_index,
                count=count,
                ranked=ranked,
            )
            if not proposals:
                stop_reason = "exhausted"
                break

            try:
                if self.trial_executor is None:
                    results = self._evaluate_batch(evaluator, proposals, spec.parallel)
                else:
                    results = self.trial_executor(spec, proposals)
                    if len(results) != len(proposals):
                        raise InvalidOperationParameters(
                            "trial executor returned a different number of outcomes than candidates"
                        )
            except CampaignTimeout:
                # The batch is not lost: its jobs remain in the queue and the same
                # idempotency keys recover their results when the campaign is resumed.
                self.store.update_campaign(
                    campaign_id,
                    status="paused",
                    stop_reason="worker_timeout",
                    rounds=round_count,
                )
                raise
            except UzenError:
                # A coordinator-side submission failure must not leave the campaign
                # looking active forever; trials already queued remain resumable.
                self.store.update_campaign(
                    campaign_id,
                    status="paused",
                    stop_reason="trial_executor_error",
                    rounds=round_count,
                )
                raise
            inserted = 0
            for sequence, (candidate, outcome) in enumerate(zip(proposals, results)):
                if outcome["status"] == "ok":
                    failures = _constraint_failures(spec, outcome["metrics"])
                    status = "constraint" if failures else "ok"
                else:
                    failures = []
                    status = outcome["status"]
                trial = Trial(
                    candidate=candidate,
                    round_index=round_index,
                    sequence=sequence,
                    status=status,
                    metrics=outcome.get("metrics") or {},
                    error=outcome.get("error", ""),
                    package_fingerprint=outcome.get("package_fingerprint", ""),
                    graph_digest=outcome.get("graph_digest", ""),
                    constraint_failures=tuple(failures),
                )
                if self.store.record_trial(campaign_id, trial):
                    inserted += 1
                    if status == "ok" and spec.write_corpus:
                        self.corpus.append(
                            spec,
                            candidate,
                            trial.metrics,
                            package_fingerprint=trial.package_fingerprint,
                            graph_digest=trial.graph_digest,
                        )
            total += inserted
            round_count = max(round_count, round_index + 1)
            existing = self.store.trials(campaign_id)

            scores = score_trials(existing, spec.objectives)
            current_best = scores[0].score if scores else None
            improved = current_best is not None and (
                best_score is None or current_best > best_score + 1e-12
            )
            if improved:
                best_score = current_best
                last_improvement = round_index

            self.store.update_campaign(
                campaign_id,
                status="running",
                stop_reason="",
                rounds=round_count,
            )
            round_index += 1

            if inserted == 0:
                stop_reason = "exhausted"
                break
            if spec.patience and last_improvement >= 0:
                if round_index - last_improvement > spec.patience:
                    stop_reason = "patience"
                    break
            elif spec.patience and last_improvement < 0:
                if round_index >= spec.patience:
                    stop_reason = "patience"
                    break

        if not stop_reason:
            stop_reason = "budget"
        paused = stop_reason == PAUSE_STOP_REASON
        self.store.update_campaign(
            campaign_id,
            status="paused" if paused else "completed",
            stop_reason=stop_reason,
            rounds=round_count,
        )
        return self.report(campaign_id)

    def _history(
        self, trials: Sequence[Trial], objectives: Sequence[Objective]
    ) -> tuple[float | None, int]:
        if not trials:
            return None, -1
        score_trials(trials, objectives)
        by_round: dict[int, float] = {}
        for trial in trials:
            if trial.score is not None:
                by_round[trial.round_index] = max(
                    by_round.get(trial.round_index, -math.inf), trial.score
                )
        best: float | None = None
        last_improvement = -1
        for round_index in sorted(by_round):
            value = by_round[round_index]
            if best is None or value > best + 1e-12:
                best = value
                last_improvement = round_index
        return best, last_improvement

    def _evaluate_batch(
        self, evaluator: Evaluator, candidates: Sequence[Candidate], parallel: int
    ) -> list[dict[str, Any]]:
        def evaluate_one(candidate: Candidate) -> dict[str, Any]:
            try:
                metrics, fingerprint, graph_digest = evaluator.evaluate(candidate)
                return {
                    "status": "ok",
                    "metrics": metrics,
                    "package_fingerprint": fingerprint,
                    "graph_digest": graph_digest,
                    "error": "",
                }
            except (InvalidOperationParameters, GraphValidationError, ShapeMismatch) as exc:
                return {"status": "invalid", "error": f"{type(exc).__name__}: {exc}"}
            except UzenError as exc:
                return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            except Exception as exc:  # noqa: BLE001 - recorded as a failed trial
                return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

        workers = max(1, min(int(parallel), len(candidates)))
        if workers == 1:
            return [evaluate_one(candidate) for candidate in candidates]
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="uzen-campaign") as pool:
            return list(pool.map(evaluate_one, candidates))

    # ------------------------------------------------------------------ #
    # reports
    # ------------------------------------------------------------------ #

    def report(self, campaign_id: str) -> CampaignReport:
        row = self.store.campaign(campaign_id)
        spec = CampaignSpec.from_dict(json.loads(row["spec_json"]))
        trials = self.store.trials(campaign_id)
        ranked = score_trials(trials, spec.objectives)
        front = pareto_front(trials, spec.objectives)
        best = _trial_entry(ranked[0]) if ranked else None
        return CampaignReport(
            campaign_id=campaign_id,
            name=str(row["name"]),
            mode=str(row["mode"]),
            status=str(row["status"]),
            stop_reason=str(row["stop_reason"]),
            spec_digest=str(row["spec_digest"]),
            seed=int(spec.seed),
            budget=int(spec.budget),
            batch_size=int(spec.batch_size),
            strategy=str(spec.strategy),
            trials=trials,
            pareto=[_trial_entry(trial) for trial in front],
            best=best,
            rounds=int(row["rounds"]),
            corpus_digest=self.corpus.merkle() if spec.write_corpus else "",
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            objectives=[objective.to_dict() for objective in spec.objectives],
        )

    def list_campaigns(self, *, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.store.list_campaigns(limit=limit)
        summaries: list[dict[str, Any]] = []
        for row in rows:
            summaries.append(
                {
                    "campaign_id": row["campaign_id"],
                    "name": row["name"],
                    "mode": row["mode"],
                    "status": row["status"],
                    "stop_reason": row["stop_reason"],
                    "seed": row["seed"],
                    "budget": row["budget"],
                    "rounds": row["rounds"],
                    "spec_digest": row["spec_digest"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            )
        return summaries

    def get(self, campaign_id: str) -> CampaignReport:
        if not campaign_id:
            raise CampaignNotFound("campaign id is required")
        return self.report(campaign_id)

    def verify_corpus(self, digests: Sequence[str] | None = None) -> dict[str, Any]:
        return self.corpus.verify(digests)

    def replay(self, digest: str) -> dict[str, Any]:
        return self.corpus.replay(digest)

    def events(self, campaign_id: str, *, after: int = 0) -> list[dict[str, Any]]:
        return self.store.events(campaign_id, after=after)
