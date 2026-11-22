"""Deterministic candidate generation for campaigns.

Grid proposals are a pure function of the flat grid index. Random proposals are a
pure function of ``(seed, round_index)``. Evolution proposals are a pure function
of the ranked completed trials and ``(seed, round_index)``. None of them depend on
wall-clock time, process scheduling or dictionary order, so stopping a campaign at
any batch boundary and resuming it reproduces the same candidate sequence.
"""

from __future__ import annotations

import itertools
from typing import Any, Iterable, Sequence

import numpy as np

from uzen.core.campaign.models import Candidate
from uzen.core.campaign.spec import (
    ARCHITECTURE_MODE,
    Axis,
    CampaignSpec,
    WEIGHT_MODE,
)


class GridExhausted(Exception):
    """The finite grid has no candidate at the requested index."""


def _candidate_from_values(spec: CampaignSpec, values: Sequence[Any]) -> Candidate:
    resolved: dict[str, Any] = {}
    for axis, value in zip(spec.axes, values):
        resolved[axis.name] = axis.cast(value)
    if spec.mode == ARCHITECTURE_MODE:
        config = {"spec": {**dict(spec.base_spec), **resolved}, "transforms": {}}
    elif spec.mode == WEIGHT_MODE:
        config = {"spec": dict(spec.base_spec), "transforms": resolved}
    else:  # validated by CampaignSpec; defensive for direct construction
        raise ValueError(f"unsupported campaign mode '{spec.mode}'")
    return Candidate.build(spec.mode, spec.family, tuple(resolved.values()), config)


def grid_candidate(spec: CampaignSpec, index: int) -> Candidate:
    if index < 0:
        raise GridExhausted("negative grid index")
    axis_values = [axis.grid_values() for axis in spec.axes]
    total = 1
    for values in axis_values:
        total *= len(values)
    if index >= total:
        raise GridExhausted(f"grid index {index} exceeds size {total}")
    mixed: list[Any] = []
    remainder = index
    for values in reversed(axis_values):
        remainder, digit = divmod(remainder, len(values))
        mixed.append(values[digit])
    mixed.reverse()
    return _candidate_from_values(spec, mixed)


def grid_size(spec: CampaignSpec) -> int:
    total = 1
    for axis in spec.axes:
        total *= len(axis.grid_values())
    return total


def _completed_digests(completed: Iterable[Any]) -> set[str]:
    digests: set[str] = set()
    for item in completed:
        candidate = getattr(item, "candidate", None)
        if candidate is not None:
            digests.add(candidate.digest)
        elif isinstance(item, str):
            digests.add(item)
        elif hasattr(item, "digest"):
            # A bare Candidate is accepted too; the previous implementation silently
            # ignored it, which could re-propose an already evaluated candidate.
            digests.add(str(item.digest))
    return digests


def _random_values(spec: CampaignSpec, rng: np.random.Generator) -> tuple[Any, ...]:
    return tuple(axis.sample(rng) for axis in spec.axes)


def _mutate(
    spec: CampaignSpec,
    parent: Candidate,
    rng: np.random.Generator,
) -> tuple[Any, ...]:
    values = list(parent.values)
    axes = list(spec.axes)
    # One or two coordinates keeps children near their parent without collapsing
    # the whole population onto a single point.
    count = 1 if len(axes) == 1 else int(rng.integers(1, min(3, len(axes) + 1)))
    chosen = rng.choice(len(axes), size=count, replace=False)
    for index in chosen:
        values[int(index)] = axes[int(index)].mutate(values[int(index)], rng)
    return tuple(values)


def _crossover(
    spec: CampaignSpec,
    left: Candidate,
    right: Candidate,
    rng: np.random.Generator,
) -> tuple[Any, ...]:
    if len(spec.axes) == 1:
        return tuple(left.values)
    cut = int(rng.integers(1, len(spec.axes)))
    return tuple(left.values[:cut] + right.values[cut:])


def propose(
    spec: CampaignSpec,
    completed: Sequence[Any],
    *,
    round_index: int,
    count: int,
    ranked: Sequence[Any] = (),
) -> list[Candidate]:
    """Return at most ``count`` fresh candidates for this round.

    ``ranked`` is an optional best-first list of successful trials used as
    evolutionary parents. It is ignored by grid and random search.
    """
    if count <= 0:
        return []
    done = _completed_digests(completed)
    proposals: list[Candidate] = []
    seen: set[str] = set()

    def add(candidate: Candidate) -> bool:
        if candidate.digest in done or candidate.digest in seen:
            return False
        seen.add(candidate.digest)
        proposals.append(candidate)
        return len(proposals) >= count

    if spec.strategy == "grid":
        start = round_index * spec.batch_size
        for index in range(start, start + count):
            try:
                candidate = grid_candidate(spec, index)
            except GridExhausted:
                break
            if add(candidate):
                break
        return proposals

    rng = np.random.default_rng(spec.seed + round_index * 1_000_003)

    if spec.strategy == "random" or not ranked:
        attempts = 0
        limit = max(64, count * 64)
        while len(proposals) < count and attempts < limit:
            attempts += 1
            candidate = _candidate_from_values(spec, _random_values(spec, rng))
            if add(candidate):
                break
        return proposals

    # Evolution: elites produce mutated children and crossover; a small immigrant
    # share keeps the population from freezing when every axis is discrete.
    elites = [trial.candidate for trial in ranked[: max(2, spec.batch_size)]]
    immigrant_share = 0.25
    attempts = 0
    limit = max(128, count * 128)
    while len(proposals) < count and attempts < limit:
        attempts += 1
        if rng.random() < immigrant_share or len(elites) == 1:
            values = _random_values(spec, rng)
        else:
            parent = elites[int(rng.integers(0, len(elites)))]
            if rng.random() < 0.25:
                other = elites[int(rng.integers(0, len(elites)))]
                values = _crossover(spec, parent, other, rng)
            else:
                values = _mutate(spec, parent, rng)
        candidate = _candidate_from_values(spec, values)
        if add(candidate):
            break
    return proposals


def axis_summary(spec: CampaignSpec) -> list[dict[str, Any]]:
    """Human-readable axis domain, used in reports and docs."""
    summary: list[dict[str, Any]] = []
    for axis in spec.axes:
        entry: dict[str, Any] = {"name": axis.name, "kind": axis.kind}
        if axis.kind == "choice":
            entry["values"] = list(axis.values)
        else:
            entry.update({"low": axis.low, "high": axis.high, "step": axis.step})
        summary.append(entry)
    return summary


def total_grid_size(spec: CampaignSpec) -> int:
    return int(np.prod([len(axis.grid_values()) for axis in spec.axes], dtype=np.int64))


def iter_grid(spec: CampaignSpec) -> Iterable[Candidate]:
    for index in itertools.count():
        try:
            yield grid_candidate(spec, index)
        except GridExhausted:
            return
