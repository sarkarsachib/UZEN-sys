"""Typed campaign specifications.

A campaign is a bounded, seeded search over either architecture fields or weight
transformations. The specification is immutable and canonicalizable so the same
document always produces the same campaign digest, candidate order and benchmark
record.

No field here enables training, gradients, fine-tuning or a learned optimizer.
Campaigns evaluate inference-only proxies over the existing deterministic engines.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from uzen.errors import InvalidOperationParameters
from uzen.util.canonical import canonical_bytes

#: Metrics a campaign may optimize or constrain. Every one is computed from a real
#: package/graph forward pass or from the package bytes; none is a fabricated score.
KNOWN_METRICS = (
    "parameters",
    "parameter_bytes",
    "nodes",
    "output_rms",
    "output_diversity",
    "output_bytes",
    "weight_sparsity",
    "quantization_rmse",
    "distance_from_base",
    "activation_agreement",
)

ARCHITECTURE_MODE = "architecture"
WEIGHT_MODE = "weights"
CAMPAIGN_MODES = (ARCHITECTURE_MODE, WEIGHT_MODE)
STRATEGIES = ("grid", "random", "evolution")
WEIGHT_AXES = ("scale", "sparsity", "quantize")


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidOperationParameters(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise InvalidOperationParameters(f"{field} must be finite")
    return number


def _positive_int(value: Any, field: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidOperationParameters(f"{field} must be an integer")
    if value < minimum:
        raise InvalidOperationParameters(f"{field} must be >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class Axis:
    """One search dimension.

    ``int`` and ``float`` axes are bounded (and stepped for grid search); ``choice``
    axes carry an explicit ordered value list. ``default`` must be reachable by the
    declared grid so a resume can always regenerate the same candidates.
    """

    name: str
    kind: str
    low: float | None = None
    high: float | None = None
    step: float | None = None
    values: tuple[Any, ...] = ()
    default: Any = None

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise InvalidOperationParameters("axis name must be a non-empty string")
        if self.kind not in ("int", "float", "choice"):
            raise InvalidOperationParameters(
                f"axis '{self.name}' kind must be int|float|choice"
            )
        if self.kind == "choice":
            if not self.values:
                raise InvalidOperationParameters(
                    f"choice axis '{self.name}' requires values"
                )
            if len(set(self.values)) != len(self.values):
                raise InvalidOperationParameters(
                    f"choice axis '{self.name}' values must be unique"
                )
            if self.default is not None and self.default not in self.values:
                raise InvalidOperationParameters(
                    f"axis '{self.name}' default is not one of its values"
                )
        else:
            if self.low is None or self.high is None:
                raise InvalidOperationParameters(
                    f"axis '{self.name}' requires low and high"
                )
            low = _finite(self.low, f"axis '{self.name}' low")
            high = _finite(self.high, f"axis '{self.name}' high")
            if low > high:
                raise InvalidOperationParameters(
                    f"axis '{self.name}' low exceeds high"
                )
            if self.kind == "int" and (not low.is_integer() or not high.is_integer()):
                raise InvalidOperationParameters(
                    f"int axis '{self.name}' bounds must be integers"
                )
            if self.step is not None:
                step = _finite(self.step, f"axis '{self.name}' step")
                if step <= 0:
                    raise InvalidOperationParameters(
                        f"axis '{self.name}' step must be positive"
                    )
                if self.kind == "int" and not step.is_integer():
                    raise InvalidOperationParameters(
                        f"int axis '{self.name}' step must be an integer"
                    )
            if self.default is not None:
                default = _finite(self.default, f"axis '{self.name}' default")
                if default < low or default > high:
                    raise InvalidOperationParameters(
                        f"axis '{self.name}' default is outside its bounds"
                    )

    def grid_values(self) -> tuple[Any, ...]:
        if self.kind == "choice":
            return self.values
        low = float(self.low)  # type: ignore[arg-type]
        high = float(self.high)  # type: ignore[arg-type]
        if self.kind == "int":
            step = int(self.step or 1)
            start = int(low)
            stop = int(high)
            return tuple(range(start, stop + 1, step))
        if self.step is None:
            raise InvalidOperationParameters(
                f"grid search over float axis '{self.name}' requires a step"
            )
        step = float(self.step)
        count = int(math.floor((high - low) / step + 1e-12)) + 1
        return tuple(low + index * step for index in range(count))

    def sample(self, rng: Any) -> Any:
        if self.kind == "choice":
            return self.values[int(rng.integers(0, len(self.values)))]
        if self.kind == "int":
            step = int(self.step or 1)
            values = list(range(int(self.low), int(self.high) + 1, step))  # type: ignore[arg-type]
            return values[int(rng.integers(0, len(values)))]
        low = float(self.low)  # type: ignore[arg-type]
        high = float(self.high)  # type: ignore[arg-type]
        step = float(self.step) if self.step else None
        if step is None:
            return float(rng.uniform(low, high))
        values = self.grid_values()
        return values[int(rng.integers(0, len(values)))]

    def cast(self, value: Any) -> Any:
        if self.kind == "choice":
            if value not in self.values:
                raise InvalidOperationParameters(
                    f"axis '{self.name}' value {value!r} is not in {self.values}"
                )
            return value
        number = _finite(value, f"axis '{self.name}'")
        if self.kind == "int":
            if not number.is_integer():
                raise InvalidOperationParameters(
                    f"int axis '{self.name}' requires an integer value"
                )
            return int(number)
        return number

    def mutate(self, value: Any, rng: Any) -> Any:
        if self.kind == "choice":
            alternatives = [item for item in self.values if item != value]
            if not alternatives:
                return value
            return alternatives[int(rng.integers(0, len(alternatives)))]
        values = self.grid_values()
        if len(values) <= 1:
            return value
        try:
            index = values.index(value)
        except ValueError:
            return self.sample(rng)
        offset = int(rng.integers(-2, 3))
        if offset == 0:
            offset = 1
        return values[max(0, min(len(values) - 1, index + offset))]

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": self.name, "kind": self.kind}
        if self.kind == "choice":
            payload["values"] = list(self.values)
        else:
            payload.update({"low": self.low, "high": self.high, "step": self.step})
        if self.default is not None:
            payload["default"] = self.default
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Axis":
        if not isinstance(data, Mapping):
            raise InvalidOperationParameters("axis must be an object")
        name = data.get("name")
        kind = data.get("kind")
        if not isinstance(name, str) or not name:
            raise InvalidOperationParameters("axis requires a non-empty name")
        if kind == "choice":
            values = data.get("values")
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise InvalidOperationParameters(
                    f"choice axis '{name}' requires a values list"
                )
            return cls(name=name, kind="choice", values=tuple(values), default=data.get("default"))
        return cls(
            name=name,
            kind=str(kind),
            low=data.get("low"),
            high=data.get("high"),
            step=data.get("step"),
            default=data.get("default"),
        )


@dataclass(frozen=True)
class Objective:
    name: str
    direction: str = "min"
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.name not in KNOWN_METRICS:
            raise InvalidOperationParameters(
                f"unknown campaign objective '{self.name}'; available: "
                f"{', '.join(KNOWN_METRICS)}"
            )
        if self.direction not in ("min", "max"):
            raise InvalidOperationParameters(
                f"objective '{self.name}' direction must be min|max"
            )
        if _finite(self.weight, f"objective '{self.name}' weight") <= 0:
            raise InvalidOperationParameters(
                f"objective '{self.name}' weight must be positive"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "direction": self.direction, "weight": self.weight}


@dataclass(frozen=True)
class CampaignSpec:
    name: str
    family: str
    base_spec: Mapping[str, Any]
    mode: str = ARCHITECTURE_MODE
    axes: tuple[Axis, ...] = ()
    objectives: tuple[Objective, ...] = ()
    constraints: Mapping[str, Mapping[str, float]] | None = None
    seed: int = 0
    budget: int = 16
    batch_size: int = 4
    strategy: str = "random"
    patience: int = 3
    max_rounds: int | None = None
    parallel: int = 1
    evaluation: Mapping[str, Any] | None = None
    campaign_id: str | None = None
    write_corpus: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise InvalidOperationParameters("campaign name must be non-empty")
        if not isinstance(self.family, str) or not self.family:
            raise InvalidOperationParameters("campaign family must be non-empty")
        if not isinstance(self.base_spec, Mapping):
            raise InvalidOperationParameters("campaign base_spec must be an object")
        if self.mode not in CAMPAIGN_MODES:
            raise InvalidOperationParameters(
                f"campaign mode must be {'|'.join(CAMPAIGN_MODES)}"
            )
        if self.strategy not in STRATEGIES:
            raise InvalidOperationParameters(
                f"campaign strategy must be {'|'.join(STRATEGIES)}"
            )
        if not self.axes:
            raise InvalidOperationParameters("campaign requires at least one axis")
        names = [axis.name for axis in self.axes]
        if len(set(names)) != len(names):
            raise InvalidOperationParameters("campaign axis names must be unique")
        if self.mode == ARCHITECTURE_MODE:
            for axis in self.axes:
                if axis.name in WEIGHT_AXES:
                    raise InvalidOperationParameters(
                        f"architecture axis '{axis.name}' collides with a weight "
                        "transform knob"
                    )
                if axis.name not in self.base_spec:
                    raise InvalidOperationParameters(
                        f"architecture axis '{axis.name}' is not a base_spec field; "
                        "typos would otherwise be silently ignored"
                    )
        else:
            for axis in self.axes:
                if axis.name not in WEIGHT_AXES:
                    raise InvalidOperationParameters(
                        f"weight campaign axis '{axis.name}' must be one of "
                        f"{', '.join(WEIGHT_AXES)}"
                    )
        if not self.objectives:
            raise InvalidOperationParameters("campaign requires at least one objective")
        objective_names = [objective.name for objective in self.objectives]
        if len(set(objective_names)) != len(objective_names):
            raise InvalidOperationParameters("campaign objective names must be unique")
        budget = _positive_int(self.budget, "budget")
        batch = _positive_int(self.batch_size, "batch_size")
        if batch > budget:
            raise InvalidOperationParameters("batch_size cannot exceed budget")
        _positive_int(self.patience, "patience")
        _positive_int(self.parallel, "parallel")
        if self.parallel > 64:
            raise InvalidOperationParameters("parallel cannot exceed 64")
        if self.seed < 0:
            raise InvalidOperationParameters("seed must be non-negative")
        if self.max_rounds is not None:
            _positive_int(self.max_rounds, "max_rounds")
        if self.campaign_id is not None and (
            not isinstance(self.campaign_id, str) or not self.campaign_id.strip()
        ):
            raise InvalidOperationParameters("campaign_id must be a non-empty string")
        constraints = dict(self.constraints or {})
        for metric, bound in constraints.items():
            if metric not in KNOWN_METRICS:
                raise InvalidOperationParameters(
                    f"constraint metric '{metric}' is not one of {', '.join(KNOWN_METRICS)}"
                )
            if not isinstance(bound, Mapping) or not bound:
                raise InvalidOperationParameters(
                    f"constraint '{metric}' must be an object with min and/or max"
                )
            for kind, value in bound.items():
                if kind not in ("min", "max"):
                    raise InvalidOperationParameters(
                        f"constraint '{metric}' supports min and max only"
                    )
                _finite(value, f"constraint '{metric}' {kind}")
        evaluation = dict(self.evaluation or {})
        for key, default in (("batch", 2), ("seq", 8), ("init_seed", 0), ("input_seed", 0)):
            evaluation.setdefault(key, default)
        _positive_int(evaluation["batch"], "evaluation.batch")
        _positive_int(evaluation["seq"], "evaluation.seq")
        if not isinstance(evaluation["init_seed"], int) or evaluation["init_seed"] < 0:
            raise InvalidOperationParameters("evaluation.init_seed must be a non-negative integer")
        if not isinstance(evaluation["input_seed"], int) or evaluation["input_seed"] < 0:
            raise InvalidOperationParameters("evaluation.input_seed must be a non-negative integer")
        object.__setattr__(self, "base_spec", dict(self.base_spec))
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "evaluation", evaluation)

    @property
    def candidate_id(self) -> str:
        if self.campaign_id:
            return self.campaign_id
        return f"{self.name.lower().replace(' ', '-')[:40]}-{self.digest()[:12]}"

    def digest(self) -> str:
        return hashlib.sha256(canonical_bytes(self._canonical())).hexdigest()

    def _canonical(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "base_spec": dict(self.base_spec),
            "mode": self.mode,
            "axes": [axis.to_dict() for axis in self.axes],
            "objectives": [objective.to_dict() for objective in self.objectives],
            "constraints": dict(self.constraints or {}),
            "seed": self.seed,
            "budget": self.budget,
            "batch_size": self.batch_size,
            "strategy": self.strategy,
            "patience": self.patience,
            "max_rounds": self.max_rounds,
            "parallel": self.parallel,
            "evaluation": dict(self.evaluation or {}),
            "write_corpus": self.write_corpus,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._canonical()
        if self.campaign_id:
            payload["campaign_id"] = self.campaign_id
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CampaignSpec":
        if not isinstance(data, Mapping):
            raise InvalidOperationParameters("campaign spec must be an object")
        axes = data.get("axes")
        objectives = data.get("objectives")
        if not isinstance(axes, Sequence) or isinstance(axes, (str, bytes)):
            raise InvalidOperationParameters("campaign spec requires an axes list")
        if not isinstance(objectives, Sequence) or isinstance(objectives, (str, bytes)):
            raise InvalidOperationParameters("campaign spec requires an objectives list")
        parsed_objectives = []
        for item in objectives:
            if not isinstance(item, Mapping) or "name" not in item:
                raise InvalidOperationParameters(
                    "each campaign objective must be an object with a name"
                )
            parsed_objectives.append(
                Objective(
                    name=str(item["name"]),
                    direction=str(item.get("direction", "min")),
                    weight=float(item.get("weight", 1.0)),
                )
            )
        return cls(
            name=str(data.get("name", "")),
            family=str(data.get("family", "")),
            base_spec=dict(data.get("base_spec") or {}),
            mode=str(data.get("mode", ARCHITECTURE_MODE)),
            axes=tuple(Axis.from_dict(axis) for axis in axes),
            objectives=tuple(parsed_objectives),
            constraints=data.get("constraints") or None,
            seed=int(data.get("seed", 0)),
            budget=int(data.get("budget", 16)),
            batch_size=int(data.get("batch_size", 4)),
            strategy=str(data.get("strategy", "random")),
            patience=int(data.get("patience", 3)),
            max_rounds=int(data["max_rounds"]) if data.get("max_rounds") is not None else None,
            parallel=int(data.get("parallel", 1)),
            evaluation=data.get("evaluation") or None,
            campaign_id=data.get("campaign_id"),
            write_corpus=bool(data.get("write_corpus", True)),
        )
