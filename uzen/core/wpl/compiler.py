"""WPL compiler: program AST -> typed execution plan -> executed artifacts.

The compiler performs five checks before anything runs:

1. **Parameter typing** — every declared parameter is resolved to a typed value,
   with unknown overrides rejected.
2. **Constraints** — ``constraint`` expressions are evaluated against the parameter
   values; a false constraint refuses the program before execution.
3. **Control flow** — ``if``/``else`` conditions are evaluated against parameters
   only. **Both** branches are compiled (operation resolution, parameter typing,
   dataflow and policy effects); only the selected branch enters the plan. A
   condition that depends on tensors is rejected by the expression evaluator.
4. **Reference/dataflow checking** — a step may only reference declared inputs or
   outputs of earlier steps on the selected path.
5. **Policy authorization** — each stage's declared effect is authorized for the
   program's scope *before* execution (STRICT_PLUS dry run).

``expect`` annotations become postconditions on the produced package and are
enforced by :func:`run_program` after execution; a failed postcondition raises
:class:`~uzen.errors.ValidationFailed` and never reports success.

Execution is delegated to :class:`uzen.core.wcomp.compiler.CompiledPipeline`, which
calls the operation registry. There is no interpreter that evaluates arbitrary code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from uzen.core.nwpc.registry import OperationRegistry, default_registry
from uzen.core.wcomp.compiler import (
    CompiledPipeline,
    CompilationReport,
    PipelineResult,
    Stage,
    compile_stages,
)
from uzen.core.wpl.parser import (
    Binary,
    Call,
    Expression,
    Expectation,
    If,
    Literal,
    Program,
    Reference,
    Step,
    Unary,
    evaluate_expression,
    parse_program,
)
from uzen.errors import ProgramInvalid, ValidationFailed
from uzen.policies.strict_plus.effects import TaskScope
from uzen.policies.strict_plus.engine import StrictPlusEngine, default_engine

TYPE_MAP = {"float": float, "int": int, "bool": bool, "str": str}
EXPECTATION_KEYS = frozenset({"finite", "tensors", "max_abs", "min_abs"})


@dataclass
class CompiledProgram:
    program: Program
    pipeline: CompiledPipeline
    report: CompilationReport
    decisions: list[dict[str, Any]] = field(default_factory=list)
    constraints: list[dict[str, Any]] = field(default_factory=list)
    expectations: list[Expectation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "program": self.program.name,
            "inputs": list(self.program.inputs),
            "params": {k: {"type": v[0], "default": v[1]} for k, v in self.program.params.items()},
            "outputs": dict(self.program.outputs),
            "compilation": self.report.to_dict(),
            "effects": self.pipeline.effects(),
            "control_flow": list(self.decisions),
            "constraints": list(self.constraints),
            "expectations": [expectation.describe() for expectation in self.expectations],
        }


def resolve_operation_code(code: str, registry: OperationRegistry) -> str:
    """Resolve a full code or a unique shorthand into a registered operation code."""
    if code.startswith("w."):
        registry.get(code)  # raises if unknown
        return code
    matches = [spec.code for spec in registry.all() if spec.code.split(".")[-1] == code]
    if not matches:
        matches = [spec.code for spec in registry.all() if code in spec.code]
    if not matches:
        raise ProgramInvalid(
            f"unknown operation '{code}' in WPL program; see GET /v1/operations for the catalogue"
        )
    if len(matches) > 1:
        raise ProgramInvalid(
            f"operation shorthand '{code}' is ambiguous: {matches}; use the full code"
        )
    return matches[0]


def _resolve_value(value: Any, params: Mapping[str, tuple[str, Any]]) -> Any:
    if isinstance(value, str) and value in params and not value.startswith("w."):
        return params[value][1]
    return value


def _param_values(params: Mapping[str, tuple[str, Any]]) -> dict[str, Any]:
    return {name: value for name, (_type, value) in params.items()}


def _stages_for_body(
    body: Sequence[Any],
    *,
    params: Mapping[str, tuple[str, Any]],
    registry: OperationRegistry,
    scope: TaskScope,
    engine: StrictPlusEngine,
    available: set[str],
    decisions: list[dict[str, Any]],
    validate_unselected: bool = True,
) -> tuple[list[Stage], set[str]]:
    """Flatten one body into stages, evaluating compile-time branches."""
    stages: list[Stage] = []
    values = _param_values(params)
    for statement in body:
        if isinstance(statement, Step):
            code = resolve_operation_code(statement.call.code, registry)
            kwargs = {k: _resolve_value(v, params) for k, v in statement.call.kwargs.items()}
            stages.append(
                Stage(name=statement.name, code=code, args=list(statement.call.args), params=kwargs)
            )
            available.add(statement.name)
            continue
        if isinstance(statement, If):
            condition_value = bool(evaluate_expression(statement.condition, values))
            chosen = statement.then_body if condition_value else statement.else_body
            decision_index = len(decisions)
            unselected_label = "else" if condition_value else "then"
            unselected = statement.else_body if condition_value else statement.then_body
            if validate_unselected and unselected:
                try:
                    branch_stages, _branch_available = _stages_for_body(
                        unselected,
                        params=params,
                        registry=registry,
                        scope=scope,
                        engine=engine,
                        available=set(available),
                        decisions=[],
                        validate_unselected=False,
                    )
                    _pipeline, branch_report = compile_stages(
                        branch_stages,
                        scope=scope,
                        registry=registry,
                        engine=engine,
                        available_inputs=sorted(available),
                        outputs=[],
                    )
                except ProgramInvalid as exc:
                    raise ProgramInvalid(
                        f"the '{unselected_label}' branch of 'if "
                        f"{statement.condition.describe()}' does not compile even though it "
                        f"is not selected: {exc.message}",
                        branch=unselected_label,
                    ) from exc
                if not branch_report.ok:
                    raise ProgramInvalid(
                        f"the '{unselected_label}' branch of 'if "
                        f"{statement.condition.describe()}' does not compile even though it "
                        f"is not selected: {branch_report.to_dict()}",
                        branch=unselected_label,
                    )
            chosen_stages, available = _stages_for_body(
                chosen,
                params=params,
                registry=registry,
                scope=scope,
                engine=engine,
                available=set(available),
                decisions=decisions,
                validate_unselected=True,
            )
            stages.extend(chosen_stages)
            decisions.insert(
                decision_index,
                {
                    "condition": statement.condition.describe(),
                    "value": condition_value,
                    "branch": "then" if condition_value else "else",
                    "steps": [stage.name for stage in chosen_stages],
                },
            )
            continue
        raise ProgramInvalid(f"unsupported WPL statement {statement!r}")
    return stages, available


def _resolve_expectations(
    program: Program, expectations: Sequence[Expectation]
) -> list[Expectation]:
    """Map each expectation target onto a concrete pipeline output name."""
    resolved: list[Expectation] = []
    references = {value: key for key, value in program.outputs.items()}
    for expectation in expectations:
        unknown = sorted(set(expectation.checks) - EXPECTATION_KEYS)
        if unknown:
            raise ProgramInvalid(
                f"'expect {expectation.target}' has unsupported annotations {unknown}; "
                f"supported: {sorted(EXPECTATION_KEYS)}"
            )
        target = expectation.target
        if target in program.outputs:
            output = program.outputs[target]
        elif target in references:
            output = target
        else:
            raise ProgramInvalid(
                f"'expect {target}' does not name a program output; outputs: "
                f"{sorted(program.outputs)}"
            )
        resolved.append(
            Expectation(
                target=target,
                checks=dict(expectation.checks),
                line=expectation.line,
                pipeline_output=output,
            )
        )
    return resolved


def compile_program(
    source: str,
    *,
    scope: TaskScope = TaskScope.WEIGHT_MODIFICATION,
    registry: OperationRegistry | None = None,
    engine: StrictPlusEngine | None = None,
    param_overrides: Mapping[str, Any] | None = None,
) -> CompiledProgram:
    """Parse and compile WPL source into an executable pipeline."""
    program = parse_program(source)
    registry = registry or default_registry()
    engine = engine or default_engine()

    params: dict[str, tuple[str, Any]] = {}
    overrides = dict(param_overrides or {})
    unknown_overrides = set(overrides) - set(program.params)
    if unknown_overrides:
        raise ProgramInvalid(
            f"override for undeclared program parameters: {sorted(unknown_overrides)}"
        )
    for name, (ptype, default) in program.params.items():
        value = overrides.get(name, default)
        if value is None:
            raise ProgramInvalid(
                f"program parameter '{name}' has no default and no override was supplied"
            )
        expected = TYPE_MAP[ptype]
        if expected is bool:
            ok = isinstance(value, bool)
        elif expected is int:
            ok = isinstance(value, int) and not isinstance(value, bool)
        elif expected is float:
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ProgramInvalid(
                    f"program parameter '{name}' must be float, got {value!r}"
                ) from exc
        else:
            ok = isinstance(value, str)
        if not ok:
            raise ProgramInvalid(
                f"program parameter '{name}' must be {ptype}, got {type(value).__name__}"
            )
        params[name] = (ptype, value)

    values = _param_values(params)
    constraint_records: list[dict[str, Any]] = []
    for constraint in program.constraints:
        result = bool(evaluate_expression(constraint, values))
        constraint_records.append({"expression": constraint.describe(), "value": result})
        if not result:
            raise ProgramInvalid(
                f"constraint '{constraint.describe()}' is false for {values}",
                constraint=constraint.describe(),
                parameters=values,
            )

    decisions: list[dict[str, Any]] = []
    stages, _available = _stages_for_body(
        program.body,
        params=params,
        registry=registry,
        scope=scope,
        engine=engine,
        available=set(program.inputs),
        decisions=decisions,
    )

    pipeline, report = compile_stages(
        stages,
        scope=scope,
        registry=registry,
        engine=engine,
        available_inputs=program.inputs,
        outputs=list(program.outputs.values()),
        metadata={
            "program": program.name,
            "outputs": sorted(set(program.outputs.values())),
            "control_flow": decisions,
            "constraints": constraint_records,
        },
    )
    report.require()
    return CompiledProgram(
        program=program,
        pipeline=pipeline,
        report=report,
        decisions=decisions,
        constraints=constraint_records,
        expectations=_resolve_expectations(program, program.expectations),
    )


def enforce_expectations(result: PipelineResult, expectations: Sequence[Expectation]) -> None:
    """Check ``expect`` postconditions against the produced packages."""
    for expectation in expectations:
        package = result.package(expectation.pipeline_output or expectation.target)
        tensors = list(package.tensors.values())
        for key, value in expectation.checks.items():
            if key == "finite":
                if value is False:
                    continue
                bad = [name for name, tensor in package.tensors.items() if not np.isfinite(tensor).all()]
                if bad:
                    raise ValidationFailed(
                        f"expect {expectation.target} finite: tensors {bad} contain non-finite values",
                        expected="finite",
                        tensors=bad,
                    )
            elif key == "tensors":
                count = int(value)
                if len(tensors) != count:
                    raise ValidationFailed(
                        f"expect {expectation.target} tensors={count}: package has {len(tensors)}",
                        expected=count,
                        actual=len(tensors),
                    )
            elif key in ("max_abs", "min_abs"):
                limit = float(value)
                if not tensors:
                    raise ValidationFailed(
                        f"expect {expectation.target} {key}={limit}: package has no tensors"
                    )
                maximum = max(float(np.max(np.abs(tensor))) for tensor in tensors)
                if key == "max_abs" and maximum > limit:
                    raise ValidationFailed(
                        f"expect {expectation.target} max_abs={limit}: observed {maximum}",
                        expected=limit,
                        observed=maximum,
                    )
                if key == "min_abs" and maximum < limit:
                    raise ValidationFailed(
                        f"expect {expectation.target} min_abs={limit}: observed maximum {maximum}",
                        expected=limit,
                        observed=maximum,
                    )
            else:  # pragma: no cover - compiler validates annotation keys
                raise ValidationFailed(f"unsupported expectation '{key}'")


def run_program(
    source: str,
    inputs: Mapping[str, Any],
    *,
    scope: TaskScope = TaskScope.WEIGHT_MODIFICATION,
    seed: int | str | None = None,
    registry: OperationRegistry | None = None,
    engine: StrictPlusEngine | None = None,
    param_overrides: Mapping[str, Any] | None = None,
    job_id: str | None = None,
) -> tuple[PipelineResult, CompiledProgram]:
    compiled = compile_program(
        source,
        scope=scope,
        registry=registry,
        engine=engine,
        param_overrides=param_overrides,
    )
    result = compiled.pipeline.execute(inputs, seed=seed, scope=scope, job_id=job_id)
    enforce_expectations(result, compiled.expectations)
    return result, compiled


def describe_program(
    source: str,
    *,
    scope: TaskScope | None = None,
    registry: OperationRegistry | None = None,
    param_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Static analysis of a program without executing it.

    ``if`` conditions are evaluated when their parameters have defaults (or an
    override); otherwise both branches are listed with ``"branch": null``.
    """
    program = parse_program(source)
    registry = registry or default_registry()
    overrides = dict(param_overrides or {})
    values = {
        name: overrides.get(name, default) for name, (_type, default) in program.params.items()
    }
    stages: list[dict[str, Any]] = []
    effects: list[str] = []
    decisions: list[dict[str, Any]] = []
    constraint_records: list[dict[str, Any]] = []

    for constraint in program.constraints:
        try:
            constraint_records.append(
                {"expression": constraint.describe(), "value": bool(evaluate_expression(constraint, values))}
            )
        except ProgramInvalid:
            constraint_records.append({"expression": constraint.describe(), "value": None})

    def walk(body: Sequence[Any], *, branch: str | None, depth: int) -> None:
        for statement in body:
            if isinstance(statement, Step):
                code = resolve_operation_code(statement.call.code, registry)
                spec = registry.get(code)
                effects.append(spec.effect.value)
                stages.append(
                    {
                        "name": statement.name,
                        "code": code,
                        "title": spec.title,
                        "effect": spec.effect.value,
                        "maturity": spec.maturity.value,
                        "args": list(statement.call.args),
                        "params": dict(statement.call.kwargs),
                        "assumptions": list(spec.assumptions),
                        "branch": branch,
                        "depth": depth,
                    }
                )
                continue
            assert isinstance(statement, If)
            try:
                condition: bool | None = bool(evaluate_expression(statement.condition, values))
            except ProgramInvalid:
                condition = None
            decisions.append(
                {
                    "condition": statement.condition.describe(),
                    "value": condition,
                    "branch": None if condition is None else ("then" if condition else "else"),
                }
            )
            if condition is None:
                walk(statement.then_body, branch="then", depth=depth + 1)
                walk(statement.else_body, branch="else", depth=depth + 1)
            else:
                walk(
                    statement.then_body if condition else statement.else_body,
                    branch="then" if condition else "else",
                    depth=depth + 1,
                )

    walk(program.body, branch=None, depth=0)
    return {
        "program": program.name,
        "inputs": program.inputs,
        "params": {k: {"type": v[0], "default": v[1]} for k, v in program.params.items()},
        "steps": stages,
        "outputs": program.outputs,
        "effects": sorted(set(effects)),
        "constraints": constraint_records,
        "control_flow": decisions,
        "expectations": [expectation.describe() for expectation in program.expectations],
        "scope_requested": scope.value if scope else None,
        "executable": True,
    }
