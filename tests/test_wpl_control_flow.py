"""P14: WPL compile-time control flow, constraints and postconditions."""

from __future__ import annotations

import numpy as np
import pytest

from uzen.core.nws.tensors import WeightPackage
from uzen.core.wpl.compiler import compile_program, describe_program, run_program
from uzen.errors import ProgramInvalid, ValidationFailed


PROGRAM = """
program "control_flow" {
    input base : package
    param high : bool = true
    param factor : float = 0.5
    constraint factor >= 0.0 and factor <= 4.0
    if high {
        step doubled = scale(base, factor=2.0)
        if factor > 1.0 {
            step out = scale(doubled, factor=factor)
        } else {
            step out = scale(doubled, factor=1.0)
        }
    } else {
        step out = scale(base, factor=0.25)
    }
    expect weights finite
    expect weights tensors=1
    output weights = out
}
"""


def _base(value: float = 1.0) -> WeightPackage:
    return WeightPackage.from_arrays({"w": np.full((2, 2), value, dtype=np.float32)}, name="base")


def _values(result) -> np.ndarray:
    # Pipeline results are keyed by producing step; this program has one output.
    return result.package().tensors["w"]


def test_if_else_selects_a_branch_and_executes_only_that_one() -> None:
    high, _ = run_program(PROGRAM, {"base": _base(1.0)})
    np.testing.assert_allclose(_values(high), np.full((2, 2), 2.0, dtype=np.float32))

    low, compiled = run_program(PROGRAM, {"base": _base(1.0)}, param_overrides={"high": False})
    np.testing.assert_allclose(_values(low), np.full((2, 2), 0.25, dtype=np.float32))
    assert [decision["branch"] for decision in compiled.decisions] == ["else"]
    assert compiled.decisions[0]["steps"] == ["out"]

    np.testing.assert_allclose(high.operations()[0]["params"]["factor"], 2.0)


def test_nested_conditions_evaluate_arithmetic_comparisons() -> None:
    result, compiled = run_program(PROGRAM, {"base": _base(1.0)}, param_overrides={"factor": 2.0})
    np.testing.assert_allclose(_values(result), np.full((2, 2), 4.0, dtype=np.float32))
    assert [decision["branch"] for decision in compiled.decisions] == ["then", "then"]

    result, compiled = run_program(PROGRAM, {"base": _base(1.0)}, param_overrides={"factor": 0.5})
    np.testing.assert_allclose(_values(result), np.full((2, 2), 2.0, dtype=np.float32))
    assert [decision["branch"] for decision in compiled.decisions] == ["then", "else"]


def test_false_constraint_refuses_before_any_execution() -> None:
    with pytest.raises(ProgramInvalid) as exc:
        compile_program(PROGRAM, param_overrides={"factor": 5.0})
    assert "constraint" in str(exc.value)
    assert "factor" in str(exc.value)
    assert exc.value.details["parameters"]["factor"] == 5.0


def test_boolean_operators_and_negative_literals() -> None:
    source = """
    program "logic" {
        input base : package
        param factor : float = 0.5
        param offset : float = -0.25
        constraint not (factor > 4.0) and factor + offset >= 0.0
        step out = scale(base, factor=factor)
        output weights = out
    }
    """
    compiled = compile_program(source)
    assert compiled.constraints[0]["value"] is True

    with pytest.raises(ProgramInvalid):
        compile_program(source, param_overrides={"factor": 4.5})


def test_unselected_branch_is_still_validated() -> None:
    source = """
    program "bad_else" {
        input base : package
        param high : bool = true
        if high {
            step out = scale(base, factor=2.0)
        } else {
            step out = no_such_operation(base)
        }
        output weights = out
    }
    """
    with pytest.raises(ProgramInvalid) as exc:
        compile_program(source)
    assert "else" in str(exc.value)
    assert "unknown operation" in str(exc.value)


def test_unselected_branch_parameter_typing_is_checked() -> None:
    source = """
    program "bad_param" {
        input base : package
        param high : bool = true
        if high {
            step out = scale(base, factor=2.0)
        } else {
            step out = scale(base, factor="not-a-number")
        }
        output weights = out
    }
    """
    with pytest.raises(ProgramInvalid) as exc:
        compile_program(source)
    assert "else" in str(exc.value)


def test_selected_branch_dataflow_is_enforced() -> None:
    source = """
    program "bad_ref" {
        input base : package
        param high : bool = true
        if high {
            step out = scale(missing, factor=2.0)
        } else {
            step out = scale(base, factor=1.0)
        }
        output weights = out
    }
    """
    with pytest.raises(Exception) as exc:
        compile_program(source)
    assert "missing" in str(exc.value)


def test_condition_may_not_depend_on_tensors() -> None:
    source = """
    program "tensor_cond" {
        input base : package
        param factor : float = 1.0
        if base > 0 {
            step out = scale(base, factor=factor)
        }
        output weights = out
    }
    """
    with pytest.raises(ProgramInvalid) as exc:
        compile_program(source)
    assert "not a declared parameter" in str(exc.value)


def test_describe_program_reports_constraints_and_selected_branches() -> None:
    report = describe_program(PROGRAM)
    assert report["constraints"] == [
        {"expression": "((factor >= 0.0) and (factor <= 4.0))", "value": True}
    ]
    assert [decision["branch"] for decision in report["control_flow"]] == ["then", "else"]
    assert [step["name"] for step in report["steps"]] == ["doubled", "out"]
    # "doubled" is in the outer then-branch; "out" is in the inner else-branch.
    assert [step["branch"] for step in report["steps"]] == ["then", "else"]
    assert report["expectations"] == ["expect weights finite", "expect weights tensors=1"]

    alternative = describe_program(PROGRAM, param_overrides={"high": False})
    assert [step["name"] for step in alternative["steps"]] == ["out"]
    assert alternative["steps"][0]["branch"] == "else"


def test_non_finite_output_fails_the_finite_expectation() -> None:
    from uzen.core.wcomp.compiler import PipelineResult
    from uzen.core.wpl.compiler import enforce_expectations

    compiled = compile_program(PROGRAM)
    poisoned = WeightPackage.from_arrays({"w": np.full((2, 2), np.inf, dtype=np.float32)})
    result = PipelineResult(outputs={"out": poisoned}, trace=[], policy=[], seeds={})
    with pytest.raises(ValidationFailed) as exc:
        enforce_expectations(result, compiled.expectations)
    assert "finite" in str(exc.value)
    assert exc.value.details["tensors"] == ["w"]

    # The engine itself refuses to emit non-finite tensors, so a NaN never reaches
    # the postcondition through a normal run; both layers are checked here.
    with pytest.raises(ValueError):
        run_program(PROGRAM, {"base": _base(np.nan)})


def test_max_abs_postcondition_reports_the_observed_value() -> None:
    source = """
    program "bounded" {
        input base : package
        expect weights finite
        expect weights max_abs=1.5
        step out = scale(base, factor=2.0)
        output weights = out
    }
    """
    with pytest.raises(ValidationFailed) as exc:
        run_program(source, {"base": _base(1.0)})
    assert exc.value.details["observed"] == pytest.approx(2.0)

    ok = """
    program "bounded_ok" {
        input base : package
        expect weights max_abs=1.5
        step out = scale(base, factor=1.0)
        output weights = out
    }
    """
    result, _ = run_program(ok, {"base": _base(1.0)})
    np.testing.assert_allclose(_values(result), np.ones((2, 2), dtype=np.float32))


def test_unknown_expectation_annotation_and_target_are_rejected() -> None:
    bad_annotation = """
    program "bad_expect" {
        input base : package
        expect weights wobbly
        step out = scale(base, factor=1.0)
        output weights = out
    }
    """
    with pytest.raises(ProgramInvalid) as exc:
        compile_program(bad_annotation)
    assert "unsupported annotations" in str(exc.value)

    bad_target = """
    program "bad_target" {
        input base : package
        expect other finite
        step out = scale(base, factor=1.0)
        output weights = out
    }
    """
    with pytest.raises(ProgramInvalid) as exc:
        compile_program(bad_target)
    assert "does not name a program output" in str(exc.value)


def test_tensor_count_postcondition() -> None:
    source = """
    program "counted" {
        input base : package
        expect weights tensors=2
        step out = scale(base, factor=1.0)
        output weights = out
    }
    """
    with pytest.raises(ValidationFailed) as exc:
        run_program(source, {"base": _base(1.0)})
    assert exc.value.details == {"expected": 2, "actual": 1}
