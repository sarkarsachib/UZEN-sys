# P14 — WPL control flow, constraints and postconditions

WPL compiles a declarative weight-engineering program into a typed execution plan.
P14 adds declarative control flow and validation annotations. All of it is
**compile-time**: conditions are evaluated against declared parameters before any
weights are read, so the executed plan is fully analysable and reproducible.

## What was added

* `if` / `else` blocks containing `step` and nested `if` statements.
* `constraint <expr>` declarations, evaluated against parameter values; a false
  constraint raises `ProgramInvalid` (`wpl.invalid_program`) before execution.
* `expect <output> <annotation>…` postconditions, enforced after execution by
  `run_program`:

  | Annotation | Meaning |
  | --- | --- |
  | `finite` | every tensor in the output package is finite |
  | `tensors=N` | the package contains exactly `N` tensors |
  | `max_abs=X` | the largest absolute value across tensors is `≤ X` |
  | `min_abs=X` | the largest absolute value across tensors is `≥ X` |

* A real expression grammar with precedence:
  `or` < `and` < `not` < comparison (`== != < <= > >=`) < `+ -` < `* / %` < unary
  `-`, plus parentheses, string equality, booleans and list literals.
* Typed parameter resolution (`float`, `int`, `bool`, `str`), including negative
  literals and overrides; unknown overrides and missing defaults are refused.

## Semantics

1. **Both branches are validated.** For every `if`, the compiler compiles the
   unselected branch too: operation resolution, parameter typing, dataflow and
   STRICT_PLUS policy effects. A broken dead branch is a compile error rather than a
   latent failure. Only the selected branch's stages enter the plan.
2. **Branch selection is recorded.** `CompiledProgram.decisions` (and the
   `/v1/wpl/describe` report) lists each condition, its evaluated value, the chosen
   branch and the steps it contributed.
3. **Conditions may not depend on tensors.** A condition referencing an input
   package raises `ProgramInvalid`: WPL has no data-dependent interpreter, and
   pretending to branch on values that do not exist at compile time would be a lie.
4. **Dataflow follows the selected path.** A step may reference declared inputs and
   outputs of earlier steps on that path; referencing the other branch's step is an
   error.
5. **Postconditions fail loudly.** `run_program` enforces `expect` after execution
   and raises `ValidationFailed` (`vel.validation_failed`) with the expected and
   observed values; it never returns a package that violated its own contract.

## Example

```wpl
program "aligned_blend" {
    input base : package
    input donor : package
    param factor : float = 0.35
    constraint factor >= 0.0 and factor <= 1.0

    if factor > 0.2 {
        step aligned = w.align.procrustes(donor, base, threshold=0.4)
        step out = w.arithmetic.linear_interpolate(base, aligned, t=factor)
    } else {
        step out = w.arithmetic.scale(base, factor=1.0)
    }

    expect weights finite
    expect weights tensors=1
    output weights = out
}
```

`compile_program(source, param_overrides={"factor": 0.0})` selects the `else`
branch, records that decision, and skips Procrustes entirely.

## Verification

`tests/test_wpl_control_flow.py` (13 tests) covers:

* branch selection changing the executed values, and the execution trace containing
  only the selected stages;
* nested conditions with arithmetic comparisons;
* false constraints refusing before execution, with the offending parameters in the
  error details;
* boolean precedence, `not`, and negative literals;
* dead-branch validation for unknown operations and bad parameter types;
* dataflow errors on the selected branch;
* refusal of tensor-dependent conditions;
* `describe_program` reporting constraints, decisions and branch-tagged steps;
* `finite`, `tensors=N`, `max_abs` postconditions and their failure details;
* unknown expectation annotations and unknown output targets.

The pre-existing WPL tests (`tests/test_wpl_compiler.py`, 13 tests) still pass: the
parser keeps the previous grammar as a subset and `Program.steps` still lists every
step in source order.

## Limits

* Control flow is compile-time only; there are no loops, no runtime branching and no
  user-defined functions.
* Expressions operate on parameters and constants, never on tensors.
* `expect` checks package-level properties, not graph or semantic equivalence; VEL
  remains the place for structural/numerical/functional verification.
* The parser accepts `step name = op(...)`; a bare `step name = other_step` alias is
  not part of the grammar (use an operation or a dedicated stage).
