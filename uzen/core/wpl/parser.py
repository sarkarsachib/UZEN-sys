"""WPL: the Weight Programming Language.

WPL is a small, real, declaratively-checked language for expressing weight
engineering programs. It is compiled to an execution plan (WComp) and executed by
the operation registry (NWPC) under STRICT_PLUS.

Grammar (EBNF-ish)::

    program     := "program" STRING? "{" declaration* "}"
    declaration := input | param | constraint | expect | output | step | if
    input       := "input" IDENT ":" "package"
    param       := "param" IDENT ":" TYPE ("=" literal)?
    constraint  := "constraint" expr
    expect      := "expect" IDENT annotation+
    annotation  := IDENT ("=" literal)?
    let         := "step" IDENT "=" call
    if          := "if" expr "{" (step | if)* "}" ("else" "{" (step | if)* "}")?
    output      := "output" (IDENT "=")? reference
    call        := opcode "(" arglist? ")"
    opcode      := IDENT
    arglist     := arg ("," arg)*
    arg         := IDENT "=" literal | reference
    literal     := NUMBER | STRING | "true" | "false" | list
    reference   := IDENT
    expr        := or_expr
    or_expr     := and_expr ("or" and_expr)*
    and_expr    := not_expr ("and" not_expr)*
    not_expr    := "not" not_expr | comparison
    comparison  := sum (("==" | "!=" | "<" | "<=" | ">" | ">=") sum)?
    sum         := product (("+" | "-") product)*
    product     := unary (("*" | "/" | "%") unary)*
    unary       := "-" unary | primary
    primary     := NUMBER | STRING | "true" | "false" | IDENT | "(" expr ")" | list

Example:::

    program "aligned_blend" {
      input base : package
      input donor : package
      param factor : float = 0.35
      param method : str = "procrustes"
      constraint factor >= 0.0 and factor <= 1.0
      step aligned  = w.align.procrustes(donor, base, threshold=0.4)
      if factor > 0.2 {
        step scaled = w.arithmetic.scale(aligned, factor=factor)
        step combined = w.arithmetic.delta(base, scaled)
      } else {
        step combined = w.arithmetic.scale(base, factor=1.0)
      }
      expect weights finite
      output weights = combined
    }

Control flow is **compile-time**: conditions are evaluated against declared
parameters, every branch is parsed and validated (operation resolution, parameter
typing, dataflow, policy effects), and only the selected branch enters the plan. A
condition that depends on runtime tensors is not expressible — WPL does not execute a
data-dependent interpreter, which keeps the plan fully analysable before any weights
are touched.

The lexer, parser and compiler are implemented here; there is no hidden Python
``eval`` — parameters are literals or declared program parameters only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from uzen.errors import ProgramInvalid

KEYWORDS = {
    "program",
    "input",
    "param",
    "step",
    "let",
    "output",
    "package",
    "float",
    "int",
    "bool",
    "str",
    "if",
    "else",
    "constraint",
    "expect",
    "and",
    "or",
    "not",
}
COMPARISONS = {"==", "!=", "<", "<=", ">", ">="}
OPERATORS = {"+", "-", "*", "/", "%"}
_PUNCTUATION = set("{}()=,:[]<>!+-*/%")
_SIGN_CONTEXT = {":", "=", ",", "(", "[", "{", "<", ">", "==", "!=", "<=", ">=", "+", "-", "*", "/", "%"}


# --------------------------------------------------------------------------- #
# lexer
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Token:
    kind: str  # IDENT | NUMBER | STRING | PUNCT | EOF
    value: str
    line: int
    column: int

    def __str__(self) -> str:
        return f"{self.kind}({self.value!r})@{self.line}:{self.column}"


def _sign_allowed(tokens: Sequence[Token]) -> bool:
    """A ``+``/``-`` starts a signed number only where a value may begin."""
    if not tokens:
        return True
    last = tokens[-1]
    if last.kind == "PUNCT":
        return last.value in _SIGN_CONTEXT
    if last.kind == "IDENT":
        return last.value in {"and", "or", "not", "constraint", "if", "else", "expect"}
    return False


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    i, line, col = 0, 1, 1
    n = len(source)
    while i < n:
        ch = source[i]
        if ch in " \t\r":
            i += 1
            col += 1
            continue
        if ch == "\n":
            i += 1
            line += 1
            col = 1
            continue
        if ch == "#":
            while i < n and source[i] != "\n":
                i += 1
                col += 1
            continue
        if ch == '"':
            start_line, start_col = line, col
            i += 1
            col += 1
            buf: list[str] = []
            while i < n and source[i] != '"':
                if source[i] == "\\" and i + 1 < n:
                    nxt = source[i + 1]
                    buf.append({"n": "\n", "t": "\t", '"': '"', "\\": "\\"}.get(nxt, nxt))
                    i += 2
                    col += 2
                    continue
                if source[i] == "\n":
                    raise ProgramInvalid(f"unterminated string at line {start_line}")
                buf.append(source[i])
                i += 1
                col += 1
            if i >= n:
                raise ProgramInvalid(f"unterminated string starting at line {start_line}")
            i += 1
            col += 1
            tokens.append(Token("STRING", "".join(buf), start_line, start_col))
            continue
        if ch.isdigit() or (
            ch in "+-"
            and _sign_allowed(tokens)
            and i + 1 < n
            and (source[i + 1].isdigit() or source[i + 1] == ".")
        ):
            start = i
            start_col = col
            if source[i] in "+-":
                i += 1
                col += 1
            while i < n and (source[i].isdigit() or source[i] in ".eE"):
                if source[i] in "eE" and i + 1 < n and source[i + 1] in "+-":
                    i += 1
                    col += 1
                i += 1
                col += 1
            tokens.append(Token("NUMBER", source[start:i], line, start_col))
            continue
        if ch.isalpha() or ch == "_":
            start = i
            start_col = col
            while i < n and (source[i].isalnum() or source[i] in "_."):
                i += 1
                col += 1
            tokens.append(Token("IDENT", source[start:i], line, start_col))
            continue
        two = source[i : i + 2]
        if two in ("==", "!=", "<=", ">="):
            tokens.append(Token("PUNCT", two, line, col))
            i += 2
            col += 2
            continue
        if ch in _PUNCTUATION:
            tokens.append(Token("PUNCT", ch, line, col))
            i += 1
            col += 1
            continue
        raise ProgramInvalid(f"unexpected character {ch!r} at line {line}, column {col}")
    tokens.append(Token("EOF", "", line, col))
    return tokens


# --------------------------------------------------------------------------- #
# AST
# --------------------------------------------------------------------------- #


@dataclass
class Call:
    code: str
    args: list[str] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Literal:
    value: Any

    def describe(self) -> str:
        return repr(self.value)


@dataclass(frozen=True)
class Reference:
    name: str

    def describe(self) -> str:
        return self.name


@dataclass(frozen=True)
class Unary:
    op: str
    operand: "Expression"

    def describe(self) -> str:
        return f"{self.op}{self.operand.describe()}"


@dataclass(frozen=True)
class Binary:
    op: str
    left: "Expression"
    right: "Expression"

    def describe(self) -> str:
        return f"({self.left.describe()} {self.op} {self.right.describe()})"


Expression = Literal | Reference | Unary | Binary


@dataclass
class Expectation:
    """A postcondition checked on the produced package after execution."""

    target: str
    checks: dict[str, Any] = field(default_factory=dict)
    line: int = 0
    #: Pipeline result key (the producing step) when ``target`` is an output alias.
    pipeline_output: str | None = None

    def describe(self) -> str:
        parts = [
            key if value is True else f"{key}={value!r}"
            for key, value in sorted(self.checks.items())
        ]
        return f"expect {self.target} {' '.join(parts)}".strip()


@dataclass
class Step:
    name: str
    call: Call
    line: int = 0


@dataclass
class If:
    condition: Expression
    then_body: list["Statement"] = field(default_factory=list)
    else_body: list["Statement"] = field(default_factory=list)
    line: int = 0

    def describe(self) -> str:
        return f"if {self.condition.describe()}"


Statement = Step | If


@dataclass
class Program:
    name: str
    inputs: list[str]
    params: dict[str, tuple[str, Any]]
    steps: list[tuple[str, Call]]
    outputs: dict[str, str]
    body: list[Statement] = field(default_factory=list)
    constraints: list[Expression] = field(default_factory=list)
    expectations: list[Expectation] = field(default_factory=list)

    def all_step_names(self) -> list[str]:
        return [name for name, _call in self.steps]


# --------------------------------------------------------------------------- #
# expressions
# --------------------------------------------------------------------------- #


def _require_number(value: Any, op: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProgramInvalid(f"operator '{op}' needs numeric operands, got {value!r}")
    return float(value)


def evaluate_expression(expression: Expression, values: Mapping[str, Any]) -> Any:
    """Evaluate a declaration-time expression (parameters only, no tensors)."""
    if isinstance(expression, Literal):
        return expression.value
    if isinstance(expression, Reference):
        if expression.name not in values:
            raise ProgramInvalid(
                f"expression references '{expression.name}', which is not a declared "
                "parameter; control flow may only depend on parameters and constants"
            )
        return values[expression.name]
    if isinstance(expression, Unary):
        value = evaluate_expression(expression.operand, values)
        if expression.op == "not":
            return not bool(value)
        number = _require_number(value, expression.op)
        return -number
    if isinstance(expression, Binary):
        op = expression.op
        if op == "and":
            left = evaluate_expression(expression.left, values)
            return bool(left) and bool(evaluate_expression(expression.right, values))
        if op == "or":
            left = evaluate_expression(expression.left, values)
            return bool(left) or bool(evaluate_expression(expression.right, values))
        left = evaluate_expression(expression.left, values)
        right = evaluate_expression(expression.right, values)
        if op in COMPARISONS:
            if isinstance(left, str) or isinstance(right, str):
                if op == "==":
                    return left == right
                if op == "!=":
                    return left != right
                raise ProgramInvalid(f"operator '{op}' cannot compare strings")
            a = _require_number(left, op)
            b = _require_number(right, op)
            return {
                "==": a == b,
                "!=": a != b,
                "<": a < b,
                "<=": a <= b,
                ">": a > b,
                ">=": a >= b,
            }[op]
        if op in ("+", "-"):
            if isinstance(left, str) and isinstance(right, str) and op == "+":
                return left + right
        a = _require_number(left, op)
        b = _require_number(right, op)
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            if b == 0:
                raise ProgramInvalid("division by zero in a WPL expression")
            return a / b
        if op == "%":
            if b == 0:
                raise ProgramInvalid("modulo by zero in a WPL expression")
            return a % b
        raise ProgramInvalid(f"unknown operator '{op}'")
    raise ProgramInvalid(f"cannot evaluate expression {expression!r}")


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #


class Parser:
    def __init__(self, tokens: Sequence[Token]) -> None:
        self.tokens = list(tokens)
        self.pos = 0

    # -- helpers ------------------------------------------------------- #

    def peek(self, offset: int = 0) -> Token:
        idx = self.pos + offset
        return self.tokens[idx] if idx < len(self.tokens) else self.tokens[-1]

    def next(self) -> Token:
        tok = self.peek()
        self.pos += 1
        return tok

    def expect(self, kind: str, value: str | None = None) -> Token:
        tok = self.peek()
        if tok.kind != kind or (value is not None and tok.value != value):
            expected = f"{kind}({value})" if value else kind
            raise ProgramInvalid(
                f"expected {expected} but found {tok} at line {tok.line}, column {tok.column}"
            )
        return self.next()

    def at_punct(self, value: str) -> bool:
        return self.peek().kind == "PUNCT" and self.peek().value == value

    def at_keyword(self, value: str) -> bool:
        return self.peek().kind == "IDENT" and self.peek().value == value

    # -- grammar ------------------------------------------------------- #

    def parse(self) -> Program:
        self.expect("IDENT", "program")
        name = "program"
        if self.peek().kind == "STRING":
            name = self.next().value
        self.expect("PUNCT", "{")
        inputs: list[str] = []
        params: dict[str, tuple[str, Any]] = {}
        steps: list[tuple[str, Call]] = []
        outputs: dict[str, str] = {}
        constraints: list[Expression] = []
        expectations: list[Expectation] = []
        body: list[Statement] = []
        while not self.at_punct("}"):
            if self.peek().kind == "EOF":
                raise ProgramInvalid(f"unterminated program '{name}': missing '}}'")
            if self.at_keyword("input"):
                inputs.append(self._parse_input())
            elif self.at_keyword("param"):
                pname, ptype, pdefault = self._parse_param()
                params[pname] = (ptype, pdefault)
            elif self.at_keyword("step") or self.at_keyword("let"):
                step = self._parse_step()
                steps.append((step.name, step.call))
                body.append(step)
            elif self.at_keyword("if"):
                node = self._parse_if(steps)
                body.append(node)
            elif self.at_keyword("constraint"):
                constraints.append(self._parse_constraint())
            elif self.at_keyword("expect"):
                expectations.append(self._parse_expect())
            elif self.at_keyword("output"):
                oname, ref = self._parse_output()
                outputs[oname] = ref
            else:
                tok = self.peek()
                raise ProgramInvalid(f"unexpected token {tok} in program body at line {tok.line}")
        self.expect("PUNCT", "}")
        self.expect("EOF")
        if not steps:
            raise ProgramInvalid("program contains no steps")
        if not outputs:
            outputs = {steps[-1][0]: steps[-1][0]}
        return Program(
            name=name,
            inputs=inputs,
            params=params,
            steps=steps,
            outputs=outputs,
            body=body,
            constraints=constraints,
            expectations=expectations,
        )

    def _parse_input(self) -> str:
        self.expect("IDENT", "input")
        name = self.expect("IDENT").value
        self.expect("PUNCT", ":")
        type_tok = self.expect("IDENT")
        if type_tok.value != "package":
            raise ProgramInvalid(
                f"input '{name}' must be declared ': package' (WPL transports parameter "
                f"packages, not raw tensors), found '{type_tok.value}'"
            )
        return name

    def _parse_param(self) -> tuple[str, str, Any]:
        self.expect("IDENT", "param")
        name = self.expect("IDENT").value
        self.expect("PUNCT", ":")
        ptype = self.expect("IDENT").value
        if ptype not in ("float", "int", "bool", "str"):
            raise ProgramInvalid(f"unsupported parameter type '{ptype}'")
        default = None
        if self.at_punct("="):
            self.next()
            default = self._parse_literal()
        return name, ptype, default

    def _parse_step(self) -> Step:
        line = self.next().line  # 'step' or 'let'
        name = self.expect("IDENT").value
        self.expect("PUNCT", "=")
        call = self._parse_call()
        return Step(name=name, call=call, line=line)

    def _parse_constraint(self) -> Expression:
        self.expect("IDENT", "constraint")
        return self._parse_expression()

    def _parse_expect(self) -> Expectation:
        line = self.expect("IDENT", "expect").line
        target = self.expect("IDENT").value
        checks: dict[str, Any] = {}
        while True:
            tok = self.peek()
            if tok.kind == "IDENT" and tok.value not in KEYWORDS:
                key = self.next().value
                if self.at_punct("="):
                    self.next()
                    checks[key] = self._parse_literal()
                else:
                    checks[key] = True
                continue
            break
        if not checks:
            raise ProgramInvalid(
                f"'expect {target}' needs at least one annotation (finite, tensors=N, max_abs=X)"
            )
        return Expectation(target=target, checks=checks, line=line)

    def _parse_if(self, steps: list[tuple[str, Call]]) -> If:
        line = self.expect("IDENT", "if").line
        condition = self._parse_expression()
        self.expect("PUNCT", "{")
        then_body = self._parse_statements(steps)
        self.expect("PUNCT", "}")
        else_body: list[Statement] = []
        if self.at_keyword("else"):
            self.next()
            self.expect("PUNCT", "{")
            else_body = self._parse_statements(steps)
            self.expect("PUNCT", "}")
        return If(condition=condition, then_body=then_body, else_body=else_body, line=line)

    def _parse_statements(self, steps: list[tuple[str, Call]]) -> list[Statement]:
        body: list[Statement] = []
        while not self.at_punct("}"):
            if self.peek().kind == "EOF":
                raise ProgramInvalid("unterminated 'if' block: missing '}'")
            if self.at_keyword("step") or self.at_keyword("let"):
                step = self._parse_step()
                steps.append((step.name, step.call))
                body.append(step)
            elif self.at_keyword("if"):
                body.append(self._parse_if(steps))
            else:
                tok = self.peek()
                raise ProgramInvalid(
                    f"only 'step' and nested 'if' are allowed inside an if block; found {tok} "
                    f"at line {tok.line}"
                )
        return body

    def _parse_output(self) -> tuple[str, str]:
        self.expect("IDENT", "output")
        if self.peek().kind == "IDENT" and self.peek(1).kind == "PUNCT" and self.peek(1).value == "=":
            name = self.next().value
            self.expect("PUNCT", "=")
            ref = self.expect("IDENT").value
            return name, ref
        ref = self.expect("IDENT").value
        return ref, ref

    def _parse_call(self) -> Call:
        code_parts = [self.expect("IDENT").value]
        while self.at_punct("."):
            self.next()
            code_parts.append(self.expect("IDENT").value)
        code = ".".join(code_parts)
        self.expect("PUNCT", "(")
        args: list[str] = []
        kwargs: dict[str, Any] = {}
        if not self.at_punct(")"):
            while True:
                if (
                    self.peek().kind == "IDENT"
                    and self.peek(1).kind == "PUNCT"
                    and self.peek(1).value == "="
                ):
                    key = self.next().value
                    self.expect("PUNCT", "=")
                    kwargs[key] = self._parse_literal()
                else:
                    args.append(self.expect("IDENT").value)
                if self.at_punct(","):
                    self.next()
                    continue
                break
        self.expect("PUNCT", ")")
        return Call(code=code, args=args, kwargs=kwargs)

    # -- expressions --------------------------------------------------- #

    def _parse_expression(self) -> Expression:
        return self._parse_or()

    def _parse_or(self) -> Expression:
        node = self._parse_and()
        while self.at_keyword("or"):
            self.next()
            node = Binary("or", node, self._parse_and())
        return node

    def _parse_and(self) -> Expression:
        node = self._parse_not()
        while self.at_keyword("and"):
            self.next()
            node = Binary("and", node, self._parse_not())
        return node

    def _parse_not(self) -> Expression:
        if self.at_keyword("not"):
            self.next()
            return Unary("not", self._parse_not())
        return self._parse_comparison()

    def _parse_comparison(self) -> Expression:
        node = self._parse_sum()
        tok = self.peek()
        if tok.kind == "PUNCT" and tok.value in COMPARISONS:
            self.next()
            node = Binary(tok.value, node, self._parse_sum())
        return node

    def _parse_sum(self) -> Expression:
        node = self._parse_product()
        while self.peek().kind == "PUNCT" and self.peek().value in ("+", "-"):
            op = self.next().value
            node = Binary(op, node, self._parse_product())
        return node

    def _parse_product(self) -> Expression:
        node = self._parse_unary()
        while self.peek().kind == "PUNCT" and self.peek().value in ("*", "/", "%"):
            op = self.next().value
            node = Binary(op, node, self._parse_unary())
        return node

    def _parse_unary(self) -> Expression:
        if self.at_punct("-"):
            self.next()
            return Unary("-", self._parse_unary())
        return self._parse_primary()

    def _parse_primary(self) -> Expression:
        tok = self.peek()
        if tok.kind == "NUMBER":
            self.next()
            text = tok.value
            try:
                value: Any = (
                    int(text) if ("." not in text and "e" not in text.lower()) else float(text)
                )
            except ValueError as exc:
                raise ProgramInvalid(f"invalid number literal {text!r}") from exc
            return Literal(value)
        if tok.kind == "STRING":
            self.next()
            return Literal(tok.value)
        if self.at_punct("("):
            self.next()
            node = self._parse_expression()
            self.expect("PUNCT", ")")
            return node
        if self.at_punct("["):
            return Literal(self._parse_literal())
        if tok.kind == "IDENT":
            self.next()
            if tok.value in ("true", "false"):
                return Literal(tok.value == "true")
            if tok.value in KEYWORDS and tok.value not in ("and", "or", "not"):
                raise ProgramInvalid(
                    f"keyword '{tok.value}' cannot appear in an expression at line {tok.line}"
                )
            return Reference(tok.value)
        raise ProgramInvalid(f"expected an expression, found {tok}")

    def _parse_literal(self) -> Any:
        tok = self.peek()
        if tok.kind == "NUMBER":
            self.next()
            text = tok.value
            try:
                return int(text) if ("." not in text and "e" not in text.lower()) else float(text)
            except ValueError as exc:
                raise ProgramInvalid(f"invalid number literal {text!r}") from exc
        if tok.kind == "STRING":
            self.next()
            return tok.value
        if tok.kind == "IDENT":
            self.next()
            if tok.value in ("true", "false"):
                return tok.value == "true"
            # Identifier literals are symbolic references to declared program params;
            # the compiler resolves them and rejects anything undeclared.
            return tok.value
        if self.at_punct("["):
            self.next()
            items: list[Any] = []
            if not self.at_punct("]"):
                while True:
                    items.append(self._parse_literal())
                    if self.at_punct(","):
                        self.next()
                        continue
                    break
            self.expect("PUNCT", "]")
            return items
        raise ProgramInvalid(f"expected a literal, found {tok}")


def parse_program(source: str) -> Program:
    if not isinstance(source, str) or not source.strip():
        raise ProgramInvalid("empty WPL source")
    return Parser(tokenize(source)).parse()
