"""A constrained, fail-closed expression language for pipeline conditions.

Pipeline content (``condition:``, ``if:``, ``while:`` ...) is data, not trusted
code, but it used to reach :func:`eval`. The worst instance combined four
separate problems::

    eval_condition = condition
    for var_name, value in variables.items():           # 1. textual substitution
        eval_condition = eval_condition.replace(var_name, repr(value))
    if any(op in eval_condition for op in ["import", "exec", "eval", "__"]):
        return True                                     # 2. fail-OPEN on "unsafe"
    return eval(eval_condition)                         # 3. full builtins

1. Substituting variable names as *text* corrupts unrelated substrings: a
   variable named ``a`` rewrites the ``a`` inside ``max``.
2. A blocklist over strings is bypassable (``getattr(x, chr(95)+...)``), and
   returning ``True`` when it *does* fire means the guard grants execution.
3. ``eval`` without ``__builtins__`` exposes ``__import__``, ``open``, etc.

This module replaces all of that with an allowlist over the parsed AST. Names
are resolved from an explicit context mapping -- never from globals -- and
anything not explicitly permitted raises :class:`ExpressionError`.

Design rule: **fail closed.** An expression that is malformed, unsupported, or
references something undefined does not execute the guarded step.
"""

from __future__ import annotations

import ast
import logging
import operator
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)

__all__ = [
    "ExpressionError",
    "evaluate_expression",
    "evaluate_condition",
    "SAFE_FUNCTIONS",
]


class ExpressionError(ValueError):
    """Raised when an expression is malformed or uses a disallowed construct."""


# Operators are mapped explicitly rather than executed by name, so only these
# exact behaviors are reachable.
_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.Not: operator.not_,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_COMPARE_OPS: dict[type[ast.cmpop], Callable[[Any, Any], Any]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}

#: Callables a pipeline author may invoke. Everything here is pure and cannot
#: reach the filesystem, network, or interpreter internals.
SAFE_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sorted": sorted,
    "sum": sum,
    "any": any,
    "all": all,
}

#: Exponentiation is excluded on purpose: `10**10**10` is a cheap way to hang
#: the process. Power is not needed to express a pipeline condition.

# `**` is deliberately absent from _BIN_OPS (see above).

_MAX_EXPRESSION_LENGTH = 4096


class _Evaluator(ast.NodeVisitor):
    """Walks a parsed expression, evaluating only allowlisted node types."""

    def __init__(self, context: Mapping[str, Any]) -> None:
        self.context = context

    # Any node type without an explicit visitor is rejected here.
    def generic_visit(self, node: ast.AST) -> Any:
        raise ExpressionError(
            f"unsupported expression element: {type(node).__name__}"
        )

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in self.context:
            return self.context[node.id]
        if node.id in SAFE_FUNCTIONS:
            return SAFE_FUNCTIONS[node.id]
        raise ExpressionError(f"undefined name: {node.id!r}")

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        # Short-circuits, matching Python semantics.
        if isinstance(node.op, ast.And):
            result: Any = True
            for value in node.values:
                result = self.visit(value)
                if not result:
                    return result
            return result
        result = False
        for value in node.values:
            result = self.visit(value)
            if result:
                return result
        return result

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        func = _UNARY_OPS.get(type(node.op))
        if func is None:
            raise ExpressionError(
                f"unsupported unary operator: {type(node.op).__name__}"
            )
        return func(self.visit(node.operand))

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        func = _BIN_OPS.get(type(node.op))
        if func is None:
            raise ExpressionError(
                f"unsupported operator: {type(node.op).__name__}"
            )
        return func(self.visit(node.left), self.visit(node.right))

    def visit_Compare(self, node: ast.Compare) -> Any:
        left = self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            func = _COMPARE_OPS.get(type(op))
            if func is None:
                raise ExpressionError(
                    f"unsupported comparison: {type(op).__name__}"
                )
            right = self.visit(comparator)
            if not func(left, right):
                return False
            left = right  # chained comparison: a < b < c
        return True

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        # Attribute access is the usual escape hatch out of a sandbox
        # (`x.__class__.__bases__`...). Private/dunder names are refused.
        if node.attr.startswith("_"):
            raise ExpressionError(f"access to private attribute {node.attr!r} denied")
        value = self.visit(node.value)
        try:
            return getattr(value, node.attr)
        except AttributeError as exc:
            raise ExpressionError(str(exc)) from exc

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        value = self.visit(node.value)
        key = self.visit(node.slice)
        try:
            return value[key]
        except (KeyError, IndexError, TypeError) as exc:
            raise ExpressionError(f"invalid subscript: {exc}") from exc

    def visit_Slice(self, node: ast.Slice) -> Any:
        return slice(
            self.visit(node.lower) if node.lower else None,
            self.visit(node.upper) if node.upper else None,
            self.visit(node.step) if node.step else None,
        )

    def visit_Call(self, node: ast.Call) -> Any:
        # Only bare names from SAFE_FUNCTIONS are callable. Calling an
        # arbitrary expression result (`obj.method()`) is refused, since that
        # would reach code this module cannot vet.
        if not isinstance(node.func, ast.Name):
            raise ExpressionError("only direct calls to built-in helpers are allowed")
        if node.func.id not in SAFE_FUNCTIONS:
            raise ExpressionError(f"call to {node.func.id!r} is not allowed")
        if node.keywords:
            raise ExpressionError("keyword arguments are not supported")
        args = [self.visit(arg) for arg in node.args]
        return SAFE_FUNCTIONS[node.func.id](*args)

    def visit_List(self, node: ast.List) -> Any:
        return [self.visit(e) for e in node.elts]

    def visit_Tuple(self, node: ast.Tuple) -> Any:
        return tuple(self.visit(e) for e in node.elts)

    def visit_Set(self, node: ast.Set) -> Any:
        return {self.visit(e) for e in node.elts}

    def visit_Dict(self, node: ast.Dict) -> Any:
        return {
            self.visit(k) if k is not None else None: self.visit(v)
            for k, v in zip(node.keys, node.values)
        }

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        return self.visit(node.body) if self.visit(node.test) else self.visit(node.orelse)


def evaluate_expression(expression: str, context: Mapping[str, Any] | None = None) -> Any:
    """Evaluate ``expression`` against ``context`` and return its value.

    Args:
        expression: A single Python-syntax expression. Statements, assignments,
            comprehensions, lambdas, imports and f-strings are not supported.
        context: Names available to the expression. Nothing else is in scope.

    Raises:
        ExpressionError: if the expression is malformed, too long, or uses any
            construct outside the allowlist.
    """
    if not isinstance(expression, str):
        raise ExpressionError(f"expression must be a string, got {type(expression).__name__}")

    stripped = expression.strip()
    if not stripped:
        raise ExpressionError("empty expression")
    if len(stripped) > _MAX_EXPRESSION_LENGTH:
        raise ExpressionError(
            f"expression exceeds {_MAX_EXPRESSION_LENGTH} characters"
        )

    try:
        tree = ast.parse(stripped, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"invalid syntax: {exc.msg}") from exc

    return _Evaluator(context or {}).visit(tree)


def evaluate_condition(
    condition: str,
    context: Mapping[str, Any] | None = None,
    *,
    default: bool = False,
) -> bool:
    """Evaluate ``condition`` as a boolean, **failing closed**.

    A condition that cannot be evaluated returns ``default`` (``False``), so an
    unparseable or malicious condition does *not* run the step it guards. The
    previous behavior returned ``True`` on both the "unsafe input" branch and
    the generic exception handler, which meant a broken condition executed the
    step it was supposed to gate.

    Args:
        condition: The expression to evaluate.
        context: Names available to the expression.
        default: Value returned when evaluation fails. Keep this ``False``
            unless a caller has a specific, documented reason.
    """
    try:
        return bool(evaluate_expression(condition, context))
    except ExpressionError as exc:
        logger.warning(
            "Condition %r rejected (%s); treating as %s.", condition, exc, default
        )
        return default
    except Exception as exc:  # noqa: BLE001 - a condition must never propagate
        logger.warning(
            "Condition %r failed to evaluate (%s: %s); treating as %s.",
            condition,
            type(exc).__name__,
            exc,
            default,
        )
        return default
