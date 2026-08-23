"""Fail-closed expression evaluator for plan guards and bindings (issue #492).

Guards, branch conditions and input templates must never reach ``eval``. This
module compiles a strictly whitelisted expression subset via :mod:`ast` and
evaluates it against a plain mapping scope. Anything not on the whitelist —
calls, lambdas, comprehensions, attribute access beyond shallow dotted names —
raises :class:`ExpressionError` at compile time. A condition that cannot be
evaluated fails closed.
"""

from __future__ import annotations

import ast
import operator
from typing import Any, Mapping

_MAX_ATTR_DEPTH = 4


class ExpressionError(Exception):
    """Raised for any disallowed construct or failed evaluation."""


_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_CMPOPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Attribute,
    ast.Subscript,
    ast.Tuple,
    ast.List,
    ast.And,
    ast.Or,
    ast.Not,
    ast.USub,
    ast.UAdd,
    ast.In,
    ast.NotIn,
) + tuple(_BINOPS) + tuple(_CMPOPS)


class ExprObj:
    """A compiled, validated expression ready for :func:`evaluate`."""

    __slots__ = ("source", "_tree")

    def __init__(self, source: str, tree: ast.Expression) -> None:
        self.source = source
        self._tree = tree

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ExprObj({self.source!r})"


def _validate(node: ast.AST, depth: int = 0) -> None:
    if not isinstance(node, _ALLOWED_NODES):
        raise ExpressionError(f"disallowed syntax: {type(node).__name__}")
    if isinstance(node, ast.Attribute):
        if not node.attr.isidentifier() or node.attr.startswith("_"):
            raise ExpressionError("invalid attribute name")
        if depth >= _MAX_ATTR_DEPTH:
            raise ExpressionError("attribute chain too deep")
    for child in ast.iter_child_nodes(node):
        _validate(child, depth + 1)


def compile_expr(src: str) -> ExprObj:
    """Compile *src*, raising :class:`ExpressionError` on anything unlisted."""
    try:
        tree = ast.parse(src.strip(), mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"syntax error in {src!r}: {exc.msg}") from exc
    _validate(tree)
    return ExprObj(src.strip(), tree)


def _resolve_name(name: str, scope: Mapping[str, Any]) -> Any:
    parts = name.split(".")
    cur: Any = scope
    for part in parts:
        if isinstance(cur, Mapping):
            if part not in cur:
                raise ExpressionError(f"unknown name {name!r}")
            cur = cur[part]
        else:
            try:
                cur = getattr(cur, part)
            except AttributeError as exc:
                raise ExpressionError(f"unknown name {name!r}") from exc
    return cur


def _eval(node: ast.AST, scope: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body, scope)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return _resolve_name(node.id, scope)
    if isinstance(node, ast.Attribute):
        base = _eval(node.value, scope)
        holder: Any = base
        if isinstance(holder, Mapping):
            if node.attr not in holder:
                raise ExpressionError(f"unknown key {node.attr!r}")
            return holder[node.attr]
        if not hasattr(holder, node.attr):
            raise ExpressionError(f"unknown attribute {node.attr!r}")
        return getattr(holder, node.attr)
    if isinstance(node, ast.Subscript):
        base = _eval(node.value, scope)
        idx = node.slice
        if isinstance(idx, ast.Slice):
            raise ExpressionError("slices are not allowed")
        key = _eval(idx, scope)
        if isinstance(key, bool) or not isinstance(key, (int, str)):
            raise ExpressionError("subscript index must be int or str")
        try:
            return base[key]  # type: ignore[index]
        except (KeyError, IndexError, TypeError) as exc:
            raise ExpressionError(f"bad subscript {key!r}") from exc
    if isinstance(node, ast.Tuple):
        return tuple(_eval(e, scope) for e in node.elts)
    if isinstance(node, ast.List):
        return [_eval(e, scope) for e in node.elts]
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return not _eval(node.operand, scope)
        val = _eval(node.operand, scope)
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise ExpressionError("unary +/- needs a number")
        return -val if isinstance(node.op, ast.USub) else +val
    if isinstance(node, ast.BoolOp):
        results = [_eval(v, scope) for v in node.values]
        if isinstance(node.op, ast.And):
            return all(results)
        return any(results)
    if isinstance(node, ast.Compare):
        left = _eval(node.left, scope)
        for op, comp in zip(node.ops, node.comparators):
            right = _eval(comp, scope)
            if isinstance(op, (ast.In, ast.NotIn)):
                try:
                    contains = left in right
                except TypeError as exc:
                    raise ExpressionError("bad `in` operand") from exc
                ok = contains if isinstance(op, ast.In) else not contains
            else:
                fn = _CMPOPS[type(op)]
                try:
                    ok = bool(fn(left, right))
                except TypeError as exc:
                    raise ExpressionError(f"bad comparison {left!r} / {right!r}") from exc
            if not ok:
                return False
            left = right
        return True
    if isinstance(node, ast.BinOp):
        left = _eval(node.left, scope)
        right = _eval(node.right, scope)
        if isinstance(left, bool) or isinstance(right, bool) or not all(
            isinstance(v, (int, float)) for v in (left, right)
        ):
            raise ExpressionError("arithmetic needs numbers")
        try:
            return _BINOPS[type(op := node.op)](left, right)
        except ZeroDivisionError as exc:
            raise ExpressionError("division by zero") from exc
    raise ExpressionError(f"unsupported node {type(node).__name__}")  # pragma: no cover


def evaluate(expr: "ExprObj | str", scope: Mapping[str, Any]) -> Any:
    """Evaluate a compiled expression (or source string) against *scope*."""
    obj = expr if isinstance(expr, ExprObj) else compile_expr(expr)
    return _eval(obj._tree, scope)
