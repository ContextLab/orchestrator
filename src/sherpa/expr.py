"""Fail-closed expression evaluator for plan guards and bindings (issue #492).

Guards, branch conditions and input templates must never reach ``eval``. This
module compiles a strictly whitelisted expression subset via :mod:`ast` and
evaluates it against a plain mapping scope. Anything not on the whitelist —
calls, lambdas, comprehensions, attribute access beyond shallow dotted names —
raises :class:`ExpressionError` at compile time. A condition that cannot be
evaluated fails closed.

Deliberate semantic choices
---------------------------
``and`` / ``or`` follow **full Python semantics**: they short-circuit, and they
return the deciding *operand* rather than a coerced ``bool``. This module binds
``{{ ... }}`` input templates (:func:`sherpa.capabilities.resolve_inputs`) as
well as guards, so ``inputs.path or 'README.md'`` must yield the string. Guard
call sites consume the result by truthiness (or coerce with ``bool``), so the
richer return value costs them nothing. Short-circuiting is load-bearing: the
``Branch``/``While`` call sites in :mod:`sherpa.kernel` do not catch
:class:`ExpressionError`, so an eagerly evaluated right-hand operand
(``'k' in d and d['k'] > 1``) would abort an entire run.

Attribute access is restricted to **plain mappings** — ``obj.attr`` is a key
lookup on a :class:`~collections.abc.Mapping`, never a :func:`getattr`. A
deny-list of dunder names is not sufficient: frame traversal
(``gen.gi_frame.f_builtins``) uses names with no leading underscore and reaches
``eval``/``exec``/``open``. Every scope the kernel builds holds plain data
(``inputs``, ``{node_id: {"result": ...}}``), so mapping-only traversal is
fail-closed without losing any real usage.
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


def _attr_chain_length(node: ast.Attribute) -> int:
    """Length of the dotted chain ending at *node* (``a.b.c`` -> 3)."""
    length = 0
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        length += 1
        current = current.value
    return length


def _validate(node: ast.AST) -> None:
    if not isinstance(node, _ALLOWED_NODES):
        raise ExpressionError(f"disallowed syntax: {type(node).__name__}")
    if isinstance(node, ast.Attribute):
        if not node.attr.isidentifier() or node.attr.startswith("_"):
            raise ExpressionError("invalid attribute name")
        # Count the attribute chain itself, not the depth of whatever
        # arithmetic or `not` happens to enclose it.
        if _attr_chain_length(node) >= _MAX_ATTR_DEPTH:
            raise ExpressionError("attribute chain too deep")
    for child in ast.iter_child_nodes(node):
        _validate(child)


def compile_expr(src: str) -> ExprObj:
    """Compile *src*, raising :class:`ExpressionError` on anything unlisted."""
    try:
        tree = ast.parse(src.strip(), mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"syntax error in {src!r}: {exc.msg}") from exc
    _validate(tree)
    return ExprObj(src.strip(), tree)


def _resolve_name(name: str, scope: Mapping[str, Any]) -> Any:
    # ``ast.Name.id`` is always a bare identifier, so this is a single lookup
    # in the scope mapping; dotted access is handled by the Attribute branch.
    if name not in scope:
        raise ExpressionError(f"unknown name {name!r}")
    return scope[name]


def _eval(node: ast.AST, scope: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body, scope)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return _resolve_name(node.id, scope)
    if isinstance(node, ast.Attribute):
        holder = _eval(node.value, scope)
        # Fail-closed: dotted access is a mapping key lookup, never getattr.
        # Real object attributes (gi_frame -> f_builtins -> eval/exec/open)
        # are unreachable by construction rather than by deny-list.
        if not isinstance(holder, Mapping):
            raise ExpressionError(
                f"attribute access is only allowed on mappings, not "
                f"{type(holder).__name__}"
            )
        if node.attr not in holder:
            raise ExpressionError(f"unknown key {node.attr!r}")
        return holder[node.attr]
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
        # Python semantics: short-circuit and return the deciding OPERAND.
        result: Any = None
        if isinstance(node.op, ast.And):
            for value in node.values:
                result = _eval(value, scope)
                if not result:
                    return result
            return result
        for value in node.values:
            result = _eval(value, scope)
            if result:
                return result
        return result
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
