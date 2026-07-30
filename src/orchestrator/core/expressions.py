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
import json
import logging
import operator
from collections import ChainMap
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)

__all__ = [
    "ExpressionError",
    "evaluate_expression",
    "evaluate_condition",
    "SAFE_FUNCTIONS",
    "SafeNamespace",
    "JSON_NAMESPACE",
]


class ExpressionError(ValueError):
    """Raised when an expression is malformed or uses a disallowed construct."""


class SafeNamespace:
    """An immutable namespace exposing a fixed set of pure callables.

    This exists so an expression can call something like ``json.loads(data)``
    *without* the ``json`` module itself being in scope.

    Putting a module in the evaluation context is the single most effective way
    to destroy a sandbox. Every Python function carries ``__globals__``: a live
    reference to the globals of the module that defined it, which contains the
    real, unrestricted ``__builtins__``. So with ``json`` in scope::

        json.loads.__globals__["__builtins__"]["__import__"]("os").system(...)

    reaches the host regardless of what ``__builtins__`` the caller passed to
    :func:`eval`. That was a confirmed remote-code-execution path in this
    codebase (``hybrid_control_system`` ``transform_spec``).

    Members are reachable **only in call position** -- the evaluator refuses
    attribute access on a ``SafeNamespace`` -- so no function object is ever
    produced as a value and there is nothing to walk ``__globals__`` on.
    """

    __slots__ = ("name", "members")

    def __init__(self, name: str, members: Mapping[str, Callable[..., Any]]) -> None:
        self.name = name
        self.members = dict(members)


#: The only module-like surface a transform expression may reach. ``loads`` is
#: pure parsing: it cannot open files, reach the network, or import anything.
JSON_NAMESPACE = SafeNamespace("json", {"loads": json.loads})


# Operators are mapped explicitly rather than executed by name, so only these
# exact behaviors are reachable.
_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    # Exponentiation is permitted but bounded (see _check_power): unbounded
    # `10**10**10` allocates an astronomically large integer and hangs the
    # process, so the exponent and operand magnitudes are capped.
    ast.Pow: operator.pow,
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

#: Methods callable on a value, keyed by the value's EXACT type. Exact-type
#: matching (``type(x) is dict``) is deliberate: a subclass could override
#: ``get`` with arbitrary code, so subclasses are not accepted.
SAFE_METHODS: dict[type, frozenset[str]] = {
    dict: frozenset({"get", "keys", "values", "items", "copy"}),
    list: frozenset({"count", "index", "copy"}),
    tuple: frozenset({"count", "index"}),
    set: frozenset({"issubset", "issuperset", "union", "intersection", "difference"}),
    frozenset: frozenset(
        {"issubset", "issuperset", "union", "intersection", "difference"}
    ),
    str: frozenset(
        {
            "startswith", "endswith", "lower", "upper", "strip", "lstrip",
            "rstrip", "split", "replace", "find", "count", "isdigit",
            "isalpha", "isnumeric", "title", "casefold",
        }
    ),
}

_MAX_EXPRESSION_LENGTH = 4096

#: Caps that bound resource use. Removing `**` alone does not close the
#: denial-of-service class: `'x' * 55000 * 55000` allocates ~3 GB from a
#: 20-character expression, and a flat chain like `not not not ...` recurses
#: deeply enough to exhaust the interpreter stack while staying under the
#: character cap. These bound both.
_MAX_AST_NODES = 500
_MAX_AST_DEPTH = 40
_MAX_SEQUENCE_LENGTH = 1_000_000
#: Bounds on `**` so exponentiation cannot allocate an astronomical integer.
_MAX_POWER_EXPONENT = 1024
_MAX_POWER_BASE = 1_000_000
#: Total comprehension iterations allowed per evaluation. This is a whole-
#: expression budget, not a per-comprehension one: nested comprehensions
#: multiply, so `[[y for y in xs] for x in xs]` over a 10k list would be 100M
#: iterations if each were bounded independently.
_MAX_COMPREHENSION_ITEMS = 100_000


def _ast_depth(node: ast.AST) -> int:
    """Maximum child-nesting depth of a parsed expression."""
    children = list(ast.iter_child_nodes(node))
    return 1 + max((_ast_depth(child) for child in children), default=0)


class _Evaluator(ast.NodeVisitor):
    """Walks a parsed expression, evaluating only allowlisted node types."""

    def __init__(self, context: Mapping[str, Any]) -> None:
        self.context = context
        # Whole-expression comprehension budget; see _MAX_COMPREHENSION_ITEMS.
        self._iterations = 0

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
        left = self.visit(node.left)
        right = self.visit(node.right)

        if isinstance(node.op, ast.Mult):
            self._check_repetition(left, right)
        elif isinstance(node.op, ast.Pow):
            self._check_power(left, right)
        elif isinstance(node.op, ast.Mod) and isinstance(left, (str, bytes)):
            # printf-style formatting can allocate arbitrarily large output
            # from a tiny expression, e.g. '%.300000000f' % 1.0 (~600 MB).
            raise ExpressionError(
                "string formatting with '%' is not supported in expressions"
            )

        return func(left, right)

    @staticmethod
    def _check_result_size(value: Any, what: str) -> Any:
        """Reject an oversized value produced by an allowed method call.

        Individually-safe methods compose into an amplifier: each
        ``.replace('x','xx')`` doubles its input, so a 334-character expression
        chaining ~18 of them allocated 769 MB. The AST depth budget bounds the
        chain length but not the growth rate, so the size is capped here too.
        """
        if isinstance(value, (str, bytes, list, tuple, set, frozenset, dict)):
            if len(value) > _MAX_SEQUENCE_LENGTH:
                raise ExpressionError(
                    f"result of {what}() exceeds {_MAX_SEQUENCE_LENGTH} elements"
                )
        return value

    @staticmethod
    def _check_power(base: Any, exponent: Any) -> None:
        """Bound exponentiation so it cannot allocate an enormous integer."""
        if not isinstance(base, (int, float)) or not isinstance(exponent, (int, float)):
            raise ExpressionError("'**' requires numeric operands")
        if isinstance(exponent, bool) or isinstance(base, bool):
            raise ExpressionError("'**' requires numeric operands")
        if abs(exponent) > _MAX_POWER_EXPONENT:
            raise ExpressionError(
                f"exponent exceeds {_MAX_POWER_EXPONENT}"
            )
        if abs(base) > _MAX_POWER_BASE:
            raise ExpressionError(f"base of '**' exceeds {_MAX_POWER_BASE}")

    @staticmethod
    def _check_repetition(left: Any, right: Any) -> None:
        """Reject sequence repetition that would allocate an enormous result."""
        for seq, count in ((left, right), (right, left)):
            if isinstance(seq, (str, bytes, list, tuple)) and isinstance(count, int):
                if len(seq) * max(count, 0) > _MAX_SEQUENCE_LENGTH:
                    raise ExpressionError(
                        "sequence repetition would exceed "
                        f"{_MAX_SEQUENCE_LENGTH} elements"
                    )

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
        if isinstance(value, SafeNamespace):
            # Reachable only in call position (see visit_Call). Returning the
            # member here would hand the expression a real function object, and
            # a function object is exactly what carries __globals__.
            raise ExpressionError(
                f"{value.name}.{node.attr} may only be called, not referenced"
            )
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
        if node.keywords:
            raise ExpressionError("keyword arguments are not supported")

        # Method call on a value, e.g. `state.get('n', 0)` or
        # `items.startswith('x')`. Permitted only on exact built-in container
        # types and only for methods in SAFE_METHODS, so no user-defined code
        # can be reached. Subclasses are refused because they can override the
        # method with arbitrary code.
        if isinstance(node.func, ast.Attribute):
            if node.func.attr.startswith("_"):
                raise ExpressionError(
                    f"access to private attribute {node.func.attr!r} denied"
                )
            target = self.visit(node.func.value)

            # Namespace call, e.g. `json.loads(data)`. Exact-type matched so a
            # subclass cannot smuggle in a different `members` mapping.
            if type(target) is SafeNamespace:
                member = target.members.get(node.func.attr)
                if member is None:
                    raise ExpressionError(
                        f"{target.name}.{node.func.attr}() is not allowed"
                    )
                args = [self.visit(arg) for arg in node.args]
                try:
                    result = member(*args)
                except ExpressionError:
                    raise
                except Exception as exc:
                    raise ExpressionError(
                        f"{target.name}.{node.func.attr}() failed: {exc}"
                    ) from exc
                return self._check_result_size(result, node.func.attr)

            allowed = SAFE_METHODS.get(type(target))
            if allowed is None:
                raise ExpressionError(
                    f"method calls are not allowed on {type(target).__name__}"
                )
            if node.func.attr not in allowed:
                raise ExpressionError(
                    f"{type(target).__name__}.{node.func.attr}() is not allowed"
                )
            args = [self.visit(arg) for arg in node.args]
            return self._check_result_size(
                getattr(target, node.func.attr)(*args), node.func.attr
            )

        # Otherwise only bare names from SAFE_FUNCTIONS are callable.
        if not isinstance(node.func, ast.Name):
            raise ExpressionError("only direct calls to built-in helpers are allowed")
        if node.func.id not in SAFE_FUNCTIONS:
            raise ExpressionError(f"call to {node.func.id!r} is not allowed")
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

    # -- comprehensions ----------------------------------------------------
    # Supported because transform expressions genuinely need them, e.g.
    # "sum(item['price'] for item in json.loads(data)['items'])". They are
    # bounded by a whole-expression iteration budget and a result-size cap:
    # a comprehension is the cheapest way to turn a short expression into an
    # arbitrarily large allocation.

    def _spend_iteration(self) -> None:
        self._iterations += 1
        if self._iterations > _MAX_COMPREHENSION_ITEMS:
            raise ExpressionError(
                f"comprehension exceeded {_MAX_COMPREHENSION_ITEMS} iterations"
            )

    def _bind_target(self, target: ast.expr, value: Any) -> None:
        """Bind a comprehension loop variable in the innermost scope."""
        if isinstance(target, ast.Name):
            if target.id.startswith("_"):
                raise ExpressionError(
                    f"comprehension variable {target.id!r} may not start with '_'"
                )
            self.context[target.id] = value  # type: ignore[index]
            return
        if isinstance(target, ast.Tuple):
            try:
                items = list(value)
            except TypeError as exc:
                raise ExpressionError(
                    f"cannot unpack {type(value).__name__} in comprehension"
                ) from exc
            if len(items) != len(target.elts):
                raise ExpressionError(
                    f"cannot unpack {len(items)} values into "
                    f"{len(target.elts)} targets"
                )
            for sub_target, item in zip(target.elts, items):
                self._bind_target(sub_target, item)
            return
        raise ExpressionError(
            f"unsupported comprehension target: {type(target).__name__}"
        )

    def _iterate(self, generators: list[ast.comprehension], index: int) -> Any:
        """Yield once per surviving combination of loop bindings."""
        if index == len(generators):
            yield
            return
        generator = generators[index]
        if generator.is_async:
            raise ExpressionError("async comprehensions are not supported")

        iterable = self.visit(generator.iter)
        try:
            items = list(iterable)
        except TypeError as exc:
            raise ExpressionError(
                f"cannot iterate over {type(iterable).__name__}"
            ) from exc
        if len(items) > _MAX_COMPREHENSION_ITEMS:
            raise ExpressionError(
                f"comprehension source of {len(items)} elements exceeds "
                f"{_MAX_COMPREHENSION_ITEMS}"
            )

        for item in items:
            self._spend_iteration()
            self._bind_target(generator.target, item)
            if all(self.visit(condition) for condition in generator.ifs):
                yield from self._iterate(generators, index + 1)

    def _comprehend(
        self,
        generators: list[ast.comprehension],
        element: ast.expr,
        value: ast.expr | None = None,
    ) -> list[Any]:
        """Evaluate a comprehension body in a child scope.

        Loop variables are written to a fresh mapping layered over the caller's
        context, so a comprehension can never mutate or shadow the real
        pipeline context after it finishes.
        """
        outer = self.context
        self.context = ChainMap({}, dict(outer))
        try:
            results = []
            for _ in self._iterate(generators, 0):
                if value is None:
                    results.append(self.visit(element))
                else:
                    results.append((self.visit(element), self.visit(value)))
            return results
        finally:
            self.context = outer

    def visit_ListComp(self, node: ast.ListComp) -> Any:
        return self._check_result_size(
            self._comprehend(node.generators, node.elt), "comprehension"
        )

    def visit_SetComp(self, node: ast.SetComp) -> Any:
        return self._check_result_size(
            set(self._comprehend(node.generators, node.elt)), "comprehension"
        )

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> Any:
        # Materialized eagerly rather than returned lazily: a lazy generator
        # would escape the size cap and could be consumed outside the budget.
        return self._check_result_size(
            self._comprehend(node.generators, node.elt), "comprehension"
        )

    def visit_DictComp(self, node: ast.DictComp) -> Any:
        pairs = self._comprehend(node.generators, node.key, node.value)
        return self._check_result_size(dict(pairs), "comprehension")


def evaluate_expression(expression: str, context: Mapping[str, Any] | None = None) -> Any:
    """Evaluate ``expression`` against ``context`` and return its value.

    Args:
        expression: A single Python-syntax expression. Statements, assignments,
            lambdas, imports and f-strings are not supported. Comprehensions
            are supported but bounded (see ``_MAX_COMPREHENSION_ITEMS``).
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
    except (ValueError, MemoryError, RecursionError) as exc:
        # ast.parse itself rejects some pathological inputs (e.g. null bytes,
        # excessive nesting) with these.
        raise ExpressionError(f"expression could not be parsed: {exc}") from exc

    # Bound the shape of the tree before walking it. A flat operator chain such
    # as ("not " * 800 + "1") stays under the character cap but exhausts the
    # interpreter stack inside the recursive visitor.
    node_count = sum(1 for _ in ast.walk(tree))
    if node_count > _MAX_AST_NODES:
        raise ExpressionError(
            f"expression is too complex ({node_count} nodes, limit {_MAX_AST_NODES})"
        )
    depth = _ast_depth(tree)
    if depth > _MAX_AST_DEPTH:
        raise ExpressionError(
            f"expression is nested too deeply ({depth} levels, limit {_MAX_AST_DEPTH})"
        )

    try:
        return _Evaluator(context or {}).visit(tree)
    except ExpressionError:
        raise
    except Exception as exc:
        # Keep the documented contract: callers of this function should only
        # have to handle ExpressionError. Ordinary evaluation errors
        # (ZeroDivisionError, TypeError from `'a' + 1`, ValueError from
        # `int('nope')`, RecursionError) are wrapped rather than leaking out.
        raise ExpressionError(
            f"{type(exc).__name__} while evaluating expression: {exc}"
        ) from exc


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
