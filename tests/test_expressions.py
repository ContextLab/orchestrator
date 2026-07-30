"""Tests for the constrained pipeline expression language.

These are adversarial by design: pipeline conditions are untrusted input, so
the interesting cases are the ones that try to escape the sandbox or to make a
guard fail open.
"""

import pytest

from orchestrator.core.expressions import (
    ExpressionError,
    evaluate_condition,
    evaluate_expression,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Expressions that must work
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("expression", "context", "expected"),
    [
        ("1 + 1", {}, 2),
        ("10 - 4 * 2", {}, 2),
        ("7 // 2", {}, 3),
        ("7 % 2", {}, 1),
        ("-5", {}, -5),
        ("True and False", {}, False),
        ("True or False", {}, True),
        ("not False", {}, True),
        ("x > 3", {"x": 5}, True),
        ("x > 3", {"x": 1}, False),
        ("1 < x < 10", {"x": 5}, True),
        ("1 < x < 10", {"x": 50}, False),
        ("x == 'done'", {"x": "done"}, True),
        ("x != None", {"x": 1}, True),
        ("'a' in items", {"items": ["a", "b"]}, True),
        ("'z' not in items", {"items": ["a", "b"]}, True),
        ("len(items) == 2", {"items": [1, 2]}, True),
        ("max(a, b)", {"a": 1, "b": 9}, 9),
        ("sum(items) > 5", {"items": [3, 4]}, True),
        ("items[0]", {"items": [42]}, 42),
        ("data['k']", {"data": {"k": "v"}}, "v"),
        ("items[1:]", {"items": [1, 2, 3]}, [2, 3]),
        ("'yes' if flag else 'no'", {"flag": True}, "yes"),
        ("[1, 2]", {}, [1, 2]),
        ("{'a': 1}", {}, {"a": 1}),
    ],
)
def test_supported_expressions(expression, context, expected):
    assert evaluate_expression(expression, context) == expected


def test_attribute_access_on_plain_object():
    class Result:
        status = "complete"

    assert evaluate_expression("r.status == 'complete'", {"r": Result()}) is True


# ---------------------------------------------------------------------------
# Sandbox escapes that must be refused
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "().__class__.__bases__[0].__subclasses__()",
        "x.__class__",
        "x.__dict__",
        "eval('1+1')",
        "exec('x=1')",
        "globals()",
        "locals()",
        "getattr(x, '__class__')",
        "[c for c in items]",           # comprehensions are not supported
        "(lambda: 1)()",                # lambdas are not supported
        "x := 5",                       # walrus / statements
        "print('hi')",                  # not in SAFE_FUNCTIONS
        "x.method()",                   # calls on arbitrary values
        "status.upper()",               # method calls on values are refused
        "2 ** 10 ** 10",                # power excluded: cheap DoS
        "f'{x}'",                       # f-strings are not supported
    ],
)
def test_sandbox_escapes_are_refused(expression):
    with pytest.raises(ExpressionError):
        evaluate_expression(expression, {"x": 1, "items": [1, 2]})


def test_undefined_name_is_refused():
    with pytest.raises(ExpressionError):
        evaluate_expression("does_not_exist > 1", {})


def test_context_is_the_only_scope():
    """A name defined in this module's globals must not be visible."""
    with pytest.raises(ExpressionError):
        evaluate_expression("evaluate_expression", {})


def test_empty_and_oversized_expressions_are_refused():
    with pytest.raises(ExpressionError):
        evaluate_expression("   ", {})
    with pytest.raises(ExpressionError):
        evaluate_expression("1 + " * 5000 + "1", {})


def test_syntax_error_is_wrapped():
    with pytest.raises(ExpressionError):
        evaluate_expression("1 +", {})


# ---------------------------------------------------------------------------
# Fail-closed behavior
# ---------------------------------------------------------------------------

def test_condition_evaluates_normally():
    assert evaluate_condition("x > 1", {"x": 2}) is True
    assert evaluate_condition("x > 1", {"x": 0}) is False


@pytest.mark.parametrize(
    "condition",
    [
        "__import__('os')",     # malicious
        "1 +",                  # malformed
        "undefined_thing",      # undefined
        "",                     # empty
    ],
)
def test_condition_fails_closed(condition):
    """A condition that cannot be evaluated must NOT run the guarded step.

    The replaced implementation returned True here -- both when its blocklist
    fired and in its generic exception handler -- so a malformed or hostile
    condition executed the step it was meant to gate.
    """
    assert evaluate_condition(condition, {}) is False


def test_fail_closed_default_is_overridable_but_explicit():
    assert evaluate_condition("bogus(", {}, default=True) is True


def test_textual_substitution_bug_does_not_recur():
    """Variable names must not be substituted as raw text.

    The old implementation did `condition.replace(var_name, repr(value))`, so a
    context variable named `a` rewrote the `a` inside `max(...)`, silently
    corrupting unrelated expressions.
    """
    assert evaluate_expression("max(a, 10)", {"a": 3}) == 10
