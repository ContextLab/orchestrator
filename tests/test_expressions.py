"""Tests for the constrained pipeline expression language.

These are adversarial by design: pipeline conditions are untrusted input, so
the interesting cases are the ones that try to escape the sandbox or to make a
guard fail open.
"""

import pytest

from orchestrator.core.expressions import (
    JSON_NAMESPACE,
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


# ---------------------------------------------------------------------------
# Bounded method calls and '**' (capabilities added so real pipelines could
# migrate off eval(); each one widened the sandbox and is attacked here)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("expression", "context", "expected"),
    [
        ("state.get('counter', 0) >= 5", {"state": {"counter": 7}}, True),
        ("sum(ex.values()) == 4", {"ex": {"a": 2, "b": 2}}, True),
        ("len(items.keys()) == 2", {"items": {"a": 1, "b": 2}}, True),
        ("name.startswith('pre')", {"name": "prefix"}, True),
        ("s.split(',')", {"s": "a,b"}, ["a", "b"]),
        ("2 ** 10", {}, 1024),
    ],
)
def test_bounded_capabilities_work(expression, context, expected):
    """Real pipeline conditions depend on these; they must keep working."""
    assert evaluate_expression(expression, context) == expected


class _EvilDict(dict):
    """A dict subclass whose get() would return something dangerous."""

    def get(self, *args, **kwargs):
        import os

        return os


@pytest.mark.parametrize(
    "expression",
    [
        # Method results must not be walked back to type objects.
        "d.get('x').__class__",
        "d.keys().__class__",
        "s.upper().__class__",
        # Exact-type matching: a dict SUBCLASS may override get(), so refuse it.
        "evil.get('x')",
        # Methods on modules / arbitrary objects.
        "mod.system('echo pwned')",
        # str.format is an information-disclosure vector; must not be allowed.
        "s.format(1)",
        "'{0.__class__}'.format(s)",
        # Not on the str allowlist.
        "s.encode()",
        "s.join(['a', 'b'])",
        # '**' must stay bounded and numeric.
        "10 ** 10 ** 10",
        "10 ** 100000",
        "(10 ** 1000) ** 1000",
        "'a' ** 2",
        "True ** 2",
    ],
)
def test_bounded_capabilities_cannot_be_abused(expression):
    import os

    context = {"d": {"x": 1}, "s": "a.b", "evil": _EvilDict(), "mod": os}
    with pytest.raises(ExpressionError):
        evaluate_expression(expression, context)


def test_chained_methods_cannot_amplify_memory():
    """Individually safe methods must not compose into an allocation bomb.

    Each `.replace('x','xx')` doubles its input. Chaining ~18 of them in a
    334-character expression allocated 769 MB before method results were
    size-capped; the AST depth budget bounded the chain length but not the
    growth rate.
    """
    expression = "('x'*1000)" + ".replace('x','xx')" * 18
    assert len(expression) < 400, "the attack must stay small to be meaningful"
    with pytest.raises(ExpressionError):
        evaluate_expression(expression, {})


# ---------------------------------------------------------------------------
# Bounded comprehensions and the json namespace (added so `transform_spec`
# could migrate off eval(); both widen the sandbox and are attacked here)
# ---------------------------------------------------------------------------

ITEMS_JSON = '{"items": [{"price": 2}, {"price": 3}]}'
LIST_JSON = '["a", "b", "c", "d"]'


@pytest.mark.parametrize(
    ("expression", "data", "expected"),
    [
        # Every one of these is a real transform_spec expression taken from
        # tests/test_action_loop.py and tests/integration/test_tools_real_world.py.
        ("sum(item['price'] for item in json.loads(data)['items'])", ITEMS_JSON, 5),
        ("len(json.loads(data)['items'])", ITEMS_JSON, 2),
        (
            "sum(item['price'] for item in json.loads(data)['items'])"
            " / len(json.loads(data)['items'])",
            ITEMS_JSON,
            2.5,
        ),
        ("len(json.loads(data))", LIST_JSON, 4),
        ("json.loads(data)[0]", LIST_JSON, "a"),
        ("[item.upper() for item in json.loads(data)]", LIST_JSON, ["A", "B", "C", "D"]),
    ],
)
def test_real_transform_expressions_evaluate(expression, data, expected):
    context = {"json": JSON_NAMESPACE, "data": data}
    assert evaluate_expression(expression, context) == expected


def test_module_globals_rce_is_refused():
    """The confirmed host-RCE payload must not evaluate.

    `transform_spec` used to run eval() with the real `json` MODULE in scope.
    Every Python function carries __globals__ -- a live reference to its
    defining module's globals, which holds the REAL builtins -- so

        json.loads.__globals__["__builtins__"]["__import__"]("os").system(...)

    reached the host no matter what __builtins__ the caller passed to eval().
    This was verified exploitable (it returned the live process id) before the
    fix, so it is pinned here permanently.
    """
    context = {"json": JSON_NAMESPACE, "data": "[]"}
    payload = 'json.loads.__globals__["__builtins__"]["__import__"]("os").getpid()'
    with pytest.raises(ExpressionError):
        evaluate_expression(payload, context)


@pytest.mark.parametrize(
    "expression",
    [
        # A function object must never be produced as a VALUE -- that object is
        # precisely what carries __globals__.
        "json.loads",
        "json.loads.__globals__",
        # Only the explicitly allowed members are callable.
        "json.dumps(data)",
        "json.load(data)",
        "json.JSONDecoder()",
        # The namespace must not leak its own internals.
        "json.name",
        "json.members",
        # Comprehensions must not become a new route to type objects.
        "[x.__class__ for x in [1]]",
        "[c for c in ().__class__.__bases__]",
        "{x: x.__class__ for x in [1]}",
        # Comprehension targets must not be able to shadow internals.
        "[_x for _x in [1]]",
    ],
)
def test_namespace_and_comprehension_escapes_are_refused(expression):
    context = {"json": JSON_NAMESPACE, "data": LIST_JSON}
    with pytest.raises(ExpressionError):
        evaluate_expression(expression, context)


def test_comprehension_iterations_are_bounded():
    """Nested comprehensions multiply; the budget is per-expression, not per-loop.

    A 10k-element source nested two deep is 100M iterations. Bounding each
    comprehension independently would let that through.
    """
    context = {"xs": list(range(10_000))}
    with pytest.raises(ExpressionError, match="iterations"):
        evaluate_expression("[[y for y in xs] for x in xs]", context)


def test_comprehension_cannot_leak_variables_into_context():
    """A loop variable must not survive, or overwrite a real context name."""
    context = {"data": "original"}
    assert evaluate_expression("[data for data in [1, 2]]", context) == [1, 2]
    assert context["data"] == "original", "comprehension mutated caller context"


def test_generator_expression_is_not_lazy():
    """A lazy generator would escape the size cap by being consumed later."""
    result = evaluate_expression("(x for x in [1, 2, 3])", {})
    assert result == [1, 2, 3], "generator expressions must be materialized"


def test_textual_substitution_bug_does_not_recur():
    """Variable names must not be substituted as raw text.

    The old implementation did `condition.replace(var_name, repr(value))`, so a
    context variable named `a` rewrote the `a` inside `max(...)`, silently
    corrupting unrelated expressions.
    """
    assert evaluate_expression("max(a, 10)", {"a": 3}) == 10
