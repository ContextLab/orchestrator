"""Hermetic tests for the fail-closed expression evaluator."""

from __future__ import annotations

import time

import pytest

from sherpa.expr import ExpressionError, compile_expr, evaluate

pytestmark = [pytest.mark.unit]

SCOPE = {"x": 3, "y": "abc", "items": [1, 2, 3], "meta": {"depth": 2}, "flag": True}


class TestAllowed:
    def test_comparisons(self) -> None:
        assert evaluate("x == 3", SCOPE) is True
        assert evaluate("x < 2", SCOPE) is False
        assert evaluate("y != 'zzz'", SCOPE) is True

    def test_boolean_logic(self) -> None:
        assert evaluate("x > 1 and flag", SCOPE) is True
        assert evaluate("not flag or x > 9", SCOPE) is False

    def test_membership(self) -> None:
        assert evaluate("x in items", SCOPE) is True
        assert evaluate("'z' in y", SCOPE) is False
        assert evaluate("x not in items", SCOPE) is False

    def test_arithmetic(self) -> None:
        assert evaluate("x * 2 + 1", SCOPE) == 7
        assert evaluate("(x - 1) / 2", SCOPE) == 1.0

    def test_dotted_names_and_subscript(self) -> None:
        assert evaluate("meta.depth == 2", SCOPE) is True
        assert evaluate("meta['depth'] == 2", SCOPE) is True
        assert evaluate("items[0] == 1", SCOPE) is True


class TestRejected:
    @pytest.mark.parametrize(
        "src",
        [
            "__import__('os').system('true')",
            "open('/etc/passwd')",
            "(lambda: 1)()",
            "[i for i in items]",
            "f'{x}'",
            "exec('1')",
            "meta.__class__",
        ],
    )
    def test_disallowed_syntax(self, src: str) -> None:
        with pytest.raises(ExpressionError):
            compile_expr(src)

    def test_attribute_chain_too_deep(self) -> None:
        with pytest.raises(ExpressionError):
            compile_expr("a.b.c.d.e")

    def test_division_by_zero_fail_closed(self) -> None:
        with pytest.raises(ExpressionError):
            evaluate("x / 0", SCOPE)

    def test_unknown_name(self) -> None:
        with pytest.raises(ExpressionError):
            evaluate("missing == 1", SCOPE)

    def test_bad_comparison_type(self) -> None:
        with pytest.raises(ExpressionError):
            evaluate("x < y", SCOPE)

    def test_arithmetic_on_strings_denied(self) -> None:
        with pytest.raises(ExpressionError):
            evaluate("y + 'd'", SCOPE)

    def test_compile_returns_reusable_obj(self) -> None:
        obj = compile_expr("x > 2")
        assert evaluate(obj, SCOPE) is True
        assert evaluate(obj, {"x": 1}) is False


class TestShortCircuit:
    """D1: `and`/`or` must not evaluate operands past the decisive one.

    kernel.py:372 (Branch `when`) and kernel.py:383 (While guard) call
    ``evaluate`` WITHOUT catching ExpressionError, so an eager right-hand
    operand aborts the whole run for the most idiomatic guard people write.
    """

    def test_and_does_not_evaluate_rhs_when_lhs_false(self) -> None:
        # The right operand would raise "bad subscript 'zz'" if evaluated.
        scope = {"d": {"a": 1}}
        assert evaluate("'zz' in d and d['zz'] > 1", scope) is False

    def test_or_does_not_evaluate_rhs_when_lhs_true(self) -> None:
        # The right operand would raise "division by zero" if evaluated.
        scope = {"n": 0}
        assert evaluate("n == 0 or 10 / n > 1", scope) is True

    def test_and_chain_stops_at_first_false(self) -> None:
        scope = {"d": {"a": 1}, "ok": True}
        assert evaluate("ok and 'zz' in d and d['zz'] and missing", scope) is False

    def test_or_chain_stops_at_first_true(self) -> None:
        scope = {"n": 0}
        assert evaluate("n == 0 or missing or 1 / n", scope) is True

    def test_short_circuit_still_raises_when_decisive_operand_fails(self) -> None:
        # Fail-closed is preserved: a *reachable* bad operand still raises.
        with pytest.raises(ExpressionError):
            evaluate("d['zz'] > 1 and True", {"d": {"a": 1}})

    def test_guard_shape_used_by_kernel_while_loops(self) -> None:
        # Realistic While guard: node output may not exist on the first pass.
        scope = {"verify": {"result": {"passed": False}}}
        assert evaluate("'verify' in scope_names and verify.result.passed",
                        {**scope, "scope_names": list(scope)}) is False


class TestBoolOpReturnsOperand:
    """D2: `and`/`or` return the deciding OPERAND, as Python does.

    Decision: full Python semantics (option (a)). This module serves BOTH
    guards and ``{{ ... }}`` input templates (capabilities.resolve_inputs),
    so `path or 'default.txt'` must yield the string, not True. Guard call
    sites already coerce (kernel.py:569 wraps in ``bool``; Branch/While use
    truthiness), so nothing loses correctness from the richer return value.
    """

    def test_or_returns_fallback_operand_not_bool(self) -> None:
        assert evaluate("y or 'fallback'", {"y": ""}) == "fallback"

    def test_or_returns_first_truthy_operand(self) -> None:
        assert evaluate("y or 'fallback'", {"y": "real"}) == "real"

    def test_and_returns_last_operand_when_all_truthy(self) -> None:
        assert evaluate("a and b", {"a": 1, "b": "kept"}) == "kept"

    def test_and_returns_first_falsy_operand(self) -> None:
        assert evaluate("a and b", {"a": 0, "b": "unused"}) == 0

    def test_or_chain_returns_first_truthy(self) -> None:
        scope = {"a": "", "b": [], "c": {"k": 1}}
        assert evaluate("a or b or c", scope) == {"k": 1}

    def test_and_chain_returns_first_falsy(self) -> None:
        scope = {"a": 1, "b": [], "c": "never"}
        assert evaluate("a and b and c", scope) == []

    def test_input_template_fallback_matches_resolve_inputs_usage(self) -> None:
        # This is exactly what capabilities.resolve_inputs binds for
        # `{{ inputs.path or 'README.md' }}`.
        scope = {"inputs": {"path": ""}}
        assert evaluate("inputs.path or 'README.md'", scope) == "README.md"

    def test_truthiness_of_result_still_drives_guards(self) -> None:
        # Guard call sites care only about truthiness; operand return is safe.
        assert bool(evaluate("y or 'fallback'", {"y": ""})) is True
        assert bool(evaluate("a and b", {"a": 0, "b": "unused"})) is False


def _live_generator():
    """A real generator object — the kind of value that leaks frames."""
    yield 1
    yield 2


class TestAttributeSandbox:
    """D3: attribute access must be fail-closed, not deny-list filtered.

    The old `_`-prefix filter blocked dunders but frame-traversal names
    (gi_frame, f_globals, f_builtins) have no leading underscore, so an
    object in scope handed out the builtins dict containing eval/exec/open.
    """

    def test_generator_frame_traversal_is_blocked(self) -> None:
        gen = _live_generator()
        next(gen)  # make gi_frame non-None so the escape is genuinely live
        try:
            assert gen.gi_frame is not None, "precondition: frame really exists"
            with pytest.raises(ExpressionError):
                evaluate("g.gi_frame", {"g": gen})
        finally:
            gen.close()

    def test_generator_frame_globals_is_blocked(self) -> None:
        gen = _live_generator()
        next(gen)
        try:
            with pytest.raises(ExpressionError):
                evaluate("g.gi_frame.f_globals", {"g": gen})
        finally:
            gen.close()

    def test_generator_frame_builtins_is_blocked(self) -> None:
        gen = _live_generator()
        next(gen)
        try:
            # Prove the escape target is real before proving it is unreachable.
            assert "eval" in gen.gi_frame.f_builtins
            with pytest.raises(ExpressionError):
                evaluate("g.gi_frame.f_builtins", {"g": gen})
        finally:
            gen.close()

    def test_function_globals_traversal_is_blocked(self) -> None:
        def carrier() -> None:
            return None

        assert isinstance(carrier.__globals__, dict)
        with pytest.raises(ExpressionError):
            evaluate("fn.__globals__", {"fn": carrier})

    def test_attribute_access_on_arbitrary_object_is_blocked(self) -> None:
        class Holder:
            secret = "leaked"

        with pytest.raises(ExpressionError):
            evaluate("h.secret", {"h": Holder()})

    def test_attribute_access_on_builtin_types_is_blocked(self) -> None:
        # str/list/int expose real attributes; none of them are reachable.
        for src, scope in (
            ("s.capitalize", {"s": "abc"}),
            ("lst.append", {"lst": [1, 2]}),
            ("n.numerator", {"n": 7}),
        ):
            with pytest.raises(ExpressionError):
                evaluate(src, scope)

    def test_legitimate_mapping_traversal_still_works(self) -> None:
        # The shape kernel/test_kernel actually bind: "{{ w1.result.file }}"
        scope = {"w1": {"result": {"file": "out.txt", "count": 3}}}
        assert evaluate("w1.result.file", scope) == "out.txt"
        assert evaluate("w1.result.count > 2", scope) is True

    def test_legitimate_traversal_of_child_plan_outputs(self) -> None:
        # kernel.py:116 in repair_planner binds "{{ verify.result }}"
        scope = {"verify": {"result": {"passed": True}}}
        assert evaluate("verify.result", scope) == {"passed": True}
        assert evaluate("verify.result.passed", scope) is True

    def test_missing_mapping_key_still_fails_closed(self) -> None:
        with pytest.raises(ExpressionError):
            evaluate("w1.result.nope", {"w1": {"result": {"file": "x"}}})


class TestAttributeChainDepth:
    """D4: the depth cap must count the ATTRIBUTE CHAIN, not generic AST depth.

    Nesting a shallow `a.b` inside arithmetic or `not` used to trip the cap,
    which rejects perfectly ordinary guards.
    """

    @pytest.mark.parametrize(
        "src",
        [
            "a.b",
            "1 + (2 + (3 + a.b))",
            "not (not (not (not a.b)))",
            "((((a.b))))",
            "1 + (2 + (3 + (4 + (5 + a.b))))",
            "a.b and (1 + (2 + (3 + c.d)))",
        ],
    )
    def test_shallow_chain_nested_deeply_still_compiles(self, src: str) -> None:
        # Compiles without error; depth of surrounding AST is irrelevant.
        assert compile_expr(src).source == src

    def test_nested_shallow_chain_evaluates(self) -> None:
        scope = {"a": {"b": 4}}
        assert evaluate("1 + (2 + (3 + a.b))", scope) == 10
        assert evaluate("not (not (not (not a.b)))", scope) is True

    @pytest.mark.parametrize("src", ["a.b.c", "a.b.c.d"])
    def test_chains_within_limit_compile(self, src: str) -> None:
        assert compile_expr(src).source == src

    @pytest.mark.parametrize("src", ["a.b.c.d.e", "a.b.c.d.e.f", "1 + a.b.c.d.e"])
    def test_chains_over_limit_are_rejected(self, src: str) -> None:
        with pytest.raises(ExpressionError, match="attribute chain too deep"):
            compile_expr(src)

    def test_long_chain_rejected_even_when_nested_in_subscript(self) -> None:
        with pytest.raises(ExpressionError, match="attribute chain too deep"):
            compile_expr("d[a.b.c.d.e]")

    def test_chain_length_counts_per_chain_not_cumulatively(self) -> None:
        # Two independent short chains must not add up to a rejection.
        assert compile_expr("a.b.c == d.e.f").source == "a.b.c == d.e.f"


class TestFailuresAreAlwaysExpressionError:
    """Evaluation failure must surface as ExpressionError, never a bare
    Python exception. kernel.py:372/383 only ever tolerate ExpressionError
    (and in fact do not catch even that), so a leaked ZeroDivisionError or
    TypeError would escape the fail-closed contract entirely.
    """

    @pytest.mark.parametrize(
        ("src", "scope", "why"),
        [
            ("x / 0", SCOPE, "true division by zero"),
            ("x // 0", SCOPE, "floor division by zero"),
            ("x % 0", SCOPE, "modulo by zero"),
            ("1 / (x - 3)", SCOPE, "division by a computed zero"),
            ("meta['nope']", SCOPE, "missing dict key"),
            ("items[99]", SCOPE, "list index out of range"),
            ("x[0]", SCOPE, "subscript of a non-container"),
            ("y[99]", SCOPE, "string index out of range"),
            ("missing == 1", SCOPE, "unknown name"),
            ("meta.nope", SCOPE, "unknown mapping key via dotted access"),
            ("x < y", SCOPE, "int vs str comparison"),
            ("y + 'd'", SCOPE, "arithmetic on strings"),
            ("flag + 1", SCOPE, "arithmetic on a bool"),
            ("-y", SCOPE, "unary minus on a string"),
            ("x in y", SCOPE, "`in` with an int needle and str haystack"),
            ("items in meta", SCOPE, "`in` with an unhashable needle"),
            ("x in flag", SCOPE, "`in` against a non-container"),
        ],
    )
    def test_failure_raises_expression_error(self, src: str, scope: dict, why: str) -> None:
        try:
            result = evaluate(src, scope)
        except ExpressionError:
            return
        except Exception as exc:  # noqa: BLE001 - this is the thing under test
            pytest.fail(
                f"{why}: evaluate({src!r}) leaked a bare "
                f"{type(exc).__name__}: {exc} instead of ExpressionError"
            )
        pytest.fail(f"{why}: evaluate({src!r}) unexpectedly returned {result!r}")

    def test_expression_error_is_not_a_python_builtin_subclass(self) -> None:
        with pytest.raises(ExpressionError) as caught:
            evaluate("x / 0", SCOPE)
        assert not isinstance(caught.value, (ZeroDivisionError, TypeError, KeyError))
        # The original cause is preserved for debugging, not re-raised.
        assert isinstance(caught.value.__cause__, ZeroDivisionError)

    def test_bad_subscript_reports_the_offending_key(self) -> None:
        with pytest.raises(ExpressionError, match="bad subscript"):
            evaluate("d['zz']", {"d": {"a": 1}})

    def test_unknown_name_reports_the_offending_name(self) -> None:
        with pytest.raises(ExpressionError, match="unknown name 'nope'"):
            evaluate("nope", SCOPE)


class TestResourceExhaustion:
    """A guard expression must never be able to wedge the kernel."""

    def test_huge_exponent_does_not_hang(self) -> None:
        started = time.monotonic()
        with pytest.raises(ExpressionError):
            evaluate("2 ** 99999999", {})
        elapsed = time.monotonic() - started
        assert elapsed < 1.0, f"took {elapsed:.3f}s; expected an immediate rejection"

    def test_pow_operator_is_rejected_at_compile_time(self) -> None:
        # Rejected structurally, so no operand values are ever computed.
        with pytest.raises(ExpressionError, match="disallowed syntax: Pow"):
            compile_expr("2 ** 99999999")

    @pytest.mark.parametrize(
        "src",
        ["x ** 99999999", "2 ** x", "(2 ** 64) ** (2 ** 64)"],
    )
    def test_all_exponentiation_forms_rejected(self, src: str) -> None:
        with pytest.raises(ExpressionError):
            evaluate(src, SCOPE)

    def test_left_shift_is_rejected(self) -> None:
        # The other cheap way to build a giant int.
        with pytest.raises(ExpressionError):
            evaluate("1 << 99999999", {})

    def test_large_but_allowed_arithmetic_stays_bounded(self) -> None:
        started = time.monotonic()
        assert evaluate("999999999 * 999999999", {}) == 999999999 * 999999999
        assert time.monotonic() - started < 1.0
