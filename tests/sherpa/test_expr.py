"""Hermetic tests for the fail-closed expression evaluator."""

from __future__ import annotations

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
