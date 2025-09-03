"""Unit tests for condition evaluators."""

import pytest
import asyncio
from unittest.mock import Mock

from src.orchestrator.actions.condition_evaluator import (

from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider
    BooleanEvaluator,
    ComparisonEvaluator,
    LogicalEvaluator,
    TemplateEvaluator,
    ExpressionEvaluator,
    ConditionEvaluationError,
    get_condition_evaluator,
)


class TestBooleanEvaluator:
    """Test BooleanEvaluator."""
    
    @pytest.fixture
    def evaluator(self):
        return BooleanEvaluator()
    
    @pytest.mark.asyncio
    async def test_simple_boolean_strings(self, evaluator):
        """Test evaluation of simple boolean strings."""
        # True values
        assert await evaluator.evaluate("true", {}) is True
        assert await evaluator.evaluate("True", {}) is True
        assert await evaluator.evaluate("TRUE", {}) is True
        assert await evaluator.evaluate("yes", {}) is True
        assert await evaluator.evaluate("on", {}) is True
        assert await evaluator.evaluate("enabled", {}) is True
        assert await evaluator.evaluate("1", {}) is True
        
        # False values
        assert await evaluator.evaluate("false", {}) is False
        assert await evaluator.evaluate("False", {}) is False
        assert await evaluator.evaluate("FALSE", {}) is False
        assert await evaluator.evaluate("no", {}) is False
        assert await evaluator.evaluate("off", {}) is False
        assert await evaluator.evaluate("disabled", {}) is False
        assert await evaluator.evaluate("0", {}) is False
        assert await evaluator.evaluate("none", {}) is False
        assert await evaluator.evaluate("", {}) is False
    
    @pytest.mark.asyncio
    async def test_context_variables(self, evaluator):
        """Test evaluation with context variables."""
        context = {
            "enabled": True,
            "disabled": False,
            "count": 5,
            "empty": "",
            "zero": 0,
        }
        
        assert await evaluator.evaluate("enabled", context) is True
        assert await evaluator.evaluate("disabled", context) is False
        assert await evaluator.evaluate("count", context) is True
        assert await evaluator.evaluate("empty", context) is False
        assert await evaluator.evaluate("zero", context) is False
    
    @pytest.mark.asyncio
    async def test_python_expressions(self, evaluator):
        """Test evaluation of Python expressions."""
        context = {"x": 5, "y": 10}
        
        assert await evaluator.evaluate("x > 0", context) is True
        assert await evaluator.evaluate("x > y", context) is False
        assert await evaluator.evaluate("x + y > 10", context) is True
    
    @pytest.mark.asyncio
    async def test_invalid_expressions(self, evaluator):
        """Test handling of invalid expressions."""
        with pytest.raises(ConditionEvaluationError):
            await evaluator.evaluate("invalid expression !@#", {})


class TestComparisonEvaluator:
    """Test ComparisonEvaluator."""
    
    @pytest.fixture
    def evaluator(self):
        return ComparisonEvaluator()
    
    @pytest.mark.asyncio
    async def test_numeric_comparisons(self, evaluator):
        """Test numeric comparison operations."""
        context = {"x": 5, "y": 10}
        
        assert await evaluator.evaluate("x == 5", context) is True
        assert await evaluator.evaluate("x != 5", context) is False
        assert await evaluator.evaluate("x < y", context) is True
        assert await evaluator.evaluate("x > y", context) is False
        assert await evaluator.evaluate("x <= 5", context) is True
        assert await evaluator.evaluate("y >= 10", context) is True
    
    @pytest.mark.asyncio
    async def test_string_comparisons(self, evaluator):
        """Test string comparison operations."""
        context = {"name": "test", "status": "active"}
        
        assert await evaluator.evaluate('name == "test"', context) is True
        assert await evaluator.evaluate("name != 'test'", context) is False
        assert await evaluator.evaluate('status == "active"', context) is True
    
    @pytest.mark.asyncio
    async def test_membership_operations(self, evaluator):
        """Test in/not in operations."""
        context = {
            "items": [1, 2, 3],
            "text": "hello world",
            "data": {"key": "value"},
        }
        
        assert await evaluator.evaluate("2 in items", context) is True
        assert await evaluator.evaluate("4 in items", context) is False
        assert await evaluator.evaluate("4 not in items", context) is True
        assert await evaluator.evaluate('"world" in text', context) is True
        assert await evaluator.evaluate('"key" in data', context) is True
    
    @pytest.mark.asyncio
    async def test_parameter_based_comparison(self, evaluator):
        """Test comparison using parameters."""
        context = {
            "operator": "==",
            "left": 5,
            "right": 5,
        }
        
        assert await evaluator.evaluate("", context) is True
        
        context["operator"] = "!="
        assert await evaluator.evaluate("", context) is False
        
        context["operator"] = "<"
        context["right"] = 10
        assert await evaluator.evaluate("", context) is True
    
    @pytest.mark.asyncio
    async def test_dot_notation(self, evaluator):
        """Test dot notation for nested objects."""
        context = {
            "user": {"name": "John", "age": 25},
            "config": {"enabled": True, "limit": 100},
        }
        
        assert await evaluator.evaluate('user.name == "John"', context) is True
        assert await evaluator.evaluate("user.age > 20", context) is True
        assert await evaluator.evaluate("config.enabled == true", context) is True
        assert await evaluator.evaluate("config.limit >= 100", context) is True


class TestLogicalEvaluator:
    """Test LogicalEvaluator."""
    
    @pytest.fixture
    def evaluator(self):
        return LogicalEvaluator()
    
    @pytest.mark.asyncio
    async def test_and_operations(self, evaluator):
        """Test AND operations."""
        context = {"x": 5, "y": 10, "enabled": True}
        
        assert await evaluator.evaluate("x > 0 and y > 0", context) is True
        assert await evaluator.evaluate("x > 0 and y < 0", context) is False
        assert await evaluator.evaluate("enabled and x == 5", context) is True
        assert await evaluator.evaluate("enabled and x == 10", context) is False
    
    @pytest.mark.asyncio
    async def test_or_operations(self, evaluator):
        """Test OR operations."""
        context = {"x": 5, "y": 10, "enabled": False}
        
        assert await evaluator.evaluate("x > 0 or y < 0", context) is True
        assert await evaluator.evaluate("x < 0 or y < 0", context) is False
        assert await evaluator.evaluate("enabled or x == 5", context) is True
        assert await evaluator.evaluate("enabled or x == 10", context) is False
    
    @pytest.mark.asyncio
    async def test_not_operations(self, evaluator):
        """Test NOT operations."""
        context = {"x": 5, "enabled": True, "disabled": False}
        
        assert await evaluator.evaluate("not false", context) is True
        assert await evaluator.evaluate("not true", context) is False
        assert await evaluator.evaluate("not disabled", context) is True
        assert await evaluator.evaluate("not enabled", context) is False
        assert await evaluator.evaluate("not x > 10", context) is True
    
    @pytest.mark.asyncio
    async def test_complex_logical_expressions(self, evaluator):
        """Test complex logical expressions."""
        context = {"a": 5, "b": 10, "c": 15, "enabled": True}
        
        # Precedence: NOT > AND > OR
        assert await evaluator.evaluate("a > 0 and b > 0 or c < 0", context) is True
        assert await evaluator.evaluate("a > 0 or b > 0 and c < 0", context) is True
        assert await evaluator.evaluate("not a > 10 and b > 5", context) is True
        assert await evaluator.evaluate("enabled and (a > 0 or b < 0)", context) is True
    
    @pytest.mark.asyncio
    async def test_parentheses_handling(self, evaluator):
        """Test handling of parentheses."""
        context = {"x": 5, "y": 10, "z": 15}
        
        assert await evaluator.evaluate("(x > 0)", context) is True
        assert await evaluator.evaluate("(x > 0 and y > 0)", context) is True
        assert await evaluator.evaluate("(x > 0) and (y > 0)", context) is True
        assert await evaluator.evaluate("(x > 10 or y > 5) and z > 10", context) is True


class TestTemplateEvaluator:
    """Test TemplateEvaluator."""
    
    @pytest.fixture
    def evaluator(self):
        return TemplateEvaluator()
    
    @pytest.fixture
    def template_manager(self):
        """Mock template manager."""
        manager = Mock()
        manager.render = Mock(side_effect=lambda template, **kwargs: 
            template.replace("{{ x }}", "5").replace("{{ y }}", "10"))
        return manager
    
    @pytest.mark.asyncio
    async def test_simple_template_replacement(self, evaluator):
        """Test simple template variable replacement."""
        context = {"x": 5, "y": 10, "name": "test"}
        
        assert await evaluator.evaluate("{{ x }} > 0", context) is True
        assert await evaluator.evaluate("{{ x }} > {{ y }}", context) is False
        assert await evaluator.evaluate('{{ name }} == "test"', context) is True
    
    @pytest.mark.asyncio
    async def test_dot_notation_templates(self, evaluator):
        """Test template replacement with dot notation."""
        context = {
            "user": {"name": "John", "age": 25},
            "items": [1, 2, 3],
        }
        
        assert await evaluator.evaluate('{{ user.name }} == "John"', context) is True
        assert await evaluator.evaluate("{{ user.age }} > 20", context) is True
    
    @pytest.mark.asyncio
    async def test_with_template_manager(self):
        """Test with actual template manager."""
        template_manager = Mock()
        template_manager.render = Mock(return_value="5 > 0")
        evaluator = TemplateEvaluator(template_manager)
        
        result = await evaluator.evaluate("{{ x }} > 0", {"x": 5})
        assert result is True
        template_manager.render.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_no_templates(self, evaluator):
        """Test conditions without templates."""
        context = {"x": 5}
        
        # Should fall through to logical evaluator
        assert await evaluator.evaluate("x > 0", context) is True
        assert await evaluator.evaluate("true and false", context) is False


class TestExpressionEvaluator:
    """Test ExpressionEvaluator."""
    
    @pytest.fixture
    def evaluator(self):
        return ExpressionEvaluator()
    
    @pytest.mark.asyncio
    async def test_mathematical_expressions(self, evaluator):
        """Test mathematical expressions."""
        context = {"x": 5, "y": 10, "z": 2}
        
        assert await evaluator.evaluate("x + y > 10", context) is True
        assert await evaluator.evaluate("x * z == 10", context) is True
        assert await evaluator.evaluate("y / z == 5", context) is True
        assert await evaluator.evaluate("y % 3 == 1", context) is True
        assert await evaluator.evaluate("z ** 3 == 8", context) is True
    
    @pytest.mark.asyncio
    async def test_function_calls(self, evaluator):
        """Test allowed function calls."""
        context = {"items": [1, 2, 3, 4, 5], "value": 3.7}
        
        assert await evaluator.evaluate("len(items) == 5", context) is True
        assert await evaluator.evaluate("max(items) == 5", context) is True
        assert await evaluator.evaluate("min(items) == 1", context) is True
        assert await evaluator.evaluate("sum(items) == 15", context) is True
        assert await evaluator.evaluate("round(value) == 4", context) is True
    
    @pytest.mark.asyncio
    async def test_type_conversions(self, evaluator):
        """Test type conversion functions."""
        context = {"x": "5", "y": 10.5, "z": 0}
        
        assert await evaluator.evaluate("int(x) == 5", context) is True
        assert await evaluator.evaluate("float(x) == 5.0", context) is True
        assert await evaluator.evaluate("str(y) == '10.5'", context) is True
        assert await evaluator.evaluate("bool(z) == False", context) is True
    
    @pytest.mark.asyncio
    async def test_unsafe_operations(self, evaluator):
        """Test that unsafe operations are blocked."""
        context = {}
        
        # These should raise errors or fall back to template evaluator
        with pytest.raises(Exception):
            await evaluator.evaluate("__import__('os')", context)
        
        with pytest.raises(Exception):
            await evaluator.evaluate("exec('print(1)')", context)
    
    @pytest.mark.asyncio
    async def test_complex_expressions(self, evaluator):
        """Test complex nested expressions."""
        context = {
            "scores": [85, 90, 78, 92, 88],
            "threshold": 80,
            "bonus": 5,
        }
        
        # Average score calculation
        assert await evaluator.evaluate(
            "sum(scores) / len(scores) > threshold", 
            context
        ) is True
        
        # Complex condition
        assert await evaluator.evaluate(
            "(max(scores) - min(scores)) < 20 and sum(scores) > 400",
            context
        ) is True


class TestGetConditionEvaluator:
    """Test the get_condition_evaluator factory function."""
    
    def test_boolean_evaluator_selection(self):
        """Test selection of BooleanEvaluator."""
        evaluator = get_condition_evaluator("true")
        assert isinstance(evaluator, BooleanEvaluator)
        
        evaluator = get_condition_evaluator("enabled")
        assert isinstance(evaluator, BooleanEvaluator)
    
    def test_comparison_evaluator_selection(self):
        """Test selection of ComparisonEvaluator."""
        evaluator = get_condition_evaluator("x > 5")
        assert isinstance(evaluator, ComparisonEvaluator)
        
        evaluator = get_condition_evaluator("name == 'test'")
        assert isinstance(evaluator, ComparisonEvaluator)
        
        evaluator = get_condition_evaluator("item in list")
        assert isinstance(evaluator, ComparisonEvaluator)
    
    def test_logical_evaluator_selection(self):
        """Test selection of LogicalEvaluator."""
        evaluator = get_condition_evaluator("x > 5 and y < 10")
        assert isinstance(evaluator, LogicalEvaluator)
        
        evaluator = get_condition_evaluator("enabled or disabled")
        assert isinstance(evaluator, LogicalEvaluator)
        
        evaluator = get_condition_evaluator("not active")
        assert isinstance(evaluator, LogicalEvaluator)
    
    def test_template_evaluator_selection(self):
        """Test selection of TemplateEvaluator."""
        evaluator = get_condition_evaluator("{{ x }} > 5")
        assert isinstance(evaluator, TemplateEvaluator)
        
        evaluator = get_condition_evaluator("{{ user.enabled }} == true")
        assert isinstance(evaluator, TemplateEvaluator)
    
    def test_expression_evaluator_selection(self):
        """Test selection of ExpressionEvaluator."""
        evaluator = get_condition_evaluator("len(items) > 0")
        assert isinstance(evaluator, ExpressionEvaluator)
        
        evaluator = get_condition_evaluator("(x + y) * z > 100")
        assert isinstance(evaluator, ExpressionEvaluator)


class TestConditionEvaluatorIntegration:
    """Integration tests for condition evaluators."""
    
    @pytest.mark.asyncio
    async def test_execute_impl_success(self):
        """Test successful execution through _execute_impl."""
        evaluator = BooleanEvaluator()
        result = await evaluator.execute(condition="true")
        
        assert result["status"] == "success"
        assert result["result"] is True
        assert result["condition"] == "true"
    
    @pytest.mark.asyncio 
    async def test_execute_impl_error(self):
        """Test error handling in _execute_impl."""
        evaluator = BooleanEvaluator()
        result = await evaluator.execute(condition="invalid !@#")
        
        assert result["status"] == "error"
        assert result["result"] is False
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_context_merging(self):
        """Test that additional parameters are merged into context."""
        evaluator = ComparisonEvaluator()
        result = await evaluator.execute(
            condition="value > threshold",
            value=10,
            threshold=5
        )
        
        assert result["status"] == "success"
        assert result["result"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])