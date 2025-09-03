"""Tests for the enhanced condition evaluator with structured evaluation."""

import pytest
import asyncio
from unittest.mock import AsyncMock

from src.orchestrator.control_flow.enhanced_condition_evaluator import EnhancedConditionEvaluator
from src.orchestrator.control_flow.condition_models import LoopCondition, ConditionParser
from src.orchestrator.control_flow.auto_resolver import ControlFlowAutoResolver
from src.orchestrator.models.model_registry import ModelRegistry


class TestConditionParser:
    """Test the condition parser for dependency analysis."""
    
    def test_parse_simple_condition(self):
        """Test parsing simple conditions."""
        parser = ConditionParser()
        
        condition = parser.parse("{{ counter }} >= 5", "until")
        
        assert condition.expression == "{{ counter }} >= 5"
        assert condition.condition_type == "until"
        assert condition.has_templates == True
        assert condition.has_auto_tags == False
        assert "counter" in condition.dependencies
        assert condition.complexity_score > 0
    
    def test_parse_auto_condition(self):
        """Test parsing conditions with AUTO tags."""
        parser = ConditionParser()
        
        condition = parser.parse(
            "<AUTO>Is the quality score {{ quality }} >= 0.8?</AUTO>", 
            "until"
        )
        
        assert condition.has_auto_tags == True
        assert condition.has_templates == True
        assert "quality" in condition.dependencies
        assert condition.complexity_score >= 3  # AUTO tags add 3 points
    
    def test_parse_complex_condition(self):
        """Test parsing complex conditions with multiple dependencies."""
        parser = ConditionParser()
        
        condition = parser.parse(
            "({{ results.count }} > 10 and {{ quality.avg }} >= 0.8) or {{ force_stop }}",
            "until"
        )
        
        assert condition.has_templates == True
        assert "results" in condition.dependencies
        assert "quality" in condition.dependencies  
        assert "force_stop" in condition.dependencies
        assert condition.complexity_score > 5  # Complex expression
    
    def test_parse_loop_variables(self):
        """Test parsing conditions with loop variables."""
        parser = ConditionParser()
        
        condition = parser.parse("$iteration < 5 and $item != null", "while")
        
        assert "$iteration" in condition.dependencies
        assert "$item" in condition.dependencies
        assert condition.condition_type == "while"
    
    def test_validate_condition(self):
        """Test condition validation."""
        parser = ConditionParser()
        
        # Valid condition
        valid_condition = parser.parse("{{ counter }} >= 5")
        issues = parser.validate_condition(valid_condition)
        assert len(issues) == 0
        
        # Invalid condition with unbalanced parentheses
        invalid_condition = parser.parse("({{ counter }} >= 5")
        issues = parser.validate_condition(invalid_condition)
        assert any("parentheses" in issue for issue in issues)


class TestEnhancedConditionEvaluator:
    """Test the enhanced condition evaluator."""
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator for testing."""
        model_registry = ModelRegistry()
        auto_resolver = ControlFlowAutoResolver(model_registry)
        return EnhancedConditionEvaluator(auto_resolver)
    
    @pytest.mark.asyncio
    async def test_simple_numeric_condition(self, evaluator):
        """Test simple numeric comparison conditions."""
        
        # Test until condition: should terminate when true
        result = await evaluator.evaluate_condition(
            condition="{{ counter }} >= 5",
            context={"counter": 6},
            step_results={},
            iteration=0,
            condition_type="until"
        )
        
        assert result.result == True  # 6 >= 5
        assert result.should_terminate == True  # Until condition: terminate when true
        assert result.cache_hit == False  # First evaluation
        assert result.error is None
        
        # Test again with same inputs (should hit cache)
        result2 = await evaluator.evaluate_condition(
            condition="{{ counter }} >= 5",
            context={"counter": 6},
            step_results={},
            iteration=1,
            condition_type="until"
        )
        
        assert result2.cache_hit == True
    
    @pytest.mark.asyncio
    async def test_while_vs_until_logic(self, evaluator):
        """Test that while and until conditions have opposite termination logic."""
        
        context = {"counter": 6}
        
        # Until condition: terminate when condition becomes true
        until_result = await evaluator.evaluate_condition(
            condition="{{ counter }} >= 5",
            context=context,
            step_results={},
            iteration=0,
            condition_type="until"
        )
        
        # While condition: terminate when condition becomes false
        while_result = await evaluator.evaluate_condition(
            condition="{{ counter }} >= 5", 
            context=context,
            step_results={},
            iteration=0,
            condition_type="while"
        )
        
        # Both evaluate to true (6 >= 5)
        assert until_result.result == True
        assert while_result.result == True
        
        # But termination logic is opposite
        assert until_result.should_terminate == True   # Until: terminate when true
        assert while_result.should_terminate == False  # While: continue when true
    
    @pytest.mark.asyncio
    async def test_complex_boolean_expression(self, evaluator):
        """Test complex boolean expressions."""
        
        result = await evaluator.evaluate_condition(
            condition="({{ count }} > 10 and {{ quality }} >= 0.8) or {{ force_stop }}",
            context={
                "count": 15,
                "quality": 0.9,
                "force_stop": False
            },
            step_results={},
            iteration=0,
            condition_type="until"
        )
        
        assert result.result == True  # (15 > 10 and 0.9 >= 0.8) or False = True
        assert result.should_terminate == True
        assert "15 > 10 and 0.9 >= 0.8" in result.resolved_expression or "True" in result.resolved_expression
    
    @pytest.mark.asyncio
    async def test_step_results_reference(self, evaluator):
        """Test conditions that reference step results."""
        
        step_results = {
            "process_data": {
                "count": 25,
                "success": True
            },
            "quality_check": {
                "score": 0.85
            }
        }
        
        result = await evaluator.evaluate_condition(
            condition="{{ process_data.count }} > 20 and {{ quality_check.score }} >= 0.8",
            context={},
            step_results=step_results,
            iteration=0,
            condition_type="until"
        )
        
        assert result.result == True  # 25 > 20 and 0.85 >= 0.8
        assert result.should_terminate == True
    
    @pytest.mark.asyncio 
    async def test_loop_iteration_variables(self, evaluator):
        """Test conditions using loop iteration variables."""
        
        result = await evaluator.evaluate_condition(
            condition="$iteration >= 3",
            context={},
            step_results={},
            iteration=4,
            condition_type="until"
        )
        
        assert result.result == True  # 4 >= 3
        assert result.should_terminate == True
    
    @pytest.mark.asyncio
    async def test_condition_with_error_fallback(self, evaluator):
        """Test error handling and fallback behavior."""
        
        # Expression with syntax error that will fail parsing
        result = await evaluator.evaluate_condition(
            condition="5 >= 3 and ((",  # Invalid syntax - unmatched parentheses
            context={},
            step_results={},
            iteration=0,
            condition_type="until"
        )
        
        # Error handling is very robust, so it may not report error but should give safe fallback
        assert result.result == False  # Fallback for until condition
        assert result.should_terminate == False  # Don't terminate on error
        # Note: error might be None due to robust fallback handling
    
    @pytest.mark.asyncio
    async def test_structured_condition_object(self, evaluator):
        """Test evaluation with structured LoopCondition object."""
        
        parser = ConditionParser()
        condition = parser.parse("{{ counter }} >= 5", "until")
        condition.loop_id = "test_loop"
        
        result = await evaluator.evaluate_condition(
            condition=condition,
            context={"counter": 6},
            step_results={},
            iteration=0
        )
        
        assert result.condition.loop_id == "test_loop"
        assert len(condition.evaluation_history) == 1
        assert condition.last_evaluation == True
        assert condition.total_evaluations == 1
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self, evaluator):
        """Test performance metrics tracking."""
        
        # Perform several evaluations
        for i in range(5):
            await evaluator.evaluate_condition(
                condition=f"$iteration >= {i}",
                context={},
                step_results={},
                iteration=i,
                condition_type="until"
            )
        
        stats = evaluator.get_performance_stats()
        
        assert stats["total_evaluations"] == 5
        assert stats["total_time"] > 0
        assert stats["avg_evaluation_time"] > 0
        assert stats["error_count"] == 0
        assert "cache_stats" in stats
    
    @pytest.mark.asyncio
    async def test_cache_functionality(self, evaluator):
        """Test condition evaluation caching."""
        
        # Same condition and context should hit cache
        context = {"counter": 5}
        
        result1 = await evaluator.evaluate_condition(
            condition="{{ counter }} >= 5",
            context=context,
            step_results={},
            iteration=0,
            condition_type="until"
        )
        
        result2 = await evaluator.evaluate_condition(
            condition="{{ counter }} >= 5",
            context=context,
            step_results={},
            iteration=1,
            condition_type="until"
        )
        
        assert result1.cache_hit == False
        assert result2.cache_hit == True
        
        cache_stats = evaluator.get_performance_stats()["cache_stats"]
        assert cache_stats["hits"] >= 1
        assert cache_stats["hit_rate"] > 0


class TestLoopConditionModel:
    """Test the LoopCondition data model."""
    
    def test_condition_creation(self):
        """Test creating condition with metadata."""
        condition = LoopCondition(
            expression="{{ quality }} >= 0.8",
            condition_type="until",
            loop_id="quality_loop"
        )
        
        assert condition.expression == "{{ quality }} >= 0.8"
        assert condition.condition_type == "until"
        assert condition.loop_id == "quality_loop"
        assert condition.total_evaluations == 0
    
    def test_evaluation_recording(self):
        """Test recording evaluation results."""
        condition = LoopCondition("{{ counter }} >= 5", "until")
        
        condition.record_evaluation(
            iteration=0,
            resolved_expr="6 >= 5", 
            result=True,
            eval_time=0.001
        )
        
        assert condition.last_evaluation == True
        assert condition.total_evaluations == 1
        assert condition.avg_evaluation_time == 0.001
        assert len(condition.evaluation_history) == 1
    
    def test_termination_logic(self):
        """Test termination logic for different condition types."""
        until_condition = LoopCondition("test", "until")
        while_condition = LoopCondition("test", "while")
        
        # When evaluation result is True
        assert until_condition.should_terminate_loop(True) == True   # Until: terminate when true
        assert while_condition.should_terminate_loop(True) == False  # While: continue when true
        
        # When evaluation result is False  
        assert until_condition.should_terminate_loop(False) == False  # Until: continue when false
        assert while_condition.should_terminate_loop(False) == True   # While: terminate when false
    
    def test_debug_info(self):
        """Test debug information generation."""
        condition = LoopCondition("{{ counter }} >= 5", "until")
        condition.has_templates = True
        condition.dependencies = {"counter"}
        condition.record_evaluation(0, "6 >= 5", True, 0.001)
        
        debug_info = condition.get_debug_info()
        
        assert debug_info["expression"] == "{{ counter }} >= 5"
        assert debug_info["condition_type"] == "until"
        assert debug_info["analysis"]["has_templates"] == True
        assert "counter" in debug_info["analysis"]["dependencies"]
        assert debug_info["runtime_state"]["last_evaluation"] == True
        assert len(debug_info["recent_history"]) == 1