"""Tests for Phase 3: Conditional execution, loops, and error handling."""

import pytest
import asyncio
from orchestrator.engine.advanced_executor import (
    ConditionalExecutor, LoopExecutor, ErrorRecoveryExecutor, AdvancedTaskExecutor
)
from orchestrator.engine.pipeline_spec import TaskSpec, ErrorHandling, LoopSpec


class TestConditionalExecutor:
    """Test conditional execution logic."""
    
    @pytest.fixture
    def conditional_executor(self):
        """Create a test conditional executor instance."""
        return ConditionalExecutor()
    
    def test_simple_conditions(self, conditional_executor):
        """Test simple conditional expressions."""
        test_cases = [
            ("{{status}} == 'success'", {"status": "success"}, True),
            ("{{count}} > 5", {"count": 10}, True),
            ("{{count}} < 5", {"count": 3}, True),
            ("{{enabled}} == false", {"enabled": False}, True),
            ("{{status}} != 'failed'", {"status": "success"}, True),
            ("{{value}} >= 10", {"value": 10}, True),
            ("{{value}} <= 10", {"value": 5}, True),
        ]
        
        for condition, context, expected in test_cases:
            result = conditional_executor.evaluate_condition(condition, context)
            assert result == expected, f"Failed for condition: {condition}"
    
    def test_nested_variable_access(self, conditional_executor):
        """Test conditions with nested variable access."""
        context = {
            "results": {"length": 5, "status": "complete"},
            "config": {"enabled": True}
        }
        
        assert conditional_executor.evaluate_condition("{{results.length}} >= 3", context) is True
        assert conditional_executor.evaluate_condition("{{results.status}} == 'complete'", context) is True
        assert conditional_executor.evaluate_condition("{{config.enabled}} == true", context) is True
    
    def test_boolean_operators(self, conditional_executor):
        """Test boolean operators in conditions."""
        context = {"a": 5, "b": 10, "status": "active"}
        
        # AND operator
        condition = "{{a}} > 0 and {{b}} > 0"
        assert conditional_executor.evaluate_condition(condition, context) is True
        
        # OR operator
        condition = "{{a}} > 10 or {{b}} > 5"
        assert conditional_executor.evaluate_condition(condition, context) is True
    
    def test_edge_cases(self, conditional_executor):
        """Test edge cases in condition evaluation."""
        # Empty condition - should be true
        assert conditional_executor.evaluate_condition("", {}) is True
        
        # Missing variable - should be false
        assert conditional_executor.evaluate_condition("{{missing}} == true", {}) is False
        
        # Invalid syntax - should be false
        assert conditional_executor.evaluate_condition("invalid syntax", {}) is False


class TestLoopExecutor:
    """Test loop execution functionality."""
    
    @pytest.fixture
    def mock_executor(self):
        """Create a mock task executor for testing."""
        class MockTaskExecutor:
            async def execute_task(self, task_spec, context):
                return {
                    "task_id": task_spec.id,
                    "success": True,
                    "result": f"Processed: {context.get('loop_item', 'unknown')}",
                    "loop_index": context.get("loop_index", -1)
                }
        return MockTaskExecutor()
    
    @pytest.fixture
    def loop_executor(self, mock_executor):
        """Create a test loop executor instance."""
        return LoopExecutor(mock_executor)
    
    @pytest.mark.asyncio
    async def test_sequential_loop(self, loop_executor):
        """Test sequential loop execution."""
        task_spec = TaskSpec(
            id="test_loop",
            action="process item",
            loop=LoopSpec(
                foreach="{{items}}",
                collect_results=True,
                parallel=False
            )
        )
        
        context = {"items": ["apple", "banana", "cherry"]}
        result = await loop_executor.execute_loop(task_spec, context)
        
        assert result["iteration_count"] == 3
        assert result["loop_completed"] is True
        assert len(result["loop_results"]) == 3
        assert all(r["success"] for r in result["loop_results"])
    
    @pytest.mark.asyncio
    async def test_parallel_loop(self, loop_executor):
        """Test parallel loop execution."""
        task_spec = TaskSpec(
            id="test_parallel",
            action="process item",
            loop=LoopSpec(
                foreach="{{items}}",
                collect_results=True,
                parallel=True
            )
        )
        
        context = {"items": [1, 2, 3, 4, 5]}
        result = await loop_executor.execute_loop(task_spec, context)
        
        assert result["iteration_count"] == 5
        assert result["execution_mode"] == "parallel"
        assert len(result["loop_results"]) == 5
    
    @pytest.mark.asyncio
    async def test_max_iterations(self, loop_executor):
        """Test max iterations limit."""
        task_spec = TaskSpec(
            id="test_limited",
            action="process item",
            loop=LoopSpec(
                foreach="{{items}}",
                max_iterations=3,
                collect_results=True
            )
        )
        
        context = {"items": list(range(10))}  # 10 items
        result = await loop_executor.execute_loop(task_spec, context)
        
        assert result["iteration_count"] == 3  # Limited to 3
        assert len(result["loop_results"]) == 3
    
    def test_loop_item_resolution(self, loop_executor):
        """Test resolution of loop items from context."""
        # List
        items = loop_executor._resolve_loop_items("{{data}}", {"data": [1, 2, 3]})
        assert items == [1, 2, 3]
        
        # Dict
        items = loop_executor._resolve_loop_items("{{config}}", {"config": {"a": 1, "b": 2}})
        assert len(items) == 2
        assert all(isinstance(item, tuple) for item in items)
        
        # String
        items = loop_executor._resolve_loop_items("{{text}}", {"text": "abc"})
        assert items == ["a", "b", "c"]
        
        # Empty
        items = loop_executor._resolve_loop_items("{{empty}}", {"empty": []})
        assert items == []


class TestErrorRecoveryExecutor:
    """Test error recovery and retry logic."""
    
    @pytest.fixture
    def failing_executor(self):
        """Create a failing task executor for testing."""
        class FailingTaskExecutor:
            def __init__(self, fail_count=2):
                self.attempt_count = 0
                self.fail_count = fail_count
            
            async def execute_task(self, task_spec, context):
                self.attempt_count += 1
                
                if self.attempt_count <= self.fail_count:
                    raise Exception(f"Simulated failure #{self.attempt_count}")
                
                return {
                    "task_id": task_spec.id,
                    "success": True,
                    "result": f"Success after {self.attempt_count} attempts"
                }
        
        return FailingTaskExecutor()
    
    @pytest.mark.asyncio
    async def test_retry_logic(self, failing_executor):
        """Test retry logic with exponential backoff."""
        error_recovery = ErrorRecoveryExecutor(failing_executor)
        
        task_spec = TaskSpec(
            id="retry_task",
            action="failing action",
            on_error=ErrorHandling(
                action="handle error",
                retry_count=3,
                retry_delay=0.01  # Short delay for testing
            )
        )
        
        result = await error_recovery.execute_with_error_handling(task_spec, {})
        
        assert result["success"] is True
        assert "Success after 3 attempts" in result["result"]
        assert failing_executor.attempt_count == 3
    
    @pytest.mark.asyncio
    async def test_error_handler_execution(self, failing_executor):
        """Test error handler execution when retries are exhausted."""
        failing_executor.fail_count = 5  # Will always fail
        error_recovery = ErrorRecoveryExecutor(failing_executor)
        
        task_spec = TaskSpec(
            id="always_fail",
            action="failing action",
            on_error=ErrorHandling(
                action="<AUTO>handle the error</AUTO>",
                retry_count=2,
                continue_on_error=True,
                fallback_value="default_value"
            )
        )
        
        result = await error_recovery.execute_with_error_handling(task_spec, {})
        
        assert result["success"] is False
        assert result["error_handled"] is True
        assert result["fallback_value"] == "default_value"
        assert result["continue_pipeline"] is True


class TestAdvancedTaskExecutor:
    """Test the complete advanced task executor."""
    
    @pytest.fixture
    def executor(self):
        """Create a test advanced executor instance."""
        return AdvancedTaskExecutor()
    
    def test_task_metadata_extraction(self):
        """Test extraction of task execution metadata."""
        task_spec = TaskSpec(
            id="advanced_task",
            action="<AUTO>complex task</AUTO>",
            condition="{{enabled}} == true",
            loop=LoopSpec(foreach="{{items}}"),
            on_error=ErrorHandling(action="handle error"),
            timeout=30.0,
            cache_results=True,
            tags=["test", "advanced"]
        )
        
        metadata = task_spec.get_execution_metadata()
        
        assert metadata["has_condition"] is True
        assert metadata["has_loop"] is True
        assert metadata["has_error_handling"] is True
        assert metadata["is_iterative"] is True
        assert metadata["is_conditional"] is True
        assert metadata["timeout"] == 30.0
        assert metadata["cache_results"] is True
        assert "test" in metadata["tags"]
    
    def test_cache_key_generation(self, executor):
        """Test cache key generation for task results."""
        task_spec = TaskSpec(id="test", action="test action")
        context = {"input": "test", "data": [1, 2, 3]}
        
        key1 = executor._generate_cache_key(task_spec, context)
        key2 = executor._generate_cache_key(task_spec, context)
        
        # Same inputs should generate same key
        assert key1 == key2
        
        # Different context should generate different key
        context2 = {"input": "different", "data": [1, 2, 3]}
        key3 = executor._generate_cache_key(task_spec, context2)
        assert key1 != key3
    
    @pytest.mark.asyncio
    async def test_conditional_skip(self, executor):
        """Test skipping tasks based on conditions."""
        task_spec = TaskSpec(
            id="conditional_task",
            action="test action",
            condition="{{skip}} == true"
        )
        
        context = {"skip": False}
        
        # Mock the execute_task_core to avoid model requirements
        async def mock_execute(*args):
            return {"success": True, "result": "executed"}
        
        executor._execute_task_core = mock_execute
        
        result = await executor.execute_task(task_spec, context)
        
        assert result["skipped"] is True
        assert result["reason"] == "condition_not_met"
    
    @pytest.mark.asyncio 
    async def test_timeout_handling(self, executor):
        """Test task timeout handling."""
        task_spec = TaskSpec(
            id="timeout_task",
            action="long running task",
            timeout=0.1  # 100ms timeout
        )
        
        # Mock a long-running task
        async def slow_task(*args):
            await asyncio.sleep(1)  # Sleep longer than timeout
            return {"success": True}
        
        executor._execute_task_core = slow_task
        
        result = await executor.execute_task(task_spec, {})
        
        assert result["success"] is False
        assert result["timeout"] is True
        assert "timed out" in result["error"]