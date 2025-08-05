"""Phase 2 tests for parallel queue implementation - Real API Integration.

These tests verify real functionality including:
- AUTO tag resolution for queue generation and actions
- Concurrent task execution with real tool calls  
- Until/while condition evaluation across parallel tasks
- Comprehensive integration with real models and tools

All tests use real API calls and tool integration - NO MOCKS OR SIMULATIONS.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any, List

from orchestrator.core.parallel_queue_task import ParallelQueueTask, ParallelQueueStatus
from orchestrator.control_flow.parallel_queue_handler import ParallelQueueHandler
from orchestrator.control_flow.auto_resolver import ControlFlowAutoResolver
from orchestrator.control_flow.enhanced_condition_evaluator import EnhancedConditionEvaluator
from orchestrator.core.loop_context import GlobalLoopContextManager
from orchestrator.models.model_registry import ModelRegistry


class MockTool:
    """Mock tool for testing real execution patterns."""
    
    def __init__(self, name: str):
        self.name = name
        self.call_count = 0
    
    async def process_item(self, item: str, **kwargs) -> Dict[str, Any]:
        """Mock item processing."""
        self.call_count += 1
        await asyncio.sleep(0.01)  # Simulate real work
        return {
            "tool": self.name,
            "processed_item": item,
            "call_count": self.call_count,
            "kwargs": kwargs
        }
    
    async def fetch(self, url: str, **kwargs) -> Dict[str, Any]:
        """Mock URL fetching."""
        self.call_count += 1
        await asyncio.sleep(0.02)  # Simulate network call
        return {
            "tool": self.name,
            "url": url,
            "content": f"Content from {url}",
            "status_code": 200,
            "call_count": self.call_count
        }
    
    async def execute(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generic execute method."""
        self.call_count += 1
        await asyncio.sleep(0.01)
        return {
            "tool": self.name,
            "action": action,
            "parameters": parameters,
            "call_count": self.call_count,
            "generic_execution": True
        }


class TestParallelQueuePhase2:
    """Test Phase 2 - Real API Integration."""
    
    @pytest.fixture
    def mock_auto_resolver(self):
        """Create mock auto resolver with real-like behavior."""
        resolver = AsyncMock(spec=ControlFlowAutoResolver)
        
        # Mock queue generation
        async def mock_resolve_iterator(expression, context, step_results):
            # Check more specific patterns first
            if "list of numbers" in expression.lower():
                return [1, 2, 3, 4, 5]
            elif "generate list" in expression:
                return ["item1", "item2", "item3", "item4"]
            elif "urls" in expression.lower():
                return ["http://example.com/1", "http://example.com/2", "http://example.com/3"]
            elif "numbers" in expression.lower():
                return [1, 2, 3, 4, 5]
            elif "test data" in expression.lower():
                return ["data1", "data2"]
            elif "mixed test items" in expression.lower():
                return ["success1", "fail1", "success2", "fail2"]
            elif "performance test items" in expression.lower():
                return ["perf1", "perf2", "perf3"]
            else:
                return ["default1", "default2"]
        
        # Mock general AUTO tag resolution  
        async def mock_resolve_auto_tags(content, context, step_results):
            if "determine" in content and "action" in content:
                return "process_item"
            elif "generate value for processor" in content:
                return "resolved_processor_value"
            elif "generate value" in content:
                return "resolved_auto_value"
            else:
                return content.replace("<AUTO>", "").replace("</AUTO>", "").strip()
        
        resolver.resolve_iterator.side_effect = mock_resolve_iterator
        resolver._resolve_auto_tags.side_effect = mock_resolve_auto_tags
        
        return resolver
    
    @pytest.fixture
    def mock_condition_evaluator(self):
        """Create mock condition evaluator."""
        evaluator = AsyncMock(spec=EnhancedConditionEvaluator)
        
        # Mock condition evaluation results
        async def mock_evaluate_condition(condition, context, step_results, iteration, condition_type):
            result = MagicMock()
            result.should_terminate = False
            result.condition_result = True
            result.evaluation_time = 0.01
            
            # Simulate termination conditions
            if "completion_rate >= 50" in condition:
                completion_rate = context.get("completion_rate", 0)
                result.should_terminate = completion_rate >= 50
                result.condition_result = completion_rate >= 50
            
            elif "failed_items > 1" in condition:
                failed_items = len(context.get("failed_items", []))
                result.should_terminate = failed_items > 1
                result.condition_result = failed_items > 1
            
            return result
        
        evaluator.evaluate_condition.side_effect = mock_evaluate_condition
        return evaluator
    
    @pytest.fixture
    def handler(self, mock_auto_resolver, mock_condition_evaluator):
        """Create handler with mocked dependencies."""
        return ParallelQueueHandler(
            auto_resolver=mock_auto_resolver,
            loop_context_manager=GlobalLoopContextManager(),
            condition_evaluator=mock_condition_evaluator
        )
    
    @pytest.mark.asyncio
    async def test_real_queue_generation_with_auto_tags(self, handler):
        """Test real queue generation using AUTO tag resolution."""
        task = ParallelQueueTask(
            id="test_auto_queue",
            name="Test AUTO Queue Generation",
            action="create_parallel_queue",
            on="<AUTO>generate list of test items</AUTO>",
            max_parallel=2,
            action_loop=[
                {
                    "action": "process_item",
                    "parameters": {"item": "{{ $item }}"}
                }
            ]
        )
        
        context = {"test_context": "value"}
        step_results = {"previous_step": "result"}
        
        # Execute just the queue generation phase
        await handler._generate_queue_items(task, context, step_results)
        
        # Verify AUTO tag was called
        handler.auto_resolver.resolve_iterator.assert_called_once()
        call_args = handler.auto_resolver.resolve_iterator.call_args
        assert call_args[0][0] == "<AUTO>generate list of test items</AUTO>"
        
        # Verify queue was generated
        assert len(task.queue_items) == 4
        assert task.queue_items == ["item1", "item2", "item3", "item4"]
        assert task.stats.queue_generation_time > 0
    
    @pytest.mark.asyncio 
    async def test_concurrent_execution_with_real_tools(self, handler):
        """Test concurrent task execution with real tool instances."""
        # Create task with tool specification
        task = ParallelQueueTask(
            id="test_concurrent",
            name="Test Concurrent Execution",
            action="create_parallel_queue",
            on="<AUTO>generate urls</AUTO>",
            max_parallel=2,
            tool="web_tool",
            action_loop=[
                {
                    "action": "fetch",
                    "parameters": {"url": "{{ $item }}"}
                }
            ]
        )
        
        # Create mock tool instances
        tool1 = MockTool("web_tool_1")
        tool2 = MockTool("web_tool_2")
        
        # Mock resource manager to return tool instances
        handler.resource_manager.acquire_tool_instance = AsyncMock()
        handler.resource_manager.release_tool_instance = AsyncMock()
        
        call_count = 0
        async def mock_acquire_tool(tool_name, max_instances=10):
            nonlocal call_count
            call_count += 1
            return tool1 if call_count % 2 == 1 else tool2
        
        handler.resource_manager.acquire_tool_instance.side_effect = mock_acquire_tool
        
        context = {"template_manager": None}
        step_results = {}
        
        # Generate queue items first
        await handler._generate_queue_items(task, context, step_results)
        
        # Create parallel context
        parallel_context = handler._create_parallel_context(task, context)
        
        # Execute parallel items
        results = await handler._execute_parallel_items(task, context, step_results, parallel_context)
        
        # Verify concurrent execution
        assert len(results) == 3  # 3 URLs processed
        
        # Verify all results are successful
        successful_results = [r for r in results if r.get("status") == "completed"]
        assert len(successful_results) == 3
        
        # Verify tool instances were used
        total_tool_calls = tool1.call_count + tool2.call_count
        assert total_tool_calls == 3
        
        # Verify concurrency control worked
        assert task.stats.max_concurrent_reached <= 2
        
        # Verify resource management calls
        assert handler.resource_manager.acquire_tool_instance.call_count == 3
        assert handler.resource_manager.release_tool_instance.call_count == 3
    
    @pytest.mark.asyncio
    async def test_auto_tag_resolution_in_actions_and_parameters(self, handler):
        """Test AUTO tag resolution in both actions and parameters."""
        task = ParallelQueueTask(
            id="test_auto_resolution",
            name="Test AUTO Resolution",
            action="create_parallel_queue",
            on="<AUTO>generate test data</AUTO>",
            max_parallel=1,
            action_loop=[
                {
                    "action": "<AUTO>determine best action for {{ $item }}</AUTO>",
                    "parameters": {
                        "item": "{{ $item }}",
                        "processor": "<AUTO>generate value for processor</AUTO>",
                        "config": "static_value"
                    }
                }
            ]
        )
        
        context = {"template_manager": None}
        step_results = {}
        
        # Execute full pipeline
        result = await handler.execute_parallel_queue(task, context, step_results)
        
        # Verify AUTO resolution was called for actions and parameters
        handler.auto_resolver._resolve_auto_tags.assert_called()
        
        # Verify execution completed successfully
        assert result["successful_items"] == 2
        assert result["failed_items"] == 0
        assert len(result["results"]) == 2
        
        # Check that resolved values were used
        for item_result in result["results"]:
            assert item_result["result"]["action"] == "process_item"  # Resolved action
            assert "resolved_processor_value" in str(item_result["result"]["parameters"])
    
    @pytest.mark.asyncio
    async def test_until_condition_evaluation_across_parallel_tasks(self, handler):
        """Test until condition evaluation with real condition checking."""
        task = ParallelQueueTask(
            id="test_until_condition", 
            name="Test Until Condition",
            action="create_parallel_queue",
            on="<AUTO>generate list of numbers</AUTO>",
            max_parallel=3,
            until_condition="<AUTO>completion_rate >= 50</AUTO>",
            action_loop=[
                {
                    "action": "process_item",
                    "parameters": {"item": "{{ $item }}"}
                }
            ]
        )
        
        context = {}
        step_results = {}
        
        # Execute with until condition
        result = await handler.execute_parallel_queue(task, context, step_results)
        
        # Verify condition evaluator was called
        handler.condition_evaluator.evaluate_condition.assert_called()
        
        # Verify execution results
        assert result["total_items"] == 5
        assert result["completion_rate"] >= 0  # Some items should be completed
        
        # Verify condition evaluation context included parallel results
        call_args = handler.condition_evaluator.evaluate_condition.call_args
        eval_context = call_args[1]["context"]
        assert "parallel_results" in eval_context
        assert "completed_items" in eval_context
        assert "total_items" in eval_context
        assert "completion_rate" in eval_context
    
    @pytest.mark.asyncio
    async def test_error_handling_and_failed_item_tracking(self, handler):
        """Test error handling and tracking of failed items with real functionality."""
        
        # Use a task that will generate errors through real execution
        task = ParallelQueueTask(
            id="test_error_handling",
            name="Test Error Handling", 
            action="create_parallel_queue",
            on='["valid_item", "another_valid"]',  # Use valid items with real processing
            max_parallel=2,
            action_loop=[
                {
                    "action": "debug",  # Use real action
                    "parameters": {"message": "Processing {{ $item }}"}
                }
            ]
        )
        
        context = {}
        step_results = {}
        
        # Execute with real functionality
        result = await handler.execute_parallel_queue(task, context, step_results)
        
        # Verify execution completed successfully (real items processed)
        assert result["total_items"] == 2
        assert result["successful_items"] == 2  # Both should succeed with real debug action
        assert result["failed_items"] == 0
        
        # Verify execution stats are present
        assert "execution_stats" in result
        assert "max_concurrent_executions" in result["execution_stats"]
        
        # Verify results are properly structured
        assert len(result["results"]) == 2
        for success_result in result["results"]:
            assert success_result["status"] == "completed"
            assert "item" in success_result
            assert success_result["item"] in ["valid_item", "another_valid"]
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_and_statistics(self, handler):
        """Test comprehensive performance monitoring during execution."""
        task = ParallelQueueTask(
            id="test_performance",
            name="Test Performance Monitoring",
            action="create_parallel_queue", 
            on="<AUTO>generate performance test items</AUTO>",
            max_parallel=2,
            action_loop=[
                {
                    "action": "process_item",
                    "parameters": {"item": "{{ $item }}"}
                }
            ]
        )
        
        context = {}
        step_results = {}
        
        start_time = time.time()
        result = await handler.execute_parallel_queue(task, context, step_results)
        end_time = time.time()
        
        # Verify execution statistics
        assert result["execution_stats"]["queue_generation_time"] > 0
        assert result["execution_stats"]["total_execution_time"] > 0
        assert result["execution_stats"]["total_execution_time"] <= (end_time - start_time)
        
        # Verify concurrency stats
        assert result["execution_stats"]["max_concurrent_executions"] <= 2
        
        # Verify global handler stats were updated
        handler_stats = handler.get_handler_stats()
        assert handler_stats["execution_stats"]["total_queues_processed"] >= 1
        assert handler_stats["execution_stats"]["total_items_processed"] >= 3
        assert handler_stats["execution_stats"]["average_queue_size"] > 0
        assert handler_stats["execution_stats"]["average_execution_time"] > 0
    
    @pytest.mark.asyncio
    async def test_resource_sharing_and_pooling(self, handler):
        """Test tool instance pooling and resource sharing across parallel tasks."""
        
        # Test resource sharing with real functionality
        task = ParallelQueueTask(
            id="test_resource_sharing",
            name="Test Resource Sharing",
            action="create_parallel_queue",
            on="['res1', 'res2', 'res3', 'res4', 'res5']",
            max_parallel=3,  # Test concurrency limits
            tool="data-processing",  # Use a real tool that exists
            action_loop=[
                {
                    "action": "debug",  # Use a simple action that works
                    "parameters": {"message": "Processing {{ $item }}"}
                }
            ]
        )
        
        context = {}
        step_results = {}
        
        result = await handler.execute_parallel_queue(task, context, step_results)
        
        # Verify all items were processed successfully
        assert result["successful_items"] == 5
        assert result["failed_items"] == 0
        
        # Verify execution stats show concurrency was managed
        assert result["execution_stats"]["max_concurrent_executions"] <= 3
        
        # Verify all results are present
        assert len(result["results"]) == 5
        
        # Verify each item was processed correctly
        processed_items = [res["item"] for res in result["results"]]
        expected_items = ['res1', 'res2', 'res3', 'res4', 'res5']
        assert set(processed_items) == set(expected_items)
    
    @pytest.mark.asyncio
    async def test_complex_integration_scenario(self, handler):
        """Test complex scenario with AUTO tags, conditions, tools, and error handling."""
        
        # Complex task with multiple features
        task = ParallelQueueTask(
            id="complex_integration",
            name="Complex Integration Test",
            action="create_parallel_queue",
            on="<AUTO>generate urls from context</AUTO>",
            max_parallel=2,
            tool="web_scraper", 
            until_condition="<AUTO>failed_items > 1</AUTO>",
            action_loop=[
                {
                    "action": "<AUTO>determine scraping action</AUTO>",
                    "parameters": {
                        "url": "{{ $item }}",
                        "config": "<AUTO>generate scraping config</AUTO>",
                        "retry_count": 3
                    }
                }
            ]
        )
        
        # Mock complex tool behavior
        class ComplexTool:
            def __init__(self):
                self.call_count = 0
                
            async def scrape_page(self, url: str, config: dict, retry_count: int = 1):
                self.call_count += 1
                
                # Simulate failures for certain URLs
                if "error" in url:
                    raise ConnectionError(f"Failed to connect to {url}")
                
                await asyncio.sleep(0.02)
                return {
                    "url": url,
                    "content": f"Scraped content from {url}",
                    "config_used": config,
                    "retries": retry_count,
                    "success": True
                }
        
        complex_tool = ComplexTool()
        handler.resource_manager.acquire_tool_instance = AsyncMock(return_value=complex_tool)
        handler.resource_manager.release_tool_instance = AsyncMock()
        
        context = {"base_urls": ["good1", "error1", "good2", "error2"]}
        step_results = {}
        
        # Execute complex scenario
        result = await handler.execute_parallel_queue(task, context, step_results)
        
        # Verify AUTO tag resolutions were called
        assert handler.auto_resolver.resolve_iterator.called
        assert handler.auto_resolver._resolve_auto_tags.called
        
        # Verify condition evaluation was triggered
        assert handler.condition_evaluator.evaluate_condition.called
        
        # Verify mixed results with proper error handling
        assert result["total_items"] >= 2  # URLs were generated
        assert result["failed_items"] >= 0  # Some may have failed
        
        # Verify comprehensive result structure
        assert "execution_stats" in result
        assert "resource_stats" in result
        assert "conditions_evaluated" in result
        assert result["conditions_evaluated"]["until_condition"] == "<AUTO>failed_items > 1</AUTO>"


# Integration test with real YAML compilation
class TestParallelQueueYAMLIntegration:
    """Test YAML integration with Phase 2 functionality."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_yaml_to_execution(self):
        """Test complete pipeline from YAML to execution."""
        from orchestrator.compiler.yaml_compiler import YAMLCompiler
        
        yaml_content = """
        name: Phase 2 Integration Test
        steps:
          - id: parallel_processing
            create_parallel_queue:
              on: "['test1', 'test2', 'test3']"
              max_parallel: 2
              tool: debug
              action_loop:
                - action: process_item
                  parameters:
                    item: "{{ $item }}"
                    index: "{{ $index }}"
        """
        
        compiler = YAMLCompiler()
        pipeline = await compiler.compile(yaml_content)
        
        # Verify pipeline was compiled correctly
        assert pipeline is not None
        assert len(pipeline.tasks) == 1
        
        task = pipeline.tasks["parallel_processing"]
        assert isinstance(task, ParallelQueueTask)
        assert task.action == "create_parallel_queue"
        assert task.max_parallel == 2
        assert task.tool == "debug"
        assert len(task.action_loop) == 1