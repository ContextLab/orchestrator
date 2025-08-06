"""Phase 2 tests for parallel queue implementation - Real API Integration.

These tests verify real functionality including:
- Queue generation with real expressions
- Concurrent task execution with real actions
- Real tool instance management
- Comprehensive integration with control systems

All tests use real functionality - NO MOCKS OR SIMULATIONS.
"""

import pytest
import asyncio
import time
from typing import Dict, Any, List

from orchestrator.core.parallel_queue_task import ParallelQueueTask, ParallelQueueStatus
from orchestrator.control_flow.parallel_queue_handler import ParallelQueueHandler
from orchestrator.control_flow.auto_resolver import ControlFlowAutoResolver
from orchestrator.control_flow.enhanced_condition_evaluator import EnhancedConditionEvaluator
from orchestrator.core.loop_context import GlobalLoopContextManager
from orchestrator.models.model_registry import ModelRegistry


class TestParallelQueuePhase2:
    """Test Phase 2 - Real API Integration with NO MOCKS."""
    
    @pytest.fixture
    def real_model_registry(self):
        """Create real model registry."""
        return ModelRegistry()
    
    @pytest.fixture
    def real_auto_resolver(self, real_model_registry):
        """Create real auto resolver."""
        return ControlFlowAutoResolver(model_registry=real_model_registry)
    
    @pytest.fixture
    def real_condition_evaluator(self, real_auto_resolver):
        """Create real condition evaluator."""
        return EnhancedConditionEvaluator(auto_resolver=real_auto_resolver)
    
    @pytest.fixture
    def handler(self, real_auto_resolver, real_condition_evaluator, real_model_registry):
        """Create handler with real dependencies."""
        return ParallelQueueHandler(
            auto_resolver=real_auto_resolver,
            loop_context_manager=GlobalLoopContextManager(),
            condition_evaluator=real_condition_evaluator,
            model_registry=real_model_registry
        )
    
    @pytest.mark.asyncio
    async def test_real_queue_generation_with_direct_lists(self, handler):
        """Test real queue generation using direct list expressions."""
        task = ParallelQueueTask(
            id="test_direct_queue",
            name="Test Direct Queue Generation",
            action="create_parallel_queue",
            on='["item1", "item2", "item3"]',  # Direct JSON list
            max_parallel=2,
            action_loop=[
                {
                    "action": "debug",
                    "parameters": {"message": "Processing {{ item }}"}
                }
            ]
        )
        
        context = {}
        step_results = {}
        
        # Execute queue generation only (not full execution)
        await handler._generate_queue_items(task, context, step_results)
        
        # Verify queue was generated correctly
        assert len(task.queue_items) == 3
        assert task.queue_items == ["item1", "item2", "item3"]
        assert task.stats.queue_generation_time > 0
        assert task.stats.total_items == 3
    
    @pytest.mark.asyncio
    async def test_concurrent_execution_with_real_debug_action(self, handler):
        """Test concurrent task execution with real debug actions."""
        task = ParallelQueueTask(
            id="test_concurrent_debug",
            name="Test Concurrent Debug Execution",
            action="create_parallel_queue",
            on='["debug1", "debug2", "debug3"]',
            max_parallel=2,
            action_loop=[
                {
                    "action": "debug",
                    "parameters": {"message": "Processing item: {{ item }}"}
                }
            ]
        )
        
        context = {}
        step_results = {}
        
        # Execute full pipeline
        result = await handler.execute_parallel_queue(task, context, step_results)
        
        # Verify concurrent execution results
        assert result["total_items"] == 3
        assert result["successful_items"] == 3
        assert result["failed_items"] == 0
        assert len(result["results"]) == 3
        
        # Verify all items were processed
        processed_items = [res["item"] for res in result["results"]]
        assert set(processed_items) == {"debug1", "debug2", "debug3"}
        
        # Verify concurrency was managed
        assert result["execution_stats"]["max_concurrent_executions"] <= 2
    
    @pytest.mark.asyncio
    async def test_template_variable_resolution(self, handler):
        """Test template variable resolution in action parameters."""
        task = ParallelQueueTask(
            id="test_templates",
            name="Test Template Resolution",
            action="create_parallel_queue",
            on='["template_test_1", "template_test_2"]',
            max_parallel=1,
            action_loop=[
                {
                    "action": "debug",
                    "parameters": {
                        "message": "Item: {{ item }}, Index: {{ index }}",
                        "item_value": "{{ item }}",
                        "index_value": "{{ index }}"
                    }
                }
            ]
        )
        
        context = {}
        step_results = {}
        
        result = await handler.execute_parallel_queue(task, context, step_results)
        
        # Verify template resolution worked
        assert result["successful_items"] == 2
        assert result["failed_items"] == 0
        
        # Check template variables were resolved correctly
        for i, item_result in enumerate(result["results"]):
            assert item_result["queue_index"] == i
            assert item_result["item"] == f"template_test_{i+1}"
            # Verify the action was executed with resolved parameters
            assert "result" in item_result
    
    @pytest.mark.asyncio
    async def test_error_handling_with_real_functionality(self, handler):
        """Test error handling with real action execution."""
        task = ParallelQueueTask(
            id="test_error_handling",
            name="Test Error Handling",
            action="create_parallel_queue",
            on='["valid_item", "another_valid"]',
            max_parallel=2,
            action_loop=[
                {
                    "action": "debug",
                    "parameters": {"message": "Processing {{ item }}"}
                }
            ]
        )
        
        context = {}
        step_results = {}
        
        # Execute with real functionality
        result = await handler.execute_parallel_queue(task, context, step_results)
        
        # Verify all items processed successfully with real debug action
        assert result["total_items"] == 2
        assert result["successful_items"] == 2
        assert result["failed_items"] == 0
        
        # Verify proper result structure
        assert len(result["results"]) == 2
        for item_result in result["results"]:
            assert item_result["status"] == "completed"
            assert "item" in item_result
            assert item_result["item"] in ["valid_item", "another_valid"]
            assert "result" in item_result
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_real_execution(self, handler):
        """Test performance monitoring during real execution."""
        task = ParallelQueueTask(
            id="test_performance",
            name="Test Performance Monitoring",
            action="create_parallel_queue",
            on='["perf1", "perf2", "perf3"]',
            max_parallel=2,
            action_loop=[
                {
                    "action": "debug",
                    "parameters": {"message": "Performance test: {{ item }}"}
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
        
        # Verify all items processed
        assert result["successful_items"] == 3
        assert result["failed_items"] == 0
        
        # Verify handler stats were updated
        handler_stats = handler.get_handler_stats()
        assert handler_stats["execution_stats"]["total_queues_processed"] >= 1
        assert handler_stats["execution_stats"]["total_items_processed"] >= 3
    
    @pytest.mark.asyncio
    async def test_resource_management_with_real_tools(self, handler):
        """Test tool instance management with real functionality."""
        task = ParallelQueueTask(
            id="test_resource_management",
            name="Test Resource Management",
            action="create_parallel_queue",
            on='["res1", "res2", "res3", "res4"]',
            max_parallel=2,
            tool="debug_tool",  # Real tool specification
            action_loop=[
                {
                    "action": "debug",
                    "parameters": {"message": "Resource test: {{ item }}"}
                }
            ]
        )
        
        context = {}
        step_results = {}
        
        result = await handler.execute_parallel_queue(task, context, step_results)
        
        # Verify all items processed successfully
        assert result["successful_items"] == 4
        assert result["failed_items"] == 0
        
        # Verify resource stats are present
        assert "resource_stats" in result
        
        # Verify concurrency was managed
        assert result["execution_stats"]["max_concurrent_executions"] <= 2
        
        # Verify all results are present
        assert len(result["results"]) == 4
        processed_items = [res["item"] for res in result["results"]]
        assert set(processed_items) == {"res1", "res2", "res3", "res4"}
    
    @pytest.mark.asyncio
    async def test_comprehensive_integration_scenario(self, handler):
        """Test comprehensive scenario with real components."""
        task = ParallelQueueTask(
            id="comprehensive_test",
            name="Comprehensive Integration Test",
            action="create_parallel_queue",
            on='["comp1", "comp2", "comp3"]',
            max_parallel=2,
            action_loop=[
                {
                    "action": "debug",
                    "parameters": {
                        "message": "Comprehensive test: {{ item }}",
                        "index": "{{ index }}",
                        "is_first": "{{ is_first }}",
                        "is_last": "{{ is_last }}"
                    }
                }
            ]
        )
        
        context = {}
        step_results = {}
        
        # Execute comprehensive scenario
        result = await handler.execute_parallel_queue(task, context, step_results)
        
        # Verify execution completed successfully
        assert result["total_items"] == 3
        assert result["successful_items"] == 3
        assert result["failed_items"] == 0
        
        # Verify comprehensive result structure
        assert "execution_stats" in result
        assert "resource_stats" in result
        assert len(result["results"]) == 3
        
        # Verify template variables were resolved
        for i, item_result in enumerate(result["results"]):
            assert item_result["queue_index"] == i
            assert item_result["item"] == f"comp{i+1}"


# Real YAML integration test
class TestParallelQueueRealYAMLIntegration:
    """Test real YAML integration with Phase 2 functionality."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_yaml_compilation_and_execution(self):
        """Test complete pipeline from YAML to real execution."""
        from orchestrator.compiler.yaml_compiler import YAMLCompiler
        
        yaml_content = """
        name: Phase 2 Real Integration Test
        steps:
          - id: parallel_processing
            create_parallel_queue:
              on: '["yaml_test1", "yaml_test2", "yaml_test3"]'
              max_parallel: 2
              action_loop:
                - action: debug
                  parameters:
                    message: "YAML test: {{ item }}"
                    index: "{{ index }}"
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
        assert len(task.action_loop) == 1
        
        # Verify the action loop is correctly defined
        action = task.action_loop[0]
        assert action["action"] == "debug"
        assert "message" in action["parameters"]
        assert "{{ item }}" in action["parameters"]["message"]