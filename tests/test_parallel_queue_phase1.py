"""Phase 1 tests for parallel queue implementation - Core Infrastructure."""

import pytest
import asyncio
from unittest.mock import AsyncMock

from src.orchestrator.core.parallel_queue_task import ParallelQueueTask, ParallelQueueStatus, ParallelSubtask
from src.orchestrator.core.parallel_loop_context import ParallelQueueContext, ParallelLoopContextManager
from src.orchestrator.control_flow.parallel_queue_handler import ParallelQueueHandler, ParallelResourceManager
from src.orchestrator.compiler.yaml_compiler import YAMLCompiler
from src.orchestrator.control_flow.auto_resolver import ControlFlowAutoResolver
from src.orchestrator.models.model_registry import ModelRegistry

from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider


class TestParallelQueueTask:
    """Test the ParallelQueueTask model."""
    
    def test_parallel_queue_task_creation(self):
        """Test creating a ParallelQueueTask with required fields."""
        task = ParallelQueueTask(
            id="test_queue",
            name="Test Parallel Queue",
            action="create_parallel_queue",
            on="['item1', 'item2', 'item3']",
            max_parallel=2,
            action_loop=[
                {
                    "action": "process_item",
                    "parameters": {"item": "{{ $item }}"}
                }
            ]
        )
        
        assert task.id == "test_queue"
        assert task.action == "create_parallel_queue"
        assert task.on == "['item1', 'item2', 'item3']"
        assert task.max_parallel == 2
        assert len(task.action_loop) == 1
        assert task.queue_status == ParallelQueueStatus.INITIALIZING
        assert task.semaphore is not None
        assert task.semaphore._value == 2  # Max parallel limit
    
    def test_parallel_queue_task_validation(self):
        """Test validation of ParallelQueueTask fields."""
        # Test missing 'on' field
        with pytest.raises(ValueError, match="must have 'on' expression"):
            ParallelQueueTask(
                id="test",
                name="Test",
                action="create_parallel_queue",
                on="",  # Empty
                action_loop=[{"action": "test"}]
            )
        
        # Test invalid max_parallel
        with pytest.raises(ValueError, match="max_parallel must be positive"):
            ParallelQueueTask(
                id="test",
                name="Test", 
                action="create_parallel_queue",
                on="['item1']",
                max_parallel=0,  # Invalid
                action_loop=[{"action": "test"}]
            )
        
        # Test missing action_loop
        with pytest.raises(ValueError, match="must have action_loop defined"):
            ParallelQueueTask(
                id="test",
                name="Test",
                action="create_parallel_queue", 
                on="['item1']",
                action_loop=[]  # Empty
            )
    
    def test_queue_item_management(self):
        """Test adding items to queue and creating subtasks."""
        task = ParallelQueueTask(
            id="test_queue",
            name="Test Queue",
            action="create_parallel_queue",
            on="['item1', 'item2']",
            action_loop=[{"action": "process"}]
        )
        
        # Add items to queue
        idx1 = task.add_queue_item("item1")
        idx2 = task.add_queue_item("item2")
        
        assert idx1 == 0
        assert idx2 == 1
        assert len(task.queue_items) == 2
        assert task.stats.total_items == 2
        
        # Create subtask
        action_def = {"action": "process", "parameters": {"test": True}}
        subtask = task.create_subtask(0, "item1", action_def)
        
        assert subtask.queue_index == 0
        assert subtask.item == "item1"
        assert subtask.task.action == "process"
        assert subtask.task.metadata["parent_parallel_queue"] == "test_queue"
        assert subtask.task.metadata["is_parallel_subtask"] == True
    
    def test_context_variables(self):
        """Test context variable generation for queue items."""
        task = ParallelQueueTask(
            id="test_queue",
            name="Test Queue", 
            action="create_parallel_queue",
            on="['a', 'b', 'c']",
            action_loop=[{"action": "process"}]
        )
        
        # Add items
        for item in ['a', 'b', 'c']:
            task.add_queue_item(item)
        
        # Test context variables for each item
        ctx0 = task.get_context_variables(0)
        assert ctx0["$item"] == "a"
        assert ctx0["$index"] == 0
        assert ctx0["$is_first"] == True
        assert ctx0["$is_last"] == False
        assert ctx0["$queue_size"] == 3
        
        ctx2 = task.get_context_variables(2)
        assert ctx2["$item"] == "c"
        assert ctx2["$index"] == 2
        assert ctx2["$is_first"] == False
        assert ctx2["$is_last"] == True
    
    def test_from_task_definition(self):
        """Test creating ParallelQueueTask from YAML definition."""
        task_def = {
            "id": "parallel_test",
            "name": "Parallel Test",
            "create_parallel_queue": {
                "on": "<AUTO>generate list</AUTO>",
                "max_parallel": 5,
                "tool": "web",
                "action_loop": [
                    {
                        "action": "fetch",
                        "parameters": {"url": "{{ $item }}"}
                    }
                ],
                "until": "<AUTO>all items processed</AUTO>"
            }
        }
        
        task = ParallelQueueTask.from_task_definition(task_def)
        
        assert task.id == "parallel_test"
        assert task.name == "Parallel Test"
        assert task.action == "create_parallel_queue"
        assert task.on == "<AUTO>generate list</AUTO>"
        assert task.max_parallel == 5
        assert task.tool == "web"
        assert task.until_condition == "<AUTO>all items processed</AUTO>"
        assert len(task.action_loop) == 1


class TestParallelLoopContext:
    """Test the parallel loop context system."""
    
    def test_parallel_queue_context_creation(self):
        """Test creating parallel queue context."""
        context = ParallelQueueContext(
            item="test_item",
            index=0,
            items=["test_item", "item2"],
            length=2,
            loop_name="test_loop",
            loop_id="test_step",
            is_auto_generated=False,
            nesting_depth=0,
            is_first=True,
            is_last=False,
            queue_id="test_queue",
            max_parallel=3
        )
        
        assert context.queue_id == "test_queue"
        assert context.max_parallel == 3
        assert context.current_parallel_count == 0
        assert len(context.completed_items) == 0
        assert len(context.failed_items) == 0
        assert len(context.active_items) == 0
    
    def test_item_lifecycle_tracking(self):
        """Test tracking item execution lifecycle."""
        context = ParallelQueueContext(
            item="item1",
            index=0,
            items=["item1", "item2"],
            length=2,
            loop_name="test_loop",
            loop_id="test_step",
            is_auto_generated=False,
            nesting_depth=0,
            is_first=True,
            is_last=False,
            queue_id="test_queue",
            max_parallel=2
        )
        
        # Start item
        context.mark_item_started(0)
        assert 0 in context.active_items
        assert context.current_parallel_count == 1
        assert 0 in context.item_start_times
        
        # Complete item
        context.mark_item_completed(0, {"result": "success"})
        assert 0 not in context.active_items
        assert 0 in context.completed_items
        assert context.current_parallel_count == 0
        assert context.item_results[0] == {"result": "success"}
        
        # Fail item
        context.mark_item_started(1)
        context.mark_item_failed(1, ValueError("test error"))
        assert 1 not in context.active_items
        assert 1 in context.failed_items
        assert "error" in context.item_results[1]
    
    def test_completion_tracking(self):
        """Test completion and failure rate calculation."""
        context = ParallelQueueContext(
            item="item1",
            index=0,
            items=["item1", "item2", "item3", "item4"],
            length=4,
            loop_name="test_loop",
            loop_id="test_step",
            is_auto_generated=False,
            nesting_depth=0,
            is_first=True,
            is_last=False,
            queue_id="test_queue"
        )
        
        # Process items with different outcomes
        context.mark_item_completed(0)  # Success
        context.mark_item_completed(1)  # Success  
        context.mark_item_failed(2)     # Failure
        # Item 3 still pending
        
        assert context.get_completion_rate() == 50.0  # 2/4 = 50%
        assert context.get_failure_rate() == 25.0     # 1/4 = 25%
        assert not context.is_execution_complete()    # Item 3 still pending
        
        # Complete final item
        context.mark_item_completed(3)
        assert context.is_execution_complete()
    
    def test_template_variables(self):
        """Test template variable generation."""
        context = ParallelQueueContext(
            item="current_item",
            index=1,
            items=["item0", "current_item", "item2"],
            length=3,
            loop_name="my_queue",
            loop_id="test_step",
            is_auto_generated=False,
            nesting_depth=0,
            is_first=False,
            is_last=False,
            queue_id="test_queue",
            max_parallel=2
        )
        
        # Mark some progress
        context.mark_item_completed(0)
        context.mark_item_started(1)
        
        template_vars = context.to_template_dict(item_index=1, is_current_loop=True)
        
        # Check parallel queue specific variables
        assert template_vars["$my_queue.queue_id"] == "test_queue"
        assert template_vars["$my_queue.max_parallel"] == 2
        assert template_vars["$my_queue.completed_count"] == 1
        assert template_vars["$my_queue.active_count"] == 1
        assert template_vars["$queue_index"] == 1
        assert template_vars["$queue_item"] == "current_item"
        assert template_vars["$is_parallel"] == True


class TestParallelLoopContextManager:
    """Test the parallel loop context manager."""
    
    def test_context_creation_and_management(self):
        """Test creating and managing parallel contexts."""
        manager = ParallelLoopContextManager()
        
        context = manager.create_parallel_queue_context(
            queue_id="test_queue",
            items=["a", "b", "c"],
            max_parallel=2,
            explicit_loop_name="my_parallel_loop"
        )
        
        assert context.queue_id == "test_queue"
        assert context.loop_name == "my_parallel_loop"
        assert len(context.items) == 3
        assert context.max_parallel == 2
        assert "test_queue" in manager.parallel_queues
        
        # Push to active loops
        manager.push_parallel_queue(context)
        assert "my_parallel_loop" in manager.active_loops
    
    def test_auto_generated_loop_names(self):
        """Test auto-generation of loop names."""
        manager = ParallelLoopContextManager()
        
        # Create without explicit name
        context1 = manager.create_parallel_queue_context(
            queue_id="queue1",
            items=["a", "b"],
            max_parallel=1
        )
        
        context2 = manager.create_parallel_queue_context(
            queue_id="queue2", 
            items=["c", "d"],
            max_parallel=1
        )
        
        # Should have auto-generated unique names
        assert context1.loop_name.startswith("parallel_queue_")
        assert context2.loop_name.startswith("parallel_queue_")
        assert context1.loop_name != context2.loop_name
    
    def test_template_variable_access(self):
        """Test accessing template variables for specific items."""
        manager = ParallelLoopContextManager()
        
        context = manager.create_parallel_queue_context(
            queue_id="test_queue",
            items=["item1", "item2", "item3"],
            max_parallel=2
        )
        
        # Get template variables for specific item
        template_vars = manager.get_parallel_template_variables("test_queue", 1)
        
        assert template_vars["$queue_index"] == 1
        assert template_vars["$queue_item"] == "item2"
        assert template_vars["$queue_size"] == 3
    
    def test_cleanup(self):
        """Test cleanup of parallel contexts."""
        manager = ParallelLoopContextManager()
        
        context = manager.create_parallel_queue_context(
            queue_id="test_queue",
            items=["a", "b"],
            max_parallel=1,
            explicit_loop_name="test_loop"
        )
        
        manager.push_parallel_queue(context)
        
        # Verify active
        assert "test_queue" in manager.parallel_queues
        assert "test_loop" in manager.active_loops
        
        # Cleanup
        manager.cleanup_parallel_queue("test_queue")
        
        # Verify cleaned up
        assert "test_queue" not in manager.parallel_queues
        assert "test_loop" not in manager.active_loops


class TestParallelResourceManager:
    """Test resource management for parallel execution."""
    
    @pytest.mark.asyncio
    async def test_tool_resource_pooling(self):
        """Test tool instance pooling and sharing."""
        manager = ParallelResourceManager()
        
        # Acquire tool instances
        instance1 = await manager.acquire_tool_instance("web_tool", max_instances=2)
        instance2 = await manager.acquire_tool_instance("web_tool", max_instances=2)
        
        assert instance1 is not None
        assert instance2 is not None
        assert instance1 != instance2
        
        # Check stats
        stats = manager.get_resource_stats()
        assert stats["stats"]["web_tool"]["total_acquisitions"] == 2
        assert stats["stats"]["web_tool"]["current_usage"] == 2
        assert stats["stats"]["web_tool"]["peak_usage"] == 2
        
        # Release instances
        await manager.release_tool_instance("web_tool", instance1)
        await manager.release_tool_instance("web_tool", instance2)
        
        stats = manager.get_resource_stats()
        assert stats["stats"]["web_tool"]["current_usage"] == 0
        assert stats["pools"]["web_tool"] == 2  # Back in pool (pool size)


class TestYAMLCompilerIntegration:
    """Test YAML compiler integration with parallel queues."""
    
    @pytest.mark.asyncio
    async def test_parallel_queue_detection(self):
        """Test that compiler detects create_parallel_queue syntax."""
        yaml_content = """
        name: Test Pipeline
        steps:
          - id: parallel_task
            create_parallel_queue:
              on: "['item1', 'item2']"
              max_parallel: 2
              action_loop:
                - action: process
                  parameters:
                    item: "{{ $item }}"
        """
        
        compiler = YAMLCompiler()
        pipeline = await compiler.compile(yaml_content)
        
        # Should have compiled successfully
        assert pipeline is not None
        assert len(pipeline.tasks) == 1
        
        task = pipeline.tasks["parallel_task"]
        assert isinstance(task, ParallelQueueTask)
        assert task.action == "create_parallel_queue"
        assert task.on == "['item1', 'item2']"
        assert task.max_parallel == 2
    
    @pytest.mark.asyncio
    async def test_parallel_queue_action_syntax(self):
        """Test alternate syntax with action: create_parallel_queue."""
        yaml_content = """
        name: Test Pipeline  
        steps:
          - id: parallel_task
            action: create_parallel_queue
            parameters:
              on: "<AUTO>generate items</AUTO>"
              max_parallel: 3
              action_loop:
                - action: fetch
                  parameters:
                    url: "{{ $item }}"
        """
        
        compiler = YAMLCompiler()
        pipeline = await compiler.compile(yaml_content)
        
        task = pipeline.tasks["parallel_task"]
        assert isinstance(task, ParallelQueueTask)
        assert task.action == "create_parallel_queue"
    
    @pytest.mark.asyncio
    async def test_template_field_skipping(self):
        """Test that 'on' fields are not processed at compile time."""
        yaml_content = """
        name: Test Pipeline
        steps:
          - id: parallel_task
            create_parallel_queue:
              on: "<AUTO>extract URLs from {{ previous_step.result }}</AUTO>"
              max_parallel: 2
              action_loop:
                - action: process
        """
        
        compiler = YAMLCompiler()
        
        # Should compile without trying to process the 'on' field
        pipeline = await compiler.compile(yaml_content)
        
        task = pipeline.tasks["parallel_task"]
        # The 'on' field should remain unprocessed for runtime evaluation
        assert "<AUTO>" in task.on
        assert "{{ previous_step.result }}" in task.on


# Placeholder test for Phase 2 (will be implemented next)
class TestParallelQueueHandlerBasic:
    """Basic tests for ParallelQueueHandler (Phase 1 functionality only)."""
    
    def test_handler_initialization(self):
        """Test basic handler initialization."""
        handler = ParallelQueueHandler()
        
        assert handler.auto_resolver is not None
        assert handler.loop_context_manager is not None
        assert handler.parallel_context_manager is not None
        assert handler.resource_manager is not None
        assert len(handler.active_queues) == 0
    
    def test_handler_stats(self):
        """Test handler statistics tracking."""
        handler = ParallelQueueHandler()
        
        stats = handler.get_handler_stats()
        
        assert "execution_stats" in stats
        assert "active_queues" in stats
        assert "resource_stats" in stats
        assert stats["active_queues"] == 0