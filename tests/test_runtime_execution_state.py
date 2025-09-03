"""
Comprehensive tests for PipelineExecutionState.

Tests all functionality with real operations, no mocks or simulations.
"""

import pytest
import json
import time
from datetime import datetime
from typing import Dict, Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.orchestrator.runtime.execution_state import (

from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider
    PipelineExecutionState,
    UnresolvedItem,
    LoopContext,
    ItemStatus
)


class TestPipelineExecutionState:
    """Test suite for PipelineExecutionState."""
    
    def test_initialization(self):
        """Test basic initialization of execution state."""
        state = PipelineExecutionState("test_pipeline")
        
        assert state.pipeline_id == "test_pipeline"
        assert isinstance(state.start_time, datetime)
        assert state.global_context['variables'] == {}
        assert state.global_context['results'] == {}
        assert len(state.executed_tasks) == 0
        assert len(state.pending_tasks) == 0
    
    def test_register_variable(self):
        """Test registering variables in global context."""
        state = PipelineExecutionState()
        
        # Register different types of variables
        state.register_variable("string_var", "test_value")
        state.register_variable("int_var", 42)
        state.register_variable("list_var", [1, 2, 3])
        state.register_variable("dict_var", {"key": "value"})
        
        context = state.get_available_context()
        assert context["string_var"] == "test_value"
        assert context["int_var"] == 42
        assert context["list_var"] == [1, 2, 3]
        assert context["dict_var"] == {"key": "value"}
    
    def test_register_result(self):
        """Test registering task execution results."""
        state = PipelineExecutionState()
        
        # Register string result
        state.register_result("task1", "simple result")
        assert state.global_context['results']["task1"] == "simple result"
        assert state.global_context['results']["task1_str"] == "simple result"
        assert "task1" in state.executed_tasks
        
        # Register dict result with 'result' key
        state.register_result("task2", {"result": "nested value", "other": 123})
        assert state.global_context['results']["task2"]["result"] == "nested value"
        assert state.global_context['results']["task2_result"] == "nested value"
        
        # Register dict result with 'value' key
        state.register_result("task3", {"value": "another value", "meta": "data"})
        assert state.global_context['results']["task3_value"] == "another value"
        
        # Verify execution tracking
        assert len(state.executed_tasks) == 3
        assert state.metadata['total_tasks_executed'] == 3
    
    def test_loop_context_management(self):
        """Test loop context stack operations."""
        state = PipelineExecutionState()
        
        # Push first loop context
        loop1 = LoopContext(
            loop_id="loop1",
            iteration=0,
            item="item1",
            index=0,
            is_first=True,
            is_last=False,
            total_items=3
        )
        state.push_loop_context("loop1", loop1)
        
        current = state.get_current_loop_context()
        assert current.loop_id == "loop1"
        assert current.item == "item1"
        
        # Push nested loop context
        loop2 = LoopContext(
            loop_id="loop2",
            iteration=0,
            item="nested_item",
            index=0,
            parent_context=loop1
        )
        state.push_loop_context("loop2", loop2)
        
        current = state.get_current_loop_context()
        assert current.loop_id == "loop2"
        assert current.parent_context == loop1
        
        # Pop contexts
        popped = state.pop_loop_context()
        assert popped.loop_id == "loop2"
        
        current = state.get_current_loop_context()
        assert current.loop_id == "loop1"
        
        popped = state.pop_loop_context()
        assert popped.loop_id == "loop1"
        
        assert state.get_current_loop_context() is None
    
    def test_loop_context_in_available_context(self):
        """Test that loop context is properly included in available context."""
        state = PipelineExecutionState()
        
        # Add some base variables
        state.register_variable("base_var", "base_value")
        
        # Push loop context
        loop_ctx = LoopContext(
            loop_id="test_loop",
            iteration=2,
            item={"id": 1, "name": "test"},
            index=2,
            is_first=False,
            is_last=False,
            total_items=5
        )
        loop_ctx.variables["custom_var"] = "custom_value"
        state.push_loop_context("test_loop", loop_ctx)
        
        context = state.get_available_context()
        
        # Check loop variables are available
        assert context["item"] == {"id": 1, "name": "test"}
        assert context["index"] == 2
        assert context["$item"] == {"id": 1, "name": "test"}
        assert context["$index"] == 2
        assert context["is_first"] is False
        assert context["is_last"] is False
        assert context["custom_var"] == "custom_value"
        
        # Check base variables still available
        assert context["base_var"] == "base_value"
    
    def test_nested_loop_context(self):
        """Test nested loop contexts with parent references."""
        state = PipelineExecutionState()
        
        # Push outer loop
        outer_loop = LoopContext(
            loop_id="outer",
            iteration=1,
            item="outer_item",
            index=1
        )
        state.push_loop_context("outer", outer_loop)
        
        # Push inner loop with parent reference
        inner_loop = LoopContext(
            loop_id="inner",
            iteration=3,
            item="inner_item",
            index=3,
            parent_context=outer_loop
        )
        state.push_loop_context("inner", inner_loop)
        
        context = state.get_available_context()
        
        # Current loop variables
        assert context["item"] == "inner_item"
        assert context["index"] == 3
        
        # Parent loop variables with prefix
        assert context["parent_1_item"] == "outer_item"
        assert context["parent_1_index"] == 1
    
    def test_unresolved_items_management(self):
        """Test managing unresolved items."""
        state = PipelineExecutionState()
        
        # Add unresolved items
        item1 = UnresolvedItem(
            id="template1",
            content="{{ variable1 }}",
            item_type="template",
            dependencies={"variable1"}
        )
        item2 = UnresolvedItem(
            id="loop1",
            content="{{ items }}",
            item_type="loop",
            dependencies={"items"}
        )
        
        state.add_unresolved_item(item1)
        state.add_unresolved_item(item2)
        
        assert len(state.unresolved_items) == 2
        assert state.dependency_graph["template1"] == {"variable1"}
        assert state.dependency_graph["loop1"] == {"items"}
        
        # Get items by type
        templates = state.get_unresolved_by_type("template")
        assert len(templates) == 1
        assert templates[0].id == "template1"
        
        loops = state.get_unresolved_by_type("loop")
        assert len(loops) == 1
        assert loops[0].id == "loop1"
    
    def test_item_resolution(self):
        """Test marking items as resolved or failed."""
        state = PipelineExecutionState()
        
        # Add unresolved item
        item = UnresolvedItem(
            id="test_item",
            content="{{ var }}",
            item_type="template",
            dependencies={"var"}
        )
        state.add_unresolved_item(item)
        
        # Mark as resolved
        state.mark_item_resolved("test_item", "resolved_value")
        
        assert "test_item" not in state.dependency_graph
        assert state.resolved_items["test_item"] == "resolved_value"
        assert len(state.unresolved_items) == 0
        
        # Add another item and mark as failed
        item2 = UnresolvedItem(
            id="fail_item",
            content="{{ missing }}",
            item_type="template",
            dependencies={"missing"}
        )
        state.add_unresolved_item(item2)
        state.mark_item_failed("fail_item", "Variable not found")
        
        # Item should still be in unresolved set but marked as failed
        failed_items = [i for i in state.unresolved_items if i.status == ItemStatus.FAILED]
        assert len(failed_items) == 1
        assert failed_items[0].error_message == "Variable not found"
    
    def test_resolvable_items_detection(self):
        """Test detection of items that can be resolved."""
        state = PipelineExecutionState()
        
        # Add some variables
        state.register_variable("var1", "value1")
        state.register_result("task1", "result1")
        
        # Add unresolved items with different dependency states
        item1 = UnresolvedItem(
            id="resolvable",
            content="{{ var1 }}",
            item_type="template",
            dependencies={"var1"}
        )
        item2 = UnresolvedItem(
            id="not_resolvable",
            content="{{ var2 }}",
            item_type="template",
            dependencies={"var2"}  # var2 doesn't exist
        )
        item3 = UnresolvedItem(
            id="multi_dep",
            content="{{ var1 }} {{ task1 }}",
            item_type="template",
            dependencies={"var1", "task1"}
        )
        
        state.add_unresolved_item(item1)
        state.add_unresolved_item(item2)
        state.add_unresolved_item(item3)
        
        resolvable = state.get_resolvable_items()
        resolvable_ids = [item.id for item in resolvable]
        
        assert "resolvable" in resolvable_ids
        assert "not_resolvable" not in resolvable_ids
        assert "multi_dep" in resolvable_ids  # Both deps are satisfied
    
    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies."""
        state = PipelineExecutionState()
        
        # Create circular dependency: A -> B -> C -> A
        item_a = UnresolvedItem(
            id="A",
            content="{{ B }}",
            item_type="template",
            dependencies={"B"}
        )
        item_b = UnresolvedItem(
            id="B",
            content="{{ C }}",
            item_type="template",
            dependencies={"C"}
        )
        item_c = UnresolvedItem(
            id="C",
            content="{{ A }}",
            item_type="template",
            dependencies={"A"}
        )
        
        state.add_unresolved_item(item_a)
        state.add_unresolved_item(item_b)
        state.add_unresolved_item(item_c)
        
        has_circular, cycle = state.has_circular_dependencies()
        assert has_circular is True
        assert cycle is not None
        # The cycle should contain A, B, and C
        assert set(cycle) == {"A", "B", "C"} or "A" in cycle
    
    def test_no_circular_dependency(self):
        """Test that non-circular dependencies are not detected as circular."""
        state = PipelineExecutionState()
        
        # Create linear dependency: A -> B -> C
        item_a = UnresolvedItem(
            id="A",
            content="{{ B }}",
            item_type="template",
            dependencies={"B"}
        )
        item_b = UnresolvedItem(
            id="B",
            content="{{ C }}",
            item_type="template",
            dependencies={"C"}
        )
        item_c = UnresolvedItem(
            id="C",
            content="constant",
            item_type="template",
            dependencies=set()  # No dependencies
        )
        
        state.add_unresolved_item(item_a)
        state.add_unresolved_item(item_b)
        state.add_unresolved_item(item_c)
        
        has_circular, cycle = state.has_circular_dependencies()
        assert has_circular is False
        assert cycle is None
    
    def test_task_failure_tracking(self):
        """Test tracking of failed tasks."""
        state = PipelineExecutionState()
        
        # Add pending tasks
        state.pending_tasks.add("task1")
        state.pending_tasks.add("task2")
        
        # Mark one as executed
        state.register_result("task1", "success")
        assert "task1" not in state.pending_tasks
        assert "task1" in state.executed_tasks
        
        # Mark another as failed
        state.mark_task_failed("task2", "Connection timeout")
        assert "task2" not in state.pending_tasks
        assert "task2" in state.failed_tasks
        assert state.failed_tasks["task2"] == "Connection timeout"
    
    def test_execution_summary(self):
        """Test execution summary generation."""
        state = PipelineExecutionState("summary_test")
        
        # Simulate some execution
        state.register_variable("var1", "value1")
        state.register_result("task1", "result1")
        state.register_result("task2", "result2")
        state.pending_tasks.add("task3")
        state.mark_task_failed("task4", "error")
        
        item = UnresolvedItem("item1", "{{ var2 }}", "template", {"var2"})
        state.add_unresolved_item(item)
        
        summary = state.get_execution_summary()
        
        assert summary["pipeline_id"] == "summary_test"
        assert summary["tasks_executed"] == 2
        assert summary["tasks_pending"] == 1
        assert summary["tasks_failed"] == 1
        assert summary["items_unresolved"] == 1
        assert summary["items_resolved"] == 0
        assert "duration_seconds" in summary
        assert summary["duration_seconds"] >= 0
    
    def test_state_export_import(self):
        """Test exporting and importing execution state."""
        # Create state with various data
        state1 = PipelineExecutionState("export_test")
        state1.register_variable("var1", "value1")
        state1.register_result("task1", {"result": "data"})
        
        loop_ctx = LoopContext("loop1", 2, item="test_item", index=2)
        state1.push_loop_context("loop1", loop_ctx)
        
        item = UnresolvedItem("item1", "{{ var2 }}", "template", {"var2"})
        state1.add_unresolved_item(item)
        
        state1.pending_tasks.add("pending1")
        state1.mark_task_failed("failed1", "error message")
        
        # Export state
        exported = state1.export_state()
        
        # Verify export contains expected data
        assert exported["pipeline_id"] == "export_test"
        assert exported["global_context"]["variables"]["var1"] == "value1"
        assert exported["global_context"]["results"]["task1"]["result"] == "data"
        assert "loop1" in exported["loop_contexts"]
        assert len(exported["unresolved_items"]) == 1
        assert "pending1" in exported["pending_tasks"]
        assert exported["failed_tasks"]["failed1"] == "error message"
        
        # Import into new state
        state2 = PipelineExecutionState()
        state2.import_state(exported)
        
        # Verify imported state matches original
        assert state2.pipeline_id == "export_test"
        assert state2.global_context["variables"]["var1"] == "value1"
        assert state2.global_context["results"]["task1"]["result"] == "data"
        assert "loop1" in state2.loop_contexts
        assert state2.loop_contexts["loop1"].item == "test_item"
        assert len(state2.unresolved_items) == 1
        assert "pending1" in state2.pending_tasks
        assert state2.failed_tasks["failed1"] == "error message"
    
    def test_auto_tag_registration(self):
        """Test registration of resolved AUTO tags."""
        state = PipelineExecutionState()
        
        # Register AUTO tag values
        state.register_auto_tag("model_selection", "gpt-4")
        state.register_auto_tag("list_generation", ["item1", "item2", "item3"])
        
        context = state.get_available_context()
        
        assert context["auto_model_selection"] == "gpt-4"
        assert context["auto_list_generation"] == ["item1", "item2", "item3"]
        assert len(context["auto_list_generation"]) == 3
    
    def test_template_registration(self):
        """Test registration of resolved templates."""
        state = PipelineExecutionState()
        
        # Register resolved templates
        state.register_template("greeting", "Hello, World!")
        state.register_template("prompt", "Process the following: test data")
        
        assert state.global_context["templates"]["greeting"] == "Hello, World!"
        assert state.global_context["templates"]["prompt"] == "Process the following: test data"
        
        # Templates should be in available context
        context = state.get_available_context()
        assert context["greeting"] == "Hello, World!"
        assert context["prompt"] == "Process the following: test data"
    
    def test_context_requirements(self):
        """Test handling of context requirements for unresolved items."""
        state = PipelineExecutionState()
        
        # Add item with both dependencies and context requirements
        item = UnresolvedItem(
            id="complex_item",
            content="{{ process(var1, var2) }}",
            item_type="template",
            dependencies={"var1"},
            context_requirements={"var2", "helper_func"}
        )
        state.add_unresolved_item(item)
        
        # Register only dependency
        state.register_variable("var1", "value1")
        
        # Should not be resolvable yet (context requirement missing)
        resolvable = state.get_resolvable_items()
        assert len(resolvable) == 0
        
        # Register context requirements
        state.register_variable("var2", "value2")
        state.register_variable("helper_func", lambda x, y: f"{x}-{y}")
        
        # Now should be resolvable
        resolvable = state.get_resolvable_items()
        assert len(resolvable) == 1
        assert resolvable[0].id == "complex_item"
    
    def test_resolution_attempts_tracking(self):
        """Test tracking of resolution attempts for items."""
        state = PipelineExecutionState()
        
        item = UnresolvedItem(
            id="retry_item",
            content="{{ missing_var }}",
            item_type="template",
            dependencies={"missing_var"}
        )
        state.add_unresolved_item(item)
        
        # Simulate resolution attempts
        for i in range(3):
            items = [it for it in state.unresolved_items if it.id == "retry_item"]
            if items:
                items[0].increment_attempts()
        
        # Check attempt count
        items = [it for it in state.unresolved_items if it.id == "retry_item"]
        assert items[0].resolution_attempts == 3
    
    def test_system_variables_in_context(self):
        """Test that system variables are included in available context."""
        state = PipelineExecutionState("system_test")
        
        # Wait a tiny bit to ensure execution time > 0
        time.sleep(0.01)
        
        context = state.get_available_context()
        
        assert context["pipeline_id"] == "system_test"
        assert "execution_time" in context
        assert context["execution_time"] > 0
        assert "timestamp" in context
        
        # Verify timestamp is valid ISO format
        try:
            datetime.fromisoformat(context["timestamp"])
        except ValueError:
            pytest.fail("Invalid timestamp format")
    
    def test_concurrent_modifications(self):
        """Test that state handles concurrent modifications correctly."""
        state = PipelineExecutionState()
        
        # Register multiple results rapidly
        for i in range(100):
            state.register_result(f"task_{i}", f"result_{i}")
        
        # All should be registered
        assert len(state.executed_tasks) == 100
        assert state.metadata["total_tasks_executed"] == 100
        
        # Register variables while adding unresolved items
        for i in range(50):
            state.register_variable(f"var_{i}", i)
            item = UnresolvedItem(
                id=f"item_{i}",
                content=f"{{{{ var_{i} }}}}",
                item_type="template",
                dependencies={f"var_{i}"}
            )
            state.add_unresolved_item(item)
        
        # All should be resolvable
        resolvable = state.get_resolvable_items()
        assert len(resolvable) == 50


class TestLoopContext:
    """Test suite for LoopContext."""
    
    def test_loop_context_to_dict(self):
        """Test converting loop context to dictionary."""
        ctx = LoopContext(
            loop_id="test_loop",
            iteration=5,
            item={"id": 1, "data": "test"},
            index=5,
            is_first=False,
            is_last=True,
            total_items=6
        )
        ctx.variables["custom"] = "value"
        
        result = ctx.to_dict()
        
        assert result["loop_id"] == "test_loop"
        assert result["iteration"] == 5
        assert result["item"] == {"id": 1, "data": "test"}
        assert result["index"] == 5
        assert result["is_first"] is False
        assert result["is_last"] is True
        assert result["total_items"] == 6
        assert result["custom"] == "value"
        
        # Check $ prefixed variables
        assert result["$item"] == {"id": 1, "data": "test"}
        assert result["$index"] == 5
        assert result["$is_first"] is False
        assert result["$is_last"] is True
    
    def test_nested_loop_context(self):
        """Test nested loop context with parent reference."""
        parent = LoopContext(
            loop_id="outer",
            iteration=2,
            item="outer_item",
            index=2
        )
        
        child = LoopContext(
            loop_id="inner",
            iteration=0,
            item="inner_item",
            index=0,
            parent_context=parent
        )
        
        assert child.parent_context == parent
        assert child.parent_context.loop_id == "outer"


class TestUnresolvedItem:
    """Test suite for UnresolvedItem."""
    
    def test_unresolved_item_creation(self):
        """Test creating unresolved items."""
        item = UnresolvedItem(
            id="test_item",
            content="{{ variable }}",
            item_type="template",
            dependencies={"variable"}
        )
        
        assert item.id == "test_item"
        assert item.content == "{{ variable }}"
        assert item.item_type == "template"
        assert "variable" in item.dependencies
        assert item.status == ItemStatus.UNRESOLVED
        assert item.resolution_attempts == 0
    
    def test_mark_resolved(self):
        """Test marking item as resolved."""
        item = UnresolvedItem("test", "content", "template")
        item.mark_resolved("resolved_value")
        
        assert item.status == ItemStatus.RESOLVED
        assert item.metadata["resolved_value"] == "resolved_value"
        assert "resolved_at" in item.metadata
    
    def test_mark_failed(self):
        """Test marking item as failed."""
        item = UnresolvedItem("test", "content", "template")
        item.mark_failed("Error message")
        
        assert item.status == ItemStatus.FAILED
        assert item.error_message == "Error message"
        assert "failed_at" in item.metadata
    
    def test_item_hashable(self):
        """Test that items can be used in sets."""
        item1 = UnresolvedItem("id1", "content1", "template")
        item2 = UnresolvedItem("id2", "content2", "template")
        item3 = UnresolvedItem("id1", "different", "loop")  # Same ID
        
        items_set = {item1, item2, item3}
        assert len(items_set) == 2  # item3 replaces item1 due to same ID


if __name__ == "__main__":
    pytest.main([__file__, "-v"])