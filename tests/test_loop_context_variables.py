"""
Comprehensive tests for loop context variables with named loops.
Tests the complete loop variable workflow with real pipeline execution.
NO MOCKS - all tests use real loop processing and execution.
"""

import pytest
import asyncio
from typing import Dict, Any, List

from src.orchestrator.core.loop_context import (
    LoopContextVariables, 
    GlobalLoopContextManager, 
    ItemListAccessor
)
from src.orchestrator.control_flow.loops import ForLoopHandler, WhileLoopHandler
from src.orchestrator.control_flow.auto_resolver import ControlFlowAutoResolver
from src.orchestrator.core.template_manager import TemplateManager
from src.orchestrator.core.context_manager import ContextManager


class TestLoopContextVariables:
    """Test core loop context variables functionality."""
    
    def test_item_list_accessor_basic_operations(self):
        """Test ItemListAccessor with various operations."""
        items = ["apple", "banana", "cherry", "date"]
        accessor = ItemListAccessor(items, "fruit_loop")
        
        # Test basic indexing
        assert accessor[0] == "apple"
        assert accessor[1] == "banana"
        assert accessor[-1] == "date"
        assert accessor[-2] == "cherry"
        
        # Test bounds checking
        assert accessor[10] is None
        assert accessor[-10] is None
        
        # Test properties
        assert accessor.first == "apple"
        assert accessor.last == "date"
        assert accessor.second == "banana"
        assert accessor.middle == "cherry"  # 4//2 = 2, items[2] = cherry
        
        # Test methods
        assert accessor.get(1) == "banana"
        assert accessor.get(10, "fallback") == "fallback"
        assert accessor.next_item(1) == "cherry"
        assert accessor.prev_item(2) == "banana"
        assert accessor.find("cherry") == 2
        assert accessor.find("orange") == -1
        assert accessor.contains("banana") is True
        assert accessor.contains("orange") is False
        
        # Test utility methods
        assert len(accessor) == 4
        assert not accessor.is_empty()
        assert accessor.has_multiple()
        assert list(accessor) == items
    
    def test_loop_context_variables_creation(self):
        """Test creating LoopContextVariables with all properties."""
        items = ["red", "green", "blue"]
        
        loop_context = LoopContextVariables(
            item="green",
            index=1,
            items=items,
            length=3,
            loop_name="color_loop",
            loop_id="process_colors",
            is_auto_generated=False,
            nesting_depth=0,
            is_first=False,
            is_last=False
        )
        
        # Test basic properties
        assert loop_context.item == "green"
        assert loop_context.index == 1
        assert loop_context.position == 2  # index + 1
        assert loop_context.remaining == 1  # length - position
        assert loop_context.has_next is True
        assert loop_context.has_prev is True
        assert loop_context.is_first is False
        assert loop_context.is_last is False
        
        # Test template dict generation
        template_vars = loop_context.to_template_dict(is_current_loop=True)
        
        # Check named variables
        assert template_vars["$color_loop.item"] == "green"
        assert template_vars["$color_loop.index"] == 1
        assert template_vars["$color_loop.length"] == 3
        assert template_vars["$color_loop.position"] == 2
        assert template_vars["$color_loop.remaining"] == 1
        
        # Check default variables (since is_current_loop=True)
        assert template_vars["$item"] == "green"
        assert template_vars["$index"] == 1
        assert template_vars["$length"] == 3
        assert isinstance(template_vars["$items"], ItemListAccessor)
    
    def test_auto_generated_loop_names(self):
        """Test auto-generation of loop names."""
        # Test auto-generation pattern
        auto_name = LoopContextVariables.generate_loop_name("process_data", 2)
        assert auto_name == "process_data_loop_2"
        
        # Test with special characters in step_id (should be cleaned)
        auto_name2 = LoopContextVariables.generate_loop_name("step-with.dots", 0)
        assert auto_name2 == "step_with_dots_loop_0"


class TestGlobalLoopContextManager:
    """Test the global loop context management system."""
    
    def test_loop_context_creation_and_management(self):
        """Test creating and managing loop contexts."""
        manager = GlobalLoopContextManager()
        
        # Test creating explicit named loop
        loop_context1 = manager.create_loop_context(
            step_id="outer_step",
            item="item1",
            index=0,
            items=["item1", "item2"],
            explicit_loop_name="outer_loop"
        )
        
        assert loop_context1.loop_name == "outer_loop"
        assert loop_context1.is_auto_generated is False
        
        # Test creating auto-generated loop
        loop_context2 = manager.create_loop_context(
            step_id="inner_step",
            item="subitem1",
            index=0,
            items=["subitem1", "subitem2"]
        )
        
        assert loop_context2.loop_name == "inner_step_loop_0"
        assert loop_context2.is_auto_generated is True
        
        # Test pushing and retrieving contexts
        manager.push_loop(loop_context1)
        manager.push_loop(loop_context2)
        
        # Test current loop retrieval
        current = manager.get_current_loop()
        assert current.loop_name == "inner_step_loop_0"
        
        # Test accessing by name
        outer = manager.get_loop_by_name("outer_loop")
        assert outer.loop_name == "outer_loop"
        
        # Test getting all accessible variables
        all_vars = manager.get_accessible_loop_variables()
        
        # Should have both named and default variables
        assert "$outer_loop.item" in all_vars
        assert "$inner_step_loop_0.item" in all_vars
        assert "$item" in all_vars  # Default from current loop
        assert all_vars["$item"] == "subitem1"  # From current loop
        
        # Test popping contexts
        manager.pop_loop("inner_step_loop_0")
        current_after_pop = manager.get_current_loop()
        assert current_after_pop.loop_name == "outer_loop"
    
    def test_nested_loop_variable_access(self):
        """Test accessing variables from nested loops."""
        manager = GlobalLoopContextManager()
        
        # Create nested structure: categories -> items -> variants
        categories = ["electronics", "books"]
        items = ["laptop", "tablet"]
        variants = ["red", "blue"]
        
        # Level 1: categories
        cat_context = manager.create_loop_context(
            step_id="process_categories",
            item="electronics", 
            index=0,
            items=categories,
            explicit_loop_name="category_loop"
        )
        manager.push_loop(cat_context)
        
        # Level 2: items
        item_context = manager.create_loop_context(
            step_id="process_items",
            item="laptop",
            index=0, 
            items=items,
            explicit_loop_name="item_loop"
        )
        manager.push_loop(item_context)
        
        # Level 3: variants (auto-generated name)
        variant_context = manager.create_loop_context(
            step_id="process_variants",
            item="red",
            index=0,
            items=variants
        )
        manager.push_loop(variant_context)
        
        # Test accessing all loop variables
        all_vars = manager.get_accessible_loop_variables()
        
        # Should have variables from all three loops
        assert all_vars["$category_loop.item"] == "electronics"
        assert all_vars["$item_loop.item"] == "laptop"
        assert all_vars["$process_variants_loop_2.item"] == "red"
        
        # Default variables should be from current (deepest) loop
        assert all_vars["$item"] == "red"
        assert all_vars["$index"] == 0
        
        # Test list access from different loops
        category_items = all_vars["$category_loop.items"]
        assert isinstance(category_items, ItemListAccessor)
        assert category_items[1] == "books"
        
        item_items = all_vars["$item_loop.items"]
        assert item_items[1] == "tablet"


class TestForLoopHandlerWithNamedLoops:
    """Test ForLoopHandler with named loop support."""
    
    @pytest.fixture
    def loop_manager(self):
        """Create a loop context manager for testing."""
        return GlobalLoopContextManager()
    
    @pytest.fixture  
    def for_handler(self, loop_manager):
        """Create ForLoopHandler with loop context manager."""
        auto_resolver = ControlFlowAutoResolver()
        return ForLoopHandler(auto_resolver, loop_manager)
    
    @pytest.mark.asyncio
    async def test_basic_for_loop_expansion_with_named_loops(self, for_handler):
        """Test basic for loop expansion with named loop variables."""
        loop_def = {
            "id": "process_fruits",
            "loop_name": "fruit_loop",  # Explicit name
            "for_each": ["apple", "banana", "cherry"],
            "steps": [
                {
                    "id": "analyze_fruit",
                    "action": "log",
                    "parameters": {
                        "message": "Processing {{ $fruit_loop.item }} ({{ $fruit_loop.position }}/{{ $fruit_loop.length }})"
                    }
                }
            ]
        }
        
        context = {"pipeline_id": "test"}
        step_results = {}
        
        # Expand the loop
        expanded_tasks = await for_handler.expand_for_loop(loop_def, context, step_results)
        
        # Should have 3 iteration tasks + 1 completion task = 4 total
        assert len(expanded_tasks) == 4
        
        # Check first iteration task
        first_task = expanded_tasks[0]
        assert first_task.id == "process_fruits_0_analyze_fruit"
        assert "fruit_loop" in first_task.metadata["loop_name"]
        assert first_task.metadata["loop_index"] == 0
        
        # Check that parameters were processed with loop variables
        message = first_task.parameters["message"]
        assert "apple" in message
        assert "(1/3)" in message  # position/length
        
        # Check completion task
        completion_task = expanded_tasks[-1]
        assert completion_task.id == "process_fruits"
        assert completion_task.metadata["is_loop_completion"] is True
    
    @pytest.mark.asyncio
    async def test_nested_for_loops_with_cross_access(self, for_handler):
        """Test nested for loops with cross-loop variable access."""
        categories = [
            {"name": "fruits", "items": ["apple", "banana"]},
            {"name": "colors", "items": ["red", "blue"]}
        ]
        
        outer_loop_def = {
            "id": "process_categories", 
            "loop_name": "category_loop",
            "for_each": categories,
            "steps": [
                {
                    "id": "process_items",
                    "loop_name": "item_loop", 
                    "for_each": "{{ $item.items }}",  # Access outer loop item
                    "steps": [
                        {
                            "id": "log_item",
                            "action": "log",
                            "parameters": {
                                "message": "Category: {{ $category_loop.item.name }}, Item: {{ $item_loop.item }}"
                            }
                        }
                    ]
                }
            ]
        }
        
        context = {}
        step_results = {}
        
        # This should recursively expand nested loops
        expanded_tasks = await for_handler.expand_for_loop(outer_loop_def, context, step_results)
        
        # Should have tasks for all combinations
        # 2 categories * 2 items each = 4 item processing tasks + nested completion tasks
        assert len(expanded_tasks) >= 4
        
        # Find a task that should have cross-loop access
        fruit_apple_task = None
        for task in expanded_tasks:
            if "log_item" in task.id and task.parameters.get("message"):
                message = task.parameters["message"]
                if "fruits" in message and "apple" in message:
                    fruit_apple_task = task
                    break
        
        assert fruit_apple_task is not None
        assert "Category: fruits, Item: apple" in fruit_apple_task.parameters["message"]
    
    @pytest.mark.asyncio
    async def test_auto_generated_loop_names_in_expansion(self, for_handler):
        """Test that auto-generated loop names work correctly."""
        loop_def = {
            "id": "auto_loop_test",
            # No explicit loop_name - should auto-generate
            "for_each": ["x", "y", "z"],
            "steps": [
                {
                    "id": "process_item", 
                    "action": "log",
                    "parameters": {
                        "message": "Item: {{ $item }}, Index: {{ $index }}"
                    }
                }
            ]
        }
        
        context = {}
        step_results = {}
        
        expanded_tasks = await for_handler.expand_for_loop(loop_def, context, step_results)
        
        # Check that auto-generated loop name was used
        first_task = expanded_tasks[0]
        loop_name = first_task.metadata["loop_name"]
        assert loop_name.startswith("auto_loop_test_loop_")
        assert first_task.parameters["message"] == "Item: x, Index: 0"


class TestWhileLoopHandlerWithNamedLoops:
    """Test WhileLoopHandler with named loop support."""
    
    @pytest.fixture
    def loop_manager(self):
        """Create a loop context manager for testing."""
        return GlobalLoopContextManager()
    
    @pytest.fixture
    def while_handler(self, loop_manager):
        """Create WhileLoopHandler with loop context manager."""
        auto_resolver = ControlFlowAutoResolver()
        return WhileLoopHandler(auto_resolver, loop_manager)
    
    @pytest.mark.asyncio
    async def test_while_loop_iteration_tasks_with_named_loops(self, while_handler):
        """Test creating while loop iteration tasks with named loops."""
        loop_def = {
            "id": "counting_loop",
            "loop_name": "counter_loop",  # Explicit name
            "while": "{{ $index < 3 }}",
            "steps": [
                {
                    "id": "count_step",
                    "action": "log", 
                    "parameters": {
                        "message": "Iteration {{ $counter_loop.index }}, Position {{ $counter_loop.position }}"
                    }
                }
            ]
        }
        
        context = {}
        step_results = {}
        iteration = 1
        
        # Create iteration tasks
        iteration_tasks = await while_handler.create_iteration_tasks(
            loop_def, iteration, context, step_results
        )
        
        # Should have the step task + result capture task
        assert len(iteration_tasks) == 2
        
        # Check the main task
        count_task = iteration_tasks[0]
        assert count_task.id == "counting_loop_1_count_step"
        assert count_task.metadata["loop_name"] == "counter_loop"
        assert count_task.metadata["loop_iteration"] == 1
        
        # Check that loop variables were processed
        message = count_task.parameters["message"]
        assert "Iteration 1" in message
        assert "Position 2" in message  # index + 1
        
        # Check result capture task
        result_task = iteration_tasks[1]
        assert result_task.id == "counting_loop_1_result"
        assert result_task.metadata["is_loop_result"] is True
        assert result_task.metadata["loop_name"] == "counter_loop"


class TestTemplateManagerLoopIntegration:
    """Test template manager integration with loop variables."""
    
    @pytest.fixture
    def loop_manager(self):
        """Create a loop context manager."""
        return GlobalLoopContextManager()
    
    @pytest.fixture
    def template_manager(self, loop_manager):
        """Create template manager with loop support."""
        return TemplateManager(debug_mode=True, loop_context_manager=loop_manager)
    
    def test_template_rendering_with_loop_variables(self, template_manager, loop_manager):
        """Test template rendering with loop variables available."""
        # Set up loop context
        items = ["alpha", "beta", "gamma"]
        loop_context = LoopContextVariables(
            item="beta",
            index=1,
            items=items,
            length=3,
            loop_name="test_loop",
            loop_id="test_step",
            is_auto_generated=False,
            nesting_depth=0,
            is_first=False,
            is_last=False
        )
        
        # Register loop context
        template_manager.register_loop_context(loop_context)
        
        # Test named loop variable access
        template1 = "Current: {{ $test_loop.item }}, Position: {{ $test_loop.position }}"
        result1 = template_manager.render(template1)
        assert result1 == "Current: beta, Position: 2"
        
        # Test default loop variable access
        template2 = "Item: {{ $item }}, Index: {{ $index }}, Remaining: {{ $remaining }}"
        result2 = template_manager.render(template2)
        assert result2 == "Item: beta, Index: 1, Remaining: 1"
        
        # Test list access
        template3 = "First: {{ $items.first }}, Last: {{ $items.last }}, Next: {{ $items.next_item($index) }}"
        result3 = template_manager.render(template3)
        assert result3 == "First: alpha, Last: gamma, Next: gamma"
        
        # Test template functions
        template4 = "Current loop: {{ current_loop_name() }}, Active loops: {{ active_loops() | length }}"
        result4 = template_manager.render(template4)
        assert "test_loop" in result4
        assert "Active loops: 1" in result4
    
    def test_nested_loop_template_rendering(self, template_manager, loop_manager):
        """Test template rendering with nested loop contexts."""
        # Set up nested loop contexts
        outer_items = ["group1", "group2"]
        inner_items = ["item1", "item2", "item3"]
        
        # Outer loop
        outer_context = LoopContextVariables(
            item="group1",
            index=0,
            items=outer_items,
            length=2,
            loop_name="outer_loop",
            loop_id="outer_step",
            is_auto_generated=False,
            nesting_depth=0,
            is_first=True,
            is_last=False
        )
        
        # Inner loop
        inner_context = LoopContextVariables(
            item="item2",
            index=1, 
            items=inner_items,
            length=3,
            loop_name="inner_loop",
            loop_id="inner_step",
            is_auto_generated=False,
            nesting_depth=1,
            is_first=False,
            is_last=False
        )
        
        # Register both contexts
        template_manager.register_loop_context(outer_context)
        template_manager.register_loop_context(inner_context)
        
        # Test cross-loop access
        template = "Outer: {{ $outer_loop.item }}, Inner: {{ $inner_loop.item }}, Default: {{ $item }}"
        result = template_manager.render(template)
        assert result == "Outer: group1, Inner: item2, Default: item2"
        
        # Test complex list access
        template2 = "Next outer: {{ $outer_loop.items.next_item($outer_loop.index) }}, Prev inner: {{ $inner_loop.items.prev_item($inner_loop.index) }}"
        result2 = template_manager.render(template2)
        assert result2 == "Next outer: group2, Prev inner: item1"


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling for loop variables."""
    
    def test_empty_loop_handling(self):
        """Test handling of empty loops."""
        manager = GlobalLoopContextManager()
        
        # Create context with empty items
        empty_context = manager.create_loop_context(
            step_id="empty_loop",
            item=None,
            index=0,
            items=[],
            explicit_loop_name="empty_loop"
        )
        
        assert empty_context.length == 0
        assert empty_context.is_first is True  # First of zero items
        assert empty_context.is_last is True   # Also last of zero items
        assert empty_context.remaining == 0
        
        # Test ItemListAccessor with empty list
        accessor = ItemListAccessor([], "empty")
        assert accessor.first is None
        assert accessor.last is None
        assert accessor.is_empty() is True
        assert not accessor.has_multiple()
    
    def test_single_item_loop_handling(self):
        """Test handling of single-item loops."""
        manager = GlobalLoopContextManager()
        
        single_context = manager.create_loop_context(
            step_id="single_loop",
            item="only_item",
            index=0,
            items=["only_item"],
            explicit_loop_name="single_loop"
        )
        
        assert single_context.length == 1
        assert single_context.is_first is True
        assert single_context.is_last is True
        assert single_context.position == 1
        assert single_context.remaining == 0
        assert not single_context.has_next
        assert not single_context.has_prev
    
    def test_loop_name_conflicts(self):
        """Test handling of loop name conflicts."""
        manager = GlobalLoopContextManager()
        
        # Create first loop with name
        context1 = manager.create_loop_context(
            step_id="step1",
            item="item1", 
            index=0,
            items=["item1"],
            explicit_loop_name="shared_name"
        )
        manager.push_loop(context1)
        
        # Create second loop with same name (should log warning but work)
        context2 = manager.create_loop_context(
            step_id="step2",
            item="item2",
            index=0, 
            items=["item2"],
            explicit_loop_name="shared_name"
        )
        manager.push_loop(context2)
        
        # Should still work - latest context takes precedence
        current = manager.get_current_loop()
        assert current.item == "item2"
    
    def test_bounds_checking_in_item_accessor(self):
        """Test bounds checking in ItemListAccessor."""
        items = ["a", "b", "c"]
        accessor = ItemListAccessor(items, "test")
        
        # Test various out-of-bounds scenarios
        assert accessor[100] is None
        assert accessor[-100] is None
        assert accessor.get(100) is None
        assert accessor.get(100, "default") == "default"
        assert accessor.next_item(10) is None
        assert accessor.prev_item(-5) is None
        
        # Test slice operations
        assert accessor.slice(1, 2) == ["b"]
        assert accessor.slice(10, 20) == []  # Out of bounds slice
        assert accessor.slice(1) == ["b", "c"]  # Open-ended slice


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])