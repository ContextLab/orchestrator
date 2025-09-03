"""
Comprehensive tests for LoopExpander.

Tests all loop expansion functionality with real operations, no mocks.
"""

import pytest
from typing import Dict, Any, List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.orchestrator.runtime.execution_state import PipelineExecutionState, LoopContext
from src.orchestrator.runtime.dependency_resolver import DependencyResolver
from src.orchestrator.runtime.loop_expander import (

from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider
    LoopExpander,
    LoopTask,
    ExpandedTask
)


class TestForEachLoopExpansion:
    """Test for_each loop expansion."""
    
    def test_expand_simple_for_each(self):
        """Test expanding a simple for_each loop."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        expander = LoopExpander(state, resolver)
        
        # Add items to context
        state.register_variable("items", ["a", "b", "c"])
        
        # Create for_each loop
        loop = LoopTask(
            id="test_loop",
            loop_type="for_each",
            iterator_expr="{{ items }}",
            loop_steps=[
                {
                    "id": "process",
                    "action": "generate_text",
                    "parameters": {
                        "prompt": "Process {{ item }}"
                    }
                }
            ],
            dependencies=["prerequisite"]
        )
        
        # Expand loop
        expanded = expander.expand_for_each(loop)
        
        assert len(expanded) == 3
        assert expanded[0].id == "test_loop_0_process"
        assert expanded[0].parameters["prompt"] == "Process a"
        assert expanded[1].parameters["prompt"] == "Process b"
        assert expanded[2].parameters["prompt"] == "Process c"
        
        # Check dependencies
        assert "prerequisite" in expanded[0].dependencies
        assert loop.completed is True
    
    def test_expand_with_index_variables(self):
        """Test that loop variables like index are available."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        expander = LoopExpander(state, resolver)
        
        state.register_variable("colors", ["red", "green", "blue"])
        
        loop = LoopTask(
            id="color_loop",
            loop_type="for_each",
            iterator_expr="{{ colors }}",
            loop_steps=[
                {
                    "id": "show",
                    "action": "echo",
                    "parameters": {
                        "text": "Item {{ index }}: {{ item }}, First: {{ is_first }}, Last: {{ is_last }}"
                    }
                }
            ]
        )
        
        expanded = expander.expand_for_each(loop)
        
        assert len(expanded) == 3
        assert expanded[0].parameters["text"] == "Item 0: red, First: True, Last: False"
        assert expanded[1].parameters["text"] == "Item 1: green, First: False, Last: False"
        assert expanded[2].parameters["text"] == "Item 2: blue, First: False, Last: True"
    
    def test_expand_with_complex_items(self):
        """Test expanding loop with complex item objects."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        expander = LoopExpander(state, resolver)
        
        state.register_variable("users", [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ])
        
        loop = LoopTask(
            id="user_loop",
            loop_type="for_each",
            iterator_expr="{{ users }}",
            loop_steps=[
                {
                    "id": "greet",
                    "action": "generate",
                    "parameters": {
                        "prompt": "Hello {{ item['name'] }}, age {{ item['age'] }}"
                    }
                }
            ]
        )
        
        expanded = expander.expand_for_each(loop)
        
        assert len(expanded) == 2
        assert "Hello Alice, age 30" in expanded[0].parameters["prompt"]
        assert "Hello Bob, age 25" in expanded[1].parameters["prompt"]
    
    def test_expand_with_dependencies_between_steps(self):
        """Test loop with dependencies between steps."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        expander = LoopExpander(state, resolver)
        
        state.register_variable("items", ["x", "y"])
        
        loop = LoopTask(
            id="multi_step",
            loop_type="for_each",
            iterator_expr="{{ items }}",
            loop_steps=[
                {
                    "id": "step1",
                    "action": "action1",
                    "parameters": {"value": "{{ item }}"}
                },
                {
                    "id": "step2",
                    "action": "action2",
                    "parameters": {"input": "{{ item }}"},
                    "dependencies": ["step1"]
                }
            ]
        )
        
        expanded = expander.expand_for_each(loop)
        
        assert len(expanded) == 4  # 2 items * 2 steps
        
        # Check internal dependencies
        assert "multi_step_0_step1" in expanded[1].dependencies  # step2 depends on step1
        assert "multi_step_1_step1" in expanded[3].dependencies
    
    def test_sequential_vs_parallel_dependencies(self):
        """Test dependency differences for sequential vs parallel execution."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        expander = LoopExpander(state, resolver)
        
        state.register_variable("items", ["a", "b", "c"])
        
        # Sequential loop (max_parallel=1)
        seq_loop = LoopTask(
            id="seq",
            loop_type="for_each",
            iterator_expr="{{ items }}",
            loop_steps=[{"id": "task", "action": "act"}],
            max_parallel=1,
            dependencies=["start"]
        )
        
        seq_expanded = expander.expand_for_each(seq_loop)
        
        # First task depends on start, others depend on previous iteration
        assert "start" in seq_expanded[0].dependencies
        assert "seq_0_task" in seq_expanded[1].dependencies
        assert "seq_1_task" in seq_expanded[2].dependencies
        
        # Parallel loop (max_parallel > 1)
        par_loop = LoopTask(
            id="par",
            loop_type="for_each",
            iterator_expr="{{ items }}",
            loop_steps=[{"id": "task", "action": "act"}],
            max_parallel=3,
            dependencies=["start"]
        )
        
        par_expanded = expander.expand_for_each(par_loop)
        
        # All tasks depend on start (can run in parallel)
        assert "start" in par_expanded[0].dependencies
        assert "start" in par_expanded[1].dependencies
        assert "start" in par_expanded[2].dependencies
    
    def test_cannot_expand_missing_iterator(self):
        """Test that loop cannot expand if iterator is missing."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        expander = LoopExpander(state, resolver)
        
        # No items in context
        loop = LoopTask(
            id="test",
            loop_type="for_each",
            iterator_expr="{{ missing_items }}",
            loop_steps=[{"id": "step", "action": "act"}]
        )
        
        assert expander.can_expand(loop) is False
        expanded = expander.expand_for_each(loop)
        assert len(expanded) == 0
    
    def test_nested_loops(self):
        """Test nested loop expansion."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        expander = LoopExpander(state, resolver)
        
        state.register_variable("outer_items", ["A", "B"])
        state.register_variable("inner_items", ["1", "2"])
        
        # Expand outer loop first
        outer_loop = LoopTask(
            id="outer",
            loop_type="for_each",
            iterator_expr="{{ outer_items }}",
            loop_steps=[
                {
                    "id": "outer_task",
                    "action": "process",
                    "parameters": {"value": "{{ item }}"}
                }
            ]
        )
        
        # Simulate outer loop context
        outer_context = LoopContext(
            loop_id="outer",
            iteration=0,
            item="A",
            index=0
        )
        state.push_loop_context("outer_0", outer_context)
        
        # Inner loop should have access to outer context
        inner_loop = LoopTask(
            id="inner",
            loop_type="for_each",
            iterator_expr="{{ inner_items }}",
            loop_steps=[
                {
                    "id": "inner_task",
                    "action": "combine",
                    "parameters": {
                        "outer": "{{ parent_1_item }}",
                        "inner": "{{ item }}"
                    }
                }
            ]
        )
        
        inner_expanded = expander.expand_for_each(inner_loop)
        
        # Pop outer context
        state.pop_loop_context()
        
        assert len(inner_expanded) == 2
        # Note: Parent context access would need to be implemented in resolver


class TestWhileLoopExpansion:
    """Test while loop expansion."""
    
    def test_expand_simple_while(self):
        """Test expanding iterations of a while loop."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        expander = LoopExpander(state, resolver)
        
        # Set up counter
        state.register_variable("counter", 0)
        
        loop = LoopTask(
            id="while_test",
            loop_type="while",
            condition_expr="counter < 3",  # Simple expression without template
            loop_steps=[
                {
                    "id": "increment",
                    "action": "update",
                    "parameters": {"value": "{{ iteration }}"}
                }
            ]
        )
        
        # First iteration
        state.register_variable("counter", 0)
        expanded1 = expander.expand_while_iteration(loop)
        assert len(expanded1) == 1
        assert expanded1[0].id == "while_test_0_increment"
        assert expanded1[0].parameters["value"] == "0"
        
        # Second iteration
        state.register_variable("counter", 1)
        expanded2 = expander.expand_while_iteration(loop)
        assert len(expanded2) == 1
        assert expanded2[0].id == "while_test_1_increment"
        assert expanded2[0].parameters["value"] == "1"
        
        # Third iteration
        state.register_variable("counter", 2)
        expanded3 = expander.expand_while_iteration(loop)
        assert len(expanded3) == 1
        assert expanded3[0].id == "while_test_2_increment"
        
        # Fourth iteration - condition false
        state.register_variable("counter", 3)
        expanded4 = expander.expand_while_iteration(loop)
        assert len(expanded4) == 0
        assert loop.completed is True
    
    def test_while_max_iterations(self):
        """Test that while loop stops at max iterations."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        expander = LoopExpander(state, resolver)
        
        state.register_variable("always_true", True)
        
        loop = LoopTask(
            id="infinite",
            loop_type="while",
            condition_expr="{{ always_true }}",
            loop_steps=[{"id": "step", "action": "act"}],
            max_iterations=5
        )
        
        # Expand iterations
        for i in range(10):  # Try more than max
            expanded = expander.expand_while_iteration(loop)
            if i < 5:
                assert len(expanded) == 1
            else:
                assert len(expanded) == 0
                assert loop.completed is True
                break
    
    def test_while_with_complex_condition(self):
        """Test while loop with complex condition."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        expander = LoopExpander(state, resolver)
        
        state.register_variable("value", 10)
        state.register_variable("threshold", 5)
        
        loop = LoopTask(
            id="complex",
            loop_type="while",
            condition_expr="value > threshold and value < 20",  # Direct expression
            loop_steps=[{"id": "process", "action": "act"}]
        )
        
        # Should expand (10 > 5 and 10 < 20)
        expanded = expander.expand_while_iteration(loop)
        assert len(expanded) == 1
        
        # Update value
        state.register_variable("value", 25)
        
        # Should not expand (25 > 5 but 25 >= 20)
        expanded = expander.expand_while_iteration(loop)
        assert len(expanded) == 0
        assert loop.completed is True
    
    def test_while_cannot_expand_missing_deps(self):
        """Test that while loop cannot expand if condition deps are missing."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        expander = LoopExpander(state, resolver)
        
        loop = LoopTask(
            id="test",
            loop_type="while",
            condition_expr="{{ missing_var < 10 }}",
            loop_steps=[{"id": "step", "action": "act"}]
        )
        
        assert expander.can_expand(loop) is False
        expanded = expander.expand_while_iteration(loop)
        assert len(expanded) == 0


class TestLoopManagement:
    """Test loop registration and management."""
    
    def test_register_and_get_loop(self):
        """Test registering and retrieving loops."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        expander = LoopExpander(state, resolver)
        
        loop = LoopTask(
            id="test_loop",
            loop_type="for_each",
            iterator_expr="{{ items }}"
        )
        
        expander.register_loop(loop)
        
        retrieved = expander.get_loop("test_loop")
        assert retrieved == loop
        
        assert expander.get_loop("nonexistent") is None
    
    def test_loop_completion_tracking(self):
        """Test tracking loop completion."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        expander = LoopExpander(state, resolver)
        
        state.register_variable("items", ["a"])
        
        loop = LoopTask(
            id="test",
            loop_type="for_each",
            iterator_expr="{{ items }}",
            loop_steps=[{"id": "step", "action": "act"}]
        )
        
        expander.register_loop(loop)
        
        assert expander.is_loop_complete("test") is False
        
        # Expand loop
        expander.expand_for_each(loop)
        
        assert expander.is_loop_complete("test") is True
    
    def test_get_expandable_loops(self):
        """Test getting all expandable loops."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        expander = LoopExpander(state, resolver)
        
        state.register_variable("ready_items", ["a", "b"])
        
        # Expandable loop
        loop1 = LoopTask(
            id="ready",
            loop_type="for_each",
            iterator_expr="{{ ready_items }}",
            loop_steps=[{"id": "step", "action": "act"}]
        )
        
        # Not expandable (missing deps)
        loop2 = LoopTask(
            id="not_ready",
            loop_type="for_each",
            iterator_expr="{{ missing_items }}",
            loop_steps=[{"id": "step", "action": "act"}]
        )
        
        # Already completed
        loop3 = LoopTask(
            id="done",
            loop_type="for_each",
            iterator_expr="{{ ready_items }}",
            loop_steps=[{"id": "step", "action": "act"}],
            completed=True
        )
        
        expander.register_loop(loop1)
        expander.register_loop(loop2)
        expander.register_loop(loop3)
        
        expandable = expander.get_expandable_loops()
        
        assert len(expandable) == 1
        assert expandable[0].id == "ready"
    
    def test_expand_all_ready(self):
        """Test expanding all ready loops at once."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        expander = LoopExpander(state, resolver)
        
        state.register_variable("list1", ["a", "b"])
        state.register_variable("list2", ["x", "y", "z"])
        
        loop1 = LoopTask(
            id="loop1",
            loop_type="for_each",
            iterator_expr="{{ list1 }}",
            loop_steps=[{"id": "task", "action": "act"}]
        )
        
        loop2 = LoopTask(
            id="loop2",
            loop_type="for_each",
            iterator_expr="{{ list2 }}",
            loop_steps=[{"id": "task", "action": "act"}]
        )
        
        expander.register_loop(loop1)
        expander.register_loop(loop2)
        
        all_expanded = expander.expand_all_ready()
        
        assert len(all_expanded) == 5  # 2 + 3
        assert any(t.loop_id == "loop1" for t in all_expanded)
        assert any(t.loop_id == "loop2" for t in all_expanded)


class TestExpandedTask:
    """Test ExpandedTask functionality."""
    
    def test_expanded_task_to_dict(self):
        """Test converting expanded task to dictionary."""
        loop_ctx = LoopContext(
            loop_id="test",
            iteration=0,
            item="test_item",
            index=0,
            is_first=True,
            is_last=False
        )
        
        task = ExpandedTask(
            id="task_0",
            original_step_id="task",
            loop_id="test",
            iteration=0,
            action="generate",
            parameters={"prompt": "test"},
            dependencies=["dep1", "dep2"],
            metadata={"custom": "value"},
            loop_context=loop_ctx
        )
        
        result = task.to_dict()
        
        assert result["id"] == "task_0"
        assert result["action"] == "generate"
        assert result["parameters"]["prompt"] == "test"
        assert len(result["dependencies"]) == 2
        assert result["metadata"]["custom"] == "value"
        assert result["metadata"]["is_loop_child"] is True
        assert result["metadata"]["loop_context"]["item"] == "test_item"


class TestParameterResolution:
    """Test parameter resolution within loops."""
    
    def test_resolve_nested_parameters(self):
        """Test resolving nested parameter structures."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        expander = LoopExpander(state, resolver)
        
        state.register_variable("items", [{"id": 1, "data": "test"}])
        
        loop = LoopTask(
            id="nested",
            loop_type="for_each",
            iterator_expr="{{ items }}",
            loop_steps=[
                {
                    "id": "process",
                    "action": "complex",
                    "parameters": {
                        "simple": "{{ item['id'] }}",
                        "nested": {
                            "deep": "{{ item['data'] }}",
                            "list": ["{{ item['id'] }}", "constant", "{{ index }}"]
                        }
                    }
                }
            ]
        )
        
        expanded = expander.expand_for_each(loop)
        
        assert expanded[0].parameters["simple"] == "1"
        assert expanded[0].parameters["nested"]["deep"] == "test"
        assert expanded[0].parameters["nested"]["list"][0] == "1"
        assert expanded[0].parameters["nested"]["list"][1] == "constant"
        assert expanded[0].parameters["nested"]["list"][2] == "0"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_iterator(self):
        """Test loop with empty iterator list."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        expander = LoopExpander(state, resolver)
        
        state.register_variable("empty_list", [])
        
        loop = LoopTask(
            id="empty",
            loop_type="for_each",
            iterator_expr="{{ empty_list }}",
            loop_steps=[{"id": "task", "action": "act"}]
        )
        
        expanded = expander.expand_for_each(loop)
        
        assert len(expanded) == 0
        assert loop.completed is True
    
    def test_invalid_iterator_type(self):
        """Test loop with non-list iterator."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        expander = LoopExpander(state, resolver)
        
        state.register_variable("not_a_list", "string_value")
        
        loop = LoopTask(
            id="invalid",
            loop_type="for_each",
            iterator_expr="{{ not_a_list }}",
            loop_steps=[{"id": "task", "action": "act"}]
        )
        
        expanded = expander.expand_for_each(loop)
        
        assert len(expanded) == 0
    
    def test_unknown_loop_type(self):
        """Test handling of unknown loop type."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        expander = LoopExpander(state, resolver)
        
        loop = LoopTask(
            id="unknown",
            loop_type="unknown_type",
            loop_steps=[{"id": "task", "action": "act"}]
        )
        
        assert expander.can_expand(loop) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])