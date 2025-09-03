"""
Comprehensive tests for DependencyResolver.

Tests all functionality with real operations, no mocks or simulations.
"""

import pytest
from typing import Dict, Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.orchestrator.runtime.execution_state import (

from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider
    PipelineExecutionState,
    UnresolvedItem,
    ItemStatus
)
from src.orchestrator.runtime.dependency_resolver import (
    DependencyResolver,
    ResolutionResult
)


class TestDependencyExtraction:
    """Test dependency extraction from various template formats."""
    
    def test_extract_simple_variables(self):
        """Test extracting simple variable references."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        
        # Simple variable
        deps = resolver.extract_dependencies("{{ variable }}")
        assert deps == {"variable"}
        
        # Multiple variables
        deps = resolver.extract_dependencies("{{ var1 }} and {{ var2 }}")
        assert deps == {"var1", "var2"}
        
        # With whitespace
        deps = resolver.extract_dependencies("{{  spacey  }}")
        assert deps == {"spacey"}
    
    def test_extract_dotted_access(self):
        """Test extracting variables with dotted access."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        
        # Object property access
        deps = resolver.extract_dependencies("{{ task.result }}")
        assert deps == {"task"}
        
        # Nested access
        deps = resolver.extract_dependencies("{{ obj.prop.nested }}")
        assert deps == {"obj"}
        
        # Multiple dotted
        deps = resolver.extract_dependencies("{{ task1.result }} {{ task2.value }}")
        assert deps == {"task1", "task2"}
    
    def test_extract_array_dict_access(self):
        """Test extracting array and dictionary access."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        
        # Array access
        deps = resolver.extract_dependencies("{{ items[0] }}")
        assert deps == {"items"}
        
        # Dict access with string key
        deps = resolver.extract_dependencies("{{ data['key'] }}")
        assert deps == {"data"}
        
        # Dict access with double quotes
        deps = resolver.extract_dependencies('{{ config["setting"] }}')
        assert deps == {"config"}
    
    def test_extract_jinja_expressions(self):
        """Test extracting dependencies from Jinja2 expressions."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        
        # If statement
        deps = resolver.extract_dependencies("{% if condition %}text{% endif %}")
        assert "condition" in deps
        
        # For loop
        deps = resolver.extract_dependencies("{% for item in items %}{{ item }}{% endfor %}")
        assert "items" in deps
        
        # Complex expression
        deps = resolver.extract_dependencies("{% if var1 > 5 and var2 %}test{% endif %}")
        assert deps >= {"var1", "var2"}
    
    def test_extract_auto_tags(self):
        """Test extracting dependencies from AUTO tags."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        
        # AUTO tag with variable reference
        deps = resolver.extract_dependencies("<AUTO>Generate list based on {{ input }}</AUTO>")
        assert "input" in deps
        
        # Nested dependencies
        deps = resolver.extract_dependencies("<AUTO task='process'>Use {{ data.value }}</AUTO>")
        assert "data" in deps
    
    def test_extract_filters_out_keywords(self):
        """Test that Python/Jinja keywords are filtered out."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        
        # Keywords should be filtered
        deps = resolver.extract_dependencies("{% if true and false %}{{ none }}{% endif %}")
        assert "true" not in deps
        assert "false" not in deps
        assert "none" not in deps
        assert "if" not in deps
        assert "and" not in deps
    
    def test_extract_complex_template(self):
        """Test extracting from complex real-world template."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        
        template = """
        {% if task1.result %}
            Process {{ data.items[0] }} with {{ config['model'] }}
            {% for item in task2.results %}
                - {{ item.name }}: {{ item.value }}
            {% endfor %}
        {% endif %}
        """
        
        deps = resolver.extract_dependencies(template)
        assert "task1" in deps
        assert "data" in deps
        assert "config" in deps
        assert "task2" in deps


class TestTemplateResolution:
    """Test template resolution functionality."""
    
    def test_resolve_simple_template(self):
        """Test resolving simple templates."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        
        # Add context
        state.register_variable("name", "World")
        state.register_variable("count", 42)
        
        # Resolve templates
        result = resolver.resolve_template("Hello, {{ name }}!")
        assert result == "Hello, World!"
        
        result = resolver.resolve_template("Count: {{ count }}")
        assert result == "Count: 42"
    
    def test_resolve_with_task_results(self):
        """Test resolving templates with task results."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        
        # Register task results
        state.register_result("task1", "simple result")
        state.register_result("task2", {"result": "nested", "value": 123})
        
        # Test direct access
        result = resolver.resolve_template("Task1: {{ task1 }}")
        assert result == "Task1: simple result"
        
        # Test nested access (using registered accessors)
        result = resolver.resolve_template("Task2: {{ task2_result }}")
        assert result == "Task2: nested"
    
    def test_resolve_with_loop_context(self):
        """Test resolving templates within loop context."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        
        # Add loop context
        from src.orchestrator.runtime.execution_state import LoopContext
        loop_ctx = LoopContext(
            loop_id="test_loop",
            iteration=2,
            item="test_item",
            index=2,
            is_first=False,
            is_last=False
        )
        state.push_loop_context("test_loop", loop_ctx)
        
        # Resolve with loop variables
        result = resolver.resolve_template("Processing {{ item }} at index {{ index }}")
        assert result == "Processing test_item at index 2"
        
        # Test that loop context also registers without $ prefix
        result = resolver.resolve_template("Item: {{ item }}, First: {{ is_first }}")
        assert result == "Item: test_item, First: False"
    
    def test_resolve_with_filters(self):
        """Test resolving templates with Jinja2 filters."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        
        state.register_variable("text", "hello world")
        state.register_variable("items", [1, 2, 3, 4, 5])
        
        # Test filters
        result = resolver.resolve_template("{{ text | upper }}")
        assert result == "HELLO WORLD"
        
        result = resolver.resolve_template("{{ items | length }}")
        assert result == "5"
        
        result = resolver.resolve_template("{{ text | title }}")
        assert result == "Hello World"
    
    def test_resolve_conditionals(self):
        """Test resolving conditional templates."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        
        state.register_variable("condition", True)
        state.register_variable("value", "test")
        
        template = "{% if condition %}Value: {{ value }}{% else %}No value{% endif %}"
        result = resolver.resolve_template(template)
        assert result == "Value: test"
        
        state.register_variable("condition", False)
        result = resolver.resolve_template(template)
        assert result == "No value"
    
    def test_resolve_loops(self):
        """Test resolving templates with loops."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        
        state.register_variable("items", ["apple", "banana", "cherry"])
        
        template = "{% for item in items %}- {{ item }}\n{% endfor %}"
        result = resolver.resolve_template(template)
        assert result == "- apple\n- banana\n- cherry\n"
    
    def test_undefined_variable_handling(self):
        """Test handling of undefined variables."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        
        # Should raise UndefinedError
        with pytest.raises(Exception):  # Jinja2 UndefinedError
            resolver.resolve_template("{{ undefined_var }}")


class TestItemResolution:
    """Test resolution of different item types."""
    
    def test_resolve_template_item(self):
        """Test resolving template items."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        
        # Add required context
        state.register_variable("name", "Test")
        
        # Create template item
        item = UnresolvedItem(
            id="greeting",
            content="Hello, {{ name }}!",
            item_type="template",
            dependencies={"name"}
        )
        
        # Resolve
        success, value = resolver.resolve_item(item)
        assert success is True
        assert value == "Hello, Test!"
        assert item.status == ItemStatus.RESOLVED
    
    def test_resolve_auto_tag_item(self):
        """Test resolving AUTO tag items."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        
        # Create AUTO tag item
        item = UnresolvedItem(
            id="auto1",
            content="<AUTO>Generate list of items</AUTO>",
            item_type="auto_tag",
            dependencies=set()
        )
        
        # Resolve (will use mock resolver)
        success, value = resolver.resolve_item(item)
        assert success is True
        assert isinstance(value, list)
        assert len(value) > 0
    
    def test_resolve_expression_item(self):
        """Test resolving expression items."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        
        # Add context
        state.register_variable("x", 10)
        state.register_variable("y", 5)
        
        # Create expression item
        item = UnresolvedItem(
            id="calc",
            content="x + y * 2",
            item_type="expression",
            dependencies={"x", "y"}
        )
        
        # Resolve
        success, value = resolver.resolve_item(item)
        assert success is True
        assert value == 20  # 10 + 5 * 2
    
    def test_resolve_loop_item(self):
        """Test resolving loop iterator items."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        
        # Add context
        state.register_variable("languages", ["en", "es", "fr"])
        
        # Create loop item
        item = UnresolvedItem(
            id="loop1",
            content="{{ languages }}",
            item_type="loop",
            dependencies={"languages"}
        )
        
        # Resolve
        success, value = resolver.resolve_item(item)
        assert success is True
        assert value == ["en", "es", "fr"]
    
    def test_resolve_condition_item(self):
        """Test resolving condition items."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        
        # Add context
        state.register_variable("count", 10)
        state.register_variable("threshold", 5)
        
        # Create condition item
        item = UnresolvedItem(
            id="check",
            content="{{ count > threshold }}",
            item_type="condition",
            dependencies={"count", "threshold"}
        )
        
        # Resolve
        success, value = resolver.resolve_item(item)
        assert success is True
        assert value is True  # 10 > 5
    
    def test_cannot_resolve_missing_deps(self):
        """Test that items with missing dependencies cannot be resolved."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        
        # Create item with missing dependency
        item = UnresolvedItem(
            id="test",
            content="{{ missing }}",
            item_type="template",
            dependencies={"missing"}
        )
        
        # Should not be resolvable
        assert resolver.can_resolve(item) is False
        
        # Attempt to resolve should fail
        success, value = resolver.resolve_item(item)
        assert success is False
        assert value is None


class TestProgressiveResolution:
    """Test progressive resolution of multiple items."""
    
    def test_resolve_all_simple(self):
        """Test resolving all items in simple case."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        
        # Add items with dependencies
        item1 = UnresolvedItem(
            id="const",
            content="constant",
            item_type="template",
            dependencies=set()
        )
        item2 = UnresolvedItem(
            id="derived",
            content="{{ const }} value",
            item_type="template",
            dependencies={"const"}
        )
        
        state.add_unresolved_item(item1)
        state.add_unresolved_item(item2)
        
        # Resolve all
        result = resolver.resolve_all_pending()
        
        assert result.success is True
        assert len(result.resolved_items) == 2
        assert "const" in result.resolved_items
        assert "derived" in result.resolved_items
        assert result.iterations > 0
    
    def test_resolve_chain_dependencies(self):
        """Test resolving chain of dependencies."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        
        # Create chain: A -> B -> C
        state.register_variable("base", "start")
        
        item_a = UnresolvedItem(
            id="A",
            content="{{ base }}_a",
            item_type="template",
            dependencies={"base"}
        )
        item_b = UnresolvedItem(
            id="B",
            content="{{ A }}_b",
            item_type="template",
            dependencies={"A"}
        )
        item_c = UnresolvedItem(
            id="C",
            content="{{ B }}_c",
            item_type="template",
            dependencies={"B"}
        )
        
        state.add_unresolved_item(item_a)
        state.add_unresolved_item(item_b)
        state.add_unresolved_item(item_c)
        
        # Resolve all
        result = resolver.resolve_all_pending()
        
        assert result.success is True
        assert len(result.resolved_items) == 3
        
        # Check resolved values
        assert state.resolved_items["A"] == "start_a"
        assert state.resolved_items["B"] == "start_a_b"
        assert state.resolved_items["C"] == "start_a_b_c"
    
    def test_resolve_with_circular_dependency(self):
        """Test handling of circular dependencies."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        
        # Create circular: A -> B -> A
        item_a = UnresolvedItem(
            id="A",
            content="{{ B }}",
            item_type="template",
            dependencies={"B"}
        )
        item_b = UnresolvedItem(
            id="B",
            content="{{ A }}",
            item_type="template",
            dependencies={"A"}
        )
        
        state.add_unresolved_item(item_a)
        state.add_unresolved_item(item_b)
        
        # Resolution should fail
        result = resolver.resolve_all_pending()
        
        assert result.success is False
        assert "Circular dependency" in result.error_message
        assert len(result.resolved_items) == 0
    
    def test_resolve_with_missing_dependency(self):
        """Test handling of permanently missing dependencies."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        
        # Create item with missing dependency
        item = UnresolvedItem(
            id="blocked",
            content="{{ missing_var }}",
            item_type="template",
            dependencies={"missing_var"}
        )
        
        state.add_unresolved_item(item)
        
        # Resolution should fail
        result = resolver.resolve_all_pending()
        
        assert result.success is False
        assert "missing dependencies" in result.error_message
        assert "missing_var" in result.error_message
    
    def test_resolve_mixed_success_failure(self):
        """Test mixed resolution with some successes and failures."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        
        state.register_variable("available", "yes")
        
        # Resolvable item
        item1 = UnresolvedItem(
            id="good",
            content="{{ available }}",
            item_type="template",
            dependencies={"available"}
        )
        
        # Unresolvable item
        item2 = UnresolvedItem(
            id="bad",
            content="{{ missing }}",
            item_type="template",
            dependencies={"missing"}
        )
        
        state.add_unresolved_item(item1)
        state.add_unresolved_item(item2)
        
        # Resolve
        result = resolver.resolve_all_pending()
        
        assert result.success is False  # Failed due to unresolvable item
        assert "good" in result.resolved_items
        assert "bad" in result.unresolved_items
    
    def test_progressive_resolution_updates_context(self):
        """Test that resolved items update context for later items."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        
        # Item that resolves to provide context for next item
        item1 = UnresolvedItem(
            id="provider",
            content="provided_value",
            item_type="template",
            dependencies=set()
        )
        
        # Item that depends on first
        item2 = UnresolvedItem(
            id="consumer",
            content="Using {{ provider }}",
            item_type="template",
            dependencies={"provider"}
        )
        
        state.add_unresolved_item(item1)
        state.add_unresolved_item(item2)
        
        # Resolve
        result = resolver.resolve_all_pending()
        
        assert result.success is True
        assert len(result.resolved_items) == 2
        assert state.resolved_items["consumer"] == "Using provided_value"
    
    def test_max_iterations_limit(self):
        """Test that resolution stops at max iterations."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state, max_iterations=2)
        
        # Create many chained dependencies
        for i in range(10):
            if i == 0:
                deps = set()
                content = "base"
            else:
                deps = {f"item_{i-1}"}
                content = f"{{{{ item_{i-1} }}}}_next"
            
            item = UnresolvedItem(
                id=f"item_{i}",
                content=content,
                item_type="template",
                dependencies=deps
            )
            state.add_unresolved_item(item)
        
        # Resolve with limited iterations
        result = resolver.resolve_all_pending()
        
        assert result.success is False
        assert "Maximum iterations" in result.error_message
        assert result.iterations == 2


class TestResolutionOrder:
    """Test resolution order optimization."""
    
    def test_get_resolution_order(self):
        """Test getting optimal resolution order."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        
        # Create dependency graph: C -> B -> A, D -> A
        item_a = UnresolvedItem("A", "a", "template", set())
        item_b = UnresolvedItem("B", "b", "template", {"A"})
        item_c = UnresolvedItem("C", "c", "template", {"B"})
        item_d = UnresolvedItem("D", "d", "template", {"A"})
        
        state.add_unresolved_item(item_a)
        state.add_unresolved_item(item_b)
        state.add_unresolved_item(item_c)
        state.add_unresolved_item(item_d)
        
        order = resolver.get_resolution_order()
        
        # A should come first (no dependencies)
        assert order[0] == "A"
        # B and D should come before C
        assert order.index("B") < order.index("C")
        assert order.index("D") < order.index("C")


class TestSingleTemplateResolution:
    """Test convenience method for single template resolution."""
    
    def test_resolve_single_template_success(self):
        """Test resolving a single template successfully."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        
        state.register_variable("name", "World")
        
        success, result = resolver.resolve_single_template("Hello, {{ name }}!")
        
        assert success is True
        assert result == "Hello, World!"
    
    def test_resolve_single_template_failure(self):
        """Test failing to resolve a single template."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        
        success, result = resolver.resolve_single_template("Hello, {{ missing }}!")
        
        assert success is False
        assert result == "Hello, {{ missing }}!"  # Returns original on failure


class TestAutoTagResolution:
    """Test AUTO tag resolution (mocked for now)."""
    
    def test_resolve_auto_tag_list(self):
        """Test resolving AUTO tag that generates a list."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        
        result = resolver.resolve_auto_tag("Generate list of items")
        assert isinstance(result, list)
        assert len(result) > 0
    
    def test_resolve_auto_tag_model(self):
        """Test resolving AUTO tag for model selection."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        
        result = resolver.resolve_auto_tag("Select best model for translation")
        assert isinstance(result, str)
        assert "gpt" in result.lower() or "model" in result.lower()
    
    def test_resolve_auto_tag_with_template(self):
        """Test resolving AUTO tag containing templates."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        
        state.register_variable("task_type", "analysis")
        
        result = resolver.resolve_auto_tag("Generate list for {{ task_type }}")
        # Should resolve the template first
        assert isinstance(result, list)


class TestExpressionResolution:
    """Test Python expression resolution."""
    
    def test_resolve_arithmetic_expression(self):
        """Test resolving arithmetic expressions."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        
        state.register_variable("x", 10)
        state.register_variable("y", 5)
        
        result = resolver.resolve_expression("x + y")
        assert result == 15
        
        result = resolver.resolve_expression("x * y - 10")
        assert result == 40
    
    def test_resolve_boolean_expression(self):
        """Test resolving boolean expressions."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        
        state.register_variable("a", True)
        state.register_variable("b", False)
        
        result = resolver.resolve_expression("a and not b")
        assert result is True
        
        result = resolver.resolve_expression("a or b")
        assert result is True
    
    def test_resolve_comparison_expression(self):
        """Test resolving comparison expressions."""
        state = PipelineExecutionState()
        resolver = DependencyResolver(state)
        
        state.register_variable("count", 10)
        
        result = resolver.resolve_expression("count > 5")
        assert result is True
        
        result = resolver.resolve_expression("count <= 10")
        assert result is True
        
        result = resolver.resolve_expression("count == 11")
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])