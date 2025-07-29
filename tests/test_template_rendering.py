"""Test template rendering in pipelines."""

import pytest
from datetime import datetime

from orchestrator.compiler.template_renderer import TemplateRenderer
from orchestrator.control_systems.hybrid_control_system import HybridControlSystem
from orchestrator.models import get_model_registry


class TestTemplateRendering:
    """Test template rendering functionality."""

    def test_simple_variable_replacement(self):
        """Test simple {{ variable }} replacement."""
        template = "Hello {{ name }}, welcome to {{ place }}!"
        context = {"name": "Alice", "place": "Wonderland"}
        
        result = TemplateRenderer.render(template, context)
        assert result == "Hello Alice, welcome to Wonderland!"

    def test_nested_variable_access(self):
        """Test nested object access with dot notation."""
        template = "User: {{ user.name }}, Age: {{ user.age }}"
        context = {
            "user": {
                "name": "Bob",
                "age": 30
            }
        }
        
        result = TemplateRenderer.render(template, context)
        assert result == "User: Bob, Age: 30"

    def test_loop_variables(self):
        """Test Jinja2 for loops with loop variables."""
        template = """Items:
{% for item in items %}
{{ loop.index }}. {{ item.name }} - ${{ item.price }}
{% endfor %}"""
        context = {
            "items": [
                {"name": "Apple", "price": 1.50},
                {"name": "Banana", "price": 0.75}
            ]
        }
        
        result = TemplateRenderer.render(template, context)
        assert "1. Apple - $1.5" in result
        assert "2. Banana - $0.75" in result

    def test_conditional_rendering(self):
        """Test Jinja2 if conditions."""
        template = """Status: {% if status == "active" %}✓ Active{% else %}✗ Inactive{% endif %}"""
        
        # Test active case
        result = TemplateRenderer.render(template, {"status": "active"})
        assert result == "Status: ✓ Active"
        
        # Test inactive case
        result = TemplateRenderer.render(template, {"status": "inactive"})
        assert result == "Status: ✗ Inactive"

    def test_escaped_curly_braces(self):
        """Test escaped curly braces should not be processed."""
        template = "Use \\{\\{ variable \\}\\} for templates"
        context = {"variable": "value"}
        
        result = TemplateRenderer.render(template, context)
        # Should preserve escaped braces
        assert "\\{\\{ variable \\}\\}" in result or "{{ variable }}" in result

    def test_double_escaped_braces(self):
        """Test double-escaped braces."""
        template = "Show literal: \\{\\{\\{ text \\}\\}\\}"
        context = {"text": "hello"}
        
        result = TemplateRenderer.render(template, context)
        # Should show literal braces
        assert "{{{" in result or "\\{\\{\\{" in result

    def test_filters(self):
        """Test Jinja2-style filters."""
        template = "Name: {{ name | upper }}, Slug: {{ title | slugify }}"
        context = {"name": "john doe", "title": "Hello World!"}
        
        result = TemplateRenderer.render(template, context)
        assert "JOHN DOE" in result
        assert "hello-world" in result

    def test_execution_timestamp(self):
        """Test special execution.timestamp variable."""
        template = "Generated at: {{ execution.timestamp }}"
        context = {}
        
        result = TemplateRenderer.render(template, context)
        # Should contain a timestamp
        assert "Generated at: 20" in result  # Starts with year

    def test_missing_variable(self):
        """Test behavior with missing variables."""
        template = "Hello {{ name }}, from {{ missing_var }}"
        context = {"name": "Alice"}
        
        result = TemplateRenderer.render(template, context)
        assert "Hello Alice" in result
        # Missing variable should remain as-is
        assert "{{ missing_var }}" in result

    def test_multiline_templates(self):
        """Test multiline template processing."""
        template = """# Report for {{ topic }}

## Summary
{{ summary }}

## Items
{% for item in items %}
- {{ item }}
{% endfor %}

Generated: {{ execution.timestamp }}"""
        
        context = {
            "topic": "Test Report",
            "summary": "This is a test",
            "items": ["Item 1", "Item 2", "Item 3"]
        }
        
        result = TemplateRenderer.render(template, context)
        assert "# Report for Test Report" in result
        assert "This is a test" in result
        assert "- Item 1" in result
        assert "- Item 2" in result
        assert "- Item 3" in result
        assert "Generated: 20" in result

    def test_special_loop_variables(self):
        """Test special loop variables like $item, $index."""
        # This would typically be handled by the loop processor
        template = "Processing {{ $item }} at index {{ $index }}"
        context = {"$item": "file.txt", "$index": 0}
        
        result = TemplateRenderer.render(template, context)
        assert "Processing file.txt at index 0" in result

    def test_json_filter(self):
        """Test JSON serialization filter."""
        template = "Data: {{ data | json }}"
        context = {"data": {"key": "value", "number": 42}}
        
        result = TemplateRenderer.render(template, context)
        assert '{"key": "value", "number": 42}' in result or \
               '{\\"key\\": \\"value\\", \\"number\\": 42}' in result

    def test_default_filter(self):
        """Test default value filter."""
        template = "Value: {{ missing | default('N/A') }}"
        context = {}
        
        result = TemplateRenderer.render(template, context)
        assert "Value: N/A" in result

    def test_complex_nested_template(self):
        """Test complex nested template with multiple features."""
        template = """# {{ title }}

{% for section in sections %}
## {{ section.name }}

{% if section.items %}
Items ({{ section.items | length }}):
{% for item in section.items %}
{{ loop.index }}. {{ item.name }}: {{ item.value | default('N/A') }}
{% endfor %}
{% else %}
No items in this section.
{% endif %}

{% endfor %}

Summary: {{ stats.total }} items processed at {{ execution.timestamp }}"""
        
        context = {
            "title": "Complex Report",
            "sections": [
                {
                    "name": "Section A",
                    "items": [
                        {"name": "Item 1", "value": 100},
                        {"name": "Item 2"}
                    ]
                },
                {
                    "name": "Section B",
                    "items": []
                }
            ],
            "stats": {"total": 2}
        }
        
        result = TemplateRenderer.render(template, context)
        assert "# Complex Report" in result
        assert "## Section A" in result
        assert "Items (2):" in result
        assert "1. Item 1: 100" in result
        assert "2. Item 2: N/A" in result
        assert "## Section B" in result
        assert "No items in this section." in result
        assert "Summary: 2 items processed at" in result


class TestHybridControlSystemTemplating:
    """Test template handling in HybridControlSystem."""

    def test_build_template_context_basic(self):
        """Test basic context building."""
        control_system = HybridControlSystem(get_model_registry())
        
        context = {
            "task_id": "test_task",
            "some_param": "value"
        }
        
        template_context = control_system._build_template_context(context)
        
        # Should have execution metadata
        assert "execution" in template_context
        assert "timestamp" in template_context["execution"]
        
        # Should preserve original values
        assert template_context["task_id"] == "test_task"
        assert template_context["some_param"] == "value"

    def test_build_template_context_with_previous_results(self):
        """Test context building with previous results."""
        control_system = HybridControlSystem(get_model_registry())
        
        context = {
            "previous_results": {
                "step1": {"result": "data1", "status": "success"},
                "step2": "simple_result"
            }
        }
        
        template_context = control_system._build_template_context(context)
        
        # Should flatten previous results
        assert "step1" in template_context
        assert template_context["step1"]["result"] == "data1"
        assert template_context["step1"]["status"] == "success"
        
        assert "step2" in template_context
        assert template_context["step2"]["result"] == "simple_result"

    def test_build_template_context_with_pipeline_params(self):
        """Test context building with pipeline parameters."""
        control_system = HybridControlSystem(get_model_registry())
        
        context = {
            "pipeline_metadata": {
                "parameters": {
                    "output_dir": "/tmp/output",
                    "max_items": 10
                }
            }
        }
        
        template_context = control_system._build_template_context(context)
        
        # Should include pipeline parameters
        assert template_context["output_dir"] == "/tmp/output"
        assert template_context["max_items"] == 10

    def test_resolve_templates_integration(self):
        """Test full template resolution through control system."""
        control_system = HybridControlSystem(get_model_registry())
        
        context = {
            "pipeline_metadata": {
                "parameters": {
                    "project_name": "TestProject"
                }
            },
            "previous_results": {
                "analyze": {"summary": "Analysis complete", "score": 95}
            }
        }
        
        template = """Project: {{ project_name }}
Summary: {{ analyze.summary }}
Score: {{ analyze.score }}
Generated: {{ execution.timestamp }}"""
        
        template_context = control_system._build_template_context(context)
        result = control_system._resolve_templates(template, template_context)
        
        assert "Project: TestProject" in result
        assert "Summary: Analysis complete" in result
        assert "Score: 95" in result
        assert "Generated: 20" in result  # Has timestamp


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_template(self):
        """Test empty template."""
        result = TemplateRenderer.render("", {"key": "value"})
        assert result == ""

    def test_empty_context(self):
        """Test template with empty context."""
        template = "Hello {{ name }}"
        result = TemplateRenderer.render(template, {})
        assert result == "Hello {{ name }}"

    def test_none_values(self):
        """Test None values in context."""
        template = "Value: {{ value }}"
        result = TemplateRenderer.render(template, {"value": None})
        assert "Value: None" in result

    def test_numeric_values(self):
        """Test numeric values in templates."""
        template = "Integer: {{ int_val }}, Float: {{ float_val }}"
        context = {"int_val": 42, "float_val": 3.14}
        
        result = TemplateRenderer.render(template, context)
        assert "Integer: 42" in result
        assert "Float: 3.14" in result

    def test_boolean_values(self):
        """Test boolean values in templates."""
        template = "Active: {{ is_active }}, Done: {{ is_done }}"
        context = {"is_active": True, "is_done": False}
        
        result = TemplateRenderer.render(template, context)
        assert "Active: True" in result
        assert "Done: False" in result

    def test_malformed_templates(self):
        """Test malformed template syntax."""
        # Unclosed braces
        template = "Hello {{ name"
        result = TemplateRenderer.render(template, {"name": "Alice"})
        assert "Hello {{ name" in result
        
        # Mismatched braces
        template = "Hello {{ name }}"
        result = TemplateRenderer.render(template, {"name": "Alice"})
        assert "Hello Alice}" in result

    def test_recursive_templates(self):
        """Test templates that reference themselves."""
        template = "Value: {{ value }}"
        context = {"value": "{{ value }}"}  # Recursive reference
        
        result = TemplateRenderer.render(template, context)
        assert "Value: {{ value }}" in result  # Should not infinite loop

    def test_special_characters_in_values(self):
        """Test special characters in template values."""
        template = "Message: {{ message }}"
        context = {"message": "Hello {{ world }} & <tags>"}
        
        result = TemplateRenderer.render(template, context)
        assert "Hello {{ world }} & <tags>" in result