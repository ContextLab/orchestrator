"""Tests for unified template resolution system."""

import pytest
from src.orchestrator.core.unified_template_resolver import (
    UnifiedTemplateResolver, 
    TemplateResolutionContext
)
from src.orchestrator.core.template_manager import TemplateManager
from src.orchestrator.core.context_manager import ContextManager
from src.orchestrator.core.loop_context import GlobalLoopContextManager, LoopContextVariables


class TestUnifiedTemplateResolver:
    """Test the unified template resolution system."""
    
    def test_basic_template_resolution(self):
        """Test basic template resolution with simple context."""
        resolver = UnifiedTemplateResolver(debug_mode=True)
        
        # Create context
        context = resolver.collect_context(
            pipeline_id="test_pipeline",
            task_id="test_task",
            pipeline_inputs={"name": "World"},
            step_results={"previous_step": "Hello"}
        )
        
        # Resolve simple template
        template = "{{ previous_step }}, {{ name }}!"
        result = resolver.resolve_templates(template, context)
        
        assert result == "Hello, World!"
    
    def test_nested_data_resolution(self):
        """Test template resolution in nested data structures."""
        resolver = UnifiedTemplateResolver(debug_mode=True)
        
        # Create context with nested data
        context = resolver.collect_context(
            pipeline_id="test_pipeline",
            step_results={
                "analyze": {
                    "result": "Analysis complete",
                    "score": 95
                }
            }
        )
        
        # Resolve templates in nested structure
        data = {
            "summary": "{{ analyze.result }}",
            "details": {
                "score": "Score: {{ analyze.score }}",
                "items": ["Item 1: {{ analyze.result }}", "Item 2: Score {{ analyze.score }}"]
            }
        }
        
        result = resolver.resolve_templates(data, context)
        
        assert result["summary"] == "Analysis complete"
        assert result["details"]["score"] == "Score: 95"
        assert result["details"]["items"][0] == "Item 1: Analysis complete"
        assert result["details"]["items"][1] == "Item 2: Score 95"
    
    def test_loop_variable_resolution(self):
        """Test template resolution with loop variables.""" 
        resolver = UnifiedTemplateResolver(debug_mode=True)
        
        # Create loop context using the manager
        loop_context = resolver.loop_context_manager.create_loop_context(
            step_id="process_items",
            item="apple", 
            index=2,
            items=["banana", "cherry", "apple"],
            explicit_loop_name="test_loop"
        )
        
        # Register loop context
        resolver.loop_context_manager.push_loop(loop_context)
        
        # The push_loop method should add the loop to the current stack too
        # Let's ensure it's properly added to the stack
        if loop_context.loop_name not in resolver.loop_context_manager.current_loop_stack:
            resolver.loop_context_manager.current_loop_stack.append(loop_context.loop_name)
        
        # Create resolution context
        context = resolver.collect_context(
            pipeline_id="test_pipeline",
            task_id="process_item",
            additional_context={"prefix": "Processing"}
        )
        
        # Resolve template with loop variables (without $ prefix for Jinja2 compatibility)
        template = "{{ prefix }}: {{ item }} at index {{ index }}"
        result = resolver.resolve_templates(template, context)
        
        assert result == "Processing: apple at index 2"
    
    def test_tool_parameter_resolution(self):
        """Test template resolution for tool parameters."""
        resolver = UnifiedTemplateResolver(debug_mode=True)
        
        # Create context
        context = resolver.collect_context(
            pipeline_id="test_pipeline",
            task_id="write_file",
            tool_name="filesystem",
            pipeline_inputs={"output_dir": "/tmp/output"},
            step_results={"generate_content": "Hello, World!"},
            tool_parameters={
                "action": "write",
                "path": "{{ output_dir }}/result.txt",
                "content": "{{ generate_content }}"
            }
        )
        
        # Use resolve_before_tool_execution method
        resolved_params = resolver.resolve_before_tool_execution(
            tool_name="filesystem",
            tool_parameters={
                "action": "write",
                "path": "{{ output_dir }}/result.txt", 
                "content": "{{ generate_content }}"
            },
            context=context
        )
        
        assert resolved_params["action"] == "write"
        assert resolved_params["path"] == "/tmp/output/result.txt"
        assert resolved_params["content"] == "Hello, World!"
    
    def test_context_hierarchy(self):
        """Test that context is properly merged from different sources."""
        resolver = UnifiedTemplateResolver(debug_mode=True)
        
        # Create comprehensive context
        context = resolver.collect_context(
            pipeline_id="test_pipeline",
            task_id="test_task",
            pipeline_inputs={"global_var": "global_value"},
            pipeline_parameters={"param_var": "param_value"},
            step_results={"step_var": "step_value"},
            additional_context={"additional_var": "additional_value"}
        )
        
        # Test that all context sources are available
        flat_context = context.to_flat_dict()
        
        assert flat_context["global_var"] == "global_value"
        assert flat_context["param_var"] == "param_value"
        assert flat_context["step_var"] == "step_value"
        assert flat_context["additional_var"] == "additional_value"
        
        # Test template resolution with all variables
        template = "{{ global_var }}-{{ param_var }}-{{ step_var }}-{{ additional_var }}"
        result = resolver.resolve_templates(template, context)
        
        assert result == "global_value-param_value-step_value-additional_value"
    
    def test_context_manager_integration(self):
        """Test integration with context manager."""
        # Use resolver with context manager
        resolver = UnifiedTemplateResolver(debug_mode=True)
        
        # Test resolution context manager
        with resolver.resolution_context(
            pipeline_id="test_pipeline",
            task_id="test_task",
            pipeline_inputs={"name": "Integration Test"},
            step_results={"previous": "Success"}
        ) as context:
            # Resolve template within context
            template = "{{ previous }}: {{ name }}"
            result = resolver.resolve_templates(template)
            
            assert result == "Success: Integration Test"
    
    def test_error_handling(self):
        """Test error handling in template resolution."""
        resolver = UnifiedTemplateResolver(debug_mode=True)
        
        # Create context with missing variables
        context = resolver.collect_context(
            pipeline_id="test_pipeline",
            step_results={"available_var": "available"}
        )
        
        # Try to resolve template with undefined variable
        template = "{{ available_var }} and {{ undefined_var }}"
        result = resolver.resolve_templates(template, context)
        
        # Should handle gracefully (depends on template manager implementation)
        assert "available" in result
    
    def test_debug_info(self):
        """Test debug information retrieval."""
        resolver = UnifiedTemplateResolver(debug_mode=True)
        
        # Create and register context
        context = resolver.collect_context(
            pipeline_id="test_pipeline",
            task_id="test_task",
            pipeline_inputs={"debug_var": "debug_value"}
        )
        resolver.register_context(context)
        
        # Get debug info
        debug_info = resolver.get_debug_info()
        
        assert debug_info["has_current_context"] is True
        assert "current_context" in debug_info
        assert debug_info["current_context"]["pipeline_id"] == "test_pipeline"
        assert debug_info["current_context"]["task_id"] == "test_task"
        assert "debug_var" in debug_info["current_context"]["context_keys"]


class TestTemplateResolutionContext:
    """Test the template resolution context class."""
    
    def test_context_initialization(self):
        """Test context initialization with defaults."""
        context = TemplateResolutionContext()
        
        # Check that empty dicts are initialized
        assert context.pipeline_inputs == {}
        assert context.step_results == {}
        assert context.loop_variables == {}
        assert context.additional_context == {}
    
    def test_context_with_data(self):
        """Test context initialization with data."""
        context = TemplateResolutionContext(
            pipeline_id="test",
            task_id="task1",
            pipeline_inputs={"input1": "value1"},
            step_results={"step1": "result1"}
        )
        
        assert context.pipeline_id == "test"
        assert context.task_id == "task1"
        assert context.pipeline_inputs["input1"] == "value1"
        assert context.step_results["step1"] == "result1"
    
    def test_to_flat_dict(self):
        """Test context flattening to dictionary."""
        context = TemplateResolutionContext(
            pipeline_id="test",
            pipeline_inputs={"input1": "value1"},
            pipeline_parameters={"param1": "pvalue1"},
            step_results={"step1": "result1"},
            additional_context={"extra": "extra_value"}
        )
        
        flat = context.to_flat_dict()
        
        # Check that all data is included
        assert flat["input1"] == "value1"
        assert flat["param1"] == "pvalue1"
        assert flat["step1"] == "result1"
        assert flat["extra"] == "extra_value"
        assert flat["pipeline_id"] == "test"
        
        # Check that parameters are also available as 'parameters'
        assert flat["parameters"]["param1"] == "pvalue1"
        
        # Check that step results are also available as 'step_results' 
        assert flat["step_results"]["step1"] == "result1"


if __name__ == "__main__":
    pytest.main([__file__])