"""
Comprehensive integration tests for template resolution system.
Tests all aspects of template resolution across the orchestrator system.
"""

import pytest
import os
import sys
import tempfile
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List

# Add orchestrator to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.orchestrator.core.unified_template_resolver import (
    UnifiedTemplateResolver, 
    TemplateResolutionContext
)
from src.orchestrator.core.template_manager import TemplateManager
from src.orchestrator.core.context_manager import ContextManager
from src.orchestrator.core.loop_context import GlobalLoopContextManager, LoopContextVariables
from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.compiler.yaml_compiler import YAMLCompiler
from orchestrator import init_models
from orchestrator.control_systems.hybrid_control_system import HybridControlSystem


class TestTemplateResolutionIntegration:
    """Comprehensive integration tests for template resolution."""
    
    @pytest.fixture
    def resolver(self):
        """Create a template resolver for testing."""
        return UnifiedTemplateResolver(debug_mode=True)
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_stream_a_core_resolution_fixes(self, resolver):
        """Test all core template resolution fixes from Stream A."""
        
        # Test 1: Jinja2 syntax compatibility fix ($variable syntax)
        context = resolver.collect_context(
            pipeline_id="test_pipeline",
            additional_context={
                "item": "test_value",
                "index": 5,
                "is_first": True
            }
        )
        
        # Test preprocessing of dollar variables
        template_with_dollars = "{{ $item }} at index {{ $index }}, first: {{ $is_first }}"
        result = resolver.resolve_templates(template_with_dollars, context)
        
        assert result == "test_value at index 5, first: True"
        assert "{{" not in result and "}}" not in result
        
        # Test 2: Cross-step template variable resolution
        context_with_steps = resolver.collect_context(
            pipeline_id="test_pipeline",
            step_results={
                "read_file": {"content": "Sample content", "size": 1024},
                "analyze_content": {"result": "Analysis successful", "score": 95}
            }
        )
        
        cross_step_template = "Content: {{ read_file.content }} | Size: {{ read_file.size }} | Analysis: {{ analyze_content.result }}"
        result = resolver.resolve_templates(cross_step_template, context_with_steps)
        
        assert result == "Content: Sample content | Size: 1024 | Analysis: Analysis successful"
        assert "{{" not in result and "}}" not in result
    
    def test_stream_a_nested_data_resolution(self, resolver):
        """Test recursive template resolution for nested data structures."""
        
        context = resolver.collect_context(
            pipeline_id="test_pipeline",
            step_results={
                "data_source": {
                    "metadata": {"title": "Test Data", "version": "1.0"},
                    "results": ["item1", "item2", "item3"]
                }
            }
        )
        
        # Test nested data structure with templates
        nested_data = {
            "title": "Processing {{ data_source.metadata.title }}",
            "details": {
                "version": "Version: {{ data_source.metadata.version }}",
                "items": [
                    "First: {{ data_source.results.0 }}", 
                    "Count: {{ data_source.results | length }}"
                ]
            }
        }
        
        result = resolver.resolve_templates(nested_data, context)
        
        assert result["title"] == "Processing Test Data"
        assert result["details"]["version"] == "Version: 1.0"
        assert result["details"]["items"][0] == "First: item1"
        assert result["details"]["items"][1] == "Count: 3"
    
    def test_stream_a_error_handling_and_debugging(self, resolver):
        """Test error handling and debugging capabilities from Stream A."""
        
        # Test debug info retrieval
        context = resolver.collect_context(
            pipeline_id="test_pipeline",
            task_id="test_task",
            pipeline_inputs={"debug_var": "debug_value"}
        )
        resolver.register_context(context)
        
        debug_info = resolver.get_debug_info()
        
        assert debug_info["has_current_context"] is True
        assert debug_info["current_context"]["pipeline_id"] == "test_pipeline"
        assert debug_info["current_context"]["task_id"] == "test_task"
        assert "debug_var" in debug_info["current_context"]["context_keys"]
        
        # Test unresolved variables detection
        unresolved_template = "{{ defined_var }} and {{ undefined_var }}"
        context_partial = resolver.collect_context(
            additional_context={"defined_var": "defined_value"}
        )
        
        # This should handle gracefully and provide debugging info
        result = resolver.resolve_templates(unresolved_template, context_partial)
        unresolved = resolver.get_unresolved_variables(unresolved_template, context_partial)
        
        assert "undefined_var" in unresolved
        assert "defined_var" not in unresolved
    
    def test_stream_b_loop_variable_integration(self, resolver):
        """Test loop variable integration fixes from Stream B."""
        
        # Create comprehensive loop context
        loop_context = resolver.loop_context_manager.create_loop_context(
            step_id="process_items",
            item="test_item",
            index=2,
            items=["item1", "item2", "test_item", "item4"],
            explicit_loop_name="main_loop"
        )
        
        # Register loop context
        resolver.loop_context_manager.push_loop(loop_context)
        
        # Ensure it's in the current stack
        if loop_context.loop_name not in resolver.loop_context_manager.current_loop_stack:
            resolver.loop_context_manager.current_loop_stack.append(loop_context.loop_name)
        
        # Test all loop variables with step results
        context = resolver.collect_context(
            pipeline_id="test_pipeline",
            task_id="process_item",
            step_results={
                "read_file": {"content": "File content", "size": 500}
            }
        )
        
        # Test loop variables and cross-step references together
        complex_template = """
        Processing item {{ $item }} at position {{ $index }}
        First item: {{ $is_first }}
        Last item: {{ $is_last }}
        File content: {{ read_file.content }}
        File size: {{ read_file.size }} bytes
        Total items: {{ $items | length }}
        """
        
        result = resolver.resolve_templates(complex_template, context)
        
        # Verify all variables resolved correctly
        assert "test_item" in result
        assert "position 2" in result
        assert "First item: False" in result  # index 2 is not first
        assert "Last item: False" in result   # index 2 is not last (4 items total)
        assert "File content: File content" in result
        assert "File size: 500 bytes" in result
        assert "Total items: 4" in result
        assert "{{" not in result and "}}" not in result
    
    def test_stream_b_multi_level_loop_support(self, resolver):
        """Test multi-level nested loop support from Stream B."""
        
        # Create nested loop contexts
        outer_loop = resolver.loop_context_manager.create_loop_context(
            step_id="outer_loop",
            item="outer_item",
            index=1,
            items=["outer1", "outer_item", "outer3"],
            explicit_loop_name="outer"
        )
        
        inner_loop = resolver.loop_context_manager.create_loop_context(
            step_id="inner_loop", 
            item="inner_item",
            index=0,
            items=["inner_item", "inner2"],
            explicit_loop_name="inner"
        )
        
        # Push both loops
        resolver.loop_context_manager.push_loop(outer_loop)
        resolver.loop_context_manager.push_loop(inner_loop)
        
        # Add to current stack
        for loop in [outer_loop, inner_loop]:
            if loop.loop_name not in resolver.loop_context_manager.current_loop_stack:
                resolver.loop_context_manager.current_loop_stack.append(loop.loop_name)
        
        # Test nested loop variable access
        context = resolver.collect_context(pipeline_id="test_pipeline")
        
        nested_template = """
        Outer: {{ $outer.item }} ({{ $outer.index }})
        Inner: {{ $inner.item }} ({{ $inner.index }})
        Current item: {{ $item }}
        Current index: {{ $index }}
        """
        
        result = resolver.resolve_templates(nested_template, context)
        
        # Should access both outer and inner loop variables
        assert "Outer: outer_item (1)" in result
        assert "Inner: inner_item (0)" in result
        assert "Current item: inner_item" in result  # Most recent loop
        assert "Current index: 0" in result
        assert "{{" not in result and "}}" not in result
    
    def test_tool_parameter_resolution_integration(self, resolver):
        """Test tool parameter resolution integration."""
        
        # Set up context with step results and loop variables
        context = resolver.collect_context(
            pipeline_id="test_pipeline",
            task_id="file_operation",
            pipeline_inputs={"output_dir": "/tmp/test"},
            step_results={
                "generate_content": {"result": "Generated text content", "metadata": {"words": 150}}
            },
            additional_context={
                "item": "document.txt",
                "index": 1
            }
        )
        
        # Test comprehensive tool parameter resolution
        tool_parameters = {
            "action": "write",
            "path": "{{ output_dir }}/{{ $item }}_processed_{{ $index }}.txt",
            "content": """
            # Generated Document {{ $index }}
            
            Original file: {{ $item }}
            Content: {{ generate_content.result }}
            Word count: {{ generate_content.metadata.words }}
            """,
            "metadata": {
                "source": "{{ $item }}",
                "generated_words": "{{ generate_content.metadata.words }}"
            }
        }
        
        resolved_params = resolver.resolve_before_tool_execution(
            tool_name="filesystem",
            tool_parameters=tool_parameters,
            context=context
        )
        
        # Verify all parameters resolved correctly
        assert resolved_params["action"] == "write"
        assert resolved_params["path"] == "/tmp/test/document.txt_processed_1.txt"
        assert "Generated text content" in resolved_params["content"]
        assert "Word count: 150" in resolved_params["content"]
        assert resolved_params["metadata"]["source"] == "document.txt"
        assert resolved_params["metadata"]["generated_words"] == "150"
        
        # Ensure no unresolved templates remain
        resolved_str = str(resolved_params)
        assert "{{" not in resolved_str and "}}" not in resolved_str
    
    def test_context_hierarchy_and_precedence(self, resolver):
        """Test context hierarchy and variable precedence."""
        
        # Create context with overlapping variable names from different sources
        context = resolver.collect_context(
            pipeline_id="test_pipeline",
            task_id="test_task",
            pipeline_inputs={"var1": "pipeline_value", "var2": "pipeline_only"},
            pipeline_parameters={"var1": "param_value", "var3": "param_only"},
            step_results={"var1": "step_value", "var4": "step_only"},
            additional_context={"var1": "additional_value", "var5": "additional_only"}
        )
        
        flat_context = context.to_flat_dict()
        
        # Test that all variables are available
        assert flat_context["var2"] == "pipeline_only"
        assert flat_context["var3"] == "param_only"
        assert flat_context["var4"] == "step_only"
        assert flat_context["var5"] == "additional_only"
        
        # Test variable precedence (additional_context should override others)
        assert flat_context["var1"] == "additional_value"
        
        # Test template resolution with hierarchy
        template = "{{ var1 }}-{{ var2 }}-{{ var3 }}-{{ var4 }}-{{ var5 }}"
        result = resolver.resolve_templates(template, context)
        
        assert result == "additional_value-pipeline_only-param_only-step_only-additional_only"


class TestRealPipelineTemplateValidation:
    """Test template resolution with real pipeline scenarios."""
    
    @pytest.fixture
    async def orchestrator_setup(self):
        """Set up orchestrator with models for real pipeline testing."""
        model_registry = init_models()
        if not model_registry or not model_registry.models:
            pytest.skip("No models available for testing")
        
        control_system = HybridControlSystem(model_registry)
        return Orchestrator(model_registry=model_registry, control_system=control_system)
    
    def create_test_data_files(self, temp_dir: Path):
        """Create test data files for pipeline testing."""
        data_dir = temp_dir / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Create test files
        (data_dir / "file1.txt").write_text("This is the content of file1 for testing.")
        (data_dir / "file2.txt").write_text("File2 contains different content for analysis.")
        (data_dir / "file3.txt").write_text("The third file has unique text data.")
        
        return data_dir
    
    def create_loop_test_pipeline(self, temp_dir: Path, data_dir: Path) -> str:
        """Create a test pipeline that exercises loop template resolution."""
        
        output_dir = temp_dir / "output"
        
        pipeline_yaml = f'''
id: template-resolution-test
name: Template Resolution Integration Test
description: Test all template resolution features
version: "1.0.0"

parameters:
  file_list:
    type: array
    default: ["file1.txt", "file2.txt", "file3.txt"]
    description: List of files to process
  output_dir:
    type: string
    default: "{output_dir}"
    description: Output directory

steps:
  # Create output directory
  - id: create_output_dir
    tool: filesystem
    action: write
    parameters:
      path: "{{{{ output_dir }}}}/.gitkeep"
      content: "# Template resolution test output"
    
  # Process each file with comprehensive template testing
  - id: process_files
    for_each: "{{{{ file_list }}}}"
    steps:
      # Read file
      - id: read_file
        tool: filesystem
        action: read
        parameters:
          path: "{data_dir}/{{{{ $item }}}}"
          
      # Test comprehensive template resolution in file output
      - id: save_processed_file
        tool: filesystem
        action: write
        parameters:
          path: "{{{{ output_dir }}}}/processed_{{{{ $item }}}}"
          content: |
            # Template Resolution Test: {{{{ $item }}}}
            
            ## Loop Variables
            Item: {{{{ $item }}}}
            Index: {{{{ $index }}}}
            Is First: {{{{ $is_first }}}}
            Is Last: {{{{ $is_last }}}}
            
            ## Cross-Step References  
            File Size: {{{{ read_file.size }}}} bytes
            File Content Length: {{{{ read_file.content | length }}}}
            
            ## Original Content
            {{{{ read_file.content }}}}
            
            ## Metadata
            Processing timestamp: {{{{ $index }}}
            Total files: {{{{ file_list | length }}}}
        dependencies:
          - read_file
    dependencies:
      - create_output_dir

outputs:
  total_files: "{{{{ file_list | length }}}}"
  output_directory: "{{{{ output_dir }}}}"
'''
        return pipeline_yaml
    
    @pytest.mark.asyncio
    async def test_real_pipeline_template_resolution(self, orchestrator_setup, temp_dir):
        """Test template resolution with a real pipeline execution."""
        
        orchestrator = orchestrator_setup
        if not orchestrator:
            pytest.skip("Orchestrator setup failed")
        
        # Create test data
        data_dir = self.create_test_data_files(temp_dir)
        
        # Create test pipeline
        pipeline_yaml = self.create_loop_test_pipeline(temp_dir, data_dir)
        
        # Execute pipeline
        inputs = {
            "file_list": ["file1.txt", "file2.txt", "file3.txt"],
            "output_dir": str(temp_dir / "output")
        }
        
        try:
            results = await orchestrator.execute_yaml(pipeline_yaml, inputs)
            
            # Verify pipeline executed successfully
            assert results is not None
            
            # Check output files were created
            output_dir = temp_dir / "output"
            assert output_dir.exists()
            
            output_files = list(output_dir.glob("processed_*.txt"))
            assert len(output_files) == 3
            
            # Verify template resolution in output files
            for i, output_file in enumerate(sorted(output_files)):
                content = output_file.read_text()
                
                # Check that NO unresolved templates remain
                assert "{{" not in content and "}}" not in content, f"Unresolved templates in {output_file.name}: {content}"
                
                # Check that loop variables resolved correctly
                assert f"Index: {i}" in content
                assert f"Is First: {i == 0}" in content
                assert f"Is Last: {i == 2}" in content
                
                # Check that cross-step references resolved
                assert "File Size:" in content and "bytes" in content
                assert "File Content Length:" in content
                
                # Check that file content was included
                assert len(content.strip()) > 100  # Should have substantial content
                
        except Exception as e:
            pytest.fail(f"Pipeline execution failed: {str(e)}")


class TestTemplateArtifactDetection:
    """Test detection and validation of template artifacts."""
    
    def test_template_artifact_scanner(self):
        """Test detection of unresolved template artifacts."""
        
        # Test data with various template artifacts
        test_outputs = [
            "Normal text with no templates",
            "Text with {{ unresolved_var }} template",
            "Loop variable $item not resolved",
            "Multiple {{ var1 }} and {{ var2 }} artifacts",
            "Mixed {{ template }} and $loop_var artifacts",
            "Properly resolved content only"
        ]
        
        def has_template_artifacts(text: str) -> List[str]:
            """Detect template artifacts in text."""
            artifacts = []
            
            # Check for unresolved Jinja2 templates
            if "{{" in text or "}}" in text:
                artifacts.append("Unresolved Jinja2 templates")
            
            # Check for unresolved loop variables
            if "$item" in text or "$index" in text or "$is_first" in text or "$is_last" in text:
                artifacts.append("Unresolved loop variables")
            
            return artifacts
        
        # Test artifact detection
        results = [has_template_artifacts(output) for output in test_outputs]
        
        assert len(results[0]) == 0  # Normal text
        assert "Unresolved Jinja2 templates" in results[1]
        assert "Unresolved loop variables" in results[2]
        assert "Unresolved Jinja2 templates" in results[3]
        assert len(results[4]) == 2  # Both types of artifacts
        assert len(results[5]) == 0  # Properly resolved
    
    def test_ai_response_quality_validation(self):
        """Test validation of AI model response quality."""
        
        # Test AI responses indicating template issues
        test_ai_responses = [
            "I analyzed the content and found...",  # Good response
            "I don't have access to {{ read_file.content }}",  # Template artifact issue
            "The variable $item was not provided",  # Loop variable issue
            "I can't see the content in {{ analyze_result }}",  # Cross-step reference issue
            "Based on the provided content: [actual analysis]"  # Good response
        ]
        
        def validate_ai_response_quality(response: str) -> List[str]:
            """Validate AI response for template-related issues."""
            issues = []
            
            # Check for template placeholder confusion
            template_confusion_markers = [
                "I don't have access to {{",
                "variable {{ ",
                "The variable $",
                "I can't see",
                "was not provided",
                "placeholder"
            ]
            
            for marker in template_confusion_markers:
                if marker.lower() in response.lower():
                    issues.append("AI received template placeholders instead of resolved content")
                    break
            
            return issues
        
        # Test response validation
        results = [validate_ai_response_quality(response) for response in test_ai_responses]
        
        assert len(results[0]) == 0  # Good response
        assert len(results[1]) > 0   # Template artifact issue
        assert len(results[2]) > 0   # Loop variable issue  
        assert len(results[3]) > 0   # Cross-step reference issue
        assert len(results[4]) == 0  # Good response


class TestRegressionPrevention:
    """Test framework for preventing template resolution regressions."""
    
    def test_comprehensive_template_scenario_matrix(self, resolver):
        """Test matrix of all template resolution scenarios."""
        
        # Define comprehensive test matrix
        test_scenarios = [
            # Basic scenarios
            {"template": "{{ simple_var }}", "context": {"simple_var": "value"}, "expected": "value"},
            
            # Loop variable scenarios  
            {"template": "{{ $item }}", "context": {"item": "loop_value"}, "expected": "loop_value"},
            {"template": "{{ $index }}", "context": {"index": 5}, "expected": "5"},
            {"template": "{{ $is_first }}", "context": {"is_first": True}, "expected": "True"},
            {"template": "{{ $is_last }}", "context": {"is_last": False}, "expected": "False"},
            
            # Cross-step reference scenarios
            {"template": "{{ step_result.data }}", "context": {"step_result": {"data": "step_value"}}, "expected": "step_value"},
            {"template": "{{ read_file.content }}", "context": {"read_file": {"content": "file_content"}}, "expected": "file_content"},
            
            # Complex nested scenarios
            {"template": "{{ data.nested.value }}", "context": {"data": {"nested": {"value": "nested_value"}}}, "expected": "nested_value"},
            
            # Mixed scenarios
            {"template": "Item {{ $item }} has {{ result.count }} entries", 
             "context": {"item": "A", "result": {"count": 3}}, 
             "expected": "Item A has 3 entries"},
        ]
        
        # Test each scenario
        for i, scenario in enumerate(test_scenarios):
            context = resolver.collect_context(
                pipeline_id=f"test_{i}",
                additional_context=scenario["context"]
            )
            
            result = resolver.resolve_templates(scenario["template"], context)
            
            assert result == scenario["expected"], f"Scenario {i} failed: {scenario['template']} -> {result} (expected {scenario['expected']})"
            assert "{{" not in result and "}}" not in result, f"Unresolved templates in scenario {i}: {result}"
    
    def test_performance_regression_prevention(self, resolver):
        """Test that template resolution performance is acceptable."""
        import time
        
        # Create large context
        large_context = resolver.collect_context(
            pipeline_id="performance_test",
            step_results={f"step_{i}": {"result": f"result_{i}"} for i in range(100)},
            additional_context={f"var_{i}": f"value_{i}" for i in range(100)}
        )
        
        # Test template resolution performance
        template = "Processing {{ $item }} with results: " + " | ".join([f"{{{{ step_{i}.result }}}}" for i in range(10)])
        
        start_time = time.time()
        
        for _ in range(10):  # Multiple iterations
            result = resolver.resolve_templates(template, large_context)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        
        # Performance should be reasonable (less than 100ms per resolution)
        assert avg_time < 0.1, f"Template resolution too slow: {avg_time:.3f}s average"
        
        # Result should be properly resolved
        assert "{{" not in result and "}}" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])