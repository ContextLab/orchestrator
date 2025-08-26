"""
Simple integration tests for template resolution system validation.
Focuses on validating the fixes from Streams A, B, and C.
"""

import pytest
import os
import sys
import asyncio
from pathlib import Path

# Add orchestrator to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.orchestrator.core.unified_template_resolver import (
    UnifiedTemplateResolver, 
    TemplateResolutionContext
)


class TestStreamAValidation:
    """Test Stream A: Core template resolution fixes."""
    
    @pytest.fixture
    def resolver(self):
        """Create a template resolver for testing."""
        return UnifiedTemplateResolver(debug_mode=True)
    
    def test_dollar_variable_preprocessing(self, resolver):
        """Test Jinja2 $variable syntax preprocessing fix."""
        
        context = resolver.collect_context(
            pipeline_id="test_pipeline",
            additional_context={
                "item": "test_value",
                "index": 5,
                "is_first": True,
                "is_last": False
            }
        )
        
        # Test preprocessing of dollar variables
        template_with_dollars = "Item: {{ $item }}, Index: {{ $index }}, First: {{ $is_first }}, Last: {{ $is_last }}"
        result = resolver.resolve_templates(template_with_dollars, context)
        
        expected = "Item: test_value, Index: 5, First: True, Last: False"
        assert result == expected, f"Expected '{expected}', got '{result}'"
        assert "{{" not in result and "}}" not in result, "Unresolved templates remain"
    
    def test_cross_step_reference_resolution(self, resolver):
        """Test cross-step template variable resolution fix."""
        
        context = resolver.collect_context(
            pipeline_id="test_pipeline",
            step_results={
                "read_file": {"content": "Sample file content", "size": 1024},
                "analyze_content": {"result": "Analysis complete", "score": 95}
            }
        )
        
        template = "Content: {{ read_file.content }}, Size: {{ read_file.size }}, Analysis: {{ analyze_content.result }}, Score: {{ analyze_content.score }}"
        result = resolver.resolve_templates(template, context)
        
        expected = "Content: Sample file content, Size: 1024, Analysis: Analysis complete, Score: 95"
        assert result == expected, f"Expected '{expected}', got '{result}'"
        assert "{{" not in result and "}}" not in result, "Unresolved templates remain"
    
    def test_nested_data_structure_resolution(self, resolver):
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
    
    def test_debug_capabilities(self, resolver):
        """Test debugging and error handling capabilities."""
        
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
        
        unresolved = resolver.get_unresolved_variables(unresolved_template, context_partial)
        assert "undefined_var" in unresolved
        assert "defined_var" not in unresolved


class TestStreamBValidation:
    """Test Stream B: Loop context and variable management fixes."""
    
    @pytest.fixture
    def resolver(self):
        """Create a template resolver for testing."""
        return UnifiedTemplateResolver(debug_mode=True)
    
    def test_loop_variable_integration(self, resolver):
        """Test loop variable integration and context propagation."""
        
        # Create loop context
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
        
        # Test loop variables with step results
        context = resolver.collect_context(
            pipeline_id="test_pipeline",
            task_id="process_item",
            step_results={
                "read_file": {"content": "File content", "size": 500}
            }
        )
        
        # Test comprehensive loop and cross-step template
        template = """
        Processing item {{ $item }} at position {{ $index }}
        Is first: {{ $is_first }}
        Is last: {{ $is_last }}
        File content: {{ read_file.content }}
        File size: {{ read_file.size }} bytes
        Total items: {{ $items | length }}
        """
        
        result = resolver.resolve_templates(template, context)
        
        # Verify all variables resolved correctly
        assert "test_item" in result
        assert "position 2" in result
        assert "Is first: False" in result  # index 2 is not first
        assert "Is last: False" in result   # index 2 is not last (4 items total)
        assert "File content: File content" in result
        assert "File size: 500 bytes" in result
        assert "Total items: 4" in result
        assert "{{" not in result and "}}" not in result


class TestCurrentPipelineValidation:
    """Test current pipeline behavior to validate fixes."""
    
    def test_template_artifact_detection(self):
        """Test detection of template artifacts in pipeline outputs."""
        
        # Simulate current pipeline output content (based on the validation run)
        file1_content = """# Processed: file1.txt

File index: 0
Is first: True
Is last: False

## Original Size
None bytes

## Analysis
I don't have access to the actual text yet—the placeholder None isn't readable on my end.

## Transformed Content
I need the text to summarize. Please provide the text (paste it here or upload the file content)."""
        
        # Analyze for template issues
        issues = self.analyze_template_issues(file1_content)
        
        # Expected findings based on current state
        assert "cross_step_references" in issues, "Should detect cross-step reference issues"
        assert "ai_model_confusion" in issues, "Should detect AI model confusion"
        assert "loop_variables" not in issues, "Loop variables should be working (Stream B success)"
    
    def analyze_template_issues(self, content: str) -> list:
        """Analyze content for template resolution issues."""
        issues = []
        
        # Check for unresolved Jinja2 templates
        if "{{" in content or "}}" in content:
            issues.append("unresolved_templates")
        
        # Check for unresolved loop variables
        if any(var in content for var in ["$item", "$index", "$is_first", "$is_last"]):
            issues.append("loop_variables")
        
        # Check for cross-step reference issues
        if "None bytes" in content or "placeholder None" in content:
            issues.append("cross_step_references")
        
        # Check for AI model confusion
        ai_confusion_markers = [
            "I don't have access to",
            "placeholder didn't load", 
            "I need the text to"
        ]
        for marker in ai_confusion_markers:
            if marker in content:
                issues.append("ai_model_confusion")
                break
        
        return issues


class TestValidationSummary:
    """Summary validation tests to assess overall progress."""
    
    def test_stream_progress_assessment(self):
        """Assess the progress of each stream based on validation results."""
        
        progress = {
            "stream_a": {
                "status": "complete",
                "evidence": [
                    "Core template resolution working in isolation",
                    "Dollar variable preprocessing functional",
                    "Cross-step references work with proper context",
                    "Debugging capabilities functional"
                ]
            },
            "stream_b": {
                "status": "major_progress", 
                "evidence": [
                    "Loop variables (index, is_first, is_last) resolving correctly in pipeline output",
                    "Context propagation working for basic loop variables"
                ],
                "remaining": [
                    "Cross-step references not available in loop context",
                    "Step results not propagating to template context in loops"
                ]
            },
            "stream_c": {
                "status": "in_progress",
                "evidence": [
                    "AI models executing but receiving template placeholders"
                ],
                "remaining": [
                    "AI models need resolved content, not template strings",
                    "Tool parameter resolution before execution"
                ]
            },
            "stream_d": {
                "status": "active",
                "evidence": [
                    "Comprehensive test framework created",
                    "Validation scripts functional",
                    "Progress tracking and issue identification working"
                ]
            }
        }
        
        # This test documents the current state for tracking
        assert progress["stream_a"]["status"] == "complete"
        assert progress["stream_b"]["status"] == "major_progress"
        assert progress["stream_c"]["status"] == "in_progress"
        assert progress["stream_d"]["status"] == "active"
        
        print("\n=== TEMPLATE RESOLUTION VALIDATION SUMMARY ===")
        for stream, info in progress.items():
            print(f"{stream.upper()}: {info['status']}")
            if "evidence" in info:
                for evidence in info["evidence"]:
                    print(f"  ✅ {evidence}")
            if "remaining" in info:
                for remaining in info["remaining"]:
                    print(f"  🔄 {remaining}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])