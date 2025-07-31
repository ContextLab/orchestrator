#!/usr/bin/env python3
"""Test script for ContextManager proof-of-concept."""

import asyncio
import json
from pathlib import Path

from src.orchestrator.core.context_manager import ContextManager
from src.orchestrator.core.template_manager import TemplateManager
from src.orchestrator.core.task import Task


def test_basic_context():
    """Test basic context management."""
    print("\n=== Testing Basic Context Management ===")
    
    cm = ContextManager()
    cm.initialize_template_manager()
    
    # Set global context
    cm.register_variable("version", "1.0.0", "global")
    cm.register_variable("environment", "production", "global")
    
    # Push pipeline context
    with cm.pipeline_context("test-pipeline", {"topic": "AI Safety", "language": "en"}):
        print(f"Pipeline context: {cm.get_merged_context()}")
        
        # Test template rendering
        template = "Research on {{topic}} in {{language}} (v{{version}})"
        rendered = cm.render_template(template)
        print(f"Rendered: {rendered}")
        assert "AI Safety" in rendered
        assert "en" in rendered
        assert "1.0.0" in rendered
        
        # Push task context
        task = Task(id="analyze", name="Analyze Topic", action="analyze_text", parameters={})
        with cm.task_context(task, cm.get_merged_context()):
            print(f"Task context: {cm.get_merged_context()}")
            
            # Register task result (simulate how orchestrator does it)
            cm.register_variable("analyze", {"result": "Analysis complete"}, "current")
            
            # Test access to task result
            template2 = "Task {{task_name}} completed: {{analyze.result}}"
            rendered2 = cm.render_template(template2)
            print(f"Rendered: {rendered2}")
            assert "Analyze Topic" in rendered2
            assert "Analysis complete" in rendered2
    
    print("✓ Basic context test passed")


def test_nested_loops():
    """Test nested loop contexts."""
    print("\n=== Testing Nested Loop Contexts ===")
    
    cm = ContextManager()
    cm.initialize_template_manager()
    
    # Pipeline context
    with cm.pipeline_context("loop-test", {"title": "Loop Test"}):
        languages = ["en", "es", "fr"]
        formats = ["pdf", "html"]
        
        # Outer loop
        for i, lang in enumerate(languages):
            with cm.loop_context("translate", lang, i):
                print(f"Outer loop - Language: {lang}")
                
                # Inner loop
                for j, fmt in enumerate(formats):
                    with cm.loop_context("format", fmt, j):
                        # Test access to both loop variables
                        template = "Generate {{title}} in {{item}} format for language {{index}}"
                        rendered = cm.render_template(template)
                        print(f"  Inner loop - Rendered: {rendered}")
                        
                        # Verify both loop contexts are accessible
                        ctx = cm.get_merged_context()
                        assert ctx["$item"] == fmt  # Current loop item
                        assert ctx["item"] == fmt  # Alternative syntax
                        
                        # Test that nested contexts work
                        assert "Loop Test" in rendered
                        assert fmt in rendered
                        assert str(j) in rendered
    
    print("✓ Nested loop test passed")


def test_advanced_control_flow():
    """Test advanced control flow with conditionals and loops."""
    print("\n=== Testing Advanced Control Flow ===")
    
    cm = ContextManager()
    cm.initialize_template_manager()
    
    # Simulate control_flow_advanced pipeline
    inputs = {
        "input_text": "Climate change impacts global food security",
        "languages": ["es", "fr", "de"],
        "quality_threshold": 0.7,
        "output": "test_output"
    }
    
    with cm.pipeline_context("control-flow-advanced", inputs):
        # Simulate analyze_text task
        with cm.task_context(
            Task(id="analyze_text", name="Analyze Text", action="analyze_text", parameters={}),
            cm.get_merged_context()
        ):
            # Register at pipeline level so it persists after task context
            cm.register_variable("analyze_text", {"result": "Text quality: 0.5"}, "pipeline")
            
        # Simulate check_quality task
        with cm.task_context(
            Task(id="check_quality", name="Check Quality", action="generate_text", parameters={}),
            cm.get_merged_context()
        ):
            # Template should have access to threshold and analysis
            template = "Quality {{analyze_text.result}} vs threshold {{quality_threshold}}"
            rendered = cm.render_template(template)
            print(f"Quality check: {rendered}")
            assert "0.5" in rendered
            assert "0.7" in rendered
            
            cm.register_variable("check_quality", {"result": "improve"}, "pipeline")
            
        # Simulate enhance_text task (conditional)
        check_result = cm.get_merged_context().get("check_quality", {})
        if "improve" in check_result.get("result", ""):
            with cm.task_context(
                Task(id="enhance_text", name="Enhance Text", action="generate_text", parameters={}),
                cm.get_merged_context()
            ):
                enhanced = f"Enhanced: {inputs['input_text']}"
                cm.register_variable("enhance_text", {"result": enhanced}, "pipeline")
                
        # Simulate translation loop
        for i, lang in enumerate(inputs["languages"]):
            with cm.loop_context("translate_text", lang, i):
                print(f"\nTranslating to {lang}:")
                
                # Test complex template with conditionals
                template = """
Translating to {{item}}:
Original: {% if enhance_text.result %}{{enhance_text.result}}{% else %}{{input_text}}{% endif %}
Output path: {{output}}/translations/{{input_text[:50] | slugify}}_{{item}}.txt
"""
                rendered = cm.render_template(template)
                print(rendered)
                
                # Verify all variables are accessible
                assert lang in rendered
                assert "Enhanced:" in rendered  # Should use enhanced text
                assert "test_output/translations/" in rendered
    
    print("✓ Advanced control flow test passed")


def test_tool_contexts():
    """Test tool execution contexts."""
    print("\n=== Testing Tool Contexts ===")
    
    cm = ContextManager()
    cm.initialize_template_manager()
    
    # Simulate different tool executions
    tools = [
        ("filesystem", {"action": "write", "path": "{{output}}/{{filename}}", "content": "{{data}}"}),
        ("web-search", {"query": "{{topic}} {{subtopic}}", "max_results": 10}),
        ("generate-text", {"prompt": "Write about {{topic}} for {{audience}}", "model": "gpt-4"})
    ]
    
    with cm.pipeline_context("tool-test", {"output": "/tmp", "topic": "AI Ethics"}):
        for tool_name, params in tools:
            # Set up additional context
            if tool_name == "filesystem":
                cm.register_variable("filename", "report.txt", "current")
                cm.register_variable("data", "Test content", "current")
            elif tool_name == "web-search":
                cm.register_variable("subtopic", "bias", "current")
            elif tool_name == "generate-text":
                cm.register_variable("audience", "students", "current")
                
            with cm.tool_context(tool_name, params):
                print(f"\nTool: {tool_name}")
                
                # Render all parameters
                rendered_params = {}
                for key, value in params.items():
                    if isinstance(value, str):
                        rendered_params[key] = cm.render_template(value)
                    else:
                        rendered_params[key] = value
                        
                print(f"Rendered params: {json.dumps(rendered_params, indent=2)}")
                
                # Verify rendering
                if tool_name == "filesystem":
                    assert rendered_params["path"] == "/tmp/report.txt"
                    assert rendered_params["content"] == "Test content"
                elif tool_name == "web-search":
                    assert "AI Ethics bias" in rendered_params["query"]
                elif tool_name == "generate-text":
                    assert "AI Ethics" in rendered_params["prompt"]
                    assert "students" in rendered_params["prompt"]
    
    print("✓ Tool context test passed")


def test_context_validation():
    """Test template validation."""
    print("\n=== Testing Context Validation ===")
    
    cm = ContextManager()
    cm.initialize_template_manager()
    
    # Test with missing variables
    template1 = "Hello {{name}}, your score is {{score}}"
    errors1 = cm.validate_template(template1)
    print(f"Validation errors (no context): {errors1}")
    assert len(errors1) > 0
    
    # Add partial context
    cm.register_variable("name", "Alice", "global")
    errors2 = cm.validate_template(template1)
    print(f"Validation errors (partial context): {errors2}")
    
    # Add complete context
    cm.register_variable("score", 95, "global")
    errors3 = cm.validate_template(template1)
    print(f"Validation errors (complete context): {errors3}")
    assert len(errors3) == 0
    
    # Test invalid syntax
    template2 = "Hello {{name"
    errors4 = cm.validate_template(template2)
    print(f"Validation errors (syntax error): {errors4}")
    assert len(errors4) > 0
    
    print("✓ Validation test passed")


def test_debug_output():
    """Test debug context display."""
    print("\n=== Testing Debug Output ===")
    
    cm = ContextManager()
    cm.initialize_template_manager()
    
    # Build up complex context
    cm.register_variable("version", "1.0", "global")
    
    with cm.pipeline_context("debug-test", {"input": "test"}):
        with cm.task_context(Task(id="task1", name="Task 1", action="test", parameters={}), {}):
            with cm.loop_context("loop1", "item1", 0):
                with cm.tool_context("test-tool", {"param": "value"}):
                    print(cm.debug_context())
    
    print("✓ Debug output test passed")


if __name__ == "__main__":
    print("Starting ContextManager tests...")
    
    test_basic_context()
    test_nested_loops()
    test_advanced_control_flow()
    test_tool_contexts()
    test_context_validation()
    test_debug_output()
    
    print("\n✅ All tests passed!")