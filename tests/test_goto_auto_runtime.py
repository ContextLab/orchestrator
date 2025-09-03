"""Test that goto AUTO tags are resolved at runtime with template variables."""

import pytest
import asyncio
import yaml

from src.orchestrator.compiler.control_flow_compiler import ControlFlowCompiler
from src.orchestrator.compiler.yaml_compiler import YAMLCompiler
from src.orchestrator.models.model_registry import ModelRegistry
from src.orchestrator.control_flow.auto_resolver import ControlFlowAutoResolver

from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider


class TestGotoAutoRuntime:
    """Test cases for runtime resolution of AUTO tags in goto fields."""
    
    @pytest.mark.asyncio
    async def test_goto_auto_preserved_in_yaml_compiler(self):
        """Test that YAMLCompiler preserves AUTO tags in goto fields."""
        
        pipeline_yaml = """
id: test-goto-auto-yaml
name: Test Goto AUTO with YAMLCompiler

steps:
  - id: get_value
    action: generate
    parameters:
      prompt: "Generate a value"
      
  - id: router
    action: process
    goto: "<AUTO>Based on {{ get_value.result }}, choose target</AUTO>"
    depends_on: [get_value]
"""
        
        # Create compiler without model registry
        compiler = YAMLCompiler(model_registry=None)
        
        # Compile without resolving ambiguities
        pipeline = await compiler.compile(pipeline_yaml, resolve_ambiguities=False)
        
        # Check router task
        router_task = pipeline.tasks.get("router")
        assert router_task is not None
        
        # The goto should be in metadata and contain AUTO tag
        goto_value = router_task.metadata.get("goto", "")
        assert "<AUTO>" in goto_value, f"AUTO tag was resolved! Got: {goto_value}"
        assert "{{ get_value.result }}" in goto_value, f"Template was resolved! Got: {goto_value}"
        
    @pytest.mark.asyncio
    async def test_goto_auto_preserved_in_control_flow_compiler(self):
        """Test that ControlFlowCompiler preserves AUTO tags in goto fields."""
        
        pipeline_yaml = """
id: test-goto-auto-cf
name: Test Goto AUTO with ControlFlowCompiler

steps:
  - id: check_status
    action: evaluate_condition
    parameters:
      condition: "status == 'success'"
      
  - id: router
    action: process
    goto: "<AUTO>If {{ check_status.result }} go to 'success_path' else 'failure_path'</AUTO>"
    depends_on: [check_status]
    
  - id: success_path
    action: echo
    parameters:
      message: "Success!"
      
  - id: failure_path
    action: echo
    parameters:
      message: "Failed!"
"""
        
        # Create compiler
        compiler = ControlFlowCompiler(model_registry=None)
        
        # Compile without resolving ambiguities
        pipeline = await compiler.compile(pipeline_yaml, resolve_ambiguities=False)
        
        # Check router task
        router_task = pipeline.tasks.get("router")
        assert router_task is not None
        
        # The goto should contain AUTO tag
        goto_value = router_task.metadata.get("goto", "")
        assert "<AUTO>" in goto_value, f"AUTO tag was resolved! Got: {goto_value}"
        assert "{{ check_status.result }}" in goto_value, f"Template was resolved! Got: {goto_value}"
        
    @pytest.mark.asyncio
    async def test_goto_auto_runtime_resolution(self):
        """Test that AUTO tags in goto are resolved at runtime with context."""
        
        # Create auto resolver
        auto_resolver = ControlFlowAutoResolver(model_registry=None)
        
        # Test goto expression with template variable
        goto_expr = "<AUTO>Based on result {{ check_result.status }}, go to 'handler_{{ check_result.status }}'</AUTO>"
        
        # Mock context with step results
        context = {}
        step_results = {
            "check_result": {
                "status": "success",
                "value": 42
            }
        }
        
        # Test that template variables are replaced before AUTO resolution
        resolved = await auto_resolver._resolve_auto_tags(goto_expr, context, step_results)
        
        # Since we don't have a model, it should return a default
        # But the template variable should have been replaced
        assert "{{ check_result.status }}" not in str(resolved), "Template variable not replaced"
        
    def test_goto_field_skipped_in_auto_resolution(self):
        """Test that goto fields are skipped during compile-time AUTO resolution."""
        
        yaml_content = """
id: test-skip-goto
name: Test Skip Goto

steps:
  - id: step1
    action: generate
    parameters:
      prompt: "<AUTO>Generate something</AUTO>"  # This should be resolved
      
  - id: step2
    action: process
    goto: "<AUTO>Choose next step</AUTO>"  # This should NOT be resolved
    parameters:
      data: "test"
"""
        
        from src.orchestrator.compiler.auto_tag_yaml_parser import AutoTagYAMLParser
        
        parser = AutoTagYAMLParser()
        parsed = parser.parse(yaml_content)
        
        # The parsed YAML should have AUTO tags preserved in goto
        step2 = next(s for s in parsed["steps"] if s["id"] == "step2")
        assert "<AUTO>" in step2["goto"], "AUTO tag in goto was modified during parsing"
        
    @pytest.mark.asyncio
    async def test_complex_goto_auto_expression(self):
        """Test complex goto AUTO expressions with multiple template variables."""
        
        pipeline_yaml = """
id: test-complex-goto
name: Test Complex Goto AUTO

steps:
  - id: analyze
    action: analyze
    parameters:
      data: "input"
      
  - id: validate
    action: validate
    parameters:
      result: "{{ analyze.result }}"
      
  - id: router
    action: process
    goto: "<AUTO>Based on analysis {{ analyze.result.type }} and validation {{ validate.passed }}, route to appropriate handler</AUTO>"
    depends_on: [analyze, validate]
"""
        
        compiler = ControlFlowCompiler(model_registry=None)
        pipeline = await compiler.compile(pipeline_yaml, resolve_ambiguities=False)
        
        router_task = pipeline.tasks.get("router")
        goto_value = router_task.metadata.get("goto", "")
        
        # All template variables should be preserved
        assert "{{ analyze.result.type }}" in goto_value
        assert "{{ validate.passed }}" in goto_value
        assert "<AUTO>" in goto_value
        

if __name__ == "__main__":
    pytest.main([__file__, "-v"])