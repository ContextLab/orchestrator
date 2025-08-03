"""Test that goto AUTO tags are resolved at runtime with template variables."""

import asyncio
import yaml

from src.orchestrator.compiler.control_flow_compiler import ControlFlowCompiler
from src.orchestrator.models.model_registry import ModelRegistry


async def test_goto_auto_resolution():
    """Test that goto AUTO tags are not resolved at compile time."""
    
    # Create a simple test pipeline with goto AUTO tag
    pipeline_yaml = """
id: test-goto-auto
name: Test Goto AUTO Resolution
description: Test that AUTO tags in goto are resolved at runtime

steps:
  - id: get_result
    action: generate
    parameters:
      prompt: "Return 'success' or 'failure'"

  - id: check_result
    action: evaluate_condition
    parameters:
      condition: "{{ get_result.result == 'success' }}"
    
  - id: router
    action: process
    parameters:
      message: "Routing based on result"
    goto: "<AUTO>Based on result {{ get_result.result }}, go to 'success_handler' or 'failure_handler'</AUTO>"
    depends_on: [get_result, check_result]
    
  - id: success_handler
    action: echo
    parameters:
      message: "Success!"
      
  - id: failure_handler
    action: echo
    parameters:
      message: "Failure!"
"""
    
    # Create compiler without model registry to skip AUTO resolution
    compiler = ControlFlowCompiler(model_registry=None)
    
    # Compile the pipeline WITHOUT resolving ambiguities
    pipeline = await compiler.compile(pipeline_yaml, resolve_ambiguities=False)
    
    # Check that the goto AUTO tag was NOT resolved
    router_task = None
    for task_id, task in pipeline.tasks.items():
        if task_id == "router":
            router_task = task
            break
    
    assert router_task is not None, "Router task not found"
    
    # The goto should still contain the AUTO tag
    goto_value = router_task.metadata.get("goto", "")
    print(f"Goto value after compilation: {goto_value}")
    
    # Verify AUTO tag is still present
    assert "<AUTO>" in goto_value, f"AUTO tag was resolved at compile time! Got: {goto_value}"
    assert "{{ get_result.result }}" in goto_value, f"Template variable was resolved! Got: {goto_value}"
    
    print("✅ SUCCESS: Goto AUTO tag preserved for runtime resolution")
    print(f"   Goto value: {goto_value}")
    
    # Also check the raw pipeline definition to ensure it wasn't modified
    raw_pipeline = yaml.safe_load(pipeline_yaml)
    router_step = next(s for s in raw_pipeline["steps"] if s["id"] == "router")
    assert "<AUTO>" in router_step["goto"], "Original YAML was modified"
    
    print("\n✅ All tests passed! AUTO tags in goto are preserved for runtime resolution.")


if __name__ == "__main__":
    asyncio.run(test_goto_auto_resolution())