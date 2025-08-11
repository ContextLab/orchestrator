#!/usr/bin/env python
"""Test template rendering in filesystem operations."""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, 'src')

from orchestrator.orchestrator import Orchestrator
from orchestrator.core.pipeline import Pipeline


async def test_filesystem_templates():
    """Test that filesystem operations properly render templates."""
    
    # Create a simple pipeline with filesystem write
    pipeline_yaml = """
id: test-filesystem-templates
name: Test Filesystem Template Rendering
parameters:
  test_value:
    type: string
    default: "Hello World"
  output_dir:
    type: string
    default: "examples/outputs/template_test"

steps:
  - id: generate_content
    action: generate_text
    parameters:
      prompt: "Say exactly: The test worked perfectly"
      max_tokens: 50
      
  - id: save_file
    tool: filesystem
    action: write
    dependencies:
      - generate_content
    parameters:
      path: "{{ output_dir }}/test_output.txt"
      content: |
        Test Value: {{ test_value }}
        Generated: {{ generate_content.result }}
        Timestamp: {{ execution.timestamp }}
"""
    
    # Initialize orchestrator with models
    from orchestrator.models.model_registry import ModelRegistry
    registry = ModelRegistry()
    
    orchestrator = Orchestrator(model_registry=registry)
    await orchestrator.initialize()
    
    # Compile and execute pipeline
    pipeline = await orchestrator.yaml_compiler.compile(pipeline_yaml, {})
    
    # Execute with test inputs
    results = await orchestrator.execute_pipeline(
        pipeline,
        parameters={
            "test_value": "Template Test Success",
            "output_dir": "examples/outputs/template_test"
        }
    )
    
    print("\n=== Execution Results ===")
    for step_id, result in results.items():
        print(f"{step_id}: {result}")
    
    # Check the output file
    output_file = Path("examples/outputs/template_test/test_output.txt")
    if output_file.exists():
        print(f"\n=== Output File Content ===")
        print(output_file.read_text())
        
        # Check for unrendered templates
        content = output_file.read_text()
        if "{{" in content or "{%" in content:
            print("\n❌ FAILED: Output contains unrendered templates!")
            return False
        else:
            print("\n✅ SUCCESS: All templates were rendered!")
            return True
    else:
        print(f"\n❌ FAILED: Output file not created at {output_file}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_filesystem_templates())
    sys.exit(0 if success else 1)