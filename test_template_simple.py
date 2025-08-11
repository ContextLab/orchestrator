#!/usr/bin/env python3
"""
Simple test for template rendering - uses real APIs
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import Orchestrator, init_models


async def test_simple_template():
    """Test simple template rendering."""
    
    yaml_content = """
id: test-template-simple
name: Simple Template Test
parameters:
  message:
    type: string
    default: "Testing templates"
  output_dir:
    type: string  
    default: "examples/outputs/template_test"

steps:
  - id: create_content
    action: generate_text
    parameters:
      prompt: "Reply with exactly: SUCCESS"
      max_tokens: 10
      
  - id: save_result
    tool: filesystem
    action: write
    dependencies:
      - create_content
    parameters:
      path: "{{ output_dir }}/result.txt"
      content: |
        Message: {{ message }}
        Result: {{ create_content }}
        Done!
"""
    
    # Initialize models
    print("Initializing models...")
    model_registry = init_models()
    
    # Initialize orchestrator
    orchestrator = Orchestrator(model_registry=model_registry)
    
    # Run pipeline
    print("Running simple template test...")
    inputs = {
        "message": "Hello from template test",
        "output_dir": "examples/outputs/simple_template_test"
    }
    
    # Create output directory
    Path(inputs["output_dir"]).mkdir(parents=True, exist_ok=True)
    
    results = await orchestrator.execute_yaml(yaml_content, inputs)
    
    print(f"Results: {results}")
    
    # Check the output file
    output_file = Path(inputs["output_dir"]) / "result.txt"
    if output_file.exists():
        content = output_file.read_text()
        print(f"\nFile content:\n{content}")
        
        # Check for templates
        if "{{" in content or "{%" in content:
            print("❌ ERROR: Unrendered templates found in output!")
            return False
        
        if "Hello from template test" in content:
            print("✅ SUCCESS: Template rendering works!")
            return True
        else:
            print("❌ ERROR: Message not found in output")
            return False
    else:
        print(f"❌ ERROR: Output file not created: {output_file}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_simple_template())
    sys.exit(0 if success else 1)