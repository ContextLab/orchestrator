#!/usr/bin/env python3
"""Test how results are registered in template manager."""

import asyncio
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Enable debug logging for template manager
logging.getLogger('orchestrator.core.template_manager').setLevel(logging.DEBUG)
logging.getLogger('orchestrator.orchestrator').setLevel(logging.DEBUG)

async def main():
    from src.orchestrator import Orchestrator
    from scripts.run_pipeline import init_models
    
    # Initialize models first
    await init_models()
    
    # Create a minimal test pipeline
    test_yaml = """
name: Test Registration
description: Test result registration

parameters:
  topic: "test"

steps:
  - id: step1
    action: generate_text
    parameters:
      prompt: "Say hello"
      max_tokens: 20
      
  - id: step2
    tool: filesystem  
    action: write
    parameters:
      path: "test_reg.txt"
      content: |
        Topic: {{ topic }}
        Step1 result: {{ step1.result }}
        Step1 type: {{ step1.__class__.__name__ }}
    dependencies:
      - step1
"""
    
    # Initialize orchestrator with debug
    orchestrator = Orchestrator(debug_templates=True)
    
    # Run the test pipeline
    result = await orchestrator.execute_yaml(
        test_yaml,
        context={"topic": "testing"}
    )
    
    print("\n=== Pipeline Result ===")
    print(f"Success: {'steps' in result}")
    if 'steps' in result:
        print(f"Steps executed: {list(result['steps'].keys())}")
        print(f"\nStep1 result: {result['steps'].get('step1')}")
    
    # Check the output file
    from pathlib import Path
    output_file = Path("test_reg.txt")
    if output_file.exists():
        content = output_file.read_text()
        print(f"\n=== File Content ===")
        print(content)
        if "{{" in content:
            print("\nERROR: Templates not rendered!")
        else:
            print("\nSUCCESS: Templates rendered!")
        # Clean up
        output_file.unlink()

if __name__ == "__main__":
    asyncio.run(main())