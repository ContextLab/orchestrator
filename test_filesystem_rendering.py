#!/usr/bin/env python3
"""Test filesystem template rendering issue."""

import asyncio
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Ensure all loggers are at INFO level
logging.getLogger().setLevel(logging.INFO)
logging.getLogger('orchestrator').setLevel(logging.INFO)

async def main():
    from src.orchestrator import Orchestrator
    from scripts.run_pipeline import init_models
    
    # Initialize models first
    await init_models()
    
    # Create a simple test pipeline
    test_yaml = """
name: Test Filesystem Rendering
description: Test if filesystem templates are rendered

parameters:
  test_value: "RENDERED_VALUE"

steps:
  - id: test_write
    tool: filesystem
    action: write
    parameters:
      path: "test_output.md"
      content: |
        # Test Report
        Value: {{ test_value }}
        Timestamp: {{ execution.timestamp }}
"""
    
    # Initialize orchestrator
    orchestrator = Orchestrator()
    
    # Run the test pipeline
    result = await orchestrator.execute_yaml(
        test_yaml,
        context={"test_value": "SUCCESS"}
    )
    
    print("\n=== Pipeline Result ===")
    print(f"Steps: {list(result.get('steps', {}).keys())}")
    if 'test_write' in result.get('steps', {}):
        print(f"test_write result: {result['steps']['test_write']}")
    
    # Check the output file
    from pathlib import Path
    output_file = Path("test_output.md")
    if output_file.exists():
        content = output_file.read_text()
        print(f"\n=== File Content ===")
        print(content)
        if "{{" in content:
            print("\nERROR: Templates not rendered!")
        else:
            print("\nSUCCESS: Templates rendered correctly!")
        # Clean up
        output_file.unlink()

if __name__ == "__main__":
    asyncio.run(main())