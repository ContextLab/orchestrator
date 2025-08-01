#!/usr/bin/env python3
"""Test what's available in template context."""

import asyncio
import logging
import sys
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

async def main():
    from src.orchestrator import Orchestrator
    from scripts.run_pipeline import init_models
    
    # Initialize models first
    await init_models()
    
    # Create a test pipeline that captures context
    test_yaml = """
name: Test Template Context
description: Test what's available in template context

parameters:
  topic: "test"

steps:
  - id: generate_data
    action: generate_text
    parameters:
      prompt: "Generate a test result"
      max_tokens: 50
      
  - id: search_data
    tool: web-search
    action: search
    parameters:
      query: "test query"
      max_results: 3
      
  - id: capture_context
    tool: filesystem  
    action: write
    parameters:
      path: "test_context.json"
      content: |
        {
          "topic": "{{ topic }}",
          "generate_data_type": "{{ generate_data.__class__.__name__ if generate_data else 'Not found' }}",
          "generate_data_keys": "{{ generate_data.keys() if generate_data and generate_data.keys else 'No keys' }}",
          "generate_data_result": "{{ generate_data.result if generate_data and generate_data.result else 'No result' }}",
          "search_data_type": "{{ search_data.__class__.__name__ if search_data else 'Not found' }}",
          "search_data_keys": "{{ search_data.keys() if search_data and search_data.keys else 'No keys' }}",
          "search_data_total": "{{ search_data.total_results if search_data and search_data.total_results else 'No total' }}",
          "execution_timestamp": "{{ execution.timestamp }}"
        }
    dependencies:
      - generate_data
      - search_data
"""
    
    # Initialize orchestrator
    orchestrator = Orchestrator()
    
    # Run the test pipeline
    result = await orchestrator.execute_yaml(
        test_yaml,
        context={"topic": "test context"}
    )
    
    print("\n=== Pipeline Result ===")
    if 'steps' in result:
        for step_id, step_result in result['steps'].items():
            print(f"\n{step_id}:")
            if isinstance(step_result, dict):
                for k, v in step_result.items():
                    if k != 'results':  # Skip large results
                        print(f"  {k}: {v}")
            else:
                print(f"  {step_result}")
    
    # Check the output file
    from pathlib import Path
    output_file = Path("test_context.json")
    if output_file.exists():
        content = output_file.read_text()
        print(f"\n=== Captured Context ===")
        print(content)
        try:
            context_data = json.loads(content)
            print("\n=== Parsed Context ===")
            for k, v in context_data.items():
                print(f"{k}: {v}")
        except:
            pass
        # Clean up
        output_file.unlink()

if __name__ == "__main__":
    asyncio.run(main())