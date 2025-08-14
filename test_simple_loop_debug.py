#!/usr/bin/env python3
"""
Simple test to debug loop template rendering.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, '/Users/jmanning/orchestrator/src')

from orchestrator.orchestrator import Orchestrator
from orchestrator import init_models


async def test_simple_loop():
    """Test simple loop with template rendering."""
    
    yaml_content = """
id: test-simple-loop
name: Simple Loop Test
parameters:
  items: ["A", "B"]
  output_dir: "examples/outputs/simple_loop_debug"
steps:
  - id: process
    for_each: "{{ items }}"
    steps:
      - id: step1
        action: generate_text
        parameters:
          prompt: "Return: STEP1_{{ item }}_{{ index }}"
          model: openai/gpt-3.5-turbo
          max_tokens: 10
      
      - id: step2
        action: generate_text
        parameters:
          prompt: "Return: STEP2_{{ step1 }}"
          model: openai/gpt-3.5-turbo
          max_tokens: 10
        dependencies:
          - step1
      
      - id: save
        tool: filesystem
        action: write
        parameters:
          path: "{{ output_dir }}/{{ item }}.txt"
          content: |
            Item: {{ item }}
            Index: {{ index }}
            Step1 result: {{ step1 }}
            Step2 result: {{ step2 }}
        dependencies:
          - step2
"""
    
    print("="*60)
    print("SIMPLE LOOP DEBUG TEST")
    print("="*60)
    
    # Initialize orchestrator
    model_registry = init_models()
    orchestrator = Orchestrator(model_registry=model_registry)
    
    # Execute pipeline
    context = {
        "items": ["A", "B"],
        "output_dir": "examples/outputs/simple_loop_debug"
    }
    print(f"Executing with items: {context['items']}")
    
    result = await orchestrator.execute_yaml(yaml_content, context=context)
    
    # Check outputs
    output_dir = Path("examples/outputs/simple_loop_debug")
    
    print("\n" + "="*60)
    print("CHECKING OUTPUTS")
    print("="*60)
    
    for item in ["A", "B"]:
        file_path = output_dir / f"{item}.txt"
        if file_path.exists():
            content = file_path.read_text()
            print(f"\n--- {item}.txt ---")
            print(content)
            
            # Check for unrendered templates
            if "{{" in content:
                print(f"⚠️  WARNING: Unrendered templates found in {item}.txt!")
        else:
            print(f"❌ Missing file: {item}.txt")
    
    return result


if __name__ == "__main__":
    asyncio.run(test_simple_loop())