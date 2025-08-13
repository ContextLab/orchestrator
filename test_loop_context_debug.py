#!/usr/bin/env python3
"""
Debug test to check loop context mapping and template rendering.
"""

import asyncio
import sys
from pathlib import Path
import logging

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add the src directory to the path
sys.path.insert(0, '/Users/jmanning/orchestrator/src')

from orchestrator.orchestrator import Orchestrator
from orchestrator import init_models


async def test_simple_loop_templates():
    """Test the simplest possible loop with filesystem templates."""
    
    yaml_content = """
id: debug-loop-templates
name: Debug Loop Templates
parameters:
  items: ["test1", "test2"]
  output_dir: "examples/outputs/debug_loop"
steps:
  - id: process_items
    for_each: "{{ items }}"
    steps:
      - id: generate
        action: generate_text
        parameters:
          prompt: "Say hello to {{ $item }}"
          model: openai/gpt-3.5-turbo
          max_tokens: 20
      
      - id: save
        tool: filesystem
        action: write
        parameters:
          path: "{{ output_dir }}/{{ $item }}.txt"
          content: |
            Item: {{ $item }}
            Generated: {{ generate }}
        dependencies:
          - generate
"""
    
    print("\n" + "="*60)
    print("DEBUG TEST: Simple Loop with Templates")
    print("="*60)
    
    # Initialize orchestrator
    model_registry = init_models()
    orchestrator = Orchestrator(model_registry=model_registry)
    
    # Enable debug logging for key modules
    logging.getLogger("orchestrator.orchestrator").setLevel(logging.DEBUG)
    logging.getLogger("orchestrator.control_systems.hybrid_control_system").setLevel(logging.DEBUG)
    logging.getLogger("orchestrator.tools.system_tools").setLevel(logging.DEBUG)
    
    # Execute pipeline
    print("\nExecuting pipeline...")
    result = await orchestrator.execute_yaml(yaml_content)
    
    # Check output files
    print("\n" + "="*60)
    print("CHECKING OUTPUT FILES")
    print("="*60)
    
    output_dir = Path("examples/outputs/debug_loop")
    
    for item in ["test1", "test2"]:
        file_path = output_dir / f"{item}.txt"
        
        if file_path.exists():
            content = file_path.read_text()
            print(f"\n--- {item}.txt ---")
            print(content)
            
            # Check for templates
            if "{{" in content:
                print(f"❌ FAIL: {item}.txt contains unrendered templates!")
                # Show which templates weren't rendered
                import re
                templates = re.findall(r'{{.*?}}', content)
                print(f"   Unrendered templates: {templates}")
            else:
                print(f"✅ PASS: {item}.txt has no template placeholders")
                
            # Check for expected content
            if f"Item: {item}" in content:
                print(f"✅ PASS: Item name '{item}' is present")
            else:
                print(f"❌ FAIL: Item name '{item}' is missing")
                
            if "Generated:" in content and "{{ generate }}" not in content:
                print(f"✅ PASS: Generated content is rendered")
            else:
                print(f"❌ FAIL: Generated content not rendered properly")
        else:
            print(f"\n❌ FAIL: {file_path} does not exist!")
    
    print("\n" + "="*60)
    print("END OF DEBUG TEST")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(test_simple_loop_templates())