#!/usr/bin/env python3
"""
Test to document current loop variable behavior in the orchestrator.
This will provide evidence of how the system currently handles loop variables.
"""

import asyncio
import sys
import json
from pathlib import Path

sys.path.insert(0, '/Users/jmanning/orchestrator/src')

from orchestrator.orchestrator import Orchestrator
from orchestrator import init_models


async def test_loop_behavior():
    """Document current loop variable behavior."""
    
    # Test 1: Simple for_each loop with filesystem write
    yaml_simple = """
id: test-loop-vars
name: Loop Variable Test
parameters:
  items: ["alpha", "beta"]
  output_dir: "examples/outputs/behavior_test"
steps:
  - id: process
    for_each: "{{ items }}"
    steps:
      - id: generate
        action: generate_text
        parameters:
          prompt: "Say: Processing {{ item }} at index {{ index }}"
          model: openai/gpt-3.5-turbo
          max_tokens: 20
      
      - id: save
        tool: filesystem
        action: write
        parameters:
          path: "{{ output_dir }}/result_{{ item }}_{{ index }}.txt"
          content: |
            Item: {{ item }}
            Index: {{ index }}
            Generated: {{ generate }}
        dependencies:
          - generate
"""
    
    print("="*80)
    print("CURRENT BEHAVIOR TEST - Simple Loop with Filesystem Write")
    print("="*80)
    
    # Initialize orchestrator
    model_registry = init_models()
    orchestrator = Orchestrator(model_registry=model_registry)
    
    # Execute pipeline
    context = {
        "items": ["alpha", "beta"],
        "output_dir": "examples/outputs/behavior_test"
    }
    
    print(f"\nContext provided: {json.dumps(context, indent=2)}")
    
    try:
        result = await orchestrator.execute_yaml(yaml_simple, context=context)
        print(f"\nExecution completed successfully")
    except Exception as e:
        print(f"\nExecution failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    # Check outputs
    output_dir = Path("examples/outputs/behavior_test")
    print(f"\n--- Checking outputs in {output_dir} ---")
    
    if output_dir.exists():
        files = list(output_dir.glob("*.txt"))
        print(f"Files created: {[f.name for f in files]}")
        
        for f in sorted(files):
            print(f"\n--- Content of {f.name} ---")
            content = f.read_text()
            print(content)
            
            # Check for template issues
            if "{{" in content:
                print(f"⚠️  UNRENDERED TEMPLATES in {f.name}")
            
            # Check filename for template issues
            if "{{" in f.name:
                print(f"⚠️  UNRENDERED TEMPLATES in filename: {f.name}")
    else:
        print(f"Output directory does not exist!")
    
    return result


async def test_runtime_expansion():
    """Test runtime ForEachTask expansion with AUTO tags."""
    
    yaml_runtime = """
id: test-runtime-expansion
name: Runtime Expansion Test
parameters:
  base_items: ["one", "two"]
  output_dir: "examples/outputs/runtime_test"
steps:
  - id: generate_list
    action: generate_text
    parameters:
      prompt: "Return exactly: ['item_one', 'item_two']"
      model: openai/gpt-3.5-turbo
      max_tokens: 30
  
  - id: process_runtime
    for_each: "AUTO[generate_list]"
    steps:
      - id: save_item
        tool: filesystem
        action: write
        parameters:
          path: "{{ output_dir }}/runtime_{{ item }}.txt"
          content: "Runtime item: {{ item }}"
"""
    
    print("\n" + "="*80)
    print("CURRENT BEHAVIOR TEST - Runtime ForEach Expansion")
    print("="*80)
    
    model_registry = init_models()
    orchestrator = Orchestrator(model_registry=model_registry)
    
    context = {
        "base_items": ["one", "two"],
        "output_dir": "examples/outputs/runtime_test"
    }
    
    print(f"\nContext provided: {json.dumps(context, indent=2)}")
    
    try:
        result = await orchestrator.execute_yaml(yaml_runtime, context=context)
        print(f"\nExecution completed successfully")
    except Exception as e:
        print(f"\nExecution failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    # Check outputs
    output_dir = Path("examples/outputs/runtime_test")
    print(f"\n--- Checking runtime outputs in {output_dir} ---")
    
    if output_dir.exists():
        files = list(output_dir.glob("*.txt"))
        print(f"Files created: {[f.name for f in files]}")
        
        for f in sorted(files):
            print(f"\n--- Content of {f.name} ---")
            content = f.read_text()
            print(content)
            
            if "{{" in content or "{{" in f.name:
                print(f"⚠️  UNRENDERED TEMPLATES detected")
    else:
        print(f"Output directory does not exist!")


async def test_nested_context():
    """Test how nested contexts and dependencies work."""
    
    yaml_nested = """
id: test-nested-context
name: Nested Context Test
parameters:
  categories: ["A", "B"]
  output_dir: "examples/outputs/nested_test"
steps:
  - id: outer_loop
    for_each: "{{ categories }}"
    steps:
      - id: gen_prefix
        action: generate_text
        parameters:
          prompt: "Return exactly: PREFIX_{{ item }}"
          model: openai/gpt-3.5-turbo
          max_tokens: 10
      
      - id: gen_suffix
        action: generate_text
        parameters:
          prompt: "Return exactly: SUFFIX_{{ index }}"
          model: openai/gpt-3.5-turbo
          max_tokens: 10
        dependencies:
          - gen_prefix
      
      - id: combine
        tool: filesystem
        action: write
        parameters:
          path: "{{ output_dir }}/combined_{{ item }}.txt"
          content: |
            Category: {{ item }}
            Index: {{ index }}
            Prefix: {{ gen_prefix }}
            Suffix: {{ gen_suffix }}
        dependencies:
          - gen_suffix
"""
    
    print("\n" + "="*80)
    print("CURRENT BEHAVIOR TEST - Nested Context and Dependencies")
    print("="*80)
    
    model_registry = init_models()
    orchestrator = Orchestrator(model_registry=model_registry)
    
    context = {
        "categories": ["A", "B"],
        "output_dir": "examples/outputs/nested_test"
    }
    
    print(f"\nContext provided: {json.dumps(context, indent=2)}")
    
    try:
        result = await orchestrator.execute_yaml(yaml_nested, context=context)
        print(f"\nExecution completed successfully")
    except Exception as e:
        print(f"\nExecution failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    # Check outputs
    output_dir = Path("examples/outputs/nested_test")
    print(f"\n--- Checking nested outputs in {output_dir} ---")
    
    if output_dir.exists():
        files = list(output_dir.glob("*.txt"))
        print(f"Files created: {[f.name for f in files]}")
        
        for f in sorted(files):
            print(f"\n--- Content of {f.name} ---")
            content = f.read_text()
            print(content)
            
            # Analyze what worked and what didn't
            lines = content.split('\n')
            for line in lines:
                if '{{' in line:
                    print(f"  ❌ Unrendered: {line}")
                elif ':' in line and line.split(':')[1].strip():
                    print(f"  ✅ Rendered: {line}")


if __name__ == "__main__":
    print("DOCUMENTING CURRENT ORCHESTRATOR LOOP BEHAVIOR")
    print("=" * 80)
    
    # Clean up previous test outputs
    import shutil
    for dir_name in ["behavior_test", "runtime_test", "nested_test"]:
        dir_path = Path(f"examples/outputs/{dir_name}")
        if dir_path.exists():
            shutil.rmtree(dir_path)
    
    # Run tests
    asyncio.run(test_loop_behavior())
    asyncio.run(test_runtime_expansion())
    asyncio.run(test_nested_context())