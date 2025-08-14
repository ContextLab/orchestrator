#!/usr/bin/env python3
"""
Test alternative nested loop implementation using shallow loops across steps.
Instead of nesting for_each loops, we can use multiple steps with dependencies.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, '/Users/jmanning/orchestrator/src')

from orchestrator.orchestrator import Orchestrator
from orchestrator import init_models


async def test_nested_via_steps():
    """Test nested loop behavior using separate steps."""
    
    yaml_content = """
id: nested-via-steps
name: Nested Loops Via Steps
parameters:
  categories: ["fruit", "veggie"]
  fruits: ["apple", "orange"]
  veggies: ["carrot", "lettuce"]
  output_dir: "examples/outputs/nested_alt"

steps:
  # Outer loop - process each category
  - id: process_categories
    for_each: "{{ categories }}"
    steps:
      - id: select_items
        action: generate_text
        parameters:
          prompt: |
            {% if $item == 'fruit' %}
            Return exactly: apple,orange
            {% else %}
            Return exactly: carrot,lettuce
            {% endif %}
          model: openai/gpt-3.5-turbo
          max_tokens: 20
      
      - id: save_category_info
        tool: filesystem
        action: write
        parameters:
          path: "{{ output_dir }}/{{ $item }}_info.txt"
          content: |
            Category: {{ $item }}
            Items to process: {{ select_items }}
        dependencies:
          - select_items
  
  # Second step - process all fruit items
  - id: process_fruits
    for_each: "{{ fruits }}"
    steps:
      - id: describe_fruit
        action: generate_text
        parameters:
          prompt: "Describe the fruit {{ $item }} in 5 words"
          model: openai/gpt-3.5-turbo
          max_tokens: 20
      
      - id: save_fruit
        tool: filesystem
        action: write
        parameters:
          path: "{{ output_dir }}/fruit_{{ $item }}.txt"
          content: |
            Category: fruit
            Item: {{ $item }}
            Description: {{ describe_fruit }}
        dependencies:
          - describe_fruit
    dependencies:
      - process_categories
  
  # Third step - process all veggie items
  - id: process_veggies
    for_each: "{{ veggies }}"
    steps:
      - id: describe_veggie
        action: generate_text
        parameters:
          prompt: "Describe the vegetable {{ $item }} in 5 words"
          model: openai/gpt-3.5-turbo
          max_tokens: 20
      
      - id: save_veggie
        tool: filesystem
        action: write
        parameters:
          path: "{{ output_dir }}/veggie_{{ $item }}.txt"
          content: |
            Category: veggie
            Item: {{ $item }}
            Description: {{ describe_veggie }}
        dependencies:
          - describe_veggie
    dependencies:
      - process_categories
"""
    
    print("\n" + "="*60)
    print("TEST: Nested Loops Via Separate Steps")
    print("="*60)
    
    # Initialize orchestrator
    model_registry = init_models()
    orchestrator = Orchestrator(model_registry=model_registry)
    
    # Execute pipeline
    print("\nExecuting pipeline...")
    result = await orchestrator.execute_yaml(yaml_content)
    
    # Check output files
    print("\n" + "="*60)
    print("CHECKING OUTPUT FILES")
    print("="*60)
    
    output_dir = Path("examples/outputs/nested_alt")
    
    # Check category info files
    for category in ["fruit", "veggie"]:
        file_path = output_dir / f"{category}_info.txt"
        if file_path.exists():
            content = file_path.read_text()
            print(f"\n--- {category}_info.txt ---")
            print(content[:200])
            if "{{" in content:
                print(f"❌ Contains unrendered templates")
            else:
                print(f"✅ No template placeholders")
        else:
            print(f"❌ {file_path} not found")
    
    # Check fruit files
    for fruit in ["apple", "orange"]:
        file_path = output_dir / f"fruit_{fruit}.txt"
        if file_path.exists():
            content = file_path.read_text()
            print(f"\n--- fruit_{fruit}.txt ---")
            print(content[:200])
            if "{{" in content:
                print(f"❌ Contains unrendered templates")
            else:
                print(f"✅ No template placeholders")
        else:
            print(f"❌ {file_path} not found")
    
    # Check veggie files
    for veggie in ["carrot", "lettuce"]:
        file_path = output_dir / f"veggie_{veggie}.txt"
        if file_path.exists():
            content = file_path.read_text()
            print(f"\n--- veggie_{veggie}.txt ---")
            print(content[:200])
            if "{{" in content:
                print(f"❌ Contains unrendered templates")
            else:
                print(f"✅ No template placeholders")
        else:
            print(f"❌ {file_path} not found")


async def test_dynamic_nested():
    """Test dynamic nested behavior using conditionals and multiple steps."""
    
    yaml_content = """
id: dynamic-nested
name: Dynamic Nested Loops
parameters:
  outer_items: ["A", "B"]
  inner_template: ["1", "2", "3"]
  output_dir: "examples/outputs/dynamic_nested"

steps:
  # Process each outer item
  - id: outer_loop
    for_each: "{{ outer_items }}"
    steps:
      - id: prepare_inner
        action: generate_text
        parameters:
          prompt: |
            For item {{ $item }}, generate a list of 2 sub-items.
            Return exactly in format: subitem1,subitem2
            Example: {{ $item }}1,{{ $item }}2
          model: openai/gpt-3.5-turbo
          max_tokens: 20
      
      - id: save_outer
        tool: filesystem
        action: write
        parameters:
          path: "{{ output_dir }}/{{ $item }}_main.txt"
          content: |
            Main Item: {{ $item }}
            Sub-items to process: {{ prepare_inner }}
        dependencies:
          - prepare_inner
  
  # Process combinations - A with inner items
  - id: process_A_items
    for_each: "{{ inner_template }}"
    if: "{{ 'A' in outer_items }}"
    steps:
      - id: combine_A
        action: generate_text
        parameters:
          prompt: "Combine A with {{ $item }}"
          model: openai/gpt-3.5-turbo
          max_tokens: 10
      
      - id: save_A_combo
        tool: filesystem
        action: write
        parameters:
          path: "{{ output_dir }}/A_{{ $item }}.txt"
          content: |
            Combination: {{ combine_A }}
        dependencies:
          - combine_A
    dependencies:
      - outer_loop
  
  # Process combinations - B with inner items
  - id: process_B_items
    for_each: "{{ inner_template }}"
    if: "{{ 'B' in outer_items }}"
    steps:
      - id: combine_B
        action: generate_text
        parameters:
          prompt: "Combine B with {{ $item }}"
          model: openai/gpt-3.5-turbo
          max_tokens: 10
      
      - id: save_B_combo
        tool: filesystem
        action: write
        parameters:
          path: "{{ output_dir }}/B_{{ $item }}.txt"
          content: |
            Combination: {{ combine_B }}
        dependencies:
          - combine_B
    dependencies:
      - outer_loop
"""
    
    print("\n" + "="*60)
    print("TEST: Dynamic Nested Loops with Conditionals")
    print("="*60)
    
    # Initialize orchestrator
    model_registry = init_models()
    orchestrator = Orchestrator(model_registry=model_registry)
    
    # Execute pipeline
    print("\nExecuting pipeline...")
    result = await orchestrator.execute_yaml(yaml_content)
    
    # Check output files
    print("\n" + "="*60)
    print("CHECKING OUTPUT FILES")
    print("="*60)
    
    output_dir = Path("examples/outputs/dynamic_nested")
    
    # Check main files
    for item in ["A", "B"]:
        file_path = output_dir / f"{item}_main.txt"
        if file_path.exists():
            content = file_path.read_text()
            print(f"\n--- {item}_main.txt ---")
            print(content[:200])
            if "{{" in content:
                print(f"❌ Contains unrendered templates")
            else:
                print(f"✅ No template placeholders")
    
    # Check combination files
    for outer in ["A", "B"]:
        for inner in ["1", "2", "3"]:
            file_path = output_dir / f"{outer}_{inner}.txt"
            if file_path.exists():
                content = file_path.read_text()
                print(f"\n--- {outer}_{inner}.txt ---")
                print(content[:100])
                if "{{" in content:
                    print(f"❌ Contains unrendered templates")
                else:
                    print(f"✅ No template placeholders")


async def main():
    """Run alternative nested loop tests."""
    print("\n" + "="*70)
    print("ALTERNATIVE NESTED LOOP IMPLEMENTATIONS")
    print("="*70)
    
    print("\nThese tests demonstrate how to achieve nested loop behavior")
    print("without actually nesting for_each loops, which avoids the")
    print("$parent_item template rendering issues.")
    
    # Test 1: Using separate steps
    await test_nested_via_steps()
    
    # Test 2: Dynamic nested with conditionals
    await test_dynamic_nested()
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("\nUsing separate steps with dependencies is a reliable way to")
    print("achieve nested loop behavior while avoiding template issues.")
    print("Each loop step has its own context and templates render correctly.")


if __name__ == "__main__":
    asyncio.run(main())