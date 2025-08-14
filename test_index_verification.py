#!/usr/bin/env python3
"""
Test to verify loop indices are correct and check for off-by-one errors.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, '/Users/jmanning/orchestrator/src')

from orchestrator.orchestrator import Orchestrator
from orchestrator import init_models


async def test_index_accuracy():
    """Test that loop indices and items match correctly."""
    
    yaml_content = """
id: test-indices
name: Index Verification Test
parameters:
  items: ["zero", "one", "two", "three", "four"]
  output_dir: "examples/outputs/index_test"
steps:
  - id: process_items
    for_each: "{{ items }}"
    steps:
      - id: generate
        action: generate_text
        parameters:
          prompt: |
            Current item: {{ $item }}
            Current index: {{ $index }}
            Return exactly: "Item {{ $item }} at index {{ $index }}"
          model: openai/gpt-3.5-turbo
          max_tokens: 30
      
      - id: save
        tool: filesystem
        action: write
        parameters:
          path: "{{ output_dir }}/item_{{ $index }}_{{ $item }}.txt"
          content: |
            Expected Index: {{ $index }}
            Expected Item: {{ $item }}
            Generated Result: {{ generate }}
            
            Verification:
            - File name: item_{{ $index }}_{{ $item }}.txt
            - Should be: item {{ $item }} at position {{ $index }}
        dependencies:
          - generate
"""
    
    print("\n" + "="*60)
    print("INDEX VERIFICATION TEST")
    print("="*60)
    
    # Initialize orchestrator
    model_registry = init_models()
    orchestrator = Orchestrator(model_registry=model_registry)
    
    # Execute pipeline
    print("\nExecuting pipeline with items: ['zero', 'one', 'two', 'three', 'four']")
    result = await orchestrator.execute_yaml(yaml_content)
    
    # Check output files
    print("\n" + "="*60)
    print("VERIFYING INDICES")
    print("="*60)
    
    output_dir = Path("examples/outputs/index_test")
    expected_mapping = {
        0: "zero",
        1: "one", 
        2: "two",
        3: "three",
        4: "four"
    }
    
    errors = []
    
    for expected_index, expected_item in expected_mapping.items():
        # Check if file exists with correct name
        file_path = output_dir / f"item_{expected_index}_{expected_item}.txt"
        
        print(f"\n--- Checking index {expected_index} ---")
        print(f"Expected: item_{expected_index}_{expected_item}.txt")
        
        if file_path.exists():
            content = file_path.read_text()
            
            # Check content
            lines = content.split('\n')
            
            # Extract actual values from content
            index_line = [l for l in lines if l.startswith("Expected Index:")]
            item_line = [l for l in lines if l.startswith("Expected Item:")]
            generated_line = [l for l in lines if l.startswith("Generated Result:")]
            
            if index_line:
                actual_index = index_line[0].split(":")[1].strip()
                if str(actual_index) != str(expected_index):
                    errors.append(f"Index mismatch in {expected_item}: expected {expected_index}, got {actual_index}")
                    print(f"❌ Index mismatch: expected {expected_index}, got {actual_index}")
                else:
                    print(f"✅ Index correct: {expected_index}")
            
            if item_line:
                actual_item = item_line[0].split(":")[1].strip()
                if actual_item != expected_item:
                    errors.append(f"Item mismatch at index {expected_index}: expected {expected_item}, got {actual_item}")
                    print(f"❌ Item mismatch: expected {expected_item}, got {actual_item}")
                else:
                    print(f"✅ Item correct: {expected_item}")
            
            if generated_line:
                generated = generated_line[0].split(":", 1)[1].strip()
                print(f"   Generated: {generated}")
                
                # Check if generated contains correct values
                if str(expected_index) not in generated:
                    errors.append(f"Generated text missing index {expected_index}")
                if expected_item not in generated:
                    errors.append(f"Generated text missing item {expected_item}")
        else:
            errors.append(f"Missing file: item_{expected_index}_{expected_item}.txt")
            print(f"❌ File not found!")
            
            # Check if there's a file with wrong index
            for alt_file in output_dir.glob(f"*_{expected_item}.txt"):
                print(f"   Found alternative: {alt_file.name}")
    
    # Also check for any unexpected files
    print("\n--- Checking for unexpected files ---")
    all_files = list(output_dir.glob("*.txt"))
    for file_path in all_files:
        file_name = file_path.name
        # Check if this file matches expected pattern
        expected = False
        for idx, item in expected_mapping.items():
            if file_name == f"item_{idx}_{item}.txt":
                expected = True
                break
        if not expected:
            errors.append(f"Unexpected file: {file_name}")
            print(f"❌ Unexpected file: {file_name}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if errors:
        print(f"❌ Found {len(errors)} errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("✅ All indices and items match correctly!")
        print("No off-by-one errors detected.")
        return True


async def main():
    """Run index verification test."""
    print("\n" + "="*70)
    print("LOOP INDEX VERIFICATION")
    print("="*70)
    
    print("\nThis test verifies that loop indices ($index) and items ($item)")
    print("are correctly mapped and there are no off-by-one errors.")
    
    success = await test_index_accuracy()
    
    if not success:
        print("\n⚠️  WARNING: Index errors detected!")
        print("This could indicate an off-by-one error in the loop implementation.")
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)