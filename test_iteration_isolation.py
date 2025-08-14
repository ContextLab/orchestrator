#!/usr/bin/env python3
"""
Test to verify that each iteration properly isolates its results.
This checks for any leakage between iterations.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, '/Users/jmanning/orchestrator/src')

from orchestrator.orchestrator import Orchestrator
from orchestrator import init_models


async def test_iteration_isolation():
    """Test that each iteration has isolated results."""
    
    yaml_content = """
id: test-isolation
name: Iteration Isolation Test
parameters:
  items: ["first", "second", "third"]
  output_dir: "examples/outputs/isolation_test"
steps:
  - id: process_items
    for_each: "{{ items }}"
    steps:
      - id: generate_unique
        action: generate_text
        parameters:
          prompt: |
            Generate a unique identifier for item "{{ item }}".
            Return exactly: "UNIQUE_{{ item | upper }}_{{ index }}"
          model: openai/gpt-3.5-turbo
          max_tokens: 20
      
      - id: process_data
        action: generate_text
        parameters:
          prompt: |
            Process this data:
            Item: {{ item }}
            Index: {{ index }}
            Previous step result: {{ generate_unique }}
            
            Return: "Processed {{ item }} with {{ generate_unique }}"
          model: openai/gpt-3.5-turbo
          max_tokens: 50
        dependencies:
          - generate_unique
      
      - id: save_results
        tool: filesystem
        action: write
        parameters:
          path: "{{ output_dir }}/{{ item }}_results.txt"
          content: |
            === Iteration {{ index }} ===
            Item: {{ item }}
            
            Step 1 (generate_unique): {{ generate_unique }}
            Step 2 (process_data): {{ process_data }}
            
            Validation Checks:
            - Should contain "{{ item }}" in both results
            - generate_unique should be: UNIQUE_{{ item | upper }}_{{ index }}
            - process_data should reference {{ item }} and its unique ID
        dependencies:
          - process_data
"""
    
    print("\n" + "="*60)
    print("ITERATION ISOLATION TEST")
    print("="*60)
    
    # Initialize orchestrator
    model_registry = init_models()
    orchestrator = Orchestrator(model_registry=model_registry)
    
    # Execute pipeline
    print("\nExecuting pipeline with items: ['first', 'second', 'third']")
    result = await orchestrator.execute_yaml(yaml_content)
    
    # Check output files
    print("\n" + "="*60)
    print("CHECKING ITERATION ISOLATION")
    print("="*60)
    
    output_dir = Path("examples/outputs/isolation_test")
    items = ["first", "second", "third"]
    
    errors = []
    
    for idx, item in enumerate(items):
        file_path = output_dir / f"{item}_results.txt"
        
        print(f"\n--- Checking {item} (index {idx}) ---")
        
        if file_path.exists():
            content = file_path.read_text()
            
            # Check that content contains correct item references
            expected_unique = f"UNIQUE_{item.upper()}_{idx}"
            
            print(f"Expected unique ID: {expected_unique}")
            
            # Check for the unique ID
            if expected_unique in content:
                print(f"✅ Correct unique ID found")
            else:
                print(f"❌ Unique ID not found or incorrect")
                errors.append(f"{item}: Missing or wrong unique ID")
            
            # Check for cross-contamination from other iterations
            for other_idx, other_item in enumerate(items):
                if other_item != item:
                    other_unique = f"UNIQUE_{other_item.upper()}_{other_idx}"
                    if other_unique in content:
                        print(f"❌ CONTAMINATION: Found {other_item}'s unique ID in {item}'s file!")
                        errors.append(f"{item}: Contains data from {other_item} iteration")
                    
                    # Check for other item names where they shouldn't be
                    lines = content.split('\n')
                    for line in lines:
                        if line.startswith("Step 1") or line.startswith("Step 2"):
                            if other_item in line and other_item not in ["first", "second", "third"]:
                                print(f"❌ CONTAMINATION: {other_item} found in {item}'s results")
                                errors.append(f"{item}: References {other_item} in results")
            
            # Show actual content of key lines
            lines = content.split('\n')
            for line in lines:
                if line.startswith("Step 1"):
                    print(f"  {line}")
                elif line.startswith("Step 2"):
                    print(f"  {line}")
        else:
            print(f"❌ File not found!")
            errors.append(f"Missing file: {item}_results.txt")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if errors:
        print(f"❌ Found {len(errors)} isolation errors:")
        for error in errors:
            print(f"  - {error}")
        print("\n⚠️  CRITICAL: Iterations are not properly isolated!")
        return False
    else:
        print("✅ All iterations are properly isolated!")
        print("Each iteration only sees its own results.")
        return True


async def main():
    """Run iteration isolation test."""
    print("\n" + "="*70)
    print("ITERATION ISOLATION VERIFICATION")
    print("="*70)
    
    print("\nThis test verifies that each loop iteration is properly isolated")
    print("and doesn't see results from other iterations.")
    
    success = await test_iteration_isolation()
    
    if not success:
        print("\n⚠️  WARNING: Iteration isolation issues detected!")
        print("This is a critical bug that needs to be fixed.")
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)