#!/usr/bin/env python3
"""
Summary test for Issue #208 - Verify template rendering in loop contexts.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, '/Users/jmanning/orchestrator/src')

from orchestrator.orchestrator import Orchestrator
from orchestrator import init_models


async def run_test(name: str, yaml_content: str, check_files: list):
    """Run a test and check output files."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print('='*60)
    
    # Initialize orchestrator
    model_registry = init_models()
    orchestrator = Orchestrator(model_registry=model_registry)
    
    try:
        # Execute pipeline
        result = await orchestrator.execute_yaml(yaml_content)
        print("✅ Pipeline executed successfully")
        
        # Check output files
        all_good = True
        for file_path in check_files:
            path = Path(file_path)
            if path.exists():
                content = path.read_text()
                if "{{" in content or "{%" in content:
                    print(f"❌ {file_path}: Contains unrendered templates")
                    all_good = False
                else:
                    print(f"✅ {file_path}: No template placeholders")
            else:
                print(f"⚠️  {file_path}: File not created")
                all_good = False
        
        return all_good
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return False


async def main():
    """Run all summary tests."""
    print("\n" + "="*70)
    print("ISSUE #208 SUMMARY TEST - Template Rendering in Loops")
    print("="*70)
    
    results = []
    
    # Test 1: Simple for_each loop
    test1 = """
id: test-simple-loop
name: Simple Loop Test
parameters:
  items: ["A", "B"]
steps:
  - id: process
    for_each: "{{ items }}"
    steps:
      - id: gen
        action: generate_text
        parameters:
          prompt: "Say {{ $item }}"
          model: openai/gpt-3.5-turbo
          max_tokens: 10
      - id: save
        tool: filesystem
        action: write
        parameters:
          path: "examples/outputs/summary_test/{{ $item }}.txt"
          content: "Result: {{ gen }}"
        dependencies: [gen]
"""
    
    result1 = await run_test(
        "Simple For-Each Loop",
        test1,
        ["examples/outputs/summary_test/A.txt", "examples/outputs/summary_test/B.txt"]
    )
    results.append(("Simple Loop", result1))
    
    # Test 2: Nested loops (simplified)
    test2 = """
id: test-nested
name: Nested Loop Test
parameters:
  outer: ["X"]
  inner: ["1", "2"]
steps:
  - id: outer_loop
    for_each: "{{ outer }}"
    steps:
      - id: inner_loop
        for_each: "{{ inner }}"
        steps:
          - id: process
            action: generate_text
            parameters:
              prompt: "Combine {{ $parent_item }} and {{ $item }}"
              model: openai/gpt-3.5-turbo
              max_tokens: 10
          - id: save
            tool: filesystem
            action: write
            parameters:
              path: "examples/outputs/summary_test/{{ $parent_item }}_{{ $item }}.txt"
              content: "Result: {{ process }}"
            dependencies: [process]
"""
    
    # Note: Nested loops might have issues with $parent_item
    result2 = await run_test(
        "Nested Loops",
        test2,
        ["examples/outputs/summary_test/X_1.txt", "examples/outputs/summary_test/X_2.txt"]
    )
    results.append(("Nested Loops", result2))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_pass = all(r[1] for r in results)
    if all_pass:
        print("\n🎉 All tests passed! Template rendering is working correctly.")
    else:
        print("\n⚠️  Some tests failed. See details above.")
    
    return all_pass


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)