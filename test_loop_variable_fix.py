#!/usr/bin/env python3
"""Test that loop variables are properly resolved after fix."""

import asyncio
import sys
from pathlib import Path
import shutil

sys.path.insert(0, '/Users/jmanning/orchestrator/src')

from orchestrator.orchestrator import Orchestrator
from orchestrator import init_models


async def test_loop_variable_resolution():
    """Test that loop variables are properly rendered in filesystem operations."""
    
    yaml_content = """
id: test-loop-fix
name: Loop Variable Fix Test
parameters:
  items: ["apple", "banana"]
  output_dir: "test_outputs/loop_fix"
steps:
  - id: process
    for_each: "{{ items }}"
    steps:
      - id: write
        tool: filesystem
        action: write
        parameters:
          path: "{{ output_dir }}/{{ item }}_{{ index }}.txt"
          content: "Processing {{ item }} at index {{ index }}"
"""
    
    print("=" * 80)
    print("LOOP VARIABLE FIX VERIFICATION TEST")
    print("=" * 80)
    
    model_registry = init_models()
    orchestrator = Orchestrator(model_registry=model_registry)
    
    context = {
        "items": ["apple", "banana"],
        "output_dir": "test_outputs/loop_fix"
    }
    
    # Clean output directory
    output_dir = Path("test_outputs/loop_fix")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    print(f"\nExecuting pipeline with items: {context['items']}")
    
    # Execute pipeline
    try:
        result = await orchestrator.execute_yaml(yaml_content, context=context)
        print("Pipeline executed successfully")
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Verify outputs
    print("\n" + "-" * 40)
    print("VERIFICATION RESULTS")
    print("-" * 40)
    
    expected_files = [
        ("apple_0.txt", "Processing apple at index 0"),
        ("banana_1.txt", "Processing banana at index 1")
    ]
    
    all_passed = True
    
    for filename, expected_content in expected_files:
        filepath = output_dir / filename
        
        if not filepath.exists():
            print(f"❌ File {filename} not created")
            # Check if wrong filename was created
            wrong_path = output_dir / f"{{{{item}}}}_{{{{index}}}}.txt"
            if wrong_path.exists():
                print(f"   Found wrong file: {wrong_path.name}")
            all_passed = False
            continue
        
        actual_content = filepath.read_text().strip()
        
        if actual_content == expected_content:
            print(f"✅ {filename}: Content correct")
            print(f"   Content: '{actual_content}'")
        else:
            print(f"❌ {filename}: Content mismatch")
            print(f"   Expected: '{expected_content}'")
            print(f"   Actual:   '{actual_content}'")
            all_passed = False
    
    # Also check for any unexpected files
    all_files = list(output_dir.glob("*.txt")) if output_dir.exists() else []
    for f in all_files:
        if f.name not in [ef[0] for ef in expected_files]:
            print(f"⚠️  Unexpected file: {f.name}")
            print(f"   Content: '{f.read_text().strip()}'")
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED! Loop variables are working correctly!")
    else:
        print("❌ TESTS FAILED! Loop variables are still not rendering properly.")
    print("=" * 80)
    
    return all_passed


async def test_nested_loops():
    """Test nested loops with variable access."""
    
    yaml_content = """
id: test-nested
name: Nested Loop Test
parameters:
  categories: ["fruit", "veggie"]
  output_dir: "test_outputs/nested_fix"
steps:
  - id: outer
    for_each: "{{ categories }}"
    steps:
      - id: inner
        for_each: "[1, 2]"
        steps:
          - id: write
            tool: filesystem
            action: write
            parameters:
              path: "{{ output_dir }}/{{ $outer.item }}_{{ item }}.txt"
              content: "Outer: {{ $outer.item }}, Inner: {{ item }}"
"""
    
    print("\n" + "=" * 80)
    print("NESTED LOOP TEST")
    print("=" * 80)
    
    model_registry = init_models()
    orchestrator = Orchestrator(model_registry=model_registry)
    
    context = {
        "categories": ["fruit", "veggie"],
        "output_dir": "test_outputs/nested_fix"
    }
    
    # Clean output directory
    output_dir = Path("test_outputs/nested_fix")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    print(f"\nExecuting nested loop pipeline")
    
    # Execute pipeline
    try:
        result = await orchestrator.execute_yaml(yaml_content, context=context)
        print("Nested pipeline executed successfully")
    except Exception as e:
        print(f"Nested pipeline execution failed: {e}")
        # This might fail due to $outer.item syntax not being supported yet
        print("Note: Nested loop variable access may require additional implementation")
        return False
    
    # Check outputs
    if output_dir.exists():
        files = list(output_dir.glob("*.txt"))
        print(f"\nFiles created: {[f.name for f in files]}")
        for f in files:
            print(f"  {f.name}: {f.read_text().strip()}")
    
    return True


async def test_loop_with_dependencies():
    """Test loop with dependencies between steps."""
    
    yaml_content = """
id: test-deps
name: Loop Dependencies Test
parameters:
  items: ["X", "Y"]
  output_dir: "test_outputs/deps_fix"
steps:
  - id: process
    for_each: "{{ items }}"
    steps:
      - id: generate
        action: generate_text
        parameters:
          prompt: "Say exactly: PROCESSED_{{ item }}"
          model: openai/gpt-3.5-turbo
          max_tokens: 20
      
      - id: save
        tool: filesystem
        action: write
        parameters:
          path: "{{ output_dir }}/{{ item }}_result.txt"
          content: |
            Item: {{ item }}
            Index: {{ index }}
            Generated: {{ generate }}
        dependencies:
          - generate
"""
    
    print("\n" + "=" * 80)
    print("LOOP WITH DEPENDENCIES TEST")
    print("=" * 80)
    
    model_registry = init_models()
    orchestrator = Orchestrator(model_registry=model_registry)
    
    context = {
        "items": ["X", "Y"],
        "output_dir": "test_outputs/deps_fix"
    }
    
    # Clean output directory
    output_dir = Path("test_outputs/deps_fix")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    print(f"\nExecuting loop with dependencies")
    
    # Execute pipeline
    try:
        result = await orchestrator.execute_yaml(yaml_content, context=context)
        print("Dependencies pipeline executed successfully")
    except Exception as e:
        print(f"Dependencies pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check outputs
    print("\nChecking outputs with dependencies:")
    all_passed = True
    
    for item in ["X", "Y"]:
        filepath = output_dir / f"{item}_result.txt"
        if filepath.exists():
            content = filepath.read_text()
            print(f"\n{filepath.name}:")
            print(content)
            
            # Check for unrendered templates
            if "{{" in content:
                print(f"  ❌ Unrendered templates found")
                all_passed = False
            else:
                print(f"  ✅ All templates rendered")
                
            # Check that item variable was rendered
            if f"Item: {item}" in content:
                print(f"  ✅ Item variable rendered correctly")
            else:
                print(f"  ❌ Item variable not rendered")
                all_passed = False
        else:
            print(f"❌ File {filepath.name} not created")
            all_passed = False
    
    return all_passed


if __name__ == "__main__":
    print("TESTING LOOP VARIABLE FIX")
    print("=" * 80)
    
    # Run tests
    loop_result = asyncio.run(test_loop_variable_resolution())
    
    # Try nested loops (may not work yet)
    # nested_result = asyncio.run(test_nested_loops())
    
    # Test with dependencies
    deps_result = asyncio.run(test_loop_with_dependencies())
    
    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Simple loop test: {'✅ PASSED' if loop_result else '❌ FAILED'}")
    print(f"Dependencies test: {'✅ PASSED' if deps_result else '❌ FAILED'}")
    
    if loop_result and deps_result:
        print("\n🎉 SUCCESS! The loop variable fix is working!")
    else:
        print("\n⚠️  Some tests failed. Additional work may be needed.")