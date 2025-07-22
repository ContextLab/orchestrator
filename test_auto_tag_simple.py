#!/usr/bin/env python3
"""Simple test of AUTO tag functionality."""

import asyncio
import os
import tempfile
from pathlib import Path

# Add orchestrator to path
import sys
sys.path.insert(0, 'src')

from orchestrator import Orchestrator


async def test_simple_auto_tag():
    """Test a simple AUTO tag example."""
    
    # Create a simple pipeline with AUTO tag
    pipeline_yaml = """
name: simple-auto-test
description: Test basic AUTO tag functionality
version: "1.0.0"

steps:
  - id: make_decision
    action: llm-generate
    parameters:
      prompt: "We need to generate a report."
      format: <AUTO>Choose the best format for a technical report: 'markdown', 'html', or 'pdf'. Answer with just one word.</AUTO>
      
  - id: show_result
    action: report-generator
    parameters:
      title: "Test Report"
      format: "{{ make_decision.format }}"
      content: |
        # AUTO Tag Test
        
        The AUTO tag resolved to: {{ make_decision.format }}
        
        This demonstrates that AUTO tags are working correctly.
"""
    
    # Save pipeline
    pipeline_file = Path(tempfile.mktemp(suffix='.yaml'))
    pipeline_file.write_text(pipeline_yaml)
    
    try:
        # Create orchestrator
        orchestrator = Orchestrator()
        
        # Execute pipeline
        print("Executing pipeline with AUTO tag...")
        result = await orchestrator.execute_yaml(str(pipeline_file))
        
        print("\n✅ Pipeline executed successfully!")
        print(f"AUTO tag resolved to: {result.get('make_decision', {}).get('format', 'N/A')}")
        
        # Show report output
        if 'show_result' in result:
            print(f"\nReport generated with format: {result['make_decision']['format']}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if pipeline_file.exists():
            pipeline_file.unlink()


async def test_auto_tag_with_validation():
    """Test AUTO tag with validation tool."""
    
    # Create test data
    test_data = {
        "name": "Test Product",
        "price": 99.99,
        "category": "Electronics"
    }
    
    data_file = Path(tempfile.mktemp(suffix='.json'))
    import json
    data_file.write_text(json.dumps(test_data, indent=2))
    
    pipeline_yaml = f"""
name: auto-validation-test
description: Test AUTO tag with validation
version: "1.0.0"

steps:
  - id: read_data
    action: filesystem
    tool_config:
      action: read
    parameters:
      path: "{data_file}"
      
  - id: analyze_data
    action: llm-generate
    parameters:
      prompt: |
        Analyze this product data:
        {{{{ read_data.result }}}}
      validation_needed: <AUTO>Looking at this product data, should we validate the price field? Answer only 'yes' or 'no'.</AUTO>
      
  - id: validate_if_needed
    action: validation
    condition: "{{{{ analyze_data.validation_needed == 'yes' }}}}"
    parameters:
      data: "{{{{ read_data.result }}}}"
      schema:
        type: object
        properties:
          price:
            type: number
            minimum: 0
            maximum: <AUTO>For an electronics product, what's a reasonable maximum price? Answer with just a number like 10000.</AUTO>
"""
    
    # Save pipeline
    pipeline_file = Path(tempfile.mktemp(suffix='.yaml'))
    pipeline_file.write_text(pipeline_yaml)
    
    try:
        # Create orchestrator
        orchestrator = Orchestrator()
        
        # Execute pipeline
        print("Testing AUTO tag with validation...")
        result = await orchestrator.execute_yaml(str(pipeline_file))
        
        print("\n✅ Pipeline executed successfully!")
        print(f"Validation needed decision: {result.get('analyze_data', {}).get('validation_needed', 'N/A')}")
        
        if 'validate_if_needed' in result:
            print("Validation was performed")
            print(f"Validation result: {result['validate_if_needed'].get('is_valid', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if pipeline_file.exists():
            pipeline_file.unlink()
        if data_file.exists():
            data_file.unlink()


async def main():
    """Run simple AUTO tag tests."""
    print("🧪 SIMPLE AUTO TAG TESTS")
    print("="*50)
    
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("\n⚠️  No API keys found!")
        print("Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        return
    
    # Run tests
    tests = [
        ("Basic AUTO Tag", test_simple_auto_tag),
        ("AUTO Tag with Validation", test_auto_tag_with_validation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print("="*50)
        
        success = await test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    passed = sum(1 for _, s in results if s)
    print(f"\nTotal: {passed}/{len(results)} tests passed")


if __name__ == "__main__":
    asyncio.run(main())