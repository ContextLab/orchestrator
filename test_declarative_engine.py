#!/usr/bin/env python3
"""Test the new declarative pipeline engine."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from orchestrator.engine import DeclarativePipelineEngine


async def test_simple_pipeline():
    """Test a simple declarative pipeline."""
    
    # Simple pipeline YAML
    pipeline_yaml = """
name: "Simple Test Pipeline"
description: "Test the declarative engine"
version: "1.0.0"

inputs:
  message:
    type: string
    description: "Message to process"

steps:
  - id: process
    action: <AUTO>process the message {{message}} and return a summary</AUTO>
    
  - id: format
    action: <AUTO>format the processed result as a nice output</AUTO>
    depends_on: [process]

outputs:
  result: "{{format.result}}"
"""
    
    # Create engine
    engine = DeclarativePipelineEngine()
    
    # Validate pipeline
    print("🔍 Validating pipeline...")
    validation = await engine.validate_pipeline(pipeline_yaml)
    print(f"Validation result: {validation}")
    
    if not validation["valid"]:
        print("❌ Pipeline validation failed")
        return False
    
    # Execute pipeline
    print("\n🚀 Executing pipeline...")
    try:
        result = await engine.execute_pipeline(
            pipeline_yaml, 
            {"message": "Hello from declarative engine!"}
        )
        
        print("✅ Pipeline executed successfully!")
        print(f"Result: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        return False


async def test_pipeline_parsing():
    """Test pipeline parsing and specification creation."""
    
    pipeline_yaml = """
name: "Test Pipeline"
description: "Test parsing"

inputs:
  topic: {type: string}

steps:
  - id: search
    action: <AUTO>search for {{topic}}</AUTO>
    tools: [web-search]
    
  - id: analyze
    action: <AUTO>analyze search results</AUTO>
    depends_on: [search]

outputs:
  analysis: "{{analyze.result}}"
"""
    
    engine = DeclarativePipelineEngine()
    
    try:
        spec = engine._parse_yaml_to_spec(pipeline_yaml)
        print("✅ Pipeline parsing successful!")
        print(f"Pipeline: {spec.name}")
        print(f"Steps: {[step.id for step in spec.steps]}")
        print(f"Execution order: {[step.id for step in spec.get_execution_order()]}")
        print(f"AUTO tag steps: {[step.id for step in spec.get_steps_with_auto_tags()]}")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline parsing failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🧪 Testing Declarative Pipeline Engine - Phase 1")
    print("=" * 60)
    
    tests = [
        ("Pipeline Parsing", test_pipeline_parsing),
        ("Simple Pipeline Execution", test_simple_pipeline),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Test: {test_name}")
        print("-" * 40)
        
        success = await test_func()
        results.append((test_name, success))
        
        if success:
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Phase 1 implementation successful.")
    else:
        print("⚠️  Some tests failed. Implementation needs review.")


if __name__ == "__main__":
    asyncio.run(main())