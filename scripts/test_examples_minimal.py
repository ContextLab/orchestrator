#!/usr/bin/env python3
"""Minimal test of example pipelines."""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Reduce logging
os.environ['ORCHESTRATOR_LOG_LEVEL'] = 'ERROR'

import asyncio
import orchestrator


async def test_example(name: str, yaml_file: str, inputs: dict):
    """Test a single example."""
    print(f"\nTesting {name}...")
    try:
        pipeline = await orchestrator.compile_async(yaml_file)
        result = await pipeline.run_async(**inputs)
        print(f"✅ {name}: Success")
        return True
    except Exception as e:
        print(f"❌ {name}: {type(e).__name__}: {str(e)[:100]}")
        return False


async def main():
    """Test key examples."""
    # Initialize models
    print("Initializing models...")
    orchestrator.init_models()
    
    # Create test data directory
    test_data_dir = Path("examples/test_data")
    test_data_dir.mkdir(exist_ok=True)
    
    # Create test CSV
    csv_file = test_data_dir / "test.csv"
    with open(csv_file, 'w') as f:
        f.write("id,value\n1,100\n2,200\n")
    
    # Test cases
    tests = [
        # Control flow examples (should work without API)
        ("For Loop", "examples/control_flow_for_loop.yaml", {
            "items": ["a", "b", "c"]
        }),
        
        ("Conditional", "examples/control_flow_conditional.yaml", {
            "value": 85,
            "threshold": 70
        }),
        
        ("While Loop", "examples/control_flow_while_loop.yaml", {
            "max_iterations": 3,
            "target_value": 10
        }),
        
        # Simple examples that don't need external APIs
        ("Terminal", "examples/terminal_automation.yaml", {
            "commands": ["echo 'test'"],
            "safe_mode": True
        }),
        
        ("Validation", "examples/validation_pipeline.yaml", {
            "data": {"name": "Test", "age": 30},
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                }
            }
        })
    ]
    
    # Run tests
    results = []
    for name, yaml_file, inputs in tests:
        success = await test_example(name, yaml_file, inputs)
        results.append((name, success))
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    
    success_count = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"Passed: {success_count}/{total}")
    
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {name}")


if __name__ == "__main__":
    asyncio.run(main())