#!/usr/bin/env python3
"""Test a few example pipelines to verify the testing framework."""

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Reduce logging
os.environ['ORCHESTRATOR_LOG_LEVEL'] = 'WARNING'

import orchestrator

async def test_example(yaml_path: str, inputs: dict):
    """Test a single example."""
    print(f"\n{'='*70}")
    print(f"Testing: {yaml_path}")
    print(f"{'='*70}")
    
    try:
        # Compile pipeline
        print("Compiling pipeline...")
        pipeline = await orchestrator.compile_async(yaml_path)
        
        # Run pipeline
        print("Running pipeline...")
        start = datetime.now()
        outputs = await pipeline.run_async(**inputs)
        duration = (datetime.now() - start).total_seconds()
        
        print(f"✅ Success! Duration: {duration:.2f}s")
        print(f"Outputs: {json.dumps(outputs, indent=2, default=str) if isinstance(outputs, dict) else outputs}")
        
        return True
    except Exception as e:
        print(f"❌ Failed: {type(e).__name__}: {str(e)}")
        return False

async def main():
    """Test a few examples."""
    print("Initializing models...")
    orchestrator.init_models()
    
    # Test cases
    tests = [
        ("examples/terminal_automation.yaml", {}),
        ("examples/validation_pipeline.yaml", {
            "data": {"name": "Test", "email": "test@example.com"},
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string", "format": "email"}
                },
                "required": ["name", "email"]
            }
        })
    ]
    
    results = []
    for yaml_path, inputs in tests:
        success = await test_example(yaml_path, inputs)
        results.append((yaml_path, success))
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    for path, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {path}")

if __name__ == "__main__":
    asyncio.run(main())