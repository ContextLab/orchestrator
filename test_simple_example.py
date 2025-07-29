#!/usr/bin/env python3
"""Test a simple example pipeline."""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import orchestrator

async def test_simple():
    """Test a simple working example."""
    print("Testing working_web_search.yaml...")
    
    # Initialize models
    orchestrator.init_models()
    
    # Compile pipeline
    pipeline = await orchestrator.compile_async("examples/working_web_search.yaml")
    
    # Run with test inputs
    result = await pipeline.run_async(
        query="Python programming best practices",
        num_results=3
    )
    
    print(f"Result type: {type(result)}")
    if isinstance(result, dict):
        print(f"Result keys: {result.keys()}")
        for key, value in result.items():
            print(f"{key}: {value}")
    else:
        print(f"Result: {result}")
    
    return result

if __name__ == "__main__":
    asyncio.run(test_simple())