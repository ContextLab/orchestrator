#!/usr/bin/env python3
"""Test a single example pipeline."""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import orchestrator

async def test_auto_tags_demo():
    """Test the AUTO tags demo pipeline."""
    print("Testing auto_tags_demo.yaml...")
    
    # Initialize models
    orchestrator.init_models()
    
    # Compile pipeline
    pipeline = await orchestrator.compile_async("examples/auto_tags_demo.yaml")
    
    # Run with test inputs
    result = await pipeline.run_async(
        task="summarize",
        content="Artificial intelligence is rapidly transforming industries worldwide. From healthcare to finance, AI systems are improving efficiency and enabling new capabilities."
    )
    
    print(f"Result: {result}")
    return result

if __name__ == "__main__":
    asyncio.run(test_auto_tags_demo())