#!/usr/bin/env python3
"""Very basic test of a single tool."""

import asyncio
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from src.orchestrator import Orchestrator, init_models


async def test_basic():
    """Test basic functionality."""
    print("Initializing models...")
    model_registry = init_models()
    print(f"Models initialized: {len(model_registry.models)} models")
    
    print("\nCreating orchestrator...")
    orchestrator = Orchestrator(model_registry=model_registry)
    print("Orchestrator created")
    
    # Simplest possible pipeline
    pipeline_yaml = """
name: basic-test
description: Basic test

steps:
  - id: simple_task
    tool: report-generator
    action: generate
    parameters:
      title: "Test"
      format: "markdown"
      content: "Hello World"
"""
    
    print("\nExecuting pipeline...")
    try:
        result = await orchestrator.execute_yaml(pipeline_yaml)
        print("✅ Success!")
        print(f"Result: {result}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    asyncio.run(test_basic())