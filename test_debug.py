#!/usr/bin/env python3
"""Debug test for pipeline execution"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import Orchestrator, init_models

async def test_simple():
    print("1. Initializing models...")
    model_registry = init_models()
    print(f"   Models: {len(model_registry.list_models())}")
    
    print("\n2. Creating orchestrator...")
    orchestrator = Orchestrator(model_registry=model_registry)
    print("   Orchestrator created")
    print(f"   Control system: {orchestrator.control_system}")
    
    yaml_content = """
name: "Test Pipeline"
description: "Simple test"

inputs:
  message:
    type: string
    default: "Hello"

steps:
  - id: echo
    action: "Echo message: {{message}}"

outputs:
  result: "{{echo.result}}"
"""
    
    print("\n3. Compiling YAML...")
    try:
        pipeline = await orchestrator.yaml_compiler.compile(yaml_content, {"message": "Test"})
        print(f"   Pipeline compiled: {pipeline.name}")
        print(f"   Tasks: {[t.id for t in pipeline.tasks.values()]}")
    except Exception as e:
        print(f"   Compile error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n4. Executing pipeline...")
    try:
        # Add timeout
        results = await asyncio.wait_for(
            orchestrator.execute_pipeline(pipeline),
            timeout=10.0
        )
        print("   Results:", results)
    except asyncio.TimeoutError:
        print("   ERROR: Pipeline execution timed out!")
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_simple())