#!/usr/bin/env python3
"""Debug exactly where orchestrator hangs during execution."""

import asyncio
import sys
from pathlib import Path
import time
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)

async def test_orchestrator_step_by_step():
    """Test orchestrator execution step by step with debugging."""
    print("🔍 Testing orchestrator execution step by step...")
    
    try:
        from orchestrator import Orchestrator, init_models
        from orchestrator.compiler.yaml_compiler import YAMLCompiler
        
        # Initialize components
        print("\n1. Initializing model registry...")
        model_registry = init_models()
        print("✅ Model registry initialized")
        
        print("\n2. Creating orchestrator...")
        orchestrator = Orchestrator(model_registry=model_registry)
        print("✅ Orchestrator created")
        
        print("\n3. Compiling YAML...")
        yaml_content = """
id: test-basic-minimal
name: Basic No-AUTO Test
description: Test without ANY AUTO tags

parameters:
  test_input:
    type: string
    default: "test"

steps:
  - id: simple_generate
    action: generate_text
    parameters:
      prompt: "Say hello to {{ test_input }}"
      model: "gpt-3.5-turbo"
      max_tokens: 10

outputs:
  result: "{{ simple_generate.result }}"
"""
        
        pipeline = await orchestrator.yaml_compiler.compile(yaml_content, context={"test_input": "world"})
        print("✅ Pipeline compiled")
        print(f"   Tasks: {pipeline.tasks}")
        
        print("\n4. Testing individual task execution...")
        # Get the task using the pipeline method
        task_id = 'simple_generate'
        task = pipeline.get_task(task_id)
        print(f"   Task: {task.id} - {task.action}")
        print(f"   Parameters: {task.parameters}")
        
        # Test the control system directly
        print("\n5. Testing control system task execution...")
        start = time.time()
        
        context = {
            "pipeline_id": pipeline.id,
            "test_input": "world"
        }
        
        try:
            result = await asyncio.wait_for(
                orchestrator.control_system.execute_task(task, context),
                timeout=20.0
            )
            print(f"✅ Control system execution successful in {time.time() - start:.2f}s")
            print(f"   Result: {result}")
            return True
            
        except asyncio.TimeoutError:
            print(f"⏱️  Control system execution timed out after 20s")
            print(f"   Task status: {task.status}")
            
            # Let's check what's in the thread pool or async execution
            print("   Checking execution state...")
            if hasattr(orchestrator, 'parallel_executor'):
                stats = orchestrator.parallel_executor.get_statistics()
                print(f"   Executor stats: {stats}")
            
            return False
        
    except Exception as e:
        print(f"❌ Orchestrator execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    print("🔧 Debugging orchestrator execution hang...")
    print("=" * 60)
    
    success = await test_orchestrator_step_by_step()
    
    if success:
        print("\n🎉 Orchestrator execution successful!")
    else:
        print("\n❌ Found the exact hanging point!")

if __name__ == "__main__":
    asyncio.run(main())