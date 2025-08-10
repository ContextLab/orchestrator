#!/usr/bin/env python3
"""Debug execute_yaml to find hanging point."""

import asyncio
import sys
from pathlib import Path
import time
import signal

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

async def test_yaml_compile():
    """Test YAML compilation directly."""
    print("Testing YAML compilation...")
    
    try:
        from orchestrator.compiler.yaml_compiler import YAMLCompiler
        from orchestrator import init_models
        
        # Create YAML compiler
        compiler = YAMLCompiler()
        
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
        
        start = time.time()
        
        # Compile with timeout
        try:
            pipeline = await asyncio.wait_for(
                compiler.compile(yaml_content, context={"test_input": "world"}),
                timeout=10.0
            )
            print(f"✅ YAML compilation successful in {time.time() - start:.2f}s")
            print(f"   Pipeline ID: {pipeline.id}")
            return pipeline
            
        except asyncio.TimeoutError:
            print(f"⏱️  YAML compilation timed out after 10s")
            return None
        
    except Exception as e:
        print(f"❌ YAML compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_execute_yaml():
    """Test execute_yaml method."""
    print("Testing execute_yaml...")
    
    try:
        from orchestrator import Orchestrator, init_models
        
        # Initialize orchestrator
        model_registry = init_models()
        orchestrator = Orchestrator(model_registry=model_registry)
        
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
        
        start = time.time()
        
        # Execute with timeout
        try:
            result = await asyncio.wait_for(
                orchestrator.execute_yaml(yaml_content, context={"test_input": "world"}),
                timeout=30.0
            )
            print(f"✅ Execute YAML successful in {time.time() - start:.2f}s")
            print(f"   Result: {result}")
            return True
            
        except asyncio.TimeoutError:
            print(f"⏱️  Execute YAML timed out after 30s")
            return False
        
    except Exception as e:
        print(f"❌ Execute YAML failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    print("🔍 Debugging YAML compilation and execution...")
    print("=" * 60)
    
    # Test compilation first
    print("\n[1/2] Testing YAML compilation...")
    pipeline = await test_yaml_compile()
    if pipeline is None:
        print("❌ YAML compilation failed, stopping")
        return
    
    print("✅ YAML compilation passed")
    
    # Test execution
    print("\n[2/2] Testing execute_yaml...")
    success = await test_execute_yaml()
    if not success:
        print("❌ Execute YAML failed")
        return
        
    print("✅ Execute YAML passed")
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())