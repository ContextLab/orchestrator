#!/usr/bin/env python3
"""Debug YAML compilation to find hanging point."""

import sys
from pathlib import Path
import time
import signal

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_yaml_compilation():
    """Test YAML compilation of the simple test pipeline."""
    print("Testing YAML compilation...")
    start = time.time()
    
    try:
        from orchestrator.compiler.yaml_compiler import YAMLCompiler
        print(f"✅ Import YAMLCompiler in {time.time() - start:.2f}s")
        
        # Read the simple test YAML
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
        
        # Compile the pipeline
        start = time.time()
        
        def timeout_handler(signum, frame):
            raise TimeoutError("YAML compilation took too long")
            
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)  # 10 second timeout
        
        compiler = YAMLCompiler()
        pipeline = compiler.compile_from_string(yaml_content, inputs={"test_input": "world"})
        
        signal.alarm(0)  # Cancel alarm
        
        print(f"✅ YAML compilation successful in {time.time() - start:.2f}s")
        print(f"   Pipeline ID: {pipeline.id}")
        print(f"   Pipeline tasks: {[task.id for task in pipeline.tasks]}")
        return True
        
    except TimeoutError as e:
        print(f"⏱️  YAML compilation timed out: {e}")
        return False
    except Exception as e:
        print(f"❌ YAML compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_orchestrator_creation():
    """Test orchestrator creation."""
    print("Testing orchestrator creation...")
    start = time.time()
    
    try:
        from orchestrator import Orchestrator, init_models
        
        print("Creating model registry...")
        model_registry = init_models()
        print(f"✅ Model registry created")
        
        print("Creating orchestrator...")
        orchestrator = Orchestrator(model_registry=model_registry)
        print(f"✅ Orchestrator created in {time.time() - start:.2f}s")
        return orchestrator
        
    except Exception as e:
        print(f"❌ Orchestrator creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_pipeline_execution():
    """Test basic pipeline execution."""
    print("Testing pipeline execution...")
    
    try:
        # Get orchestrator
        orchestrator = test_orchestrator_creation()
        if not orchestrator:
            return False
            
        # Create simple pipeline
        from orchestrator.compiler.yaml_compiler import YAMLCompiler
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
        
        compiler = YAMLCompiler()
        pipeline = compiler.compile_from_string(yaml_content, inputs={"test_input": "world"})
        
        print("Executing pipeline...")
        start = time.time()
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Pipeline execution took too long")
            
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout
        
        # Execute pipeline (this is async)
        import asyncio
        result = asyncio.run(orchestrator.execute_pipeline(pipeline))
        
        signal.alarm(0)  # Cancel alarm
        
        print(f"✅ Pipeline execution successful in {time.time() - start:.2f}s")
        print(f"   Result: {result}")
        return True
        
    except TimeoutError as e:
        print(f"⏱️  Pipeline execution timed out: {e}")
        return False
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 Debugging YAML compilation and pipeline execution...")
    print("=" * 60)
    
    # Test each component
    tests = [
        test_yaml_compilation,
        test_orchestrator_creation,
        test_pipeline_execution,
    ]
    
    for i, test in enumerate(tests):
        print(f"\n[{i+1}/{len(tests)}] ", end="")
        if test.__name__ == 'test_orchestrator_creation':
            # Special handling for this test
            result = test()
            success = result is not None
        else:
            success = test()
            
        if not success:
            print(f"❌ Test failed: {test.__name__}")
            print("Stopping here to investigate...")
            break
        print("✅ Test passed")
    
    print("\n🎉 All tests completed!")