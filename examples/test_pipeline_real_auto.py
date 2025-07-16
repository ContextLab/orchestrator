#!/usr/bin/env python3
"""Test pipeline AUTO resolution with real models."""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from orchestrator.orchestrator import Orchestrator
from orchestrator.core.control_system import MockControlSystem
from orchestrator.core.task import Task
from orchestrator.integrations.ollama_model import OllamaModel


class SimpleControlSystem(MockControlSystem):
    """Simple control system for testing AUTO resolution."""
    
    def __init__(self):
        super().__init__(name="simple-control")
        self._results = {}
    
    async def execute_task(self, task: Task, context: dict = None):
        """Execute task with simple implementation."""
        # Just return a simple result based on action
        result = {
            "status": "completed",
            "action": task.action,
            "parameters": dict(task.parameters),
            "message": f"Executed {task.action} successfully"
        }
        self._results[task.id] = result
        return result


async def test_auto_resolution():
    """Test AUTO tag resolution with real Ollama model."""
    print("🎯 Testing AUTO Resolution with Real Model")
    print("="*50)
    
    try:
        # Create Ollama model
        print("📥 Setting up Ollama model...")
        model = OllamaModel(model_name="llama3.2:1b", timeout=15)
        
        if not model._is_available:
            print("❌ Ollama model not available")
            return False
        
        print(f"✅ Using model: {model.name}")
        
        # Test direct AUTO resolution
        from orchestrator.compiler.ambiguity_resolver import AmbiguityResolver
        resolver = AmbiguityResolver(model=model)
        
        print("\n🧪 Testing direct AUTO resolution:")
        
        simple_tests = [
            ("Choose format: json or csv", "json"),
            ("Select size: small or large", "small"),  
            ("Pick method: fast or thorough", "fast"),
        ]
        
        for content, expected_type in simple_tests:
            try:
                print(f"🔍 Resolving: '{content}'")
                resolved = await resolver.resolve(content, "test.parameter")
                print(f"✅ Result: '{resolved}'")
                
                # Just check it's not empty
                if not resolved or resolved.strip() == "":
                    print(f"❌ Empty resolution for '{content}'")
                    return False
                    
            except Exception as e:
                print(f"❌ Failed to resolve '{content}': {e}")
                return False
        
        print("\n🎉 AUTO resolution tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_simple_pipeline():
    """Test a simple pipeline with real AUTO resolution."""
    print("\n🚀 Testing Simple Pipeline with Real AUTO")
    print("="*50)
    
    try:
        # Simple pipeline YAML with AUTO tags
        pipeline_yaml = """
name: "simple_auto_test"
description: "Test AUTO resolution"

steps:
  - id: test_step
    action: process
    parameters:
      format: <AUTO>Choose output format: json or csv</AUTO>
      mode: <AUTO>Select processing mode: fast or thorough</AUTO>
      size: <AUTO>Pick batch size: small or large</AUTO>
"""
        
        # Set up orchestrator
        control_system = SimpleControlSystem()
        orchestrator = Orchestrator(control_system=control_system)
        
        # Use real Ollama model for AUTO resolution
        model = OllamaModel(model_name="llama3.2:1b", timeout=15)
        if not model._is_available:
            print("❌ Ollama model not available")
            return False
        
        orchestrator.yaml_compiler.ambiguity_resolver.model = model
        print(f"✅ Using model: {model.name}")
        
        # Execute pipeline
        print("\n⚙️  Executing pipeline...")
        results = await orchestrator.execute_yaml(pipeline_yaml, context={})
        
        print("✅ Pipeline completed!")
        print(f"📊 Tasks: {len(results)}")
        
        # Check results
        for task_id, result in results.items():
            print(f"\n📋 Task: {task_id}")
            if isinstance(result, dict):
                params = result.get("parameters", {})
                print(f"   📄 Format: {params.get('format', 'unknown')}")
                print(f"   ⚙️  Mode: {params.get('mode', 'unknown')}")
                print(f"   📊 Size: {params.get('size', 'unknown')}")
                
                # Verify AUTO tags were resolved
                for key, value in params.items():
                    if isinstance(value, str) and ("<AUTO>" in value or "AUTO>" in value):
                        print(f"❌ AUTO tag not resolved: {key} = {value}")
                        return False
        
        print("\n🎉 Pipeline test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run AUTO resolution tests with real models."""
    print("🚀 REAL MODEL AUTO RESOLUTION TESTS")
    print("="*50)
    
    results = []
    
    # Test 1: Direct AUTO resolution  
    success = await test_auto_resolution()
    results.append(("AUTO Resolution", success))
    
    if success:
        # Test 2: Pipeline with AUTO tags
        success = await test_simple_pipeline()
        results.append(("Pipeline AUTO", success))
    else:
        print("⏭️  Skipping pipeline test due to AUTO resolution failure")
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 TEST RESULTS")
    print("="*50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    overall_success = passed == total
    print(f"\n📈 Tests: {passed}/{total} passed ({passed/total*100:.1f}%)")
    
    if overall_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Real model AUTO resolution working")
        print("✅ Pipeline integration successful")
    else:
        print("\n⚠️ SOME TESTS FAILED")
    
    return overall_success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)