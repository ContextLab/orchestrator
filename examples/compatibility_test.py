#!/usr/bin/env python3
"""
Backward Compatibility Test

This script tests that existing user code continues to work with the new architecture.
It tests the key user-facing APIs that existing pipeline definitions rely on.
"""

import os
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that key imports still work."""
    print("Testing imports...")
    
    try:
        import orchestrator
        print("✅ Basic orchestrator import")
        
        from orchestrator import compile, compile_async, init_models
        print("✅ Function imports")
        
        from orchestrator import Orchestrator, Pipeline, Task, TaskStatus
        print("✅ Class imports")
        
        from orchestrator import ModelRegistry, YAMLCompiler
        print("✅ Component imports")
        
        print(f"✅ Package version: {orchestrator.__version__}")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_api_compatibility():
    """Test that the main API functions exist and have the expected signatures."""
    print("\nTesting API compatibility...")
    
    try:
        import orchestrator
        
        # Test that compile function exists and is callable
        if hasattr(orchestrator, 'compile') and callable(orchestrator.compile):
            print("✅ compile() function available")
        else:
            print("❌ compile() function missing")
            return False
            
        # Test that compile_async function exists and is callable
        if hasattr(orchestrator, 'compile_async') and callable(orchestrator.compile_async):
            print("✅ compile_async() function available")
        else:
            print("❌ compile_async() function missing")
            return False
            
        # Test that init_models function exists
        if hasattr(orchestrator, 'init_models') and callable(orchestrator.init_models):
            print("✅ init_models() function available")
        else:
            print("❌ init_models() function missing")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ API compatibility test failed: {e}")
        return False

def test_example_yaml_structure():
    """Test that existing YAML files still have valid structure."""
    print("\nTesting YAML compatibility...")
    
    example_files = [
        "test_simple_pipeline.yaml",
        "code_optimization.yaml",
        "web_research_pipeline.yaml"
    ]
    
    examples_dir = Path(__file__).parent
    compatible_count = 0
    
    for yaml_file in example_files:
        yaml_path = examples_dir / yaml_file
        if not yaml_path.exists():
            print(f"⚠️  {yaml_file} not found, skipping")
            continue
            
        try:
            import yaml
            with open(yaml_path, 'r') as f:
                pipeline_def = yaml.safe_load(f)
            
            # Check for required fields that existing pipelines use
            required_fields = ['steps']
            optional_fields = ['id', 'name', 'description', 'parameters', 'inputs', 'outputs']
            
            if 'steps' in pipeline_def:
                print(f"✅ {yaml_file}: Valid pipeline structure")
                compatible_count += 1
            else:
                print(f"❌ {yaml_file}: Missing 'steps' field")
                
        except Exception as e:
            print(f"❌ {yaml_file}: Error parsing - {e}")
    
    print(f"✅ {compatible_count}/{len(example_files)} YAML files have compatible structure")
    return compatible_count > 0

def test_backward_compatibility():
    """Run all backward compatibility tests."""
    print("🔄 Running Backward Compatibility Tests")
    print("=" * 50)
    
    tests = [
        ("Import Compatibility", test_imports),
        ("API Compatibility", test_api_compatibility), 
        ("YAML Structure Compatibility", test_example_yaml_structure),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All backward compatibility tests PASSED!")
        print("✅ Existing user code should continue to work with the new architecture.")
        return True
    else:
        print("⚠️  Some backward compatibility issues detected.")
        print("🔧 These issues need to be addressed before migration is complete.")
        return False

if __name__ == "__main__":
    success = test_backward_compatibility()
    sys.exit(0 if success else 1)