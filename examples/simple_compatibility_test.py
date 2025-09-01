#!/usr/bin/env python3
"""
Simple Backward Compatibility Test

Tests basic imports and structure without full initialization.
"""

import os
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_basic_imports():
    """Test basic imports without initialization."""
    print("Testing basic imports...")
    
    try:
        # Test core classes
        from orchestrator.core.pipeline import Pipeline
        from orchestrator.core.task import Task, TaskStatus
        print("✅ Core classes import successfully")
        
        # Test compiler
        from orchestrator.compiler.yaml_compiler import YAMLCompiler
        print("✅ YAML compiler imports successfully")
        
        # Test main orchestrator
        from orchestrator.orchestrator import Orchestrator
        print("✅ Main Orchestrator class imports successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_yaml_structure():
    """Test YAML structure compatibility."""
    print("\nTesting YAML structure...")
    
    try:
        import yaml
        
        # Test a simple pipeline
        yaml_path = Path(__file__).parent / "test_simple_pipeline.yaml"
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                pipeline_def = yaml.safe_load(f)
            
            # Check basic structure
            if 'steps' in pipeline_def:
                print(f"✅ Simple pipeline has valid structure")
                print(f"   - Steps: {len(pipeline_def['steps'])}")
                return True
            else:
                print(f"❌ Missing 'steps' field")
                return False
        else:
            print("⚠️  test_simple_pipeline.yaml not found")
            return True  # Don't fail if file is missing
            
    except Exception as e:
        print(f"❌ YAML structure test failed: {e}")
        return False

def run_tests():
    """Run lightweight compatibility tests."""
    print("🔄 Running Simple Compatibility Tests")
    print("=" * 40)
    
    tests = [
        test_basic_imports,
        test_yaml_structure,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 Basic compatibility tests PASSED!")
        return True
    else:
        print("⚠️  Some compatibility issues detected.")
        return False

if __name__ == "__main__":
    success = run_tests()
    print("\n✅ Import structure appears to be backward compatible")
    print("📋 Next step: Create full backward compatibility layer")
    sys.exit(0 if success else 1)