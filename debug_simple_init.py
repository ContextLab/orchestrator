#!/usr/bin/env python3
"""Debug simple initialization to find the hanging point."""

import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_import_orchestrator():
    """Test importing orchestrator."""
    print("Testing import orchestrator...")
    start = time.time()
    try:
        from orchestrator import Orchestrator
        print(f"✅ Import successful in {time.time() - start:.2f}s")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_init_models():
    """Test init_models function."""
    print("Testing init_models...")
    start = time.time()
    try:
        from orchestrator import init_models
        print(f"✅ init_models import successful in {time.time() - start:.2f}s")
        
        print("Calling init_models()...")
        start = time.time()
        # Set a shorter timeout by checking time
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("init_models() took too long")
            
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)  # 10 second timeout
        
        model_registry = init_models()
        signal.alarm(0)  # Cancel alarm
        
        print(f"✅ init_models() successful in {time.time() - start:.2f}s")
        return True
        
    except TimeoutError as e:
        print(f"⏱️  init_models() timed out: {e}")
        return False
    except Exception as e:
        print(f"❌ init_models() failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_registry():
    """Test model registry creation."""
    print("Testing model registry...")
    start = time.time()
    try:
        from orchestrator.models.model_registry import ModelRegistry
        print(f"✅ Import ModelRegistry in {time.time() - start:.2f}s")
        
        start = time.time()
        registry = ModelRegistry()
        print(f"✅ Create ModelRegistry in {time.time() - start:.2f}s")
        return True
        
    except Exception as e:
        print(f"❌ ModelRegistry failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_keys():
    """Test API key loading."""
    print("Testing API key loading...")
    start = time.time()
    try:
        from orchestrator.utils.api_keys_flexible import load_api_keys_optional
        keys = load_api_keys_optional()
        print(f"✅ API keys loaded in {time.time() - start:.2f}s: {list(keys.keys())}")
        return True
        
    except Exception as e:
        print(f"❌ API keys failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Debugging orchestrator initialization hang...")
    print("=" * 50)
    
    # Test each component individually
    tests = [
        test_import_orchestrator,
        test_api_keys, 
        test_model_registry,
        test_init_models,
    ]
    
    for i, test in enumerate(tests):
        print(f"\n[{i+1}/{len(tests)}] ", end="")
        success = test()
        if not success:
            print(f"❌ Test failed: {test.__name__}")
            print("Stopping here to investigate...")
            break
        print("✅ Test passed")
    
    print("\n🎉 All tests completed!")