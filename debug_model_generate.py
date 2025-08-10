#!/usr/bin/env python3
"""Debug model.generate call that's hanging."""

import asyncio
import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

async def test_model_generate_exact():
    """Test the exact model.generate call that's hanging."""
    print("Testing exact model.generate call...")
    
    try:
        from orchestrator import init_models
        
        # Initialize models
        model_registry = init_models()
        
        # Select a model exactly like the control system does
        requirements = {
            'tasks': ['generate'],
            'context_window': 19,  # From our previous debug
            'expertise': ['general']
        }
        
        print("Selecting model...")
        model = await model_registry.select_model(requirements)
        print(f"Selected model: {model.name} (type: {type(model)})")
        
        # Test the exact parameters that are hanging
        gen_kwargs = {
            "prompt": "Say hello to world",  # This is what gets rendered
            "temperature": 0.7,
            "max_tokens": 10,
        }
        
        print(f"Calling model.generate with kwargs: {gen_kwargs}")
        start = time.time()
        
        try:
            result = await asyncio.wait_for(
                model.generate(**gen_kwargs),
                timeout=15.0
            )
            print(f"✅ Model generate successful in {time.time() - start:.2f}s")
            print(f"   Result: {result}")
            return True
            
        except asyncio.TimeoutError:
            print(f"⏱️  Model generate timed out after 15s")
            return False
        
    except Exception as e:
        print(f"❌ Model generate failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_model_generate_simple():
    """Test simpler model.generate call."""
    print("Testing simple model.generate call...")
    
    try:
        from orchestrator import init_models
        
        # Initialize and select model
        model_registry = init_models()
        requirements = {'tasks': ['generate'], 'context_window': 19, 'expertise': ['general']}
        model = await model_registry.select_model(requirements)
        
        print(f"Testing simple call on {model.name}...")
        start = time.time()
        
        try:
            # Test the simple call that worked before
            result = await asyncio.wait_for(
                model.generate("Say hello", max_tokens=5),
                timeout=10.0
            )
            print(f"✅ Simple generate successful in {time.time() - start:.2f}s")
            print(f"   Result: {result}")
            return True
            
        except asyncio.TimeoutError:
            print(f"⏱️  Simple generate timed out after 10s")
            return False
        
    except Exception as e:
        print(f"❌ Simple generate failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    print("🔍 Debugging model.generate call that's hanging...")
    print("=" * 60)
    
    # Test simple call first
    print("\n[1/2] Testing simple model.generate...")
    success = await test_model_generate_simple()
    if not success:
        print("❌ Simple generate failed")
        return
    print("✅ Simple generate passed")
    
    # Test exact call
    print("\n[2/2] Testing exact model.generate call...")
    success = await test_model_generate_exact()
    if not success:
        print("❌ Exact generate failed")
        return
    print("✅ Exact generate passed")
    
    print("\n🎉 All model.generate tests completed!")

if __name__ == "__main__":
    asyncio.run(main())