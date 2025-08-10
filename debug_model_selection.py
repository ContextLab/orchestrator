#!/usr/bin/env python3
"""Debug model selection to find hanging point."""

import asyncio
import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

async def test_direct_model_selection():
    """Test model selection directly."""
    print("Testing direct model selection...")
    
    try:
        from orchestrator import init_models
        
        # Initialize models
        print("Initializing models...")
        model_registry = init_models()
        print("Models initialized")
        
        # Get available models
        print("Getting available models...")
        available_models = await model_registry.get_available_models()
        print(f"Available models: {len(available_models)}")
        
        # Test simple model selection
        print("Testing model selection...")
        start = time.time()
        
        requirements = {
            'tasks': ['generate'],
            'context_window': 19,
            'expertise': ['general']
        }
        
        try:
            selected_model = await asyncio.wait_for(
                model_registry.select_model(requirements),
                timeout=10.0
            )
            print(f"✅ Model selection successful in {time.time() - start:.2f}s")
            print(f"   Selected: {selected_model.name}")
            return selected_model
            
        except asyncio.TimeoutError:
            print(f"⏱️  Model selection timed out after 10s")
            return None
        
    except Exception as e:
        print(f"❌ Model selection failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_direct_model_call():
    """Test calling a model directly."""
    print("Testing direct model call...")
    
    try:
        from orchestrator.integrations.openai_model import OpenAIModel
        
        # Create model directly
        model = OpenAIModel("gpt-3.5-turbo")
        
        print("Calling model...")
        start = time.time()
        
        try:
            response = await asyncio.wait_for(
                model.generate('Say hello', max_tokens=5),
                timeout=10.0
            )
            print(f"✅ Model call successful in {time.time() - start:.2f}s")
            print(f"   Response: {response}")
            return True
            
        except asyncio.TimeoutError:
            print(f"⏱️  Model call timed out after 10s")
            return False
        
    except Exception as e:
        print(f"❌ Model call failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_model_filtering():
    """Test model filtering process."""
    print("Testing model filtering...")
    
    try:
        from orchestrator import init_models
        
        model_registry = init_models()
        
        # Access the filtering method directly
        print("Testing _filter_by_capabilities...")
        start = time.time()
        
        requirements = {
            'tasks': ['generate'],
            'context_window': 19,
            'expertise': ['general']
        }
        
        try:
            eligible_models = await asyncio.wait_for(
                model_registry._filter_by_capabilities(requirements),
                timeout=5.0
            )
            print(f"✅ Model filtering successful in {time.time() - start:.2f}s")
            print(f"   Eligible models: {len(eligible_models)}")
            if eligible_models:
                print(f"   First model: {eligible_models[0].name}")
            return eligible_models
            
        except asyncio.TimeoutError:
            print(f"⏱️  Model filtering timed out after 5s")
            return None
        
    except Exception as e:
        print(f"❌ Model filtering failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    """Main test function."""
    print("🔍 Debugging model selection process...")
    print("=" * 60)
    
    # Test direct model call first
    print("\n[1/4] Testing direct model call...")
    success = await test_direct_model_call()
    if not success:
        print("❌ Direct model call failed")
        return
    print("✅ Direct model call passed")
    
    # Test model filtering
    print("\n[2/4] Testing model filtering...")
    eligible_models = await test_model_filtering()
    if eligible_models is None:
        print("❌ Model filtering failed")
        return
    print("✅ Model filtering passed")
    
    # Test model selection
    print("\n[3/4] Testing model selection...")
    selected_model = await test_direct_model_selection()
    if selected_model is None:
        print("❌ Model selection failed")
        return
    print("✅ Model selection passed")
    
    print("\n🎉 All model tests completed!")

if __name__ == "__main__":
    asyncio.run(main())