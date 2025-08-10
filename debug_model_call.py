#!/usr/bin/env python3
"""Debug model call directly."""

import asyncio
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'src'))

async def test_direct_model_call():
    """Test calling model directly."""
    try:
        from orchestrator import init_models
        
        print("Initializing models...")
        model_registry = init_models()
        print("Models initialized successfully")
        
        # Try to get available models
        print("Getting available models...")
        available_models = await model_registry.get_available_models()
        print(f"Available models: {available_models}")
        
        # Try model selection
        print("Selecting model...")
        selected_model = await model_registry.select_model({
            'tasks': ['generate'],
            'context_window': 100
        })
        print(f"Selected model: {selected_model}")
        
        # Try actual model call
        print("Making model call...")
        from orchestrator.integrations.openai_model import OpenAIModel
        
        openai_model = OpenAIModel("gpt-3.5-turbo")
        response = await openai_model.call({
            'prompt': 'Say hello',
            'max_tokens': 5
        })
        print(f"Model response: {response}")
        
        print("✅ Direct model call successful!")
        
    except Exception as e:
        print(f"❌ Error in direct model call: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_direct_model_call())