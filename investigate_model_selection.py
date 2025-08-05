#!/usr/bin/env python3

import asyncio
import sys
import os
sys.path.insert(0, '/Users/jmanning/orchestrator/src')

from orchestrator.models.model_registry import ModelRegistry


async def investigate_model_selection():
    """Investigate model selection to understand gated repo issues."""
    
    print("=== Model Registry Investigation ===")
    
    # Create model registry
    registry = ModelRegistry()
    
    # List all available models
    all_models = registry.list_models()
    print(f"\nTotal registered models: {len(all_models)}")
    
    # Group by provider
    providers = {}
    for model_name in all_models:
        if ':' in model_name:
            provider, name = model_name.split(':', 1)
            if provider not in providers:
                providers[provider] = []
            providers[provider].append(name)
        else:
            if 'unknown' not in providers:
                providers['unknown'] = []
            providers['unknown'].append(model_name)
    
    print("\nModels by provider:")
    for provider, models in providers.items():
        print(f"  {provider}: {len(models)} models")
        for model in models[:3]:  # Show first 3
            print(f"    - {model}")
        if len(models) > 3:
            print(f"    ... and {len(models) - 3} more")
    
    # Try the same selection as the failing tests
    requirements = {"tasks": ["generate"]}
    print(f"\nTesting model selection with requirements: {requirements}")
    
    try:
        selected_model = await registry.select_model(requirements)
        print(f"Selected model: {selected_model.name}")
        print(f"  Provider: {selected_model.provider if hasattr(selected_model, 'provider') else 'Unknown'}")
        print(f"  Capabilities: {selected_model.capabilities}")
        
        # Check if it's a gated model
        if 'huggingface' in selected_model.name.lower():
            print(f"  ⚠️  This is a HuggingFace model - may be gated")
            
        # Try to identify non-gated alternatives
        print(f"\nLooking for non-gated alternatives...")
        non_huggingface_models = [name for name in all_models if not name.startswith('huggingface:')]
        print(f"Non-HuggingFace models: {len(non_huggingface_models)}")
        for model in non_huggingface_models[:5]:
            print(f"  - {model}")
        
    except Exception as e:
        print(f"Model selection failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test individual provider models
    print(f"\n=== Testing Provider-Specific Models ===")
    providers_to_test = ['openai', 'anthropic', 'google']
    
    for provider in providers_to_test:
        provider_models = [name for name in all_models if name.startswith(f"{provider}:")]
        if provider_models:
            print(f"\n{provider.upper()} models ({len(provider_models)}):")
            for model in provider_models[:3]:
                print(f"  - {model}")
            
            # Try to get one of these models
            try:
                model_obj = registry.get_model(provider_models[0])
                print(f"  ✅ Successfully retrieved: {model_obj.name}")
            except Exception as e:
                print(f"  ❌ Failed to retrieve: {e}")


if __name__ == "__main__":
    asyncio.run(investigate_model_selection())