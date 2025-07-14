#!/usr/bin/env python3
"""Debug AUTO resolution issues."""

import asyncio
import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


async def test_step_by_step():
    """Test each step individually."""
    print("🔍 Step-by-step AUTO resolution debug")
    print("="*40)
    
    try:
        # Step 1: Test Ollama model directly
        print("1️⃣ Testing Ollama model...")
        from orchestrator.integrations.ollama_model import OllamaModel
        model = OllamaModel(model_name="llama3.2:1b", timeout=10)
        print(f"   Available: {model._is_available}")
        
        if not model._is_available:
            print("❌ Ollama not available")
            return
        
        # Step 2: Test simple generation
        print("2️⃣ Testing generation...")
        result = await model.generate("Say 'hello'", max_tokens=5, temperature=0.0)
        print(f"   Result: '{result}'")
        
        # Step 3: Test ambiguity resolver creation
        print("3️⃣ Testing resolver creation...")
        from orchestrator.compiler.ambiguity_resolver import AmbiguityResolver
        resolver = AmbiguityResolver(model=model)
        print(f"   Resolver model: {resolver.model.name}")
        
        # Step 4: Test simple resolution
        print("4️⃣ Testing simple resolution...")
        resolved = await resolver.resolve("json", "test.format")
        print(f"   Resolved: '{resolved}'")
        
        # Step 5: Test complex resolution
        print("5️⃣ Testing complex resolution...")
        resolved = await resolver.resolve("Choose format: json or csv", "test.format")
        print(f"   Resolved: '{resolved}'")
        
        print("✅ All steps completed!")
        
    except Exception as e:
        print(f"❌ Error at step: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_step_by_step())