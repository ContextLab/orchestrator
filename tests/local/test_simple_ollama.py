#!/usr/bin/env python3
"""Simple Ollama integration test."""

import asyncio
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Mark all tests in this file as local-only (not run in CI)
pytestmark = pytest.mark.local


async def test_ollama_direct():
    """Test Ollama integration directly."""
    print("🦙 Testing Ollama Integration")
    print("=" * 40)

    try:
        from orchestrator.integrations.ollama_model import OllamaModel

        # Create model with longer timeout
        print("📥 Creating Ollama model...")
        model = OllamaModel(model_name="llama3.2:1b", timeout=60)

        print(f"✅ Model: {model.name}")
        print(f"🔍 Available: {model._is_available}")

        if not model._is_available:
            print("❌ Model not available")
            return False

        # Test health check first
        print("\n🏥 Running health check...")
        healthy = await model.health_check()
        print(f"✅ Health check: {'PASS' if healthy else 'FAIL'}")

        if not healthy:
            return False

        # Test simple generation
        print("\n🧪 Testing generation...")
        result = await model.generate("2+2=", max_tokens=3, temperature=0.0)
        print(f"✅ Result: '{result}'")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    success = await test_ollama_direct()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}")
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
