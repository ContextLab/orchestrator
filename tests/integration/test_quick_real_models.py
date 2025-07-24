#!/usr/bin/env python3
"""Quick test of real model integration."""

import asyncio
import sys
import os

from orchestrator.integrations.ollama_model import OllamaModel


async def test_ollama_model():
    """Test Ollama model integration."""
    print("🦙 Testing Ollama Model Integration")
    print("=" * 50)

    try:
        # Test with llama3.2:1b (fastest available model)
        print("📥 Loading llama3.2:1b...")
        model = OllamaModel(model_name="llama3.2:1b")

        print(f"✅ Model created: {model.name}")
        print(f"🔍 Available: {model._is_available}")

        if not model._is_available:
            print("❌ Model not available")
            return False

        # Test simple generation
        print("\n🧪 Testing simple generation...")
        result = await model.generate("What is 2+2?", max_tokens=10, temperature=0.1)
        print(f"✅ Generated: {result}")

        # Test AUTO-style resolution
        print("\n🎯 Testing AUTO resolution scenarios...")

        prompts = [
            "Choose the best format for data output: json, csv, or xml",
            "Select appropriate batch size: small, medium, or large",
            "Pick suitable timeout value: 10, 30, or 60 seconds",
        ]

        for prompt in prompts:
            try:
                result = await model.generate(prompt, max_tokens=20, temperature=0.1)
                # Extract just the choice
                choice = result.split()[0] if result else "unknown"
                print(f"✅ '{prompt}' → '{choice}'")
            except Exception as e:
                print(f"❌ Failed: {e}")
                return False

        print("\n🎉 All Ollama tests passed!")
        return True

    except Exception as e:
        print(f"❌ Ollama test failed: {e}")
        return False


async def test_ambiguity_resolver():
    """Test the ambiguity resolver with real model."""
    print("\n🔧 Testing Ambiguity Resolver with Real Model")
    print("=" * 50)

    try:
        from orchestrator.compiler.ambiguity_resolver import AmbiguityResolver

        # Create resolver (should auto-detect and use Ollama model)
        print("🔍 Creating ambiguity resolver...")
        resolver = AmbiguityResolver()

        print(f"✅ Using model: {resolver.model.name}")
        print(f"📍 Provider: {resolver.model.provider}")

        # Test AUTO resolution
        test_cases = [
            ("Choose output format", "config.format"),
            ("Select batch size", "settings.batch_size"),
            ("Pick analysis method", "task.method"),
        ]

        print("\n🎯 Testing AUTO resolution:")
        for content, context in test_cases:
            try:
                resolved = await resolver.resolve(content, context)
                print(f"✅ '{content}' → '{resolved}'")
            except Exception as e:
                print(f"❌ Failed to resolve '{content}': {e}")
                return False

        print("\n🎉 Ambiguity resolver tests passed!")
        return True

    except Exception as e:
        print(f"❌ Ambiguity resolver test failed: {e}")
        return False


async def main():
    """Run quick real model tests."""
    print("🚀 QUICK REAL MODEL TESTS")
    print("Testing with available Ollama models")
    print("=" * 50)

    results = []

    # Test 1: Direct Ollama model
    success = await test_ollama_model()
    results.append(("Ollama Model", success))

    # Test 2: Ambiguity resolver integration
    success = await test_ambiguity_resolver()
    results.append(("Ambiguity Resolver", success))

    # Summary
    print(f"\n{'='*50}")
    print("📊 TEST RESULTS")
    print("=" * 50)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")

    overall_success = passed == total
    print(f"\n📈 Tests: {passed}/{total} passed ({passed/total*100:.1f}%)")

    if overall_success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Real model integration working")
    else:
        print("⚠️ SOME TESTS FAILED")

    return overall_success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
