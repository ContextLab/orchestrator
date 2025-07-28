#!/usr/bin/env python3
"""Test AUTO resolution quality with real models."""

import asyncio
import sys

from orchestrator.compiler.ambiguity_resolver import AmbiguityResolver
from orchestrator.integrations.ollama_model import OllamaModel


async def test_auto_resolution():
    """Test various AUTO resolution scenarios."""
    print("🧪 TESTING AUTO RESOLUTION QUALITY")
    print("=" * 50)

    # Create resolver with real model
    model = OllamaModel(model_name="llama3.2:1b")
    if not model._is_available:
        print("❌ Ollama model not available")
        return False

    print(f"✅ Using model: {model.name}")
    resolver = AmbiguityResolver(model=model)

    # Test cases
    test_cases = [
        # (content, context, expected_type)
        (
            "Choose best sources for healthcare AI research",
            "parameters.sources",
            "list"),
        ("Determine appropriate search depth", "parameters.depth", "string"),
        ("Select analysis method for research data", "parameters.method", "string"),
        ("Set relevance threshold", "parameters.threshold", "number"),
        ("Choose output format", "parameters.format", "string"),
        ("Determine summary length", "parameters.length", "string"),
        ("Detect programming language", "parameters.language", "string"),
        ("Choose appropriate scan type", "parameters.scan_type", "string"),
        ("Determine load level for testing", "parameters.load", "string"),
        ("Match source code language", "parameters.language", "string"),
        ("Choose optimization focus", "parameters.type", "string"),
        ("Select appropriate metrics for customer data", "parameters.metrics", "list"),
        ("Choose segmentation method", "parameters.method", "string"),
        ("Set significance threshold", "parameters.threshold", "number"),
        ("Choose format for business audience", "parameters.format", "string"),
        ("Choose ML framework language", "parameters.language", "string"),
    ]

    print(f"\n📋 Testing {len(test_cases)} AUTO resolution scenarios:\n")

    results = []
    for content, context, expected_type in test_cases:
        try:
            print(f"🔍 Content: '{content}'")
            print(f"   Context: {context}")
            print(f"   Expected: {expected_type}")

            resolved = await resolver.resolve(content, context)

            print(f"   ✅ Resolved: '{resolved}' (type: {type(resolved).__name__})")

            # Check if resolution makes sense
            is_valid = True
            issues = []

            if expected_type == "list" and not isinstance(resolved, list):
                # For lists, check if it's a comma-separated string at least
                if isinstance(resolved, str) and "," not in resolved:
                    is_valid = False
                    issues.append("Expected list or comma-separated values")

            if expected_type == "number" and isinstance(resolved, str):
                try:
                    float(resolved)
                except ValueError:
                    is_valid = False
                    issues.append("Expected numeric value")

            if isinstance(resolved, str) and len(resolved) < 2:
                is_valid = False
                issues.append("Resolution too short")

            if not is_valid:
                print(f"   ⚠️  Issues: {', '.join(issues)}")

            results.append((content, resolved, is_valid))
            print()

        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append((content, None, False))
            print()

    # Summary
    print("=" * 50)
    print("📊 RESOLUTION QUALITY SUMMARY")
    print("=" * 50)

    valid_count = sum(1 for _, _, valid in results if valid)
    total_count = len(results)

    print(
        f"\n✅ Valid resolutions: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)"
    )

    if valid_count < total_count:
        print("\n❌ Failed resolutions:")
        for content, resolved, valid in results:
            if not valid:
                print(f"   - '{content}' → '{resolved}'")

    return valid_count / total_count >= 0.7


async def test_specific_resolutions():
    """Test specific problematic resolutions."""
    print("\n🔧 TESTING SPECIFIC RESOLUTIONS")
    print("=" * 50)

    model = OllamaModel(model_name="llama3.2:1b")
    if not model._is_available:
        return False

    resolver = AmbiguityResolver(model=model)

    # Test the problematic "sources" resolution
    print("\n1️⃣ Testing sources resolution:")
    sources_prompt = "Choose best sources for healthcare AI research"
    resolved = await resolver.resolve(sources_prompt, "parameters.sources")
    print(f"   Prompt: '{sources_prompt}'")
    print(f"   Resolved: '{resolved}'")
    print(f"   Type: {type(resolved).__name__}")

    # Direct model test
    print("\n2️⃣ Testing direct model response:")
    direct_prompt = "List the best sources for healthcare AI research. Answer with comma-separated values:"
    direct_result = await model.generate(direct_prompt, max_tokens=20, temperature=0.1)
    print(f"   Prompt: '{direct_prompt}'")
    print(f"   Response: '{direct_result}'")

    return True


async def main():
    """Run AUTO resolution quality tests."""
    print("🚀 AUTO RESOLUTION QUALITY TESTING")
    print("Testing how well real models resolve AUTO tags")
    print("=" * 50)

    # Run tests
    test1_passed = await test_auto_resolution()
    test2_passed = await test_specific_resolutions()

    # Summary
    print("\n" + "=" * 50)
    print("📊 FINAL RESULTS")
    print("=" * 50)

    if test1_passed and test2_passed:
        print("✅ AUTO resolution quality is acceptable")
        print("💡 Some resolutions may need improvement")
    else:
        print("❌ AUTO resolution quality needs improvement")
        print("💡 Consider fine-tuning prompts or using a larger model")

    return test1_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
