#!/usr/bin/env python3
"""Local Ollama testing - only runs when Ollama is available."""

import sys
import subprocess
import pytest

# Mark all tests in this file as local-only (not run in CI)
pytestmark = pytest.mark.local


def check_ollama_available():
    """Check if Ollama is available and has models."""
    try:
        # Check if ollama command exists
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            # Check for specific models
            output = result.stdout
            available_models = []
            for line in output.split("\n")[1:]:  # Skip header
                if line.strip():
                    model_name = line.split()[0]
                    available_models.append(model_name)
            return available_models
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return []


# Skip all tests if Ollama not available
available_models = check_ollama_available()
skip_reason = "Ollama not available or no models installed"
if not available_models:
    pytestmark = pytest.mark.skip(reason=skip_reason)


class TestOllamaIntegration:
    """Test Ollama model integration when available."""

    @pytest.mark.asyncio
    async def test_ollama_model_creation(self):
        """Test creating Ollama model."""
        from orchestrator.integrations.ollama_model import OllamaModel

        # Use first available model
        model_name = available_models[0] if available_models else "llama3.2:1b"
        model = OllamaModel(model_name=model_name)

        assert model.name == model_name
        assert model.provider == "ollama"
        assert model._is_available

    @pytest.mark.asyncio
    async def test_ollama_generation(self):
        """Test Ollama text generation."""
        from orchestrator.integrations.ollama_model import OllamaModel

        model_name = available_models[0] if available_models else "llama3.2:1b"
        model = OllamaModel(model_name=model_name, timeout=30)

        result = await model.generate("2+2=", max_tokens=5, temperature=0.0)
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_ollama_health_check(self):
        """Test Ollama health check."""
        from orchestrator.integrations.ollama_model import OllamaModel

        model_name = available_models[0] if available_models else "llama3.2:1b"
        model = OllamaModel(model_name=model_name)

        healthy = await model.health_check()
        assert healthy is True

    @pytest.mark.asyncio
    async def test_ambiguity_resolver_with_ollama(self):
        """Test ambiguity resolver with Ollama model."""
        from orchestrator.integrations.ollama_model import OllamaModel
        from orchestrator.compiler.ambiguity_resolver import AmbiguityResolver

        model_name = available_models[0] if available_models else "llama3.2:1b"
        model = OllamaModel(model_name=model_name, timeout=20)
        resolver = AmbiguityResolver(model=model)

        # Test simple resolution
        result = await resolver.resolve("Choose format", "test.format")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_auto_model_detection(self):
        """Test automatic Ollama model detection."""
        from orchestrator.compiler.ambiguity_resolver import AmbiguityResolver
        from orchestrator.integrations.ollama_model import OllamaModel
        from orchestrator.models.model_registry import ModelRegistry

        # Create a model registry and register an Ollama model
        registry = ModelRegistry()
        model_name = available_models[0] if available_models else "llama3.2:1b"
        model = OllamaModel(name=model_name, model_size="1b")
        registry.register_model(model)

        # Create resolver with registry
        resolver = AmbiguityResolver(model_registry=registry)
        assert resolver.model is None  # Model selected lazily
        assert resolver.model_registry is registry

        # Test resolution works - this will trigger model selection
        result = await resolver.resolve("json or csv", "data.format")
        assert isinstance(result, str)
        assert len(result) > 0

        # Now model should be selected
        assert resolver.model is not None
        assert resolver.model.provider == "ollama"


@pytest.mark.asyncio
async def test_performance_comparison():
    """Compare performance of different available models."""
    if len(available_models) < 2:
        pytest.skip("Need at least 2 models for comparison")

    from orchestrator.integrations.ollama_model import OllamaModel
    import time

    results = {}

    for model_name in available_models[:3]:  # Test up to 3 models
        model = OllamaModel(model_name=model_name, timeout=30)

        start_time = time.time()
        try:
            result = await model.generate("Hello", max_tokens=5, temperature=0.0)
            duration = time.time() - start_time
            results[model_name] = {
                "duration": duration,
                "success": True,
                "result": result,
            }
        except Exception as e:
            duration = time.time() - start_time
            results[model_name] = {
                "duration": duration,
                "success": False,
                "error": str(e),
            }

    print("\n🏁 Performance Results:")
    for model, data in results.items():
        if data["success"]:
            print(f"✅ {model}: {data['duration']:.2f}s - '{data['result']}'")
        else:
            print(f"❌ {model}: {data['duration']:.2f}s - {data['error']}")

    # At least one model should work
    assert any(data["success"] for data in results.values())


if __name__ == "__main__":
    if not available_models:
        print("⚠️ Ollama not available - skipping tests")
        print("💡 To run these tests:")
        print("   1. Install Ollama: https://ollama.ai")
        print("   2. Pull a model: ollama pull llama3.2:1b")
        print("   3. Run tests: python test_ollama_local.py")
        sys.exit(0)

    print(f"🦙 Available Ollama models: {', '.join(available_models)}")
    print("🧪 Running Ollama integration tests...")

    # Run tests manually
    import pytest

    exit_code = pytest.main([__file__, "-v"])
    sys.exit(exit_code)
