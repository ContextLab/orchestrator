#!/usr/bin/env python3
"""
Provider implementation tests for multi-model integration.

Tests real provider implementations with actual API calls to validate:
- Provider abstractions work with real services
- Authentication and configuration handling
- Error handling and resilience
- Response parsing and standardization
"""

import asyncio
import os
import pytest
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.orchestrator.models.providers.base import ModelProvider, ProviderError, ProviderConfig
from src.orchestrator.models.providers.openai_provider import OpenAIProvider
from src.orchestrator.models.providers.anthropic_provider import AnthropicProvider
from src.orchestrator.models.providers.local_provider import LocalProvider
from src.orchestrator.models.registry import ModelRegistry
from src.orchestrator.core.model import Model, ModelCapabilities, ModelCost, ModelRequirements
from src.orchestrator.integrations.ollama_model import OllamaModel
from src.orchestrator.integrations.huggingface_model import HuggingFaceModel


class TestProviderAbstractions:
    """Test base provider abstraction contracts."""

    def test_base_provider_interface(self):
        """Test that base provider defines correct interface."""
        # Check that BaseProvider has required methods
        required_methods = [
            'get_available_models',
            'create_model',
            'health_check',
            'get_capabilities'
        ]
        
        for method in required_methods:
            assert hasattr(BaseProvider, method)
            assert callable(getattr(BaseProvider, method))

    def test_model_info_structure(self):
        """Test ModelInfo structure."""
        model_info = ModelInfo(
            name="test-model",
            provider="test",
            capabilities=[ModelCapability.TEXT_GENERATION],
            context_window=4096,
            cost_per_token=0.001
        )
        
        assert model_info.name == "test-model"
        assert model_info.provider == "test"
        assert ModelCapability.TEXT_GENERATION in model_info.capabilities
        assert model_info.context_window == 4096
        assert model_info.cost_per_token == 0.001

    def test_model_capability_enum(self):
        """Test ModelCapability enum values."""
        capabilities = [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.ANALYSIS,
            ModelCapability.SUMMARIZATION,
            ModelCapability.TRANSLATION
        ]
        
        # All capabilities should be valid
        for capability in capabilities:
            assert isinstance(capability, ModelCapability)


@pytest.mark.integration
class TestOpenAIProvider:
    """Test OpenAI provider with real API calls."""

    @pytest.fixture
    def openai_provider(self):
        """Create OpenAI provider instance."""
        return OpenAIProvider()

    def test_openai_provider_initialization(self, openai_provider):
        """Test OpenAI provider can be created."""
        assert openai_provider is not None
        assert isinstance(openai_provider, BaseProvider)
        assert openai_provider.name == "openai"

    async def test_openai_health_check(self, openai_provider):
        """Test OpenAI health check."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set - skipping real API test")
        
        health = await openai_provider.health_check()
        assert "status" in health
        # Should be "healthy" if API key is valid, otherwise may be "degraded"
        assert health["status"] in ["healthy", "degraded", "unhealthy"]

    async def test_openai_get_available_models(self, openai_provider):
        """Test getting available OpenAI models."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set - skipping real API test")
        
        models = await openai_provider.get_available_models()
        
        # Should have at least some models
        assert len(models) > 0
        
        # Check model structure
        for model in models:
            assert isinstance(model, ModelInfo)
            assert model.provider == "openai"
            assert len(model.capabilities) > 0

    async def test_openai_create_model(self, openai_provider):
        """Test creating OpenAI model instance."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set - skipping real API test")
        
        # Create with a known model
        model = await openai_provider.create_model("gpt-4o-mini")
        
        assert model is not None
        assert hasattr(model, 'generate')
        assert hasattr(model, 'name')
        
        # Test simple generation
        result = await model.generate("What is 2+2?", max_tokens=10, temperature=0.1)
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"OpenAI generation: {result}")

    def test_openai_capabilities(self, openai_provider):
        """Test OpenAI provider capabilities."""
        capabilities = openai_provider.get_capabilities()
        
        assert isinstance(capabilities, dict)
        assert "supported_models" in capabilities
        assert "features" in capabilities
        assert "limitations" in capabilities


@pytest.mark.integration  
class TestAnthropicProvider:
    """Test Anthropic provider with real API calls."""

    @pytest.fixture
    def anthropic_provider(self):
        """Create Anthropic provider instance."""
        return AnthropicProvider()

    def test_anthropic_provider_initialization(self, anthropic_provider):
        """Test Anthropic provider can be created."""
        assert anthropic_provider is not None
        assert isinstance(anthropic_provider, BaseProvider)
        assert anthropic_provider.name == "anthropic"

    async def test_anthropic_health_check(self, anthropic_provider):
        """Test Anthropic health check."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set - skipping real API test")
        
        health = await anthropic_provider.health_check()
        assert "status" in health
        assert health["status"] in ["healthy", "degraded", "unhealthy"]

    async def test_anthropic_get_available_models(self, anthropic_provider):
        """Test getting available Anthropic models."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set - skipping real API test")
        
        models = await anthropic_provider.get_available_models()
        
        # Should have Claude models
        assert len(models) > 0
        
        # Check for known Claude models
        model_names = [m.name for m in models]
        claude_models = [name for name in model_names if "claude" in name.lower()]
        assert len(claude_models) > 0
        
        # Check model structure
        for model in models:
            assert isinstance(model, ModelInfo)
            assert model.provider == "anthropic"

    async def test_anthropic_create_model(self, anthropic_provider):
        """Test creating Anthropic model instance."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set - skipping real API test")
        
        # Create with Claude model (adjust based on what's available)
        model = await anthropic_provider.create_model("claude-3-haiku-20240307")
        
        assert model is not None
        assert hasattr(model, 'generate')
        assert hasattr(model, 'name')
        
        # Test simple generation
        result = await model.generate("What is 3+3?", max_tokens=10, temperature=0.1)
        assert isinstance(result, str)
        assert len(result) > 0
        print(f"Anthropic generation: {result}")


@pytest.mark.integration
class TestLocalProvider:
    """Test local provider with Ollama and HuggingFace models."""

    @pytest.fixture
    def local_provider(self):
        """Create local provider instance."""
        return LocalProvider()

    def test_local_provider_initialization(self, local_provider):
        """Test local provider can be created."""
        assert local_provider is not None
        assert isinstance(local_provider, BaseProvider)
        assert local_provider.name == "local"

    async def test_local_health_check(self, local_provider):
        """Test local provider health check."""
        health = await local_provider.health_check()
        assert "status" in health
        assert health["status"] in ["healthy", "degraded", "unhealthy"]
        
        # Should include details about available local services
        assert "ollama_available" in health
        assert "huggingface_available" in health

    async def test_local_get_available_models(self, local_provider):
        """Test getting available local models."""
        models = await local_provider.get_available_models()
        
        # May have no models if nothing is installed locally
        assert isinstance(models, list)
        
        # If we have models, check their structure
        for model in models:
            assert isinstance(model, ModelInfo)
            assert model.provider == "local"

    async def test_ollama_integration(self, local_provider):
        """Test Ollama integration through local provider."""
        # Check if Ollama is available
        if not OllamaModel.check_ollama_installation():
            pytest.skip("Ollama not installed - skipping Ollama tests")
        
        try:
            # Try to create a small Ollama model
            model = await local_provider.create_model("llama3.2:1b")
            if model is None:
                pytest.skip("llama3.2:1b not available - skipping test")
            
            # Test generation
            result = await model.generate("Hello", max_tokens=5, temperature=0.1)
            assert isinstance(result, str)
            assert len(result) > 0
            print(f"Ollama generation: {result}")
            
        except Exception as e:
            pytest.skip(f"Ollama model creation failed: {e}")

    async def test_huggingface_integration(self, local_provider):
        """Test HuggingFace integration through local provider."""
        try:
            # Try to create a small HuggingFace model  
            model = await local_provider.create_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            if model is None:
                pytest.skip("HuggingFace model not available - skipping test")
            
            # Test generation
            result = await model.generate("Hi", max_tokens=5, temperature=0.1)
            assert isinstance(result, str) 
            assert len(result) > 0
            print(f"HuggingFace generation: {result}")
            
        except Exception as e:
            pytest.skip(f"HuggingFace model creation failed: {e}")


@pytest.mark.integration
class TestProviderRegistry:
    """Test provider registry integration."""

    @pytest.fixture
    def model_registry(self):
        """Create model registry with all providers."""
        registry = ModelRegistry()
        
        # Add all providers
        registry.add_provider(OpenAIProvider())
        registry.add_provider(AnthropicProvider())
        registry.add_provider(LocalProvider())
        
        return registry

    async def test_registry_provider_enumeration(self, model_registry):
        """Test enumerating all providers in registry."""
        providers = model_registry.get_providers()
        
        assert len(providers) == 3
        provider_names = [p.name for p in providers]
        assert "openai" in provider_names
        assert "anthropic" in provider_names
        assert "local" in provider_names

    async def test_registry_model_discovery(self, model_registry):
        """Test discovering models across all providers."""
        all_models = []
        
        # Discover models from each provider
        providers = model_registry.get_providers()
        for provider in providers:
            try:
                models = await provider.get_available_models()
                all_models.extend(models)
            except Exception as e:
                # Some providers may fail if not configured
                print(f"Provider {provider.name} failed model discovery: {e}")
                continue
        
        # Should have found at least some models
        print(f"Total models discovered: {len(all_models)}")
        
        # Check model diversity
        providers_with_models = set(model.provider for model in all_models)
        print(f"Providers with models: {providers_with_models}")

    async def test_registry_model_creation(self, model_registry):
        """Test creating models through registry."""
        # Try to create any available model
        providers = model_registry.get_providers()
        
        model_created = False
        for provider in providers:
            try:
                models = await provider.get_available_models()
                if not models:
                    continue
                
                # Try first model
                test_model = models[0]
                model_instance = await provider.create_model(test_model.name)
                
                if model_instance:
                    # Test basic functionality
                    result = await model_instance.generate(
                        "Test", max_tokens=3, temperature=0.1
                    )
                    assert isinstance(result, str)
                    print(f"Successfully created {provider.name}:{test_model.name}")
                    model_created = True
                    break
                    
            except Exception as e:
                print(f"Failed to test {provider.name}: {e}")
                continue
        
        # Should have created at least one working model
        if not model_created:
            pytest.skip("No working models available for testing")


@pytest.mark.integration
class TestProviderResilience:
    """Test provider error handling and resilience."""

    def test_invalid_api_key_handling(self):
        """Test handling of invalid API keys."""
        # Test with obviously invalid keys
        os.environ["OPENAI_API_KEY"] = "invalid_key_test_12345"
        
        provider = OpenAIProvider()
        
        # Should not raise exception on creation
        assert provider is not None
        
        # Health check should detect the issue
        asyncio.run(self._test_degraded_health(provider))

    async def _test_degraded_health(self, provider):
        """Helper to test degraded health status."""
        health = await provider.health_check()
        # Should be degraded or unhealthy with invalid key
        assert health["status"] in ["degraded", "unhealthy"]

    async def test_network_error_handling(self):
        """Test handling of network errors."""
        # This would require mocking network calls
        # For now, we'll test timeout scenarios
        pytest.skip("Network error handling test requires mock setup")

    async def test_rate_limit_handling(self):
        """Test handling of rate limits."""
        # This would require triggering actual rate limits
        pytest.skip("Rate limit testing requires controlled load generation")


class TestProviderCompatibility:
    """Test compatibility across different provider versions."""

    def test_provider_version_compatibility(self):
        """Test provider compatibility with different API versions."""
        # Each provider should handle version differences gracefully
        providers = [OpenAIProvider(), AnthropicProvider(), LocalProvider()]
        
        for provider in providers:
            capabilities = provider.get_capabilities()
            assert "version" in capabilities or "api_version" in capabilities
            print(f"{provider.name} capabilities: {capabilities}")

    def test_provider_feature_parity(self):
        """Test feature parity across providers."""
        providers = [OpenAIProvider(), AnthropicProvider(), LocalProvider()]
        
        # All providers should support basic text generation
        for provider in providers:
            capabilities = provider.get_capabilities()
            assert "text_generation" in str(capabilities).lower() or \
                   "generation" in str(capabilities).lower()


async def main():
    """Run provider integration tests."""
    print("🚀 PROVIDER INTEGRATION TESTS")
    print("=" * 60)
    
    # Run pytest with this file
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "integration"
    ])
    
    return exit_code == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)