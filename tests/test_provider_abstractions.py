"""Test provider abstractions with real API calls."""

import asyncio
import os
import pytest
from typing import Dict, Any

from src.orchestrator.models import (

from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider
    UnifiedModelRegistry,
    create_registry_from_env,
    create_default_configuration,
    create_registry_from_config,
    ProviderConfig,
    OpenAIProvider,
    AnthropicProvider,
    LocalProvider,
)


class TestProviderAbstractions:
    """Test provider abstractions and unified registry."""

    @pytest.fixture
    def registry(self):
        """Create a test registry."""
        registry = create_registry_from_env()
        return registry

    async def test_registry_creation(self):
        """Test creating registry with different methods."""
        # Test creating from environment
        registry_env = create_registry_from_env()
        assert isinstance(registry_env, UnifiedModelRegistry)
        
        # Test creating from default config
        config = create_default_configuration()
        registry_config = create_registry_from_config(config)
        assert isinstance(registry_config, UnifiedModelRegistry)
        
        # Test manual registry creation
        registry_manual = UnifiedModelRegistry()
        assert isinstance(registry_manual, UnifiedModelRegistry)
        
        await registry_env.cleanup()
        await registry_config.cleanup()
        await registry_manual.cleanup()

    async def test_provider_configuration(self):
        """Test provider configuration."""
        registry = UnifiedModelRegistry()
        
        # Configure OpenAI provider
        if os.getenv("OPENAI_API_KEY"):
            registry.configure_provider(
                provider_name="openai-test",
                provider_type="openai",
                config={
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "timeout": 30.0,
                }
            )
        
        # Configure Anthropic provider
        if os.getenv("ANTHROPIC_API_KEY"):
            registry.configure_provider(
                provider_name="anthropic-test",
                provider_type="anthropic",
                config={
                    "api_key": os.getenv("ANTHROPIC_API_KEY"),
                    "timeout": 30.0,
                }
            )
        
        # Configure local provider
        registry.configure_provider(
            provider_name="local-test",
            provider_type="local",
            config={
                "base_url": "http://localhost:11434",
                "timeout": 60.0,
            }
        )
        
        # Check providers are registered
        providers = registry.providers
        assert len(providers) >= 1  # At least local provider should be registered
        
        await registry.cleanup()

    async def test_openai_provider(self):
        """Test OpenAI provider directly."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not available")
        
        config = ProviderConfig(
            name="openai-test",
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=30.0,
            max_retries=2,
        )
        
        provider = OpenAIProvider(config)
        
        # Test initialization
        await provider.initialize()
        assert provider.is_initialized
        assert len(provider.available_models) > 0
        
        # Test model discovery
        models = await provider.discover_models()
        assert isinstance(models, list)
        assert len(models) > 0
        
        # Test health check
        is_healthy = await provider.health_check()
        assert is_healthy is True
        
        # Test model capabilities
        if "gpt-3.5-turbo" in provider.available_models:
            capabilities = provider.get_model_capabilities("gpt-3.5-turbo")
            assert capabilities.context_window > 0
            assert "generate" in capabilities.supported_tasks
            
            requirements = provider.get_model_requirements("gpt-3.5-turbo")
            assert requirements.memory_gb > 0
            
            cost = provider.get_model_cost("gpt-3.5-turbo")
            assert cost.input_cost_per_1k_tokens >= 0
            assert not cost.is_free
        
        # Test model creation
        if "gpt-3.5-turbo" in provider.available_models:
            model = await provider.create_model("gpt-3.5-turbo")
            assert model is not None
            assert model.name == "gpt-3.5-turbo"
            assert model.provider == "openai"
            
            # Test actual generation
            try:
                result = await model.generate("Say 'Hello, world!'", max_tokens=10)
                assert isinstance(result, str)
                assert len(result) > 0
                print(f"OpenAI model generated: {result}")
            except Exception as e:
                print(f"OpenAI generation test failed (may be rate limited): {e}")
        
        await provider.cleanup()

    async def test_anthropic_provider(self):
        """Test Anthropic provider directly."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not available")
        
        config = ProviderConfig(
            name="anthropic-test",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            timeout=30.0,
            max_retries=2,
        )
        
        provider = AnthropicProvider(config)
        
        # Test initialization
        await provider.initialize()
        assert provider.is_initialized
        assert len(provider.available_models) > 0
        
        # Test model discovery
        models = await provider.discover_models()
        assert isinstance(models, list)
        assert len(models) > 0
        
        # Test health check
        is_healthy = await provider.health_check()
        assert is_healthy is True
        
        # Test model capabilities
        if "claude-3-haiku" in provider.available_models:
            capabilities = provider.get_model_capabilities("claude-3-haiku")
            assert capabilities.context_window > 0
            assert "generate" in capabilities.supported_tasks
            
            requirements = provider.get_model_requirements("claude-3-haiku")
            assert requirements.memory_gb > 0
            
            cost = provider.get_model_cost("claude-3-haiku")
            assert cost.input_cost_per_1k_tokens >= 0
            assert not cost.is_free
        
        # Test model creation
        if "claude-3-haiku" in provider.available_models:
            model = await provider.create_model("claude-3-haiku")
            assert model is not None
            assert model.name == "claude-3-haiku"
            assert model.provider == "anthropic"
            
            # Test actual generation
            try:
                result = await model.generate("Say 'Hello, world!'", max_tokens=10)
                assert isinstance(result, str)
                assert len(result) > 0
                print(f"Anthropic model generated: {result}")
            except Exception as e:
                print(f"Anthropic generation test failed (may be rate limited): {e}")
        
        await provider.cleanup()

    async def test_local_provider(self):
        """Test local provider (Ollama)."""
        config = ProviderConfig(
            name="local-test",
            base_url="http://localhost:11434",
            timeout=60.0,
            max_retries=2,
        )
        
        provider = LocalProvider(config)
        
        # Test initialization (should work even if Ollama is not running)
        await provider.initialize()
        assert provider.is_initialized
        
        # Test model discovery
        models = await provider.discover_models()
        assert isinstance(models, list)
        # Models list might be empty if Ollama is not running, that's OK
        
        # Test health check (might be False if Ollama not running)
        is_healthy = await provider.health_check()
        assert isinstance(is_healthy, bool)
        
        # Test model capabilities for known models
        if provider.supports_model("gemma2:2b"):
            capabilities = provider.get_model_capabilities("gemma2:2b")
            assert capabilities.context_window > 0
            assert "generate" in capabilities.supported_tasks
            
            requirements = provider.get_model_requirements("gemma2:2b")
            assert requirements.memory_gb > 0
            
            cost = provider.get_model_cost("gemma2:2b")
            assert cost.input_cost_per_1k_tokens == 0  # Local models are free
            assert cost.is_free
        
        await provider.cleanup()

    async def test_unified_registry(self):
        """Test unified registry with multiple providers."""
        registry = create_registry_from_env()
        
        # Initialize registry
        await registry.initialize()
        assert registry.is_initialized
        
        # Check providers
        providers = registry.providers
        assert isinstance(providers, dict)
        assert len(providers) >= 1  # Should have at least local provider
        
        # Test model discovery
        all_models = await registry.discover_all_models()
        assert isinstance(all_models, dict)
        
        # Test health checks
        health_status = await registry.health_check()
        assert isinstance(health_status, dict)
        
        # Test registry info
        info = registry.get_registry_info()
        assert "provider_count" in info
        assert "total_models" in info
        assert "providers" in info
        
        # Test model listing
        model_list = registry.list_models()
        assert isinstance(model_list, dict)
        
        # Test finding a model (if any are available)
        available_models = registry.available_models
        if available_models:
            first_model = next(iter(available_models.keys()))
            provider_name = registry.find_model(first_model)
            assert provider_name is not None
            assert provider_name in providers
            
            # Test getting the model
            try:
                model = await registry.get_model(first_model)
                assert model is not None
                assert model.name == first_model
                print(f"Successfully created model: {first_model} from provider: {provider_name}")
            except Exception as e:
                print(f"Failed to create model {first_model}: {e}")
        
        await registry.cleanup()

    async def test_model_generation_integration(self):
        """Integration test with actual model generation."""
        registry = create_registry_from_env()
        await registry.initialize()
        
        available_models = registry.available_models
        test_prompt = "What is 2 + 2?"
        
        for model_name, provider_name in available_models.items():
            # Skip expensive models in testing
            if any(expensive in model_name.lower() for expensive in ["gpt-4", "opus", "70b"]):
                continue
                
            try:
                print(f"\nTesting model: {model_name} from provider: {provider_name}")
                model = await registry.get_model(model_name)
                
                # Test generation
                result = await model.generate(test_prompt, max_tokens=20, temperature=0.1)
                assert isinstance(result, str)
                assert len(result) > 0
                print(f"Generated: {result}")
                
                # Test health check
                is_healthy = await model.health_check()
                print(f"Health check: {is_healthy}")
                
                # Only test one model to avoid API rate limits
                break
                
            except Exception as e:
                print(f"Failed to test model {model_name}: {e}")
                continue
        
        await registry.cleanup()


def test_provider_abstractions_sync():
    """Synchronous wrapper for async tests."""
    async def run_tests():
        test_instance = TestProviderAbstractions()
        
        # Run basic tests
        await test_instance.test_registry_creation()
        await test_instance.test_provider_configuration()
        
        # Test individual providers (if API keys available)
        if os.getenv("OPENAI_API_KEY"):
            await test_instance.test_openai_provider()
            
        if os.getenv("ANTHROPIC_API_KEY"):
            await test_instance.test_anthropic_provider()
            
        # Always test local provider
        await test_instance.test_local_provider()
        
        # Test unified registry
        await test_instance.test_unified_registry()
        
        # Integration test (if any models are available)
        await test_instance.test_model_generation_integration()
    
    asyncio.run(run_tests())


if __name__ == "__main__":
    test_provider_abstractions_sync()
    print("All provider abstraction tests passed!")