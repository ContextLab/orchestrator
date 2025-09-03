"""Tests for ambiguity resolver error handling and retry logic with real models."""

import pytest
import asyncio
from src.orchestrator.compiler.ambiguity_resolver import (
    AmbiguityResolver,
    AmbiguityResolutionError)
from src.orchestrator.compiler.structured_ambiguity_resolver import (
    StructuredAmbiguityResolver)
from src.orchestrator.compiler.utils import is_transient_error
from src.orchestrator.models.registry_singleton import get_model_registry
from orchestrator import init_models


class TestErrorHandling:
    """Test error handling in ambiguity resolvers with real models."""

    @pytest.fixture(scope="class")
    def real_registry(self):
        """Get real model registry."""
        # Initialize models if not already done
        try:
            init_models()
        except Exception:
            pass  # May already be initialized
        
        registry = get_model_registry()
        if not registry.list_models():
            pytest.skip("No models available for testing")
        return registry

    @pytest.fixture
    def real_model(self, real_registry):
        """Get a real model for testing."""
        models = real_registry.list_models()
        if not models:
            pytest.skip("No models available")
        
        # Try to get a small, fast model
        preferred_models = [
            "openai:gpt-4.1-nano",
            "openai:gpt-4o-mini", 
            "anthropic:claude-3-5-haiku-20241022",
            "google:gemini-2.0-flash-lite"
        ]
        
        model_name = None
        for preferred in preferred_models:
            if preferred in models:
                model_name = preferred
                break
        
        if not model_name:
            model_name = models[0]  # Use first available
        
        return real_registry.get_model(model_name)

    def test_no_model_available_error_real(self):
        """Test error when no model is available at initialization."""
        # Create a registry with no models by passing an empty dict
        from src.orchestrator.models.model_registry import ModelRegistry
        empty_registry = ModelRegistry()
        
        # Should raise ValueError at initialization
        with pytest.raises(ValueError, match="No AI model available"):
            AmbiguityResolver(model_registry=empty_registry)

    @pytest.mark.asyncio
    async def test_model_resolution_real(self, real_registry, real_model):
        """Test real model resolution."""
        resolver = AmbiguityResolver(model=real_model)
        
        # Test with a simple ambiguity
        result = await resolver.resolve("Choose a format: json or yaml", "output.format")
        
        # Should resolve to one of the options
        assert result.lower() in ["json", "yaml", "format"]

    @pytest.mark.asyncio
    async def test_cache_behavior_real(self, real_registry, real_model):
        """Test that caching works with real models."""
        resolver = AmbiguityResolver(model=real_model)
        
        # First call - should hit the model
        content = "Select output format"
        context = "config.output.format"
        
        result1 = await resolver.resolve(content, context)
        
        # Second call with same inputs - should use cache
        result2 = await resolver.resolve(content, context)
        
        # Results should be identical (from cache)
        assert result1 == result2
        
        # Clear cache
        resolver.clear_cache()
        
        # Third call - should hit model again
        result3 = await resolver.resolve(content, context)
        
        # Result might be different since it's a new model call
        assert isinstance(result3, str)

    @pytest.mark.asyncio
    async def test_invalid_prompt_handling_real(self, real_registry, real_model):
        """Test handling of invalid or nonsensical prompts."""
        resolver = AmbiguityResolver(model=real_model)
        
        # Test with empty content
        try:
            result = await resolver.resolve("", "test.path")
            # Model might still return something
            assert isinstance(result, str)
        except AmbiguityResolutionError:
            # This is also acceptable
            pass

    @pytest.mark.asyncio
    async def test_structured_resolver_real(self, real_registry):
        """Test structured resolver with real models."""
        # Get a model that supports structured output
        models = real_registry.list_models()
        
        # Find a model with structured output support
        structured_model = None
        for model_name in models:
            model = real_registry.get_model(model_name)
            if hasattr(model, 'capabilities') and model.capabilities.supports_structured_output:
                structured_model = model
                break
        
        if not structured_model:
            # Fall back to regular resolver
            resolver = StructuredAmbiguityResolver(model_registry=real_registry)
        else:
            resolver = StructuredAmbiguityResolver(model=structured_model)
        
        # Test boolean resolution
        result = await resolver.resolve("Enable feature?", "config.feature.enabled")
        assert isinstance(result, (bool, str))
        
        # Test list resolution  
        result = await resolver.resolve("List supported formats", "config.formats")
        assert isinstance(result, (list, str))

    @pytest.mark.asyncio
    async def test_model_switching_real(self, real_registry):
        """Test switching between different real models."""
        models = real_registry.list_models()
        if len(models) < 2:
            pytest.skip("Need at least 2 models for this test")
        
        # Create resolver with first model
        model1 = real_registry.get_model(models[0])
        resolver = AmbiguityResolver(model=model1)
        
        result1 = await resolver.resolve("Pick a number", "config.number")
        
        # Switch to second model
        model2 = real_registry.get_model(models[1])
        resolver.model = model2
        resolver.clear_cache()  # Clear cache to ensure new model is used
        
        result2 = await resolver.resolve("Pick a number", "config.number")
        
        # Both should return valid results
        assert isinstance(result1, (str, int, float))
        assert isinstance(result2, (str, int, float))

    @pytest.mark.asyncio
    async def test_concurrent_resolutions_real(self, real_registry, real_model):
        """Test concurrent resolution requests with real models."""
        resolver = AmbiguityResolver(model=real_model)
        
        # Create multiple resolution tasks
        tasks = []
        prompts = [
            ("Choose format", "config.format"),
            ("Select mode", "config.mode"),
            ("Pick color", "config.color"),
        ]
        
        for content, context in prompts:
            task = resolver.resolve(content, context)
            tasks.append(task)
        
        # Run concurrently
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(results) == 3
        for result in results:
            assert isinstance(result, str)

    def test_is_transient_error_real(self):
        """Test transient error detection with real error types."""
        # Network errors are transient
        assert is_transient_error(ConnectionError("Network unreachable"))
        assert is_transient_error(TimeoutError("Request timed out"))
        
        # API errors might be transient
        assert is_transient_error(Exception("rate limit exceeded"))
        assert is_transient_error(Exception("Service temporarily unavailable"))
        
        # Programming errors are not transient
        assert not is_transient_error(ValueError("Invalid argument"))
        assert not is_transient_error(TypeError("Wrong type"))
        assert not is_transient_error(AttributeError("No such attribute"))

    @pytest.mark.asyncio
    async def test_long_context_handling_real(self, real_registry, real_model):
        """Test handling of long context with real models."""
        resolver = AmbiguityResolver(model=real_model)
        
        # Create a long context
        long_content = "Choose the best option: " + ", ".join([f"option{i}" for i in range(50)])
        
        result = await resolver.resolve(long_content, "config.selection")
        
        # Should still work
        assert isinstance(result, str)