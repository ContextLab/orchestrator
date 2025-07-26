"""Tests for ambiguity resolver error handling and retry logic."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from orchestrator.compiler.ambiguity_resolver import (
    AmbiguityResolver,
    AmbiguityResolutionError,
)
from orchestrator.compiler.structured_ambiguity_resolver import (
    StructuredAmbiguityResolver,
)
from orchestrator.compiler.utils import is_transient_error


class TestErrorHandling:
    """Test error handling in ambiguity resolvers."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = MagicMock()
        model.name = "test-model"
        model.generate = AsyncMock()
        model.generate_structured = AsyncMock()
        model.capabilities = MagicMock()
        model.capabilities.supports_structured_output = True
        return model

    @pytest.fixture
    def mock_registry(self, mock_model):
        """Create a mock model registry."""
        registry = MagicMock()
        registry.select_model = AsyncMock(return_value=mock_model)
        registry.get_model = MagicMock(return_value=mock_model)
        registry.list_models = MagicMock(return_value=["test-model"])
        return registry

    def test_no_model_available_error(self):
        """Test error when no model is available at initialization."""
        # Create a registry with no models
        empty_registry = MagicMock()
        empty_registry.list_models = MagicMock(return_value=[])

        # Should raise ValueError at initialization
        with pytest.raises(ValueError, match="No AI model available"):
            AmbiguityResolver(model_registry=empty_registry)

    @pytest.mark.asyncio
    async def test_model_becomes_unavailable(self, mock_registry):
        """Test error when model becomes unavailable after initialization."""
        resolver = AmbiguityResolver(model_registry=mock_registry)

        # Make model selection fail
        mock_registry.select_model.side_effect = Exception("Model unavailable")
        mock_registry.list_models.return_value = []
        mock_registry.get_model.return_value = None

        with pytest.raises(AmbiguityResolutionError, match="No AI model available"):
            await resolver.resolve("test content", "test.path")

    @pytest.mark.asyncio
    async def test_model_failure_with_retry(self, mock_registry, mock_model):
        """Test that model failures trigger retries."""
        resolver = AmbiguityResolver(model_registry=mock_registry)

        # Make model fail twice then succeed
        mock_model.generate.side_effect = [
            Exception("Network error"),
            Exception("Timeout error"),
            "Success",
        ]

        result = await resolver.resolve("test content", "test.path")

        # Should have been called 3 times due to retries
        assert mock_model.generate.call_count == 3
        assert result == "Success"

    @pytest.mark.asyncio
    async def test_permanent_failure_after_retries(self, mock_registry, mock_model):
        """Test that permanent failures raise after all retries."""
        resolver = AmbiguityResolver(model_registry=mock_registry)

        # Make model always fail
        mock_model.generate.side_effect = Exception("Permanent error")

        with pytest.raises(AmbiguityResolutionError, match="Permanent error"):
            await resolver.resolve("test content", "test.path")

        # Should have been called 3 times (initial + 2 retries)
        assert mock_model.generate.call_count == 3

    @pytest.mark.asyncio
    async def test_structured_resolver_fallback(self, mock_registry, mock_model):
        """Test that structured resolver falls back to parsing on failure."""
        resolver = StructuredAmbiguityResolver(model_registry=mock_registry)

        # Make structured call fail, regular call succeed
        mock_model.generate_structured.side_effect = Exception(
            "Structured not supported"
        )
        mock_model.generate.return_value = "true"

        result = await resolver.resolve("Should we enable caching?", "test.boolean")

        # Should have fallen back to regular generate
        assert mock_model.generate_structured.call_count >= 1
        assert mock_model.generate.call_count >= 1
        assert result is True

    @pytest.mark.asyncio
    async def test_both_methods_fail(self, mock_registry, mock_model):
        """Test error when both structured and fallback methods fail."""
        resolver = StructuredAmbiguityResolver(model_registry=mock_registry)

        # Make both methods fail
        mock_model.generate_structured.side_effect = Exception("Structured failed")
        mock_model.generate.side_effect = Exception("Generate failed")

        with pytest.raises(
            AmbiguityResolutionError,
            match="Both structured output and fallback parsing failed",
        ):
            await resolver.resolve("Should we enable caching?", "test.boolean")

    def test_transient_error_detection(self):
        """Test detection of transient errors."""
        # Transient errors
        assert is_transient_error(Exception("Connection timeout"))
        assert is_transient_error(Exception("Network error"))
        assert is_transient_error(Exception("Rate limit exceeded"))
        assert is_transient_error(Exception("429 Too Many Requests"))
        assert is_transient_error(Exception("503 Service Unavailable"))
        assert is_transient_error(Exception("Request throttled"))

        # Non-transient errors
        assert not is_transient_error(Exception("Invalid API key"))
        assert not is_transient_error(Exception("Model not found"))
        assert not is_transient_error(Exception("Syntax error"))

    @pytest.mark.asyncio
    async def test_cache_prevents_retries(self, mock_registry, mock_model):
        """Test that cached results prevent unnecessary retries."""
        resolver = AmbiguityResolver(model_registry=mock_registry)

        # First call succeeds
        mock_model.generate.return_value = "cached result"
        result1 = await resolver.resolve("test content", "test.path")

        # Reset mock to fail
        mock_model.generate.reset_mock()
        mock_model.generate.side_effect = Exception("Should not be called")

        # Second call should use cache
        result2 = await resolver.resolve("test content", "test.path")

        assert result1 == result2 == "cached result"
        assert mock_model.generate.call_count == 0  # Not called due to cache

    @pytest.mark.asyncio
    async def test_model_selection_fallback(self, mock_registry):
        """Test fallback when model selection fails."""
        resolver = AmbiguityResolver(model_registry=mock_registry)

        # Make select_model fail but list_models succeed
        mock_registry.select_model.side_effect = Exception("No suitable model")
        mock_model = MagicMock()
        mock_model.generate = AsyncMock(return_value="fallback result")
        mock_registry.get_model.return_value = mock_model

        result = await resolver.resolve("test content", "test.path")

        assert result == "fallback result"
        assert mock_registry.select_model.called
        assert mock_registry.get_model.called

    @pytest.mark.asyncio
    async def test_retry_with_backoff(self, mock_registry, mock_model):
        """Test that retries use exponential backoff."""
        resolver = AmbiguityResolver(model_registry=mock_registry)

        # Track timing
        import time

        call_times = []

        async def mock_generate(*args, **kwargs):
            call_times.append(time.time())
            if len(call_times) < 3:
                raise Exception("Transient error")
            return "Success"

        mock_model.generate = mock_generate

        start = time.time()
        result = await resolver.resolve("test content", "test.path")

        assert result == "Success"
        assert len(call_times) == 3

        # Check delays are increasing (with some tolerance)
        if len(call_times) >= 3:
            delay1 = call_times[1] - call_times[0]
            delay2 = call_times[2] - call_times[1]
            assert delay2 > delay1 * 1.5  # Backoff factor
