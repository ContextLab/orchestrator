"""Tests for structured ambiguity resolver."""

import pytest
import logging
from orchestrator.compiler.structured_ambiguity_resolver import (
    StructuredAmbiguityResolver)
from orchestrator import init_models

# Enable debug logging for tests
logging.basicConfig(
    level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s"
)


@pytest.fixture(scope="module")
def model_registry():
    """Initialize models once for all tests."""
    return init_models()


@pytest.fixture
def structured_resolver(model_registry):
    """Create structured ambiguity resolver."""
    return StructuredAmbiguityResolver(model_registry=model_registry)


class TestStructuredAmbiguityResolver:
    """Test structured ambiguity resolution with real models."""

    @pytest.mark.asyncio
    async def test_type_inference(self, structured_resolver):
        """Test that type inference works correctly."""
        test_cases = [
            # Boolean cases
            ("Should we enable caching?", "config.enable_cache", "boolean"),
            ("Is this valid? Answer true or false", "validation.result", "boolean"),
            ("Enable parallel processing: yes or no?", "settings.parallel", "boolean"),
            # Number cases
            ("How many workers should we use?", "config.worker_count", "number"),
            ("What's the optimal batch size?", "processing.batch_size", "number"),
            ("Set timeout duration in seconds", "config.timeout", "number"),
            # List cases
            ("List the supported formats", "config.formats", "list"),
            ("Provide a comma-separated list of features", "features", "list"),
            ("What items should we include?", "config.items", "list"),
            # Choice cases
            (
                "Choose the strategy: aggressive, moderate, or conservative",
                "config.strategy",
                "choice"),
            ("Select output format: json or yaml", "output.format", "choice"),
            ("Which approach: parallel or sequential?", "processing.mode", "choice"),
            # String cases (default)
            ("Describe the process", "description", "string"),
            ("What should we name this?", "config.name", "string"),
        ]

        for content, path, expected_type in test_cases:
            inferred = structured_resolver._infer_type_from_context(content, path)
            assert (
                inferred == expected_type
            ), f"Expected {expected_type} for '{content}', got {inferred}"

    @pytest.mark.asyncio
    async def test_boolean_resolution(self, structured_resolver):
        """Test boolean value resolution."""
        test_cases = [
            (
                "Should we enable caching for better performance?",
                True),  # Positive framing
            ("Should we disable logging in production?", None),  # Model decides
            ("Is 5 greater than 3?", True),  # Factual
            ("Is 2 greater than 10?", False),  # Factual
        ]

        for content, expected in test_cases:
            result = await structured_resolver.resolve(content, "test.boolean")
            assert isinstance(result, bool), f"Expected bool, got {type(result)}"
            if expected is not None:
                assert (
                    result == expected
                ), f"Expected {expected} for '{content}', got {result}"

    @pytest.mark.asyncio
    async def test_number_resolution(self, structured_resolver):
        """Test number value resolution."""
        result = await structured_resolver.resolve(
            "For a dataset with 1000 items, what's a good batch size?",
            "config.batch_size")

        assert isinstance(result, (int, float)), f"Expected number, got {type(result)}"
        assert 1 <= result <= 1000, f"Batch size {result} seems unreasonable"

    @pytest.mark.asyncio
    async def test_list_resolution(self, structured_resolver):
        """Test list value resolution."""
        result = await structured_resolver.resolve(
            "What are the top 3 programming languages for data science?",
            "config.languages")

        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert len(result) >= 1, "Expected at least one item"
        assert all(
            isinstance(item, str) for item in result
        ), "All items should be strings"

    @pytest.mark.asyncio
    async def test_choice_resolution(self, structured_resolver):
        """Test choice selection."""
        result = await structured_resolver.resolve(
            "For high-performance computing, which is better: CPU or GPU?",
            "config.processor")

        assert isinstance(result, str), f"Expected string, got {type(result)}"
        assert result.upper() in ["CPU", "GPU"], f"Unexpected choice: {result}"

    @pytest.mark.asyncio
    async def test_cache_behavior(self, structured_resolver):
        """Test that caching works correctly."""
        content = "Should we enable debug mode?"
        path = "config.debug"

        # First call
        result1 = await structured_resolver.resolve(content, path)
        cache_size_1 = len(structured_resolver.resolution_cache)

        # Second call (should use cache)
        result2 = await structured_resolver.resolve(content, path)
        cache_size_2 = len(structured_resolver.resolution_cache)

        assert result1 == result2, "Cached result should be identical"
        assert cache_size_1 == cache_size_2, "Cache size should not increase"

    @pytest.mark.asyncio
    async def test_structured_output_fallback(self, structured_resolver):
        """Test fallback when structured output fails."""
        # Use a very ambiguous prompt that might fail structured parsing
        result = await structured_resolver.resolve(
            "What about the thing with the stuff?", "config.unknown"
        )

        # Should still return something (string by default)
        assert result is not None
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_context_path_hints(self, structured_resolver):
        """Test that context path provides type hints."""
        # Path suggests number
        result = await structured_resolver.resolve(
            "Set the value",  # Ambiguous content
            "config.max_count",  # Path hints at number
        )
        assert isinstance(
            result, (int, float)
        ), "Path 'max_count' should hint at number type"

        # Path suggests boolean
        result = await structured_resolver.resolve(
            "Configure this setting",  # Ambiguous content
            "config.enable_feature",  # Path hints at boolean
        )
        assert isinstance(
            result, bool
        ), "Path 'enable_feature' should hint at boolean type"

    @pytest.mark.asyncio
    async def test_error_handling(self, structured_resolver):
        """Test error handling for invalid inputs."""
        with pytest.raises(Exception):  # Could be AmbiguityResolutionError or TypeError
            await structured_resolver.resolve(None, "test.path")

        with pytest.raises(Exception):
            await structured_resolver.resolve("", None)
