"""Tests for ambiguity resolver functionality with real AI models."""

import pytest

from orchestrator.compiler.ambiguity_resolver import (
    AmbiguityResolutionError,
    AmbiguityResolver,
)
from orchestrator import init_models


@pytest.fixture(scope="module")
def model_registry():
    """Initialize models once for all tests in this module."""
    return init_models()


@pytest.fixture
def test_model(model_registry):
    """Get a test model for use in tests."""
    model = None
    for model_id in ["gpt-4o-mini", "claude-3-5-haiku-20241022", "gemini-2.0-flash-lite"]:
        try:
            model = model_registry.get_model(model_id)
            if model:
                break
        except:
            pass

    if not model:
        pytest.skip("No AI models available for testing")

    return model


class TestAmbiguityResolver:
    """Test cases for AmbiguityResolver class using real AI models."""

    def test_resolver_creation_with_model(self, test_model):
        """Test basic resolver creation with a real model."""
        resolver = AmbiguityResolver(model=test_model)
        assert resolver.model is not None
        assert resolver.resolution_cache == {}

    @pytest.mark.asyncio
    async def test_resolver_with_model_registry(self, model_registry):
        """Test resolver with model registry."""
        # Verify the registry has models
        available_models = model_registry.list_models()
        if not available_models:
            pytest.skip("No AI models available for testing")

        resolver = AmbiguityResolver(model_registry=model_registry)
        assert resolver.model is None  # Model is selected lazily
        assert resolver.model_registry is model_registry
        
        # Trigger model selection by making a resolution
        # Use a clearer prompt that's more likely to get a direct answer
        result = await resolver.resolve("Select either 'option1' or 'option2'. Reply with only the option name.", "test.choice")
        
        # Now the model should be selected
        assert resolver.model is not None
        # Accept any result that contains option1 or option2
        assert "option1" in result.lower() or "option2" in result.lower() or result == ""

    def test_resolver_without_model_fails(self):
        """Test that resolver fails without a model."""
        with pytest.raises(ValueError, match="No AI model available"):
            AmbiguityResolver()

    @pytest.mark.asyncio
    async def test_resolve_format_ambiguity(self, test_model):
        """Test resolving format selection with real AI."""
        resolver = AmbiguityResolver(model=test_model)

        result = await resolver.resolve(
            "Choose the best output format for structured data: json, yaml, or xml", "output.format"
        )

        # AI should choose one of the valid formats
        assert result in ["json", "yaml", "xml"]

    @pytest.mark.asyncio
    async def test_resolve_boolean_ambiguity(self, test_model):
        """Test resolving boolean decisions with real AI."""
        resolver = AmbiguityResolver(model=test_model)

        result = await resolver.resolve(
            "Should we enable caching for better performance? Consider memory constraints.",
            "config.enable_cache",
        )

        # AI should return a boolean
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_resolve_number_ambiguity(self, test_model):
        """Test resolving numeric values with real AI."""
        resolver = AmbiguityResolver(model=test_model)

        result = await resolver.resolve(
            "Choose an optimal batch size between 8 and 128 for processing large datasets",
            "config.batch_size",
        )

        # AI should return a reasonable number
        assert isinstance(result, (int, float))
        assert 8 <= result <= 128

    @pytest.mark.asyncio
    async def test_resolve_list_ambiguity(self, test_model):
        """Test resolving list selections with real AI."""
        resolver = AmbiguityResolver(model=test_model)

        result = await resolver.resolve(
            "Select the most important programming languages to support: python, javascript, java, go, rust, c++",
            "config.languages",
        )

        # AI might return a list or comma-separated string
        if isinstance(result, str):
            # Convert comma-separated string to list
            result = [lang.strip() for lang in result.split(",")]

        assert isinstance(result, list)
        assert len(result) > 0
        # Check that returned items are from the provided options
        valid_langs = ["python", "javascript", "java", "go", "rust", "c++"]
        for lang in result:
            assert lang.lower() in valid_langs

    @pytest.mark.asyncio
    async def test_resolve_with_cache(self, test_model):
        """Test that resolution results are cached."""
        resolver = AmbiguityResolver(model=test_model)

        # First resolution
        content = "Choose the best compression algorithm"
        context = "config.compression"
        result1 = await resolver.resolve(content, context)

        # Check cache was populated
        cache_key = f"{content}:{context}"
        assert cache_key in resolver.resolution_cache
        assert resolver.get_cache_size() == 1

        # Second resolution should use cache
        result2 = await resolver.resolve(content, context)
        assert result1 == result2
        assert resolver.get_cache_size() == 1  # Still only 1 entry

    @pytest.mark.asyncio
    async def test_resolve_real_yaml_examples(self, test_model):
        """Test resolving actual AUTO tag scenarios from YAML files."""
        resolver = AmbiguityResolver(model=test_model)

        # Real examples from YAML pipelines
        test_cases = [
            {
                "content": "Choose best analysis method for this data type",
                "context": "steps.analyze_data.parameters.method",
                "valid_options": [
                    "statistical",
                    "machine_learning",
                    "deep_learning",
                    "hybrid",
                    "basic",
                    "advanced",
                ],
            },
            {
                "content": "Determine appropriate timeout in seconds for network requests",
                "context": "steps.fetch_data.parameters.timeout",
                "valid_range": (5, 300),
            },
            {
                "content": "Select data validation strategy: strict, moderate, or lenient",
                "context": "steps.validate.parameters.strategy",
                "valid_options": ["strict", "moderate", "lenient"],
            },
        ]

        for test in test_cases:
            result = await resolver.resolve(test["content"], test["context"])

            if "valid_options" in test:
                # For choice-based ambiguities
                assert any(
                    opt in str(result).lower() for opt in test["valid_options"]
                ), f"AI chose '{result}' which doesn't match any of {test['valid_options']}"
            elif "valid_range" in test:
                # For numeric ambiguities
                assert isinstance(result, (int, float))
                assert (
                    test["valid_range"][0] <= result <= test["valid_range"][1]
                ), f"AI chose {result} which is outside range {test['valid_range']}"

    @pytest.mark.asyncio
    async def test_resolve_algorithm_selection(self, test_model):
        """Test AI's ability to select appropriate algorithms."""
        resolver = AmbiguityResolver(model=test_model)

        result = await resolver.resolve(
            "Choose the most suitable sorting algorithm for a nearly sorted large dataset",
            "algorithm.sort",
        )

        # AI should choose an algorithm suitable for nearly sorted data
        # Good choices would be insertion sort, timsort, or similar
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_clear_cache(self, test_model):
        """Test clearing the resolution cache."""
        resolver = AmbiguityResolver(model=test_model)

        # Add something to cache
        await resolver.resolve("Test content", "test.path")
        assert resolver.get_cache_size() > 0

        # Clear cache
        resolver.clear_cache()
        assert resolver.get_cache_size() == 0

    @pytest.mark.asyncio
    async def test_resolve_with_different_models(self, model_registry):
        """Test that different models can resolve ambiguities."""
        models = []
        for model_id in ["gpt-4o-mini", "claude-3-5-haiku-20241022", "gemini-2.0-flash-lite"]:
            try:
                model = model_registry.get_model(model_id)
                if model:
                    models.append(model)
                if len(models) >= 2:
                    break
            except:
                pass

        if len(models) < 2:
            pytest.skip("Need at least 2 different AI models for comparison")

        # Test the same ambiguity with different models
        test_content = "Choose the most appropriate data structure for fast lookups"
        test_context = "implementation.data_structure"

        results = {}
        for model in models:
            resolver = AmbiguityResolver(model=model)
            result = await resolver.resolve(test_content, test_context)
            results[model.name] = result

        # Each model should provide a valid answer
        for model_name, result in results.items():
            assert result is not None
            assert len(str(result)) > 0

    @pytest.mark.asyncio
    async def test_error_handling(self, test_model):
        """Test error handling in ambiguity resolution."""
        resolver = AmbiguityResolver(model=test_model)

        # Test with None content - this should raise TypeError
        with pytest.raises((AmbiguityResolutionError, TypeError)):
            await resolver.resolve(None, "test.path")

    @pytest.mark.asyncio
    async def test_complex_ambiguity_resolution(self, test_model):
        """Test resolving complex, multi-faceted ambiguities."""
        resolver = AmbiguityResolver(model=test_model)

        # Complex scenario requiring AI reasoning
        result = await resolver.resolve(
            "For a web application expecting 1000-10000 concurrent users with read-heavy workload, "
            "choose the most appropriate database system considering scalability, consistency, and cost",
            "infrastructure.database",
        )

        # AI should provide a thoughtful response
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_context_aware_resolution(self, test_model):
        """Test that AI considers context path in resolution."""
        resolver = AmbiguityResolver(model=test_model)

        # Same content, different contexts should potentially yield different results
        content = "Choose the best approach"

        security_result = await resolver.resolve(content, "security.encryption.method")
        performance_result = await resolver.resolve(content, "performance.caching.method")

        # Both should be valid responses
        assert security_result is not None
        assert performance_result is not None

        # Results might differ based on context (though not guaranteed)
        # The important thing is that both are reasonable for their contexts
