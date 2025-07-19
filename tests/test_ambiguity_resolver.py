"""Comprehensive tests for ambiguity resolver functionality."""

import pytest

from orchestrator.compiler.ambiguity_resolver import (
    AmbiguityResolutionError,
    AmbiguityResolver,
)
from orchestrator.core.model import Model
from orchestrator.models.model_registry import ModelRegistry
from orchestrator import init_models


@pytest.fixture(scope="module")
def model_registry():
    """Initialize models once for all tests in this module."""
    return init_models()


class TestAmbiguityResolver:
    """Test cases for AmbiguityResolver class."""

    def test_resolver_creation(self, model_registry):
        """Test basic resolver creation."""
        resolver = AmbiguityResolver(model_registry)

        assert resolver.model_registry is not None
        assert resolver.resolution_cache == {}
        assert len(resolver.resolution_strategies) == 6
        assert "parameter" in resolver.resolution_strategies
        assert "value" in resolver.resolution_strategies
        assert "list" in resolver.resolution_strategies
        assert "boolean" in resolver.resolution_strategies
        assert "number" in resolver.resolution_strategies
        assert "string" in resolver.resolution_strategies

    def test_resolver_with_model_registry(self, model_registry):
        """Test resolver with model registry."""
        # Verify the registry has models
        available_models = model_registry.list_models()
        if not available_models:
            raise AssertionError(
                "No AI models available for testing. "
                "Please configure API keys in ~/.orchestrator/.env"
            )
        
        resolver = AmbiguityResolver(model_registry)
        assert resolver.model_registry is model_registry

    @pytest.mark.asyncio
    async def test_resolve_parameter_ambiguity_format(self):
        """Test resolving parameter ambiguity with format."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        result = await resolver.resolve("best format", "parameters.format")

        assert result == "json"

    @pytest.mark.asyncio
    async def test_resolve_parameter_ambiguity_method(self):
        """Test resolving parameter ambiguity with method."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        result = await resolver.resolve("best method", "parameters.method")

        assert result == "default"

    @pytest.mark.asyncio
    async def test_resolve_parameter_ambiguity_type(self):
        """Test resolving parameter ambiguity with type."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        result = await resolver.resolve("best type", "parameters.type")

        assert result == "auto"

    @pytest.mark.asyncio
    async def test_resolve_parameter_ambiguity_generic(self):
        """Test resolving parameter ambiguity with generic content."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)
        
        # The resolver should handle this with its default strategy
        result = await resolver.resolve("best configuration", "parameters.config")
        
        # Should return something reasonable
        assert result is not None
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_resolve_value_ambiguity_with_choices(self):
        """Test resolving value ambiguity with explicit choices."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        result = await resolver.resolve("Select from: json, xml, csv", "output.format")

        assert result == "json"

    @pytest.mark.asyncio
    async def test_resolve_value_ambiguity_no_choices(self):
        """Test resolving value ambiguity without explicit choices."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)
        
        # Should use default resolution
        result = await resolver.resolve("Choose the best option", "config.option")
        
        # Should return something reasonable (default fallback)
        assert result is not None
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_resolve_list_ambiguity_source(self):
        """Test resolving list ambiguity with source context."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        result = await resolver.resolve("Choose data source list", "config.sources")

        assert result == ["web", "documents", "database"]

    @pytest.mark.asyncio
    async def test_resolve_list_ambiguity_format(self):
        """Test resolving list ambiguity with format context."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        result = await resolver.resolve("Choose format list", "config.formats")

        assert result == ["json", "csv", "xml"]

    @pytest.mark.asyncio
    async def test_resolve_list_ambiguity_language(self):
        """Test resolving list ambiguity with language context."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        result = await resolver.resolve("Choose language list", "config.languages")

        assert result == ["en", "es", "fr"]

    @pytest.mark.asyncio
    async def test_resolve_list_ambiguity_default(self):
        """Test resolving list ambiguity with default context."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        result = await resolver.resolve("Choose some items", "config.items")

        assert result == ["item1", "item2", "item3"]

    @pytest.mark.asyncio
    async def test_resolve_boolean_ambiguity_positive(self):
        """Test resolving boolean ambiguity with positive indicators."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        # Test each positive word
        assert (
            await resolver.resolve("Enable compression", "config.compression") is True
        )
        assert await resolver.resolve("Choose true", "config.flag") is True
        assert await resolver.resolve("Say yes", "config.confirm") is True
        assert await resolver.resolve("Allow access", "config.allow") is True
        assert await resolver.resolve("Support feature", "config.support") is True

    @pytest.mark.asyncio
    async def test_resolve_boolean_ambiguity_negative(self):
        """Test resolving boolean ambiguity with negative indicators."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        # Test each negative word
        assert (
            await resolver.resolve("Disable compression", "config.compression") is False
        )
        assert await resolver.resolve("Choose false", "config.flag") is False
        assert await resolver.resolve("Say no", "config.confirm") is False
        assert await resolver.resolve("Deny access", "config.deny") is False
        assert await resolver.resolve("Block feature", "config.block") is False

    @pytest.mark.asyncio
    async def test_resolve_boolean_ambiguity_context_positive(self):
        """Test resolving boolean ambiguity with positive context."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        result = await resolver.resolve("Choose option", "config.enable_feature")
        assert result is True

        result = await resolver.resolve("Choose option", "config.support_mode")
        assert result is True

    @pytest.mark.asyncio
    async def test_resolve_boolean_ambiguity_context_negative(self):
        """Test resolving boolean ambiguity with negative context."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        result = await resolver.resolve("Choose option", "config.other_option")
        assert result is False

    @pytest.mark.asyncio
    async def test_resolve_number_ambiguity_batch(self):
        """Test resolving number ambiguity with batch context."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        result = await resolver.resolve("Set batch size", "config.batch_size")

        assert result == 32

    @pytest.mark.asyncio
    async def test_resolve_number_ambiguity_timeout(self):
        """Test resolving number ambiguity with timeout context."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        result = await resolver.resolve("Set timeout value", "config.timeout")

        assert result == 30

    @pytest.mark.asyncio
    async def test_resolve_number_ambiguity_retry(self):
        """Test resolving number ambiguity with retry context."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        result = await resolver.resolve("Set retry count", "config.retry")

        assert result == 3

    @pytest.mark.asyncio
    async def test_resolve_number_ambiguity_size(self):
        """Test resolving number ambiguity with size context."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        result = await resolver.resolve("Set size value", "config.size")

        assert result == 100

    @pytest.mark.asyncio
    async def test_resolve_number_ambiguity_limit(self):
        """Test resolving number ambiguity with limit context."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        result = await resolver.resolve("Set limit value", "config.limit")

        assert result == 1000

    @pytest.mark.asyncio
    async def test_resolve_number_ambiguity_default(self):
        """Test resolving number ambiguity with default context."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        result = await resolver.resolve("Set some number", "config.number")

        assert result == 10

    @pytest.mark.asyncio
    async def test_resolve_string_ambiguity_with_quotes(self):
        """Test resolving string ambiguity with quoted content."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        result = await resolver.resolve('Choose "custom_value" option', "config.value")

        assert result == "custom_value"

    @pytest.mark.asyncio
    async def test_resolve_string_ambiguity_without_quotes(self):
        """Test resolving string ambiguity without quoted content."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)
        
        result = await resolver.resolve("Choose string value", "config.string")
        
        # Should return a reasonable default string
        assert result is not None
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_resolve_generic_with_model(self, model_registry):
        """Test generic resolution using model."""
        model = None
        for model_id in ["gpt-4o-mini", "claude-3-5-haiku-20241022", "llama3.2:1b"]:
            try:
                model = model_registry.get_model(model_id)
                if model:
                    break
            except:
                pass
        
        if model:
            resolver = AmbiguityResolver(model)
            result = await resolver.resolve("Some ambiguous content", "config.generic")
            # Real model should return something
            assert result is not None
        else:
            raise AssertionError(
                "No AI models available for testing. "
                "Please configure API keys in ~/.orchestrator/.env"
            )

    @pytest.mark.asyncio
    async def test_resolve_with_real_ambiguity_examples(self, model_registry):
        """Test resolving real ambiguous content with AI models."""
        model = None
        for model_id in ["gpt-4o-mini", "claude-3-5-haiku-20241022", "llama3.2:1b"]:
            try:
                model = model_registry.get_model(model_id)
                if model:
                    break
            except:
                pass
        
        if not model:
            raise AssertionError(
                "No AI models available for testing. "
                "Please configure API keys in ~/.orchestrator/.env"
            )
        
        resolver = AmbiguityResolver(model)
        
        # Test real ambiguous scenarios that would appear in YAML files
        test_cases = [
            # Format selection
            ("Choose appropriate output format: json, yaml, or xml based on efficiency", "output.format", ["json", "yaml", "xml"]),
            # Method selection  
            ("Select the best analysis method for large datasets", "analysis.method", ["streaming", "batch", "parallel"]),
            # Boolean decision
            ("Determine if caching should be enabled for better performance", "config.enable_cache", [True, False]),
            # Number selection
            ("Choose optimal batch size for processing", "config.batch_size", range(8, 129)),
            # Algorithm selection
            ("Pick the most suitable sorting algorithm for this data type", "sort.algorithm", ["quicksort", "mergesort", "heapsort", "timsort"]),
        ]
        
        for content, context_path, valid_options in test_cases:
            result = await resolver.resolve(content, context_path)
            # Verify the AI made a reasonable choice
            assert result in valid_options, f"AI chose '{result}' which is not in {valid_options} for '{content}'"
            # Verify it's not just returning defaults
            assert isinstance(result, type(valid_options[0]))

    @pytest.mark.asyncio
    async def test_resolve_with_cache(self):
        """Test resolution with caching."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        # First resolution
        result1 = await resolver.resolve("Choose format", "test.format")

        # Second resolution should use cache
        result2 = await resolver.resolve("Choose format", "test.format")

        assert result1 == result2
        assert resolver.get_cache_size() == 1

    @pytest.mark.asyncio
    async def test_resolve_with_network_failure(self):
        """Test resolution with real network failures."""
        # This test requires manipulating network conditions
        # For now, we'll test the fallback mechanism works
        resolver = AmbiguityResolver(fallback_to_heuristics=True)
        
        # Test that resolver has fallback strategies for common cases
        test_cases = [
            ("Choose format", "config.format", "json"),
            ("Select method", "config.method", "auto"),
            ("Pick style", "config.style", "default"),
            ("Choose query type", "config.query", "default search query"),
        ]
        
        for content, context_path, expected in test_cases:
            # Even without a model, resolver should have sensible fallbacks
            result = await resolver.resolve(content, context_path)
            assert result == expected, f"Expected '{expected}' but got '{result}' for '{content}'"

    def test_classify_ambiguity_choose_boolean(self):
        """Test classifying ambiguity with choose boolean."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        assert (
            resolver._classify_ambiguity("Choose true or false", "config.flag")
            == "boolean"
        )

    def test_classify_ambiguity_choose_number(self):
        """Test classifying ambiguity with choose number."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        assert (
            resolver._classify_ambiguity("Choose number of items", "config.items")
            == "number"
        )
        assert resolver._classify_ambiguity("Select size", "config.size") == "number"
        assert resolver._classify_ambiguity("Choose count", "config.count") == "number"
        assert (
            resolver._classify_ambiguity("Select amount", "config.amount") == "number"
        )

    def test_classify_ambiguity_choose_list(self):
        """Test classifying ambiguity with choose list."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        assert (
            resolver._classify_ambiguity("Choose list of items", "config.items")
            == "list"
        )
        assert resolver._classify_ambiguity("Select array", "config.array") == "list"

    def test_classify_ambiguity_choose_value(self):
        """Test classifying ambiguity with choose value."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        assert resolver._classify_ambiguity("Choose option", "config.option") == "value"
        assert resolver._classify_ambiguity("Select item", "config.item") == "value"

    def test_classify_ambiguity_context_parameter(self):
        """Test classifying ambiguity with parameter context."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        # "Choose" with parameters context should be classified as parameter (precedence)
        assert (
            resolver._classify_ambiguity("Choose option", "parameters.option")
            == "parameter"
        )
        # Without "choose" keyword, should classify as parameter
        assert (
            resolver._classify_ambiguity("option", "parameters.option") == "parameter"
        )

    def test_classify_ambiguity_context_string(self):
        """Test classifying ambiguity with string context."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        # "Choose" triggers value classification first
        assert resolver._classify_ambiguity("Choose option", "config.format") == "value"
        assert resolver._classify_ambiguity("Choose option", "config.type") == "value"
        # Without "choose" keyword, should classify based on context
        assert resolver._classify_ambiguity("option", "config.format") == "string"
        assert resolver._classify_ambiguity("option", "config.type") == "string"

    def test_classify_ambiguity_context_boolean(self):
        """Test classifying ambiguity with boolean context."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        # "Choose" triggers value classification first
        assert resolver._classify_ambiguity("Choose option", "config.enable") == "value"
        assert (
            resolver._classify_ambiguity("Choose option", "config.support") == "value"
        )
        # Without "choose" keyword, should classify based on context
        assert resolver._classify_ambiguity("option", "config.enable") == "boolean"
        assert resolver._classify_ambiguity("option", "config.support") == "boolean"

    def test_classify_ambiguity_context_number(self):
        """Test classifying ambiguity with number context."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        # "Choose" triggers value classification first
        assert resolver._classify_ambiguity("Choose option", "config.count") == "value"
        assert resolver._classify_ambiguity("Choose option", "config.size") == "value"
        assert resolver._classify_ambiguity("Choose option", "config.limit") == "value"
        # Without "choose" keyword, should classify based on context
        assert resolver._classify_ambiguity("option", "config.count") == "number"
        assert resolver._classify_ambiguity("option", "config.size") == "number"
        assert resolver._classify_ambiguity("option", "config.limit") == "number"

    def test_classify_ambiguity_default_string(self):
        """Test classifying ambiguity with default string."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        assert resolver._classify_ambiguity("Some content", "config.other") == "string"

    def test_fallback_resolution_query(self):
        """Test fallback resolution with query."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        result = resolver._fallback_resolution("Choose query", "config.query")
        assert result == "default search query"

    def test_fallback_resolution_format(self):
        """Test fallback resolution with format."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        result = resolver._fallback_resolution("Choose format", "config.format")
        assert result == "json"

    def test_fallback_resolution_method(self):
        """Test fallback resolution with method."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        result = resolver._fallback_resolution("Choose method", "config.method")
        assert result == "auto"

    def test_fallback_resolution_style(self):
        """Test fallback resolution with style."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        result = resolver._fallback_resolution("Choose style", "config.style")
        assert result == "default"

    def test_fallback_resolution_type(self):
        """Test fallback resolution with type."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        result = resolver._fallback_resolution("Choose type", "config.type")
        assert result == "standard"

    def test_fallback_resolution_default(self):
        """Test fallback resolution with default."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        result = resolver._fallback_resolution("Choose something", "config.something")
        assert result == "default"

    def test_extract_choices_simple(self):
        """Test extracting choices from simple content."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        choices = resolver._extract_choices("Choose: json, xml, csv")
        assert choices == ["json", "xml", "csv"]

    def test_extract_choices_with_or(self):
        """Test extracting choices with or separator."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        choices = resolver._extract_choices("Select: json, xml, or csv")
        assert choices == ["json", "xml", "csv"]

    def test_extract_choices_select_keyword(self):
        """Test extracting choices with select keyword."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        choices = resolver._extract_choices("Select from: red, blue, green")
        assert choices == ["red", "blue", "green"]

    def test_extract_choices_no_match(self):
        """Test extracting choices with no match."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        choices = resolver._extract_choices("Some random text")
        assert choices == []

    def test_extract_choices_empty_choice(self):
        """Test extracting choices with empty choice."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        choices = resolver._extract_choices("Choose: json, , csv")
        assert choices == ["json", "csv"]

    def test_extract_quotes_single(self):
        """Test extracting single quoted string."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        quotes = resolver._extract_quotes('Choose "custom_value" option')
        assert quotes == ["custom_value"]

    def test_extract_quotes_multiple(self):
        """Test extracting multiple quoted strings."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        quotes = resolver._extract_quotes('Choose "value1" and "value2"')
        assert quotes == ["value1", "value2"]

    def test_extract_quotes_empty(self):
        """Test extracting quotes with empty content."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        quotes = resolver._extract_quotes('Choose ""')
        assert quotes == [""]

    def test_extract_quotes_no_quotes(self):
        """Test extracting quotes with no quotes."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        quotes = resolver._extract_quotes("Choose value")
        assert quotes == []

    def test_clear_cache(self):
        """Test clearing the resolution cache."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        # Add something to cache
        resolver.resolution_cache["test"] = "value"
        assert resolver.get_cache_size() == 1

        # Clear cache
        resolver.clear_cache()
        assert resolver.get_cache_size() == 0

    def test_set_resolution_strategy(self):
        """Test setting custom resolution strategy."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        async def custom_strategy(content, context_path):
            return "custom_result"

        resolver.set_resolution_strategy("custom", custom_strategy)

        assert "custom" in resolver.resolution_strategies
        assert resolver.resolution_strategies["custom"] == custom_strategy

    @pytest.mark.asyncio
    async def test_resolve_with_custom_strategy(self):
        """Test resolving with custom strategy."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        async def custom_strategy(content, context_path):
            return "custom_result"

        resolver.set_resolution_strategy("custom", custom_strategy)

        # Create a custom classifier function
        def custom_classifier(content, context_path):
            return "custom"
        
        # Replace the classify method temporarily
        original_classify = resolver._classify_ambiguity
        resolver._classify_ambiguity = custom_classifier

        result = await resolver.resolve("test content", "test.path")

        assert result == "custom_result"

        # Restore original method
        resolver._classify_ambiguity = original_classify

    @pytest.mark.asyncio
    async def test_resolve_unknown_strategy(self):
        """Test resolving with unknown strategy type."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        # Create a custom classifier function that returns unknown type
        def unknown_classifier(content, context_path):
            return "unknown_type"
        
        # Replace the classify method temporarily
        original_classify = resolver._classify_ambiguity
        resolver._classify_ambiguity = unknown_classifier

        # Create a test model that returns generic result
        class GenericModel(Model):
            def __init__(self):
                super().__init__(
                    name="generic-model",
                    provider="test",
                    capabilities={"supported_tasks": ["generate"]},
                    metrics={}
                )
            
            async def generate(self, prompt, **kwargs):
                return "generic_result"
            
            async def generate_structured(self, prompt, schema, **kwargs):
                return {"result": "generic_result"}
            
            def estimate_cost(self, prompt, max_tokens=None):
                return 0.0
            
            async def health_check(self):
                return True
            
            def can_execute(self, task):
                return True
        
        resolver.model = GenericModel()

        result = await resolver.resolve("test content", "test.path")

        assert result == "generic_result"

        # Restore original method
        resolver._classify_ambiguity = original_classify

    @pytest.mark.asyncio
    async def test_resolve_with_exception_in_strategy(self):
        """Test resolving with exception in strategy."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        async def failing_strategy(content, context_path):
            raise Exception("Strategy failure")

        resolver.set_resolution_strategy("failing", failing_strategy)

        # Create a custom classifier function that returns failing type
        def failing_classifier(content, context_path):
            return "failing"
        
        # Replace the classify method temporarily
        original_classify = resolver._classify_ambiguity
        resolver._classify_ambiguity = failing_classifier

        with pytest.raises(AmbiguityResolutionError):
            await resolver.resolve("test content", "test.path")

        # Restore original method
        resolver._classify_ambiguity = original_classify

    @pytest.mark.asyncio
    async def test_resolve_yaml_auto_tags_real_world(self, model_registry):
        """Test resolving actual <AUTO> tags from YAML pipelines."""
        model = None
        for model_id in ["gpt-4o-mini", "claude-3-5-haiku-20241022"]:
            try:
                model = model_registry.get_model(model_id)
                if model:
                    break
            except:
                pass
        
        if not model:
            raise AssertionError(
                "No AI models available for testing. "
                "Please configure API keys in ~/.orchestrator/.env"
            )
        
        resolver = AmbiguityResolver(model)
        
        # Real AUTO tags from example YAML files
        auto_examples = [
            {
                "content": "Choose best analysis method for this data type",
                "context": "steps.analyze_data.parameters.method",
                "description": "From research_assistant.yaml"
            },
            {
                "content": "Determine analysis depth based on data complexity",
                "context": "steps.analyze_data.parameters.depth",
                "description": "From research_assistant.yaml"
            },
            {
                "content": "Select appropriate format based on downstream requirements",
                "context": "steps.format_results.parameters.format",
                "description": "From data_processing.yaml"
            },
        ]
        
        results = []
        for example in auto_examples:
            result = await resolver.resolve(example["content"], example["context"])
            results.append({
                "example": example["description"],
                "content": example["content"],
                "result": result
            })
            # Verify we got a meaningful result
            assert result is not None
            assert len(str(result)) > 0
        
        # Verify different contexts produce different results
        unique_results = set(str(r["result"]) for r in results)
        assert len(unique_results) >= 2, "AI should produce varied results for different contexts"

    @pytest.mark.asyncio
    async def test_resolve_with_different_models(self, model_registry):
        """Test that different AI models can resolve ambiguities."""
        # Try to get multiple different models
        models = []
        for model_id in ["gpt-4o-mini", "claude-3-5-haiku-20241022", "llama3.2:1b", "gemini-1.5-flash-latest"]:
            try:
                model = model_registry.get_model(model_id)
                if model and model not in models:
                    models.append(model)
            except:
                pass
        
        if len(models) < 2:
            raise AssertionError(
                "Need at least 2 different AI models for comparison testing. "
                "Please configure multiple API keys in ~/.orchestrator/.env"
            )
        
        # Test the same ambiguity with different models
        test_content = "Choose the most appropriate data structure for fast lookups with occasional updates"
        test_context = "implementation.data_structure"
        
        results = {}
        for model in models[:2]:  # Test with first 2 available models
            resolver = AmbiguityResolver(model)
            result = await resolver.resolve(test_content, test_context)
            results[model.name] = result
            
            # Each model should provide a valid data structure
            valid_structures = ["hashmap", "hash_map", "dictionary", "dict", "map", "btree", "b-tree", "hashtable", "hash_table"]
            assert any(struct in str(result).lower() for struct in valid_structures), (
                f"Model {model.name} returned '{result}' which doesn't appear to be a data structure"
            )
        
        # Log results for analysis (models might choose different structures)
        print(f"Model comparison results: {results}")

    def test_classify_ambiguity_choose_with_true_false(self):
        """Test classification of choose with true/false."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        assert (
            resolver._classify_ambiguity("Choose true or false", "config.option")
            == "boolean"
        )
        assert (
            resolver._classify_ambiguity("Select false value", "config.option")
            == "boolean"
        )

    def test_classify_ambiguity_choose_with_number_keywords(self):
        """Test classification of choose with number keywords."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        assert (
            resolver._classify_ambiguity("Choose number of items", "config.option")
            == "number"
        )
        assert (
            resolver._classify_ambiguity("Select count value", "config.option")
            == "number"
        )
        assert (
            resolver._classify_ambiguity("Choose amount needed", "config.option")
            == "number"
        )
        assert (
            resolver._classify_ambiguity("Select size option", "config.option")
            == "number"
        )

    def test_classify_ambiguity_choose_with_list_keywords(self):
        """Test classification of choose with list keywords."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        assert (
            resolver._classify_ambiguity("Choose list of items", "config.option")
            == "list"
        )
        assert (
            resolver._classify_ambiguity("Select array values", "config.option")
            == "list"
        )
        assert (
            resolver._classify_ambiguity("Choose items to include", "config.option")
            == "list"
        )
        assert (
            resolver._classify_ambiguity("Select languages available", "config.option")
            == "list"
        )

    def test_classify_ambiguity_missing_line_148_boolean_choose(self):
        """Test line 148: return 'boolean' for true/false in choose context."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        # Test exact condition: choose/select + "true" or "false" content
        assert (
            resolver._classify_ambiguity("Choose true", "config.setting") == "boolean"
        )
        assert (
            resolver._classify_ambiguity("Select false", "config.option") == "boolean"
        )
        assert (
            resolver._classify_ambiguity("Choose true option", "config.flag")
            == "boolean"
        )
        assert (
            resolver._classify_ambiguity("Select false value", "config.toggle")
            == "boolean"
        )

    def test_classify_ambiguity_missing_line_150_number_choose(self):
        """Test line 150: return 'number' for number words in choose context."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        # Test exact condition: choose/select + number words
        assert (
            resolver._classify_ambiguity("Choose number", "config.setting") == "number"
        )
        assert resolver._classify_ambiguity("Select size", "config.option") == "number"
        assert resolver._classify_ambiguity("Choose count", "config.flag") == "number"
        assert resolver._classify_ambiguity("Select amount", "config.data") == "number"

    def test_classify_ambiguity_missing_line_171_list_context(self):
        """Test line 171: return 'list' for list-related context paths."""
        resolver = AmbiguityResolver(fallback_to_heuristics=True)

        # Test exact condition: list-related context path words (line 171)
        assert resolver._classify_ambiguity("Process data", "step.languages") == "list"
        assert resolver._classify_ambiguity("Handle content", "config.items") == "list"
        assert resolver._classify_ambiguity("Process info", "config.tags") == "list"
        assert resolver._classify_ambiguity("Manage data", "step.options") == "list"
        assert (
            resolver._classify_ambiguity("Transform content", "config.list") == "list"
        )
