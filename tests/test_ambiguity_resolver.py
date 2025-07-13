"""Comprehensive tests for ambiguity resolver functionality."""

import pytest
import asyncio
from unittest.mock import AsyncMock

from orchestrator.compiler.ambiguity_resolver import AmbiguityResolver, AmbiguityResolutionError
from orchestrator.core.model import MockModel


class TestAmbiguityResolver:
    """Test cases for AmbiguityResolver class."""
    
    def test_resolver_creation(self):
        """Test basic resolver creation."""
        resolver = AmbiguityResolver()
        
        assert resolver.model is not None
        assert resolver.resolution_cache == {}
        assert len(resolver.resolution_strategies) == 6
        assert "parameter" in resolver.resolution_strategies
        assert "value" in resolver.resolution_strategies
        assert "list" in resolver.resolution_strategies
        assert "boolean" in resolver.resolution_strategies
        assert "number" in resolver.resolution_strategies
        assert "string" in resolver.resolution_strategies
    
    def test_resolver_with_model(self):
        """Test resolver with specific model."""
        model = MockModel()
        resolver = AmbiguityResolver(model)
        
        assert resolver.model is model
    
    @pytest.mark.asyncio
    async def test_resolve_parameter_ambiguity_format(self):
        """Test resolving parameter ambiguity with format."""
        resolver = AmbiguityResolver()
        
        result = await resolver.resolve("best format", "parameters.format")
        
        assert result == "json"
    
    @pytest.mark.asyncio
    async def test_resolve_parameter_ambiguity_method(self):
        """Test resolving parameter ambiguity with method."""
        resolver = AmbiguityResolver()
        
        result = await resolver.resolve("best method", "parameters.method")
        
        assert result == "default"
    
    @pytest.mark.asyncio
    async def test_resolve_parameter_ambiguity_type(self):
        """Test resolving parameter ambiguity with type."""
        resolver = AmbiguityResolver()
        
        result = await resolver.resolve("best type", "parameters.type")
        
        assert result == "auto"
    
    @pytest.mark.asyncio
    async def test_resolve_parameter_ambiguity_generic(self):
        """Test resolving parameter ambiguity with generic content."""
        model = MockModel()
        model.set_response("best configuration", "default_config")
        resolver = AmbiguityResolver(model)
        
        result = await resolver.resolve("best configuration", "parameters.config")
        
        assert result == "default_config"
    
    @pytest.mark.asyncio
    async def test_resolve_value_ambiguity_with_choices(self):
        """Test resolving value ambiguity with explicit choices."""
        resolver = AmbiguityResolver()
        
        result = await resolver.resolve("Select from: json, xml, csv", "output.format")
        
        assert result == "json"
    
    @pytest.mark.asyncio
    async def test_resolve_value_ambiguity_no_choices(self):
        """Test resolving value ambiguity without explicit choices."""
        model = MockModel()
        model.set_response("Choose the best option", "optimal_choice")
        resolver = AmbiguityResolver(model)
        
        result = await resolver.resolve("Choose the best option", "config.option")
        
        assert result == "optimal_choice"
    
    @pytest.mark.asyncio
    async def test_resolve_list_ambiguity_source(self):
        """Test resolving list ambiguity with source context."""
        resolver = AmbiguityResolver()
        
        result = await resolver.resolve("Choose data source list", "config.sources")
        
        assert result == ["web", "documents", "database"]
    
    @pytest.mark.asyncio
    async def test_resolve_list_ambiguity_format(self):
        """Test resolving list ambiguity with format context."""
        resolver = AmbiguityResolver()
        
        result = await resolver.resolve("Choose format list", "config.formats")
        
        assert result == ["json", "csv", "xml"]
    
    @pytest.mark.asyncio
    async def test_resolve_list_ambiguity_language(self):
        """Test resolving list ambiguity with language context."""
        resolver = AmbiguityResolver()
        
        result = await resolver.resolve("Choose language list", "config.languages")
        
        assert result == ["en", "es", "fr"]
    
    @pytest.mark.asyncio
    async def test_resolve_list_ambiguity_default(self):
        """Test resolving list ambiguity with default context."""
        resolver = AmbiguityResolver()
        
        result = await resolver.resolve("Choose some items", "config.items")
        
        assert result == ["item1", "item2", "item3"]
    
    @pytest.mark.asyncio
    async def test_resolve_boolean_ambiguity_positive(self):
        """Test resolving boolean ambiguity with positive indicators."""
        resolver = AmbiguityResolver()
        
        # Test each positive word
        assert await resolver.resolve("Enable compression", "config.compression") is True
        assert await resolver.resolve("Choose true", "config.flag") is True
        assert await resolver.resolve("Say yes", "config.confirm") is True
        assert await resolver.resolve("Allow access", "config.allow") is True
        assert await resolver.resolve("Support feature", "config.support") is True
    
    @pytest.mark.asyncio
    async def test_resolve_boolean_ambiguity_negative(self):
        """Test resolving boolean ambiguity with negative indicators."""
        resolver = AmbiguityResolver()
        
        # Test each negative word
        assert await resolver.resolve("Disable compression", "config.compression") is False
        assert await resolver.resolve("Choose false", "config.flag") is False
        assert await resolver.resolve("Say no", "config.confirm") is False
        assert await resolver.resolve("Deny access", "config.deny") is False
        assert await resolver.resolve("Block feature", "config.block") is False
    
    @pytest.mark.asyncio
    async def test_resolve_boolean_ambiguity_context_positive(self):
        """Test resolving boolean ambiguity with positive context."""
        resolver = AmbiguityResolver()
        
        result = await resolver.resolve("Choose option", "config.enable_feature")
        assert result is True
        
        result = await resolver.resolve("Choose option", "config.support_mode")
        assert result is True
    
    @pytest.mark.asyncio
    async def test_resolve_boolean_ambiguity_context_negative(self):
        """Test resolving boolean ambiguity with negative context."""
        resolver = AmbiguityResolver()
        
        result = await resolver.resolve("Choose option", "config.other_option")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_resolve_number_ambiguity_batch(self):
        """Test resolving number ambiguity with batch context."""
        resolver = AmbiguityResolver()
        
        result = await resolver.resolve("Set batch size", "config.batch_size")
        
        assert result == 32
    
    @pytest.mark.asyncio
    async def test_resolve_number_ambiguity_timeout(self):
        """Test resolving number ambiguity with timeout context."""
        resolver = AmbiguityResolver()
        
        result = await resolver.resolve("Set timeout value", "config.timeout")
        
        assert result == 30
    
    @pytest.mark.asyncio
    async def test_resolve_number_ambiguity_retry(self):
        """Test resolving number ambiguity with retry context."""
        resolver = AmbiguityResolver()
        
        result = await resolver.resolve("Set retry count", "config.retry")
        
        assert result == 3
    
    @pytest.mark.asyncio
    async def test_resolve_number_ambiguity_size(self):
        """Test resolving number ambiguity with size context."""
        resolver = AmbiguityResolver()
        
        result = await resolver.resolve("Set size value", "config.size")
        
        assert result == 100
    
    @pytest.mark.asyncio
    async def test_resolve_number_ambiguity_limit(self):
        """Test resolving number ambiguity with limit context."""
        resolver = AmbiguityResolver()
        
        result = await resolver.resolve("Set limit value", "config.limit")
        
        assert result == 1000
    
    @pytest.mark.asyncio
    async def test_resolve_number_ambiguity_default(self):
        """Test resolving number ambiguity with default context."""
        resolver = AmbiguityResolver()
        
        result = await resolver.resolve("Set some number", "config.number")
        
        assert result == 10
    
    @pytest.mark.asyncio
    async def test_resolve_string_ambiguity_with_quotes(self):
        """Test resolving string ambiguity with quoted content."""
        resolver = AmbiguityResolver()
        
        result = await resolver.resolve('Choose "custom_value" option', "config.value")
        
        assert result == "custom_value"
    
    @pytest.mark.asyncio
    async def test_resolve_string_ambiguity_without_quotes(self):
        """Test resolving string ambiguity without quoted content."""
        model = MockModel()
        model.set_response("Choose string value", "resolved_string")
        resolver = AmbiguityResolver(model)
        
        result = await resolver.resolve("Choose string value", "config.string")
        
        assert result == "resolved_string"
    
    @pytest.mark.asyncio
    async def test_resolve_generic_with_model(self):
        """Test generic resolution using model."""
        model = MockModel()
        model.set_response("Some ambiguous content", "resolved_content")
        resolver = AmbiguityResolver(model)
        
        result = await resolver.resolve("Some ambiguous content", "config.generic")
        
        assert result == "resolved_content"
    
    @pytest.mark.asyncio
    async def test_resolve_generic_with_model_failure(self):
        """Test generic resolution with model failure."""
        model = MockModel()
        model.set_response("Some ambiguous content", Exception("Model failure"))
        resolver = AmbiguityResolver(model)
        
        result = await resolver.resolve("Some ambiguous content", "config.generic")
        
        # Should fallback to heuristic resolution
        assert result == "default"
    
    @pytest.mark.asyncio
    async def test_resolve_with_cache(self):
        """Test resolution with caching."""
        resolver = AmbiguityResolver()
        
        # First resolution
        result1 = await resolver.resolve("Choose format", "test.format")
        
        # Second resolution should use cache
        result2 = await resolver.resolve("Choose format", "test.format")
        
        assert result1 == result2
        assert resolver.get_cache_size() == 1
    
    @pytest.mark.asyncio
    async def test_resolve_with_resolution_error(self):
        """Test resolution with error handling."""
        model = MockModel()
        # Mock the model to raise an exception that cannot be handled
        model.generate = AsyncMock(side_effect=Exception("Unhandleable error"))
        
        resolver = AmbiguityResolver(model)
        
        # Should still work due to fallback
        result = await resolver.resolve("Choose format", "config.format")
        assert result == "json"
    
    def test_classify_ambiguity_choose_boolean(self):
        """Test classifying ambiguity with choose boolean."""
        resolver = AmbiguityResolver()
        
        assert resolver._classify_ambiguity("Choose true or false", "config.flag") == "boolean"
    
    def test_classify_ambiguity_choose_number(self):
        """Test classifying ambiguity with choose number."""
        resolver = AmbiguityResolver()
        
        assert resolver._classify_ambiguity("Choose number of items", "config.items") == "number"
        assert resolver._classify_ambiguity("Select size", "config.size") == "number"
        assert resolver._classify_ambiguity("Choose count", "config.count") == "number"
        assert resolver._classify_ambiguity("Select amount", "config.amount") == "number"
    
    def test_classify_ambiguity_choose_list(self):
        """Test classifying ambiguity with choose list."""
        resolver = AmbiguityResolver()
        
        assert resolver._classify_ambiguity("Choose list of items", "config.items") == "list"
        assert resolver._classify_ambiguity("Select array", "config.array") == "list"
    
    def test_classify_ambiguity_choose_value(self):
        """Test classifying ambiguity with choose value."""
        resolver = AmbiguityResolver()
        
        assert resolver._classify_ambiguity("Choose option", "config.option") == "value"
        assert resolver._classify_ambiguity("Select item", "config.item") == "value"
    
    def test_classify_ambiguity_context_parameter(self):
        """Test classifying ambiguity with parameter context."""
        resolver = AmbiguityResolver()
        
        # "Choose" with parameters context should be classified as parameter (precedence)
        assert resolver._classify_ambiguity("Choose option", "parameters.option") == "parameter"
        # Without "choose" keyword, should classify as parameter
        assert resolver._classify_ambiguity("option", "parameters.option") == "parameter"
    
    def test_classify_ambiguity_context_string(self):
        """Test classifying ambiguity with string context."""
        resolver = AmbiguityResolver()
        
        # "Choose" triggers value classification first
        assert resolver._classify_ambiguity("Choose option", "config.format") == "value"
        assert resolver._classify_ambiguity("Choose option", "config.type") == "value"
        # Without "choose" keyword, should classify based on context
        assert resolver._classify_ambiguity("option", "config.format") == "string"
        assert resolver._classify_ambiguity("option", "config.type") == "string"
    
    def test_classify_ambiguity_context_boolean(self):
        """Test classifying ambiguity with boolean context."""
        resolver = AmbiguityResolver()
        
        # "Choose" triggers value classification first
        assert resolver._classify_ambiguity("Choose option", "config.enable") == "value"
        assert resolver._classify_ambiguity("Choose option", "config.support") == "value"
        # Without "choose" keyword, should classify based on context
        assert resolver._classify_ambiguity("option", "config.enable") == "boolean"
        assert resolver._classify_ambiguity("option", "config.support") == "boolean"
    
    def test_classify_ambiguity_context_number(self):
        """Test classifying ambiguity with number context."""
        resolver = AmbiguityResolver()
        
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
        resolver = AmbiguityResolver()
        
        assert resolver._classify_ambiguity("Some content", "config.other") == "string"
    
    def test_fallback_resolution_query(self):
        """Test fallback resolution with query."""
        resolver = AmbiguityResolver()
        
        result = resolver._fallback_resolution("Choose query", "config.query")
        assert result == "default search query"
    
    def test_fallback_resolution_format(self):
        """Test fallback resolution with format."""
        resolver = AmbiguityResolver()
        
        result = resolver._fallback_resolution("Choose format", "config.format")
        assert result == "json"
    
    def test_fallback_resolution_method(self):
        """Test fallback resolution with method."""
        resolver = AmbiguityResolver()
        
        result = resolver._fallback_resolution("Choose method", "config.method")
        assert result == "auto"
    
    def test_fallback_resolution_style(self):
        """Test fallback resolution with style."""
        resolver = AmbiguityResolver()
        
        result = resolver._fallback_resolution("Choose style", "config.style")
        assert result == "default"
    
    def test_fallback_resolution_type(self):
        """Test fallback resolution with type."""
        resolver = AmbiguityResolver()
        
        result = resolver._fallback_resolution("Choose type", "config.type")
        assert result == "standard"
    
    def test_fallback_resolution_default(self):
        """Test fallback resolution with default."""
        resolver = AmbiguityResolver()
        
        result = resolver._fallback_resolution("Choose something", "config.something")
        assert result == "default"
    
    def test_extract_choices_simple(self):
        """Test extracting choices from simple content."""
        resolver = AmbiguityResolver()
        
        choices = resolver._extract_choices("Choose: json, xml, csv")
        assert choices == ["json", "xml", "csv"]
    
    def test_extract_choices_with_or(self):
        """Test extracting choices with or separator."""
        resolver = AmbiguityResolver()
        
        choices = resolver._extract_choices("Select: json, xml, or csv")
        assert choices == ["json", "xml", "csv"]
    
    def test_extract_choices_select_keyword(self):
        """Test extracting choices with select keyword."""
        resolver = AmbiguityResolver()
        
        choices = resolver._extract_choices("Select from: red, blue, green")
        assert choices == ["red", "blue", "green"]
    
    def test_extract_choices_no_match(self):
        """Test extracting choices with no match."""
        resolver = AmbiguityResolver()
        
        choices = resolver._extract_choices("Some random text")
        assert choices == []
    
    def test_extract_choices_empty_choice(self):
        """Test extracting choices with empty choice."""
        resolver = AmbiguityResolver()
        
        choices = resolver._extract_choices("Choose: json, , csv")
        assert choices == ["json", "csv"]
    
    def test_extract_quotes_single(self):
        """Test extracting single quoted string."""
        resolver = AmbiguityResolver()
        
        quotes = resolver._extract_quotes('Choose "custom_value" option')
        assert quotes == ["custom_value"]
    
    def test_extract_quotes_multiple(self):
        """Test extracting multiple quoted strings."""
        resolver = AmbiguityResolver()
        
        quotes = resolver._extract_quotes('Choose "value1" and "value2"')
        assert quotes == ["value1", "value2"]
    
    def test_extract_quotes_empty(self):
        """Test extracting quotes with empty content."""
        resolver = AmbiguityResolver()
        
        quotes = resolver._extract_quotes('Choose ""')
        assert quotes == [""]
    
    def test_extract_quotes_no_quotes(self):
        """Test extracting quotes with no quotes."""
        resolver = AmbiguityResolver()
        
        quotes = resolver._extract_quotes('Choose value')
        assert quotes == []
    
    def test_clear_cache(self):
        """Test clearing the resolution cache."""
        resolver = AmbiguityResolver()
        
        # Add something to cache
        resolver.resolution_cache["test"] = "value"
        assert resolver.get_cache_size() == 1
        
        # Clear cache
        resolver.clear_cache()
        assert resolver.get_cache_size() == 0
    
    def test_set_resolution_strategy(self):
        """Test setting custom resolution strategy."""
        resolver = AmbiguityResolver()
        
        async def custom_strategy(content, context_path):
            return "custom_result"
        
        resolver.set_resolution_strategy("custom", custom_strategy)
        
        assert "custom" in resolver.resolution_strategies
        assert resolver.resolution_strategies["custom"] == custom_strategy
    
    @pytest.mark.asyncio
    async def test_resolve_with_custom_strategy(self):
        """Test resolving with custom strategy."""
        resolver = AmbiguityResolver()
        
        async def custom_strategy(content, context_path):
            return "custom_result"
        
        resolver.set_resolution_strategy("custom", custom_strategy)
        
        # Mock the classify method to return our custom type
        original_classify = resolver._classify_ambiguity
        resolver._classify_ambiguity = lambda content, context_path: "custom"
        
        result = await resolver.resolve("test content", "test.path")
        
        assert result == "custom_result"
        
        # Restore original method
        resolver._classify_ambiguity = original_classify
    
    @pytest.mark.asyncio
    async def test_resolve_unknown_strategy(self):
        """Test resolving with unknown strategy type."""
        resolver = AmbiguityResolver()
        
        # Mock the classify method to return unknown type
        original_classify = resolver._classify_ambiguity
        resolver._classify_ambiguity = lambda content, context_path: "unknown_type"
        
        model = MockModel()
        model.set_response("test content", "generic_result")
        resolver.model = model
        
        result = await resolver.resolve("test content", "test.path")
        
        assert result == "generic_result"
        
        # Restore original method
        resolver._classify_ambiguity = original_classify
    
    @pytest.mark.asyncio
    async def test_resolve_with_exception_in_strategy(self):
        """Test resolving with exception in strategy."""
        resolver = AmbiguityResolver()
        
        async def failing_strategy(content, context_path):
            raise Exception("Strategy failure")
        
        resolver.set_resolution_strategy("failing", failing_strategy)
        
        # Mock the classify method to return our failing type
        original_classify = resolver._classify_ambiguity
        resolver._classify_ambiguity = lambda content, context_path: "failing"
        
        with pytest.raises(AmbiguityResolutionError):
            await resolver.resolve("test content", "test.path")
        
        # Restore original method
        resolver._classify_ambiguity = original_classify
    
    def test_resolver_creation_no_model_available(self):
        """Test resolver creation when no model is available."""
        with pytest.raises(RuntimeError, match="No AI model available"):
            AmbiguityResolver(model=None, fallback_to_mock=False)
    
    def test_classify_ambiguity_choose_with_true_false(self):
        """Test classification of choose with true/false."""
        resolver = AmbiguityResolver()
        
        assert resolver._classify_ambiguity("Choose true or false", "config.option") == "boolean"
        assert resolver._classify_ambiguity("Select false value", "config.option") == "boolean"
    
    def test_classify_ambiguity_choose_with_number_keywords(self):
        """Test classification of choose with number keywords."""
        resolver = AmbiguityResolver()
        
        assert resolver._classify_ambiguity("Choose number of items", "config.option") == "number"
        assert resolver._classify_ambiguity("Select count value", "config.option") == "number"
        assert resolver._classify_ambiguity("Choose amount needed", "config.option") == "number"
        assert resolver._classify_ambiguity("Select size option", "config.option") == "number"
    
    def test_classify_ambiguity_choose_with_list_keywords(self):
        """Test classification of choose with list keywords."""
        resolver = AmbiguityResolver()
        
        assert resolver._classify_ambiguity("Choose list of items", "config.option") == "list"
        assert resolver._classify_ambiguity("Select array values", "config.option") == "list"
        assert resolver._classify_ambiguity("Choose items to include", "config.option") == "list"
        assert resolver._classify_ambiguity("Select languages available", "config.option") == "list"
    
    def test_classify_ambiguity_missing_line_148_boolean_choose(self):
        """Test line 148: return 'boolean' for true/false in choose context."""
        resolver = AmbiguityResolver()
        
        # Test exact condition: choose/select + "true" or "false" content
        assert resolver._classify_ambiguity("Choose true", "config.setting") == "boolean"
        assert resolver._classify_ambiguity("Select false", "config.option") == "boolean"
        assert resolver._classify_ambiguity("Choose true option", "config.flag") == "boolean"
        assert resolver._classify_ambiguity("Select false value", "config.toggle") == "boolean"
    
    def test_classify_ambiguity_missing_line_150_number_choose(self):
        """Test line 150: return 'number' for number words in choose context."""
        resolver = AmbiguityResolver()
        
        # Test exact condition: choose/select + number words
        assert resolver._classify_ambiguity("Choose number", "config.setting") == "number"
        assert resolver._classify_ambiguity("Select size", "config.option") == "number"
        assert resolver._classify_ambiguity("Choose count", "config.flag") == "number"
        assert resolver._classify_ambiguity("Select amount", "config.data") == "number"
    
    def test_classify_ambiguity_missing_line_171_list_context(self):
        """Test line 171: return 'list' for list-related context paths."""
        resolver = AmbiguityResolver()
        
        # Test exact condition: list-related context path words (line 171)
        assert resolver._classify_ambiguity("Process data", "step.languages") == "list"
        assert resolver._classify_ambiguity("Handle content", "config.items") == "list"
        assert resolver._classify_ambiguity("Process info", "config.tags") == "list"
        assert resolver._classify_ambiguity("Manage data", "step.options") == "list"
        assert resolver._classify_ambiguity("Transform content", "config.list") == "list"