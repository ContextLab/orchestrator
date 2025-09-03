"""Tests for the new lazy AUTO tag resolution system."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from src.orchestrator.auto_resolution import (

from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider
    AutoTagContext,
    AutoTagResolution,
    AutoTagConfig,
    LazyAutoTagResolver,
    RequirementsAnalysis,
    PromptConstruction,
    ActionPlan,
    NestedAutoTagHandler,
    ResolutionLogger,
    ParseError,
    AutoTagResolutionError,
    AutoTagNestingError,
)
from src.orchestrator.auto_resolution.integration import EnhancedControlFlowAutoResolver
from src.orchestrator.core.pipeline import Pipeline
from src.orchestrator.core.task import Task


class TestAutoTagModels:
    """Test AUTO tag data models."""
    
    def test_auto_tag_context_creation(self):
        """Test AutoTagContext creation and methods."""
        pipeline = Pipeline(id="test-pipeline", name="Test")
        context = AutoTagContext(
            pipeline=pipeline,
            current_task_id="task1",
            tag_location="steps[0].action",
            variables={"var1": "value1"},
            step_results={"step1": {"result": "data"}},
            loop_context={"$item": "current", "$index": 0}
        )
        
        assert context.current_task_id == "task1"
        assert context.tag_location == "steps[0].action"
        assert context.resolution_depth == 0
        
        # Test get_full_context
        full_context = context.get_full_context()
        assert "variables" in full_context
        assert "step_results" in full_context
        assert "loop" in full_context
        assert full_context["variables"]["var1"] == "value1"
    
    def test_requirements_analysis_model(self):
        """Test RequirementsAnalysis model."""
        req = RequirementsAnalysis(
            tools_needed=["web_search", "file_reader"],
            output_format="json",
            expected_output_type="object",
            model_used="gpt-4"
        )
        
        assert len(req.tools_needed) == 2
        assert req.output_format == "json"
        assert req.expected_output_type == "object"
    
    def test_auto_tag_resolution_checkpoint(self):
        """Test resolution checkpoint data."""
        resolution = AutoTagResolution(
            original_tag="Generate a summary",
            tag_location="steps[0].action",
            requirements=RequirementsAnalysis(),
            prompt_construction=PromptConstruction(prompt="test"),
            resolved_value="This is a summary",
            action_plan=ActionPlan(action_type="return_value"),
            total_time_ms=1500,
            final_model_used="gpt-4"
        )
        
        checkpoint_data = resolution.to_checkpoint_data()
        assert checkpoint_data["original_tag"] == "Generate a summary"
        assert checkpoint_data["resolved_value"] == "This is a summary"
        assert checkpoint_data["total_time_ms"] == 1500
        assert "timestamp" in checkpoint_data


class TestLazyAutoTagResolver:
    """Test the core lazy AUTO tag resolver."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for testing."""
        return {
            "requirements_analyzer": AsyncMock(),
            "prompt_constructor": AsyncMock(),
            "resolution_executor": AsyncMock(),
            "action_determiner": AsyncMock(),
            "logger": Mock()
        }
    
    @pytest.fixture
    def resolver(self, mock_components):
        """Create resolver with mocked components."""
        config = AutoTagConfig(
            model_escalation=["model1", "model2"],
            max_retries_per_model=2
        )
        
        return LazyAutoTagResolver(
            config=config,
            **mock_components
        )
    
    @pytest.fixture
    def sample_context(self):
        """Create sample AUTO tag context."""
        pipeline = Pipeline(id="test-pipeline", name="Test")
        pipeline.add_task(Task(id="task1", name="Task 1", action="test"))
        
        return AutoTagContext(
            pipeline=pipeline,
            current_task_id="task1",
            tag_location="steps[0].action",
            variables={"input": "test data"},
            step_results={"previous": {"status": "success"}}
        )
    
    @pytest.mark.asyncio
    async def test_successful_resolution(self, resolver, sample_context, mock_components):
        """Test successful AUTO tag resolution."""
        # Setup mocks
        mock_components["requirements_analyzer"].analyze.return_value = RequirementsAnalysis(
            expected_output_type="string",
            model_used="model1"
        )
        
        mock_components["prompt_constructor"].construct.return_value = PromptConstruction(
            prompt="Generated prompt",
            model_used="model1"
        )
        
        mock_components["resolution_executor"].execute.return_value = "Resolved value"
        
        mock_components["action_determiner"].determine.return_value = ActionPlan(
            action_type="return_value",
            model_used="model1"
        )
        
        # Execute resolution
        result = await resolver.resolve("Generate something", sample_context)
        
        # Verify result
        assert isinstance(result, AutoTagResolution)
        assert result.resolved_value == "Resolved value"
        assert result.final_model_used == "model1"
        assert result.action_plan.action_type == "return_value"
        
        # Verify all components were called
        assert mock_components["requirements_analyzer"].analyze.called
        assert mock_components["prompt_constructor"].construct.called
        assert mock_components["resolution_executor"].execute.called
        assert mock_components["action_determiner"].determine.called
    
    @pytest.mark.asyncio
    async def test_model_escalation(self, resolver, sample_context, mock_components):
        """Test escalation to more powerful models on failure."""
        # First model fails
        mock_components["requirements_analyzer"].analyze.side_effect = [
            ParseError("Parse failed"),
            RequirementsAnalysis(expected_output_type="string", model_used="model2")
        ]
        
        # Setup other mocks for success on second model
        mock_components["prompt_constructor"].construct.return_value = PromptConstruction(
            prompt="Generated prompt",
            model_used="model2"
        )
        mock_components["resolution_executor"].execute.return_value = "Resolved value"
        mock_components["action_determiner"].determine.return_value = ActionPlan(
            action_type="return_value",
            model_used="model2"
        )
        
        # Execute resolution
        result = await resolver.resolve("Generate something", sample_context)
        
        # Verify escalation
        assert result.final_model_used == "model2"
        assert len(result.models_attempted) == 2
        assert mock_components["requirements_analyzer"].analyze.call_count == 2
    
    @pytest.mark.asyncio
    async def test_resolution_failure(self, resolver, sample_context, mock_components):
        """Test complete resolution failure."""
        # All models fail
        mock_components["requirements_analyzer"].analyze.side_effect = ParseError("Parse failed")
        
        # Execute and expect failure
        with pytest.raises(AutoTagResolutionError) as exc_info:
            await resolver.resolve("Generate something", sample_context)
        
        assert "Failed to resolve AUTO tag after trying models" in str(exc_info.value)
    
    def test_extract_auto_tag_content(self, resolver):
        """Test AUTO tag content extraction."""
        # Single AUTO tag
        content = resolver.extract_auto_tag_content("<AUTO>Generate summary</AUTO>")
        assert content == "Generate summary"
        
        # AUTO tag with surrounding text
        content = resolver.extract_auto_tag_content("prefix <AUTO>Content</AUTO> suffix")
        assert content == "Content"
        
        # No AUTO tag
        content = resolver.extract_auto_tag_content("No auto tag here")
        assert content is None
        
        # Nested content
        content = resolver.extract_auto_tag_content("<AUTO>Line 1\nLine 2</AUTO>")
        assert content == "Line 1\nLine 2"


class TestNestedAutoTagHandler:
    """Test nested AUTO tag handling."""
    
    @pytest.fixture
    def handler(self):
        """Create nested handler with mock resolver."""
        mock_resolver = Mock()
        mock_resolver.resolve = AsyncMock()
        mock_resolver.logger = Mock()
        return NestedAutoTagHandler(mock_resolver)
    
    @pytest.fixture
    def sample_context(self):
        """Create sample context."""
        pipeline = Pipeline(id="test", name="Test")
        return AutoTagContext(
            pipeline=pipeline,
            current_task_id="task1",
            tag_location="test",
            variables={},
            step_results={}
        )
    
    def test_find_auto_tags(self, handler):
        """Test finding AUTO tags in content."""
        content = "Start <AUTO>First</AUTO> middle <AUTO>Second</AUTO> end"
        tags = handler._find_auto_tags(content)
        
        assert len(tags) == 2
        assert tags[0]["content"] == "First"
        assert tags[1]["content"] == "Second"
        
        # Test nested tags
        content = "<AUTO>Outer <AUTO>Inner</AUTO> more</AUTO>"
        tags = handler._find_auto_tags(content)
        assert len(tags) == 2
    
    def test_has_auto_tags(self, handler):
        """Test checking for AUTO tags."""
        assert handler._has_auto_tags("<AUTO>Test</AUTO>")
        assert handler._has_auto_tags("Text with <AUTO>tag</AUTO> inside")
        assert not handler._has_auto_tags("No tags here")
    
    @pytest.mark.asyncio
    async def test_resolve_nested_simple(self, handler, sample_context):
        """Test resolving simple nested AUTO tags."""
        # Setup mock resolver
        resolution = Mock()
        resolution.resolved_value = "RESOLVED"
        resolution.action_plan = Mock(action_type="return_value")
        
        handler.resolver.resolve.return_value = resolution
        
        # Test resolution
        content = "Value: <AUTO>Generate value</AUTO>"
        result = await handler.resolve_nested(content, sample_context)
        
        assert result == "Value: RESOLVED"
        assert handler.resolver.resolve.called
    
    @pytest.mark.asyncio
    async def test_resolve_deeply_nested(self, handler, sample_context):
        """Test resolving deeply nested AUTO tags."""
        # Setup mock to resolve inner tag first, then outer
        resolutions = []
        
        def create_resolution(value):
            resolution = Mock()
            resolution.resolved_value = value
            resolution.action_plan = Mock(action_type="return_value")
            return resolution
        
        handler.resolver.resolve.side_effect = [
            create_resolution("inner_resolved"),
            create_resolution("outer_resolved")
        ]
        
        # Test with nested tags
        content = "<AUTO>Process <AUTO>Inner tag</AUTO> result</AUTO>"
        result = await handler.resolve_nested(content, sample_context)
        
        # Should resolve inner first, then outer
        assert handler.resolver.resolve.call_count == 2
    
    @pytest.mark.asyncio
    async def test_max_depth_exceeded(self, handler, sample_context):
        """Test maximum nesting depth protection."""
        sample_context.resolution_depth = 5
        
        with pytest.raises(AutoTagNestingError):
            await handler.resolve_nested("<AUTO>Test</AUTO>", sample_context, max_depth=5)
    
    def test_count_auto_tags(self, handler):
        """Test counting AUTO tags."""
        assert handler.count_auto_tags("No tags") == 0
        assert handler.count_auto_tags("<AUTO>One</AUTO>") == 1
        assert handler.count_auto_tags("<AUTO>One</AUTO> and <AUTO>Two</AUTO>") == 2


class TestIntegration:
    """Test integration with existing system."""
    
    @pytest.fixture
    def enhanced_resolver(self):
        """Create enhanced resolver."""
        pipeline = Pipeline(id="test", name="Test")
        config = AutoTagConfig()
        return EnhancedControlFlowAutoResolver(
            model_registry=None,
            config=config,
            pipeline=pipeline
        )
    
    @pytest.mark.asyncio
    async def test_resolve_condition_compatibility(self, enhanced_resolver):
        """Test backward compatibility for condition resolution."""
        # Test simple boolean
        result = await enhanced_resolver.resolve_condition(
            "true",
            {},
            {},
            cache_key="test"
        )
        assert result is True
        
        # Test from cache
        result2 = await enhanced_resolver.resolve_condition(
            "false",  # Different value, but should use cache
            {},
            {},
            cache_key="test"
        )
        assert result2 is True  # From cache
    
    @pytest.mark.asyncio
    async def test_resolve_iterator_compatibility(self, enhanced_resolver):
        """Test backward compatibility for iterator resolution."""
        # Test list string
        result = await enhanced_resolver.resolve_iterator(
            '["item1", "item2", "item3"]',
            {},
            {}
        )
        assert result == ["item1", "item2", "item3"]
        
        # Test comma-separated
        result = await enhanced_resolver.resolve_iterator(
            "a, b, c",
            {},
            {}
        )
        assert result == ["a", "b", "c"]
    
    @pytest.mark.asyncio
    async def test_simple_resolution_fallback(self, enhanced_resolver):
        """Test simple resolution when full system unavailable."""
        # Remove pipeline to trigger fallback
        enhanced_resolver.pipeline = None
        
        # Test AUTO tag resolution with fallback
        result = await enhanced_resolver._resolve_with_new_system(
            '<AUTO>Choose between success_handler and error_handler</AUTO>',
            {},
            {},
            "test",
            "goto"
        )
        
        # Should use simple heuristics
        assert result in ["success_handler", "error_handler", "default_handler"]
    
    def test_type_conversions(self, enhanced_resolver):
        """Test type conversion methods."""
        # Boolean conversion
        assert enhanced_resolver._to_boolean(True) is True
        assert enhanced_resolver._to_boolean("true") is True
        assert enhanced_resolver._to_boolean("false") is False
        assert enhanced_resolver._to_boolean("yes") is True
        assert enhanced_resolver._to_boolean("no") is False
        assert enhanced_resolver._to_boolean("1") is True
        assert enhanced_resolver._to_boolean("0") is False
        assert enhanced_resolver._to_boolean("") is False
        
        # List conversion
        assert enhanced_resolver._to_list([1, 2, 3]) == [1, 2, 3]
        assert enhanced_resolver._to_list("a, b, c") == ["a", "b", "c"]
        assert enhanced_resolver._to_list('["x", "y"]') == ["x", "y"]
        assert enhanced_resolver._to_list("single") == ["single"]
        
        # Int conversion
        assert enhanced_resolver._to_int(42) == 42
        assert enhanced_resolver._to_int(42.7) == 42
        assert enhanced_resolver._to_int("123") == 123
        assert enhanced_resolver._to_int("  456  ") == 456
        
        with pytest.raises(ValueError):
            enhanced_resolver._to_int("not a number")


class TestResolutionLogger:
    """Test resolution logging."""
    
    @pytest.fixture
    def logger(self):
        """Create resolution logger."""
        return ResolutionLogger(checkpoint_resolutions=True)
    
    def test_log_resolution_complete(self, logger):
        """Test logging complete resolution."""
        resolution = AutoTagResolution(
            original_tag="test",
            tag_location="test.location",
            requirements=RequirementsAnalysis(),
            prompt_construction=PromptConstruction(prompt="test"),
            resolved_value="result",
            action_plan=ActionPlan(action_type="return_value"),
            total_time_ms=100,
            final_model_used="gpt-4"
        )
        
        logger.log_resolution_complete(resolution)
        
        # Check history
        assert len(logger.resolution_history) == 1
        assert logger.resolution_history[0] == resolution
    
    def test_checkpoint_data(self, logger):
        """Test getting checkpoint data."""
        # Add some resolutions
        for i in range(3):
            resolution = AutoTagResolution(
                original_tag=f"test{i}",
                tag_location=f"location{i}",
                requirements=RequirementsAnalysis(),
                prompt_construction=PromptConstruction(prompt="test"),
                resolved_value=f"result{i}",
                action_plan=ActionPlan(action_type="return_value")
            )
            logger.resolution_history.append(resolution)
        
        # Get checkpoint data
        checkpoint_data = logger.get_checkpoint_data()
        assert len(checkpoint_data) == 3
        assert all("timestamp" in item for item in checkpoint_data)
        
        # Clear history
        logger.clear_history()
        assert len(logger.resolution_history) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])