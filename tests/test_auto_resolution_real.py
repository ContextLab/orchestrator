"""Real integration tests for AUTO tag resolution using actual LLM APIs."""

import asyncio
import os
import pytest
from unittest.mock import MagicMock

from orchestrator.core.pipeline import Pipeline
from orchestrator.core.task import Task
from orchestrator.auto_resolution.models import AutoTagContext
from orchestrator.auto_resolution.resolver import LazyAutoTagResolver
from orchestrator.auto_resolution.model_caller import ModelCaller
from orchestrator.auto_resolution.requirements_analyzer import RequirementsAnalyzer
from orchestrator.auto_resolution.prompt_constructor import PromptConstructor
from orchestrator.auto_resolution.resolution_executor import ResolutionExecutor
from orchestrator.auto_resolution.action_determiner import ActionDeterminer
from orchestrator.auto_resolution.nested_handler import NestedAutoTagHandler


# Skip these tests if no API keys are available
pytestmark = pytest.mark.skipif(
    not any([
        os.getenv("OPENAI_API_KEY"),
        os.getenv("ANTHROPIC_API_KEY"),
        os.getenv("GOOGLE_API_KEY")
    ]),
    reason="No API keys available for real testing"
)


@pytest.fixture
def sample_pipeline():
    """Create a sample pipeline for testing."""
    pipeline = Pipeline(
        id="test-pipeline",
        name="Test Pipeline",
        tasks={
            "analyze": Task(
                id="analyze",
                name="Analyze Data",
                action="analyze_data",
                parameters={
                    "data": "{{ input_data }}",
                    "method": "<AUTO>Choose best analysis method</AUTO>"
                },
                metadata={"tool": "data_analyzer"}
            ),
            "report": Task(
                id="report",
                name="Generate Report",
                action="generate_report", 
                parameters={
                    "results": "{{ analyze.output }}",
                    "format": "<AUTO>Determine report format based on audience</AUTO>"
                },
                dependencies=["analyze"],
                metadata={"produces": "report.md"}
            )
        },
        metadata={"model": "gpt-4o-mini"}
    )
    return pipeline


@pytest.fixture
def context(sample_pipeline):
    """Create context for AUTO tag resolution."""
    return AutoTagContext(
        pipeline=sample_pipeline,
        current_task_id="analyze",
        tag_location="tasks.analyze.parameters.method",
        variables={"input_data": "Sales data for Q4 2023"},
        step_results={},
        loop_context=None,
        resolution_depth=0,
        parent_resolutions=[]
    )


class TestRealModelCaller:
    """Test real model calling infrastructure."""
    
    @pytest.mark.asyncio
    async def test_openai_call(self):
        """Test real OpenAI API call."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not available")
            
        caller = ModelCaller()
        response = await caller.call_model(
            model="gpt-4o-mini",
            prompt="What is 2+2? Answer with just the number.",
            temperature=0
        )
        
        assert "4" in response
    
    @pytest.mark.asyncio
    async def test_anthropic_call(self):
        """Test real Anthropic API call."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("Anthropic API key not available")
            
        caller = ModelCaller()
        response = await caller.call_model(
            model="claude-3-haiku-20240307",
            prompt="What is the capital of France? Answer with just the city name.",
            temperature=0
        )
        
        assert "Paris" in response
    
    @pytest.mark.asyncio
    async def test_gemini_call(self):
        """Test real Gemini API call."""
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("Google API key not available")
            
        caller = ModelCaller()
        response = await caller.call_model(
            model="gemini-pro",
            prompt="Complete this sequence: 1, 2, 3, ___. Answer with just the next number.",
            temperature=0
        )
        
        assert "4" in response
    
    @pytest.mark.asyncio
    async def test_json_mode(self):
        """Test JSON mode with real API."""
        caller = ModelCaller()
        
        if caller.is_model_available("gpt-4o-mini"):
            response = await caller.call_model(
                model="gpt-4o-mini",
                prompt='Return a JSON object with keys "name" and "age" for a person named John who is 30.',
                json_mode=True,
                temperature=0
            )
            
            import json
            data = json.loads(response)
            assert data.get("name") == "John"
            assert data.get("age") == 30


class TestRealRequirementsAnalyzer:
    """Test requirements analysis with real LLMs."""
    
    @pytest.mark.asyncio
    async def test_simple_requirement_analysis(self, context):
        """Test analyzing simple AUTO tag requirements."""
        analyzer = RequirementsAnalyzer()
        
        # Use first available model
        model_caller = ModelCaller()
        models = model_caller.get_available_models()
        if not models:
            pytest.skip("No models available")
        
        model = models[0]
        
        requirements = await analyzer.analyze(
            "Choose the best data analysis method for sales data",
            context,
            model
        )
        
        assert requirements.expected_output_type == "string"
        assert requirements.model_used == model
    
    @pytest.mark.asyncio
    async def test_complex_requirement_analysis(self, context):
        """Test analyzing complex AUTO tag with dependencies."""
        analyzer = RequirementsAnalyzer()
        
        # Update context with more complex scenario
        context.tag_location = "tasks.process.parameters.config"
        
        model_caller = ModelCaller()
        models = model_caller.get_available_models()
        if not models:
            pytest.skip("No models available")
        
        requirements = await analyzer.analyze(
            "Generate JSON configuration for processing {{ input_data }} with optimal settings",
            context,
            models[0]
        )
        
        assert requirements.expected_output_type in ["object", "string"]
        assert "input_data" in requirements.data_dependencies
        assert requirements.output_format == "json"


class TestRealPromptConstructor:
    """Test prompt construction with real LLMs."""
    
    @pytest.mark.asyncio 
    async def test_prompt_construction(self, context):
        """Test constructing prompts based on requirements."""
        analyzer = RequirementsAnalyzer()
        constructor = PromptConstructor()
        
        model_caller = ModelCaller()
        models = model_caller.get_available_models()
        if not models:
            pytest.skip("No models available")
        
        # First analyze requirements
        requirements = await analyzer.analyze(
            "Choose the best visualization type for {{ input_data }}",
            context,
            models[0]
        )
        
        # Then construct prompt
        prompt_data = await constructor.construct(
            "Choose the best visualization type for {{ input_data }}",
            context,
            requirements,
            models[0]
        )
        
        assert prompt_data.prompt
        assert "sales data for Q4 2023" in prompt_data.prompt.lower()  # Variable should be resolved
        assert prompt_data.model_used == models[0]


class TestRealResolutionExecutor:
    """Test resolution execution with real LLMs."""
    
    @pytest.mark.asyncio
    async def test_string_resolution(self, context):
        """Test resolving AUTO tag to string value."""
        analyzer = RequirementsAnalyzer()
        constructor = PromptConstructor()
        executor = ResolutionExecutor()
        
        model_caller = ModelCaller()
        models = model_caller.get_available_models()
        if not models:
            pytest.skip("No models available")
        
        # Analyze
        requirements = await analyzer.analyze(
            "Choose between 'regression' or 'classification' for this data",
            context,
            models[0]
        )
        
        # Construct
        prompt_data = await constructor.construct(
            "Choose between 'regression' or 'classification' for this data",
            context,
            requirements,
            models[0]
        )
        
        # Execute
        result = await executor.execute(prompt_data, context, requirements)
        
        assert isinstance(result, str)
        assert result.lower() in ["regression", "classification"]
    
    @pytest.mark.asyncio
    async def test_json_resolution(self, context):
        """Test resolving AUTO tag to JSON object."""
        analyzer = RequirementsAnalyzer()
        constructor = PromptConstructor()
        executor = ResolutionExecutor()
        
        model_caller = ModelCaller()
        models = model_caller.get_available_models()
        if not models:
            pytest.skip("No models available")
        
        # Update context
        context.tag_location = "tasks.config.parameters.settings"
        
        # Analyze
        requirements = await analyzer.analyze(
            "Generate a JSON config with keys: 'batch_size' (number) and 'enabled' (boolean)",
            context,
            models[0]
        )
        
        # Override requirements to ensure JSON output
        requirements.expected_output_type = "object"
        requirements.output_format = "json"
        
        # Construct
        prompt_data = await constructor.construct(
            "Generate a JSON config with keys: 'batch_size' (number) and 'enabled' (boolean)",
            context,
            requirements,
            models[0]
        )
        
        # Add output schema
        prompt_data.output_schema = {
            "type": "object",
            "properties": {
                "batch_size": {"type": "number"},
                "enabled": {"type": "boolean"}
            },
            "required": ["batch_size", "enabled"]
        }
        
        # Execute
        result = await executor.execute(prompt_data, context, requirements)
        
        assert isinstance(result, dict)
        assert "batch_size" in result
        assert "enabled" in result
        assert isinstance(result["batch_size"], (int, float))
        assert isinstance(result["enabled"], bool)


class TestRealActionDeterminer:
    """Test action determination with real LLMs."""
    
    @pytest.mark.asyncio
    async def test_return_value_action(self, context):
        """Test determining return_value action."""
        analyzer = RequirementsAnalyzer()
        determiner = ActionDeterminer()
        
        model_caller = ModelCaller()
        models = model_caller.get_available_models()
        if not models:
            pytest.skip("No models available")
        
        # Create mock requirements
        requirements = MagicMock()
        requirements.tools_needed = []
        requirements.output_format = "text"
        requirements.expected_output_type = "string"
        
        # Determine action for a simple string value
        action = await determiner.determine(
            "regression",
            requirements,
            context,
            models[0]
        )
        
        assert action.action_type == "return_value"
        assert action.model_used == models[0]
    
    @pytest.mark.asyncio
    async def test_save_file_action(self, context):
        """Test determining save_file action."""
        analyzer = RequirementsAnalyzer()
        determiner = ActionDeterminer()
        
        model_caller = ModelCaller()
        models = model_caller.get_available_models()
        if not models:
            pytest.skip("No models available")
        
        # Update context to indicate file output
        context.tag_location = "tasks.report.metadata.produces"
        
        # Create mock requirements
        requirements = MagicMock()
        requirements.tools_needed = []
        requirements.output_format = "markdown"
        requirements.expected_output_type = "string"
        
        # Determine action for a report content
        action = await determiner.determine(
            "# Analysis Report\n\nThis is the report content...",
            requirements,
            context,
            models[0]
        )
        
        # Should suggest saving to file since it's in 'produces' metadata
        assert action.action_type in ["save_file", "return_value"]


class TestRealEndToEndResolution:
    """Test complete AUTO tag resolution with real LLMs."""
    
    @pytest.mark.asyncio
    async def test_simple_resolution(self, sample_pipeline):
        """Test resolving a simple AUTO tag end-to-end."""
        resolver = LazyAutoTagResolver()
        
        context = AutoTagContext(
            pipeline=sample_pipeline,
            current_task_id="analyze",
            tag_location="tasks.analyze.parameters.method",
            variables={"input_data": "Customer purchase history"},
            step_results={},
            loop_context=None,
            resolution_depth=0,
            parent_resolutions=[]
        )
        
        resolution = await resolver.resolve(
            "Choose the best analysis method: 'clustering', 'regression', or 'classification'",
            context
        )
        
        # Check that resolved value contains one of the expected methods
        resolved_lower = resolution.resolved_value.lower().strip("'\"")
        assert any(method in resolved_lower for method in ["clustering", "regression", "classification"])
        assert resolution.action_plan.action_type == "return_value"
    
    @pytest.mark.asyncio
    async def test_nested_resolution(self, sample_pipeline):
        """Test resolving nested AUTO tags."""
        resolver = LazyAutoTagResolver()
        handler = NestedAutoTagHandler(resolver)
        
        context = AutoTagContext(
            pipeline=sample_pipeline,
            current_task_id="process",
            tag_location="tasks.process.parameters.config",
            variables={
                "data_type": "numeric",
                "data_size": 1000
            },
            step_results={},
            loop_context=None,
            resolution_depth=0,
            parent_resolutions=[]
        )
        
        # Content with nested AUTO tags
        content = """
        {
            "method": "<AUTO>Choose method based on {{ data_type }}</AUTO>",
            "batch_size": <AUTO>Optimal batch size for {{ data_size }} records</AUTO>
        }
        """
        
        result = await handler.resolve_nested(content, context)
        
        # Should have resolved both AUTO tags
        assert "<AUTO>" not in result
        assert "method" in result
        assert "batch_size" in result
    
    @pytest.mark.asyncio
    async def test_model_escalation(self, sample_pipeline):
        """Test model escalation on parsing failure."""
        resolver = LazyAutoTagResolver(
            default_models=["gpt-4o-mini", "gpt-4o", "claude-sonnet-4-20250514"]
        )
        
        context = AutoTagContext(
            pipeline=sample_pipeline,
            current_task_id="complex",
            tag_location="tasks.complex.parameters.algorithm",
            variables={"requirements": "Need highly optimized algorithm"},
            step_results={},
            loop_context=None,
            resolution_depth=0,
            parent_resolutions=[]
        )
        
        # Complex AUTO tag that might need a more powerful model
        resolution = await resolver.resolve(
            """Generate a complex JSON schema for a machine learning pipeline configuration
            that includes hyperparameters, data preprocessing steps, model architecture,
            and evaluation metrics. Ensure all fields have appropriate constraints.""",
            context
        )
        
        # Should have resolved successfully (potentially after escalation)
        assert resolution.resolved_value is not None
        assert resolution.requirements_analysis is not None
        assert resolution.prompt_construction is not None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])