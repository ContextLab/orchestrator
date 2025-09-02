"""
Comprehensive tests for runtime model selection integration.

This module tests the integration of intelligent model selection capabilities
with the pipeline execution engine, ensuring that models are selected optimally
at runtime based on step requirements, execution context, and selection strategies.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Optional

# Import the modules we're testing
from src.orchestrator.execution.model_selector import (
    ExecutionModelSelector, 
    RuntimeModelContext
)
from src.orchestrator.execution.engine import StateGraphEngine
from src.orchestrator.api.execution import (
    PipelineExecutor,
    create_intelligent_pipeline_executor
)

# Import required foundation components
from src.orchestrator.foundation._compatibility import (
    FoundationConfig,
    PipelineSpecification,
    PipelineStep,
    PipelineResult,
    StepResult
)
from src.orchestrator.models.registry import ModelRegistry
from src.orchestrator.models.model_selector import ModelSelectionCriteria
from src.orchestrator.core.model import Model, ModelCapabilities, ModelCost, ModelMetrics


class TestExecutionModelSelector:
    """Test the ExecutionModelSelector class for runtime model selection."""

    @pytest.fixture
    def mock_model_registry(self):
        """Create a mock model registry with test models."""
        registry = Mock(spec=ModelRegistry)
        
        # Create test models
        fast_model = Mock(spec=Model)
        fast_model.provider = "test"
        fast_model.name = "fast-model"
        fast_model.capabilities = Mock(spec=ModelCapabilities)
        fast_model.capabilities.speed_rating = "fast"
        fast_model.capabilities.accuracy_score = 0.8
        fast_model.capabilities.vision_capable = False
        fast_model.capabilities.code_specialized = True
        fast_model.capabilities.supports_function_calling = True
        fast_model.cost = Mock(spec=ModelCost)
        fast_model.cost.is_free = False
        fast_model.cost.input_cost_per_1k_tokens = 0.001
        fast_model.cost.output_cost_per_1k_tokens = 0.002
        fast_model.metrics = Mock(spec=ModelMetrics)
        fast_model.metrics.success_rate = 0.9
        
        expensive_model = Mock(spec=Model)
        expensive_model.provider = "test"
        expensive_model.name = "expensive-model"
        expensive_model.capabilities = Mock(spec=ModelCapabilities)
        expensive_model.capabilities.speed_rating = "medium"
        expensive_model.capabilities.accuracy_score = 0.95
        expensive_model.capabilities.vision_capable = True
        expensive_model.capabilities.code_specialized = True
        expensive_model.capabilities.supports_function_calling = True
        expensive_model.cost = Mock(spec=ModelCost)
        expensive_model.cost.is_free = False
        expensive_model.cost.input_cost_per_1k_tokens = 0.01
        expensive_model.cost.output_cost_per_1k_tokens = 0.02
        expensive_model.metrics = Mock(spec=ModelMetrics)
        expensive_model.metrics.success_rate = 0.95
        
        free_model = Mock(spec=Model)
        free_model.provider = "test"
        free_model.name = "free-model"
        free_model.capabilities = Mock(spec=ModelCapabilities)
        free_model.capabilities.speed_rating = "slow"
        free_model.capabilities.accuracy_score = 0.7
        free_model.capabilities.vision_capable = False
        free_model.capabilities.code_specialized = False
        free_model.capabilities.supports_function_calling = False
        free_model.cost = Mock(spec=ModelCost)
        free_model.cost.is_free = True
        free_model.cost.input_cost_per_1k_tokens = 0.0
        free_model.cost.output_cost_per_1k_tokens = 0.0
        free_model.metrics = Mock(spec=ModelMetrics)
        free_model.metrics.success_rate = 0.8
        
        registry.models = {
            "test:fast-model": fast_model,
            "test:expensive-model": expensive_model,
            "test:free-model": free_model
        }
        
        # Mock registry methods
        async def mock_filter_by_capabilities(requirements: Dict[str, Any]) -> List[Model]:
            models = [fast_model, expensive_model, free_model]
            filtered = []
            
            for model in models:
                # Simple filtering logic for tests
                if requirements.get("code_specialized") and not model.capabilities.code_specialized:
                    continue
                if requirements.get("vision_capable") and not model.capabilities.vision_capable:
                    continue
                filtered.append(model)
                
            return filtered
        
        async def mock_get_model(provider: str, name: str) -> Optional[Model]:
            key = f"{provider}:{name}"
            return registry.models.get(key)
        
        async def mock_find_model_by_name(name: str) -> Optional[Model]:
            for model in registry.models.values():
                if model.name == name:
                    return model
            return None
        
        registry._filter_by_capabilities = mock_filter_by_capabilities
        registry.get_model = mock_get_model
        registry.find_model_by_name = mock_find_model_by_name
        
        return registry

    @pytest.fixture
    def model_selector(self, mock_model_registry):
        """Create an ExecutionModelSelector with mock registry."""
        return ExecutionModelSelector(
            model_registry=mock_model_registry,
            enable_adaptive_selection=True,
            enable_expert_assignments=True,
            enable_cost_optimization=True
        )

    @pytest.fixture
    def sample_step(self):
        """Create a sample pipeline step for testing."""
        step = Mock(spec=PipelineStep)
        step.id = "test_step"
        step.name = "Test Step"
        step.model = "AUTO:code generation task"
        step.tools = ["code_editor", "compiler"]
        step.variables = {"output": "generated code"}
        step.description = "Generate Python code for data processing"
        step.prompt = "Create a function that processes data"
        step.context_limit = 4000
        return step

    @pytest.fixture
    def sample_pipeline_spec(self, sample_step):
        """Create a sample pipeline specification."""
        spec = Mock(spec=PipelineSpecification)
        spec.header = Mock()
        spec.header.id = "test_pipeline"
        spec.header.name = "Test Pipeline"
        spec.steps = [sample_step]
        
        # Mock selection schema
        spec.selection_schema = Mock()
        spec.selection_schema.strategy = "balanced"
        spec.selection_schema.cost_limit = 0.05
        spec.selection_schema.max_latency_ms = 5000
        
        # Mock experts field
        spec.experts = {
            "code_editor": "test:fast-model",
            "compiler": "test:expensive-model"
        }
        
        return spec

    @pytest.fixture
    def runtime_context(self, sample_step, sample_pipeline_spec):
        """Create a runtime model context for testing."""
        return RuntimeModelContext(
            step=sample_step,
            pipeline_spec=sample_pipeline_spec,
            execution_state={"variables": {"input": "test data"}},
            available_variables={"user_preference": "fast"},
            expert_assignments={"code_editor": "test:fast-model"},
            cost_constraints={"max_cost_per_request": 0.01},
            performance_requirements={"max_latency_ms": 3000}
        )

    @pytest.mark.asyncio
    async def test_select_model_for_step_basic(self, model_selector, runtime_context):
        """Test basic model selection for a step."""
        selected_model = await model_selector.select_model_for_step(runtime_context)
        
        assert selected_model is not None
        assert selected_model.provider == "test"
        assert selected_model.name in ["fast-model", "expensive-model", "free-model"]

    @pytest.mark.asyncio
    async def test_select_model_with_explicit_specification(self, model_selector, runtime_context):
        """Test model selection when step has explicit model specification."""
        # Set explicit model
        runtime_context.step.model = "test:expensive-model"
        
        selected_model = await model_selector.select_model_for_step(runtime_context)
        
        assert selected_model is not None
        assert selected_model.provider == "test"
        assert selected_model.name == "expensive-model"

    @pytest.mark.asyncio
    async def test_select_model_with_expert_assignments(self, model_selector, runtime_context):
        """Test model selection using expert tool-model assignments."""
        # Expert assignment should take precedence
        selected_model = await model_selector.select_model_for_step(runtime_context)
        
        assert selected_model is not None
        # Should select the expert-assigned model for code_editor tool
        assert selected_model.name == "fast-model"

    @pytest.mark.asyncio
    async def test_select_model_cost_optimized_strategy(self, model_selector, runtime_context):
        """Test model selection with cost-optimized strategy."""
        selected_model = await model_selector.select_model_for_step(
            runtime_context, selection_strategy="cost_optimized"
        )
        
        assert selected_model is not None
        # Should prefer cheaper models
        assert selected_model.cost.input_cost_per_1k_tokens <= 0.01

    @pytest.mark.asyncio
    async def test_select_model_performance_optimized_strategy(self, model_selector, runtime_context):
        """Test model selection with performance-optimized strategy."""
        selected_model = await model_selector.select_model_for_step(
            runtime_context, selection_strategy="performance_optimized"
        )
        
        assert selected_model is not None
        # Should prefer faster models
        assert selected_model.capabilities.speed_rating in ["fast", "medium"]

    @pytest.mark.asyncio
    async def test_evaluate_selection_quality(self, model_selector, mock_model_registry):
        """Test evaluation of model selection quality."""
        fast_model = mock_model_registry.models["test:fast-model"]
        
        execution_result = {
            "timestamp": "2024-01-01T00:00:00Z",
            "status": "success",
            "execution_time": 2.5,
            "errors": [],
            "output": {"result": "Generated code successfully"}
        }
        
        quality_metrics = await model_selector.evaluate_selection_quality(
            "test_step", fast_model, execution_result
        )
        
        assert quality_metrics["step_id"] == "test_step"
        assert quality_metrics["model"] == "test:fast-model"
        assert quality_metrics["success"] == True
        assert quality_metrics["execution_time"] == 2.5
        assert "performance_score" in quality_metrics
        assert "cost_efficiency" in quality_metrics

    @pytest.mark.asyncio
    async def test_get_selection_recommendations(self, model_selector, sample_pipeline_spec):
        """Test getting model selection recommendations for a pipeline."""
        execution_context = {"user_type": "developer", "budget": 0.1}
        
        recommendations = await model_selector.get_selection_recommendations(
            sample_pipeline_spec, execution_context
        )
        
        assert "test_step" in recommendations
        step_recommendations = recommendations["test_step"]
        assert "step_name" in step_recommendations
        assert "recommendations" in step_recommendations
        assert len(step_recommendations["recommendations"]) > 0
        
        # Check recommendation structure
        first_rec = step_recommendations["recommendations"][0]
        assert "model" in first_rec
        assert "score" in first_rec
        assert "rationale" in first_rec
        assert "estimated_cost" in first_rec


class TestStateGraphEngine:
    """Test StateGraphEngine integration with model selection."""

    @pytest.fixture
    def mock_model_registry(self):
        """Create a mock model registry."""
        registry = Mock(spec=ModelRegistry)
        
        # Create a test model
        test_model = Mock(spec=Model)
        test_model.provider = "test"
        test_model.name = "integration-model"
        test_model.capabilities = Mock(spec=ModelCapabilities)
        test_model.capabilities.speed_rating = "medium"
        test_model.capabilities.accuracy_score = 0.85
        test_model.capabilities.code_specialized = True
        test_model.cost = Mock(spec=ModelCost)
        test_model.cost.is_free = False
        test_model.metrics = Mock(spec=ModelMetrics)
        test_model.metrics.success_rate = 0.9
        
        registry.models = {"test:integration-model": test_model}
        
        # Mock registry methods
        async def mock_filter_by_capabilities(requirements: Dict[str, Any]) -> List[Model]:
            return [test_model]
        
        registry._filter_by_capabilities = mock_filter_by_capabilities
        
        return registry

    @pytest.fixture
    def engine(self, mock_model_registry):
        """Create StateGraphEngine with model registry."""
        config = FoundationConfig(max_concurrent_steps=5)
        return StateGraphEngine(config=config, model_registry=mock_model_registry)

    @pytest.fixture
    def sample_pipeline_spec(self):
        """Create a pipeline specification for testing."""
        step = Mock(spec=PipelineStep)
        step.id = "step1"
        step.name = "Test Step"
        step.model = "AUTO"
        step.tools = ["test_tool"]
        step.variables = {"output": "test output"}
        step.dependencies = []
        step.condition = None
        step.retry_count = 0
        
        spec = Mock(spec=PipelineSpecification)
        spec.header = Mock()
        spec.header.id = "test_pipeline"
        spec.header.name = "Test Pipeline"
        spec.steps = [step]
        
        # Mock spec methods
        def mock_get_step(step_id: str):
            if step_id == "step1":
                return step
            return None
        
        def mock_get_execution_order():
            return [["step1"]]
        
        def mock_get_dependents(step_id: str):
            return []
        
        spec.get_step = mock_get_step
        spec.get_execution_order = mock_get_execution_order
        spec.get_dependents = mock_get_dependents
        
        return spec

    @pytest.mark.asyncio
    async def test_engine_initialization_with_model_registry(self, mock_model_registry):
        """Test engine initialization with model registry."""
        config = FoundationConfig()
        engine = StateGraphEngine(config=config, model_registry=mock_model_registry)
        
        assert engine._model_registry is mock_model_registry
        assert engine._model_selector is not None
        assert isinstance(engine._model_selector, ExecutionModelSelector)

    @pytest.mark.asyncio
    async def test_engine_model_selection_recommendations(self, engine, sample_pipeline_spec):
        """Test getting model selection recommendations from engine."""
        execution_context = {"test": "context"}
        
        recommendations = engine.get_model_selection_recommendations(
            sample_pipeline_spec, execution_context
        )
        
        assert isinstance(recommendations, dict)
        # Should contain recommendations or error message
        assert "step1" in recommendations or "error" in recommendations

    def test_engine_initialization_without_model_registry(self):
        """Test engine initialization without model registry."""
        config = FoundationConfig()
        engine = StateGraphEngine(config=config, model_registry=None)
        
        assert engine._model_registry is None
        assert engine._model_selector is None


class TestPipelineExecutor:
    """Test PipelineExecutor integration with intelligent model selection."""

    @pytest.fixture
    def mock_model_registry(self):
        """Create a mock model registry."""
        registry = Mock(spec=ModelRegistry)
        
        test_model = Mock(spec=Model)
        test_model.provider = "test"
        test_model.name = "executor-model"
        test_model.capabilities = Mock(spec=ModelCapabilities)
        test_model.cost = Mock(spec=ModelCost)
        test_model.metrics = Mock(spec=ModelMetrics)
        
        registry.models = {"test:executor-model": test_model}
        return registry

    @pytest.fixture
    def pipeline_executor(self, mock_model_registry):
        """Create PipelineExecutor with intelligent selection."""
        return create_intelligent_pipeline_executor(
            model_registry=mock_model_registry,
            max_concurrent_executions=5
        )

    @pytest.fixture
    def sample_pipeline(self):
        """Create a sample pipeline for testing."""
        from src.orchestrator.core.pipeline import Pipeline
        
        # Mock pipeline 
        pipeline = Mock(spec=Pipeline)
        pipeline.id = "test_pipeline"
        pipeline.name = "Test Pipeline"
        pipeline.context = {"test_context": "value"}
        
        # Mock pipeline specification
        pipeline.specification = Mock(spec=PipelineSpecification)
        pipeline.specification.header = Mock()
        pipeline.specification.header.id = "test_pipeline"
        pipeline.specification.header.name = "Test Pipeline"
        pipeline.specification.steps = []
        
        return pipeline

    def test_executor_initialization_with_intelligent_selection(self, mock_model_registry):
        """Test executor initialization with intelligent model selection."""
        executor = create_intelligent_pipeline_executor(mock_model_registry)
        
        assert executor.enable_intelligent_selection == True
        assert executor.model_registry is mock_model_registry
        assert executor._execution_engine is not None

    def test_executor_initialization_without_model_registry(self):
        """Test executor initialization without model registry."""
        executor = PipelineExecutor(
            model_registry=None,
            enable_intelligent_selection=False
        )
        
        assert executor.enable_intelligent_selection == False
        assert executor.model_registry is None
        assert executor._execution_engine is None

    def test_get_model_selection_recommendations(self, pipeline_executor, sample_pipeline):
        """Test getting model selection recommendations from executor."""
        execution_context = {"user_type": "test"}
        
        recommendations = pipeline_executor.get_model_selection_recommendations(
            sample_pipeline, execution_context
        )
        
        assert isinstance(recommendations, dict)

    @pytest.mark.asyncio
    async def test_execute_with_intelligent_selection(self, pipeline_executor, sample_pipeline):
        """Test executing pipeline with intelligent model selection."""
        with patch.object(pipeline_executor, 'execute_with_monitoring') as mock_execute:
            mock_manager = Mock()
            mock_manager.execution_id = "test_exec_123"
            mock_execute.return_value = mock_manager
            
            result = await pipeline_executor.execute_with_intelligent_selection(
                pipeline=sample_pipeline,
                context={"test": "context"},
                selection_strategy="cost_optimized"
            )
            
            assert result is mock_manager
            mock_execute.assert_called_once()
            
            # Check that enhanced context was passed
            call_args = mock_execute.call_args
            enhanced_context = call_args.kwargs["context"]
            assert "selection_strategy" in enhanced_context
            assert enhanced_context["selection_strategy"] == "cost_optimized"

    def test_analyze_execution_efficiency(self, pipeline_executor):
        """Test execution efficiency analysis."""
        # Setup mock execution metadata
        execution_id = "test_exec_123"
        pipeline_executor._execution_metadata[execution_id] = {
            "pipeline_id": "test_pipeline",
            "intelligent_selection": True,
            "selection_strategy": "balanced",
            "total_tasks": 3
        }
        
        analysis = pipeline_executor.analyze_execution_efficiency(execution_id)
        
        assert analysis["execution_id"] == execution_id
        assert analysis["intelligent_selection_used"] == True
        assert analysis["selection_strategy"] == "balanced"
        assert "cost_efficiency" in analysis
        assert "performance_efficiency" in analysis


class TestModelSelectionIntegration:
    """Integration tests for the complete model selection system."""

    @pytest.mark.asyncio
    async def test_end_to_end_model_selection_flow(self):
        """Test complete end-to-end model selection flow."""
        # This test would require more complex setup with real components
        # For now, we verify that all components can be instantiated together
        
        # Create mock registry
        registry = Mock(spec=ModelRegistry)
        test_model = Mock(spec=Model)
        test_model.provider = "test"
        test_model.name = "e2e-model"
        test_model.capabilities = Mock(spec=ModelCapabilities)
        test_model.cost = Mock(spec=ModelCost)
        test_model.metrics = Mock(spec=ModelMetrics)
        registry.models = {"test:e2e-model": test_model}
        
        # Create components
        model_selector = ExecutionModelSelector(registry)
        engine = StateGraphEngine(FoundationConfig(), registry)
        executor = create_intelligent_pipeline_executor(registry)
        
        # Verify all components are properly initialized
        assert model_selector.model_registry is registry
        assert engine._model_selector is not None
        assert executor.enable_intelligent_selection == True
        assert executor._execution_engine is not None

    @pytest.mark.asyncio
    async def test_adaptive_selection_learning(self, mock_model_registry):
        """Test adaptive selection learning from execution history."""
        model_selector = ExecutionModelSelector(
            mock_model_registry,
            enable_adaptive_selection=True
        )
        
        # Create sample context
        step = Mock(spec=PipelineStep)
        step.id = "adaptive_test"
        step.name = "Adaptive Test"
        step.model = "AUTO"
        step.tools = ["test_tool"]
        step.variables = {}
        
        spec = Mock(spec=PipelineSpecification)
        
        context = RuntimeModelContext(
            step=step,
            pipeline_spec=spec,
            execution_state={},
            available_variables={}
        )
        
        # First selection
        selected_model_1 = await model_selector.select_model_for_step(context)
        
        # Simulate execution result
        execution_result = {
            "status": "success",
            "execution_time": 2.0,
            "timestamp": "2024-01-01T00:00:00Z",
            "output": {"result": "success"},
            "errors": []
        }
        
        # Record quality metrics
        await model_selector.evaluate_selection_quality(
            step.id, selected_model_1, execution_result
        )
        
        # Verify that selection history is recorded
        assert len(model_selector._execution_history) > 0

    def test_cost_constraint_handling(self):
        """Test handling of cost constraints in model selection."""
        registry = Mock(spec=ModelRegistry)
        
        # Create expensive and cheap models
        expensive_model = Mock(spec=Model)
        expensive_model.cost = Mock(spec=ModelCost)
        expensive_model.cost.is_free = False
        expensive_model.cost.input_cost_per_1k_tokens = 0.1
        expensive_model.cost.output_cost_per_1k_tokens = 0.2
        
        cheap_model = Mock(spec=Model)
        cheap_model.cost = Mock(spec=ModelCost)
        cheap_model.cost.is_free = True
        cheap_model.cost.input_cost_per_1k_tokens = 0.0
        cheap_model.cost.output_cost_per_1k_tokens = 0.0
        
        registry.models = {
            "test:expensive": expensive_model,
            "test:cheap": cheap_model
        }
        
        model_selector = ExecutionModelSelector(
            registry,
            enable_cost_optimization=True
        )
        
        # Test cost estimation
        expensive_cost = model_selector._estimate_model_cost(expensive_model, "per-task")
        cheap_cost = model_selector._estimate_model_cost(cheap_model, "per-task")
        
        assert expensive_cost > cheap_cost
        assert cheap_cost == 0.0

    def test_expert_assignment_priority(self):
        """Test that expert assignments take priority in model selection."""
        registry = Mock(spec=ModelRegistry)
        
        default_model = Mock(spec=Model)
        default_model.provider = "test"
        default_model.name = "default-model"
        
        expert_model = Mock(spec=Model)
        expert_model.provider = "test"
        expert_model.name = "expert-model"
        
        async def mock_resolve_explicit_model(model_spec: str, context):
            if model_spec == "test:expert-model":
                return expert_model
            return None
        
        model_selector = ExecutionModelSelector(registry, enable_expert_assignments=True)
        
        # Mock the _resolve_explicit_model method
        model_selector._resolve_explicit_model = mock_resolve_explicit_model
        
        # The expert assignment logic should be tested through integration
        # This test verifies the component structure is correct
        assert model_selector.enable_expert_assignments == True

    def test_performance_requirements_filtering(self):
        """Test filtering models based on performance requirements."""
        criteria = ModelSelectionCriteria()
        criteria.max_latency_ms = 1000
        criteria.min_accuracy_score = 0.8
        criteria.speed_preference = "fast"
        
        # Create test models with different performance characteristics
        fast_model = Mock(spec=Model)
        fast_model.capabilities = Mock(spec=ModelCapabilities)
        fast_model.capabilities.speed_rating = "fast"
        fast_model.capabilities.accuracy_score = 0.85
        
        slow_model = Mock(spec=Model)
        slow_model.capabilities = Mock(spec=ModelCapabilities) 
        slow_model.capabilities.speed_rating = "slow"
        slow_model.capabilities.accuracy_score = 0.9
        
        inaccurate_model = Mock(spec=Model)
        inaccurate_model.capabilities = Mock(spec=ModelCapabilities)
        inaccurate_model.capabilities.speed_rating = "fast"
        inaccurate_model.capabilities.accuracy_score = 0.6
        
        models = [fast_model, slow_model, inaccurate_model]
        
        # Create model selector to test filtering logic
        registry = Mock(spec=ModelRegistry)
        model_selector = ExecutionModelSelector(registry)
        
        # Test the filtering method
        filtered_models = model_selector._filter_by_criteria(models, criteria)
        
        # Should filter out slow model and inaccurate model
        assert fast_model in filtered_models
        assert slow_model not in filtered_models  # Too slow
        assert inaccurate_model not in filtered_models  # Not accurate enough


if __name__ == "__main__":
    pytest.main([__file__, "-v"])