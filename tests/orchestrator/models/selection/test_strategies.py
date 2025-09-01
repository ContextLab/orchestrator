"""Tests for model selection strategies."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.orchestrator.core.model import Model, ModelCapabilities, ModelCost, ModelRequirements
from src.orchestrator.models.registry import ModelRegistry
from src.orchestrator.models.selection.strategies import (
    TaskRequirements,
    SelectionResult,
    TaskBasedStrategy,
    CostAwareStrategy, 
    PerformanceBasedStrategy,
    WeightedStrategy,
    FallbackStrategy,
)


class MockModel(Model):
    """Mock model for testing."""
    
    def __init__(
        self,
        name: str,
        provider: str,
        capabilities: ModelCapabilities,
        cost: ModelCost,
        requirements: ModelRequirements = None,
    ):
        super().__init__(name, provider, capabilities, requirements, cost=cost)
        self._is_available = True
    
    async def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = None, **kwargs):
        return f"Generated response from {self.name}"
    
    async def generate_structured(self, prompt: str, schema: dict, temperature: float = 0.7, **kwargs):
        return {"response": f"Structured response from {self.name}"}
    
    async def health_check(self) -> bool:
        return True
    
    async def estimate_cost(self, prompt: str, max_tokens: int = None) -> float:
        return 0.001  # $0.001


@pytest.fixture
def sample_models():
    """Create sample models for testing."""
    models = []
    
    # Fast, accurate, expensive model
    models.append(MockModel(
        name="gpt-4",
        provider="openai",
        capabilities=ModelCapabilities(
            supported_tasks=["text_generation", "analysis", "code_generation"],
            context_window=8192,
            supports_function_calling=True,
            accuracy_score=0.95,
            speed_rating="medium",
        ),
        cost=ModelCost(
            input_cost_per_1k_tokens=0.03,
            output_cost_per_1k_tokens=0.06,
        ),
    ))
    
    # Fast, moderate accuracy, cheap model  
    models.append(MockModel(
        name="gpt-3.5-turbo",
        provider="openai",
        capabilities=ModelCapabilities(
            supported_tasks=["text_generation", "analysis"],
            context_window=4096,
            accuracy_score=0.85,
            speed_rating="fast",
        ),
        cost=ModelCost(
            input_cost_per_1k_tokens=0.001,
            output_cost_per_1k_tokens=0.002,
        ),
    ))
    
    # High accuracy, slow, expensive model
    models.append(MockModel(
        name="claude-3-opus",
        provider="anthropic", 
        capabilities=ModelCapabilities(
            supported_tasks=["text_generation", "analysis", "creative_writing"],
            context_window=200000,
            accuracy_score=0.98,
            speed_rating="slow",
            domains=["creative", "analysis"],
        ),
        cost=ModelCost(
            input_cost_per_1k_tokens=0.015,
            output_cost_per_1k_tokens=0.075,
        ),
    ))
    
    # Free local model
    models.append(MockModel(
        name="llama2-7b",
        provider="local",
        capabilities=ModelCapabilities(
            supported_tasks=["text_generation"],
            context_window=4096,
            accuracy_score=0.75,
            speed_rating="medium",
            code_specialized=True,
        ),
        cost=ModelCost(is_free=True),
    ))
    
    return models


@pytest.fixture
def mock_registry(sample_models):
    """Create mock registry with sample models."""
    registry = MagicMock(spec=ModelRegistry)
    registry.is_initialized = True
    
    # Mock list_models to return model info
    model_info = {}
    for model in sample_models:
        model_info[model.name] = {
            "provider": model.provider,
            "capabilities": model.capabilities.to_dict(),
            "cost": model.cost.to_dict(),
        }
    registry.list_models.return_value = model_info
    
    # Mock get_model to return the actual models
    async def get_model(name, provider):
        for model in sample_models:
            if model.name == name and model.provider == provider:
                return model
        raise ValueError(f"Model {name} not found")
    
    registry.get_model.side_effect = get_model
    
    return registry


class TestTaskBasedStrategy:
    """Test TaskBasedStrategy."""
    
    @pytest.mark.asyncio
    async def test_text_generation_selection(self, mock_registry, sample_models):
        """Test selection for text generation task."""
        strategy = TaskBasedStrategy()
        requirements = TaskRequirements(
            task_type="text_generation",
            context_window=4096,
        )
        
        result = await strategy.select_model(mock_registry, requirements, sample_models)
        
        assert isinstance(result, SelectionResult)
        assert result.model.capabilities.supports_task("text_generation")
        assert result.confidence_score > 0
        assert len(result.alternatives) <= 4
    
    @pytest.mark.asyncio 
    async def test_code_generation_selection(self, mock_registry, sample_models):
        """Test selection for code generation task."""
        strategy = TaskBasedStrategy()
        requirements = TaskRequirements(
            task_type="code_generation",
            required_capabilities=["code_specialized"],
        )
        
        result = await strategy.select_model(mock_registry, requirements, sample_models)
        
        # Should prefer the local model with code specialization
        assert result.model.capabilities.code_specialized or result.model.capabilities.supports_task("code_generation")
    
    @pytest.mark.asyncio
    async def test_large_context_requirement(self, mock_registry, sample_models):
        """Test selection with large context requirement."""
        strategy = TaskBasedStrategy()
        requirements = TaskRequirements(
            task_type="text_generation",
            context_window=100000,  # Very large context
        )
        
        result = await strategy.select_model(mock_registry, requirements, sample_models)
        
        # Should select claude-3-opus with 200k context
        assert result.model.capabilities.context_window >= 100000
    
    def test_score_model(self, sample_models):
        """Test model scoring."""
        strategy = TaskBasedStrategy()
        requirements = TaskRequirements(task_type="text_generation")
        
        for model in sample_models:
            score = strategy.score_model(model, requirements)
            assert 0 <= score <= 1
    
    def test_compatibility_check(self, sample_models):
        """Test compatibility checking."""
        strategy = TaskBasedStrategy()
        
        # Test with exclude providers
        requirements = TaskRequirements(
            task_type="text_generation",
            exclude_providers={"openai"},
        )
        
        openai_model = next(m for m in sample_models if m.provider == "openai")
        assert not strategy._is_compatible(openai_model, requirements)
        
        anthropic_model = next(m for m in sample_models if m.provider == "anthropic")
        assert strategy._is_compatible(anthropic_model, requirements)


class TestCostAwareStrategy:
    """Test CostAwareStrategy."""
    
    @pytest.mark.asyncio
    async def test_cost_optimization(self, mock_registry, sample_models):
        """Test cost-aware selection."""
        strategy = CostAwareStrategy(cost_weight=0.8)  # Heavy cost emphasis
        requirements = TaskRequirements(
            task_type="text_generation",
            budget_limit=0.005,  # Low budget
        )
        
        result = await strategy.select_model(mock_registry, requirements, sample_models)
        
        # Should prefer free or cheap models
        assert result.model.cost.is_free or result.model.cost.input_cost_per_1k_tokens <= 0.005
    
    @pytest.mark.asyncio
    async def test_budget_constraint(self, mock_registry, sample_models):
        """Test budget constraint filtering."""
        strategy = CostAwareStrategy()
        requirements = TaskRequirements(
            task_type="text_generation",
            budget_limit=0.001,  # Very tight budget
            budget_period="per-task",
        )
        
        result = await strategy.select_model(mock_registry, requirements, sample_models)
        
        # Should select a model within budget
        estimated_cost = result.model.cost.estimate_cost_for_budget_period("per-task")
        assert estimated_cost <= 0.001 or result.model.cost.is_free
    
    def test_score_model_cost_emphasis(self, sample_models):
        """Test cost-emphasized scoring."""
        strategy = CostAwareStrategy(cost_weight=0.9)
        requirements = TaskRequirements(task_type="text_generation")
        
        free_model = next(m for m in sample_models if m.cost.is_free)
        expensive_model = next(m for m in sample_models if m.cost.input_cost_per_1k_tokens > 0.01)
        
        free_score = strategy.score_model(free_model, requirements)
        expensive_score = strategy.score_model(expensive_model, requirements)
        
        # Free model should score higher due to cost emphasis
        assert free_score > expensive_score


class TestPerformanceBasedStrategy:
    """Test PerformanceBasedStrategy."""
    
    @pytest.mark.asyncio
    async def test_performance_selection(self, mock_registry, sample_models):
        """Test performance-based selection."""
        strategy = PerformanceBasedStrategy(accuracy_weight=0.8, speed_weight=0.2)
        requirements = TaskRequirements(task_type="analysis")
        
        result = await strategy.select_model(mock_registry, requirements, sample_models)
        
        # Should prefer high-accuracy models
        assert result.model.capabilities.accuracy_score >= 0.8
    
    @pytest.mark.asyncio
    async def test_speed_emphasis(self, mock_registry, sample_models):
        """Test speed-emphasized selection."""
        strategy = PerformanceBasedStrategy(accuracy_weight=0.2, speed_weight=0.8)
        requirements = TaskRequirements(
            task_type="text_generation",
            max_latency_ms=1000,  # Low latency requirement
        )
        
        result = await strategy.select_model(mock_registry, requirements, sample_models)
        
        # Should prefer fast models
        assert result.model.capabilities.speed_rating in ["fast", "medium"]
    
    def test_score_model_performance(self, sample_models):
        """Test performance scoring."""
        strategy = PerformanceBasedStrategy()
        requirements = TaskRequirements(task_type="text_generation")
        
        high_accuracy_model = max(sample_models, key=lambda m: m.capabilities.accuracy_score)
        low_accuracy_model = min(sample_models, key=lambda m: m.capabilities.accuracy_score)
        
        high_score = strategy.score_model(high_accuracy_model, requirements)
        low_score = strategy.score_model(low_accuracy_model, requirements)
        
        assert high_score > low_score


class TestWeightedStrategy:
    """Test WeightedStrategy."""
    
    @pytest.mark.asyncio
    async def test_balanced_selection(self, mock_registry, sample_models):
        """Test balanced weighted selection."""
        strategy = WeightedStrategy(
            task_weight=0.25,
            cost_weight=0.25,
            performance_weight=0.25,
            capability_weight=0.25,
        )
        requirements = TaskRequirements(task_type="text_generation")
        
        result = await strategy.select_model(mock_registry, requirements, sample_models)
        
        assert isinstance(result, SelectionResult)
        assert result.confidence_score > 0
    
    @pytest.mark.asyncio
    async def test_cost_heavy_weighting(self, mock_registry, sample_models):
        """Test cost-heavy weighted selection."""
        strategy = WeightedStrategy(
            task_weight=0.1,
            cost_weight=0.7,
            performance_weight=0.1,
            capability_weight=0.1,
        )
        requirements = TaskRequirements(task_type="text_generation")
        
        result = await strategy.select_model(mock_registry, requirements, sample_models)
        
        # Should prefer cost-effective models
        assert result.model.cost.is_free or result.model.cost.get_cost_efficiency_score() > 50
    
    def test_weight_normalization(self):
        """Test that weights are properly normalized."""
        strategy = WeightedStrategy(
            task_weight=2.0,
            cost_weight=2.0,
            performance_weight=2.0,
            capability_weight=2.0,
        )
        
        # All weights should sum to 1.0 after normalization
        total_weight = (
            strategy.task_weight +
            strategy.cost_weight +
            strategy.performance_weight +
            strategy.capability_weight
        )
        assert abs(total_weight - 1.0) < 1e-6


class TestFallbackStrategy:
    """Test FallbackStrategy."""
    
    @pytest.mark.asyncio
    async def test_successful_fallback(self, mock_registry, sample_models):
        """Test successful fallback to working strategy."""
        # Create strategy with one failing strategy
        failing_strategy = MagicMock()
        failing_strategy.select_model = AsyncMock(side_effect=Exception("Strategy failed"))
        failing_strategy.name = "failing"
        
        working_strategy = TaskBasedStrategy()
        
        fallback = FallbackStrategy(strategies=[failing_strategy, working_strategy])
        requirements = TaskRequirements(task_type="text_generation")
        
        result = await fallback.select_model(mock_registry, requirements, sample_models)
        
        assert isinstance(result, SelectionResult)
        assert "[task_based]" in result.selection_reason
    
    @pytest.mark.asyncio
    async def test_all_strategies_fail(self, mock_registry, sample_models):
        """Test behavior when all strategies fail."""
        failing_strategy1 = MagicMock()
        failing_strategy1.select_model = AsyncMock(side_effect=Exception("Strategy 1 failed"))
        failing_strategy1.name = "failing1"
        
        failing_strategy2 = MagicMock()
        failing_strategy2.select_model = AsyncMock(side_effect=Exception("Strategy 2 failed"))
        failing_strategy2.name = "failing2"
        
        fallback = FallbackStrategy(strategies=[failing_strategy1, failing_strategy2])
        requirements = TaskRequirements(task_type="text_generation")
        
        with pytest.raises(ValueError, match="All fallback strategies failed"):
            await fallback.select_model(mock_registry, requirements, sample_models)


class TestTaskRequirements:
    """Test TaskRequirements data class."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        requirements = TaskRequirements(
            task_type="text_generation",
            context_window=4096,
            max_cost_per_1k_tokens=0.01,
            required_capabilities=["function_calling"],
            exclude_providers={"provider1", "provider2"},
        )
        
        result = requirements.to_dict()
        
        assert result["task_type"] == "text_generation"
        assert result["context_window"] == 4096
        assert result["max_cost_per_1k_tokens"] == 0.01
        assert result["required_capabilities"] == ["function_calling"]
        assert set(result["exclude_providers"]) == {"provider1", "provider2"}


class TestSelectionResult:
    """Test SelectionResult data class."""
    
    def test_to_dict(self, sample_models):
        """Test conversion to dictionary."""
        model = sample_models[0]
        result = SelectionResult(
            model=model,
            provider="test_provider",
            confidence_score=0.85,
            selection_reason="Test selection",
            alternatives=[(sample_models[1], "alt_provider", 0.75)],
            estimated_cost=0.001,
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["model"]["name"] == model.name
        assert result_dict["provider"] == "test_provider"
        assert result_dict["confidence_score"] == 0.85
        assert result_dict["selection_reason"] == "Test selection"
        assert len(result_dict["alternatives"]) == 1
        assert result_dict["estimated_cost"] == 0.001