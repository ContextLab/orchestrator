"""
Test Category: Enhanced Model Selector
Real tests for the enhanced model selection logic with new scoring criteria.
"""

import pytest
from unittest.mock import MagicMock
from orchestrator.models.model_selector import ModelSelector, ModelSelectionCriteria
from orchestrator.models.model_registry import ModelRegistry
from orchestrator.core.model import Model, ModelCapabilities, ModelCost, ModelMetrics


class TestEnhancedModelSelector:
    """Test enhanced model selection with new scoring criteria."""

    @pytest.fixture
    def mock_models(self):
        """Create mock models with different characteristics for testing."""
        models = {}
        
        # Budget model - cheap but limited
        budget_model = MagicMock(spec=Model)
        budget_model.name = "gemma3-1b"
        budget_model.provider = "ollama"
        budget_model._expertise = ["fast", "compact"]
        budget_model._size_billions = 1.0
        budget_model.meets_requirements.return_value = True
        budget_model.capabilities = ModelCapabilities(
            supported_tasks=["text-generation"],
            accuracy_score=0.7,
            speed_rating="fast"
        )
        budget_model.cost = ModelCost(is_free=True)
        budget_model.metrics = ModelMetrics(success_rate=0.85, throughput=50)
        models["ollama:gemma3-1b"] = budget_model
        
        # Balanced model - good general purpose
        balanced_model = MagicMock(spec=Model)
        balanced_model.name = "gpt-4o-mini"
        balanced_model.provider = "openai"
        balanced_model._expertise = ["general"]
        balanced_model._size_billions = 8.0
        balanced_model.meets_requirements.return_value = True
        balanced_model.capabilities = ModelCapabilities(
            supported_tasks=["text-generation", "chat"],
            accuracy_score=0.85,
            speed_rating="medium"
        )
        balanced_model.cost = ModelCost(
            input_cost_per_1k_tokens=0.00015,
            output_cost_per_1k_tokens=0.0006
        )
        balanced_model.metrics = ModelMetrics(success_rate=0.92, throughput=30)
        models["openai:gpt-4o-mini"] = balanced_model
        
        # High-end model - expensive but powerful
        premium_model = MagicMock(spec=Model) 
        premium_model.name = "claude-sonnet-4"
        premium_model.provider = "anthropic"
        premium_model._expertise = ["analysis", "research", "reasoning"]
        premium_model._size_billions = 200.0
        premium_model.meets_requirements.return_value = True
        premium_model.capabilities = ModelCapabilities(
            supported_tasks=["text-generation", "analysis", "research"],
            accuracy_score=0.95,
            speed_rating="slow"
        )
        premium_model.cost = ModelCost(
            input_cost_per_1k_tokens=0.003,
            output_cost_per_1k_tokens=0.015
        )
        premium_model.metrics = ModelMetrics(success_rate=0.98, throughput=15)
        models["anthropic:claude-sonnet-4"] = premium_model
        
        # Code specialist
        code_model = MagicMock(spec=Model)
        code_model.name = "deepseek-r1-32b"
        code_model.provider = "ollama"
        code_model._expertise = ["code", "reasoning", "math"]
        code_model._size_billions = 32.0
        code_model.meets_requirements.return_value = True
        code_model.capabilities = ModelCapabilities(
            supported_tasks=["text-generation", "code-generation", "reasoning"],
            code_specialized=True,
            accuracy_score=0.92,
            speed_rating="medium"
        )
        code_model.cost = ModelCost(is_free=True)
        code_model.metrics = ModelMetrics(success_rate=0.90, throughput=25)
        models["ollama:deepseek-r1-32b"] = code_model
        
        return models

    @pytest.fixture
    def registry(self, mock_models):
        """Create registry with mock models."""
        registry = ModelRegistry()
        registry.models = mock_models
        return registry

    @pytest.fixture
    def selector(self, registry):
        """Create model selector with registry."""
        return ModelSelector(registry)

    @pytest.mark.asyncio
    async def test_expertise_level_selection(self, selector):
        """Test selection based on expertise level requirements."""
        # Low expertise requirement should prefer fast models
        criteria = ModelSelectionCriteria(expertise="low")
        model = await selector.select_model(criteria)
        assert model.name == "gemma3-1b"  # Fast, compact model
        
        # High expertise requirement should prefer specialized models
        criteria = ModelSelectionCriteria(expertise="high")
        model = await selector.select_model(criteria)
        assert model.name in ["deepseek-r1-32b", "claude-sonnet-4"]  # Code or analysis models
        
        # Very high expertise should prefer analysis models
        criteria = ModelSelectionCriteria(expertise="very-high")
        model = await selector.select_model(criteria)
        assert model.name == "claude-sonnet-4"  # Analysis/research model

    @pytest.mark.asyncio
    async def test_cost_constraint_selection(self, selector):
        """Test selection based on cost constraints."""
        # Very tight budget should prefer free models
        criteria = ModelSelectionCriteria(
            cost_limit=0.01,
            budget_period="per-task"
        )
        model = await selector.select_model(criteria)
        assert model.cost.is_free  # Should select free model
        
        # Higher budget allows paid models
        criteria = ModelSelectionCriteria(
            cost_limit=5.0,
            budget_period="per-task"
        )
        model = await selector.select_model(criteria)
        # Could be any model within budget

    @pytest.mark.asyncio
    async def test_modality_selection(self, selector):
        """Test selection based on modality requirements."""
        # Code modality should prefer code-specialized models
        criteria = ModelSelectionCriteria(modalities=["code"])
        model = await selector.select_model(criteria)
        assert model.capabilities.code_specialized or "code" in model._expertise

    @pytest.mark.asyncio
    async def test_size_preference_selection(self, selector):
        """Test selection based on size preferences."""
        # Minimum size requirement
        criteria = ModelSelectionCriteria(min_model_size=30.0)
        model = await selector.select_model(criteria)
        assert model._size_billions >= 30.0
        
        # Size range
        criteria = ModelSelectionCriteria(
            min_model_size=5.0,
            max_model_size=50.0
        )
        model = await selector.select_model(criteria)
        assert 5.0 <= model._size_billions <= 50.0

    @pytest.mark.asyncio
    async def test_performance_selection(self, selector):
        """Test selection based on performance requirements."""
        # Throughput requirement
        criteria = ModelSelectionCriteria(min_tokens_per_second=30)
        model = await selector.select_model(criteria)
        assert model.metrics.throughput >= 30

    @pytest.mark.asyncio
    async def test_fallback_strategies(self, selector):
        """Test different fallback strategies."""
        # Impossible requirements to trigger fallback
        criteria = ModelSelectionCriteria(
            min_model_size=1000.0,  # No model this large
            fallback_strategy="cheapest"
        )
        model = await selector.select_model(criteria)
        assert model.cost.is_free  # Should fallback to cheapest (free) model
        
        # Best available fallback
        criteria = ModelSelectionCriteria(
            min_model_size=1000.0,
            fallback_strategy="best_available"
        )
        model = await selector.select_model(criteria)
        # Should select high-quality model (highest accuracy)
        assert model.capabilities.accuracy_score > 0.9

    @pytest.mark.asyncio
    async def test_selection_strategy_differences(self, selector):
        """Test different selection strategies produce different results."""
        base_criteria = ModelSelectionCriteria()
        
        # Cost optimized should prefer cheaper models
        criteria = ModelSelectionCriteria(selection_strategy="cost_optimized")
        cost_model = await selector.select_model(criteria)
        
        # Accuracy optimized should prefer more accurate models
        criteria = ModelSelectionCriteria(selection_strategy="accuracy_optimized")
        accuracy_model = await selector.select_model(criteria)
        
        # Performance optimized should prefer faster models
        criteria = ModelSelectionCriteria(selection_strategy="performance_optimized")
        perf_model = await selector.select_model(criteria)
        
        # They might select different models based on strategy
        # At minimum, accuracy model should have highest accuracy
        assert accuracy_model.capabilities.accuracy_score >= cost_model.capabilities.accuracy_score

    @pytest.mark.asyncio
    async def test_preferred_models_bonus(self, selector):
        """Test that preferred models receive scoring bonuses."""
        # Without preference
        criteria = ModelSelectionCriteria()
        model1 = await selector.select_model(criteria)
        
        # With specific model preference  
        criteria = ModelSelectionCriteria(
            preferred_models=["anthropic:claude-sonnet-4"]
        )
        model2 = await selector.select_model(criteria)
        
        # Should prefer the specified model
        assert model2.name == "claude-sonnet-4"

    def test_scoring_helper_methods(self, selector):
        """Test the individual scoring helper methods."""
        # Get a test model
        test_model = list(selector.registry.models.values())[0]
        
        # Test expertise scoring
        score = selector._score_expertise_match(test_model, "low")
        assert isinstance(score, float)
        assert score >= 0.0
        
        # Test modality scoring
        score = selector._score_modality_match(test_model, ["text"])
        assert isinstance(score, float)
        assert score >= 0.0  # All models support text
        
        # Test cost estimation
        cost = selector._estimate_model_cost(test_model, "per-task")
        assert isinstance(cost, float)
        assert cost >= 0.0

    @pytest.mark.asyncio
    async def test_complex_requirements_combination(self, selector):
        """Test selection with multiple combined requirements."""
        # Combine multiple criteria
        criteria = ModelSelectionCriteria(
            expertise="high",
            modalities=["code"],
            max_model_size=100.0,
            cost_limit=1.0,
            budget_period="per-task",
            min_tokens_per_second=20,
            selection_strategy="balanced"
        )
        
        model = await selector.select_model(criteria)
        
        # Verify it meets the key requirements
        assert model._size_billions <= 100.0
        assert model.metrics.throughput >= 20 or model.metrics.throughput == 0  # 0 means no data
        # Should be a code-capable model
        assert model.capabilities.code_specialized or "code" in model._expertise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])