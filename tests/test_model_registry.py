"""Tests for ModelRegistry and UCBModelSelector."""

import pytest

from orchestrator.models.model_registry import (
    ModelRegistry,
    UCBModelSelector,
    ModelNotFoundError,
    NoEligibleModelsError,
)
from orchestrator.core.model import MockModel, ModelCapabilities, ModelMetrics


class TestModelRegistry:
    """Test cases for ModelRegistry class."""
    
    def test_registry_creation(self):
        """Test basic registry creation."""
        registry = ModelRegistry()
        
        assert len(registry.models) == 0
        assert isinstance(registry.model_selector, UCBModelSelector)
        assert registry._model_health_cache == {}
    
    def test_register_model(self):
        """Test registering a model."""
        registry = ModelRegistry()
        model = MockModel(name="test-model", provider="test-provider")
        
        registry.register_model(model)
        
        assert len(registry.models) == 1
        assert "test-provider:test-model" in registry.models
        assert registry.models["test-provider:test-model"] == model
    
    def test_register_model_duplicate(self):
        """Test registering duplicate model."""
        registry = ModelRegistry()
        model1 = MockModel(name="test-model", provider="test-provider")
        model2 = MockModel(name="test-model", provider="test-provider")
        
        registry.register_model(model1)
        
        with pytest.raises(ValueError, match="already registered"):
            registry.register_model(model2)
    
    def test_register_multiple_models(self):
        """Test registering multiple models."""
        registry = ModelRegistry()
        model1 = MockModel(name="model1", provider="provider1")
        model2 = MockModel(name="model2", provider="provider2")
        model3 = MockModel(name="model3", provider="provider1")
        
        registry.register_model(model1)
        registry.register_model(model2)
        registry.register_model(model3)
        
        assert len(registry.models) == 3
        assert "provider1:model1" in registry.models
        assert "provider2:model2" in registry.models
        assert "provider1:model3" in registry.models
    
    def test_unregister_model_by_name_and_provider(self):
        """Test unregistering model by name and provider."""
        registry = ModelRegistry()
        model = MockModel(name="test-model", provider="test-provider")
        
        registry.register_model(model)
        registry.unregister_model("test-model", "test-provider")
        
        assert len(registry.models) == 0
    
    def test_unregister_model_by_name_only(self):
        """Test unregistering model by name only."""
        registry = ModelRegistry()
        model = MockModel(name="test-model", provider="test-provider")
        
        registry.register_model(model)
        registry.unregister_model("test-model")
        
        assert len(registry.models) == 0
    
    def test_unregister_model_not_found(self):
        """Test unregistering non-existent model."""
        registry = ModelRegistry()
        
        with pytest.raises(ModelNotFoundError):
            registry.unregister_model("nonexistent")
    
    def test_get_model_by_name_and_provider(self):
        """Test getting model by name and provider."""
        registry = ModelRegistry()
        model = MockModel(name="test-model", provider="test-provider")
        
        registry.register_model(model)
        retrieved_model = registry.get_model("test-model", "test-provider")
        
        assert retrieved_model == model
    
    def test_get_model_by_name_only(self):
        """Test getting model by name only."""
        registry = ModelRegistry()
        model = MockModel(name="test-model", provider="test-provider")
        
        registry.register_model(model)
        retrieved_model = registry.get_model("test-model")
        
        assert retrieved_model == model
    
    def test_get_model_not_found(self):
        """Test getting non-existent model."""
        registry = ModelRegistry()
        
        with pytest.raises(ModelNotFoundError):
            registry.get_model("nonexistent")
    
    def test_get_model_ambiguous_name(self):
        """Test getting model with ambiguous name."""
        registry = ModelRegistry()
        model1 = MockModel(name="test-model", provider="provider1")
        model2 = MockModel(name="test-model", provider="provider2")
        
        registry.register_model(model1)
        registry.register_model(model2)
        
        # Should get the first one found
        retrieved_model = registry.get_model("test-model")
        assert retrieved_model in [model1, model2]
    
    def test_list_models_all(self):
        """Test listing all models."""
        registry = ModelRegistry()
        model1 = MockModel(name="model1", provider="provider1")
        model2 = MockModel(name="model2", provider="provider2")
        
        registry.register_model(model1)
        registry.register_model(model2)
        
        models = registry.list_models()
        
        assert len(models) == 2
        assert "provider1:model1" in models
        assert "provider2:model2" in models
    
    def test_list_models_by_provider(self):
        """Test listing models by provider."""
        registry = ModelRegistry()
        model1 = MockModel(name="model1", provider="provider1")
        model2 = MockModel(name="model2", provider="provider1")
        model3 = MockModel(name="model3", provider="provider2")
        
        registry.register_model(model1)
        registry.register_model(model2)
        registry.register_model(model3)
        
        provider1_models = registry.list_models("provider1")
        
        assert len(provider1_models) == 2
        assert "provider1:model1" in provider1_models
        assert "provider1:model2" in provider1_models
        assert "provider2:model3" not in provider1_models
    
    def test_list_providers(self):
        """Test listing providers."""
        registry = ModelRegistry()
        model1 = MockModel(name="model1", provider="provider1")
        model2 = MockModel(name="model2", provider="provider2")
        model3 = MockModel(name="model3", provider="provider1")
        
        registry.register_model(model1)
        registry.register_model(model2)
        registry.register_model(model3)
        
        providers = registry.list_providers()
        
        assert len(providers) == 2
        assert "provider1" in providers
        assert "provider2" in providers
    
    @pytest.mark.asyncio
    async def test_filter_by_capabilities(self):
        """Test filtering models by capabilities."""
        registry = ModelRegistry()
        
        # Model with function calling
        capabilities1 = ModelCapabilities(
            supported_tasks=["generate"],
            supports_function_calling=True
        )
        model1 = MockModel(name="model1", provider="provider1", capabilities=capabilities1)
        
        # Model without function calling
        capabilities2 = ModelCapabilities(
            supported_tasks=["generate"],
            supports_function_calling=False
        )
        model2 = MockModel(name="model2", provider="provider2", capabilities=capabilities2)
        
        registry.register_model(model1)
        registry.register_model(model2)
        
        # Test filtering for function calling
        requirements = {"supports_function_calling": True}
        eligible = await registry._filter_by_capabilities(requirements)
        
        assert len(eligible) == 1
        assert eligible[0] == model1
    
    @pytest.mark.asyncio
    async def test_filter_by_capabilities_context_window(self):
        """Test filtering models by context window."""
        registry = ModelRegistry()
        
        # Model with large context window
        capabilities1 = ModelCapabilities(
            supported_tasks=["generate"],
            context_window=16384
        )
        model1 = MockModel(name="model1", provider="provider1", capabilities=capabilities1)
        
        # Model with small context window
        capabilities2 = ModelCapabilities(
            supported_tasks=["generate"],
            context_window=4096
        )
        model2 = MockModel(name="model2", provider="provider2", capabilities=capabilities2)
        
        registry.register_model(model1)
        registry.register_model(model2)
        
        # Test filtering for large context window
        requirements = {"context_window": 8192}
        eligible = await registry._filter_by_capabilities(requirements)
        
        assert len(eligible) == 1
        assert eligible[0] == model1
    
    @pytest.mark.asyncio
    async def test_filter_by_capabilities_tasks(self):
        """Test filtering models by supported tasks."""
        registry = ModelRegistry()
        
        # Model supporting multiple tasks
        capabilities1 = ModelCapabilities(
            supported_tasks=["generate", "analyze", "transform"]
        )
        model1 = MockModel(name="model1", provider="provider1", capabilities=capabilities1)
        
        # Model supporting only one task
        capabilities2 = ModelCapabilities(
            supported_tasks=["generate"]
        )
        model2 = MockModel(name="model2", provider="provider2", capabilities=capabilities2)
        
        registry.register_model(model1)
        registry.register_model(model2)
        
        # Test filtering for analyze task
        requirements = {"tasks": ["analyze"]}
        eligible = await registry._filter_by_capabilities(requirements)
        
        assert len(eligible) == 1
        assert eligible[0] == model1
    
    @pytest.mark.asyncio
    async def test_filter_by_health(self):
        """Test filtering models by health."""
        registry = ModelRegistry()
        
        model1 = MockModel(name="model1", provider="provider1")
        model2 = MockModel(name="model2", provider="provider2")
        
        registry.register_model(model1)
        registry.register_model(model2)
        
        # Set health status
        registry._model_health_cache["provider1:model1"] = True
        registry._model_health_cache["provider2:model2"] = False
        
        healthy = await registry._filter_by_health([model1, model2])
        
        assert len(healthy) == 1
        assert healthy[0] == model1
    
    @pytest.mark.asyncio
    async def test_filter_by_health_all_healthy(self):
        """Test filtering when all models are healthy."""
        registry = ModelRegistry()
        
        model1 = MockModel(name="model1", provider="provider1")
        model2 = MockModel(name="model2", provider="provider2")
        
        registry.register_model(model1)
        registry.register_model(model2)
        
        # Set all models as healthy
        registry._model_health_cache["provider1:model1"] = True
        registry._model_health_cache["provider2:model2"] = True
        
        healthy = await registry._filter_by_health([model1, model2])
        
        assert len(healthy) == 2
        assert model1 in healthy
        assert model2 in healthy
    
    @pytest.mark.asyncio
    async def test_filter_by_health_none_healthy(self):
        """Test filtering when no models are healthy."""
        registry = ModelRegistry()
        
        model1 = MockModel(name="model1", provider="provider1")
        model2 = MockModel(name="model2", provider="provider2")
        
        registry.register_model(model1)
        registry.register_model(model2)
        
        # Set all models as unhealthy
        registry._model_health_cache["provider1:model1"] = False
        registry._model_health_cache["provider2:model2"] = False
        
        healthy = await registry._filter_by_health([model1, model2])
        
        assert len(healthy) == 0
    
    @pytest.mark.asyncio
    async def test_select_model_success(self):
        """Test successful model selection."""
        registry = ModelRegistry()
        
        capabilities = ModelCapabilities(
            supported_tasks=["generate"],
            supports_function_calling=True
        )
        model = MockModel(name="test-model", provider="test-provider", capabilities=capabilities)
        
        registry.register_model(model)
        
        # Set model as healthy
        registry._model_health_cache["test-provider:test-model"] = True
        
        requirements = {"supports_function_calling": True}
        selected = await registry.select_model(requirements)
        
        assert selected == model
    
    @pytest.mark.asyncio
    async def test_select_model_no_eligible(self):
        """Test model selection with no eligible models."""
        registry = ModelRegistry()
        
        capabilities = ModelCapabilities(
            supported_tasks=["generate"],
            supports_function_calling=False
        )
        model = MockModel(name="test-model", provider="test-provider", capabilities=capabilities)
        
        registry.register_model(model)
        
        requirements = {"supports_function_calling": True}
        
        with pytest.raises(NoEligibleModelsError):
            await registry.select_model(requirements)
    
    @pytest.mark.asyncio
    async def test_select_model_no_healthy(self):
        """Test model selection with no healthy models."""
        registry = ModelRegistry()
        
        capabilities = ModelCapabilities(
            supported_tasks=["generate"],
            supports_function_calling=True
        )
        model = MockModel(name="test-model", provider="test-provider", capabilities=capabilities)
        
        registry.register_model(model)
        
        # Set model as unhealthy
        registry._model_health_cache["test-provider:test-model"] = False
        
        requirements = {"supports_function_calling": True}
        
        with pytest.raises(NoEligibleModelsError):
            await registry.select_model(requirements)
    
    @pytest.mark.asyncio
    async def test_select_model_multiple_candidates(self):
        """Test model selection with multiple candidates."""
        registry = ModelRegistry()
        
        capabilities = ModelCapabilities(
            supported_tasks=["generate"],
            supports_function_calling=True
        )
        model1 = MockModel(name="model1", provider="provider1", capabilities=capabilities)
        model2 = MockModel(name="model2", provider="provider2", capabilities=capabilities)
        
        registry.register_model(model1)
        registry.register_model(model2)
        
        # Set all models as healthy
        registry._model_health_cache["provider1:model1"] = True
        registry._model_health_cache["provider2:model2"] = True
        
        requirements = {"supports_function_calling": True}
        selected = await registry.select_model(requirements)
        
        # Should select one of the eligible models
        assert selected in [model1, model2]
    
    def test_update_model_performance_success(self):
        """Test updating model performance with success."""
        registry = ModelRegistry()
        model = MockModel(name="test-model", provider="test-provider")
        
        registry.register_model(model)
        
        initial_success_rate = model.metrics.success_rate
        initial_latency = model.metrics.latency_p50
        initial_cost = model.metrics.cost_per_token
        
        # Update performance
        registry.update_model_performance(model, success=True, latency=1.5, cost=0.001)
        
        # Check that metrics were updated appropriately
        # Success rate should stay the same when starting at 1.0 and updating with success=True
        assert model.metrics.success_rate == initial_success_rate  # Should remain 1.0
        assert model.metrics.latency_p50 > initial_latency  # Should increase from 0
        assert model.metrics.cost_per_token > initial_cost  # Should increase from 0
    
    def test_update_model_performance_failure(self):
        """Test updating model performance with failure."""
        registry = ModelRegistry()
        model = MockModel(name="test-model", provider="test-provider")
        
        registry.register_model(model)
        
        initial_success_rate = model.metrics.success_rate
        
        # Update performance with failure
        registry.update_model_performance(model, success=False, latency=2.0, cost=0.002)
        
        # Success rate should decrease
        assert model.metrics.success_rate < initial_success_rate
    
    def test_calculate_reward_success(self):
        """Test reward calculation for successful operation."""
        registry = ModelRegistry()
        
        reward = registry._calculate_reward(success=True, latency=1.0, cost=0.001)
        
        assert 0 < reward <= 1.0
    
    def test_calculate_reward_failure(self):
        """Test reward calculation for failed operation."""
        registry = ModelRegistry()
        
        reward = registry._calculate_reward(success=False, latency=1.0, cost=0.001)
        
        assert reward == 0.0
    
    def test_calculate_reward_high_latency(self):
        """Test reward calculation with high latency."""
        registry = ModelRegistry()
        
        reward_low = registry._calculate_reward(success=True, latency=0.1, cost=0.001)
        reward_high = registry._calculate_reward(success=True, latency=5.0, cost=0.001)
        
        assert reward_low > reward_high
    
    def test_calculate_reward_high_cost(self):
        """Test reward calculation with high cost."""
        registry = ModelRegistry()
        
        reward_low = registry._calculate_reward(success=True, latency=1.0, cost=0.001)
        reward_high = registry._calculate_reward(success=True, latency=1.0, cost=0.01)
        
        assert reward_low > reward_high
    
    def test_calculate_reward_edge_cases(self):
        """Test reward calculation edge cases."""
        registry = ModelRegistry()
        
        # Very high latency should cap penalty
        reward_extreme = registry._calculate_reward(success=True, latency=100.0, cost=0.001)
        assert reward_extreme >= 0.1  # Should not go below minimum
        
        # Very high cost should cap penalty
        reward_expensive = registry._calculate_reward(success=True, latency=1.0, cost=1.0)
        assert reward_expensive >= 0.1  # Should not go below minimum
        
        # Zero values should work
        reward_zero = registry._calculate_reward(success=True, latency=0.0, cost=0.0)
        assert reward_zero == 1.0
    
    def test_update_model_metrics(self):
        """Test updating model metrics."""
        registry = ModelRegistry()
        model = MockModel(name="test-model", provider="test-provider")
        
        initial_success_rate = model.metrics.success_rate
        initial_latency = model.metrics.latency_p50
        initial_cost = model.metrics.cost_per_token
        
        registry._update_model_metrics(model, success=True, latency=2.0, cost=0.002)
        
        # Metrics should be updated appropriately
        # Success rate should stay the same when starting at 1.0 and updating with success=True  
        assert model.metrics.success_rate == initial_success_rate  # Should remain 1.0
        assert model.metrics.latency_p50 > initial_latency  # Should increase from 0
        assert model.metrics.cost_per_token > initial_cost  # Should increase from 0
    
    def test_update_model_metrics_failure(self):
        """Test updating model metrics with failure."""
        registry = ModelRegistry()
        model = MockModel(name="test-model", provider="test-provider")
        
        initial_success_rate = model.metrics.success_rate
        
        registry._update_model_metrics(model, success=False, latency=0.0, cost=0.0)
        
        # Success rate should decrease
        assert model.metrics.success_rate < initial_success_rate
    
    def test_get_model_key(self):
        """Test getting model key."""
        registry = ModelRegistry()
        model = MockModel(name="test-model", provider="test-provider")
        
        key = registry._get_model_key(model)
        
        assert key == "test-provider:test-model"
    
    def test_get_model_statistics(self):
        """Test getting model statistics."""
        registry = ModelRegistry()
        model1 = MockModel(name="model1", provider="provider1")
        model2 = MockModel(name="model2", provider="provider2")
        
        registry.register_model(model1)
        registry.register_model(model2)
        
        # Set health status
        registry._model_health_cache["provider1:model1"] = True
        registry._model_health_cache["provider2:model2"] = False
        
        stats = registry.get_model_statistics()
        
        assert stats["total_models"] == 2
        assert stats["providers"] == 2
        assert stats["healthy_models"] == 1
        assert "provider_breakdown" in stats
        assert stats["provider_breakdown"]["provider1"] == 1
        assert stats["provider_breakdown"]["provider2"] == 1
        assert "selection_stats" in stats
    
    def test_get_model_statistics_empty(self):
        """Test getting statistics for empty registry."""
        registry = ModelRegistry()
        
        stats = registry.get_model_statistics()
        
        assert stats["total_models"] == 0
        assert stats["providers"] == 0
        assert stats["healthy_models"] == 0
        assert stats["provider_breakdown"] == {}
    
    def test_reset_statistics(self):
        """Test resetting statistics."""
        registry = ModelRegistry()
        model = MockModel(name="test-model", provider="test-provider")
        
        registry.register_model(model)
        registry._model_health_cache["test-provider:test-model"] = True
        
        registry.reset_statistics()
        
        assert registry._model_health_cache == {}
        assert registry._last_health_check == 0


class TestUCBModelSelector:
    """Test cases for UCBModelSelector class."""
    
    def test_selector_creation(self):
        """Test basic selector creation."""
        selector = UCBModelSelector()
        
        assert selector.exploration_factor == 2.0
        assert selector.model_stats == {}
        assert selector.total_attempts == 0
    
    def test_selector_with_custom_exploration(self):
        """Test selector with custom exploration factor."""
        selector = UCBModelSelector(exploration_factor=1.5)
        
        assert selector.exploration_factor == 1.5
    
    def test_initialize_model(self):
        """Test initializing model in selector."""
        selector = UCBModelSelector()
        metrics = ModelMetrics(success_rate=0.8)
        
        selector.initialize_model("test-model", metrics)
        
        assert "test-model" in selector.model_stats
        assert selector.model_stats["test-model"]["attempts"] == 0
        assert selector.model_stats["test-model"]["successes"] == 0
        assert selector.model_stats["test-model"]["total_reward"] == 0.0
        assert selector.model_stats["test-model"]["average_reward"] == 0.8
    
    def test_initialize_multiple_models(self):
        """Test initializing multiple models."""
        selector = UCBModelSelector()
        metrics1 = ModelMetrics(success_rate=0.9)
        metrics2 = ModelMetrics(success_rate=0.7)
        
        selector.initialize_model("model1", metrics1)
        selector.initialize_model("model2", metrics2)
        
        assert len(selector.model_stats) == 2
        assert selector.model_stats["model1"]["average_reward"] == 0.9
        assert selector.model_stats["model2"]["average_reward"] == 0.7
    
    def test_select_single_model(self):
        """Test selecting single model."""
        selector = UCBModelSelector()
        
        selected = selector.select(["model1"], {})
        
        assert selected == "model1"
        assert selector.model_stats["model1"]["attempts"] == 1
        assert selector.total_attempts == 1
    
    def test_select_multiple_models_first_time(self):
        """Test selecting from multiple models for first time."""
        selector = UCBModelSelector()
        
        # First selection should pick any model (all have infinite score)
        selected = selector.select(["model1", "model2", "model3"], {})
        
        assert selected in ["model1", "model2", "model3"]
        assert selector.model_stats[selected]["attempts"] == 1
        assert selector.total_attempts == 1
    
    def test_select_with_history(self):
        """Test selection with performance history."""
        selector = UCBModelSelector()
        
        # Initialize models with different performance
        metrics1 = ModelMetrics(success_rate=0.9)
        metrics2 = ModelMetrics(success_rate=0.7)
        
        selector.initialize_model("model1", metrics1)
        selector.initialize_model("model2", metrics2)
        
        # Simulate some attempts
        selector.model_stats["model1"]["attempts"] = 10
        selector.model_stats["model1"]["total_reward"] = 8.0
        selector.model_stats["model1"]["average_reward"] = 0.8
        
        selector.model_stats["model2"]["attempts"] = 10
        selector.model_stats["model2"]["total_reward"] = 5.0
        selector.model_stats["model2"]["average_reward"] = 0.5
        
        selector.total_attempts = 20
        
        selected = selector.select(["model1", "model2"], {})
        
        # model1 should be selected due to higher average reward
        assert selected == "model1"
    
    def test_select_exploration_vs_exploitation(self):
        """Test exploration vs exploitation behavior."""
        selector = UCBModelSelector(exploration_factor=0.1)  # Low exploration
        
        # Initialize model with some history
        selector.model_stats["experienced_model"] = {
            "attempts": 100,
            "successes": 80,
            "total_reward": 80.0,
            "average_reward": 0.8
        }
        selector.total_attempts = 100
        
        # Select between experienced model and new model
        selected = selector.select(["experienced_model", "new_model"], {})
        
        # With low exploration, should often pick experienced model
        # But new model should get picked sometimes due to infinite UCB score
        assert selected in ["experienced_model", "new_model"]
    
    def test_select_no_models(self):
        """Test selection with no models."""
        selector = UCBModelSelector()
        
        with pytest.raises(ValueError, match="No models available"):
            selector.select([], {})
    
    def test_update_reward_positive(self):
        """Test updating reward with positive value."""
        selector = UCBModelSelector()
        
        # Initialize model
        selector.initialize_model("test-model", ModelMetrics())
        
        # Update reward
        selector.update_reward("test-model", 0.8)
        
        stats = selector.model_stats["test-model"]
        assert stats["total_reward"] == 0.8
        assert stats["successes"] == 1
        assert stats["average_reward"] == 0.8
    
    def test_update_reward_zero(self):
        """Test updating with zero reward."""
        selector = UCBModelSelector()
        
        # Initialize model
        selector.initialize_model("test-model", ModelMetrics())
        
        # Update with zero reward
        selector.update_reward("test-model", 0.0)
        
        stats = selector.model_stats["test-model"]
        assert stats["total_reward"] == 0.0
        assert stats["successes"] == 0
        assert stats["average_reward"] == 0.0
    
    def test_update_reward_multiple_times(self):
        """Test updating reward multiple times."""
        selector = UCBModelSelector()
        
        # Initialize model
        selector.initialize_model("test-model", ModelMetrics())
        
        # Update multiple times
        selector.update_reward("test-model", 0.8)
        selector.update_reward("test-model", 0.6)
        selector.update_reward("test-model", 0.0)
        
        stats = selector.model_stats["test-model"]
        assert stats["total_reward"] == 1.4  # 0.8 + 0.6 + 0.0
        assert stats["successes"] == 2  # Two positive rewards
        assert stats["average_reward"] == 1.4 / 3  # Total / attempts
    
    def test_update_reward_nonexistent_model(self):
        """Test updating reward for non-existent model."""
        selector = UCBModelSelector()
        
        # Should not raise error
        selector.update_reward("nonexistent", 0.8)
        
        # Should not create entry
        assert "nonexistent" not in selector.model_stats
    
    def test_remove_model(self):
        """Test removing model from selector."""
        selector = UCBModelSelector()
        
        # Initialize model
        selector.initialize_model("test-model", ModelMetrics())
        
        # Remove model
        selector.remove_model("test-model")
        
        assert "test-model" not in selector.model_stats
    
    def test_remove_nonexistent_model(self):
        """Test removing non-existent model."""
        selector = UCBModelSelector()
        
        # Should not raise error
        selector.remove_model("nonexistent")
    
    def test_get_statistics_empty(self):
        """Test getting statistics for empty selector."""
        selector = UCBModelSelector()
        
        stats = selector.get_statistics()
        
        assert stats["total_attempts"] == 0
        assert stats["models_tracked"] == 0
        assert stats["model_performance"] == {}
    
    def test_get_statistics_with_data(self):
        """Test getting selection statistics with data."""
        selector = UCBModelSelector()
        
        # Initialize and update model
        selector.initialize_model("test-model", ModelMetrics())
        selector.model_stats["test-model"]["attempts"] = 5
        selector.model_stats["test-model"]["successes"] = 3
        selector.model_stats["test-model"]["total_reward"] = 2.1
        selector.model_stats["test-model"]["average_reward"] = 0.7
        selector.total_attempts = 5
        
        stats = selector.get_statistics()
        
        assert stats["total_attempts"] == 5
        assert stats["models_tracked"] == 1
        assert "model_performance" in stats
        assert stats["model_performance"]["test-model"]["attempts"] == 5
        assert stats["model_performance"]["test-model"]["successes"] == 3
        assert stats["model_performance"]["test-model"]["success_rate"] == 0.6
        assert stats["model_performance"]["test-model"]["average_reward"] == 0.7
    
    def test_get_statistics_multiple_models(self):
        """Test getting statistics for multiple models."""
        selector = UCBModelSelector()
        
        # Initialize multiple models
        selector.initialize_model("model1", ModelMetrics())
        selector.initialize_model("model2", ModelMetrics())
        
        # Update stats
        selector.model_stats["model1"]["attempts"] = 10
        selector.model_stats["model1"]["successes"] = 8
        selector.model_stats["model2"]["attempts"] = 5
        selector.model_stats["model2"]["successes"] = 2
        selector.total_attempts = 15
        
        stats = selector.get_statistics()
        
        assert stats["total_attempts"] == 15
        assert stats["models_tracked"] == 2
        assert len(stats["model_performance"]) == 2
        assert stats["model_performance"]["model1"]["success_rate"] == 0.8
        assert stats["model_performance"]["model2"]["success_rate"] == 0.4
    
    def test_reset_statistics(self):
        """Test resetting selection statistics."""
        selector = UCBModelSelector()
        
        # Initialize and update model
        selector.initialize_model("test-model", ModelMetrics())
        selector.model_stats["test-model"]["attempts"] = 5
        selector.model_stats["test-model"]["successes"] = 3
        selector.model_stats["test-model"]["total_reward"] = 2.1
        selector.total_attempts = 5
        
        selector.reset_statistics()
        
        assert selector.total_attempts == 0
        assert selector.model_stats["test-model"]["attempts"] == 0
        assert selector.model_stats["test-model"]["successes"] == 0
        assert selector.model_stats["test-model"]["total_reward"] == 0.0
        assert selector.model_stats["test-model"]["average_reward"] == 0.5
    
    def test_get_model_confidence_nonexistent(self):
        """Test getting confidence for non-existent model."""
        selector = UCBModelSelector()
        
        confidence = selector.get_model_confidence("nonexistent")
        assert confidence == 0.0
    
    def test_get_model_confidence_no_attempts(self):
        """Test getting confidence for model with no attempts."""
        selector = UCBModelSelector()
        
        selector.initialize_model("test-model", ModelMetrics())
        confidence = selector.get_model_confidence("test-model")
        assert confidence == 0.0
    
    def test_get_model_confidence_with_attempts(self):
        """Test getting confidence for model with attempts."""
        selector = UCBModelSelector()
        
        selector.initialize_model("test-model", ModelMetrics())
        selector.model_stats["test-model"]["attempts"] = 5
        selector.model_stats["test-model"]["average_reward"] = 0.8
        
        confidence = selector.get_model_confidence("test-model")
        assert 0 < confidence <= 1.0
    
    def test_get_model_confidence_max_attempts(self):
        """Test model confidence with max attempts."""
        selector = UCBModelSelector()
        
        selector.initialize_model("test-model", ModelMetrics())
        selector.model_stats["test-model"]["attempts"] = 15  # More than cap of 10
        selector.model_stats["test-model"]["average_reward"] = 0.9
        
        confidence = selector.get_model_confidence("test-model")
        
        # Should cap at 10 attempts
        assert confidence == 0.9  # 1.0 * 0.9
    
    def test_get_model_confidence_progression(self):
        """Test confidence progression with more attempts."""
        selector = UCBModelSelector()
        
        selector.initialize_model("test-model", ModelMetrics())
        selector.model_stats["test-model"]["average_reward"] = 0.8
        
        # Confidence should increase with more attempts
        selector.model_stats["test-model"]["attempts"] = 1
        confidence1 = selector.get_model_confidence("test-model")
        
        selector.model_stats["test-model"]["attempts"] = 5
        confidence5 = selector.get_model_confidence("test-model")
        
        selector.model_stats["test-model"]["attempts"] = 10
        confidence10 = selector.get_model_confidence("test-model")
        
        assert confidence1 < confidence5 < confidence10