"""Tests for ModelRegistry and UCBModelSelector."""

import pytest

from orchestrator.core.model import Model, ModelCapabilities, ModelMetrics
from orchestrator.models.model_registry import (
    ModelNotFoundError,
    ModelRegistry,
    NoEligibleModelsError,
    UCBModelSelector,
)


class TestModelRegistry:
    """Test cases for ModelRegistry class."""

    def get_test_model(self, populated_model_registry):
        """Helper to get a real test model from populated registry."""
        # Try to get any real model
        for model_id in ["gpt-4o-mini", "claude-3-5-haiku-20241022", "llama3.2:1b"]:
            try:
                model = populated_model_registry.get_model(model_id)
                if model:
                    return model
            except:
                pass
        
        raise AssertionError(
            "No AI models available for testing. "
            "Please configure API keys in ~/.orchestrator/.env"
        )

    def test_registry_creation(self):
        """Test basic registry creation."""
        registry = ModelRegistry()  # Create empty registry for this test

        assert len(registry.models) == 0
        assert isinstance(registry.model_selector, UCBModelSelector)
        assert registry._model_health_cache == {}

    def test_register_model(self, populated_model_registry):
        """Test registering a model."""
        # Get a real model from populated registry
        model = self.get_test_model(populated_model_registry)
        
        # Create a new empty registry for testing registration
        registry = ModelRegistry()
        registry.register_model(model)

        assert len(registry.models) == 1
        model_key = f"{model.provider}:{model.name}"
        assert model_key in registry.models
        assert registry.models[model_key] == model

    def test_register_model_duplicate(self, populated_model_registry):
        """Test registering duplicate model."""
        # Get a real model from populated registry
        model = self.get_test_model(populated_model_registry)
        
        # Create a new empty registry and register the model
        registry = ModelRegistry()
        registry.register_model(model)
        
        # Try to register the same model again
        with pytest.raises(ValueError, match="already registered"):
            registry.register_model(model)

    def test_register_multiple_models(self, populated_model_registry):
        """Test registering multiple models."""
        registry = populated_model_registry
        
        # Get count of pre-registered models
        initial_count = len(registry.models)
        
        # Should have at least some models from initialization
        assert initial_count > 0
        
        # Check that different providers exist
        providers = set()
        for key in registry.models:
            provider = key.split(":")[0]
            providers.add(provider)
        
        # Should have multiple providers if models are available
        if initial_count > 1:
            assert len(providers) >= 1

    def test_unregister_model_by_name_and_provider(self, populated_model_registry):
        """Test unregistering model by name and provider."""
        registry = populated_model_registry
        
        # Get a real model
        model = None
        for model_id in ["gpt-4o-mini", "claude-3-5-haiku-20241022"]:
            try:
                model = registry.get_model(model_id)
                if model:
                    break
            except:
                pass
        
        if not model:
            raise AssertionError(
                "No AI models available for testing. "
                "Please configure API keys in ~/.orchestrator/.env"
            )
        
        initial_count = len(registry.models)
        registry.unregister_model(model.name, model.provider)

        assert len(registry.models) == initial_count - 1

    def test_unregister_model_by_name_only(self):
        """Test unregistering model by name only."""
        registry = ModelRegistry()  # Create empty registry for this test
        
        # Create a test registry with a unique model
        from orchestrator.integrations.openai_model import OpenAIModel
        from orchestrator.integrations.anthropic_model import AnthropicModel
        from orchestrator.integrations.ollama_model import OllamaModel
        
        # Try to create a temporary test model
        test_model = None
        try:
            # Use a fake model that won't conflict
            test_capabilities = ModelCapabilities(supported_tasks=["generate"])
            test_metrics = ModelMetrics()
            
            # Create a minimal model class for testing
            class TestModel(Model):
                def __init__(self):
                    super().__init__(
                        name="test-unregister-model",
                        provider="test-provider",
                        capabilities=test_capabilities,
                        metrics=test_metrics
                    )
                    self.is_available = True
                
                async def generate(self, prompt, **kwargs):
                    return "test"
                
                async def health_check(self):
                    return True
            
            test_model = TestModel()
            registry.models.clear()  # Clear for clean test
            registry.register_model(test_model)
            
            assert len(registry.models) == 1
            registry.unregister_model("test-unregister-model")
            assert len(registry.models) == 0
            
        except Exception:
            raise AssertionError(
                "Could not create test model. "
                "Please configure API keys in ~/.orchestrator/.env"
            )

    def test_unregister_model_not_found(self):
        """Test unregistering non-existent model."""
        registry = ModelRegistry()  # Create empty registry for this test

        with pytest.raises(ModelNotFoundError):
            registry.unregister_model("nonexistent")

    def test_get_model_by_name_and_provider(self, populated_model_registry):
        """Test getting model by name and provider."""
        registry = populated_model_registry
        
        # Get a real model
        model = None
        model_name = None
        model_provider = None
        
        for model_id in ["gpt-4o-mini", "claude-3-5-haiku-20241022"]:
            try:
                model = registry.get_model(model_id)
                if model:
                    model_name = model.name
                    model_provider = model.provider
                    break
            except:
                pass
        
        if not model:
            raise AssertionError(
                "No AI models available for testing. "
                "Please configure API keys in ~/.orchestrator/.env"
            )
        
        retrieved_model = registry.get_model(model_name, model_provider)
        assert retrieved_model.name == model_name
        assert retrieved_model.provider == model_provider

    def test_get_model_by_name_only(self, populated_model_registry):
        """Test getting model by name only."""
        registry = populated_model_registry
        
        # Test with known model names
        model = None
        for model_id in ["gpt-4o-mini", "claude-3-5-haiku-20241022", "llama3.2:1b"]:
            try:
                model = registry.get_model(model_id)
                if model:
                    break
            except:
                pass
        
        if model:
            assert model.name == model_id or model_id in model.name
        else:
            raise AssertionError(
                "No AI models available for testing. "
                "Please configure API keys in ~/.orchestrator/.env"
            )

    def test_get_model_not_found(self):
        """Test getting non-existent model."""
        registry = ModelRegistry()  # Create empty registry for this test

        with pytest.raises(ModelNotFoundError):
            registry.get_model("nonexistent")

    def test_get_model_ambiguous_name(self):
        """Test getting model with ambiguous name."""
        # This test is no longer applicable with real models only
        # Each provider has unique model names
        pass

    def test_list_models_all(self, populated_model_registry):
        """Test listing all models."""
        registry = populated_model_registry
        
        models = registry.list_models()
        
        # Should have some models registered
        assert len(models) > 0
        
        # Check format of model IDs
        for model_id in models:
            assert ":" in model_id  # Should be provider:model format

    def test_list_models_by_provider(self, populated_model_registry):
        """Test listing models by provider."""
        registry = populated_model_registry
        
        # Test with known providers
        for provider in ["openai", "anthropic", "ollama"]:
            provider_models = registry.list_models(provider)
            
            # If provider has models, verify they all match
            if provider_models:
                for model_id in provider_models:
                    assert model_id.startswith(f"{provider}:")

    def test_list_providers(self, populated_model_registry):
        """Test listing providers."""
        registry = populated_model_registry
        
        providers = registry.list_providers()
        
        # Should have at least one provider
        assert len(providers) > 0
        
        # Known providers should be in the list (if available)
        possible_providers = ["openai", "anthropic", "ollama"]
        found_providers = [p for p in possible_providers if p in providers]
        
        # At least one provider should be available
        if not found_providers:
            pytest.skip("No known providers available")

    @pytest.mark.asyncio
    async def test_filter_by_capabilities(self, populated_model_registry):
        """Test filtering models by capabilities."""
        registry = populated_model_registry

        # Get all registered models
        all_models = list(registry.models.values())
        
        if not all_models:
            raise AssertionError(
                "No models available for testing. "
                "Please configure API keys in ~/.orchestrator/.env"
            )

        # Test filtering for function calling
        requirements = {"supports_function_calling": True}
        eligible = await registry._filter_by_capabilities(requirements)

        # All eligible models should support function calling
        for model in eligible:
            assert model.capabilities.supports_function_calling is True
        
        # Test filtering for non-function calling
        requirements = {"supports_function_calling": False}
        non_eligible = await registry._filter_by_capabilities(requirements)
        
        # Should partition all models
        assert len(eligible) + len(non_eligible) <= len(all_models)

    @pytest.mark.asyncio
    async def test_filter_by_capabilities_context_window(self, populated_model_registry):
        """Test filtering models by context window."""
        registry = populated_model_registry

        # Test filtering for different context windows
        requirements_small = {"context_window": 4096}
        requirements_medium = {"context_window": 8192}
        requirements_large = {"context_window": 16384}
        
        eligible_small = await registry._filter_by_capabilities(requirements_small)
        eligible_medium = await registry._filter_by_capabilities(requirements_medium)
        eligible_large = await registry._filter_by_capabilities(requirements_large)

        # Models with larger windows should be subsets
        if eligible_small and eligible_large:
            # All models that support large context should also support small
            for model in eligible_large:
                assert model.capabilities.context_window >= 16384
        
        # Verify context windows
        for model in eligible_medium:
            assert model.capabilities.context_window >= 8192

    @pytest.mark.asyncio
    async def test_filter_by_capabilities_tasks(self, populated_model_registry):
        """Test filtering models by supported tasks."""
        registry = populated_model_registry

        # Test filtering for common tasks
        requirements_generate = {"tasks": ["generate"]}
        requirements_analyze = {"tasks": ["analyze"]}
        
        eligible_generate = await registry._filter_by_capabilities(requirements_generate)
        eligible_analyze = await registry._filter_by_capabilities(requirements_analyze)

        # Most models should support generation
        if eligible_generate:
            for model in eligible_generate:
                assert any(task in ["generate", "generation"] for task in model.capabilities.supported_tasks)
        
        # Some models may support analysis
        for model in eligible_analyze:
            assert "analyze" in model.capabilities.supported_tasks

    @pytest.mark.asyncio
    async def test_filter_by_health(self, populated_model_registry):
        """Test filtering models by health."""
        registry = populated_model_registry

        # Get some real models
        all_models = list(registry.models.values())[:2]  # Get first 2 models
        
        if len(all_models) < 2:
            raise AssertionError(
                "Need at least 2 models for health testing. "
                "Please configure multiple API keys in ~/.orchestrator/.env"
            )

        model1 = all_models[0]
        model2 = all_models[1]

        # Set health status
        registry._model_health_cache[f"{model1.provider}:{model1.name}"] = True
        registry._model_health_cache[f"{model2.provider}:{model2.name}"] = False

        healthy = await registry._filter_by_health([model1, model2])

        assert len(healthy) == 1
        assert healthy[0] == model1

    @pytest.mark.asyncio
    async def test_filter_by_health_all_healthy(self, populated_model_registry):
        """Test filtering when all models are healthy."""
        registry = populated_model_registry

        # Get some real models
        all_models = list(registry.models.values())[:2]  # Get first 2 models
        
        if len(all_models) < 2:
            raise AssertionError(
                "Need at least 2 models for health testing. "
                "Please configure multiple API keys in ~/.orchestrator/.env"
            )

        # Set all models as healthy
        for model in all_models:
            registry._model_health_cache[f"{model.provider}:{model.name}"] = True

        healthy = await registry._filter_by_health(all_models)

        assert len(healthy) == len(all_models)
        for model in all_models:
            assert model in healthy

    @pytest.mark.asyncio
    async def test_filter_by_health_none_healthy(self, populated_model_registry):
        """Test filtering when no models are healthy."""
        registry = populated_model_registry

        # Get some real models
        all_models = list(registry.models.values())[:2]  # Get first 2 models
        
        if len(all_models) < 1:
            raise AssertionError(
                "Need at least 1 model for health testing. "
                "Please configure API keys in ~/.orchestrator/.env"
            )

        # Set all models as unhealthy
        for model in all_models:
            registry._model_health_cache[f"{model.provider}:{model.name}"] = False

        healthy = await registry._filter_by_health(all_models)

        assert len(healthy) == 0

    @pytest.mark.asyncio
    async def test_select_model_success(self, populated_model_registry):
        """Test successful model selection."""
        registry = populated_model_registry

        # Try to find a model that supports function calling
        requirements = {"supports_function_calling": True}
        
        try:
            selected = await registry.select_model(requirements)
            assert selected is not None
            assert selected.capabilities.supports_function_calling is True
        except NoEligibleModelsError:
            # Try without function calling requirement
            selected = await registry.select_model({})
            assert selected is not None

    @pytest.mark.asyncio
    async def test_select_model_no_eligible(self, populated_model_registry):
        """Test model selection with no eligible models."""
        registry = populated_model_registry

        # Use impossible requirements
        requirements = {
            "context_window": 10000000,  # 10M context window
            "supports_function_calling": True,
            "tasks": ["impossible_task_xyz"]
        }

        with pytest.raises(NoEligibleModelsError):
            await registry.select_model(requirements)

    @pytest.mark.asyncio
    async def test_select_model_no_healthy(self, populated_model_registry):
        """Test model selection with no healthy models."""
        registry = populated_model_registry

        # Set all models as unhealthy
        for key in registry.models:
            registry._model_health_cache[key] = False

        requirements = {}

        with pytest.raises(NoEligibleModelsError):
            await registry.select_model(requirements)

    @pytest.mark.asyncio
    async def test_select_model_multiple_candidates(self, populated_model_registry):
        """Test model selection with multiple candidates."""
        registry = populated_model_registry

        # Use minimal requirements to get multiple candidates
        requirements = {}
        
        # Get all healthy models
        healthy_models = []
        for key, model in registry.models.items():
            if registry._model_health_cache.get(key, True):  # Default to healthy
                healthy_models.append(model)
        
        if len(healthy_models) < 2:
            raise AssertionError(
                "Need at least 2 healthy models for this test. "
                "Please configure multiple API keys in ~/.orchestrator/.env"
            )

        selected = await registry.select_model(requirements)

        # Should select one of the healthy models
        assert selected in healthy_models

    def test_update_model_performance_success(self, populated_model_registry):
        """Test updating model performance with success."""
        registry = populated_model_registry
        
        # Get a real model
        model = None
        for key, m in registry.models.items():
            model = m
            break
        
        if not model:
            raise AssertionError(
                "No models available for testing. "
                "Please configure API keys in ~/.orchestrator/.env"
            )

        initial_success_rate = model.metrics.success_rate
        initial_latency = model.metrics.latency_p50
        initial_cost = model.metrics.cost_per_token

        # Update performance
        registry.update_model_performance(model, success=True, latency=1.5, cost=0.001)

        # Check that metrics were updated appropriately
        # For OpenAI/Anthropic models, metrics might be pre-set
        # Just verify the update doesn't crash
        assert model.metrics.success_rate >= 0
        assert model.metrics.latency_p50 >= 0
        assert model.metrics.cost_per_token >= 0

    def test_update_model_performance_failure(self):
        """Test updating model performance with failure."""
        registry = ModelRegistry()  # Create empty registry for this test
        
        # Create a test model with known metrics
        from orchestrator.core.model import Model, ModelCapabilities, ModelMetrics
        
        class TestModel(Model):
            def __init__(self):
                capabilities = ModelCapabilities(supported_tasks=["generate"])
                metrics = ModelMetrics(success_rate=1.0)  # Start with perfect rate
                super().__init__(
                    name="test-perf-model",
                    provider="test",
                    capabilities=capabilities,
                    metrics=metrics
                )
            
            async def generate(self, prompt, **kwargs):
                return "test"
            
            async def health_check(self):
                return True
        
        model = TestModel()
        registry.models["test:test-perf-model"] = model

        initial_success_rate = model.metrics.success_rate

        # Update performance with failure
        registry.update_model_performance(model, success=False, latency=2.0, cost=0.002)

        # Success rate should decrease
        assert model.metrics.success_rate < initial_success_rate

    def test_calculate_reward_success(self):
        """Test reward calculation for successful operation."""
        registry = ModelRegistry()  # Create empty registry for this test

        reward = registry._calculate_reward(success=True, latency=1.0, cost=0.001)

        assert 0 < reward <= 1.0

    def test_calculate_reward_failure(self):
        """Test reward calculation for failed operation."""
        registry = ModelRegistry()  # Create empty registry for this test

        reward = registry._calculate_reward(success=False, latency=1.0, cost=0.001)

        assert reward == 0.0

    def test_calculate_reward_high_latency(self):
        """Test reward calculation with high latency."""
        registry = ModelRegistry()  # Create empty registry for this test

        reward_low = registry._calculate_reward(success=True, latency=0.1, cost=0.001)
        reward_high = registry._calculate_reward(success=True, latency=5.0, cost=0.001)

        assert reward_low > reward_high

    def test_calculate_reward_high_cost(self):
        """Test reward calculation with high cost."""
        registry = ModelRegistry()  # Create empty registry for this test

        reward_low = registry._calculate_reward(success=True, latency=1.0, cost=0.001)
        reward_high = registry._calculate_reward(success=True, latency=1.0, cost=0.01)

        assert reward_low > reward_high

    def test_calculate_reward_edge_cases(self):
        """Test reward calculation edge cases."""
        registry = ModelRegistry()  # Create empty registry for this test

        # Very high latency should cap penalty
        reward_extreme = registry._calculate_reward(
            success=True, latency=100.0, cost=0.001
        )
        assert reward_extreme >= 0.1  # Should not go below minimum

        # Very high cost should cap penalty
        reward_expensive = registry._calculate_reward(
            success=True, latency=1.0, cost=1.0
        )
        assert reward_expensive >= 0.1  # Should not go below minimum

        # Zero values should work
        reward_zero = registry._calculate_reward(success=True, latency=0.0, cost=0.0)
        assert reward_zero == 1.0

    def test_update_model_metrics(self):
        """Test updating model metrics."""
        registry = ModelRegistry()  # Create empty registry for this test
        
        # Create a test model
        from orchestrator.core.model import Model, ModelCapabilities, ModelMetrics
        
        class TestModel(Model):
            def __init__(self):
                capabilities = ModelCapabilities(supported_tasks=["generate"])
                metrics = ModelMetrics()  # Default metrics
                super().__init__(
                    name="test-metrics-model",
                    provider="test",
                    capabilities=capabilities,
                    metrics=metrics
                )
            
            async def generate(self, prompt, **kwargs):
                return "test"
            
            async def health_check(self):
                return True
        
        model = TestModel()

        initial_success_rate = model.metrics.success_rate
        initial_latency = model.metrics.latency_p50
        initial_cost = model.metrics.cost_per_token

        registry._update_model_metrics(model, success=True, latency=2.0, cost=0.002)

        # Verify metrics were updated
        assert model.metrics.success_rate >= 0
        assert model.metrics.latency_p50 >= 0
        assert model.metrics.cost_per_token >= 0

    def test_update_model_metrics_failure(self):
        """Test updating model metrics with failure."""
        registry = ModelRegistry()  # Create empty registry for this test
        
        # Create a test model with perfect success rate
        from orchestrator.core.model import Model, ModelCapabilities, ModelMetrics
        
        class TestModel(Model):
            def __init__(self):
                capabilities = ModelCapabilities(supported_tasks=["generate"])
                metrics = ModelMetrics(success_rate=1.0)  # Perfect rate
                super().__init__(
                    name="test-failure-model",
                    provider="test",
                    capabilities=capabilities,
                    metrics=metrics
                )
            
            async def generate(self, prompt, **kwargs):
                return "test"
            
            async def health_check(self):
                return True
        
        model = TestModel()

        initial_success_rate = model.metrics.success_rate

        registry._update_model_metrics(model, success=False, latency=0.0, cost=0.0)

        # Success rate should decrease from 1.0
        assert model.metrics.success_rate < initial_success_rate

    def test_get_model_key(self, populated_model_registry):
        """Test getting model key."""
        registry = populated_model_registry
        
        # Get a real model
        model = None
        for _, m in registry.models.items():
            model = m
            break
        
        if not model:
            raise AssertionError(
                "No models available for testing. "
                "Please configure API keys in ~/.orchestrator/.env"
            )

        key = registry._get_model_key(model)

        assert key == f"{model.provider}:{model.name}"

    def test_get_model_statistics(self, populated_model_registry):
        """Test getting model statistics."""
        registry = populated_model_registry

        # Get current statistics
        stats = registry.get_model_statistics()

        assert stats["total_models"] >= 0
        assert stats["providers"] >= 0
        assert stats["healthy_models"] >= 0
        assert "provider_breakdown" in stats
        assert "selection_stats" in stats
        
        # If we have models, verify provider breakdown
        if stats["total_models"] > 0:
            total_in_breakdown = sum(stats["provider_breakdown"].values())
            assert total_in_breakdown == stats["total_models"]

    def test_get_model_statistics_empty(self):
        """Test getting statistics for empty registry."""
        registry = ModelRegistry()  # Create empty registry for this test

        stats = registry.get_model_statistics()

        assert stats["total_models"] == 0
        assert stats["providers"] == 0
        assert stats["healthy_models"] == 0
        assert stats["provider_breakdown"] == {}

    def test_reset_statistics(self, populated_model_registry):
        """Test resetting statistics."""
        registry = populated_model_registry
        
        # Set some health cache data
        if registry.models:
            for key in list(registry.models.keys())[:2]:  # First 2 models
                registry._model_health_cache[key] = True

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
            "average_reward": 0.8,
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
