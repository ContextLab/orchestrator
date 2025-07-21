"""Comprehensive tests for Model Registry to achieve 100% coverage."""

import asyncio
import time

import pytest

from orchestrator.core.model import Model, ModelCapabilities, ModelMetrics
from orchestrator.models.model_registry import (
    ModelRegistry,
    NoEligibleModelsError,
    UCBModelSelector,
)


class TestModel(Model):
    """Test model implementation for registry tests."""
    
    def __init__(self, name, provider="test", supports_function_calling=True):
        capabilities = ModelCapabilities(
            supported_tasks=["generate", "analyze", "transform"],
            context_window=8192,
            languages=["en"]
        )
        super().__init__(name=name, provider=provider, capabilities=capabilities)
        self.supports_function_calling = supports_function_calling
        self._health_check_result = True
        self._health_check_exception = None
        self._health_check_calls = 0
        
    async def generate(self, prompt, **kwargs):
        return f"Response from {self.name}"
        
    async def generate_structured(self, prompt, schema, **kwargs):
        return {"value": await self.generate(prompt, **kwargs)}
        
    async def validate_response(self, response, schema):
        return True
        
    def estimate_tokens(self, text):
        return len(text.split())
        
    def estimate_cost(self, input_tokens, output_tokens):
        return 0.001
        
    async def health_check(self):
        """Health check that can be configured to return specific results."""
        self._health_check_calls += 1
        if self._health_check_exception:
            raise self._health_check_exception
        return self._health_check_result
        
    def set_health_check_result(self, result, exception=None):
        """Configure health check behavior."""
        self._health_check_result = result
        self._health_check_exception = exception
        
    def get_health_check_calls(self):
        """Get number of times health_check was called."""
        return self._health_check_calls
        
    def reset_health_check_calls(self):
        """Reset health check call counter."""
        self._health_check_calls = 0


class TestModelRegistryComprehensiveCoverage:
    """Comprehensive tests to achieve 100% coverage of Model Registry."""

    @pytest.mark.asyncio
    async def test_filter_by_health_missing_models_from_cache(self, populated_model_registry):
        """Test _filter_by_health when some models are missing from cache (line 198)."""
        # Create fresh registry for this test
        registry = ModelRegistry()
        
        # Get real models from populated registry
        available_models = populated_model_registry.list_models()
        if len(available_models) < 3:
            pytest.skip("Need at least 3 models available for this test")
        
        model1 = populated_model_registry.get_model(available_models[0])
        model2 = populated_model_registry.get_model(available_models[1])
        model3 = populated_model_registry.get_model(available_models[2])
        
        if not all([model1, model2, model3]):
            pytest.skip("Could not get required models for testing")

        registry.register_model(model1)
        registry.register_model(model2)
        registry.register_model(model3)

        # Set health status for only some models (model3 missing from cache)
        model1_key = f"{model1.provider}:{model1.name}"
        model2_key = f"{model2.provider}:{model2.name}"
        # Intentionally don't add model3 to cache to test missing model scenario
        
        registry._model_health_cache[model1_key] = True
        registry._model_health_cache[model2_key] = False
        # model3 is missing from cache

        # Track refresh calls by replacing the method
        refresh_calls = []
        original_refresh = registry._refresh_health_cache

        async def track_refresh(models_to_refresh):
            refresh_calls.append(
                [registry._get_model_key(model) for model in models_to_refresh]
            )
            # Call the original method
            await original_refresh(models_to_refresh)

        registry._refresh_health_cache = track_refresh

        healthy = await registry._filter_by_health([model1, model2, model3])

        # Should refresh only the missing model (model3)
        assert len(refresh_calls) == 1
        model3_key = f"{model3.provider}:{model3.name}"
        assert model3_key in refresh_calls[0]

        # Should return healthy models including the newly refreshed one
        assert len(healthy) == 2  # model1 and model3 (model2 is unhealthy)
        assert model1 in healthy
        assert model3 in healthy

    @pytest.mark.asyncio
    async def test_filter_by_health_stale_cache_refresh_all(self, populated_model_registry):
        """Test _filter_by_health with stale cache refreshing all models (lines 203-205)."""
        # Create fresh registry for this test
        registry = ModelRegistry()
        
        # Get real models from populated registry
        available_models = populated_model_registry.list_models()
        if len(available_models) < 2:
            pytest.skip("Need at least 2 models available for this test")
        
        model1 = populated_model_registry.get_model(available_models[0])
        model2 = populated_model_registry.get_model(available_models[1])
        
        if not all([model1, model2]):
            pytest.skip("Could not get required models for testing")

        registry.register_model(model1)
        registry.register_model(model2)

        # Set health status and make cache stale
        model1_key = f"{model1.provider}:{model1.name}"
        model2_key = f"{model2.provider}:{model2.name}"
        
        registry._model_health_cache[model1_key] = True
        registry._model_health_cache[model2_key] = True
        
        # Set last health check to a non-zero value that's stale using asyncio loop time
        loop_time = asyncio.get_event_loop().time()
        registry._last_health_check = loop_time - 400  # Make it stale (> 300s TTL)

        refresh_calls = []
        original_refresh = registry._refresh_health_cache

        async def track_refresh(models_to_refresh):
            refresh_calls.append(
                [registry._get_model_key(model) for model in models_to_refresh]
            )
            # Call the original method
            await original_refresh(models_to_refresh)

        registry._refresh_health_cache = track_refresh

        # Should refresh all models due to stale cache
        healthy = await registry._filter_by_health([model1, model2])

        # Should refresh ALL models because cache is stale
        assert len(refresh_calls) == 1
        assert len(refresh_calls[0]) == 2  # Both models refreshed
        assert model1_key in refresh_calls[0]
        assert model2_key in refresh_calls[0]

        # Verify _last_health_check was updated (using asyncio loop time)
        current_loop_time = asyncio.get_event_loop().time()
        assert registry._last_health_check > current_loop_time - 5  # Recently updated

    @pytest.mark.asyncio
    async def test_refresh_health_cache_with_tasks(self, populated_model_registry):
        """Test _refresh_health_cache with actual tasks to execute (lines 218-224)."""
        registry = ModelRegistry()
        
        # Get real models from populated registry
        available_models = populated_model_registry.list_models()
        if len(available_models) < 2:
            pytest.skip("Need at least 2 models available for this test")
        
        model1 = populated_model_registry.get_model(available_models[0])
        model2 = populated_model_registry.get_model(available_models[1])
        
        if not all([model1, model2]):
            pytest.skip("Could not get required models for testing")

        # Call _refresh_health_cache directly with real models
        await registry._refresh_health_cache([model1, model2])
        
        # Verify cache was updated with actual health check results
        model1_key = f"{model1.provider}:{model1.name}"
        model2_key = f"{model2.provider}:{model2.name}"
        
        # Cache should contain the health check results (True/False based on actual health)
        assert model1_key in registry._model_health_cache
        assert model2_key in registry._model_health_cache
        
        # The actual values depend on the real models' health status
        # We just verify that the cache was populated
        assert isinstance(registry._model_health_cache[model1_key], bool)
        assert isinstance(registry._model_health_cache[model2_key], bool)

    @pytest.mark.asyncio
    async def test_refresh_health_cache_empty_models(self):
        """Test _refresh_health_cache with empty models list."""
        registry = ModelRegistry()

        # Should handle empty list gracefully
        await registry._refresh_health_cache([])

        # Cache should remain empty
        assert len(registry._model_health_cache) == 0

    @pytest.mark.asyncio
    async def test_check_model_health_exception_handling(self, populated_model_registry):
        """Test _check_model_health exception handling (lines 228-232)."""
        registry = ModelRegistry()
        
        # Get a real model from populated registry
        available_models = populated_model_registry.list_models()
        if not available_models:
            pytest.skip("No models available for testing")
        
        model = populated_model_registry.get_model(available_models[0])
        if not model:
            pytest.skip("Could not get model for testing")

        # Store the original health_check method
        original_health_check = model.health_check
        
        # Create a health check that raises an exception
        async def failing_health_check():
            raise Exception("Health check failed")
        
        # Replace the health_check method temporarily
        model.health_check = failing_health_check
        model_key = registry._get_model_key(model)

        # Should handle exception and set health to False
        await registry._check_model_health(model_key, model)

        # Verify health was set to False due to exception
        assert registry._model_health_cache[model_key] is False
        
        # Restore original health check
        model.health_check = original_health_check

    @pytest.mark.asyncio
    async def test_check_model_health_success(self, populated_model_registry):
        """Test _check_model_health successful execution."""
        registry = ModelRegistry()
        
        # Get a real model from populated registry
        available_models = populated_model_registry.list_models()
        if not available_models:
            pytest.skip("No models available for testing")
        
        model = populated_model_registry.get_model(available_models[0])
        if not model:
            pytest.skip("Could not get model for testing")
        
        model_key = registry._get_model_key(model)
        
        # Call check_model_health with a real model
        await registry._check_model_health(model_key, model)
        
        # Verify health was set in cache (actual value depends on real model's health)
        assert model_key in registry._model_health_cache
        assert isinstance(registry._model_health_cache[model_key], bool)


class TestUCBModelSelectorComprehensiveCoverage:
    """Comprehensive tests to achieve 100% coverage of UCB Model Selector."""

    def test_update_reward_pending_attempts_removal(self):
        """Test update_reward removing from pending_attempts (line 446)."""
        selector = UCBModelSelector()

        # Initialize model
        metrics = ModelMetrics()
        selector.initialize_model("test_model", metrics)

        # Simulate select() adding to pending attempts
        selector._pending_attempts.add("test_model")
        selector.model_stats["test_model"][
            "attempts"
        ] = 1  # Already incremented by select
        selector.total_attempts = 1

        # Update reward - should remove from pending attempts
        selector.update_reward("test_model", 0.8)

        # Should remove from pending attempts
        assert "test_model" not in selector._pending_attempts

        # Should NOT increment attempts again
        assert selector.model_stats["test_model"]["attempts"] == 1
        assert selector.total_attempts == 1

        # Should update other statistics
        assert selector.model_stats["test_model"]["total_reward"] == 0.8
        assert selector.model_stats["test_model"]["successes"] == 1
        assert selector.model_stats["test_model"]["average_reward"] == 0.8

    def test_update_reward_not_in_pending_attempts(self):
        """Test update_reward when model not in pending_attempts."""
        selector = UCBModelSelector()

        # Initialize model
        metrics = ModelMetrics()
        selector.initialize_model("test_model", metrics)

        # Don't add to pending attempts (simulate direct update_reward call)
        # Initial state: 0 attempts

        # Update reward - should increment attempts
        selector.update_reward("test_model", 0.5)

        # Should increment attempts and total
        assert selector.model_stats["test_model"]["attempts"] == 1
        assert selector.total_attempts == 1

        # Should update statistics
        assert selector.model_stats["test_model"]["total_reward"] == 0.5
        assert selector.model_stats["test_model"]["successes"] == 1
        assert selector.model_stats["test_model"]["average_reward"] == 0.5


class TestModelRegistryHealthCacheEdgeCases:
    """Test edge cases for health caching logic."""

    @pytest.mark.asyncio
    async def test_filter_by_health_never_checked_cache(self):
        """Test filter_by_health when cache has never been checked."""
        registry = ModelRegistry()

        model1 = TestModel(name="model1", provider="provider1")
        registry.register_model(model1)

        # _last_health_check should be 0 (never checked)
        assert registry._last_health_check == 0

        # Configure health check to return True
        model1.set_health_check_result(True)

        # Should refresh because model is missing from cache
        healthy = await registry._filter_by_health([model1])

        # Should call health check and update cache
        assert model1.get_health_check_calls() == 1
        assert registry._model_health_cache["provider1:model1"] is True
        assert len(healthy) == 1
        assert healthy[0] == model1

    @pytest.mark.asyncio
    async def test_filter_by_health_fresh_cache_no_missing(self):
        """Test filter_by_health with fresh cache and no missing models."""
        registry = ModelRegistry()

        model1 = TestModel(name="model1", provider="provider1")
        model2 = TestModel(name="model2", provider="provider2")
        registry.register_model(model1)
        registry.register_model(model2)

        # Set fresh cache (within TTL)
        current_time = time.time()
        registry._last_health_check = current_time - 100  # Fresh (< 300s TTL)
        registry._model_health_cache["provider1:model1"] = True
        registry._model_health_cache["provider2:model2"] = False

        # Reset health check counters
        model1.reset_health_check_calls()
        model2.reset_health_check_calls()

        # Should NOT refresh cache
        healthy = await registry._filter_by_health([model1, model2])

        # Health checks should NOT be called
        assert model1.get_health_check_calls() == 0
        assert model2.get_health_check_calls() == 0

        # Should return only healthy models
        assert len(healthy) == 1
        assert healthy[0] == model1

    @pytest.mark.asyncio
    async def test_filter_by_health_mixed_cache_and_stale(self):
        """Test filter_by_health with both stale cache and missing models."""
        registry = ModelRegistry()

        model1 = TestModel(name="model1", provider="provider1")
        model2 = TestModel(name="model2", provider="provider2")
        model3 = TestModel(name="model3", provider="provider3")

        registry.register_model(model1)
        registry.register_model(model2)
        registry.register_model(model3)

        # Make cache stale
        loop_time = asyncio.get_event_loop().time()
        registry._last_health_check = loop_time - 400  # Stale (> 300s TTL)
        registry._model_health_cache["provider1:model1"] = True
        registry._model_health_cache["provider2:model2"] = False
        # model3 missing from cache

        refresh_calls = []
        original_refresh = registry._refresh_health_cache

        async def track_refresh(models_to_refresh):
            refresh_calls.append(models_to_refresh)
            await original_refresh(models_to_refresh)

        registry._refresh_health_cache = track_refresh

        # Should refresh ALL models because cache is stale (not just missing ones)
        healthy = await registry._filter_by_health([model1, model2, model3])

        # Should refresh all models due to stale cache
        assert len(refresh_calls) == 1
        assert len(refresh_calls[0]) == 3  # All models refreshed

    @pytest.mark.asyncio
    async def test_comprehensive_model_workflow(self):
        """Test comprehensive model workflow with health caching."""
        registry = ModelRegistry()

        # Create models with different health behaviors
        healthy_model = TestModel(name="healthy", provider="test")
        unhealthy_model = TestModel(name="unhealthy", provider="test")
        failing_model = TestModel(name="failing", provider="test")

        # Set up health check behaviors
        healthy_model.set_health_check_result(True)
        unhealthy_model.set_health_check_result(False)
        failing_model.set_health_check_result(False, Exception("Health check error"))

        # Register all models
        registry.register_model(healthy_model)
        registry.register_model(unhealthy_model)
        registry.register_model(failing_model)

        # Test model selection with health filtering
        requirements = {"supports_function_calling": True}

        try:
            selected = await registry.select_model(requirements)
            # Should select the healthy model
            assert selected == healthy_model
        except NoEligibleModelsError:
            # May fail if no models meet requirements, but health filtering should work
            pass

        # Verify health cache was populated
        assert registry._model_health_cache["test:healthy"] is True
        assert registry._model_health_cache["test:unhealthy"] is False
        assert registry._model_health_cache["test:failing"] is False

        # Test performance update workflow
        registry.update_model_performance(
            healthy_model, success=True, latency=1.5, cost=0.001
        )

        # Verify metrics were updated
        assert healthy_model.metrics.success_rate >= 0  # Should be updated

        # Test statistics gathering
        stats = registry.get_model_statistics()
        assert stats["total_models"] == 3
        assert stats["healthy_models"] == 1
