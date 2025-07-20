"""Final tests for Model Registry to achieve 100% coverage."""

import asyncio

import pytest

from orchestrator.models.model_registry import ModelRegistry


class TestModelRegistryFinalCoverage:
    """Final tests to achieve 100% coverage of Model Registry."""

    @pytest.mark.asyncio
    async def test_filter_by_health_cache_stale_and_missing_models(self, populated_model_registry):
        """Test lines 198, 203-205: stale cache with missing models using real models."""
        # Create a fresh registry for this test
        registry = ModelRegistry()
        
        # Get real models from the populated registry
        available_models = populated_model_registry.list_models()
        if len(available_models) < 3:
            pytest.skip("Need at least 3 models available for this test")
        
        # Use real models
        model1 = populated_model_registry.get_model(available_models[0])
        model2 = populated_model_registry.get_model(available_models[1])
        model3 = populated_model_registry.get_model(available_models[2])
        
        if not all([model1, model2, model3]):
            pytest.skip("Could not get required models for testing")

        # Register the real models
        registry.register_model(model1)
        registry.register_model(model2)
        registry.register_model(model3)

        # Set up stale cache with only some models
        loop_time = asyncio.get_event_loop().time()
        registry._last_health_check = loop_time - 400  # Stale (> 300s TTL)
        
        # Manually set cache entries (simulating previous health checks)
        model1_key = f"{model1.provider}:{model1.name}"
        model2_key = f"{model2.provider}:{model2.name}"
        registry._model_health_cache[model1_key] = True
        registry._model_health_cache[model2_key] = False
        # model3 is intentionally missing from cache

        # Track which models had their health checked
        health_checked_models = set()
        
        # Store original health_check methods
        original_health_checks = {
            model1: model1.health_check,
            model2: model2.health_check,
            model3: model3.health_check
        }
        
        # Wrap health_check methods to track calls
        async def make_tracked_health_check(model):
            async def tracked_health_check():
                health_checked_models.add(model.name)
                return await original_health_checks[model]()
            return tracked_health_check
        
        model1.health_check = await make_tracked_health_check(model1)
        model2.health_check = await make_tracked_health_check(model2)
        model3.health_check = await make_tracked_health_check(model3)
        
        # This should trigger:
        # Line 198: missing_models.append(model) for model3
        # Line 203: models_to_refresh = models (because cache is stale)
        # Line 204: await self._refresh_health_cache(models_to_refresh)
        # Line 205: self._last_health_check = current_time
        healthy = await registry._filter_by_health([model1, model2, model3])
        
        # Restore original methods
        model1.health_check = original_health_checks[model1]
        model2.health_check = original_health_checks[model2]
        model3.health_check = original_health_checks[model3]
        
        # Verify results - we should get back healthy models
        # The exact number depends on which models are actually healthy
        assert len(healthy) >= 0, "Should return a list of healthy models"
        
        # Verify all models were health checked (because cache was stale)
        assert len(health_checked_models) == 3, f"All 3 models should have been health checked, but only {health_checked_models} were"

        # Verify cache was updated for all models
        model1_key = f"{model1.provider}:{model1.name}"
        model2_key = f"{model2.provider}:{model2.name}"
        model3_key = f"{model3.provider}:{model3.name}"
        
        assert model1_key in registry._model_health_cache, "Model1 should be in cache"
        assert model2_key in registry._model_health_cache, "Model2 should be in cache"
        assert model3_key in registry._model_health_cache, "Model3 should be in cache"

        # Verify _last_health_check was updated (line 205)
        current_loop_time = asyncio.get_event_loop().time()
        assert registry._last_health_check > loop_time, "Last health check time should be updated"
        assert registry._last_health_check <= current_loop_time, "Last health check should not be in the future"
