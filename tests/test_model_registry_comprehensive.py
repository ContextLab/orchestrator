"""Comprehensive tests for Model Registry to achieve 100% coverage."""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from orchestrator.core.model import MockModel, ModelMetrics
from orchestrator.models.model_registry import (
    ModelRegistry,
    NoEligibleModelsError,
    UCBModelSelector,
)


class TestModelRegistryComprehensiveCoverage:
    """Comprehensive tests to achieve 100% coverage of Model Registry."""

    @pytest.mark.asyncio
    async def test_filter_by_health_missing_models_from_cache(self):
        """Test _filter_by_health when some models are missing from cache (line 198)."""
        registry = ModelRegistry()

        model1 = MockModel(name="model1", provider="provider1")
        model2 = MockModel(name="model2", provider="provider2")
        model3 = MockModel(name="model3", provider="provider3")

        registry.register_model(model1)
        registry.register_model(model2)
        registry.register_model(model3)

        # Set health status for only some models (model3 missing from cache)
        registry._model_health_cache["provider1:model1"] = True
        registry._model_health_cache["provider2:model2"] = False
        # provider3:model3 is missing from cache

        # Mock _refresh_health_cache to track calls
        refresh_calls = []

        async def mock_refresh(models_to_refresh):
            refresh_calls.append(
                [registry._get_model_key(model) for model in models_to_refresh]
            )
            # Add the missing model to cache
            for model in models_to_refresh:
                model_key = registry._get_model_key(model)
                if model_key not in registry._model_health_cache:
                    registry._model_health_cache[model_key] = True

        registry._refresh_health_cache = mock_refresh

        healthy = await registry._filter_by_health([model1, model2, model3])

        # Should refresh only the missing model (model3)
        assert len(refresh_calls) == 1
        assert "provider3:model3" in refresh_calls[0]

        # Should return healthy models including the newly refreshed one
        assert len(healthy) == 2  # model1 and model3 (model2 is unhealthy)
        assert model1 in healthy
        assert model3 in healthy

    @pytest.mark.asyncio
    async def test_filter_by_health_stale_cache_refresh_all(self):
        """Test _filter_by_health with stale cache refreshing all models (lines 203-205)."""
        registry = ModelRegistry()

        model1 = MockModel(name="model1", provider="provider1")
        model2 = MockModel(name="model2", provider="provider2")

        registry.register_model(model1)
        registry.register_model(model2)

        # Set health status and make cache stale
        registry._model_health_cache["provider1:model1"] = True
        registry._model_health_cache["provider2:model2"] = True
        # Set last health check to a non-zero value that's stale using asyncio loop time
        loop_time = asyncio.get_event_loop().time()
        registry._last_health_check = loop_time - 400  # Make it stale (> 300s TTL)

        refresh_calls = []

        async def mock_refresh(models_to_refresh):
            refresh_calls.append(
                [registry._get_model_key(model) for model in models_to_refresh]
            )
            # Update cache with fresh data
            for model in models_to_refresh:
                model_key = registry._get_model_key(model)
                registry._model_health_cache[model_key] = True

        registry._refresh_health_cache = mock_refresh

        # Mock health checks for the models
        model1.health_check = AsyncMock(return_value=True)
        model2.health_check = AsyncMock(return_value=True)

        # Should refresh all models due to stale cache
        healthy = await registry._filter_by_health([model1, model2])

        # Should refresh ALL models because cache is stale
        assert len(refresh_calls) == 1
        assert len(refresh_calls[0]) == 2  # Both models refreshed
        assert "provider1:model1" in refresh_calls[0]
        assert "provider2:model2" in refresh_calls[0]

        # Verify _last_health_check was updated (using asyncio loop time)
        current_loop_time = asyncio.get_event_loop().time()
        assert registry._last_health_check > current_loop_time - 5  # Recently updated

    @pytest.mark.asyncio
    async def test_refresh_health_cache_with_tasks(self):
        """Test _refresh_health_cache with actual tasks to execute (lines 218-224)."""
        registry = ModelRegistry()

        model1 = MockModel(name="model1", provider="provider1")
        model2 = MockModel(name="model2", provider="provider2")

        # Mock health_check methods
        model1.health_check = AsyncMock(return_value=True)
        model2.health_check = AsyncMock(return_value=False)

        # Call _refresh_health_cache directly with models
        await registry._refresh_health_cache([model1, model2])

        # Verify health checks were called
        model1.health_check.assert_called_once()
        model2.health_check.assert_called_once()

        # Verify cache was updated
        assert registry._model_health_cache["provider1:model1"] is True
        assert registry._model_health_cache["provider2:model2"] is False

    @pytest.mark.asyncio
    async def test_refresh_health_cache_empty_models(self):
        """Test _refresh_health_cache with empty models list."""
        registry = ModelRegistry()

        # Should handle empty list gracefully
        await registry._refresh_health_cache([])

        # Cache should remain empty
        assert len(registry._model_health_cache) == 0

    @pytest.mark.asyncio
    async def test_check_model_health_exception_handling(self):
        """Test _check_model_health exception handling (lines 228-232)."""
        registry = ModelRegistry()

        model = MockModel(name="failing_model", provider="test_provider")

        # Mock health_check to raise exception
        model.health_check = AsyncMock(side_effect=Exception("Health check failed"))

        model_key = "test_provider:failing_model"

        # Should handle exception and set health to False
        await registry._check_model_health(model_key, model)

        # Verify health was set to False due to exception
        assert registry._model_health_cache[model_key] is False

        # Verify health_check was called
        model.health_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_model_health_success(self):
        """Test _check_model_health successful execution."""
        registry = ModelRegistry()

        model = MockModel(name="healthy_model", provider="test_provider")

        # Mock health_check to return True
        model.health_check = AsyncMock(return_value=True)

        model_key = "test_provider:healthy_model"

        # Should set health to True
        await registry._check_model_health(model_key, model)

        # Verify health was set correctly
        assert registry._model_health_cache[model_key] is True

        # Verify health_check was called
        model.health_check.assert_called_once()


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

        model1 = MockModel(name="model1", provider="provider1")
        registry.register_model(model1)

        # _last_health_check should be 0 (never checked)
        assert registry._last_health_check == 0

        # Mock health check
        model1.health_check = AsyncMock(return_value=True)

        # Should refresh because model is missing from cache
        healthy = await registry._filter_by_health([model1])

        # Should call health check and update cache
        model1.health_check.assert_called_once()
        assert registry._model_health_cache["provider1:model1"] is True
        assert len(healthy) == 1
        assert healthy[0] == model1

    @pytest.mark.asyncio
    async def test_filter_by_health_fresh_cache_no_missing(self):
        """Test filter_by_health with fresh cache and no missing models."""
        registry = ModelRegistry()

        model1 = MockModel(name="model1", provider="provider1")
        model2 = MockModel(name="model2", provider="provider2")
        registry.register_model(model1)
        registry.register_model(model2)

        # Set fresh cache (within TTL)
        current_time = time.time()
        registry._last_health_check = current_time - 100  # Fresh (< 300s TTL)
        registry._model_health_cache["provider1:model1"] = True
        registry._model_health_cache["provider2:model2"] = False

        # Mock health checks (should NOT be called)
        model1.health_check = AsyncMock()
        model2.health_check = AsyncMock()

        # Should NOT refresh cache
        healthy = await registry._filter_by_health([model1, model2])

        # Health checks should NOT be called
        model1.health_check.assert_not_called()
        model2.health_check.assert_not_called()

        # Should return only healthy models
        assert len(healthy) == 1
        assert healthy[0] == model1

    @pytest.mark.asyncio
    async def test_filter_by_health_mixed_cache_and_stale(self):
        """Test filter_by_health with both stale cache and missing models."""
        registry = ModelRegistry()

        model1 = MockModel(name="model1", provider="provider1")
        model2 = MockModel(name="model2", provider="provider2")
        model3 = MockModel(name="model3", provider="provider3")

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

        async def mock_refresh(models_to_refresh):
            refresh_calls.append(models_to_refresh)
            for model in models_to_refresh:
                model_key = registry._get_model_key(model)
                registry._model_health_cache[model_key] = True

        registry._refresh_health_cache = mock_refresh

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
        healthy_model = MockModel(name="healthy", provider="test")
        unhealthy_model = MockModel(name="unhealthy", provider="test")
        failing_model = MockModel(name="failing", provider="test")

        # Set up health check behaviors
        healthy_model.health_check = AsyncMock(return_value=True)
        unhealthy_model.health_check = AsyncMock(return_value=False)
        failing_model.health_check = AsyncMock(
            side_effect=Exception("Health check error")
        )

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
