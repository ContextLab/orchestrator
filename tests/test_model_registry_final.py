"""Final tests for Model Registry to achieve 100% coverage."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from orchestrator.core.model import MockModel
from orchestrator.models.model_registry import ModelRegistry


class TestModelRegistryFinalCoverage:
    """Final tests to achieve 100% coverage of Model Registry."""

    @pytest.mark.asyncio
    async def test_filter_by_health_cache_stale_and_missing_models(self):
        """Test lines 198, 203-205: stale cache with missing models."""
        registry = ModelRegistry()

        model1 = MockModel(name="model1", provider="provider1")
        model2 = MockModel(name="model2", provider="provider2")
        model3 = MockModel(name="model3", provider="provider3")

        registry.register_model(model1)
        registry.register_model(model2)
        registry.register_model(model3)

        # Set up stale cache with only some models
        loop_time = asyncio.get_event_loop().time()
        registry._last_health_check = loop_time - 400  # Stale (> 300s TTL)
        registry._model_health_cache["provider1:model1"] = True
        registry._model_health_cache["provider2:model2"] = False
        # model3 is missing from cache

        # Mock health checks
        model1.health_check = AsyncMock(return_value=True)
        model2.health_check = AsyncMock(return_value=True)
        model3.health_check = AsyncMock(return_value=True)

        # This should trigger:
        # Line 198: missing_models.append(model) for model3
        # Line 203: models_to_refresh = models (because cache is stale)
        # Line 204: await self._refresh_health_cache(models_to_refresh)
        # Line 205: self._last_health_check = current_time

        healthy = await registry._filter_by_health([model1, model2, model3])

        # All models should be healthy after refresh
        assert len(healthy) == 3
        assert model1 in healthy
        assert model2 in healthy
        assert model3 in healthy

        # Verify all health checks were called (because cache was stale)
        model1.health_check.assert_called_once()
        model2.health_check.assert_called_once()
        model3.health_check.assert_called_once()

        # Verify cache was updated
        assert registry._model_health_cache["provider1:model1"] is True
        assert registry._model_health_cache["provider2:model2"] is True
        assert registry._model_health_cache["provider3:model3"] is True

        # Verify _last_health_check was updated (line 205)
        current_loop_time = asyncio.get_event_loop().time()
        assert registry._last_health_check > current_loop_time - 5  # Recently updated
