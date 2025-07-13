"""Debug test for model registry health cache issue."""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, Mock

from src.orchestrator.models.model_registry import ModelRegistry
from src.orchestrator.core.model import MockModel


@pytest.mark.asyncio
async def test_debug_health_cache_stale_logic():
    """Debug the stale cache logic."""
    registry = ModelRegistry()
    
    model1 = MockModel(name="model1", provider="provider1")
    model2 = MockModel(name="model2", provider="provider2")
    
    registry.register_model(model1)
    registry.register_model(model2)
    
    # Set health status and make cache stale
    registry._model_health_cache["provider1:model1"] = True
    registry._model_health_cache["provider2:model2"] = True
    registry._last_health_check = time.time() - 400  # Make it stale (> 300s TTL)
    
    print(f"Cache TTL: {registry._cache_ttl}")
    print(f"Last health check: {registry._last_health_check}")
    print(f"Current time: {time.time()}")
    print(f"Time diff: {time.time() - registry._last_health_check}")
    
    # Mock health checks
    model1.health_check = AsyncMock(return_value=True)
    model2.health_check = AsyncMock(return_value=True)
    
    # Check the conditions manually
    current_time = asyncio.get_event_loop().time()
    cache_is_stale = (registry._last_health_check > 0 and 
                     current_time - registry._last_health_check > registry._cache_ttl)
    
    print(f"Current time (loop): {current_time}")
    print(f"Cache is stale: {cache_is_stale}")
    print(f"Condition 1 (last_check > 0): {registry._last_health_check > 0}")
    print(f"Condition 2 (time diff > TTL): {current_time - registry._last_health_check > registry._cache_ttl}")
    
    # Check for missing models
    missing_models = []
    for model in [model1, model2]:
        model_key = registry._get_model_key(model)
        if model_key not in registry._model_health_cache:
            missing_models.append(model)
    
    print(f"Missing models: {len(missing_models)}")
    print(f"Models in cache: {list(registry._model_health_cache.keys())}")
    
    # Test the actual condition
    should_refresh = cache_is_stale or missing_models
    print(f"Should refresh: {should_refresh}")
    
    # Now test the actual method
    refresh_calls = []
    async def mock_refresh(models_to_refresh):
        refresh_calls.append([registry._get_model_key(model) for model in models_to_refresh])
        for model in models_to_refresh:
            model_key = registry._get_model_key(model)
            registry._model_health_cache[model_key] = True
    
    registry._refresh_health_cache = mock_refresh
    
    healthy = await registry._filter_by_health([model1, model2])
    
    print(f"Refresh calls made: {len(refresh_calls)}")
    if refresh_calls:
        print(f"Models refreshed: {refresh_calls[0]}")
    
    assert len(refresh_calls) >= 0  # Just to see what happens