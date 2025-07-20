"""Debug test for model registry health cache issue."""

import asyncio
import time

import pytest

from src.orchestrator.models.model_registry import ModelRegistry


@pytest.mark.asyncio
async def test_debug_health_cache_stale_logic(populated_model_registry):
    """Debug the stale cache logic using real models."""
    # Get a fresh registry instance for this test
    registry = ModelRegistry()
    
    # Get real models from the populated registry
    available_models = populated_model_registry.list_models()
    if len(available_models) < 2:
        pytest.skip("Need at least 2 models available for this test")
    
    # Use the first two available real models
    model1 = populated_model_registry.get_model(available_models[0])
    model2 = populated_model_registry.get_model(available_models[1])
    
    if not model1 or not model2:
        pytest.skip("Could not get real models for testing")

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

    # Register the real models in our test registry
    registry.register_model(model1)
    registry.register_model(model2)

    # Check the conditions manually
    current_time = asyncio.get_event_loop().time()
    cache_is_stale = (
        registry._last_health_check > 0
        and current_time - registry._last_health_check > registry._cache_ttl
    )

    print(f"Current time (loop): {current_time}")
    print(f"Cache is stale: {cache_is_stale}")
    print(f"Condition 1 (last_check > 0): {registry._last_health_check > 0}")
    print(
        f"Condition 2 (time diff > TTL): {current_time - registry._last_health_check > registry._cache_ttl}"
    )

    # Check for missing models
    missing_models = []
    for model in [model1, model2]:
        model_key = registry._get_model_key(model)
        if model_key not in registry._model_health_cache:
            missing_models.append(model)

    print(f"Missing models: {len(missing_models)}")
    print(f"Model 1: {model1.provider}:{model1.name}")
    print(f"Model 2: {model2.provider}:{model2.name}")
    print(f"Models in cache: {list(registry._model_health_cache.keys())}")

    # Test the actual condition
    should_refresh = cache_is_stale or missing_models
    print(f"Should refresh: {should_refresh}")

    # Now test the actual method
    refresh_calls = []

    # Store original method to restore later
    original_refresh = registry._refresh_health_cache
    
    async def track_refresh(models_to_refresh):
        refresh_calls.append(
            [registry._get_model_key(model) for model in models_to_refresh]
        )
        # Call the actual refresh method to test real behavior
        await original_refresh(models_to_refresh)

    registry._refresh_health_cache = track_refresh

    healthy = await registry._filter_by_health([model1, model2])

    print(f"Refresh calls made: {len(refresh_calls)}")
    if refresh_calls:
        print(f"Models refreshed: {refresh_calls[0]}")

    # Verify the refresh was called
    assert len(refresh_calls) >= 1, "Health cache should have been refreshed"
    
    # Verify we got healthy models back (real models that passed real health checks)
    assert len(healthy) > 0, "Should have at least one healthy model"
    
    # Restore original method
    registry._refresh_health_cache = original_refresh
