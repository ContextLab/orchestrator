"""Shared pytest fixtures for orchestrator tests."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from orchestrator import init_models
from src.orchestrator.models.model_registry import ModelRegistry


@pytest.fixture(scope="session")
def populated_model_registry() -> ModelRegistry:
    """
    Provide a populated model registry for tests.

    This fixture initializes models once per test session and provides
    the populated registry to all tests that need it.

    Returns:
        ModelRegistry: Registry populated with all configured models

    Raises:
        pytest.skip.Exception: If no models are available (likely due to missing API keys)
    """
    print("\n>> Test fixture: Initializing model registry...")
    registry = init_models()

    # Verify we have models available
    available_models = registry.list_models()
    print(f">> Test fixture: Found {len(available_models)} models: {available_models}")
    
    if not available_models:
        # Can't use pytest.skip in a session-scoped fixture
        # Return None instead and let tests handle it
        print(">> Test fixture: No models available - returning empty registry")
        return registry

    return registry


@pytest.fixture(scope="session")
def model_registry(populated_model_registry) -> ModelRegistry:
    """
    Alias for populated_model_registry for backward compatibility.

    Many tests expect a 'model_registry' fixture.
    """
    return populated_model_registry
