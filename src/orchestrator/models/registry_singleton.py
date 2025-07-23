"""Singleton pattern for ModelRegistry to ensure single source of truth."""

from typing import Optional
from .model_registry import ModelRegistry

_global_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """
    Get the global model registry instance.

    This ensures that all components use the same registry instance,
    preventing issues with models being registered in different registries.

    Returns:
        The global ModelRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelRegistry()
    return _global_registry


def set_model_registry(registry: ModelRegistry) -> None:
    """
    Set the global model registry instance.

    This should only be called during initialization to set a custom registry.

    Args:
        registry: The ModelRegistry instance to use globally
    """
    global _global_registry
    _global_registry = registry


def reset_model_registry() -> None:
    """
    Reset the global model registry.

    This is mainly useful for testing.
    """
    global _global_registry
    _global_registry = None
