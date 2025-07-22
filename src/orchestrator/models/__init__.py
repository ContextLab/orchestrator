"""Model management and selection."""

from .model_registry import (
    ModelNotFoundError,
    ModelRegistry,
    NoEligibleModelsError,
    UCBModelSelector,
)
from .registry_singleton import (
    get_model_registry,
    set_model_registry,
    reset_model_registry,
)

__all__ = [
    "ModelRegistry",
    "UCBModelSelector",
    "ModelNotFoundError",
    "NoEligibleModelsError",
    "get_model_registry",
    "set_model_registry",
    "reset_model_registry",
]
