"""Model management and selection.

The canonical registry is :class:`ModelRegistry` from
:mod:`orchestrator.models.model_registry`, reached through
:func:`get_model_registry`. The skills-era "unified" registry and its
provider-configuration system were retired under #430 -- they only ever
supported Anthropic, which is no longer a provider of this product.
"""

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
from .providers import (
    ModelProvider,
    ProviderConfig,
    ProviderError,
)

#: Kept for backwards compatibility with code that imported the legacy name.
LegacyModelRegistry = ModelRegistry

__all__ = [
    "ModelRegistry",
    "LegacyModelRegistry",
    "UCBModelSelector",
    "ModelNotFoundError",
    "NoEligibleModelsError",
    "get_model_registry",
    "set_model_registry",
    "reset_model_registry",
    "ModelProvider",
    "ProviderConfig",
    "ProviderError",
]
