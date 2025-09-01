"""Model management and selection with unified provider abstractions."""

# Legacy model registry (for backwards compatibility)
from .model_registry import (
    ModelNotFoundError,
    NoEligibleModelsError,
    UCBModelSelector,
)
from .registry_singleton import (
    get_model_registry,
    set_model_registry,
    reset_model_registry,
)

# New unified provider system
from .registry import ModelRegistry as UnifiedModelRegistry
from .config import (
    RegistryConfiguration,
    ModelProviderSpec,
    create_default_configuration,
    create_registry_from_config,
    create_registry_from_env,
    load_configuration_from_dict,
    configuration_to_dict,
    CLOUD_ONLY_CONFIG,
    LOCAL_ONLY_CONFIG,
    DEVELOPMENT_CONFIG,
)
from .providers import (
    ModelProvider,
    ProviderConfig,
    ProviderError,
    OpenAIProvider,
    AnthropicProvider,
    LocalProvider,
)

# Keep legacy ModelRegistry for backwards compatibility
# TODO: Eventually migrate all usage to UnifiedModelRegistry
from .model_registry import ModelRegistry as LegacyModelRegistry

__all__ = [
    # Legacy registry (backwards compatibility)
    "LegacyModelRegistry",
    "UCBModelSelector", 
    "ModelNotFoundError",
    "NoEligibleModelsError",
    "get_model_registry",
    "set_model_registry", 
    "reset_model_registry",
    
    # New unified provider system
    "UnifiedModelRegistry",
    "RegistryConfiguration",
    "ModelProviderSpec",
    "create_default_configuration",
    "create_registry_from_config",
    "create_registry_from_env",
    "load_configuration_from_dict",
    "configuration_to_dict",
    "CLOUD_ONLY_CONFIG",
    "LOCAL_ONLY_CONFIG", 
    "DEVELOPMENT_CONFIG",
    
    # Provider abstractions
    "ModelProvider",
    "ProviderConfig",
    "ProviderError",
    "OpenAIProvider",
    "AnthropicProvider",
    "LocalProvider",
]

# For backwards compatibility, keep ModelRegistry pointing to legacy
ModelRegistry = LegacyModelRegistry
