"""Model provider configuration and discovery system."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .providers.base import ProviderConfig
from .registry import ModelRegistry


@dataclass
class ModelProviderSpec:
    """Specification for a model provider."""
    
    name: str
    type: str  # "openai", "anthropic", "local"
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    priority: int = 0  # Higher priority providers are checked first
    
    def to_provider_config(self) -> ProviderConfig:
        """Convert to ProviderConfig."""
        return ProviderConfig(name=self.name, **self.config)


@dataclass 
class RegistryConfiguration:
    """Configuration for the model registry."""
    
    providers: List[ModelProviderSpec] = field(default_factory=list)
    auto_discover: bool = True
    default_timeout: float = 30.0
    max_retries: int = 3
    cache_models: bool = True
    
    def add_provider(
        self,
        name: str,
        provider_type: str,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
        priority: int = 0,
    ) -> None:
        """Add a provider specification."""
        spec = ModelProviderSpec(
            name=name,
            type=provider_type,
            config=config or {},
            enabled=enabled,
            priority=priority,
        )
        self.providers.append(spec)
    
    def get_enabled_providers(self) -> List[ModelProviderSpec]:
        """Get list of enabled providers sorted by priority."""
        enabled = [p for p in self.providers if p.enabled]
        return sorted(enabled, key=lambda p: p.priority, reverse=True)


def create_default_configuration() -> RegistryConfiguration:
    """Create a default registry configuration with common providers."""
    config = RegistryConfiguration()
    
    # OpenAI provider
    config.add_provider(
        name="openai",
        provider_type="openai",
        config={
            "api_key": None,  # Will use environment variable
            "organization": None,
            "timeout": 30.0,
            "max_retries": 3,
        },
        priority=100,  # High priority for cloud provider
    )
    
    # Anthropic provider  
    config.add_provider(
        name="anthropic", 
        provider_type="anthropic",
        config={
            "api_key": None,  # Will use environment variable
            "timeout": 30.0,
            "max_retries": 3,
        },
        priority=95,  # High priority for cloud provider
    )
    
    # Local provider (Ollama)
    config.add_provider(
        name="local",
        provider_type="local",
        config={
            "base_url": "http://localhost:11434",
            "timeout": 60.0,  # Longer timeout for local models
            "max_retries": 2,
        },
        priority=50,  # Lower priority than cloud providers
    )
    
    return config


def create_registry_from_config(config: RegistryConfiguration) -> ModelRegistry:
    """Create and configure a model registry from configuration."""
    registry = ModelRegistry()
    
    for provider_spec in config.get_enabled_providers():
        try:
            registry.configure_provider(
                provider_name=provider_spec.name,
                provider_type=provider_spec.type,
                config=provider_spec.config,
            )
        except Exception as e:
            # Log error but continue with other providers
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to configure provider {provider_spec.name}: {e}")
    
    return registry


def create_registry_from_env() -> ModelRegistry:
    """Create a registry configured from environment variables."""
    config = RegistryConfiguration()
    
    # Configure OpenAI if API key is available
    if os.getenv("OPENAI_API_KEY"):
        config.add_provider(
            name="openai",
            provider_type="openai",
            config={
                "api_key": os.getenv("OPENAI_API_KEY"),
                "organization": os.getenv("OPENAI_ORG_ID"),
                "base_url": os.getenv("OPENAI_BASE_URL"),
            },
            priority=100,
        )
    
    # Configure Anthropic if API key is available
    if os.getenv("ANTHROPIC_API_KEY"):
        config.add_provider(
            name="anthropic",
            provider_type="anthropic", 
            config={
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
                "base_url": os.getenv("ANTHROPIC_BASE_URL"),
            },
            priority=95,
        )
    
    # Always add local provider (will check Ollama availability during init)
    config.add_provider(
        name="local",
        provider_type="local",
        config={
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        },
        priority=50,
    )
    
    return create_registry_from_config(config)


def load_configuration_from_dict(config_dict: Dict[str, Any]) -> RegistryConfiguration:
    """Load registry configuration from a dictionary."""
    config = RegistryConfiguration()
    
    # Load global settings
    config.auto_discover = config_dict.get("auto_discover", True)
    config.default_timeout = config_dict.get("default_timeout", 30.0)
    config.max_retries = config_dict.get("max_retries", 3)
    config.cache_models = config_dict.get("cache_models", True)
    
    # Load providers
    for provider_dict in config_dict.get("providers", []):
        spec = ModelProviderSpec(
            name=provider_dict["name"],
            type=provider_dict["type"],
            config=provider_dict.get("config", {}),
            enabled=provider_dict.get("enabled", True),
            priority=provider_dict.get("priority", 0),
        )
        config.providers.append(spec)
    
    return config


def configuration_to_dict(config: RegistryConfiguration) -> Dict[str, Any]:
    """Convert registry configuration to dictionary."""
    return {
        "auto_discover": config.auto_discover,
        "default_timeout": config.default_timeout,
        "max_retries": config.max_retries,
        "cache_models": config.cache_models,
        "providers": [
            {
                "name": spec.name,
                "type": spec.type,
                "config": spec.config,
                "enabled": spec.enabled,
                "priority": spec.priority,
            }
            for spec in config.providers
        ],
    }


# Example configuration presets
CLOUD_ONLY_CONFIG = {
    "providers": [
        {
            "name": "openai",
            "type": "openai",
            "enabled": True,
            "priority": 100,
        },
        {
            "name": "anthropic", 
            "type": "anthropic",
            "enabled": True,
            "priority": 95,
        },
        {
            "name": "local",
            "type": "local", 
            "enabled": False,  # Disabled for cloud-only
        },
    ]
}

LOCAL_ONLY_CONFIG = {
    "providers": [
        {
            "name": "openai",
            "type": "openai",
            "enabled": False,  # Disabled for local-only
        },
        {
            "name": "anthropic",
            "type": "anthropic", 
            "enabled": False,  # Disabled for local-only
        },
        {
            "name": "local",
            "type": "local",
            "enabled": True,
            "priority": 100,  # High priority for local-only setup
        },
    ]
}

DEVELOPMENT_CONFIG = {
    "auto_discover": True,
    "cache_models": True,
    "providers": [
        {
            "name": "openai",
            "type": "openai",
            "config": {"timeout": 60.0},  # Longer timeout for development
            "enabled": True,
            "priority": 100,
        },
        {
            "name": "anthropic",
            "type": "anthropic",
            "config": {"timeout": 60.0},
            "enabled": True, 
            "priority": 95,
        },
        {
            "name": "local",
            "type": "local",
            "config": {"timeout": 120.0},  # Very long timeout for slow local models
            "enabled": True,
            "priority": 50,
        },
    ]
}