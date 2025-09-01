"""Base provider abstraction for unified model management."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ...core.model import Model, ModelCapabilities, ModelCost, ModelRequirements

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Base exception for provider-related errors."""
    pass


class ProviderInitializationError(ProviderError):
    """Raised when provider initialization fails."""
    pass


class ProviderAPIError(ProviderError):
    """Raised when provider API calls fail."""
    pass


class ModelNotSupportedError(ProviderError):
    """Raised when a requested model is not supported by the provider."""
    pass


@dataclass
class ProviderConfig:
    """Configuration for a model provider."""
    
    name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_per_minute: Optional[int] = None
    custom_headers: Dict[str, str] = field(default_factory=dict)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.name:
            raise ValueError("Provider name cannot be empty")
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("Max retries must be non-negative")
        if self.retry_delay < 0:
            raise ValueError("Retry delay must be non-negative")


class ModelProvider(ABC):
    """Abstract base class for model providers."""

    def __init__(self, config: ProviderConfig) -> None:
        """
        Initialize provider.
        
        Args:
            config: Provider configuration
        """
        self.config = config
        self.name = config.name
        self._initialized = False
        self._available_models: Set[str] = set()
        self._model_cache: Dict[str, Model] = {}
        
    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized."""
        return self._initialized

    @property 
    def available_models(self) -> Set[str]:
        """Get set of available model names."""
        return self._available_models.copy()

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the provider.
        
        This should:
        - Validate API keys and configuration
        - Test connectivity
        - Discover available models
        - Set up any required clients
        
        Raises:
            ProviderInitializationError: If initialization fails
        """
        pass

    @abstractmethod
    async def create_model(self, model_name: str, **kwargs: Any) -> Model:
        """
        Create a model instance for this provider.
        
        Args:
            model_name: Name of the model to create
            **kwargs: Additional model-specific parameters
            
        Returns:
            Model instance
            
        Raises:
            ModelNotSupportedError: If model is not supported
            ProviderAPIError: If model creation fails
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if provider is healthy and available.
        
        Returns:
            True if provider is healthy
        """
        pass

    @abstractmethod
    async def discover_models(self) -> List[str]:
        """
        Discover available models from this provider.
        
        Returns:
            List of available model names
        """
        pass

    @abstractmethod
    def get_model_capabilities(self, model_name: str) -> ModelCapabilities:
        """
        Get capabilities for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model capabilities
            
        Raises:
            ModelNotSupportedError: If model is not supported
        """
        pass

    @abstractmethod
    def get_model_requirements(self, model_name: str) -> ModelRequirements:
        """
        Get resource requirements for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model requirements
            
        Raises:
            ModelNotSupportedError: If model is not supported
        """
        pass

    @abstractmethod
    def get_model_cost(self, model_name: str) -> ModelCost:
        """
        Get cost information for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model cost information
            
        Raises:
            ModelNotSupportedError: If model is not supported
        """
        pass

    async def get_model(self, model_name: str, **kwargs: Any) -> Model:
        """
        Get or create a model instance (with caching).
        
        Args:
            model_name: Name of the model
            **kwargs: Additional model parameters
            
        Returns:
            Model instance
        """
        if not self.is_initialized:
            await self.initialize()
            
        cache_key = f"{model_name}:{hash(frozenset(kwargs.items()))}"
        
        if cache_key not in self._model_cache:
            self._model_cache[cache_key] = await self.create_model(model_name, **kwargs)
            
        return self._model_cache[cache_key]

    def supports_model(self, model_name: str) -> bool:
        """
        Check if provider supports a specific model.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model is supported
        """
        return model_name in self._available_models

    async def cleanup(self) -> None:
        """
        Clean up provider resources.
        
        Default implementation clears model cache.
        Subclasses should override to clean up additional resources.
        """
        self._model_cache.clear()
        self._initialized = False

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get provider information summary.
        
        Returns:
            Dictionary with provider info
        """
        return {
            "name": self.name,
            "initialized": self.is_initialized,
            "available_models": list(self.available_models),
            "model_count": len(self.available_models),
            "cached_models": len(self._model_cache),
            "config": {
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries,
                "rate_limit": self.config.rate_limit_per_minute,
            }
        }

    def __str__(self) -> str:
        """String representation of provider."""
        return f"{self.__class__.__name__}(name='{self.name}', models={len(self.available_models)})"

    def __repr__(self) -> str:
        """Detailed representation of provider."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"initialized={self.is_initialized}, "
            f"models={len(self.available_models)}"
            f")"
        )