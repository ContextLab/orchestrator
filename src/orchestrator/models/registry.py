"""Unified model registry with provider abstractions."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set

from ..core.model import Model
from .providers.base import ModelProvider, ProviderConfig, ProviderError
from .providers.openai_provider import OpenAIProvider
from .providers.anthropic_provider import AnthropicProvider
from .providers.local_provider import LocalProvider

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Unified model registry that manages multiple providers.
    
    This registry provides a unified interface for discovering, configuring,
    and accessing models from different providers (OpenAI, Anthropic, local, etc.).
    """
    
    def __init__(self) -> None:
        """Initialize model registry."""
        self._providers: Dict[str, ModelProvider] = {}
        self._model_cache: Dict[str, Model] = {}
        self._initialized = False
        
    @property
    def is_initialized(self) -> bool:
        """Check if registry is initialized."""
        return self._initialized
        
    @property
    def providers(self) -> Dict[str, ModelProvider]:
        """Get all registered providers."""
        return self._providers.copy()
    
    @property
    def available_models(self) -> Dict[str, str]:
        """
        Get all available models across providers.
        
        Returns:
            Dictionary mapping model names to provider names
        """
        models = {}
        for provider_name, provider in self._providers.items():
            for model_name in provider.available_models:
                models[model_name] = provider_name
        return models
    
    def register_provider(self, provider: ModelProvider) -> None:
        """
        Register a model provider.
        
        Args:
            provider: The provider to register
        """
        self._providers[provider.name] = provider
        logger.info(f"Registered provider: {provider.name}")
    
    def configure_provider(
        self,
        provider_name: str,
        provider_type: str,
        config: Dict[str, Any]
    ) -> None:
        """
        Configure and register a provider.
        
        Args:
            provider_name: Name for the provider instance
            provider_type: Type of provider (openai, anthropic, local)
            config: Provider configuration
        """
        provider_config = ProviderConfig(name=provider_name, **config)
        
        if provider_type.lower() == "openai":
            provider = OpenAIProvider(provider_config)
        elif provider_type.lower() == "anthropic":
            provider = AnthropicProvider(provider_config)
        elif provider_type.lower() == "local":
            provider = LocalProvider(provider_config)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")
        
        self.register_provider(provider)
    
    async def initialize(self) -> None:
        """Initialize all registered providers."""
        if self._initialized:
            return
            
        initialization_tasks = []
        for provider_name, provider in self._providers.items():
            if not provider.is_initialized:
                initialization_tasks.append(self._initialize_provider(provider_name, provider))
        
        if initialization_tasks:
            results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
            
            # Log any initialization failures
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    provider_name = list(self._providers.keys())[i]
                    logger.error(f"Failed to initialize provider {provider_name}: {result}")
        
        self._initialized = True
        logger.info(f"Registry initialized with {len(self._providers)} providers")
    
    async def _initialize_provider(self, provider_name: str, provider: ModelProvider) -> None:
        """Initialize a single provider with error handling."""
        try:
            await provider.initialize()
            logger.info(f"Provider {provider_name} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize provider {provider_name}: {e}")
            raise
    
    async def discover_all_models(self) -> Dict[str, List[str]]:
        """
        Discover models from all providers.
        
        Returns:
            Dictionary mapping provider names to lists of model names
        """
        if not self._initialized:
            await self.initialize()
        
        discovery_tasks = []
        provider_names = []
        
        for provider_name, provider in self._providers.items():
            if provider.is_initialized:
                discovery_tasks.append(provider.discover_models())
                provider_names.append(provider_name)
        
        if not discovery_tasks:
            return {}
        
        results = await asyncio.gather(*discovery_tasks, return_exceptions=True)
        
        discovered_models = {}
        for i, result in enumerate(results):
            provider_name = provider_names[i]
            if isinstance(result, Exception):
                logger.error(f"Failed to discover models from {provider_name}: {result}")
                discovered_models[provider_name] = []
            else:
                discovered_models[provider_name] = result
        
        return discovered_models
    
    def find_model(self, model_name: str) -> Optional[str]:
        """
        Find which provider supports a model.
        
        Args:
            model_name: Name of the model to find
            
        Returns:
            Provider name if found, None otherwise
        """
        for provider_name, provider in self._providers.items():
            if provider.supports_model(model_name):
                return provider_name
        return None
    
    async def get_model(self, model_name: str, provider_name: Optional[str] = None, **kwargs: Any) -> Model:
        """
        Get a model instance.
        
        Args:
            model_name: Name of the model
            provider_name: Specific provider to use (auto-detect if None)
            **kwargs: Additional model parameters
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model not found or provider not available
        """
        if not self._initialized:
            await self.initialize()
        
        # Auto-detect provider if not specified
        if provider_name is None:
            provider_name = self.find_model(model_name)
            if provider_name is None:
                raise ValueError(f"Model '{model_name}' not found in any provider")
        
        # Check if provider exists
        if provider_name not in self._providers:
            raise ValueError(f"Provider '{provider_name}' not registered")
        
        provider = self._providers[provider_name]
        
        # Check if provider supports the model
        if not provider.supports_model(model_name):
            raise ValueError(f"Provider '{provider_name}' does not support model '{model_name}'")
        
        # Create cache key
        cache_key = f"{provider_name}:{model_name}:{hash(frozenset(kwargs.items()))}"
        
        # Return cached model if available
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        # Create new model instance
        try:
            model = await provider.get_model(model_name, **kwargs)
            self._model_cache[cache_key] = model
            return model
        except Exception as e:
            raise ValueError(f"Failed to create model '{model_name}' from provider '{provider_name}': {e}")
    
    async def health_check(self) -> Dict[str, bool]:
        """
        Check health of all providers.
        
        Returns:
            Dictionary mapping provider names to health status
        """
        health_tasks = []
        provider_names = []
        
        for provider_name, provider in self._providers.items():
            if provider.is_initialized:
                health_tasks.append(provider.health_check())
                provider_names.append(provider_name)
        
        if not health_tasks:
            return {}
        
        results = await asyncio.gather(*health_tasks, return_exceptions=True)
        
        health_status = {}
        for i, result in enumerate(results):
            provider_name = provider_names[i]
            if isinstance(result, Exception):
                logger.error(f"Health check failed for {provider_name}: {result}")
                health_status[provider_name] = False
            else:
                health_status[provider_name] = result
        
        return health_status
    
    def get_registry_info(self) -> Dict[str, Any]:
        """
        Get registry information summary.
        
        Returns:
            Dictionary with registry information
        """
        provider_info = {}
        for provider_name, provider in self._providers.items():
            provider_info[provider_name] = provider.get_provider_info()
        
        return {
            "initialized": self._initialized,
            "provider_count": len(self._providers),
            "total_models": len(self.available_models),
            "cached_models": len(self._model_cache),
            "providers": provider_info,
        }
    
    def list_models(self, provider_name: Optional[str] = None) -> Dict[str, Any]:
        """
        List available models with details.
        
        Args:
            provider_name: Filter by provider (all providers if None)
            
        Returns:
            Dictionary with model information
        """
        models = {}
        
        providers_to_check = (
            {provider_name: self._providers[provider_name]} if provider_name
            else self._providers
        )
        
        for prov_name, provider in providers_to_check.items():
            if not provider.is_initialized:
                continue
                
            for model_name in provider.available_models:
                try:
                    capabilities = provider.get_model_capabilities(model_name)
                    requirements = provider.get_model_requirements(model_name)
                    cost = provider.get_model_cost(model_name)
                    
                    models[model_name] = {
                        "provider": prov_name,
                        "capabilities": capabilities.to_dict(),
                        "requirements": requirements.to_dict(),
                        "cost": cost.to_dict() if hasattr(cost, 'to_dict') else {
                            "input_cost_per_1k_tokens": cost.input_cost_per_1k_tokens,
                            "output_cost_per_1k_tokens": cost.output_cost_per_1k_tokens,
                            "is_free": cost.is_free,
                        },
                    }
                except Exception as e:
                    logger.warning(f"Failed to get info for model {model_name}: {e}")
                    models[model_name] = {
                        "provider": prov_name,
                        "error": str(e),
                    }
        
        return models
    
    async def cleanup(self) -> None:
        """Clean up registry and all providers."""
        cleanup_tasks = []
        for provider in self._providers.values():
            if provider.is_initialized:
                cleanup_tasks.append(provider.cleanup())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        self._model_cache.clear()
        self._providers.clear()
        self._initialized = False
        logger.info("Registry cleaned up")
    
    def __str__(self) -> str:
        """String representation of registry."""
        return f"ModelRegistry(providers={len(self._providers)}, models={len(self.available_models)})"
    
    def __repr__(self) -> str:
        """Detailed representation of registry."""
        return (
            f"ModelRegistry("
            f"providers={list(self._providers.keys())}, "
            f"models={len(self.available_models)}, "
            f"initialized={self._initialized}"
            f")"
        )