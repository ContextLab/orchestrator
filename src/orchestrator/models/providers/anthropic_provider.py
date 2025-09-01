"""Anthropic model provider implementation."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from anthropic import AsyncAnthropic

from ...core.model import ModelCapabilities, ModelCost, ModelRequirements
from ...utils.api_keys_flexible import ensure_api_key
from ...utils.auto_install import safe_import
from ..anthropic_model import AnthropicModel
from .base import ModelProvider, ProviderConfig, ProviderInitializationError, ModelNotSupportedError

logger = logging.getLogger(__name__)


class AnthropicProvider(ModelProvider):
    """Provider for Anthropic models."""
    
    # Known Anthropic models with their specifications
    KNOWN_MODELS = {
        # Claude 3.5 Sonnet / Sonnet 4 
        "claude-3-5-sonnet-20241022": {
            "context_window": 200000,
            "max_tokens": 8192,
            "input_cost": 3.0 / 1000,  # $3 per 1M tokens = $0.003 per 1K
            "output_cost": 15.0 / 1000,  # $15 per 1M tokens = $0.015 per 1K
            "vision": True,
            "function_calling": True,
        },
        "claude-3-5-sonnet": {
            "context_window": 200000,
            "max_tokens": 8192,
            "input_cost": 3.0 / 1000,
            "output_cost": 15.0 / 1000,
            "vision": True,
            "function_calling": True,
        },
        
        # Claude 3 Opus
        "claude-3-opus-20240229": {
            "context_window": 200000,
            "max_tokens": 4096,
            "input_cost": 15.0 / 1000,  # $15 per 1M tokens
            "output_cost": 75.0 / 1000,  # $75 per 1M tokens
            "vision": True,
            "function_calling": True,
        },
        "claude-3-opus": {
            "context_window": 200000,
            "max_tokens": 4096,
            "input_cost": 15.0 / 1000,
            "output_cost": 75.0 / 1000,
            "vision": True,
            "function_calling": True,
        },
        
        # Claude 3 Sonnet
        "claude-3-sonnet-20240229": {
            "context_window": 200000,
            "max_tokens": 4096,
            "input_cost": 3.0 / 1000,  # $3 per 1M tokens
            "output_cost": 15.0 / 1000,  # $15 per 1M tokens
            "vision": True,
            "function_calling": True,
        },
        "claude-3-sonnet": {
            "context_window": 200000,
            "max_tokens": 4096,
            "input_cost": 3.0 / 1000,
            "output_cost": 15.0 / 1000,
            "vision": True,
            "function_calling": True,
        },
        
        # Claude 3 Haiku
        "claude-3-haiku-20240307": {
            "context_window": 200000,
            "max_tokens": 4096,
            "input_cost": 0.25 / 1000,  # $0.25 per 1M tokens
            "output_cost": 1.25 / 1000,  # $1.25 per 1M tokens
            "vision": True,
            "function_calling": True,
        },
        "claude-3-haiku": {
            "context_window": 200000,
            "max_tokens": 4096,
            "input_cost": 0.25 / 1000,
            "output_cost": 1.25 / 1000,
            "vision": True,
            "function_calling": True,
        },
        
        # Claude 2.1
        "claude-2.1": {
            "context_window": 200000,
            "max_tokens": 4096,
            "input_cost": 8.0 / 1000,  # $8 per 1M tokens
            "output_cost": 24.0 / 1000,  # $24 per 1M tokens
            "vision": False,
            "function_calling": False,
        },
        
        # Claude 2
        "claude-2.0": {
            "context_window": 100000,
            "max_tokens": 4096,
            "input_cost": 8.0 / 1000,
            "output_cost": 24.0 / 1000,
            "vision": False,
            "function_calling": False,
        },
        "claude-2": {
            "context_window": 100000,
            "max_tokens": 4096,
            "input_cost": 8.0 / 1000,
            "output_cost": 24.0 / 1000,
            "vision": False,
            "function_calling": False,
        },
        
        # Claude Instant
        "claude-instant-1.2": {
            "context_window": 100000,
            "max_tokens": 4096,
            "input_cost": 0.8 / 1000,  # $0.80 per 1M tokens
            "output_cost": 2.4 / 1000,  # $2.40 per 1M tokens
            "vision": False,
            "function_calling": False,
        },
        "claude-instant": {
            "context_window": 100000,
            "max_tokens": 4096,
            "input_cost": 0.8 / 1000,
            "output_cost": 2.4 / 1000,
            "vision": False,
            "function_calling": False,
        },
    }

    def __init__(self, config: ProviderConfig) -> None:
        """Initialize Anthropic provider."""
        super().__init__(config)
        self._client: Optional[AsyncAnthropic] = None
        
    async def initialize(self) -> None:
        """Initialize Anthropic provider."""
        try:
            # Get API key
            api_key = ensure_api_key(
                service="anthropic",
                api_key=self.config.api_key,
                env_var="ANTHROPIC_API_KEY"
            )
            
            # Initialize client
            self._client = AsyncAnthropic(
                api_key=api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
            
            # Anthropic doesn't have a public models endpoint, so use known models
            self._available_models = set(self.KNOWN_MODELS.keys())
            
            # Test connectivity with a simple completion
            try:
                await self._client.messages.create(
                    model="claude-3-haiku-20240307",  # Use cheapest model for health check
                    max_tokens=1,
                    messages=[{"role": "user", "content": "hi"}]
                )
                logger.info(f"Anthropic provider initialized with {len(self._available_models)} models")
            except Exception as e:
                logger.warning(f"Could not test Anthropic connectivity: {e}. Provider may still work.")
                
            self._initialized = True
            
        except Exception as e:
            raise ProviderInitializationError(f"Failed to initialize Anthropic provider: {e}")

    async def create_model(self, model_name: str, **kwargs: Any) -> AnthropicModel:
        """Create an Anthropic model instance."""
        if not self.supports_model(model_name):
            raise ModelNotSupportedError(f"Model '{model_name}' not supported by Anthropic provider")
        
        # Get model specifications
        capabilities = self.get_model_capabilities(model_name)
        requirements = self.get_model_requirements(model_name)
        cost = self.get_model_cost(model_name)
        
        # Create model instance
        return AnthropicModel(
            name=model_name,
            api_key=self._client.api_key if self._client else None,
            base_url=self.config.base_url,
            capabilities=capabilities,
            requirements=requirements,
            **kwargs
        )

    async def health_check(self) -> bool:
        """Check if Anthropic provider is healthy."""
        if not self._initialized or not self._client:
            return False
            
        try:
            # Simple API call to check health
            await self._client.messages.create(
                model="claude-3-haiku-20240307",  # Use cheapest model
                max_tokens=1,
                messages=[{"role": "user", "content": "test"}]
            )
            return True
        except Exception as e:
            logger.warning(f"Anthropic health check failed: {e}")
            return False

    async def discover_models(self) -> List[str]:
        """Discover available Anthropic models."""
        # Anthropic doesn't provide a models discovery endpoint
        # Return known models
        return list(self.KNOWN_MODELS.keys())

    def get_model_capabilities(self, model_name: str) -> ModelCapabilities:
        """Get capabilities for an Anthropic model."""
        if not self.supports_model(model_name):
            raise ModelNotSupportedError(f"Model '{model_name}' not supported by Anthropic provider")
        
        name_lower = model_name.lower()
        model_info = self.KNOWN_MODELS.get(model_name, {})
        
        # Claude 3.5 Sonnet / Sonnet 4
        if "sonnet" in name_lower and ("3-5" in name_lower or "3.5" in name_lower or "sonnet-4" in name_lower):
            return ModelCapabilities(
                supported_tasks=[
                    "generate",
                    "analyze", 
                    "transform",
                    "code",
                    "reasoning",
                    "creative",
                    "chat",
                    "instruct",
                    "vision",
                    "math",
                    "research",
                ],
                context_window=model_info.get("context_window", 200000),
                supports_function_calling=model_info.get("function_calling", True),
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                max_tokens=model_info.get("max_tokens", 8192),
                temperature_range=(0.0, 1.0),
                domains=["general", "technical", "creative", "business", "visual"],
                vision_capable=model_info.get("vision", True),
                code_specialized=True,
                supports_tools=True,
                supports_json_mode=True,
                accuracy_score=0.97,
                speed_rating="medium",
            )
        
        # Claude 3 Opus
        elif "opus" in name_lower:
            return ModelCapabilities(
                supported_tasks=[
                    "generate",
                    "analyze",
                    "transform", 
                    "code",
                    "reasoning",
                    "creative",
                    "chat",
                    "instruct",
                    "vision",
                    "math",
                    "research",
                ],
                context_window=model_info.get("context_window", 200000),
                supports_function_calling=model_info.get("function_calling", True),
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                max_tokens=model_info.get("max_tokens", 4096),
                temperature_range=(0.0, 1.0),
                domains=["general", "technical", "creative", "business", "visual"],
                vision_capable=model_info.get("vision", True),
                code_specialized=True,
                supports_tools=True,
                supports_json_mode=True,
                accuracy_score=0.98,  # Highest accuracy
                speed_rating="slow",
            )
        
        # Claude 3 Sonnet
        elif "sonnet" in name_lower:
            return ModelCapabilities(
                supported_tasks=[
                    "generate",
                    "analyze",
                    "transform",
                    "code", 
                    "reasoning",
                    "creative",
                    "chat",
                    "instruct",
                    "vision",
                    "math",
                    "research",
                ],
                context_window=model_info.get("context_window", 200000),
                supports_function_calling=model_info.get("function_calling", True),
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                max_tokens=model_info.get("max_tokens", 4096),
                temperature_range=(0.0, 1.0),
                domains=["general", "technical", "creative", "business", "visual"],
                vision_capable=model_info.get("vision", True),
                code_specialized=True,
                supports_tools=True,
                supports_json_mode=True,
                accuracy_score=0.93,
                speed_rating="medium",
            )
        
        # Claude 3 Haiku
        elif "haiku" in name_lower:
            return ModelCapabilities(
                supported_tasks=[
                    "generate",
                    "analyze",
                    "transform",
                    "code",
                    "chat",
                    "instruct",
                    "vision",
                ],
                context_window=model_info.get("context_window", 200000),
                supports_function_calling=model_info.get("function_calling", True),
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                max_tokens=model_info.get("max_tokens", 4096),
                temperature_range=(0.0, 1.0),
                domains=["general", "technical", "visual"],
                vision_capable=model_info.get("vision", True),
                code_specialized=True,
                supports_tools=True,
                supports_json_mode=True,
                accuracy_score=0.88,
                speed_rating="fast",
            )
        
        # Claude 2.x models
        elif "claude-2" in name_lower:
            return ModelCapabilities(
                supported_tasks=[
                    "generate",
                    "analyze",
                    "transform",
                    "code",
                    "reasoning", 
                    "creative",
                    "chat",
                    "instruct",
                ],
                context_window=model_info.get("context_window", 100000),
                supports_function_calling=model_info.get("function_calling", False),
                supports_structured_output=False,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                max_tokens=model_info.get("max_tokens", 4096),
                temperature_range=(0.0, 1.0),
                domains=["general", "technical", "creative", "business"],
                vision_capable=False,
                code_specialized=True,
                supports_tools=False,
                supports_json_mode=False,
                accuracy_score=0.90,
                speed_rating="medium",
            )
        
        # Claude Instant
        elif "instant" in name_lower:
            return ModelCapabilities(
                supported_tasks=[
                    "generate",
                    "chat",
                    "instruct",
                    "transform",
                ],
                context_window=model_info.get("context_window", 100000),
                supports_function_calling=False,
                supports_structured_output=False,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                max_tokens=model_info.get("max_tokens", 4096),
                temperature_range=(0.0, 1.0),
                domains=["general"],
                vision_capable=False,
                code_specialized=False,
                supports_tools=False,
                supports_json_mode=False,
                accuracy_score=0.82,
                speed_rating="fast",
            )

        # Default capabilities for unknown models
        return ModelCapabilities(
            supported_tasks=["generate", "chat"],
            context_window=model_info.get("context_window", 100000),
            supports_function_calling=model_info.get("function_calling", False),
            supports_structured_output=False,
            supports_streaming=True,
            languages=["en"],
            max_tokens=model_info.get("max_tokens", 4096),
            temperature_range=(0.0, 1.0),
            domains=["general"],
            code_specialized=False,
            supports_tools=False,
            accuracy_score=0.85,
            speed_rating="medium",
        )

    def get_model_requirements(self, model_name: str) -> ModelRequirements:
        """Get resource requirements for an Anthropic model."""
        if not self.supports_model(model_name):
            raise ModelNotSupportedError(f"Model '{model_name}' not supported by Anthropic provider")
        
        # Anthropic models are cloud-hosted, so minimal local requirements
        return ModelRequirements(
            memory_gb=0.1,  # Minimal memory for API client
            gpu_memory_gb=None,  # No local GPU needed
            cpu_cores=1,
            supports_quantization=[],  # Not applicable for cloud models
            min_python_version="3.8",
            requires_gpu=False,
            disk_space_gb=0.05,  # Just for cached responses
        )

    def get_model_cost(self, model_name: str) -> ModelCost:
        """Get cost information for an Anthropic model."""
        if not self.supports_model(model_name):
            raise ModelNotSupportedError(f"Model '{model_name}' not supported by Anthropic provider")
        
        model_info = self.KNOWN_MODELS.get(model_name, {})
        
        return ModelCost(
            input_cost_per_1k_tokens=model_info.get("input_cost", 8.0 / 1000),  # Default to Claude 2 pricing
            output_cost_per_1k_tokens=model_info.get("output_cost", 24.0 / 1000),
            is_free=False,
        )