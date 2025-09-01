"""OpenAI model provider implementation."""

from __future__ import annotations

import logging
from typing import Any, List

from openai import AsyncOpenAI

from ...core.model import ModelCapabilities, ModelCost, ModelRequirements
from ...utils.api_keys_flexible import ensure_api_key
from ...utils.auto_install import safe_import
from ..openai_model import OpenAIModel
from .base import ModelProvider, ProviderConfig, ProviderInitializationError, ModelNotSupportedError

logger = logging.getLogger(__name__)


class OpenAIProvider(ModelProvider):
    """Provider for OpenAI models."""
    
    # Known OpenAI models with their specifications
    KNOWN_MODELS = {
        # GPT-4 models
        "gpt-4": {
            "context_window": 8192,
            "max_tokens": 4096,
            "input_cost": 0.03,
            "output_cost": 0.06,
            "vision": False,
            "function_calling": True,
        },
        "gpt-4-32k": {
            "context_window": 32768,
            "max_tokens": 4096,
            "input_cost": 0.06,
            "output_cost": 0.12,
            "vision": False,
            "function_calling": True,
        },
        "gpt-4-turbo": {
            "context_window": 128000,
            "max_tokens": 4096,
            "input_cost": 0.01,
            "output_cost": 0.03,
            "vision": True,
            "function_calling": True,
        },
        "gpt-4-turbo-preview": {
            "context_window": 128000,
            "max_tokens": 4096,
            "input_cost": 0.01,
            "output_cost": 0.03,
            "vision": False,
            "function_calling": True,
        },
        "gpt-4-vision-preview": {
            "context_window": 128000,
            "max_tokens": 4096,
            "input_cost": 0.01,
            "output_cost": 0.03,
            "vision": True,
            "function_calling": True,
        },
        "gpt-4o": {
            "context_window": 128000,
            "max_tokens": 4096,
            "input_cost": 0.005,
            "output_cost": 0.015,
            "vision": True,
            "function_calling": True,
        },
        "gpt-4o-mini": {
            "context_window": 128000,
            "max_tokens": 4096,
            "input_cost": 0.00015,
            "output_cost": 0.0006,
            "vision": True,
            "function_calling": True,
        },
        
        # GPT-3.5 models
        "gpt-3.5-turbo": {
            "context_window": 4096,
            "max_tokens": 4096,
            "input_cost": 0.0005,
            "output_cost": 0.0015,
            "vision": False,
            "function_calling": True,
        },
        "gpt-3.5-turbo-16k": {
            "context_window": 16385,
            "max_tokens": 4096,
            "input_cost": 0.003,
            "output_cost": 0.004,
            "vision": False,
            "function_calling": True,
        },
        "gpt-3.5-turbo-instruct": {
            "context_window": 4096,
            "max_tokens": 4096,
            "input_cost": 0.0015,
            "output_cost": 0.002,
            "vision": False,
            "function_calling": False,
        },
        
        # DALL-E models
        "dall-e-2": {
            "context_window": 1000,
            "max_tokens": 1000,
            "input_cost": 0.0,  # Charged per image, not tokens
            "output_cost": 0.0,
            "vision": False,
            "function_calling": False,
            "image_generation": True,
        },
        "dall-e-3": {
            "context_window": 4000,
            "max_tokens": 4000,
            "input_cost": 0.0,  # Charged per image, not tokens
            "output_cost": 0.0,
            "vision": False,
            "function_calling": False,
            "image_generation": True,
        },
    }

    def __init__(self, config: ProviderConfig) -> None:
        """Initialize OpenAI provider."""
        super().__init__(config)
        self._client: Optional[AsyncOpenAI] = None
        
    async def initialize(self) -> None:
        """Initialize OpenAI provider."""
        try:
            # Get API key
            api_key = ensure_api_key(
                service="openai", 
                api_key=self.config.api_key,
                env_var="OPENAI_API_KEY"
            )
            
            # Initialize client
            self._client = AsyncOpenAI(
                api_key=api_key,
                organization=self.config.organization,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
            
            # Test connectivity
            try:
                models_response = await self._client.models.list()
                available_models = [model.id for model in models_response.data]
                self._available_models = set(available_models)
                logger.info(f"OpenAI provider initialized with {len(available_models)} models")
            except Exception as e:
                # If we can't list models, fall back to known models
                logger.warning(f"Could not list OpenAI models: {e}. Using known models.")
                self._available_models = set(self.KNOWN_MODELS.keys())
                
            self._initialized = True
            
        except Exception as e:
            raise ProviderInitializationError(f"Failed to initialize OpenAI provider: {e}")

    async def create_model(self, model_name: str, **kwargs: Any) -> OpenAIModel:
        """Create an OpenAI model instance."""
        if not self.supports_model(model_name):
            raise ModelNotSupportedError(f"Model '{model_name}' not supported by OpenAI provider")
        
        # Get model specifications
        capabilities = self.get_model_capabilities(model_name)
        requirements = self.get_model_requirements(model_name)
        cost = self.get_model_cost(model_name)
        
        # Create model instance
        return OpenAIModel(
            name=model_name,
            api_key=self._client.api_key if self._client else None,
            organization=self.config.organization,
            base_url=self.config.base_url,
            capabilities=capabilities,
            requirements=requirements,
            **kwargs
        )

    async def health_check(self) -> bool:
        """Check if OpenAI provider is healthy."""
        if not self._initialized or not self._client:
            return False
            
        try:
            # Simple API call to check health
            await self._client.models.list()
            return True
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}")
            return False

    async def discover_models(self) -> List[str]:
        """Discover available OpenAI models."""
        if not self._initialized:
            await self.initialize()
        
        try:
            if self._client:
                models_response = await self._client.models.list()
                return [model.id for model in models_response.data]
            else:
                return list(self.KNOWN_MODELS.keys())
        except Exception as e:
            logger.warning(f"Could not discover OpenAI models: {e}")
            return list(self.KNOWN_MODELS.keys())

    def get_model_capabilities(self, model_name: str) -> ModelCapabilities:
        """Get capabilities for an OpenAI model."""
        if not self.supports_model(model_name):
            raise ModelNotSupportedError(f"Model '{model_name}' not supported by OpenAI provider")
        
        name_lower = model_name.lower()
        model_info = self.KNOWN_MODELS.get(model_name, {})
        
        # DALL-E models for image generation
        if "dall-e" in name_lower:
            return ModelCapabilities(
                supported_tasks=[
                    "image-generation",
                    "generate-image", 
                    "create-image"
                ],
                context_window=model_info.get("context_window", 4000),
                supports_function_calling=False,
                supports_structured_output=False,
                supports_streaming=False,
                languages=["en"],
                max_tokens=model_info.get("max_tokens", 4000),
                temperature_range=(0.0, 1.0),
                domains=["visual", "creative", "artistic"],
                vision_capable=False,  # Generates images, doesn't analyze them
                code_specialized=False,
                supports_tools=False
            )
        
        # GPT-4 models
        if "gpt-4" in name_lower:
            is_vision = model_info.get("vision", False) or "vision" in name_lower
            
            tasks = [
                "generate",
                "analyze", 
                "transform",
                "code",
                "reasoning",
                "creative",
                "chat",
                "instruct",
            ]
            
            if is_vision:
                tasks.extend(["vision", "image-analysis", "visual-reasoning"])

            return ModelCapabilities(
                supported_tasks=tasks,
                context_window=model_info.get("context_window", 8192),
                supports_function_calling=model_info.get("function_calling", True),
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                max_tokens=model_info.get("max_tokens", 4096),
                temperature_range=(0.0, 2.0),
                domains=["general", "technical", "creative", "business", "visual"] if is_vision else ["general", "technical", "creative", "business"],
                vision_capable=is_vision,
                code_specialized=True,
                supports_tools=True,
                supports_json_mode=True,
                accuracy_score=0.95,
                speed_rating="medium",
            )
        
        # GPT-3.5 models
        elif "gpt-3.5" in name_lower:
            return ModelCapabilities(
                supported_tasks=[
                    "generate",
                    "analyze",
                    "transform", 
                    "code",
                    "chat",
                    "instruct",
                ],
                context_window=model_info.get("context_window", 4096),
                supports_function_calling=model_info.get("function_calling", True),
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                max_tokens=model_info.get("max_tokens", 4096),
                temperature_range=(0.0, 2.0),
                domains=["general", "technical"],
                code_specialized=True,
                supports_tools=True,
                supports_json_mode=True,
                accuracy_score=0.85,
                speed_rating="fast",
            )

        # Default capabilities for unknown models
        return ModelCapabilities(
            supported_tasks=["generate", "chat"],
            context_window=model_info.get("context_window", 4096),
            supports_function_calling=model_info.get("function_calling", False),
            supports_structured_output=False,
            supports_streaming=True,
            languages=["en"],
            max_tokens=model_info.get("max_tokens", 4096),
            temperature_range=(0.0, 2.0),
            domains=["general"],
            code_specialized=False,
            supports_tools=False,
            accuracy_score=0.8,
            speed_rating="medium",
        )

    def get_model_requirements(self, model_name: str) -> ModelRequirements:
        """Get resource requirements for an OpenAI model."""
        if not self.supports_model(model_name):
            raise ModelNotSupportedError(f"Model '{model_name}' not supported by OpenAI provider")
        
        # OpenAI models are cloud-hosted, so minimal local requirements
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
        """Get cost information for an OpenAI model."""
        if not self.supports_model(model_name):
            raise ModelNotSupportedError(f"Model '{model_name}' not supported by OpenAI provider")
        
        model_info = self.KNOWN_MODELS.get(model_name, {})
        
        return ModelCost(
            input_cost_per_1k_tokens=model_info.get("input_cost", 0.002),
            output_cost_per_1k_tokens=model_info.get("output_cost", 0.002),
            is_free=False,
        )