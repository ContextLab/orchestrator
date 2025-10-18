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
    
    # Latest Anthropic models (2025) - Simplified for Claude Skills refactor
    KNOWN_MODELS = {
        # Claude Opus 4.1 (Released August 2025)
        "claude-opus-4-1-20250805": {
            "context_window": 200000,
            "max_tokens": 8192,
            "input_cost": 15.0 / 1000,  # $15 per 1M tokens (estimated)
            "output_cost": 75.0 / 1000,  # $75 per 1M tokens (estimated)
            "vision": True,
            "function_calling": True,
            "role": "review_and_analysis",
            "description": "Most powerful Claude model for deep analysis and review",
            "released": "2025-08-05",
        },
        "claude-opus-4.1": {
            "context_window": 200000,
            "max_tokens": 8192,
            "input_cost": 15.0 / 1000,
            "output_cost": 75.0 / 1000,
            "vision": True,
            "function_calling": True,
            "role": "review_and_analysis",
            "description": "Most powerful Claude model for deep analysis and review",
        },

        # Claude Sonnet 4.5 (Released September 2025)
        "claude-sonnet-4-5": {
            "context_window": 1000000,  # 1M token context window for API customers
            "max_tokens": 8192,
            "input_cost": 3.0 / 1000,  # $3 per 1M tokens
            "output_cost": 15.0 / 1000,  # $15 per 1M tokens
            "vision": True,
            "function_calling": True,
            "role": "orchestrator",
            "description": "World's best coding model, optimal for building agents",
            "released": "2025-09-29",
        },
        "claude-sonnet-4.5": {
            "context_window": 1000000,  # 1M token context window
            "max_tokens": 8192,
            "input_cost": 3.0 / 1000,
            "output_cost": 15.0 / 1000,
            "vision": True,
            "function_calling": True,
            "role": "orchestrator",
            "description": "World's best coding model, optimal for building agents",
        },

        # Claude Haiku 4.5 (Released October 2025)
        "claude-haiku-4-5": {
            "context_window": 200000,
            "max_tokens": 8192,
            "input_cost": 1.0 / 1000,  # $1 per 1M tokens
            "output_cost": 5.0 / 1000,  # $5 per 1M tokens
            "vision": True,
            "function_calling": True,
            "role": "simple_tasks",
            "description": "90% of Sonnet 4.5's performance at 1/3 the cost",
            "released": "2025-10-15",
        },
        "claude-haiku-4.5": {
            "context_window": 200000,
            "max_tokens": 8192,
            "input_cost": 1.0 / 1000,
            "output_cost": 5.0 / 1000,
            "vision": True,
            "function_calling": True,
            "role": "simple_tasks",
            "description": "90% of Sonnet 4.5's performance at 1/3 the cost",
        },

        # Legacy models kept for backwards compatibility (will be deprecated)
        "claude-3-5-sonnet-20241022": {
            "context_window": 200000,
            "max_tokens": 8192,
            "input_cost": 3.0 / 1000,
            "output_cost": 15.0 / 1000,
            "vision": True,
            "function_calling": True,
            "deprecated": True,
        },
        "claude-3-haiku-20240307": {
            "context_window": 200000,
            "max_tokens": 4096,
            "input_cost": 0.25 / 1000,
            "output_cost": 1.25 / 1000,
            "vision": True,
            "function_calling": True,
            "deprecated": True,
        },
    }

    def __init__(self, config: ProviderConfig) -> None:
        """Initialize Anthropic provider."""
        super().__init__(config)
        self._client: Optional[AsyncAnthropic] = None
        
    async def initialize(self) -> None:
        """Initialize Anthropic provider."""
        try:
            # Get API key - prioritize config, then use ensure_api_key
            if self.config.api_key:
                api_key = self.config.api_key
            else:
                api_key = ensure_api_key("anthropic")
            
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
                # Try new model first, fall back to current model
                test_models = ["claude-haiku-4.5", "claude-3-haiku-20240307"]
                for test_model in test_models:
                    try:
                        await self._client.messages.create(
                            model=test_model,
                            max_tokens=1,
                            messages=[{"role": "user", "content": "hi"}]
                        )
                        logger.info(f"Anthropic provider initialized with {len(self._available_models)} models (tested with {test_model})")
                        break
                    except Exception as model_error:
                        if "not_found" in str(model_error).lower():
                            continue
                        raise model_error
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

        # Try new models first, fall back to current models
        test_models = ["claude-haiku-4.5", "claude-3-haiku-20240307"]
        for test_model in test_models:
            try:
                await self._client.messages.create(
                    model=test_model,
                    max_tokens=1,
                    messages=[{"role": "user", "content": "test"}]
                )
                return True
            except Exception as e:
                if "not_found" in str(e).lower() and test_model != test_models[-1]:
                    continue  # Try next model
                logger.warning(f"Anthropic health check failed with {test_model}: {e}")

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

        # Claude Sonnet 4.5 (2025)
        if "sonnet-4" in name_lower or "sonnet-4.5" in name_lower:
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
                    "orchestration",
                    "agent_building",
                ],
                context_window=model_info.get("context_window", 1000000),  # 1M tokens
                supports_function_calling=model_info.get("function_calling", True),
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                max_tokens=model_info.get("max_tokens", 8192),
                temperature_range=(0.0, 1.0),
                domains=["general", "technical", "creative", "business", "visual", "agent"],
                vision_capable=model_info.get("vision", True),
                code_specialized=True,
                supports_tools=True,
                supports_json_mode=True,
                accuracy_score=0.98,  # World's best coding model
                speed_rating="medium",
            )

        # Claude Opus 4.1 (2025)
        elif "opus-4" in name_lower or "opus-4.1" in name_lower:
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
                    "review",
                    "deep_analysis",
                ],
                context_window=model_info.get("context_window", 200000),
                supports_function_calling=model_info.get("function_calling", True),
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                max_tokens=model_info.get("max_tokens", 8192),
                temperature_range=(0.0, 1.0),
                domains=["general", "technical", "creative", "business", "visual", "academic"],
                vision_capable=model_info.get("vision", True),
                code_specialized=True,
                supports_tools=True,
                supports_json_mode=True,
                accuracy_score=0.99,  # Most powerful for deep analysis
                speed_rating="slow",
            )

        # Claude Haiku 4.5 (2025)
        elif "haiku-4" in name_lower or "haiku-4.5" in name_lower:
            return ModelCapabilities(
                supported_tasks=[
                    "generate",
                    "analyze",
                    "transform",
                    "code",
                    "chat",
                    "instruct",
                    "vision",
                    "simple_tasks",
                ],
                context_window=model_info.get("context_window", 200000),
                supports_function_calling=model_info.get("function_calling", True),
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                max_tokens=model_info.get("max_tokens", 8192),
                temperature_range=(0.0, 1.0),
                domains=["general", "technical", "visual"],
                vision_capable=model_info.get("vision", True),
                code_specialized=True,
                supports_tools=True,
                supports_json_mode=True,
                accuracy_score=0.90,  # 90% of Sonnet 4.5's performance
                speed_rating="fast",
            )

        # Legacy Claude 3 Opus (deprecated)
        elif "opus" in name_lower and "opus-4" not in name_lower:
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