"""Local model provider implementation for Ollama and other local model frameworks."""

from __future__ import annotations

import json
import logging
import subprocess
from typing import Any, List, Optional, Set

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

from ...core.model import ModelCapabilities, ModelCost, ModelRequirements
from ...integrations.ollama_model import OllamaModel
from ...utils.auto_install import safe_import
from .base import ModelProvider, ProviderConfig, ProviderInitializationError, ModelNotSupportedError

logger = logging.getLogger(__name__)


class LocalProvider(ModelProvider):
    """Provider for local models (Ollama, HuggingFace transformers, etc.)."""
    
    # Known local model configurations
    OLLAMA_MODEL_CONFIGS = {
        "gemma2:27b": {
            "context_window": 8192,
            "max_tokens": 2048,
            "memory_gb": 16.0,
            "gpu_memory_gb": 12.0,
            "disk_space_gb": 15.0,
            "supports_streaming": True,
            "supports_function_calling": False,
        },
        "gemma2:9b": {
            "context_window": 8192,
            "max_tokens": 2048,
            "memory_gb": 8.0,
            "gpu_memory_gb": 6.0,
            "disk_space_gb": 5.5,
            "supports_streaming": True,
            "supports_function_calling": False,
        },
        "gemma2:2b": {
            "context_window": 8192,
            "max_tokens": 2048,
            "memory_gb": 2.0,
            "gpu_memory_gb": 1.5,
            "disk_space_gb": 1.6,
            "supports_streaming": True,
            "supports_function_calling": False,
        },
        "llama3.2:3b": {
            "context_window": 128000,
            "max_tokens": 2048,
            "memory_gb": 2.5,
            "gpu_memory_gb": 2.0,
            "disk_space_gb": 2.0,
            "supports_streaming": True,
            "supports_function_calling": False,
        },
        "llama3.2:1b": {
            "context_window": 128000,
            "max_tokens": 2048,
            "memory_gb": 1.0,
            "gpu_memory_gb": 0.7,
            "disk_space_gb": 0.7,
            "supports_streaming": True,
            "supports_function_calling": False,
        },
        "llama3.1:8b": {
            "context_window": 128000,
            "max_tokens": 2048,
            "memory_gb": 6.0,
            "gpu_memory_gb": 5.0,
            "disk_space_gb": 4.7,
            "supports_streaming": True,
            "supports_function_calling": True,
        },
        "llama3.1:70b": {
            "context_window": 128000,
            "max_tokens": 2048,
            "memory_gb": 40.0,
            "gpu_memory_gb": 35.0,
            "disk_space_gb": 39.0,
            "supports_streaming": True,
            "supports_function_calling": True,
        },
        "phi3:3.8b": {
            "context_window": 128000,
            "max_tokens": 2048,
            "memory_gb": 3.0,
            "gpu_memory_gb": 2.5,
            "disk_space_gb": 2.3,
            "supports_streaming": True,
            "supports_function_calling": False,
        },
        "qwen2.5:7b": {
            "context_window": 32768,
            "max_tokens": 2048,
            "memory_gb": 5.0,
            "gpu_memory_gb": 4.0,
            "disk_space_gb": 4.1,
            "supports_streaming": True,
            "supports_function_calling": False,
        },
        "mistral:7b": {
            "context_window": 32000,
            "max_tokens": 2048,
            "memory_gb": 5.5,
            "gpu_memory_gb": 4.5,
            "disk_space_gb": 4.0,
            "supports_streaming": True,
            "supports_function_calling": False,
        },
        "mixtral:8x7b": {
            "context_window": 32000,
            "max_tokens": 2048,
            "memory_gb": 30.0,
            "gpu_memory_gb": 24.0,
            "disk_space_gb": 26.0,
            "supports_streaming": True,
            "supports_function_calling": False,
        },
        "codellama:7b": {
            "context_window": 16384,
            "max_tokens": 2048,
            "memory_gb": 5.0,
            "gpu_memory_gb": 4.0,
            "disk_space_gb": 3.8,
            "supports_streaming": True,
            "supports_function_calling": False,
        },
        "codellama:13b": {
            "context_window": 16384,
            "max_tokens": 2048,
            "memory_gb": 9.0,
            "gpu_memory_gb": 7.0,
            "disk_space_gb": 7.3,
            "supports_streaming": True,
            "supports_function_calling": False,
        },
        "deepseek-coder:6.7b": {
            "context_window": 16384,
            "max_tokens": 2048,
            "memory_gb": 5.0,
            "gpu_memory_gb": 4.0,
            "disk_space_gb": 3.9,
            "supports_streaming": True,
            "supports_function_calling": False,
        },
    }

    def __init__(self, config: ProviderConfig) -> None:
        """Initialize local provider."""
        super().__init__(config)
        self._ollama_available = False
        self._ollama_base_url = config.base_url or "http://localhost:11434"
        
    async def initialize(self) -> None:
        """Initialize local provider."""
        try:
            # Check if Ollama is available
            if await self._check_ollama_available():
                self._ollama_available = True
                
                # Discover available Ollama models
                ollama_models = await self._discover_ollama_models()
                self._available_models.update(ollama_models)
                
                logger.info(f"Local provider initialized with Ollama ({len(ollama_models)} models)")
            else:
                logger.warning("Ollama not available. Local provider will have limited functionality.")
                # Still initialize with known models that can be pulled
                self._available_models = set(self.OLLAMA_MODEL_CONFIGS.keys())
            
            self._initialized = True
            
        except Exception as e:
            raise ProviderInitializationError(f"Failed to initialize local provider: {e}")

    async def _check_ollama_available(self) -> bool:
        """Check if Ollama service is available."""
        if not REQUESTS_AVAILABLE:
            logger.warning("requests package not available, cannot check Ollama service")
            return False
            
        try:
            response = requests.get(f"{self._ollama_base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            return False

    async def _discover_ollama_models(self) -> Set[str]:
        """Discover available Ollama models."""
        if not REQUESTS_AVAILABLE or not self._ollama_available:
            return set()
            
        try:
            response = requests.get(f"{self._ollama_base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = set()
                for model in data.get("models", []):
                    models.add(model["name"])
                return models
            else:
                logger.warning(f"Failed to discover Ollama models: HTTP {response.status_code}")
                return set()
        except Exception as e:
            logger.warning(f"Failed to discover Ollama models: {e}")
            return set()

    async def create_model(self, model_name: str, **kwargs: Any) -> OllamaModel:
        """Create a local model instance."""
        if not self.supports_model(model_name):
            raise ModelNotSupportedError(f"Model '{model_name}' not supported by local provider")
        
        # For now, we only support Ollama models
        # Get model specifications
        capabilities = self.get_model_capabilities(model_name)
        requirements = self.get_model_requirements(model_name)
        cost = self.get_model_cost(model_name)
        
        # Create Ollama model instance
        return OllamaModel(
            name=model_name,
            capabilities=capabilities,
            requirements=requirements,
            **kwargs
        )

    async def health_check(self) -> bool:
        """Check if local provider is healthy."""
        if not self._initialized:
            return False
            
        # Check Ollama service if available
        if self._ollama_available:
            return await self._check_ollama_available()
        
        # If no local services are available, consider it unhealthy
        return False

    async def discover_models(self) -> List[str]:
        """Discover available local models."""
        if not self._initialized:
            await self.initialize()
        
        models = []
        
        # Add Ollama models
        if self._ollama_available:
            ollama_models = await self._discover_ollama_models()
            models.extend(ollama_models)
        
        # Add known models that can be pulled
        models.extend(self.OLLAMA_MODEL_CONFIGS.keys())
        
        return list(set(models))  # Remove duplicates

    def supports_model(self, model_name: str) -> bool:
        """Check if provider supports a specific model."""
        # Support models that are either available or known (can be pulled)
        return (
            model_name in self._available_models or 
            model_name in self.OLLAMA_MODEL_CONFIGS
        )

    def get_model_capabilities(self, model_name: str) -> ModelCapabilities:
        """Get capabilities for a local model."""
        if not self.supports_model(model_name):
            raise ModelNotSupportedError(f"Model '{model_name}' not supported by local provider")
        
        model_config = self.OLLAMA_MODEL_CONFIGS.get(model_name, {})
        
        # Determine model family
        name_lower = model_name.lower()
        
        # Code-specialized models
        if "codellama" in name_lower or "code" in name_lower or "deepseek-coder" in name_lower:
            tasks = ["generate", "code", "analyze", "transform", "debug", "explain"]
            code_specialized = True
            domains = ["technical", "code"]
            accuracy_score = 0.92
        # Reasoning models (Llama 3.1)
        elif "llama3.1" in name_lower:
            tasks = ["generate", "chat", "reasoning", "code", "analyze", "transform", "math", "research"]
            code_specialized = True
            domains = ["general", "technical", "reasoning", "math"]
            accuracy_score = 0.90
        # General instruction models
        else:
            tasks = ["generate", "chat", "instruct", "analyze", "transform", "summarize"]
            code_specialized = False
            domains = ["general"]
            accuracy_score = 0.85
        
        return ModelCapabilities(
            supported_tasks=tasks,
            context_window=model_config.get("context_window", 8192),
            supports_function_calling=model_config.get("supports_function_calling", False),
            supports_structured_output=True,  # Most local models support this via formatting
            supports_streaming=model_config.get("supports_streaming", True),
            languages=["en"],  # Most local models primarily support English well
            max_tokens=model_config.get("max_tokens", 2048),
            temperature_range=(0.0, 2.0),
            domains=domains,
            vision_capable=False,  # No vision models configured yet
            audio_capable=False,
            code_specialized=code_specialized,
            supports_tools=False,  # Limited tool support in local models
            supports_json_mode=True,  # Most can output JSON with prompting
            supports_multimodal=False,
            accuracy_score=accuracy_score,
            speed_rating="medium",  # Depends on hardware but generally slower than cloud
        )

    def get_model_requirements(self, model_name: str) -> ModelRequirements:
        """Get resource requirements for a local model."""
        if not self.supports_model(model_name):
            raise ModelNotSupportedError(f"Model '{model_name}' not supported by local provider")
        
        model_config = self.OLLAMA_MODEL_CONFIGS.get(model_name, {})
        
        return ModelRequirements(
            memory_gb=model_config.get("memory_gb", 4.0),
            gpu_memory_gb=model_config.get("gpu_memory_gb", 3.0),
            cpu_cores=4,  # Recommend at least 4 cores for decent performance
            supports_quantization=["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "fp16"],
            min_python_version="3.8",
            requires_gpu=False,  # Can run on CPU but GPU recommended
            disk_space_gb=model_config.get("disk_space_gb", 4.0),
        )

    def get_model_cost(self, model_name: str) -> ModelCost:
        """Get cost information for a local model."""
        if not self.supports_model(model_name):
            raise ModelNotSupportedError(f"Model '{model_name}' not supported by local provider")
        
        # Local models are free to run (after downloading)
        return ModelCost(
            input_cost_per_1k_tokens=0.0,
            output_cost_per_1k_tokens=0.0,
            base_cost_per_request=0.0,
            is_free=True,
        )