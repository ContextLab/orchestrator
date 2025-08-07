"""LangChain model adapter that preserves orchestrator Model interface."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union

from ..core.model import Model, ModelCapabilities, ModelRequirements, ModelCost, ModelMetrics
from ..utils.api_keys_flexible import ensure_api_key
from ..utils.auto_install import safe_import

logger = logging.getLogger(__name__)


class LangChainModelAdapter(Model):
    """
    Adapter that wraps LangChain models to preserve orchestrator Model interface.
    
    This class allows seamless integration of LangChain providers while maintaining
    full compatibility with existing orchestrator code that uses the Model interface.
    """

    def __init__(
        self,
        provider: str,
        model_name: str,
        capabilities: Optional[ModelCapabilities] = None,
        requirements: Optional[ModelRequirements] = None,
        cost: Optional[ModelCost] = None,
        metrics: Optional[ModelMetrics] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize LangChain model adapter.

        Args:
            provider: Provider name (openai, anthropic, ollama, huggingface, google)
            model_name: Model name/identifier
            capabilities: Model capabilities (auto-detected if None)
            requirements: Resource requirements (defaults if None)
            cost: Cost information (auto-detected if None)
            metrics: Performance metrics (defaults if None)
            **kwargs: Additional parameters passed to LangChain model
        """
        self.provider_name = provider
        self.model_name = model_name
        self.langchain_model = None
        self._langchain_available = False

        # Try to create LangChain model
        try:
            self.langchain_model = self._create_langchain_model(provider, model_name, **kwargs)
            self._langchain_available = True
            logger.info(f"Successfully created LangChain model for {provider}:{model_name}")
        except Exception as e:
            logger.warning(f"Failed to create LangChain model for {provider}:{model_name}: {e}")
            self._langchain_available = False

        # Set up model metadata
        if capabilities is None:
            capabilities = self._detect_capabilities(provider, model_name)
        
        if requirements is None:
            requirements = self._get_default_requirements(provider, model_name)
            
        if cost is None:
            cost = self._get_default_cost(provider, model_name)
            
        if metrics is None:
            metrics = self._get_default_metrics(provider, model_name)

        super().__init__(
            name=model_name,
            provider=provider,
            capabilities=capabilities,
            requirements=requirements,
            cost=cost,
            metrics=metrics,
        )

        self._is_available = self._langchain_available

    def _create_langchain_model(self, provider: str, model_name: str, **kwargs: Any) -> Any:
        """Create LangChain model instance based on provider."""
        if provider == "openai":
            return self._create_openai_model(model_name, **kwargs)
        elif provider == "anthropic":
            return self._create_anthropic_model(model_name, **kwargs)
        elif provider == "google":
            return self._create_google_model(model_name, **kwargs)
        elif provider == "ollama":
            return self._create_ollama_model(model_name, **kwargs)
        elif provider == "huggingface":
            return self._create_huggingface_model(model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _create_openai_model(self, model_name: str, **kwargs: Any) -> Any:
        """Create LangChain OpenAI model."""
        langchain_openai = safe_import("langchain_openai", auto_install=True)
        if not langchain_openai:
            raise ImportError("Failed to install langchain-openai")

        api_key = kwargs.pop("api_key", None)
        if not api_key:
            api_key = ensure_api_key("openai")

        return langchain_openai.ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens"),
            **kwargs
        )

    def _create_anthropic_model(self, model_name: str, **kwargs: Any) -> Any:
        """Create LangChain Anthropic model."""
        langchain_anthropic = safe_import("langchain_anthropic", auto_install=True)
        if not langchain_anthropic:
            raise ImportError("Failed to install langchain-anthropic")

        api_key = kwargs.pop("api_key", None)
        if not api_key:
            api_key = ensure_api_key("anthropic")

        return langchain_anthropic.ChatAnthropic(
            model=model_name,
            api_key=api_key,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens"),
            **kwargs
        )

    def _create_google_model(self, model_name: str, **kwargs: Any) -> Any:
        """Create LangChain Google model."""
        langchain_google = safe_import("langchain_google_genai", auto_install=True)
        if not langchain_google:
            raise ImportError("Failed to install langchain-google-genai")

        api_key = kwargs.pop("api_key", None)
        if not api_key:
            api_key = ensure_api_key("google")

        return langchain_google.ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=kwargs.get("temperature", 0.7),
            max_output_tokens=kwargs.get("max_tokens"),
            **kwargs
        )

    def _create_ollama_model(self, model_name: str, **kwargs: Any) -> Any:
        """Create LangChain Ollama model."""
        langchain_community = safe_import("langchain_community", auto_install=True)
        if not langchain_community:
            raise ImportError("Failed to install langchain-community")

        # Ensure Ollama service is running
        from ..utils.service_manager import ensure_service_running
        if not ensure_service_running("ollama"):
            raise RuntimeError("Failed to start Ollama service")

        return langchain_community.chat_models.ChatOllama(
            model=model_name,
            base_url=kwargs.get("base_url", "http://localhost:11434"),
            temperature=kwargs.get("temperature", 0.7),
            **kwargs
        )

    def _create_huggingface_model(self, model_name: str, **kwargs: Any) -> Any:
        """Create LangChain HuggingFace model."""
        langchain_huggingface = safe_import("langchain_huggingface", auto_install=True)
        if not langchain_huggingface:
            raise ImportError("Failed to install langchain-huggingface")

        # Try HuggingFace token
        try:
            hf_token = ensure_api_key("huggingface")
            kwargs["huggingfacehub_api_token"] = hf_token
        except Exception:
            # HF token is optional for public models
            pass

        return langchain_huggingface.HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task="text-generation",
            model_kwargs={"temperature": kwargs.get("temperature", 0.7)},
            **kwargs
        )

    def _detect_capabilities(self, provider: str, model_name: str) -> ModelCapabilities:
        """Auto-detect model capabilities based on provider and model name."""
        name_lower = model_name.lower()
        
        # Base capabilities by provider
        if provider == "openai":
            return self._get_openai_capabilities(name_lower)
        elif provider == "anthropic":
            return self._get_anthropic_capabilities(name_lower)
        elif provider == "google":
            return self._get_google_capabilities(name_lower)
        elif provider == "ollama":
            return self._get_ollama_capabilities(name_lower)
        elif provider == "huggingface":
            return self._get_huggingface_capabilities(name_lower)
        else:
            return self._get_default_capabilities()

    def _get_openai_capabilities(self, model_name: str) -> ModelCapabilities:
        """Get OpenAI model capabilities."""
        if "gpt-4" in model_name:
            context_window = 128000 if "turbo" in model_name else 8192
            return ModelCapabilities(
                supported_tasks=["generate", "analyze", "transform", "code", "reasoning", "creative", "chat", "instruct"],
                context_window=context_window,
                supports_function_calling=True,
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                max_tokens=4096,
                vision_capable="vision" in model_name or "gpt-4-turbo" in model_name,
                code_specialized=True,
                supports_tools=True,
                supports_json_mode=True,
                accuracy_score=0.95,
                speed_rating="medium",
            )
        elif "gpt-3.5" in model_name:
            context_window = 16385 if "16k" in model_name else 4096
            return ModelCapabilities(
                supported_tasks=["generate", "analyze", "transform", "code", "chat", "instruct"],
                context_window=context_window,
                supports_function_calling=True,
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                max_tokens=4096,
                code_specialized=True,
                supports_tools=True,
                supports_json_mode=True,
                accuracy_score=0.85,
                speed_rating="fast",
            )
        else:
            return self._get_default_capabilities()

    def _get_anthropic_capabilities(self, model_name: str) -> ModelCapabilities:
        """Get Anthropic model capabilities."""
        if "opus" in model_name or "sonnet" in model_name:
            return ModelCapabilities(
                supported_tasks=["generate", "analyze", "transform", "code", "reasoning", "creative", "chat", "instruct", "vision"],
                context_window=200000,
                supports_function_calling=True,
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                max_tokens=4096 if "sonnet" in model_name else 8192,
                vision_capable=True,
                code_specialized=True,
                supports_tools=True,
                accuracy_score=0.93 if "opus" in model_name else 0.90,
                speed_rating="medium",
            )
        elif "haiku" in model_name:
            return ModelCapabilities(
                supported_tasks=["generate", "analyze", "transform", "code", "chat", "instruct", "vision"],
                context_window=200000,
                supports_function_calling=True,
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                max_tokens=4096,
                vision_capable=True,
                code_specialized=True,
                supports_tools=True,
                accuracy_score=0.85,
                speed_rating="fast",
            )
        else:
            return self._get_default_capabilities()

    def _get_google_capabilities(self, model_name: str) -> ModelCapabilities:
        """Get Google model capabilities."""
        return ModelCapabilities(
            supported_tasks=["generate", "analyze", "transform", "code", "reasoning", "creative", "chat", "instruct"],
            context_window=32000,
            supports_function_calling=True,
            supports_structured_output=True,
            supports_streaming=True,
            languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
            max_tokens=2048,
            code_specialized=True,
            supports_tools=True,
            accuracy_score=0.88,
            speed_rating="fast",
        )

    def _get_ollama_capabilities(self, model_name: str) -> ModelCapabilities:
        """Get Ollama model capabilities."""
        # Use existing OllamaModel configurations if available
        try:
            from ..integrations.ollama_model import OllamaModel
            if model_name in OllamaModel.MODEL_CONFIGS:
                return OllamaModel.MODEL_CONFIGS[model_name]["capabilities"]
        except ImportError:
            pass

        # Default Ollama capabilities
        return ModelCapabilities(
            supported_tasks=["generate", "chat", "analyze", "transform", "code"],
            context_window=4096,
            supports_function_calling=False,
            supports_structured_output=True,
            supports_streaming=True,
            languages=["en"],
            max_tokens=2048,
            accuracy_score=0.80,
            speed_rating="medium",
        )

    def _get_huggingface_capabilities(self, model_name: str) -> ModelCapabilities:
        """Get HuggingFace model capabilities."""
        return ModelCapabilities(
            supported_tasks=["generate", "chat", "analyze"],
            context_window=2048,
            supports_function_calling=False,
            supports_structured_output=False,
            supports_streaming=True,
            languages=["en"],
            max_tokens=1024,
            accuracy_score=0.75,
            speed_rating="slow",
        )

    def _get_default_capabilities(self) -> ModelCapabilities:
        """Get default capabilities."""
        return ModelCapabilities(
            supported_tasks=["generate", "chat"],
            context_window=4096,
            supports_function_calling=False,
            supports_structured_output=False,
            supports_streaming=True,
            languages=["en"],
            max_tokens=1024,
        )

    def _get_default_requirements(self, provider: str, model_name: str) -> ModelRequirements:
        """Get default resource requirements."""
        if provider == "ollama":
            # Higher requirements for local models
            return ModelRequirements(
                memory_gb=8.0,
                gpu_memory_gb=4.0,
                cpu_cores=4,
                requires_gpu=False,
                disk_space_gb=4.0,
            )
        else:
            # Lower requirements for API models
            return ModelRequirements(
                memory_gb=0.5,
                cpu_cores=1,
                requires_gpu=False,
                disk_space_gb=0.1,
            )

    def _get_default_cost(self, provider: str, model_name: str) -> ModelCost:
        """Get default cost information."""
        if provider == "ollama" or provider == "huggingface":
            return ModelCost(is_free=True)
        else:
            # Use reasonable defaults for API providers
            return ModelCost(
                input_cost_per_1k_tokens=0.002,
                output_cost_per_1k_tokens=0.006,
                is_free=False,
            )

    def _get_default_metrics(self, provider: str, model_name: str) -> ModelMetrics:
        """Get default performance metrics."""
        return ModelMetrics(
            latency_p50=2.0,
            latency_p95=5.0,
            throughput=20.0,
            accuracy=0.80,
            success_rate=0.95,
        )

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text from prompt using LangChain model.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        if not self._langchain_available:
            raise RuntimeError(f"LangChain model not available for {self.provider}:{self.name}")

        try:
            # Prepare the prompt for LangChain
            if hasattr(self.langchain_model, 'ainvoke'):
                response = await self.langchain_model.ainvoke(prompt)
            else:
                # Run sync model in thread pool
                response = await asyncio.to_thread(self.langchain_model.invoke, prompt)

            # Extract content from response
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)

        except Exception as e:
            raise RuntimeError(f"LangChain generation failed for {self.provider}:{self.name}: {str(e)}")

    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate structured output from prompt.

        Args:
            prompt: Input prompt
            schema: JSON schema for output structure
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Structured output matching schema
        """
        if not self.capabilities.supports_structured_output:
            raise ValueError(f"Model {self.name} does not support structured output")

        # Create structured prompt
        schema_prompt = f"{prompt}\n\nPlease respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"

        try:
            response = await self.generate(schema_prompt, temperature, **kwargs)
            
            # Parse JSON response
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                raise ValueError("Could not parse JSON from response")

        except Exception as e:
            raise RuntimeError(f"LangChain structured generation failed: {str(e)}")

    async def health_check(self) -> bool:
        """
        Check if model is available and healthy.

        Returns:
            True if healthy, False otherwise
        """
        if not self._langchain_available:
            return False

        try:
            # Simple health check with minimal prompt
            response = await self.generate("Hi", temperature=0.0, max_tokens=5)
            return len(response) > 0
        except Exception:
            return False

    async def estimate_cost(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
    ) -> float:
        """
        Estimate cost for generation.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Estimated cost in USD
        """
        if self.cost.is_free:
            return 0.0

        # Rough token estimation
        prompt_tokens = len(prompt) // 4
        output_tokens = max_tokens or 1000

        return self.cost.calculate_cost(prompt_tokens, output_tokens)

    @property
    def is_langchain_available(self) -> bool:
        """Check if LangChain model is available."""
        return self._langchain_available

    def get_langchain_model(self) -> Any:
        """Get the underlying LangChain model instance."""
        return self.langchain_model