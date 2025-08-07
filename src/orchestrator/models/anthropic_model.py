"""Anthropic model adapter implementation with LangChain backend support."""

from __future__ import annotations

import os
import asyncio
import logging
from typing import Any, Dict, Optional

from anthropic import AsyncAnthropic

from ..core.model import Model, ModelCapabilities, ModelRequirements, ModelCost
from ..utils.auto_install import safe_import
from ..utils.api_keys_flexible import ensure_api_key

logger = logging.getLogger(__name__)


class AnthropicModel(Model):
    """Anthropic model implementation."""

    def __init__(
        self,
        name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        capabilities: Optional[ModelCapabilities] = None,
        requirements: Optional[ModelRequirements] = None,
        use_langchain: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Anthropic model with LangChain backend support.

        Args:
            name: Model name (e.g., "claude-3-opus", "claude-3-sonnet")
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            base_url: Custom API base URL (optional)
            capabilities: Model capabilities
            requirements: Resource requirements
            use_langchain: Whether to try using LangChain backend (defaults to True)
            **kwargs: Additional arguments
        """
        # Set default capabilities based on model
        if capabilities is None:
            capabilities = self._get_default_capabilities(name)

        # Set default requirements
        if requirements is None:
            requirements = ModelRequirements(
                memory_gb=0.5,
                cpu_cores=1,
                requires_gpu=False,
                disk_space_gb=0.1,
            )

        # Set cost information
        cost = self._get_model_cost(name)

        super().__init__(
            name=name,
            provider="anthropic",
            capabilities=capabilities,
            requirements=requirements,
            cost=cost,
        )

        # Get API key using existing infrastructure
        self.api_key = api_key
        if not self.api_key:
            try:
                self.api_key = ensure_api_key("anthropic")
            except Exception:
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
                if not self.api_key:
                    raise ValueError("Anthropic API key not provided")

        # Try to initialize LangChain model first
        self.langchain_model = None
        self._use_langchain = False
        
        if use_langchain:
            try:
                langchain_anthropic = safe_import("langchain_anthropic", auto_install=True)
                if langchain_anthropic:
                    self.langchain_model = langchain_anthropic.ChatAnthropic(
                        model=self._normalize_model_name(name),
                        api_key=self.api_key,
                        base_url=base_url,
                        temperature=kwargs.get("temperature", 0.7),
                        max_tokens=kwargs.get("max_tokens"),
                        **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']}
                    )
                    self._use_langchain = True
                    logger.info(f"Using LangChain backend for Anthropic model: {name}")
                else:
                    logger.warning(f"LangChain not available, falling back to direct Anthropic for: {name}")
            except Exception as e:
                logger.warning(f"Failed to initialize LangChain Anthropic model: {e}, falling back to direct Anthropic")

        # Fallback: Initialize direct Anthropic client
        if not self._use_langchain:
            self.client = AsyncAnthropic(
                api_key=self.api_key,
                base_url=base_url,
            )
            logger.info(f"Using direct Anthropic client for model: {name}")

        # Set model-specific attributes (preserve existing functionality)
        self._model_id = self._normalize_model_name(name)
        self._expertise = self._get_model_expertise(name)
        self._size_billions = self._estimate_model_size(name)
        self._is_available = True

    def _normalize_model_name(self, name: str) -> str:
        """Normalize model name to Anthropic format."""
        name_lower = name.lower()

        # Handle Claude Sonnet 4
        if "sonnet-4" in name_lower or "sonnet4" in name_lower:
            return "claude-3-5-sonnet-20241022"  # Latest Sonnet

        # Claude 3.5 Sonnet
        if "claude-3.5-sonnet" in name_lower or "claude-3-5-sonnet" in name_lower:
            return "claude-3-5-sonnet-20241022"

        # Claude 3 Opus
        if "opus" in name_lower:
            return "claude-3-opus-20240229"

        # Claude 3 Sonnet
        if "sonnet" in name_lower and "3.5" not in name_lower:
            return "claude-3-sonnet-20240229"

        # Claude 3 Haiku
        if "haiku" in name_lower:
            return "claude-3-haiku-20240307"

        # Claude 2.1
        if "claude-2.1" in name_lower:
            return "claude-2.1"

        # Claude 2
        if "claude-2" in name_lower:
            return "claude-2.0"

        # Claude Instant
        if "instant" in name_lower:
            return "claude-instant-1.2"

        # Default
        return name

    def _get_default_capabilities(self, name: str) -> ModelCapabilities:
        """Get default capabilities based on model name."""
        name_lower = name.lower()

        # Claude 3.5 Sonnet / Sonnet 4
        if "sonnet" in name_lower and (
            "3.5" in name_lower or "sonnet-4" in name_lower or "sonnet4" in name_lower
        ):
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
                context_window=200000,
                supports_function_calling=True,
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                max_tokens=8192,
                temperature_range=(0.0, 1.0),
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
                context_window=200000,
                supports_function_calling=True,
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                max_tokens=4096,
                temperature_range=(0.0, 1.0),
                vision_capable=True,
                code_specialized=True,
                supports_tools=True,
                accuracy_score=0.95,
                speed_rating="medium",
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
                ],
                context_window=200000,
                supports_function_calling=True,
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                max_tokens=4096,
                temperature_range=(0.0, 1.0),
                vision_capable=True,
                code_specialized=True,
                supports_tools=True,
                accuracy_score=0.90,
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
                context_window=200000,
                supports_function_calling=True,
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                max_tokens=4096,
                temperature_range=(0.0, 1.0),
                vision_capable=True,
                code_specialized=True,
                supports_tools=True,
                accuracy_score=0.85,
                speed_rating="fast",
            )

        # Claude 2.x
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
                context_window=100000,
                supports_function_calling=False,
                supports_structured_output=False,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                max_tokens=4096,
                temperature_range=(0.0, 1.0),
            )

        # Claude Instant
        elif "instant" in name_lower:
            return ModelCapabilities(
                supported_tasks=["generate", "chat", "instruct"],
                context_window=100000,
                supports_function_calling=False,
                supports_structured_output=False,
                supports_streaming=True,
                languages=["en"],
                max_tokens=4096,
                temperature_range=(0.0, 1.0),
            )

        # Default
        return ModelCapabilities(
            supported_tasks=["generate", "chat"],
            context_window=100000,
            supports_function_calling=False,
            supports_structured_output=False,
            supports_streaming=True,
            languages=["en"],
            max_tokens=4096,
        )

    def _get_model_expertise(self, name: str) -> list[str]:
        """Get model expertise areas."""
        name_lower = name.lower()

        if "opus" in name_lower:
            return [
                "general",
                "reasoning",
                "code",
                "creative",
                "analysis",
                "research",
                "math",
            ]
        elif "sonnet" in name_lower:
            if "3.5" in name_lower or "sonnet-4" in name_lower:
                return [
                    "general",
                    "reasoning",
                    "code",
                    "creative",
                    "analysis",
                    "research",
                ]
            return ["general", "reasoning", "code", "analysis"]
        elif "haiku" in name_lower:
            return ["general", "chat", "code"]
        elif "instant" in name_lower:
            return ["general", "chat"]

        return ["general"]

    def _estimate_model_size(self, name: str) -> float:
        """Estimate model size in billions of parameters."""
        name_lower = name.lower()

        if "opus" in name_lower:
            return 175.0  # Estimated
        elif "sonnet" in name_lower:
            return 70.0  # Estimated
        elif "haiku" in name_lower:
            return 20.0  # Estimated
        elif "instant" in name_lower:
            return 10.0  # Estimated

        return 1.0

    def _get_model_cost(self, name: str) -> ModelCost:
        """Get cost information for Anthropic model."""
        name_lower = name.lower()

        # Anthropic pricing (as of 2024)
        if "opus" in name_lower:
            return ModelCost(
                input_cost_per_1k_tokens=15.0 / 1000,  # $15 per 1M input tokens = $0.015 per 1K
                output_cost_per_1k_tokens=75.0 / 1000,  # $75 per 1M output tokens = $0.075 per 1K
                is_free=False,
            )
        elif "sonnet" in name_lower:
            if "3-5" in name_lower or "3.5" in name_lower:
                return ModelCost(
                    input_cost_per_1k_tokens=3.0 / 1000,  # $3 per 1M tokens
                    output_cost_per_1k_tokens=15.0 / 1000,  # $15 per 1M tokens
                    is_free=False,
                )
            else:
                return ModelCost(
                    input_cost_per_1k_tokens=3.0 / 1000,  # $3 per 1M tokens
                    output_cost_per_1k_tokens=15.0 / 1000,  # $15 per 1M tokens
                    is_free=False,
                )
        elif "haiku" in name_lower:
            return ModelCost(
                input_cost_per_1k_tokens=0.25 / 1000,  # $0.25 per 1M tokens
                output_cost_per_1k_tokens=1.25 / 1000,  # $1.25 per 1M tokens
                is_free=False,
            )
        elif "instant" in name_lower:
            return ModelCost(
                input_cost_per_1k_tokens=0.8 / 1000,  # $0.80 per 1M tokens
                output_cost_per_1k_tokens=2.4 / 1000,  # $2.40 per 1M tokens
                is_free=False,
            )
        else:
            # Default pricing for unknown Anthropic models
            return ModelCost(
                input_cost_per_1k_tokens=8.0 / 1000,
                output_cost_per_1k_tokens=24.0 / 1000,
                is_free=False,
            )

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text from prompt using LangChain or direct Anthropic.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        # Use LangChain if available
        if self._use_langchain and self.langchain_model:
            try:
                return await self._langchain_generate(prompt, temperature, max_tokens, **kwargs)
            except Exception as e:
                logger.warning(f"LangChain generation failed, falling back to direct Anthropic: {e}")
                # Fall through to direct Anthropic implementation
        
        # Fallback to direct Anthropic implementation (preserve original functionality)
        return await self._direct_anthropic_generate(prompt, temperature, max_tokens, **kwargs)

    async def _langchain_generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Generate using LangChain backend."""
        try:
            # Handle system prompt for LangChain
            system_prompt = kwargs.get("system_prompt")
            if system_prompt and system_prompt.strip():
                from langchain_core.messages import SystemMessage, HumanMessage
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=prompt)
                ]
                
                if hasattr(self.langchain_model, 'ainvoke'):
                    response = await self.langchain_model.ainvoke(messages)
                else:
                    response = await asyncio.to_thread(self.langchain_model.invoke, messages)
            else:
                # Simple prompt
                if hasattr(self.langchain_model, 'ainvoke'):
                    response = await self.langchain_model.ainvoke(prompt)
                else:
                    response = await asyncio.to_thread(self.langchain_model.invoke, prompt)

            # Extract content from LangChain response
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)

        except Exception as e:
            raise RuntimeError(f"LangChain Anthropic generation failed: {str(e)}")

    async def _direct_anthropic_generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Generate using direct Anthropic client (original implementation)."""
        try:
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]

            # Add system message if provided
            system_prompt = kwargs.get("system_prompt")

            # Make API call
            api_kwargs = {
                "model": self._model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens or self.capabilities.max_tokens,
                "stream": False,
            }
            
            # Only add system prompt if it's provided and non-empty
            if system_prompt and system_prompt.strip():
                api_kwargs["system"] = system_prompt
                
            response = await self.client.messages.create(**api_kwargs)

            # Extract response
            if response.content:
                # Handle different content types
                if isinstance(response.content, list):
                    # Extract text from content blocks
                    text_parts = []
                    for block in response.content:
                        if hasattr(block, "text"):
                            text_parts.append(block.text)
                    return " ".join(text_parts)
                else:
                    return str(response.content)

            return ""

        except Exception as e:
            # Log error and raise
            raise RuntimeError(f"Anthropic generation failed: {str(e)}")

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
            schema: JSON schema for output
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Structured output
        """
        try:
            # Add schema instruction to prompt
            import json
            schema_prompt = f"{prompt}\n\nPlease respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"

            # Generate response using existing generate method (handles LangChain/direct Anthropic automatically)
            response = await self.generate(
                prompt=schema_prompt,
                temperature=temperature,
                **kwargs,
            )

            # Parse JSON response
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r"\{.*\}", response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                raise ValueError("Could not parse JSON from response")

        except Exception as e:
            raise RuntimeError(f"Anthropic structured generation failed: {str(e)}")

    async def health_check(self) -> bool:
        """
        Check if model is available and healthy.

        Returns:
            True if healthy
        """
        try:
            # Use the generate method which handles LangChain/direct Anthropic automatically
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
        # Estimate token counts (Anthropic uses similar tokenization to OpenAI)
        prompt_tokens = len(prompt) // 4  # Rough estimate
        output_tokens = max_tokens or 1000

        # Cost per 1M tokens
        model_lower = self._model_id.lower()
        if "opus" in model_lower:
            input_cost = 15.0  # $15 per 1M input tokens
            output_cost = 75.0  # $75 per 1M output tokens
        elif "sonnet" in model_lower:
            if "3-5" in model_lower or "3.5" in model_lower:
                input_cost = 3.0  # $3 per 1M input tokens
                output_cost = 15.0  # $15 per 1M output tokens
            else:
                input_cost = 3.0  # $3 per 1M input tokens
                output_cost = 15.0  # $15 per 1M output tokens
        elif "haiku" in model_lower:
            input_cost = 0.25  # $0.25 per 1M input tokens
            output_cost = 1.25  # $1.25 per 1M output tokens
        elif "instant" in model_lower:
            input_cost = 0.8  # $0.80 per 1M input tokens
            output_cost = 2.4  # $2.40 per 1M output tokens
        else:
            # Default pricing
            input_cost = 8.0
            output_cost = 24.0

        # Calculate total cost
        total_cost = (prompt_tokens / 1_000_000 * input_cost) + (
            output_tokens / 1_000_000 * output_cost
        )
        return total_cost
