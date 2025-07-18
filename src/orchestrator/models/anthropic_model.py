"""Anthropic model adapter implementation."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import anthropic
from anthropic import AsyncAnthropic

from ..core.model import Model, ModelCapabilities, ModelRequirements


class AnthropicModel(Model):
    """Anthropic model implementation."""

    def __init__(
        self,
        name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        capabilities: Optional[ModelCapabilities] = None,
        requirements: Optional[ModelRequirements] = None,
    ) -> None:
        """
        Initialize Anthropic model.

        Args:
            name: Model name (e.g., "claude-3-opus", "claude-3-sonnet")
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            base_url: Custom API base URL (optional)
            capabilities: Model capabilities
            requirements: Resource requirements
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

        super().__init__(
            name=name,
            provider="anthropic",
            capabilities=capabilities,
            requirements=requirements,
        )

        # Initialize Anthropic client
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")

        self.client = AsyncAnthropic(
            api_key=self.api_key,
            base_url=base_url,
        )

        # Set model-specific attributes
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
        if "sonnet" in name_lower and ("3.5" in name_lower or "sonnet-4" in name_lower or "sonnet4" in name_lower):
            return ModelCapabilities(
                supported_tasks=[
                    "generate", "analyze", "transform", "code", 
                    "reasoning", "creative", "chat", "instruct",
                    "vision", "math", "research"
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
                    "generate", "analyze", "transform", "code", 
                    "reasoning", "creative", "chat", "instruct",
                    "vision", "math", "research"
                ],
                context_window=200000,
                supports_function_calling=True,
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                max_tokens=4096,
                temperature_range=(0.0, 1.0),
            )

        # Claude 3 Sonnet
        elif "sonnet" in name_lower:
            return ModelCapabilities(
                supported_tasks=[
                    "generate", "analyze", "transform", "code", 
                    "reasoning", "creative", "chat", "instruct",
                    "vision"
                ],
                context_window=200000,
                supports_function_calling=True,
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                max_tokens=4096,
                temperature_range=(0.0, 1.0),
            )

        # Claude 3 Haiku
        elif "haiku" in name_lower:
            return ModelCapabilities(
                supported_tasks=[
                    "generate", "analyze", "transform", "code", 
                    "chat", "instruct", "vision"
                ],
                context_window=200000,
                supports_function_calling=True,
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                max_tokens=4096,
                temperature_range=(0.0, 1.0),
            )

        # Claude 2.x
        elif "claude-2" in name_lower:
            return ModelCapabilities(
                supported_tasks=[
                    "generate", "analyze", "transform", "code", 
                    "reasoning", "creative", "chat", "instruct"
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
            return ["general", "reasoning", "code", "creative", "analysis", "research", "math"]
        elif "sonnet" in name_lower:
            if "3.5" in name_lower or "sonnet-4" in name_lower:
                return ["general", "reasoning", "code", "creative", "analysis", "research"]
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

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        try:
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            
            # Add system message if provided
            system_prompt = kwargs.get("system_prompt", "")

            # Make API call
            response = await self.client.messages.create(
                model=self._model_id,
                messages=messages,
                system=system_prompt if system_prompt else None,
                temperature=temperature,
                max_tokens=max_tokens or self.capabilities.max_tokens,
                stream=False,
            )

            # Extract response
            if response.content:
                # Handle different content types
                if isinstance(response.content, list):
                    # Extract text from content blocks
                    text_parts = []
                    for block in response.content:
                        if hasattr(block, 'text'):
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
            schema_prompt = f"{prompt}\n\nPlease respond with valid JSON matching this schema:\n{schema}"
            
            # Generate response
            response = await self.generate(
                prompt=schema_prompt,
                temperature=temperature,
                **kwargs,
            )

            # Parse JSON response
            import json
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                raise

        except Exception as e:
            raise RuntimeError(f"Anthropic structured generation failed: {str(e)}")

    async def health_check(self) -> bool:
        """
        Check if model is available and healthy.

        Returns:
            True if healthy
        """
        try:
            # Try a simple completion
            response = await self.client.messages.create(
                model=self._model_id,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
                temperature=0.0,
            )
            return bool(response.content)
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
        total_cost = (prompt_tokens / 1_000_000 * input_cost) + (output_tokens / 1_000_000 * output_cost)
        return total_cost