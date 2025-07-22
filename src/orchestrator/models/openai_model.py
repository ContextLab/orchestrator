"""OpenAI model adapter implementation."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from openai import AsyncOpenAI

from ..core.model import Model, ModelCapabilities, ModelRequirements, ModelCost


class OpenAIModel(Model):
    """OpenAI model implementation."""

    def __init__(
        self,
        name: str,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        capabilities: Optional[ModelCapabilities] = None,
        requirements: Optional[ModelRequirements] = None,
    ) -> None:
        """
        Initialize OpenAI model.

        Args:
            name: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            organization: OpenAI organization ID (optional)
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

        # Set cost information
        cost = self._get_model_cost(name)

        super().__init__(
            name=name,
            provider="openai",
            capabilities=capabilities,
            requirements=requirements,
            cost=cost,
        )

        # Initialize OpenAI client
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            organization=organization,
            base_url=base_url,
        )

        # Set model-specific attributes
        self._model_id = self._normalize_model_name(name)
        self._expertise = self._get_model_expertise(name)
        self._size_billions = self._estimate_model_size(name)
        self._is_available = True

    def _normalize_model_name(self, name: str) -> str:
        """Normalize model name to OpenAI format."""
        # Handle common variations
        name_lower = name.lower()
        
        # GPT-4.1 variations
        if "gpt-4.1" in name_lower or "gpt-41" in name_lower:
            if "mini" in name_lower:
                return "gpt-4-0125-preview"  # Using latest GPT-4 as substitute
            return "gpt-4-turbo-preview"
        
        # GPT-4 variations
        if name_lower.startswith("gpt-4"):
            if "turbo" in name_lower:
                return "gpt-4-turbo-preview"
            elif "32k" in name_lower:
                return "gpt-4-32k"
            return "gpt-4"
        
        # GPT-3.5 variations
        if "gpt-3.5" in name_lower or "gpt-35" in name_lower:
            if "16k" in name_lower:
                return "gpt-3.5-turbo-16k"
            return "gpt-3.5-turbo"
        
        # Default: return as-is
        return name

    def _get_default_capabilities(self, name: str) -> ModelCapabilities:
        """Get default capabilities based on model name."""
        name_lower = name.lower()

        # GPT-4 models
        if "gpt-4" in name_lower:
            context_window = 128000 if "turbo" in name_lower else 8192
            if "32k" in name_lower:
                context_window = 32768
            
            return ModelCapabilities(
                supported_tasks=[
                    "generate", "analyze", "transform", "code", 
                    "reasoning", "creative", "chat", "instruct"
                ],
                context_window=context_window,
                supports_function_calling=True,
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                max_tokens=4096,
                temperature_range=(0.0, 2.0),
                domains=["general", "technical", "creative", "business"],
                vision_capable="vision" in name_lower or "gpt-4-turbo" in name_lower,
                code_specialized=True,
                supports_tools=True,
                supports_json_mode=True,
                accuracy_score=0.95,
                speed_rating="medium",
            )

        # GPT-3.5 models
        elif "gpt-3.5" in name_lower:
            context_window = 16385 if "16k" in name_lower else 4096
            
            return ModelCapabilities(
                supported_tasks=[
                    "generate", "analyze", "transform", "code", 
                    "chat", "instruct"
                ],
                context_window=context_window,
                supports_function_calling=True,
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                max_tokens=4096,
                temperature_range=(0.0, 2.0),
                domains=["general", "technical"],
                code_specialized=True,
                supports_tools=True,
                supports_json_mode=True,
                accuracy_score=0.85,
                speed_rating="fast",
            )

        # Default capabilities
        return ModelCapabilities(
            supported_tasks=["generate", "chat"],
            context_window=4096,
            supports_function_calling=False,
            supports_structured_output=False,
            supports_streaming=True,
            languages=["en"],
            max_tokens=2048,
        )

    def _get_model_expertise(self, name: str) -> list[str]:
        """Get model expertise areas."""
        name_lower = name.lower()
        
        if "gpt-4" in name_lower:
            return ["general", "reasoning", "code", "creative", "analysis"]
        elif "gpt-3.5" in name_lower:
            return ["general", "chat", "instruct"]
        
        return ["general"]

    def _estimate_model_size(self, name: str) -> float:
        """Estimate model size in billions of parameters."""
        name_lower = name.lower()
        
        if "gpt-4" in name_lower:
            return 1760.0  # Estimated
        elif "gpt-3.5" in name_lower:
            return 175.0
        
        return 1.0
    
    def _get_model_cost(self, name: str) -> ModelCost:
        """Get cost information for model."""
        name_lower = name.lower()
        
        # GPT-4 pricing (as of 2024)
        if "gpt-4" in name_lower:
            if "turbo" in name_lower or "preview" in name_lower:
                return ModelCost(
                    input_cost_per_1k_tokens=0.01,
                    output_cost_per_1k_tokens=0.03,
                    is_free=False
                )
            else:
                return ModelCost(
                    input_cost_per_1k_tokens=0.03,
                    output_cost_per_1k_tokens=0.06,
                    is_free=False
                )
        
        # GPT-3.5 pricing
        elif "gpt-3.5" in name_lower:
            return ModelCost(
                input_cost_per_1k_tokens=0.0005,
                output_cost_per_1k_tokens=0.0015,
                is_free=False
            )
        
        # Default pricing for unknown models
        else:
            return ModelCost(
                input_cost_per_1k_tokens=0.002,
                output_cost_per_1k_tokens=0.002,
                is_free=False
            )

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
            if "system_prompt" in kwargs:
                messages.insert(0, {"role": "system", "content": kwargs["system_prompt"]})

            # Make API call
            response = await self.client.chat.completions.create(
                model=self._model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or self.capabilities.max_tokens,
                n=1,
                stream=False,
            )

            # Extract response
            if response.choices and response.choices[0].message:
                return response.choices[0].message.content or ""
            
            return ""

        except Exception as e:
            # Log error and raise
            raise RuntimeError(f"OpenAI generation failed: {str(e)}")

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
            raise RuntimeError(f"OpenAI structured generation failed: {str(e)}")

    async def health_check(self) -> bool:
        """
        Check if model is available and healthy.

        Returns:
            True if healthy
        """
        try:
            # Try a simple completion
            response = await self.client.chat.completions.create(
                model=self._model_id,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
                temperature=0.0,
            )
            return bool(response.choices)
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
        # Estimate token counts
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model(self._model_id)
            prompt_tokens = len(encoding.encode(prompt))
        except Exception:
            # Fallback: rough estimate
            prompt_tokens = len(prompt) // 4

        output_tokens = max_tokens or 1000

        # Cost per 1K tokens (approximate)
        model_lower = self._model_id.lower()
        if "gpt-4" in model_lower:
            if "turbo" in model_lower:
                input_cost = 0.01  # $0.01 per 1K input tokens
                output_cost = 0.03  # $0.03 per 1K output tokens
            else:
                input_cost = 0.03  # $0.03 per 1K input tokens
                output_cost = 0.06  # $0.06 per 1K output tokens
        elif "gpt-3.5" in model_lower:
            input_cost = 0.0005  # $0.0005 per 1K input tokens
            output_cost = 0.0015  # $0.0015 per 1K output tokens
        else:
            # Default pricing
            input_cost = 0.002
            output_cost = 0.002

        # Calculate total cost
        total_cost = (prompt_tokens / 1000 * input_cost) + (output_tokens / 1000 * output_cost)
        return total_cost