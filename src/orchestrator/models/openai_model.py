"""OpenAI model adapter implementation with LangChain backend support."""

from __future__ import annotations

import os
import asyncio
import logging
from typing import Any, Dict, Optional

from openai import AsyncOpenAI

from ..core.model import Model, ModelCapabilities, ModelRequirements, ModelCost
from ..utils.auto_install import safe_import
from ..utils.api_keys_flexible import ensure_api_key

logger = logging.getLogger(__name__)


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
        use_langchain: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize OpenAI model with LangChain backend support.

        Args:
            name: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            organization: OpenAI organization ID (optional)
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
            provider="openai",
            capabilities=capabilities,
            requirements=requirements,
            cost=cost,
        )

        # Get API key using existing infrastructure
        self.api_key = api_key
        if not self.api_key:
            try:
                self.api_key = ensure_api_key("openai")
            except Exception:
                self.api_key = os.getenv("OPENAI_API_KEY")
                if not self.api_key:
                    raise ValueError("OpenAI API key not provided")

        # Try to initialize LangChain model first
        self.langchain_model = None
        self._use_langchain = False
        
        if use_langchain:
            try:
                langchain_openai = safe_import("langchain_openai", auto_install=True)
                if langchain_openai:
                    # Prepare model kwargs based on model type
                    model_kwargs = {
                        "model": name,
                        "api_key": self.api_key,
                        "organization": organization,
                        "base_url": base_url,
                        "temperature": kwargs.get("temperature", 0.7),
                    }
                    
                    # Handle max_tokens vs max_completion_tokens for GPT-5
                    max_tokens_value = kwargs.get("max_tokens")
                    if max_tokens_value:
                        if "gpt-5" in name.lower():
                            model_kwargs["max_completion_tokens"] = max_tokens_value
                        else:
                            model_kwargs["max_tokens"] = max_tokens_value
                    
                    # Add remaining kwargs
                    for k, v in kwargs.items():
                        if k not in ['temperature', 'max_tokens', 'max_completion_tokens']:
                            model_kwargs[k] = v
                    
                    self.langchain_model = langchain_openai.ChatOpenAI(**model_kwargs)
                    self._use_langchain = True
                    logger.info(f"Using LangChain backend for OpenAI model: {name}")
                else:
                    logger.warning(f"LangChain not available, falling back to direct OpenAI for: {name}")
            except Exception as e:
                logger.warning(f"Failed to initialize LangChain OpenAI model: {e}, falling back to direct OpenAI")

        # Always initialize direct OpenAI client for image generation
        # (even if using LangChain for text)
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            organization=organization,
            base_url=base_url,
        )
        
        if not self._use_langchain:
            logger.info(f"Using direct OpenAI client for model: {name}")

        # Set model-specific attributes (preserve existing functionality)
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
        
        # DALL-E models for image generation
        if "dall-e" in name_lower:
            return ModelCapabilities(
                supported_tasks=[
                    "image-generation",
                    "generate-image",
                    "create-image"
                ],
                context_window=4000,  # Prompt length limit
                supports_function_calling=False,
                supports_structured_output=False,
                supports_streaming=False,
                languages=["en"],  # DALL-E works best with English
                max_tokens=4000,  # Prompt token limit
                temperature_range=(0.0, 1.0),
                domains=["visual", "creative", "artistic"],
                vision_capable=False,  # Generates images, doesn't analyze them
                code_specialized=False,
                supports_tools=False
            )

        # GPT-4 models
        if "gpt-4" in name_lower:
            context_window = 128000 if "turbo" in name_lower else 8192
            if "32k" in name_lower:
                context_window = 32768
            
            # Check if it's a vision model
            is_vision = "vision" in name_lower or "gpt-4-turbo" in name_lower or "gpt-4o" in name_lower
            
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
                context_window=context_window,
                supports_function_calling=True,
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                max_tokens=4096,
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
            context_window = 16385 if "16k" in name_lower else 4096

            return ModelCapabilities(
                supported_tasks=[
                    "generate",
                    "analyze",
                    "transform",
                    "code",
                    "chat",
                    "instruct",
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
                    is_free=False,
                )
            else:
                return ModelCost(
                    input_cost_per_1k_tokens=0.03,
                    output_cost_per_1k_tokens=0.06,
                    is_free=False,
                )

        # GPT-3.5 pricing
        elif "gpt-3.5" in name_lower:
            return ModelCost(
                input_cost_per_1k_tokens=0.0005,
                output_cost_per_1k_tokens=0.0015,
                is_free=False,
            )

        # Default pricing for unknown models
        else:
            return ModelCost(
                input_cost_per_1k_tokens=0.002,
                output_cost_per_1k_tokens=0.002,
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
        Generate text from prompt using LangChain or direct OpenAI.

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
                logger.warning(f"LangChain generation failed, falling back to direct OpenAI: {e}")
                # Fall through to direct OpenAI implementation
        
        # Fallback to direct OpenAI implementation (preserve original functionality)
        return await self._direct_openai_generate(prompt, temperature, max_tokens, **kwargs)

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
            if "system_prompt" in kwargs:
                from langchain_core.messages import SystemMessage, HumanMessage
                messages = [
                    SystemMessage(content=kwargs["system_prompt"]),
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
            raise RuntimeError(f"LangChain OpenAI generation failed: {str(e)}")

    async def _direct_openai_generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Generate using direct OpenAI client (original implementation)."""
        try:
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]

            # Add system message if provided
            if "system_prompt" in kwargs:
                messages.insert(
                    0, {"role": "system", "content": kwargs["system_prompt"]}
                )

            # Prepare API call parameters
            api_params = {
                "model": self._model_id,
                "messages": messages,
                "temperature": temperature,
                "n": 1,
                "stream": False,
            }
            
            # Handle max_tokens vs max_completion_tokens based on model
            max_tokens_value = max_tokens or self.capabilities.max_tokens
            if "gpt-5" in self._model_id.lower():
                # GPT-5 models use max_completion_tokens
                api_params["max_completion_tokens"] = max_tokens_value
            else:
                # Older models use max_tokens
                api_params["max_tokens"] = max_tokens_value

            # Make API call
            response = await self.client.chat.completions.create(**api_params)

            # Extract response
            if response.choices and response.choices[0].message:
                return response.choices[0].message.content or ""

            return ""

        except Exception as e:
            # Log error and raise
            raise RuntimeError(f"OpenAI generation failed: {str(e)}")
    
    async def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "vivid",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate image using DALL-E 3.
        
        Args:
            prompt: Text description of the image to generate
            size: Image size (1024x1024, 1792x1024, or 1024x1792)
            quality: Image quality (standard or hd)
            style: Style (vivid or natural)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with image URL and metadata
        """
        # Check if this model supports image generation
        if "dall-e" not in self._model_id.lower():
            raise ValueError(f"Model {self._model_id} doesn't support image generation. Use dall-e-3 or dall-e-2.")
        
        try:
            # Make API call to DALL-E
            response = await self.client.images.generate(
                model=self._model_id,
                prompt=prompt,
                size=size,
                quality=quality,
                style=style,
                n=1  # DALL-E 3 only supports n=1
            )
            
            # Extract image data
            if response.data and len(response.data) > 0:
                image_data = response.data[0]
                return {
                    "url": image_data.url,
                    "revised_prompt": getattr(image_data, 'revised_prompt', prompt),
                    "size": size,
                    "quality": quality,
                    "style": style,
                    "data": [{"url": image_data.url}]  # For compatibility
                }
            
            raise RuntimeError("No image data in response")
            
        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}")
            raise RuntimeError(f"DALL-E generation failed: {str(e)}")

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

            # Generate response using existing generate method (handles LangChain/direct OpenAI automatically)
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
            raise RuntimeError(f"OpenAI structured generation failed: {str(e)}")

    async def health_check(self) -> bool:
        """
        Check if model is available and healthy.

        Returns:
            True if healthy
        """
        try:
            # Use the generate method which handles LangChain/direct OpenAI automatically
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
        total_cost = (prompt_tokens / 1000 * input_cost) + (
            output_tokens / 1000 * output_cost
        )
        return total_cost
