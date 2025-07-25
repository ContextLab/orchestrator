"""Anthropic model integration for the orchestrator framework."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

try:
    import anthropic
    from anthropic import Anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    Anthropic = None
    anthropic = None

from orchestrator.core.model import (
    Model,
    ModelCapabilities,
    ModelMetrics,
    ModelRequirements,
)


class AnthropicModel(Model):
    """Anthropic model implementation."""

    # Model configurations
    MODEL_CONFIGS = {
        "claude-3-5-sonnet-20241022": {
            "capabilities": ModelCapabilities(
                supported_tasks=[
                    "generate",
                    "analyze",
                    "transform",
                    "code",
                    "reasoning",
                    "vision",
                ],
                context_window=200000,
                supports_function_calling=True,
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                max_tokens=4096,
                temperature_range=(0.0, 1.0),
            ),
            "requirements": ModelRequirements(
                memory_gb=0.1,
                cpu_cores=1,
                disk_space_gb=0.1,
                min_python_version="3.8",
                requires_gpu=False,
            ),
            "metrics": ModelMetrics(
                latency_p50=2.5,
                latency_p95=6.0,
                throughput=12.0,
                accuracy=0.97,
                cost_per_token=0.000015,
                success_rate=0.99,
            ),
        },
        "claude-3-opus-20240229": {
            "capabilities": ModelCapabilities(
                supported_tasks=[
                    "generate",
                    "analyze",
                    "transform",
                    "code",
                    "reasoning",
                    "vision",
                ],
                context_window=200000,
                supports_function_calling=True,
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                max_tokens=4096,
                temperature_range=(0.0, 1.0),
            ),
            "requirements": ModelRequirements(
                memory_gb=0.1,
                cpu_cores=1,
                disk_space_gb=0.1,
                min_python_version="3.8",
                requires_gpu=False,
            ),
            "metrics": ModelMetrics(
                latency_p50=3.0,
                latency_p95=7.0,
                throughput=8.0,
                accuracy=0.98,
                cost_per_token=0.000075,
                success_rate=0.99,
            ),
        },
        "claude-3-haiku-20240307": {
            "capabilities": ModelCapabilities(
                supported_tasks=["generate", "analyze", "transform", "code"],
                context_window=200000,
                supports_function_calling=True,
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                max_tokens=4096,
                temperature_range=(0.0, 1.0),
            ),
            "requirements": ModelRequirements(
                memory_gb=0.1,
                cpu_cores=1,
                disk_space_gb=0.1,
                min_python_version="3.8",
                requires_gpu=False,
            ),
            "metrics": ModelMetrics(
                latency_p50=1.0,
                latency_p95=2.5,
                throughput=20.0,
                accuracy=0.90,
                cost_per_token=0.00000125,
                success_rate=0.98,
            ),
        },
    }

    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Anthropic model.

        Args:
            model_name: Anthropic model name
            api_key: Anthropic API key (if not provided, will use ANTHROPIC_API_KEY env var)
            base_url: Base URL for API calls
            max_retries: Maximum number of retries for failed requests
            timeout: Request timeout in seconds
            **kwargs: Additional arguments passed to parent class
        """
        if not ANTHROPIC_AVAILABLE:
            # Try to install on demand
            import subprocess
            import sys
            global anthropic, Anthropic, ANTHROPIC_AVAILABLE
            try:
                print("Anthropic library not found. Installing...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "anthropic"])
                # Re-import after installation
                import anthropic
                from anthropic import Anthropic
                ANTHROPIC_AVAILABLE = True
            except Exception as e:
                raise ImportError(
                    f"Failed to install Anthropic library: {e}. Install manually with: pip install anthropic"
                )

        # Get model configuration
        config = self.MODEL_CONFIGS.get(
            model_name, self.MODEL_CONFIGS["claude-3-5-sonnet-20241022"]
        )

        super().__init__(
            name=model_name,
            provider="anthropic",
            capabilities=config["capabilities"],
            requirements=config["requirements"],
            metrics=config["metrics"],
            **kwargs,
        )

        # Initialize Anthropic client
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = Anthropic(
            api_key=self.api_key,
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
        )

        self.model_name = model_name
        self.max_retries = max_retries
        self.timeout = timeout

        # Rate limiting
        self._rate_limiter = None
        self._last_request_time = 0.0
        self._min_request_interval = 0.1  # 10 requests per second max

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text using Anthropic API.

        Args:
            prompt: Input prompt (can be string or list of content blocks)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Anthropic parameters (including 'messages' for multimodal)

        Returns:
            Generated text
        """
        await self._rate_limit()

        # Validate temperature
        temp_min, temp_max = self.capabilities.temperature_range
        if not temp_min <= temperature <= temp_max:
            raise ValueError(
                f"Temperature {temperature} not in valid range {self.capabilities.temperature_range}"
            )

        # Set default max_tokens if not provided
        if max_tokens is None:
            max_tokens = self.capabilities.max_tokens

        try:
            # Check if multimodal messages are provided
            if "messages" in kwargs:
                messages = kwargs.pop("messages")
            else:
                # Create messages from prompt
                messages = [{"role": "user", "content": prompt}]

            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                **kwargs,
            )

            return response.content[0].text if response.content else ""

        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {str(e)}") from e

    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate structured output using Anthropic API.

        Args:
            prompt: Input prompt
            schema: JSON schema for output structure
            temperature: Sampling temperature
            **kwargs: Additional Anthropic parameters

        Returns:
            Structured output matching schema
        """
        if not self.capabilities.supports_structured_output:
            raise ValueError(f"Model {self.name} does not support structured output")

        await self._rate_limit()

        # Create prompt with schema instructions
        structured_prompt = f"""
        {prompt}

        Please respond with a JSON object that matches this schema:
        {json.dumps(schema, indent=2)}

        Return only the JSON object, no additional text.
        """

        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.capabilities.max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": structured_prompt}],
                **kwargs,
            )

            content = response.content[0].text if response.content else "{}"

            # Parse JSON response
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re

                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    raise ValueError("Could not parse JSON from response")

        except Exception as e:
            raise RuntimeError(
                f"Anthropic structured generation error: {str(e)}"
            ) from e

    async def health_check(self) -> bool:
        """
        Check if Anthropic API is available and healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Run synchronous client in thread pool to avoid blocking
            import asyncio
            loop = asyncio.get_event_loop()
            
            def _sync_health_check():
                self.client.messages.create(
                    model=self.model_name,
                    max_tokens=1,
                    temperature=0.0,
                    messages=[{"role": "user", "content": "Test"}],
                    timeout=5.0,  # Add explicit timeout
                )
                return True
            
            # Run in executor with timeout
            result = await asyncio.wait_for(
                loop.run_in_executor(None, _sync_health_check),
                timeout=10.0
            )
            self._is_available = result
            return result

        except Exception:
            self._is_available = False
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
        # Rough token estimation (1 token ≈ 4 characters)
        input_tokens = len(prompt) // 4
        output_tokens = max_tokens or 100

        total_tokens = input_tokens + output_tokens
        return total_tokens * self.metrics.cost_per_token

    async def _rate_limit(self) -> None:
        """Apply rate limiting to API requests."""
        import asyncio
        import time

        current_time = time.time()
        time_since_last = current_time - self._last_request_time

        if time_since_last < self._min_request_interval:
            await asyncio.sleep(self._min_request_interval - time_since_last)

        self._last_request_time = time.time()

    def supports_streaming(self) -> bool:
        """Check if model supports streaming."""
        return self.capabilities.supports_streaming

    async def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Generate text with streaming.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Anthropic parameters

        Yields:
            Streaming text chunks
        """
        if not self.supports_streaming():
            raise ValueError(f"Model {self.name} does not support streaming")

        await self._rate_limit()

        try:
            stream = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens or self.capabilities.max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                **kwargs,
            )

            for chunk in stream:
                if chunk.type == "content_block_delta":
                    yield chunk.delta.text

        except Exception as e:
            raise RuntimeError(f"Anthropic streaming error: {str(e)}") from e

    def supports_function_calling(self) -> bool:
        """Check if model supports function calling."""
        return self.capabilities.supports_function_calling

    async def generate_multimodal(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text from multimodal input using Anthropic's native vision support.

        Args:
            messages: List of message dicts with role and content (can include images)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Anthropic parameters

        Returns:
            Generated text
        """
        # For Anthropic models with vision support, we can pass messages directly
        if "vision" in self.capabilities.supported_tasks:
            kwargs["messages"] = messages
            return await self.generate("", temperature, max_tokens, **kwargs)
        else:
            # Fall back to default text-only implementation
            return await super().generate_multimodal(
                messages, temperature, max_tokens, **kwargs
            )

    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate text with tool/function calling.

        Args:
            prompt: Input prompt
            tools: List of tool definitions
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Anthropic parameters

        Returns:
            Response with tool calls
        """
        if not self.supports_function_calling():
            raise ValueError(f"Model {self.name} does not support function calling")

        await self._rate_limit()

        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens or self.capabilities.max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                tools=tools,
                **kwargs,
            )

            return {
                "content": response.content[0].text if response.content else "",
                "tool_calls": [
                    {
                        "name": block.name,
                        "input": block.input,
                    }
                    for block in response.content
                    if hasattr(block, "name")
                ],
            }

        except Exception as e:
            raise RuntimeError(f"Anthropic function calling error: {str(e)}") from e

    def get_available_models(self) -> List[str]:
        """Get list of available Anthropic models."""
        return list(self.MODEL_CONFIGS.keys())

    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> "AnthropicModel":
        """Create Anthropic model from configuration."""
        return cls(**config)
