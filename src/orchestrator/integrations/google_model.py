"""Google AI model integration for the orchestrator framework."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

try:
    import google.generativeai as genai
    from google.generativeai import GenerativeModel

    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    genai = None
    GenerativeModel = None

from orchestrator.core.model import (
    Model,
    ModelCapabilities,
    ModelMetrics,
    ModelRequirements,
)


class GoogleModel(Model):
    """Google AI model implementation."""

    # Model configurations
    MODEL_CONFIGS = {
        "gemini-1.5-pro": {
            "capabilities": ModelCapabilities(
                supported_tasks=[
                    "generate",
                    "analyze",
                    "transform",
                    "code",
                    "reasoning",
                    "vision",
                ],
                context_window=2000000,
                supports_function_calling=True,
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                max_tokens=8192,
                temperature_range=(0.0, 2.0),
            ),
            "requirements": ModelRequirements(
                memory_gb=0.1,
                cpu_cores=1,
                disk_space_gb=0.1,
                min_python_version="3.8",
                requires_gpu=False,
            ),
            "metrics": ModelMetrics(
                latency_p50=2.0,
                latency_p95=5.0,
                throughput=10.0,
                accuracy=0.95,
                cost_per_token=0.0000035,
                success_rate=0.99,
            ),
        },
        "gemini-1.5-flash": {
            "capabilities": ModelCapabilities(
                supported_tasks=[
                    "generate",
                    "analyze",
                    "transform",
                    "code",
                    "reasoning",
                    "vision",
                ],
                context_window=1000000,
                supports_function_calling=True,
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                max_tokens=8192,
                temperature_range=(0.0, 2.0),
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
                accuracy=0.92,
                cost_per_token=0.00000035,
                success_rate=0.98,
            ),
        },
        "gemini-1.0-pro": {
            "capabilities": ModelCapabilities(
                supported_tasks=[
                    "generate",
                    "analyze",
                    "transform",
                    "code",
                    "reasoning",
                ],
                context_window=32768,
                supports_function_calling=True,
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                max_tokens=8192,
                temperature_range=(0.0, 2.0),
            ),
            "requirements": ModelRequirements(
                memory_gb=0.1,
                cpu_cores=1,
                disk_space_gb=0.1,
                min_python_version="3.8",
                requires_gpu=False,
            ),
            "metrics": ModelMetrics(
                latency_p50=1.5,
                latency_p95=3.0,
                throughput=15.0,
                accuracy=0.90,
                cost_per_token=0.0000005,
                success_rate=0.97,
            ),
        },
    }

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Google AI model.

        Args:
            model_name: Google AI model name
            api_key: Google AI API key (if not provided, will use GOOGLE_AI_API_KEY env var)
            max_retries: Maximum number of retries for failed requests
            timeout: Request timeout in seconds
            **kwargs: Additional arguments passed to parent class
        """
        global GOOGLE_AI_AVAILABLE, genai, GenerativeModel
        if not GOOGLE_AI_AVAILABLE:
            # Try to install on demand
            import subprocess
            import sys

            try:
                print("Google AI library not found. Installing...")
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "google-generativeai"]
                )
                # Re-import after installation
                import google.generativeai as genai
                from google.generativeai import GenerativeModel

                GOOGLE_AI_AVAILABLE = True
            except Exception as e:
                raise ImportError(
                    f"Failed to install Google AI library: {e}. Install manually with: pip install google-generativeai"
                )

        # Get model configuration
        config = self.MODEL_CONFIGS.get(
            model_name, self.MODEL_CONFIGS["gemini-1.5-flash"]
        )

        super().__init__(
            name=model_name,
            provider="google",
            capabilities=config["capabilities"],
            requirements=config["requirements"],
            metrics=config["metrics"],
            **kwargs,
        )

        # Initialize Google AI client
        self.api_key = api_key or os.getenv("GOOGLE_AI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google AI API key not provided. Set GOOGLE_AI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Configure the API
        genai.configure(api_key=self.api_key)

        # Create model instance
        self.model = GenerativeModel(model_name)
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
        Generate text using Google AI API.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Google AI parameters (including 'contents' for multimodal)

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

        # Set up generation config
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens or self.capabilities.max_tokens,
        }

        # Remove generation config params from kwargs
        for key in ["temperature", "max_output_tokens"]:
            kwargs.pop(key, None)

        try:
            # Check if multimodal contents are provided
            if "contents" in kwargs:
                contents = kwargs.pop("contents")
                response = self.model.generate_content(
                    contents,
                    generation_config=generation_config,
                    **kwargs,
                )
            else:
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    **kwargs,
                )

            return response.text if response.text else ""

        except Exception as e:
            raise RuntimeError(f"Google AI API error: {str(e)}") from e

    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate structured output using Google AI API.

        Args:
            prompt: Input prompt
            schema: JSON schema for output structure
            temperature: Sampling temperature
            **kwargs: Additional Google AI parameters

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
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": self.capabilities.max_tokens,
                **kwargs,
            }

            response = self.model.generate_content(
                structured_prompt,
                generation_config=generation_config,
            )

            content = response.text if response.text else "{}"

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
                f"Google AI structured generation error: {str(e)}"
            ) from e

    async def health_check(self) -> bool:
        """
        Check if Google AI API is available and healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Run synchronous client in thread pool to avoid blocking
            import asyncio

            loop = asyncio.get_event_loop()

            def _sync_health_check():
                generation_config = {
                    "temperature": 0.0,
                    "max_output_tokens": 1,
                }
                self.model.generate_content(
                    "Test",
                    generation_config=generation_config,
                    request_options={"timeout": 5.0},  # Add timeout
                )
                return True

            # Run in executor with timeout
            result = await asyncio.wait_for(
                loop.run_in_executor(None, _sync_health_check), timeout=10.0
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
            **kwargs: Additional Google AI parameters

        Yields:
            Streaming text chunks
        """
        if not self.supports_streaming():
            raise ValueError(f"Model {self.name} does not support streaming")

        await self._rate_limit()

        try:
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens or self.capabilities.max_tokens,
                **kwargs,
            }

            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                stream=True,
            )

            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            raise RuntimeError(f"Google AI streaming error: {str(e)}") from e

    def supports_function_calling(self) -> bool:
        """Check if model supports function calling."""
        return self.capabilities.supports_function_calling

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
            **kwargs: Additional Google AI parameters

        Returns:
            Response with tool calls
        """
        if not self.supports_function_calling():
            raise ValueError(f"Model {self.name} does not support function calling")

        await self._rate_limit()

        try:
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens or self.capabilities.max_tokens,
                **kwargs,
            }

            # Convert tools to Google AI format
            google_tools = []
            for tool in tools:
                google_tools.append({"function_declarations": [tool]})

            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                tools=google_tools,
            )

            # Extract tool calls
            tool_calls = []
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "function_call"):
                        tool_calls.append(
                            {
                                "name": part.function_call.name,
                                "args": dict(part.function_call.args),
                            }
                        )

            return {
                "content": response.text if response.text else "",
                "tool_calls": tool_calls,
            }

        except Exception as e:
            raise RuntimeError(f"Google AI function calling error: {str(e)}") from e

    def supports_vision(self) -> bool:
        """Check if model supports vision/image processing."""
        return "vision" in self.capabilities.supported_tasks

    async def generate_with_image(
        self,
        prompt: str,
        image_data: bytes,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text with image input.

        Args:
            prompt: Input prompt
            image_data: Image data as bytes
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Google AI parameters

        Returns:
            Generated text
        """
        if not self.supports_vision():
            raise ValueError(f"Model {self.name} does not support vision")

        await self._rate_limit()

        try:
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens or self.capabilities.max_tokens,
                **kwargs,
            }

            # Create image part
            image_part = {
                "mime_type": "image/jpeg",  # Assume JPEG for simplicity
                "data": image_data,
            }

            response = self.model.generate_content(
                [prompt, image_part],
                generation_config=generation_config,
            )

            return response.text if response.text else ""

        except Exception as e:
            raise RuntimeError(f"Google AI vision error: {str(e)}") from e

    async def generate_multimodal(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text from multimodal input using Google's native vision support.

        Args:
            messages: List of message dicts with role and content (can include images)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Google parameters

        Returns:
            Generated text
        """
        # Convert our format to Google format
        contents = []

        for msg in messages:
            if isinstance(msg.get("content"), list):
                # Build parts for multimodal content
                parts = []
                for block in msg["content"]:
                    if block["type"] == "text":
                        parts.append(block["text"])
                    elif block["type"] == "image":
                        if block.get("source", {}).get("type") == "base64":
                            # Google expects PIL Image or bytes
                            import base64
                            from PIL import Image
                            import io

                            image_data = base64.b64decode(block["source"]["data"])
                            image = Image.open(io.BytesIO(image_data))
                            parts.append(image)
                contents.extend(parts)
            else:
                # Simple text content
                contents.append(msg["content"])

        kwargs["contents"] = contents
        return await self.generate("", temperature, max_tokens, **kwargs)

    def get_available_models(self) -> List[str]:
        """Get list of available Google AI models."""
        return list(self.MODEL_CONFIGS.keys())

    def list_models(self) -> List[str]:
        """List all available models from the API."""
        try:
            models = genai.list_models()
            return [model.name for model in models]
        except Exception:
            return self.get_available_models()

    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> "GoogleModel":
        """Create Google model from configuration."""
        return cls(**config)
