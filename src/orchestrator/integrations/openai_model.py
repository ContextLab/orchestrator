"""OpenAI model integration for the orchestrator framework."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Union

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None
    openai = None

from orchestrator.core.model import Model, ModelCapabilities, ModelMetrics, ModelRequirements


class OpenAIModel(Model):
    """OpenAI model implementation."""
    
    # Model configurations
    MODEL_CONFIGS = {
        "gpt-4": {
            "capabilities": ModelCapabilities(
                supported_tasks=["generate", "analyze", "transform", "code", "reasoning"],
                context_window=8192,
                supports_function_calling=True,
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                max_tokens=4096,
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
                cost_per_token=0.00003,
                success_rate=0.99,
            ),
        },
        "gpt-4-turbo": {
            "capabilities": ModelCapabilities(
                supported_tasks=["generate", "analyze", "transform", "code", "reasoning", "vision"],
                context_window=128000,
                supports_function_calling=True,
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                max_tokens=4096,
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
                latency_p95=4.0,
                throughput=15.0,
                accuracy=0.96,
                cost_per_token=0.00001,
                success_rate=0.99,
            ),
        },
        "gpt-3.5-turbo": {
            "capabilities": ModelCapabilities(
                supported_tasks=["generate", "analyze", "transform", "code"],
                context_window=16384,
                supports_function_calling=True,
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                max_tokens=4096,
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
                throughput=25.0,
                accuracy=0.88,
                cost_per_token=0.0000015,
                success_rate=0.98,
            ),
        },
    }
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize OpenAI model.
        
        Args:
            model_name: OpenAI model name
            api_key: OpenAI API key (if not provided, will use OPENAI_API_KEY env var)
            base_url: Base URL for API calls
            max_retries: Maximum number of retries for failed requests
            timeout: Request timeout in seconds
            **kwargs: Additional arguments passed to parent class
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI library not available. Install with: pip install openai"
            )
        
        # Get model configuration
        config = self.MODEL_CONFIGS.get(model_name, self.MODEL_CONFIGS["gpt-3.5-turbo"])
        
        super().__init__(
            name=model_name,
            provider="openai",
            capabilities=config["capabilities"],
            requirements=config["requirements"],
            metrics=config["metrics"],
            **kwargs,
        )
        
        # Initialize OpenAI client
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = OpenAI(
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
        Generate text using OpenAI API.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI parameters
            
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
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            
            return response.choices[0].message.content or ""
            
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}") from e
    
    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate structured output using OpenAI API.
        
        Args:
            prompt: Input prompt
            schema: JSON schema for output structure
            temperature: Sampling temperature
            **kwargs: Additional OpenAI parameters
            
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
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": structured_prompt}],
                temperature=temperature,
                **kwargs,
            )
            
            content = response.choices[0].message.content or "{}"
            
            # Parse JSON response
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    raise ValueError("Could not parse JSON from response")
                    
        except Exception as e:
            raise RuntimeError(f"OpenAI structured generation error: {str(e)}") from e
    
    async def health_check(self) -> bool:
        """
        Check if OpenAI API is available and healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Simple test request
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=1,
                temperature=0.0,
            )
            self._is_available = True
            return True
            
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
            **kwargs: Additional OpenAI parameters
            
        Yields:
            Streaming text chunks
        """
        if not self.supports_streaming():
            raise ValueError(f"Model {self.name} does not support streaming")
        
        await self._rate_limit()
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            raise RuntimeError(f"OpenAI streaming error: {str(e)}") from e
    
    def get_available_models(self) -> List[str]:
        """Get list of available OpenAI models."""
        return list(self.MODEL_CONFIGS.keys())
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> "OpenAIModel":
        """Create OpenAI model from configuration."""
        return cls(**config)