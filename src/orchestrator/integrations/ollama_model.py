"""Ollama model integration for the orchestrator framework."""

from __future__ import annotations

import asyncio
import json
import subprocess
from typing import Any, Dict, List, Optional

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

from orchestrator.core.model import (
    Model,
    ModelCapabilities,
    ModelMetrics,
    ModelRequirements,
)


class OllamaModel(Model):
    """Ollama model implementation."""

    # Model configurations for popular Ollama models
    MODEL_CONFIGS = {
        "gemma2:27b": {
            "capabilities": ModelCapabilities(
                supported_tasks=["generate", "chat", "reasoning", "code", "analyze", "transform", "summarize"],
                context_window=8192,
                supports_function_calling=False,
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh"],
                max_tokens=2048,
                temperature_range=(0.0, 2.0),
            ),
            "requirements": ModelRequirements(
                memory_gb=16.0,
                gpu_memory_gb=12.0,
                cpu_cores=8,
                supports_quantization=["q4_0", "q4_1", "q5_0", "q5_1", "q8_0"],
                min_python_version="3.8",
                requires_gpu=False,
                disk_space_gb=15.0,
            ),
            "metrics": ModelMetrics(
                latency_p50=2.5,
                latency_p95=8.0,
                throughput=15.0,
                accuracy=0.88,
                cost_per_token=0.0,
                success_rate=0.96,
            ),
        },
        "gemma2:9b": {
            "capabilities": ModelCapabilities(
                supported_tasks=["generate", "chat", "reasoning", "code", "analyze", "transform", "summarize"],
                context_window=8192,
                supports_function_calling=False,
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh"],
                max_tokens=2048,
                temperature_range=(0.0, 2.0),
            ),
            "requirements": ModelRequirements(
                memory_gb=8.0,
                gpu_memory_gb=6.0,
                cpu_cores=4,
                supports_quantization=["q4_0", "q4_1", "q5_0", "q5_1", "q8_0"],
                min_python_version="3.8",
                requires_gpu=False,
                disk_space_gb=5.5,
            ),
            "metrics": ModelMetrics(
                latency_p50=1.8,
                latency_p95=5.0,
                throughput=25.0,
                accuracy=0.85,
                cost_per_token=0.0,
                success_rate=0.95,
            ),
        },
        "llama3.2:3b": {
            "capabilities": ModelCapabilities(
                supported_tasks=["generate", "chat", "reasoning", "analyze", "transform", "summarize"],
                context_window=4096,
                supports_function_calling=False,
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr", "de", "it", "pt"],
                max_tokens=1024,
                temperature_range=(0.0, 2.0),
            ),
            "requirements": ModelRequirements(
                memory_gb=4.0,
                gpu_memory_gb=3.0,
                cpu_cores=2,
                supports_quantization=["q4_0", "q4_1", "q5_0", "q5_1", "q8_0"],
                min_python_version="3.8",
                requires_gpu=False,
                disk_space_gb=2.0,
            ),
            "metrics": ModelMetrics(
                latency_p50=1.2,
                latency_p95=3.5,
                throughput=35.0,
                accuracy=0.82,
                cost_per_token=0.0,
                success_rate=0.94,
            ),
        },
        "llama3.2:1b": {
            "capabilities": ModelCapabilities(
                supported_tasks=["generate", "chat", "analyze", "transform", "summarize"],
                context_window=4096,
                supports_function_calling=False,
                supports_structured_output=True,
                supports_streaming=True,
                languages=["en", "es", "fr"],
                max_tokens=1024,
                temperature_range=(0.0, 2.0),
            ),
            "requirements": ModelRequirements(
                memory_gb=2.0,
                gpu_memory_gb=1.5,
                cpu_cores=2,
                supports_quantization=["q4_0", "q4_1", "q5_0", "q5_1", "q8_0"],
                min_python_version="3.8",
                requires_gpu=False,
                disk_space_gb=1.0,
            ),
            "metrics": ModelMetrics(
                latency_p50=0.8,
                latency_p95=2.0,
                throughput=50.0,
                accuracy=0.78,
                cost_per_token=0.0,
                success_rate=0.93,
            ),
        },
    }

    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        timeout: int = 30,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Ollama model.

        Args:
            model_name: Ollama model name (e.g., "gemma2:27b", "llama3.2:3b")
            base_url: Ollama server URL
            timeout: Request timeout in seconds
            **kwargs: Additional arguments passed to parent class
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "Requests library not available. Install with: pip install requests"
            )

        # Get model configuration
        config = self.MODEL_CONFIGS.get(
            model_name,
            {
                "capabilities": ModelCapabilities(
                    supported_tasks=["generate", "chat", "analyze", "transform", "summarize"],
                    context_window=4096,
                    supports_function_calling=False,
                    supports_structured_output=True,
                    supports_streaming=True,
                    languages=["en"],
                    max_tokens=1024,
                    temperature_range=(0.0, 2.0),
                ),
                "requirements": ModelRequirements(
                    memory_gb=4.0,
                    gpu_memory_gb=2.0,
                    cpu_cores=2,
                    supports_quantization=["q4_0", "q4_1", "q5_0", "q5_1", "q8_0"],
                    min_python_version="3.8",
                    requires_gpu=False,
                    disk_space_gb=2.0,
                ),
                "metrics": ModelMetrics(
                    latency_p50=2.0,
                    latency_p95=6.0,
                    throughput=20.0,
                    accuracy=0.80,
                    cost_per_token=0.0,
                    success_rate=0.90,
                ),
            },
        )

        super().__init__(
            name=model_name,
            provider="ollama",
            capabilities=config["capabilities"],
            requirements=config["requirements"],
            metrics=config["metrics"],
            **kwargs,
        )

        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # Set Issue 194 enhanced attributes
        self._expertise = self._get_model_expertise(model_name)
        self._size_billions = self._estimate_model_size(model_name)
        
        # Set cost information (Ollama models are free)
        from ..core.model import ModelCost
        self.cost = ModelCost(is_free=True)

        # Check if Ollama is available
        self._check_ollama_availability()

    def _check_ollama_availability(self) -> None:
        """Check if Ollama is running and available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                self._is_available = True
                # Check if our specific model is available
                models = response.json().get("models", [])
                model_names = [model["name"] for model in models]
                if self.model_name not in model_names:
                    # Try to pull the model
                    self._pull_model()
            else:
                self._is_available = False
        except Exception:
            # Ollama might not be running, try to start it
            if self._start_ollama_if_installed():
                # Try again after starting
                try:
                    response = requests.get(f"{self.base_url}/api/tags", timeout=5)
                    if response.status_code == 200:
                        self._is_available = True
                        # Check if our specific model is available
                        models = response.json().get("models", [])
                        model_names = [model["name"] for model in models]
                        if self.model_name not in model_names:
                            # Try to pull the model
                            self._pull_model()
                    else:
                        self._is_available = False
                except Exception:
                    self._is_available = False
            else:
                self._is_available = False

    def _start_ollama_if_installed(self) -> bool:
        """Try to start Ollama service if it's installed but not running."""
        try:
            # Use the enhanced service manager for better service control
            from orchestrator.utils.service_manager import SERVICE_MANAGERS
            ollama_manager = SERVICE_MANAGERS.get("ollama")
            if ollama_manager:
                return ollama_manager.ensure_running()
            else:
                logger.error("Ollama service manager not found")
                return False
        except ImportError:
            # Fallback to old method
            try:
                from orchestrator.utils.model_utils import start_ollama_server
                return start_ollama_server()
            except ImportError:
                logger.warning("Service management modules not found, cannot auto-start Ollama")
                return False

    def _pull_model(self) -> None:
        """Pull model if not available locally."""
        try:
            # Try using the enhanced service manager first
            from orchestrator.utils.service_manager import SERVICE_MANAGERS
            ollama_manager = SERVICE_MANAGERS.get("ollama")
            if ollama_manager and hasattr(ollama_manager, 'ensure_model_available'):
                if ollama_manager.ensure_model_available(self.model_name):
                    self._is_available = True
                    return
                    
            # Fallback to direct CLI call
            result = subprocess.run(
                ["ollama", "pull", self.model_name],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes timeout for model pull
            )
            if result.returncode == 0:
                self._is_available = True
            else:
                print(
                    f"Warning: Could not pull model {self.model_name}: {result.stderr}"
                )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print(
                f"Warning: Could not pull model {self.model_name} (ollama CLI not available or timeout)"
            )

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text using Ollama model.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        if not self._is_available:
            raise RuntimeError(
                "Ollama model not available. Check if Ollama is running."
            )

        # Validate temperature
        temp_min, temp_max = self.capabilities.temperature_range
        if not temp_min <= temperature <= temp_max:
            raise ValueError(
                f"Temperature {temperature} not in valid range {self.capabilities.temperature_range}"
            )

        # Set default max_tokens if not provided
        if max_tokens is None:
            max_tokens = self.capabilities.max_tokens

        # Prepare request payload
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **kwargs,
            },
        }

        try:
            # Run the synchronous request in a thread pool to avoid blocking
            response = await asyncio.to_thread(
                requests.post,
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()
            return result.get("response", "").strip()

        except Exception as e:
            raise RuntimeError(f"Ollama generation error: {str(e)}") from e

    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate structured output using Ollama model.

        Args:
            prompt: Input prompt
            schema: JSON schema for output structure
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            Structured output matching schema
        """
        if not self.capabilities.supports_structured_output:
            raise ValueError(f"Model {self.name} does not support structured output")

        # Create prompt with schema instructions
        structured_prompt = f"""
{prompt}

Please respond with a JSON object that matches this schema:
{json.dumps(schema, indent=2)}

Return only the JSON object, no additional text.
"""

        try:
            response = await self.generate(
                structured_prompt, temperature=temperature, **kwargs
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
                else:
                    raise ValueError("Could not parse JSON from response")

        except Exception as e:
            raise RuntimeError(f"Ollama structured generation error: {str(e)}") from e

    def _get_model_expertise(self, model_name: str) -> list[str]:
        """Get model expertise areas based on model name."""
        name_lower = model_name.lower()
        
        # Code-specialized models
        if any(x in name_lower for x in ["codellama", "deepseek", "coder", "starcoder"]):
            return ["code", "reasoning", "programming"]
        
        # Fast/compact models
        elif any(x in name_lower for x in ["1b", "3b"]) or "mini" in name_lower:
            return ["fast", "compact", "general"]
        
        # Reasoning models
        elif any(x in name_lower for x in ["wizard", "orca", "vicuna", "reasoning"]):
            return ["reasoning", "analysis", "general"]
        
        # Large capable models
        elif any(x in name_lower for x in ["70b", "405b"]) or "instruct" in name_lower:
            return ["reasoning", "analysis", "creative", "general"]
        
        # Medium models
        elif any(x in name_lower for x in ["7b", "8b", "13b", "27b"]):
            return ["general", "chat", "reasoning"]
        
        # Default
        return ["general"]

    def _estimate_model_size(self, model_name: str) -> float:
        """Estimate model size in billions of parameters from name."""
        from ..utils.model_utils import parse_model_size
        return parse_model_size(model_name, None)

    async def health_check(self) -> bool:
        """
        Check if Ollama model is available and healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Check if Ollama is running
            response = await asyncio.to_thread(
                requests.get, f"{self.base_url}/api/tags", timeout=5
            )
            if response.status_code != 200:
                self._is_available = False
                return False

            # Check if our model is available
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            if self.model_name not in model_names:
                self._is_available = False
                return False

            # Simple test generation
            await self.generate("Test", max_tokens=1, temperature=0.0)
            self._is_available = True
            return True

        except Exception:
            # Try to start Ollama if it's not running
            if await asyncio.to_thread(self._start_ollama_if_installed):
                # Try health check again after starting
                try:
                    response = await asyncio.to_thread(
                        requests.get, f"{self.base_url}/api/tags", timeout=5
                    )
                    if response.status_code == 200:
                        # Check if our model is available
                        models = response.json().get("models", [])
                        model_names = [model["name"] for model in models]
                        if self.model_name not in model_names:
                            # Try to pull the model
                            await asyncio.to_thread(self._pull_model)
                            # Check again
                            response = await asyncio.to_thread(
                                requests.get, f"{self.base_url}/api/tags", timeout=5
                            )
                            models = response.json().get("models", [])
                            model_names = [model["name"] for model in models]
                            if self.model_name not in model_names:
                                self._is_available = False
                                return False
                        
                        # Simple test generation
                        await self.generate("Test", max_tokens=1, temperature=0.0)
                        self._is_available = True
                        return True
                except Exception:
                    pass
                    
            self._is_available = False
            return False

    async def estimate_cost(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
    ) -> float:
        """
        Estimate cost for generation (local models are free).

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Estimated cost in USD (0.0 for local models)
        """
        return 0.0  # Local models are free

    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
        except Exception:
            pass

        return list(self.MODEL_CONFIGS.keys())

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": self.model_name,
            "base_url": self.base_url,
            "available": self._is_available,
            "timeout": self.timeout,
            "supported_quantizations": self.requirements.supports_quantization,
        }

    def is_ollama_running(self) -> bool:
        """Check if Ollama service is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def get_ollama_models(self) -> List[Dict[str, Any]]:
        """Get list of all models available in Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                return response.json().get("models", [])
        except Exception:
            pass
        return []

    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> "OllamaModel":
        """Create Ollama model from configuration."""
        return cls(**config)

    @staticmethod
    def check_ollama_installation() -> bool:
        """Check if Ollama CLI is installed."""
        try:
            result = subprocess.run(
                ["ollama", "--version"], capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    @staticmethod
    def get_recommended_models() -> List[str]:
        """Get list of recommended models for testing."""
        return [
            "gemma2:27b",  # Best quality, high resource requirements
            "gemma2:9b",  # Good balance of quality and resources
            "llama3.2:3b",  # Good for testing, moderate resources
            "llama3.2:1b",  # Fastest, lowest resources
        ]
