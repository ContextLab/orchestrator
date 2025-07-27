"""HuggingFace model integration for the orchestrator framework."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from orchestrator.utils.auto_install import safe_import, ensure_packages

# Try to import required packages with auto-installation
torch = safe_import("torch")
transformers = safe_import("transformers")

if transformers:
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            pipeline,
        )
        TRANSFORMERS_AVAILABLE = True
    except ImportError:
        TRANSFORMERS_AVAILABLE = False
        AutoModelForCausalLM = None
        AutoTokenizer = None
        pipeline = None
        BitsAndBytesConfig = None
else:
    TRANSFORMERS_AVAILABLE = False
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None
    BitsAndBytesConfig = None

from orchestrator.core.model import (
    Model,
    ModelCapabilities,
    ModelMetrics,
    ModelRequirements,
)


class HuggingFaceModel(Model):
    """HuggingFace model implementation."""

    # Model configurations
    MODEL_CONFIGS = {
        "microsoft/DialoGPT-medium": {
            "capabilities": ModelCapabilities(
                supported_tasks=["generate", "chat"],
                context_window=1024,
                supports_function_calling=False,
                supports_structured_output=False,
                supports_streaming=False,
                languages=["en"],
                max_tokens=512,
                temperature_range=(0.0, 2.0),
            ),
            "requirements": ModelRequirements(
                memory_gb=2.0,
                gpu_memory_gb=1.0,
                cpu_cores=2,
                supports_quantization=["8bit", "4bit"],
                min_python_version="3.8",
                requires_gpu=False,
                disk_space_gb=1.0,
            ),
            "metrics": ModelMetrics(
                latency_p50=0.5,
                latency_p95=1.5,
                throughput=50.0,
                accuracy=0.80,
                cost_per_token=0.0,
                success_rate=0.95,
            ),
        },
        "microsoft/DialoGPT-small": {
            "capabilities": ModelCapabilities(
                supported_tasks=["generate", "chat"],
                context_window=1024,
                supports_function_calling=False,
                supports_structured_output=False,
                supports_streaming=False,
                languages=["en"],
                max_tokens=512,
                temperature_range=(0.0, 2.0),
            ),
            "requirements": ModelRequirements(
                memory_gb=1.0,
                gpu_memory_gb=0.5,
                cpu_cores=1,
                supports_quantization=["8bit", "4bit"],
                min_python_version="3.8",
                requires_gpu=False,
                disk_space_gb=0.5,
            ),
            "metrics": ModelMetrics(
                latency_p50=0.3,
                latency_p95=1.0,
                throughput=100.0,
                accuracy=0.75,
                cost_per_token=0.0,
                success_rate=0.95,
            ),
        },
        "gpt2": {
            "capabilities": ModelCapabilities(
                supported_tasks=["generate", "complete"],
                context_window=1024,
                supports_function_calling=False,
                supports_structured_output=False,
                supports_streaming=False,
                languages=["en"],
                max_tokens=512,
                temperature_range=(0.0, 2.0),
            ),
            "requirements": ModelRequirements(
                memory_gb=1.5,
                gpu_memory_gb=0.8,
                cpu_cores=2,
                supports_quantization=["8bit", "4bit"],
                min_python_version="3.8",
                requires_gpu=False,
                disk_space_gb=0.8,
            ),
            "metrics": ModelMetrics(
                latency_p50=0.4,
                latency_p95=1.2,
                throughput=75.0,
                accuracy=0.78,
                cost_per_token=0.0,
                success_rate=0.95,
            ),
        },
        "distilgpt2": {
            "capabilities": ModelCapabilities(
                supported_tasks=["generate", "complete"],
                context_window=1024,
                supports_function_calling=False,
                supports_structured_output=False,
                supports_streaming=False,
                languages=["en"],
                max_tokens=512,
                temperature_range=(0.0, 2.0),
            ),
            "requirements": ModelRequirements(
                memory_gb=0.5,
                gpu_memory_gb=0.3,
                cpu_cores=1,
                supports_quantization=["8bit", "4bit"],
                min_python_version="3.8",
                requires_gpu=False,
                disk_space_gb=0.3,
            ),
            "metrics": ModelMetrics(
                latency_p50=0.2,
                latency_p95=0.6,
                throughput=150.0,
                accuracy=0.70,
                cost_per_token=0.0,
                success_rate=0.95,
            ),
        },
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
            "capabilities": ModelCapabilities(
                supported_tasks=["generate", "chat", "reasoning"],
                context_window=2048,
                supports_function_calling=False,
                supports_structured_output=True,
                supports_streaming=False,
                languages=["en"],
                max_tokens=512,
                temperature_range=(0.0, 2.0),
            ),
            "requirements": ModelRequirements(
                memory_gb=1.5,
                gpu_memory_gb=1.0,
                cpu_cores=2,
                supports_quantization=["8bit", "4bit"],
                min_python_version="3.8",
                requires_gpu=False,
                disk_space_gb=1.2,
            ),
            "metrics": ModelMetrics(
                latency_p50=0.8,
                latency_p95=2.0,
                throughput=40.0,
                accuracy=0.82,
                cost_per_token=0.0,
                success_rate=0.92,
            ),
        },
    }

    def __init__(
        self,
        model_name: str = "distilgpt2",
        device: Optional[str] = None,
        quantization: Optional[str] = None,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize HuggingFace model.

        Args:
            model_name: HuggingFace model name or path
            device: Device to load model on ('cpu', 'cuda', 'auto')
            quantization: Quantization mode ('8bit', '4bit', None)
            cache_dir: Directory to cache models
            token: HuggingFace authentication token
            **kwargs: Additional arguments passed to parent class
        """
        global TRANSFORMERS_AVAILABLE, torch, AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
        if not TRANSFORMERS_AVAILABLE:
            # Try to install on demand
            import subprocess
            import sys

            try:
                print("Transformers library not found. Installing...")
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "transformers", "torch"]
                )
                # Re-import after installation
                import torch
                from transformers import (
                    AutoModelForCausalLM,
                    AutoTokenizer,
                    pipeline,
                    BitsAndBytesConfig,
                )

                TRANSFORMERS_AVAILABLE = True
            except Exception as e:
                raise ImportError(
                    f"Failed to install Transformers library: {e}. Install manually with: pip install transformers torch"
                )

        # Get model configuration
        config = self.MODEL_CONFIGS.get(
            model_name,
            {
                "capabilities": ModelCapabilities(
                    supported_tasks=["generate"],
                    context_window=1024,
                    supports_function_calling=False,
                    supports_structured_output=False,
                    supports_streaming=False,
                    languages=["en"],
                    max_tokens=512,
                    temperature_range=(0.0, 2.0),
                ),
                "requirements": ModelRequirements(
                    memory_gb=2.0,
                    gpu_memory_gb=1.0,
                    cpu_cores=2,
                    supports_quantization=["8bit", "4bit"],
                    min_python_version="3.8",
                    requires_gpu=False,
                    disk_space_gb=1.0,
                ),
                "metrics": ModelMetrics(
                    latency_p50=1.0,
                    latency_p95=3.0,
                    throughput=30.0,
                    accuracy=0.80,
                    cost_per_token=0.0,
                    success_rate=0.95,
                ),
            },
        )

        super().__init__(
            name=model_name,
            provider="huggingface",
            capabilities=config["capabilities"],
            requirements=config["requirements"],
            metrics=config["metrics"],
            **kwargs,
        )

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.quantization = quantization
        self.cache_dir = cache_dir
        self.token = token or os.getenv("HF_TOKEN")

        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._model_loaded = False

        # Quantization config
        self.quantization_config = None
        if quantization:
            if quantization == "8bit":
                self.quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            elif quantization == "4bit":
                self.quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    async def _load_model(self) -> None:
        """Load model and tokenizer if not already loaded."""
        if self._model_loaded:
            return

        print(f"[HuggingFace] Loading model: {self.model_name}")
        print(f"[HuggingFace] Device: {self.device}, Quantization: {self.quantization}")
        print(f"[HuggingFace] Cache dir: {self.cache_dir}")
        print(f"[HuggingFace] Auth token: {'Set' if self.token else 'Not set'}")
        
        try:
            # Load tokenizer
            print(f"[HuggingFace] Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                token=self.token,
            )

            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            print(f"Loading model (this may take a while)...")
            model_kwargs = {
                "cache_dir": self.cache_dir,
                "use_auth_token": self.use_auth_token,
            }

            if self.quantization_config:
                model_kwargs["quantization_config"] = self.quantization_config

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs,
            )

            # Move to device
            if self.device != "auto" and not self.quantization:
                self.model = self.model.to(self.device)

            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
            )

            self._model_loaded = True
            self._is_available = True

        except Exception as e:
            raise RuntimeError(f"Failed to load HuggingFace model: {str(e)}") from e

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text using HuggingFace model.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        await self._load_model()

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
            # Generate with pipeline
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0.0,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )

            # Extract generated text (remove prompt)
            generated_text = outputs[0]["generated_text"]
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt) :].strip()

            return generated_text

        except Exception as e:
            raise RuntimeError(f"HuggingFace generation error: {str(e)}") from e

    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate structured output using HuggingFace model.

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
            raise RuntimeError(
                f"HuggingFace structured generation error: {str(e)}"
            ) from e

    async def health_check(self) -> bool:
        """
        Check if HuggingFace model is available and healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            print(f"[HuggingFace] Starting health check for {self.model_name}")
            
            # Check if transformers is available
            if not TRANSFORMERS_AVAILABLE:
                print(f"[HuggingFace] Transformers library not available")
                return False
                
            # Try to load the model
            print(f"[HuggingFace] Loading model...")
            await self._load_model()
            
            # Check if model was loaded
            if not hasattr(self, 'model') or self.model is None:
                print(f"[HuggingFace] Model failed to load")
                return False
                
            if not hasattr(self, 'tokenizer') or self.tokenizer is None:
                print(f"[HuggingFace] Tokenizer failed to load")
                return False

            # Simple test generation
            print(f"[HuggingFace] Testing generation...")
            result = await self.generate("Test", max_tokens=1, temperature=0.0)
            print(f"[HuggingFace] Test generation result: {result}")
            
            self._is_available = True
            print(f"[HuggingFace] Health check passed")
            return True

        except Exception as e:
            print(f"[HuggingFace] Health check failed for {self.model_name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
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

    def supports_quantization(self) -> bool:
        """Check if model supports quantization."""
        return len(self.requirements.supports_quantization) > 0

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": self.model_name,
            "device": self.device,
            "quantization": self.quantization,
            "loaded": self._model_loaded,
            "supports_quantization": self.supports_quantization(),
            "supported_quantization": self.requirements.supports_quantization,
        }

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage information."""
        if not self._model_loaded or not torch.cuda.is_available():
            return {"gpu_memory_mb": 0.0, "gpu_memory_gb": 0.0}

        try:
            memory_mb = torch.cuda.memory_allocated() / (1024**2)
            return {
                "gpu_memory_mb": memory_mb,
                "gpu_memory_gb": memory_mb / 1024,
            }
        except Exception:
            return {"gpu_memory_mb": 0.0, "gpu_memory_gb": 0.0}

    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._model_loaded = False
        self._is_available = False

    def get_available_models(self) -> List[str]:
        """Get list of available HuggingFace models."""
        return list(self.MODEL_CONFIGS.keys())

    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> "HuggingFaceModel":
        """Create HuggingFace model from configuration."""
        return cls(**config)

    def __del__(self) -> None:
        """Clean up when model is destroyed."""
        try:
            self.unload_model()
        except Exception:
            pass
