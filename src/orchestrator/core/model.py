"""Model abstractions for the orchestrator framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelCapabilities:
    """Defines what a model can do."""

    supported_tasks: List[str] = field(default_factory=list)
    context_window: int = 4096
    supports_function_calling: bool = False
    supports_structured_output: bool = False
    supports_streaming: bool = False
    languages: List[str] = field(default_factory=lambda: ["en"])
    max_tokens: Optional[int] = None
    temperature_range: tuple[float, float] = (0.0, 2.0)

    # Enhanced capabilities for intelligent routing
    domains: List[str] = field(default_factory=list)  # e.g., ["medical", "legal", "creative"]
    vision_capable: bool = False
    audio_capable: bool = False
    code_specialized: bool = False
    supports_tools: bool = False  # Tool/function calling
    supports_json_mode: bool = False  # Native JSON output mode
    supports_multimodal: bool = False  # Multiple input/output modalities
    accuracy_score: float = 0.85  # 0-1 score for general accuracy
    speed_rating: str = "medium"  # "fast", "medium", "slow"

    def __post_init__(self) -> None:
        """Validate capabilities after initialization."""
        if self.context_window <= 0:
            raise ValueError("Context window must be positive")
        if not self.supported_tasks:
            raise ValueError("Model must support at least one task")
        if not self.languages:
            raise ValueError("Model must support at least one language")
        if len(self.temperature_range) != 2:
            raise ValueError("Temperature range must be a tuple of two values")
        if self.temperature_range[0] > self.temperature_range[1]:
            raise ValueError("Temperature range min must be <= max")

    def supports_task(self, task: str) -> bool:
        """Check if model supports a specific task."""
        return task in self.supported_tasks

    def supports_language(self, language: str) -> bool:
        """Check if model supports a specific language."""
        return language in self.languages

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "supported_tasks": self.supported_tasks,
            "context_window": self.context_window,
            "supports_function_calling": self.supports_function_calling,
            "supports_structured_output": self.supports_structured_output,
            "supports_streaming": self.supports_streaming,
            "languages": self.languages,
            "max_tokens": self.max_tokens,
            "temperature_range": self.temperature_range,
            "domains": self.domains,
            "vision_capable": self.vision_capable,
            "audio_capable": self.audio_capable,
            "code_specialized": self.code_specialized,
            "supports_tools": self.supports_tools,
            "supports_json_mode": self.supports_json_mode,
            "supports_multimodal": self.supports_multimodal,
            "accuracy_score": self.accuracy_score,
            "speed_rating": self.speed_rating,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelCapabilities:
        """Create from dictionary representation."""
        return cls(**data)


@dataclass
class ModelRequirements:
    """Resource requirements for a model."""

    memory_gb: float = 1.0
    gpu_memory_gb: Optional[float] = None
    cpu_cores: int = 1
    supports_quantization: List[str] = field(default_factory=list)
    min_python_version: str = "3.8"
    requires_gpu: bool = False
    disk_space_gb: float = 1.0

    def __post_init__(self) -> None:
        """Validate requirements after initialization."""
        if self.memory_gb <= 0:
            raise ValueError("Memory requirement must be positive")
        if self.gpu_memory_gb is not None and self.gpu_memory_gb <= 0:
            raise ValueError("GPU memory requirement must be positive")
        if self.cpu_cores <= 0:
            raise ValueError("CPU cores requirement must be positive")
        if self.disk_space_gb <= 0:
            raise ValueError("Disk space requirement must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "memory_gb": self.memory_gb,
            "gpu_memory_gb": self.gpu_memory_gb,
            "cpu_cores": self.cpu_cores,
            "supports_quantization": self.supports_quantization,
            "min_python_version": self.min_python_version,
            "requires_gpu": self.requires_gpu,
            "disk_space_gb": self.disk_space_gb,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelRequirements:
        """Create from dictionary representation."""
        return cls(**data)


@dataclass
class ModelCost:
    """Cost information for a model."""

    input_cost_per_1k_tokens: float = 0.0  # Cost per 1000 input tokens in USD
    output_cost_per_1k_tokens: float = 0.0  # Cost per 1000 output tokens in USD
    base_cost_per_request: float = 0.0  # Fixed cost per request in USD
    is_free: bool = False  # True for local/self-hosted models

    def __post_init__(self) -> None:
        """Validate cost information."""
        if self.input_cost_per_1k_tokens < 0:
            raise ValueError("Input cost must be non-negative")
        if self.output_cost_per_1k_tokens < 0:
            raise ValueError("Output cost must be non-negative")
        if self.base_cost_per_request < 0:
            raise ValueError("Base cost must be non-negative")

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate total cost for a request."""
        if self.is_free:
            return 0.0

        input_cost = (input_tokens / 1000) * self.input_cost_per_1k_tokens
        output_cost = (output_tokens / 1000) * self.output_cost_per_1k_tokens
        return self.base_cost_per_request + input_cost + output_cost

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "input_cost_per_1k_tokens": self.input_cost_per_1k_tokens,
            "output_cost_per_1k_tokens": self.output_cost_per_1k_tokens,
            "base_cost_per_request": self.base_cost_per_request,
            "is_free": self.is_free,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelCost:
        """Create from dictionary representation."""
        return cls(**data)


@dataclass
class ModelMetrics:
    """Performance metrics for a model."""

    latency_p50: float = 0.0
    latency_p95: float = 0.0
    throughput: float = 0.0
    accuracy: float = 0.0
    cost_per_token: float = 0.0
    success_rate: float = 1.0

    def __post_init__(self) -> None:
        """Validate metrics after initialization."""
        if self.latency_p50 < 0:
            raise ValueError("Latency P50 must be non-negative")
        if self.latency_p95 < 0:
            raise ValueError("Latency P95 must be non-negative")
        if self.throughput < 0:
            raise ValueError("Throughput must be non-negative")
        if not 0 <= self.accuracy <= 1:
            raise ValueError("Accuracy must be between 0 and 1")
        if self.cost_per_token < 0:
            raise ValueError("Cost per token must be non-negative")
        if not 0 <= self.success_rate <= 1:
            raise ValueError("Success rate must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "latency_p50": self.latency_p50,
            "latency_p95": self.latency_p95,
            "throughput": self.throughput,
            "accuracy": self.accuracy,
            "cost_per_token": self.cost_per_token,
            "success_rate": self.success_rate,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelMetrics:
        """Create from dictionary representation."""
        return cls(**data)


class Model(ABC):
    """Abstract base class for all models."""

    def __init__(
        self,
        name: str,
        provider: str,
        capabilities: Optional[ModelCapabilities] = None,
        requirements: Optional[ModelRequirements] = None,
        metrics: Optional[ModelMetrics] = None,
        cost: Optional[ModelCost] = None,
    ) -> None:
        """
        Initialize model.

        Args:
            name: Model name
            provider: Provider name (e.g., "openai", "anthropic", "local")
            capabilities: Model capabilities
            requirements: Resource requirements
            metrics: Performance metrics
            cost: Cost information
        """
        if not name:
            raise ValueError("Model name cannot be empty")
        if not provider:
            raise ValueError("Provider name cannot be empty")

        self.name = name
        self.provider = provider
        self.capabilities = capabilities or ModelCapabilities()
        self.requirements = requirements or ModelRequirements()
        self.metrics = metrics or ModelMetrics()
        self.cost = cost or ModelCost()
        self._is_available = False

    @abstractmethod
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
            **kwargs: Additional model-specific parameters

        Returns:
            Generated text
        """
        pass

    @abstractmethod
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
            **kwargs: Additional model-specific parameters

        Returns:
            Structured output matching schema
        """
        pass
    
    async def generate_multimodal(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text from multimodal input (text, images, etc.).
        
        Default implementation converts to generate() call.
        Models with native multimodal support should override this.

        Args:
            messages: List of message dicts with role and content
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional model-specific parameters

        Returns:
            Generated text
        """
        # Default: extract text content and use generate
        text_parts = []
        for msg in messages:
            if isinstance(msg.get("content"), str):
                text_parts.append(msg["content"])
            elif isinstance(msg.get("content"), list):
                for block in msg["content"]:
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
        
        prompt = "\n".join(text_parts)
        return await self.generate(prompt, temperature, max_tokens, **kwargs)

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if model is available and healthy.

        Returns:
            True if healthy, False otherwise
        """
        pass

    @abstractmethod
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
        pass

    def can_handle_task(self, task: str) -> bool:
        """Check if model can handle a specific task."""
        return self.capabilities.supports_task(task)

    def can_handle_language(self, language: str) -> bool:
        """Check if model can handle a specific language."""
        return self.capabilities.supports_language(language)

    def meets_requirements(self, requirements: Dict[str, Any]) -> bool:
        """
        Check if model meets specified requirements.

        Args:
            requirements: Dict of requirements to check

        Returns:
            True if all requirements are met
        """
        # Check context window
        if "context_window" in requirements:
            if self.capabilities.context_window < requirements["context_window"]:
                return False

        # Check function calling
        if requirements.get("supports_function_calling", False):
            if not self.capabilities.supports_function_calling:
                return False

        # Check structured output
        if requirements.get("supports_structured_output", False):
            if not self.capabilities.supports_structured_output:
                return False

        # Check supported tasks
        if "tasks" in requirements:
            required_tasks = requirements["tasks"]
            if not all(self.can_handle_task(task) for task in required_tasks):
                return False

        # Check supported languages
        if "languages" in requirements:
            required_languages = requirements["languages"]
            if not all(self.can_handle_language(lang) for lang in required_languages):
                return False

        return True

    @property
    def is_available(self) -> bool:
        """Check if model is available."""
        return self._is_available

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary representation."""
        return {
            "name": self.name,
            "provider": self.provider,
            "capabilities": self.capabilities.to_dict(),
            "requirements": self.requirements.to_dict(),
            "metrics": self.metrics.to_dict(),
            "cost": self.cost.to_dict(),
            "is_available": self.is_available,
        }

    def __repr__(self) -> str:
        """String representation of model."""
        return f"Model(name='{self.name}', provider='{self.provider}')"

    def __eq__(self, other: object) -> bool:
        """Check equality based on name and provider."""
        if not isinstance(other, Model):
            return NotImplemented
        return self.name == other.name and self.provider == other.provider

    def __hash__(self) -> int:
        """Hash based on name and provider."""
        return hash((self.name, self.provider))
