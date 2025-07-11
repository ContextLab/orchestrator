"""Model abstractions for the orchestrator framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


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
    ) -> None:
        """
        Initialize model.
        
        Args:
            name: Model name
            provider: Provider name (e.g., "openai", "anthropic", "local")
            capabilities: Model capabilities
            requirements: Resource requirements
            metrics: Performance metrics
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


class MockModel(Model):
    """Mock model implementation for testing."""
    
    def __init__(
        self,
        name: str = "mock-model",
        provider: str = "mock",
        capabilities: Optional[ModelCapabilities] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize mock model."""
        if capabilities is None:
            capabilities = ModelCapabilities(
                supported_tasks=["generate", "analyze", "transform"],
                context_window=4096,
                supports_function_calling=True,
                supports_structured_output=True,
            )
        
        super().__init__(name, provider, capabilities, **kwargs)
        self._is_available = True
        self._responses = {}
    
    def set_response(self, prompt: str, response: Union[str, Dict[str, Any]]) -> None:
        """Set canned response for a prompt."""
        self._responses[prompt] = response
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Generate mock response."""
        # First try exact match
        if prompt in self._responses:
            response = self._responses[prompt]
            if isinstance(response, str):
                return response
            elif isinstance(response, Exception):
                raise response
            return str(response)
        
        # Then try partial match (useful for ambiguity resolution)
        for key, response in self._responses.items():
            if key in prompt:
                if isinstance(response, str):
                    return response
                elif isinstance(response, Exception):
                    raise response
                return str(response)
        
        return f"Mock response for: {prompt[:50]}..."
    
    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate mock structured response."""
        if prompt in self._responses:
            response = self._responses[prompt]
            if isinstance(response, dict):
                return response
        
        # Generate mock response based on schema
        return {"result": f"Mock structured response for: {prompt[:50]}..."}
    
    async def health_check(self) -> bool:
        """Mock health check."""
        return True
    
    async def estimate_cost(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
    ) -> float:
        """Mock cost estimation."""
        return 0.001  # $0.001 per request