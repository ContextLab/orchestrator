"""Test infrastructure and utilities for orchestrator testing."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from src.orchestrator.core.model import Model, ModelCapabilities, ModelRequirements, ModelMetrics, ModelCost


class TestModel(Model):
    """Mock model for testing that implements all required abstract methods."""
    
    def __init__(
        self, 
        name: str = "test-model",
        provider: str = "test-provider",
        capabilities: Optional[ModelCapabilities] = None,
        requirements: Optional[ModelRequirements] = None,
        metrics: Optional[ModelMetrics] = None,
        cost: Optional[ModelCost] = None
    ) -> None:
        """Initialize test model with minimal defaults."""
        if capabilities is None:
            capabilities = ModelCapabilities(
                supported_tasks=["text-generation", "analysis"],
                context_window=8192,
                supports_function_calling=True,
                supports_structured_output=True
            )
        
        if requirements is None:
            requirements = ModelRequirements(
                memory_gb=0.1,
                cpu_cores=1
            )
            
        if metrics is None:
            metrics = ModelMetrics()
            
        if cost is None:
            cost = ModelCost(is_free=True)
            
        super().__init__(name, provider, capabilities, requirements, metrics, cost)
    
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any
    ) -> str:
        """Generate text response for testing."""
        return f"Test response for: {prompt[:50]}..."
    
    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Generate structured response for testing."""
        return {"test_output": f"Structured response for: {prompt[:30]}..."}
    
    async def health_check(self) -> bool:
        """Always healthy for testing."""
        return True
    
    async def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Always free for testing."""
        return 0.0


class TestProvider:
    """Mock provider for testing."""
    
    def __init__(self, name: str = "test-provider"):
        self.name = name
        self.is_initialized = True
        self._models = {"test-model": TestModel()}
    
    @property 
    def available_models(self) -> List[str]:
        """List available models."""
        return list(self._models.keys())
    
    def supports_model(self, model_name: str) -> bool:
        """Check if provider supports model."""
        return model_name in self._models
    
    def get_model_capabilities(self, model_name: str):
        """Get capabilities for test model."""
        if not self.supports_model(model_name):
            raise ValueError(f"Model '{model_name}' not supported")
        return ModelCapabilities(
            supported_tasks=["text-generation", "analysis"],
            context_window=8192,
            supports_function_calling=True,
            supports_structured_output=True
        )
    
    def get_model_requirements(self, model_name: str):
        """Get requirements for test model."""
        if not self.supports_model(model_name):
            raise ValueError(f"Model '{model_name}' not supported")
        return ModelRequirements(
            memory_gb=0.1,
            cpu_cores=1
        )
    
    def get_model_cost(self, model_name: str):
        """Get cost for test model.""" 
        if not self.supports_model(model_name):
            raise ValueError(f"Model '{model_name}' not supported")
        return ModelCost(is_free=True)
    
    def get_provider_info(self):
        """Get provider info for test provider."""
        return {
            "name": self.name,
            "type": "test",
            "models": len(self._models),
            "initialized": self.is_initialized
        }
    
    async def get_model(self, model_name: str, **kwargs) -> TestModel:
        """Get model instance."""
        return self._models[model_name]
    
    async def initialize(self) -> None:
        """Initialize provider."""
        pass


def create_test_orchestrator():
    """Create orchestrator with test model for testing."""
    from src.orchestrator.orchestrator import Orchestrator
    from src.orchestrator.models.registry import ModelRegistry
    from src.orchestrator.control_systems.hybrid_control_system import HybridControlSystem
    
    # Create registry and add test provider
    registry = ModelRegistry()
    test_provider = TestProvider()
    registry.register_provider(test_provider)
    
    # Create control system with populated registry  
    control_system = HybridControlSystem(model_registry=registry)
    
    # Create orchestrator with all required components
    orchestrator = Orchestrator(
        model_registry=registry,
        control_system=control_system
    )
    
    return orchestrator