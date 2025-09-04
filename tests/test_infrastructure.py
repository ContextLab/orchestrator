"""Test infrastructure and utilities for orchestrator testing."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from src.orchestrator.core.model import Model, ModelCapabilities, ModelRequirements, ModelMetrics, ModelCost


class MockTestModel(Model):
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


class MockTestProvider:
    """Mock provider for testing."""
    
    def __init__(self, name: str = "test-provider"):
        self.name = name
        self.is_initialized = True
        # Include common model names that tests might expect
        test_model = MockTestModel()
        self._models = {
            "test-model": test_model,
            "openai/gpt-3.5-turbo": test_model,
            "openai/gpt-4": test_model,
            "anthropic/claude-3": test_model,
        }
    
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
    
    async def get_model(self, model_name: str, **kwargs) -> MockTestModel:
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
    test_provider = MockTestProvider()
    registry.register_provider(test_provider)
    
    # Create control system with populated registry  
    control_system = HybridControlSystem(model_registry=registry)
    
    # Create orchestrator with all required components
    orchestrator = Orchestrator(
        model_registry=registry,
        control_system=control_system
    )
    
    return orchestrator


# Actual test cases for the infrastructure components
import pytest


def test_mock_test_model_creation():
    """Test that MockTestModel can be created with defaults."""
    model = MockTestModel()
    assert model.name == "test-model"
    assert model.provider == "test-provider"
    assert model.capabilities.supports_function_calling is True
    assert model.capabilities.supports_structured_output is True
    assert model.cost.is_free is True


@pytest.mark.asyncio
async def test_mock_test_model_generate():
    """Test that MockTestModel can generate text responses."""
    model = MockTestModel()
    response = await model.generate("test prompt")
    assert "Test response for: test prompt" in response


@pytest.mark.asyncio
async def test_mock_test_model_generate_structured():
    """Test that MockTestModel can generate structured responses."""
    model = MockTestModel()
    schema = {"type": "object", "properties": {"result": {"type": "string"}}}
    response = await model.generate_structured("test prompt", schema)
    assert isinstance(response, dict)
    assert "test_output" in response


@pytest.mark.asyncio
async def test_mock_test_model_health_check():
    """Test that MockTestModel health check always passes."""
    model = MockTestModel()
    health = await model.health_check()
    assert health is True


@pytest.mark.asyncio  
async def test_mock_test_model_estimate_cost():
    """Test that MockTestModel cost estimation is always free."""
    model = MockTestModel()
    cost = await model.estimate_cost(100, 50)
    assert cost == 0.0


def test_mock_test_provider_creation():
    """Test that MockTestProvider can be created."""
    provider = MockTestProvider()
    assert provider.name == "test-provider"
    assert provider.is_initialized is True


def test_mock_test_provider_available_models():
    """Test that MockTestProvider returns expected model list.""" 
    provider = MockTestProvider()
    models = provider.available_models
    expected_models = [
        "test-model",
        "openai/gpt-3.5-turbo", 
        "openai/gpt-4",
        "anthropic/claude-3"
    ]
    assert all(model in models for model in expected_models)


def test_mock_test_provider_supports_model():
    """Test that MockTestProvider supports expected models."""
    provider = MockTestProvider()
    assert provider.supports_model("test-model") is True
    assert provider.supports_model("openai/gpt-4") is True
    assert provider.supports_model("unknown-model") is False


def test_mock_test_provider_capabilities():
    """Test that MockTestProvider returns correct model capabilities."""
    provider = MockTestProvider()
    capabilities = provider.get_model_capabilities("test-model")
    assert capabilities.supports_function_calling is True
    assert capabilities.supports_structured_output is True
    assert capabilities.context_window == 8192


def test_mock_test_provider_requirements():
    """Test that MockTestProvider returns correct model requirements."""
    provider = MockTestProvider()
    requirements = provider.get_model_requirements("test-model")
    assert requirements.memory_gb == 0.1
    assert requirements.cpu_cores == 1


def test_mock_test_provider_cost():
    """Test that MockTestProvider returns correct cost info."""
    provider = MockTestProvider()
    cost = provider.get_model_cost("test-model")
    assert cost.is_free is True


def test_mock_test_provider_info():
    """Test that MockTestProvider returns correct provider info."""
    provider = MockTestProvider()
    info = provider.get_provider_info()
    assert info["name"] == "test-provider"
    assert info["type"] == "test"
    assert info["initialized"] is True
    assert info["models"] == 4


@pytest.mark.asyncio
async def test_mock_test_provider_get_model():
    """Test that MockTestProvider can return model instances."""
    provider = MockTestProvider()
    model = await provider.get_model("test-model")
    assert isinstance(model, MockTestModel)


@pytest.mark.asyncio
async def test_mock_test_provider_initialize():
    """Test that MockTestProvider initialization works."""
    provider = MockTestProvider()
    await provider.initialize()  # Should not raise any exceptions


def test_create_test_orchestrator():
    """Test that test orchestrator can be created successfully."""
    orchestrator = create_test_orchestrator()
    assert orchestrator is not None
    assert orchestrator.model_registry is not None
    assert orchestrator.control_system is not None


# Test categorization and systematic analysis utilities
import os
import subprocess
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class TestFailureInfo:
    """Information about a test failure."""
    test_file: str
    test_name: str
    failure_type: str
    error_message: str
    category: str = "unknown"


class TestCategorizer:
    """Systematic test failure categorizer."""
    
    CATEGORY_PATTERNS = {
        "infrastructure": [
            "model.*not.*found",
            "provider.*not.*available", 
            "registry.*error",
            "ModelRegistry",
            "TestModel",
            "MockTestModel",
            "create_test_orchestrator"
        ],
        "api_compatibility": [
            "unexpected keyword argument",
            "takes.*positional argument",
            "missing.*required.*argument",
            "TypeError.*argument",
            "AttributeError.*object.*has.*no.*attribute"
        ],
        "data_structure": [
            "KeyError",
            "IndexError", 
            "list index out of range",
            "dictionary.*key",
            "NoneType.*object",
            "result.*access"
        ],
        "business_logic": [
            "validation.*failed",
            "assertion.*error",
            "expected.*got", 
            "AssertionError",
            "ValueError.*invalid"
        ],
        "dependencies": [
            "ImportError",
            "ModuleNotFoundError",
            "No module named",
            "package.*not.*found",
            "install.*required"
        ],
        "environment": [
            "PermissionError",
            "FileNotFoundError",
            "ConnectionError",
            "timeout",
            "network.*error"
        ]
    }
    
    def categorize_failure(self, failure_info: TestFailureInfo) -> str:
        """Categorize a test failure based on error patterns."""
        error_text = f"{failure_info.failure_type} {failure_info.error_message}".lower()
        
        for category, patterns in self.CATEGORY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern.lower(), error_text):
                    return category
                    
        return "unknown"
    
    def run_test_categorization(self, test_pattern: str = "tests/", max_tests: int = 50) -> Dict[str, List[TestFailureInfo]]:
        """Run systematic test categorization on a subset of tests."""
        failures_by_category = {}
        
        try:
            # Get list of test files
            result = subprocess.run(
                ["find", test_pattern, "-name", "test_*.py"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            test_files = result.stdout.strip().split('\n')[:max_tests]
            
            for test_file in test_files:
                if not test_file or not os.path.exists(test_file):
                    continue
                    
                # Run just collection to avoid timeouts
                try:
                    result = subprocess.run(
                        ["python", "-m", "pytest", test_file, "--collect-only", "-q"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if result.returncode != 0:
                        failure_info = TestFailureInfo(
                            test_file=test_file,
                            test_name="collection",
                            failure_type="collection_error",
                            error_message=result.stderr[:200]
                        )
                        
                        category = self.categorize_failure(failure_info)
                        failure_info.category = category
                        
                        if category not in failures_by_category:
                            failures_by_category[category] = []
                        failures_by_category[category].append(failure_info)
                        
                except subprocess.TimeoutExpired:
                    failure_info = TestFailureInfo(
                        test_file=test_file,
                        test_name="timeout",
                        failure_type="timeout",
                        error_message="Test collection timeout",
                        category="environment"
                    )
                    if "environment" not in failures_by_category:
                        failures_by_category["environment"] = []
                    failures_by_category["environment"].append(failure_info)
                        
        except Exception as e:
            print(f"Error during test categorization: {e}")
            
        return failures_by_category
    
    def generate_categorization_report(self, failures_by_category: Dict[str, List[TestFailureInfo]]) -> str:
        """Generate a systematic categorization report."""
        report = ["# Test Failure Categorization Report\n"]
        
        total_failures = sum(len(failures) for failures in failures_by_category.values())
        report.append(f"**Total analyzed failures**: {total_failures}\n")
        
        for category, failures in failures_by_category.items():
            report.append(f"## {category.title()} ({len(failures)} failures)\n")
            
            for failure in failures[:5]:  # Show first 5 examples
                report.append(f"- **{failure.test_file}**: {failure.error_message[:100]}...")
            
            if len(failures) > 5:
                report.append(f"- ... and {len(failures) - 5} more")
            
            report.append("")
        
        return "\n".join(report)


def test_categorizer_functionality():
    """Test that the test categorizer works correctly."""
    categorizer = TestCategorizer()
    
    # Test infrastructure categorization
    failure = TestFailureInfo(
        test_file="test_example.py",
        test_name="test_model",
        failure_type="AttributeError", 
        error_message="ModelRegistry object has no attribute 'get_model'"
    )
    assert categorizer.categorize_failure(failure) == "infrastructure"
    
    # Test API compatibility categorization
    failure = TestFailureInfo(
        test_file="test_example.py",
        test_name="test_api",
        failure_type="TypeError",
        error_message="unexpected keyword argument 'temperature'"
    )
    assert categorizer.categorize_failure(failure) == "api_compatibility"


def run_systematic_test_analysis():
    """Run systematic test analysis and return categorization."""
    categorizer = TestCategorizer()
    
    print("Running systematic test categorization...")
    failures_by_category = categorizer.run_test_categorization(max_tests=20)
    
    report = categorizer.generate_categorization_report(failures_by_category)
    print(report)
    
    return failures_by_category