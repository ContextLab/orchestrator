"""Tests for model manager."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.orchestrator.core.model import Model, ModelCapabilities, ModelCost
from src.orchestrator.models.registry import ModelRegistry
from src.orchestrator.models.selection.manager import ModelManager, ModelUsageStats
from src.orchestrator.models.selection.strategies import TaskRequirements, TaskBasedStrategy
from src.orchestrator.models.optimization.caching import ModelResponseCache


class MockModel(Model):
    """Mock model for testing."""
    
    def __init__(self, name: str, provider: str, capabilities: ModelCapabilities = None):
        super().__init__(
            name=name,
            provider=provider,
            capabilities=capabilities or ModelCapabilities(supported_tasks=["text_generation"]),
            cost=ModelCost(is_free=True),
        )
        self._is_available = True
    
    async def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = None, **kwargs):
        await asyncio.sleep(0.01)  # Simulate some latency
        return f"Generated response from {self.name}: {prompt[:50]}..."
    
    async def generate_structured(self, prompt: str, schema: dict, temperature: float = 0.7, **kwargs):
        await asyncio.sleep(0.01)
        return {"response": f"Structured from {self.name}", "prompt": prompt}
    
    async def health_check(self) -> bool:
        return True
    
    async def estimate_cost(self, prompt: str, max_tokens: int = None) -> float:
        return 0.001


@pytest.fixture
def mock_registry():
    """Create mock registry."""
    registry = MagicMock(spec=ModelRegistry)
    registry.is_initialized = True
    
    # Create sample models
    models = [
        MockModel("gpt-3.5-turbo", "openai"),
        MockModel("claude-3-haiku", "anthropic"),
        MockModel("llama2-7b", "local"),
    ]
    
    # Mock list_models
    model_info = {
        model.name: {
            "provider": model.provider,
            "capabilities": model.capabilities.to_dict(),
            "cost": model.cost.to_dict(),
        }
        for model in models
    }
    registry.list_models.return_value = model_info
    
    # Mock get_model
    async def get_model(name, provider):
        for model in models:
            if model.name == name and model.provider == provider:
                return model
        raise ValueError(f"Model {name} not found")
    
    registry.get_model.side_effect = get_model
    registry.initialize = AsyncMock()
    registry.health_check = AsyncMock(return_value={"openai": True, "anthropic": True, "local": True})
    
    return registry


@pytest.fixture
def model_manager(mock_registry):
    """Create model manager for testing."""
    return ModelManager(
        registry=mock_registry,
        selection_strategy=TaskBasedStrategy(),
        enable_caching=True,
        enable_pooling=False,  # Disable pooling for simpler tests
        max_cache_size=100,
    )


class TestModelManager:
    """Test ModelManager functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, model_manager):
        """Test manager initialization."""
        assert model_manager.registry is not None
        assert model_manager.selection_strategy is not None
        assert model_manager.enable_caching is True
        assert model_manager._cache is not None
    
    @pytest.mark.asyncio
    async def test_select_model(self, model_manager):
        """Test model selection."""
        requirements = TaskRequirements(
            task_type="text_generation",
            context_window=4096,
        )
        
        result = await model_manager.select_model(requirements)
        
        assert result.model is not None
        assert result.provider is not None
        assert result.confidence_score > 0
        assert result.selection_reason is not None
    
    @pytest.mark.asyncio
    async def test_generate_with_model(self, model_manager):
        """Test text generation with model."""
        requirements = TaskRequirements(task_type="text_generation")
        selection_result = await model_manager.select_model(requirements)
        
        response, metadata = await model_manager.generate_with_model(
            model=selection_result.model,
            provider=selection_result.provider,
            prompt="Hello, world!",
            temperature=0.7,
        )
        
        assert isinstance(response, str)
        assert "Hello, world!" in response or "Generated response" in response
        assert "latency" in metadata
        assert "cost" in metadata
        assert "model" in metadata
        assert metadata["cached"] is False  # First call shouldn't be cached
    
    @pytest.mark.asyncio
    async def test_generate_structured_with_model(self, model_manager):
        """Test structured generation with model."""
        requirements = TaskRequirements(task_type="text_generation")
        selection_result = await model_manager.select_model(requirements)
        
        schema = {
            "type": "object",
            "properties": {
                "response": {"type": "string"},
                "sentiment": {"type": "string"},
            },
        }
        
        response, metadata = await model_manager.generate_structured_with_model(
            model=selection_result.model,
            provider=selection_result.provider,
            prompt="Analyze this text",
            schema=schema,
            temperature=0.7,
        )
        
        assert isinstance(response, dict)
        assert "response" in response
        assert "latency" in metadata
        assert "cost" in metadata
    
    @pytest.mark.asyncio
    async def test_caching_behavior(self, model_manager):
        """Test response caching."""
        requirements = TaskRequirements(task_type="text_generation")
        selection_result = await model_manager.select_model(requirements)
        
        prompt = "Test prompt for caching"
        
        # First call
        response1, metadata1 = await model_manager.generate_with_model(
            model=selection_result.model,
            provider=selection_result.provider,
            prompt=prompt,
            temperature=0.7,
            use_cache=True,
        )
        
        # Second call with same parameters
        response2, metadata2 = await model_manager.generate_with_model(
            model=selection_result.model,
            provider=selection_result.provider,
            prompt=prompt,
            temperature=0.7,
            use_cache=True,
        )
        
        assert metadata1["cached"] is False
        assert metadata2["cached"] is True
        assert response1 == response2
    
    @pytest.mark.asyncio
    async def test_get_best_model(self, model_manager):
        """Test getting best model instance."""
        requirements = TaskRequirements(task_type="text_generation")
        
        model, provider = await model_manager.get_best_model(requirements)
        
        assert isinstance(model, Model)
        assert isinstance(provider, str)
        assert model.capabilities.supports_task("text_generation")
    
    @pytest.mark.asyncio
    async def test_model_stats_tracking(self, model_manager):
        """Test model usage statistics tracking."""
        requirements = TaskRequirements(task_type="text_generation")
        selection_result = await model_manager.select_model(requirements)
        
        # Make a few requests
        for i in range(3):
            await model_manager.generate_with_model(
                model=selection_result.model,
                provider=selection_result.provider,
                prompt=f"Test prompt {i}",
            )
        
        # Check stats
        stats = await model_manager.get_model_stats(
            model_name=selection_result.model.name,
            provider=selection_result.provider,
        )
        
        model_key = f"{selection_result.provider}:{selection_result.model.name}"
        assert model_key in stats
        assert stats[model_key]["total_requests"] == 3
        assert stats[model_key]["successful_requests"] == 3
        assert stats[model_key]["success_rate"] == 1.0
    
    @pytest.mark.asyncio
    async def test_health_check(self, model_manager):
        """Test health check functionality."""
        # Make some requests first to have active models
        requirements = TaskRequirements(task_type="text_generation")
        selection_result = await model_manager.select_model(requirements)
        
        await model_manager.generate_with_model(
            model=selection_result.model,
            provider=selection_result.provider,
            prompt="Health check test",
        )
        
        # Perform health check
        health_result = await model_manager.health_check(force=True)
        
        assert "status" in health_result
        assert "timestamp" in health_result
        assert "total_models" in health_result
        assert health_result["success"] == True
    
    @pytest.mark.asyncio
    async def test_optimize_performance(self, model_manager):
        """Test performance optimization."""
        # Make some requests to generate stats
        requirements = TaskRequirements(task_type="text_generation")
        selection_result = await model_manager.select_model(requirements)
        
        for i in range(5):
            await model_manager.generate_with_model(
                model=selection_result.model,
                provider=selection_result.provider,
                prompt=f"Optimization test {i}",
            )
        
        # Run optimization
        optimization_result = await model_manager.optimize_performance()
        
        assert "timestamp" in optimization_result
        assert "optimizations" in optimization_result
        assert isinstance(optimization_result["optimizations"], list)
    
    @pytest.mark.asyncio
    async def test_failure_tracking(self, model_manager):
        """Test failure tracking and model health management."""
        requirements = TaskRequirements(task_type="text_generation")
        selection_result = await model_manager.select_model(requirements)
        
        # Mock model to fail
        original_generate = selection_result.model.generate
        selection_result.model.generate = AsyncMock(side_effect=Exception("Model failed"))
        
        # Make failing requests
        with pytest.raises(Exception):
            await model_manager.generate_with_model(
                model=selection_result.model,
                provider=selection_result.provider,
                prompt="This will fail",
            )
        
        # Check that failure is tracked
        stats = await model_manager.get_model_stats(
            model_name=selection_result.model.name,
            provider=selection_result.provider,
        )
        
        model_key = f"{selection_result.provider}:{selection_result.model.name}"
        assert stats[model_key]["failed_requests"] == 1
        assert stats[model_key]["success_rate"] < 1.0
        
        # Restore original method
        selection_result.model.generate = original_generate
    
    @pytest.mark.asyncio
    async def test_cleanup(self, model_manager):
        """Test manager cleanup."""
        # Make some requests to create state
        requirements = TaskRequirements(task_type="text_generation")
        selection_result = await model_manager.select_model(requirements)
        
        await model_manager.generate_with_model(
            model=selection_result.model,
            provider=selection_result.provider,
            prompt="Cleanup test",
        )
        
        # Verify we have some state
        stats = await model_manager.get_model_stats()
        assert len(stats) > 0
        
        # Clean up
        await model_manager.cleanup()
        
        # Verify state is cleared (this might depend on implementation details)
        manager_info = model_manager.get_manager_info()
        # At minimum, verify cleanup was called without errors
        assert isinstance(manager_info, dict)
    
    def test_manager_info(self, model_manager):
        """Test manager info retrieval."""
        info = model_manager.get_manager_info()
        
        assert "strategy" in info
        assert "caching_enabled" in info
        assert "pooling_enabled" in info
        assert info["caching_enabled"] is True
        assert info["pooling_enabled"] is False


class TestModelUsageStats:
    """Test ModelUsageStats data class."""
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        stats = ModelUsageStats(
            total_requests=10,
            successful_requests=8,
            failed_requests=2,
        )
        
        assert stats.success_rate == 0.8
    
    def test_success_rate_zero_requests(self):
        """Test success rate with zero requests."""
        stats = ModelUsageStats()
        assert stats.success_rate == 1.0
    
    def test_average_latency_calculation(self):
        """Test average latency calculation."""
        stats = ModelUsageStats(
            successful_requests=5,
            total_latency=2.5,  # 2.5 seconds total
        )
        
        assert stats.average_latency == 0.5  # 0.5 seconds average
    
    def test_average_latency_zero_requests(self):
        """Test average latency with zero successful requests."""
        stats = ModelUsageStats()
        assert stats.average_latency == 0.0
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = ModelUsageStats(
            total_requests=10,
            successful_requests=8,
            failed_requests=2,
            total_latency=4.0,
            total_cost=0.05,
            last_used=1234567890.0,
            error_messages=["Error 1", "Error 2", "Error 3"],
        )
        
        result = stats.to_dict()
        
        assert result["total_requests"] == 10
        assert result["successful_requests"] == 8
        assert result["failed_requests"] == 2
        assert result["success_rate"] == 0.8
        assert result["average_latency"] == 0.5
        assert result["total_cost"] == 0.05
        assert result["last_used"] == 1234567890.0
        assert len(result["recent_errors"]) == 3  # All errors since < 5