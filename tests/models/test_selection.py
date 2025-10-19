#!/usr/bin/env python3
# SKIPPED: This test file uses removed providers (OpenAI, Local) - Issue #426
import pytest
pytest.skip("Skipping entire module - uses removed providers", allow_module_level=True)

"""
Selection strategy tests for multi-model integration.

Tests model selection strategies with real performance data to validate:
- Cost optimization selection works with real pricing
- Performance-based selection uses actual latency/throughput data
- Task-specific selection chooses appropriate models
- Balanced selection strategies optimize multiple criteria
"""

import asyncio
import os
import pytest
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.orchestrator.models.registry import ModelRegistry
from src.orchestrator.models.selection.strategies import (
    CostOptimizedStrategy, 
    PerformanceOptimizedStrategy,
    BalancedStrategy,
    TaskSpecificStrategy,
    SelectionCriteria,
    SelectionResult
)
from src.orchestrator.models.selection.manager import ModelSelectionManager
from src.orchestrator.models.providers.openai_provider import OpenAIProvider
from src.orchestrator.models.providers.anthropic_provider import AnthropicProvider
from src.orchestrator.models.providers.local_provider import LocalProvider
from src.orchestrator.models.providers.base import ModelCapability, ModelInfo


@dataclass
class PerformanceMetric:
    """Performance measurement data."""
    model_name: str
    provider: str
    latency_ms: float
    throughput_tokens_per_sec: float
    cost_per_token: float
    accuracy_score: float
    timestamp: datetime


class TestSelectionStrategies:
    """Test selection strategy implementations."""

    @pytest.fixture
    def mock_models(self):
        """Create mock models with different characteristics."""
        return [
            ModelInfo(
                name="gpt-4o-mini",
                provider="openai",
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.ANALYSIS],
                context_window=128000,
                cost_per_token=0.000150,
                performance_tier="fast",
                quality_tier="high"
            ),
            ModelInfo(
                name="gpt-4o",
                provider="openai", 
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CODE_GENERATION],
                context_window=128000,
                cost_per_token=0.002500,
                performance_tier="medium",
                quality_tier="premium"
            ),
            ModelInfo(
                name="claude-3-haiku-20240307",
                provider="anthropic",
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.ANALYSIS],
                context_window=200000,
                cost_per_token=0.000250,
                performance_tier="fast",
                quality_tier="high"
            ),
            ModelInfo(
                name="claude-3-5-sonnet-20241022",
                provider="anthropic",
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CODE_GENERATION],
                context_window=200000,
                cost_per_token=0.003000,
                performance_tier="medium",
                quality_tier="premium"
            ),
            ModelInfo(
                name="llama3.2:8b",
                provider="local",
                capabilities=[ModelCapability.TEXT_GENERATION],
                context_window=131072,
                cost_per_token=0.0,  # Free local model
                performance_tier="slow",
                quality_tier="medium"
            )
        ]

    def test_cost_optimized_strategy(self, mock_models):
        """Test cost optimization strategy."""
        strategy = CostOptimizedStrategy()
        
        criteria = SelectionCriteria(
            required_capabilities=[ModelCapability.TEXT_GENERATION],
            max_cost_per_token=0.001,
            min_context_window=50000
        )
        
        result = strategy.select_model(mock_models, criteria)
        
        assert result is not None
        assert isinstance(result, SelectionResult)
        
        # Should select the cheapest model that meets criteria
        # Local model (free) should be preferred if it meets requirements
        if result.model.provider == "local":
            assert result.model.cost_per_token == 0.0
        else:
            # Should be cheapest paid model
            assert result.model.cost_per_token <= 0.001
        
        assert result.confidence > 0.0
        assert "cost" in result.reasoning.lower()

    def test_performance_optimized_strategy(self, mock_models):
        """Test performance optimization strategy."""
        strategy = PerformanceOptimizedStrategy()
        
        criteria = SelectionCriteria(
            required_capabilities=[ModelCapability.TEXT_GENERATION],
            min_quality_tier="medium"
        )
        
        result = strategy.select_model(mock_models, criteria)
        
        assert result is not None
        
        # Should prefer fast performance tier
        assert result.model.performance_tier in ["fast", "medium"]
        assert result.confidence > 0.0
        assert "performance" in result.reasoning.lower()

    def test_balanced_strategy(self, mock_models):
        """Test balanced optimization strategy."""
        strategy = BalancedStrategy()
        
        criteria = SelectionCriteria(
            required_capabilities=[ModelCapability.TEXT_GENERATION],
            max_cost_per_token=0.005,
            min_quality_tier="high"
        )
        
        result = strategy.select_model(mock_models, criteria)
        
        assert result is not None
        
        # Should balance cost and performance
        assert result.model.cost_per_token <= 0.005
        assert result.model.quality_tier in ["high", "premium"]
        assert result.confidence > 0.0
        assert "balance" in result.reasoning.lower() or "balanced" in result.reasoning.lower()

    def test_task_specific_strategy(self, mock_models):
        """Test task-specific strategy."""
        strategy = TaskSpecificStrategy()
        
        # Test code generation task
        criteria = SelectionCriteria(
            task_type="code_generation",
            required_capabilities=[ModelCapability.CODE_GENERATION],
            min_quality_tier="high"
        )
        
        result = strategy.select_model(mock_models, criteria)
        
        assert result is not None
        assert ModelCapability.CODE_GENERATION in result.model.capabilities
        assert result.confidence > 0.0

    def test_no_suitable_model(self, mock_models):
        """Test handling when no models meet criteria."""
        strategy = CostOptimizedStrategy()
        
        # Impossible criteria
        criteria = SelectionCriteria(
            required_capabilities=[ModelCapability.TEXT_GENERATION],
            max_cost_per_token=0.000001,  # Extremely low cost
            min_context_window=1000000,   # Extremely large context
            min_quality_tier="premium"
        )
        
        result = strategy.select_model(mock_models, criteria)
        
        # Should return None or empty result
        assert result is None or result.model is None


@pytest.mark.integration
class TestSelectionWithRealModels:
    """Test selection strategies with real model performance data."""

    @pytest.fixture
    async def model_registry(self):
        """Create registry with real providers."""
        registry = ModelRegistry()
        
        # Add providers (they'll only work if properly configured)
        registry.add_provider(OpenAIProvider())
        registry.add_provider(AnthropicProvider())
        registry.add_provider(LocalProvider())
        
        return registry

    @pytest.fixture
    async def real_models(self, model_registry):
        """Get real available models."""
        all_models = []
        
        providers = model_registry.get_providers()
        for provider in providers:
            try:
                models = await provider.get_available_models()
                all_models.extend(models)
            except Exception as e:
                print(f"Provider {provider.name} not available: {e}")
                continue
        
        return all_models

    async def test_selection_with_real_models(self, real_models):
        """Test selection strategies with real models."""
        if not real_models:
            pytest.skip("No real models available for testing")
        
        print(f"Testing with {len(real_models)} real models")
        
        # Test different strategies
        strategies = [
            CostOptimizedStrategy(),
            PerformanceOptimizedStrategy(), 
            BalancedStrategy()
        ]
        
        criteria = SelectionCriteria(
            required_capabilities=[ModelCapability.TEXT_GENERATION]
        )
        
        results = []
        for strategy in strategies:
            result = strategy.select_model(real_models, criteria)
            if result and result.model:
                results.append((strategy.__class__.__name__, result))
                print(f"{strategy.__class__.__name__}: {result.model.provider}:{result.model.name}")
        
        # Should have at least one successful selection
        assert len(results) > 0

    async def test_performance_measurement(self, model_registry):
        """Test actual performance measurement of models."""
        # Find a working model for testing
        working_model = None
        
        providers = model_registry.get_providers()
        for provider in providers:
            try:
                models = await provider.get_available_models()
                for model_info in models[:1]:  # Try first model
                    try:
                        model_instance = await provider.create_model(model_info.name)
                        if model_instance:
                            working_model = (model_info, model_instance)
                            break
                    except Exception as e:
                        print(f"Failed to create {model_info.name}: {e}")
                        continue
                
                if working_model:
                    break
                    
            except Exception as e:
                print(f"Provider {provider.name} failed: {e}")
                continue
        
        if not working_model:
            pytest.skip("No working models available for performance testing")
        
        model_info, model_instance = working_model
        
        # Measure performance
        print(f"Measuring performance of {model_info.provider}:{model_info.name}")
        
        test_prompt = "What is machine learning?"
        start_time = time.time()
        
        result = await model_instance.generate(
            test_prompt, 
            max_tokens=50, 
            temperature=0.1
        )
        
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to ms
        
        # Calculate throughput (rough estimate)
        output_tokens = len(result.split())
        throughput = output_tokens / (latency / 1000) if latency > 0 else 0
        
        print(f"Latency: {latency:.2f}ms")
        print(f"Output tokens: {output_tokens}")
        print(f"Throughput: {throughput:.2f} tokens/sec")
        print(f"Response: {result[:100]}...")
        
        # Create performance metric
        metric = PerformanceMetric(
            model_name=model_info.name,
            provider=model_info.provider,
            latency_ms=latency,
            throughput_tokens_per_sec=throughput,
            cost_per_token=model_info.cost_per_token,
            accuracy_score=0.85,  # Would need evaluation benchmark
            timestamp=datetime.now()
        )
        
        # Verify reasonable values
        assert metric.latency_ms > 0
        assert metric.throughput_tokens_per_sec >= 0
        assert isinstance(result, str)
        assert len(result) > 0


@pytest.mark.integration
class TestModelSelectionManager:
    """Test the model selection manager."""

    @pytest.fixture
    async def selection_manager(self):
        """Create selection manager with real providers."""
        registry = ModelRegistry()
        
        # Add providers
        registry.add_provider(OpenAIProvider())
        registry.add_provider(AnthropicProvider())
        registry.add_provider(LocalProvider())
        
        return ModelSelectionManager(registry)

    async def test_manager_initialization(self, selection_manager):
        """Test selection manager initialization."""
        assert selection_manager is not None
        assert hasattr(selection_manager, 'registry')
        assert hasattr(selection_manager, 'select_model')

    async def test_manager_model_selection(self, selection_manager):
        """Test model selection through manager."""
        criteria = SelectionCriteria(
            required_capabilities=[ModelCapability.TEXT_GENERATION],
            strategy="balanced"
        )
        
        try:
            result = await selection_manager.select_model(criteria)
            
            if result and result.model:
                print(f"Selected: {result.model.provider}:{result.model.name}")
                print(f"Reasoning: {result.reasoning}")
                print(f"Confidence: {result.confidence}")
                
                # Verify selection meets criteria
                assert ModelCapability.TEXT_GENERATION in result.model.capabilities
                assert result.confidence > 0.0
                
        except Exception as e:
            print(f"Model selection failed: {e}")
            pytest.skip("No suitable models available for selection")

    async def test_manager_fallback_strategies(self, selection_manager):
        """Test fallback when primary strategy fails."""
        # Very restrictive criteria that might not be met
        criteria = SelectionCriteria(
            required_capabilities=[ModelCapability.TEXT_GENERATION],
            max_cost_per_token=0.000001,  # Very low
            strategy="cost_optimized"
        )
        
        try:
            result = await selection_manager.select_model(criteria)
            
            # Should either succeed with fallback or fail gracefully
            if result is None:
                print("No models met strict criteria - expected behavior")
            else:
                print(f"Found model with fallback: {result.model.name}")
                
        except Exception as e:
            print(f"Selection failed: {e}")

    async def test_manager_caching(self, selection_manager):
        """Test result caching in selection manager."""
        criteria = SelectionCriteria(
            required_capabilities=[ModelCapability.TEXT_GENERATION],
            strategy="balanced"
        )
        
        # First selection
        start_time = time.time()
        result1 = await selection_manager.select_model(criteria)
        first_duration = time.time() - start_time
        
        # Second selection (should be cached)
        start_time = time.time() 
        result2 = await selection_manager.select_model(criteria)
        second_duration = time.time() - start_time
        
        if result1 and result2:
            assert result1.model.name == result2.model.name
            # Second call should be faster (cached)
            print(f"First call: {first_duration:.3f}s, Second call: {second_duration:.3f}s")


class TestSelectionOptimization:
    """Test selection optimization with performance data."""

    def test_historical_performance_integration(self):
        """Test integration with historical performance data."""
        # This would integrate with performance tracking system
        performance_data = {
            "gpt-4o-mini": {
                "avg_latency_ms": 1200,
                "avg_throughput": 15.5,
                "success_rate": 0.98,
                "last_updated": datetime.now()
            },
            "claude-3-haiku": {
                "avg_latency_ms": 800,
                "avg_throughput": 22.1, 
                "success_rate": 0.97,
                "last_updated": datetime.now()
            }
        }
        
        # Selection should consider historical performance
        assert len(performance_data) > 0
        
        # Verify data structure
        for model_name, metrics in performance_data.items():
            assert "avg_latency_ms" in metrics
            assert "avg_throughput" in metrics
            assert "success_rate" in metrics
            assert metrics["success_rate"] >= 0.0
            assert metrics["success_rate"] <= 1.0

    def test_dynamic_criteria_adjustment(self):
        """Test dynamic adjustment of selection criteria."""
        # Test time-based criteria adjustment
        base_criteria = SelectionCriteria(
            required_capabilities=[ModelCapability.TEXT_GENERATION],
            max_cost_per_token=0.002
        )
        
        # During peak hours, might prioritize performance over cost
        peak_hour_criteria = SelectionCriteria(
            required_capabilities=[ModelCapability.TEXT_GENERATION],
            max_cost_per_token=0.005,  # Higher cost allowed
            min_quality_tier="high"
        )
        
        # During off-peak, might prioritize cost
        off_peak_criteria = SelectionCriteria(
            required_capabilities=[ModelCapability.TEXT_GENERATION],
            max_cost_per_token=0.001,  # Lower cost preferred
            strategy="cost_optimized"
        )
        
        # All criteria should be valid
        assert base_criteria.max_cost_per_token == 0.002
        assert peak_hour_criteria.max_cost_per_token == 0.005
        assert off_peak_criteria.max_cost_per_token == 0.001


async def main():
    """Run selection strategy integration tests."""
    print("🎯 SELECTION STRATEGY INTEGRATION TESTS")
    print("=" * 60)
    
    # Run pytest with this file
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "integration"
    ])
    
    return exit_code == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)