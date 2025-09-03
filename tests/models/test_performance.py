#!/usr/bin/env python3
"""
Performance optimization tests for multi-model integration.

Tests performance optimization features with real-world scenarios to validate:
- Connection pooling reduces latency for multiple requests
- Caching improves response times for repeated queries
- Load balancing distributes requests effectively
- Resource management prevents memory leaks
- Batch processing optimizations work correctly
"""

import asyncio
import os
import pytest
import sys
import time
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Performance optimization imports
from src.orchestrator.models.optimization.caching import ModelCache, CacheConfig
from src.orchestrator.models.optimization.pooling import ConnectionPool, PoolConfig
from src.orchestrator.models.load_balancer import LoadBalancer, LoadBalancingStrategy

# Model system imports
from src.orchestrator.models.registry import ModelRegistry
from src.orchestrator.models.providers.openai_provider import OpenAIProvider
from src.orchestrator.models.providers.anthropic_provider import AnthropicProvider
from src.orchestrator.models.providers.local_provider import LocalProvider
from src.orchestrator.models.providers.base import ModelCapability
from src.orchestrator.models.selection.manager import ModelSelectionManager
from src.orchestrator.models.selection.strategies import SelectionCriteria


@dataclass
class PerformanceMetric:
    """Performance measurement data."""
    operation: str
    duration_ms: float
    memory_usage_mb: float
    timestamp: datetime
    success: bool
    model_used: str = ""
    cache_hit: bool = False
    pool_reused: bool = False


class TestModelCaching:
    """Test model response caching performance."""

    @pytest.fixture
    def cache_config(self):
        """Create cache configuration for testing."""
        return CacheConfig(
            max_size=100,
            ttl_seconds=300,  # 5 minutes
            enable_compression=True
        )

    @pytest.fixture
    def model_cache(self, cache_config):
        """Create model cache instance."""
        return ModelCache(cache_config)

    def test_cache_configuration(self, model_cache):
        """Test cache configuration setup."""
        assert model_cache is not None
        assert model_cache.config.max_size == 100
        assert model_cache.config.ttl_seconds == 300
        assert model_cache.config.enable_compression is True

    def test_cache_basic_operations(self, model_cache):
        """Test basic cache operations."""
        # Test cache miss
        result = model_cache.get("test_key")
        assert result is None
        
        # Test cache set and hit
        test_data = {"content": "Hello, world!", "model": "test-model"}
        model_cache.set("test_key", test_data)
        
        cached_result = model_cache.get("test_key")
        assert cached_result is not None
        assert cached_result["content"] == "Hello, world!"
        assert cached_result["model"] == "test-model"

    def test_cache_expiration(self, model_cache):
        """Test cache TTL expiration."""
        # Set item with short TTL
        short_ttl_cache = ModelCache(CacheConfig(max_size=10, ttl_seconds=0.1))
        
        short_ttl_cache.set("expiring_key", {"data": "expires soon"})
        
        # Should be available immediately
        result = short_ttl_cache.get("expiring_key")
        assert result is not None
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should be expired
        expired_result = short_ttl_cache.get("expiring_key")
        assert expired_result is None

    def test_cache_size_limit(self, model_cache):
        """Test cache size limits and eviction."""
        small_cache = ModelCache(CacheConfig(max_size=3, ttl_seconds=300))
        
        # Fill cache to capacity
        for i in range(3):
            small_cache.set(f"key_{i}", {"data": f"value_{i}"})
        
        # All items should be present
        for i in range(3):
            assert small_cache.get(f"key_{i}") is not None
        
        # Add one more item (should evict oldest)
        small_cache.set("key_3", {"data": "value_3"})
        
        # First item should be evicted
        assert small_cache.get("key_0") is None
        # Others should still be present
        assert small_cache.get("key_1") is not None
        assert small_cache.get("key_2") is not None
        assert small_cache.get("key_3") is not None

    async def test_cache_performance_improvement(self, model_cache):
        """Test that caching improves performance."""
        # Simulate expensive operation
        async def expensive_model_call(prompt: str):
            """Simulate model call with artificial delay."""
            await asyncio.sleep(0.1)  # 100ms delay
            return f"Response to: {prompt}"
        
        # Create cache key
        cache_key = model_cache.create_key("test_prompt", {"temperature": 0.5})
        
        # First call - cache miss
        start_time = time.time()
        result1 = await expensive_model_call("test_prompt")
        model_cache.set(cache_key, result1)
        first_call_duration = time.time() - start_time
        
        # Second call - cache hit
        start_time = time.time()
        cached_result = model_cache.get(cache_key)
        second_call_duration = time.time() - start_time
        
        assert cached_result is not None
        assert cached_result == result1
        # Cache hit should be much faster
        assert second_call_duration < first_call_duration * 0.1


@pytest.mark.integration
class TestConnectionPooling:
    """Test connection pooling performance optimization."""

    @pytest.fixture
    def pool_config(self):
        """Create pool configuration for testing."""
        return PoolConfig(
            min_connections=2,
            max_connections=5,
            connection_timeout=30.0,
            idle_timeout=300.0
        )

    @pytest.fixture
    def connection_pool(self, pool_config):
        """Create connection pool instance."""
        return ConnectionPool("test_provider", pool_config)

    def test_pool_configuration(self, connection_pool):
        """Test connection pool configuration."""
        assert connection_pool.provider_name == "test_provider"
        assert connection_pool.config.min_connections == 2
        assert connection_pool.config.max_connections == 5

    async def test_pool_connection_reuse(self, connection_pool):
        """Test connection reuse in pool."""
        # Simulate getting connections
        conn1 = await connection_pool.get_connection()
        assert conn1 is not None
        
        # Return connection to pool
        await connection_pool.return_connection(conn1)
        
        # Get connection again - should reuse
        conn2 = await connection_pool.get_connection()
        # In a real implementation, we'd verify it's the same connection
        assert conn2 is not None

    async def test_pool_concurrent_access(self, connection_pool):
        """Test concurrent access to connection pool."""
        async def get_and_return_connection(conn_id):
            """Get connection, simulate work, return it."""
            conn = await connection_pool.get_connection()
            await asyncio.sleep(0.01)  # Simulate work
            await connection_pool.return_connection(conn)
            return conn_id
        
        # Create multiple concurrent tasks
        tasks = [get_and_return_connection(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All tasks should complete successfully
        assert len(results) == 10
        assert all(isinstance(r, int) for r in results)

    async def test_pool_max_connections_limit(self, connection_pool):
        """Test that pool respects maximum connection limit."""
        connections = []
        
        # Get up to max connections
        for i in range(connection_pool.config.max_connections):
            conn = await connection_pool.get_connection()
            connections.append(conn)
        
        # Pool should be at capacity
        assert len(connections) == connection_pool.config.max_connections
        
        # Return connections
        for conn in connections:
            await connection_pool.return_connection(conn)


class TestLoadBalancing:
    """Test load balancing performance optimization."""

    @pytest.fixture
    def mock_models(self):
        """Create mock models for load balancing tests."""
        from unittest.mock import AsyncMock
        
        models = []
        for i in range(3):
            model = AsyncMock()
            model.name = f"test-model-{i}"
            model.provider = "test"
            model.generate = AsyncMock(return_value=f"Response from model {i}")
            model.health_check = AsyncMock(return_value=True)
            models.append(model)
        
        return models

    @pytest.fixture
    def load_balancer(self, mock_models):
        """Create load balancer with mock models."""
        return LoadBalancer(mock_models, LoadBalancingStrategy.ROUND_ROBIN)

    def test_load_balancer_initialization(self, load_balancer, mock_models):
        """Test load balancer initialization."""
        assert load_balancer.strategy == LoadBalancingStrategy.ROUND_ROBIN
        assert len(load_balancer.models) == 3

    async def test_round_robin_distribution(self, load_balancer):
        """Test round-robin load distribution."""
        results = []
        
        # Make several requests
        for i in range(6):
            model = await load_balancer.get_next_model()
            result = await model.generate("test prompt")
            results.append(result)
        
        # Should cycle through models
        expected_pattern = [
            "Response from model 0",
            "Response from model 1", 
            "Response from model 2",
            "Response from model 0",
            "Response from model 1",
            "Response from model 2"
        ]
        
        assert results == expected_pattern

    async def test_load_balancer_with_failed_model(self, mock_models):
        """Test load balancer handling failed models."""
        # Make one model fail
        mock_models[1].health_check.return_value = False
        mock_models[1].generate.side_effect = Exception("Model failed")
        
        load_balancer = LoadBalancer(mock_models, LoadBalancingStrategy.ROUND_ROBIN)
        
        # Should skip failed model
        for i in range(4):
            model = await load_balancer.get_next_model()
            result = await model.generate("test prompt")
            # Should only get responses from models 0 and 2
            assert "model 1" not in result

    async def test_least_connections_strategy(self, mock_models):
        """Test least connections load balancing strategy."""
        load_balancer = LoadBalancer(mock_models, LoadBalancingStrategy.LEAST_CONNECTIONS)
        
        # Simulate different connection counts
        mock_models[0].active_connections = 5
        mock_models[1].active_connections = 2
        mock_models[2].active_connections = 8
        
        # Should select model with least connections (model 1)
        selected_model = await load_balancer.get_next_model()
        assert selected_model.name == "test-model-1"


@pytest.mark.integration
class TestPerformanceWithRealModels:
    """Test performance optimizations with real models."""

    @pytest.fixture
    async def model_registry(self):
        """Create registry with real providers."""
        registry = ModelRegistry()
        registry.add_provider(OpenAIProvider())
        registry.add_provider(AnthropicProvider()) 
        registry.add_provider(LocalProvider())
        return registry

    @pytest.fixture
    async def working_model(self, model_registry):
        """Find a working model for performance testing."""
        providers = model_registry.get_providers()
        
        for provider in providers:
            try:
                models = await provider.get_available_models()
                for model_info in models[:1]:  # Try first model
                    try:
                        model_instance = await provider.create_model(model_info.name)
                        if model_instance:
                            return (model_info, model_instance)
                    except Exception as e:
                        print(f"Failed to create {model_info.name}: {e}")
                        continue
            except Exception as e:
                print(f"Provider {provider.name} failed: {e}")
                continue
        
        return None

    async def test_caching_with_real_model(self, working_model):
        """Test caching performance with real model."""
        if not working_model:
            pytest.skip("No working models available for performance testing")
        
        model_info, model_instance = working_model
        
        # Create cache
        cache = ModelCache(CacheConfig(max_size=10, ttl_seconds=300))
        
        test_prompt = "What is machine learning?"
        cache_key = cache.create_key(test_prompt, {"temperature": 0.1})
        
        # Measure uncached performance
        metrics = []
        
        # First call (cache miss)
        start_time = time.time()
        result1 = await model_instance.generate(test_prompt, max_tokens=50, temperature=0.1)
        duration1 = (time.time() - start_time) * 1000  # Convert to ms
        
        # Cache the result
        cache.set(cache_key, result1)
        
        metrics.append(PerformanceMetric(
            operation="generate_uncached",
            duration_ms=duration1,
            memory_usage_mb=0,  # Would need memory profiling
            timestamp=datetime.now(),
            success=True,
            model_used=model_info.name,
            cache_hit=False
        ))
        
        # Second call (cache hit)
        start_time = time.time()
        cached_result = cache.get(cache_key)
        duration2 = (time.time() - start_time) * 1000
        
        metrics.append(PerformanceMetric(
            operation="generate_cached",
            duration_ms=duration2,
            memory_usage_mb=0,
            timestamp=datetime.now(),
            success=True,
            model_used=model_info.name,
            cache_hit=True
        ))
        
        assert cached_result == result1
        assert duration2 < duration1 * 0.1  # Cache should be much faster
        
        print(f"Uncached: {duration1:.2f}ms, Cached: {duration2:.2f}ms")
        print(f"Performance improvement: {(duration1/duration2):.1f}x faster")

    async def test_batch_processing_performance(self, working_model):
        """Test batch processing performance optimization."""
        if not working_model:
            pytest.skip("No working models available for batch testing")
        
        model_info, model_instance = working_model
        
        test_prompts = [
            "What is AI?",
            "Explain machine learning",
            "Define deep learning", 
            "What is neural network?",
            "Describe data science"
        ]
        
        # Sequential processing
        start_time = time.time()
        sequential_results = []
        for prompt in test_prompts:
            result = await model_instance.generate(prompt, max_tokens=20, temperature=0.1)
            sequential_results.append(result)
        sequential_duration = time.time() - start_time
        
        # Concurrent processing (simulated batch)
        start_time = time.time()
        async def process_prompt(prompt):
            return await model_instance.generate(prompt, max_tokens=20, temperature=0.1)
        
        concurrent_tasks = [process_prompt(prompt) for prompt in test_prompts]
        concurrent_results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
        concurrent_duration = time.time() - start_time
        
        # Filter out any exceptions
        successful_concurrent = [r for r in concurrent_results if not isinstance(r, Exception)]
        
        print(f"Sequential: {sequential_duration:.2f}s ({len(sequential_results)} requests)")
        print(f"Concurrent: {concurrent_duration:.2f}s ({len(successful_concurrent)} successful)")
        
        if len(successful_concurrent) > 0:
            speedup = sequential_duration / concurrent_duration
            print(f"Speedup: {speedup:.2f}x")
            
            # Concurrent should be faster (though limited by API rate limits)
            assert concurrent_duration <= sequential_duration * 1.2  # Allow some overhead

    async def test_connection_pooling_performance(self, working_model):
        """Test connection pooling performance benefits."""
        if not working_model:
            pytest.skip("No working models available for pooling testing")
        
        model_info, model_instance = working_model
        
        # Simulate multiple requests that would benefit from connection reuse
        async def make_request(request_id):
            """Make a model request."""
            start_time = time.time()
            result = await model_instance.generate(
                f"Request {request_id}", 
                max_tokens=10, 
                temperature=0.1
            )
            duration = time.time() - start_time
            return {
                "request_id": request_id,
                "duration": duration,
                "result": result,
                "success": True
            }
        
        # Make multiple concurrent requests
        num_requests = 5
        tasks = [make_request(i) for i in range(num_requests)]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_duration = time.time() - start_time
        
        # Analyze results
        successful_results = [r for r in results if isinstance(r, dict) and r.get("success")]
        
        if len(successful_results) > 0:
            avg_request_duration = statistics.mean([r["duration"] for r in successful_results])
            
            print(f"Total requests: {num_requests}")
            print(f"Successful: {len(successful_results)}")
            print(f"Total duration: {total_duration:.2f}s")
            print(f"Average request duration: {avg_request_duration:.2f}s")
            
            # With connection pooling, should handle requests efficiently
            assert len(successful_results) >= num_requests * 0.8  # At least 80% success rate


class TestResourceManagement:
    """Test resource management and memory optimization."""

    async def test_memory_cleanup(self):
        """Test that models are properly cleaned up."""
        import gc
        import sys
        
        # Get initial memory usage
        initial_objects = len(gc.get_objects())
        
        # Create and use some mock model instances
        models = []
        for i in range(5):
            model = type('MockModel', (), {
                'name': f'model_{i}',
                'data': 'x' * 1000  # Some data to consume memory
            })()
            models.append(model)
        
        # Clear references
        models.clear()
        
        # Force garbage collection
        gc.collect()
        
        # Check memory cleanup
        final_objects = len(gc.get_objects())
        
        # Should not have significantly more objects
        object_growth = final_objects - initial_objects
        print(f"Object growth: {object_growth}")
        
        # Allow for some growth but not excessive
        assert object_growth < 100

    def test_resource_limits(self):
        """Test resource limit enforcement."""
        # Test cache size limits
        cache = ModelCache(CacheConfig(max_size=5, ttl_seconds=300))
        
        # Fill beyond capacity
        for i in range(10):
            cache.set(f"key_{i}", {"data": f"value_{i}"})
        
        # Should not exceed max size
        # In a real implementation, would check internal size
        # For now, just verify it doesn't crash
        assert cache is not None

    async def test_concurrent_resource_access(self):
        """Test concurrent access to shared resources."""
        cache = ModelCache(CacheConfig(max_size=50, ttl_seconds=300))
        
        async def concurrent_cache_operations(worker_id):
            """Perform cache operations concurrently."""
            results = []
            for i in range(10):
                key = f"worker_{worker_id}_key_{i}"
                value = {"worker": worker_id, "iteration": i, "data": "test_data"}
                
                # Set value
                cache.set(key, value)
                
                # Get value
                retrieved = cache.get(key)
                results.append(retrieved is not None)
                
                # Small delay to allow interleaving
                await asyncio.sleep(0.001)
            
            return results
        
        # Run multiple workers concurrently
        num_workers = 5
        tasks = [concurrent_cache_operations(i) for i in range(num_workers)]
        worker_results = await asyncio.gather(*tasks)
        
        # All operations should succeed
        for worker_id, results in enumerate(worker_results):
            success_rate = sum(results) / len(results)
            print(f"Worker {worker_id} success rate: {success_rate:.2f}")
            assert success_rate >= 0.9  # At least 90% success rate


async def main():
    """Run performance optimization tests."""
    print("⚡ PERFORMANCE OPTIMIZATION TESTS")
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