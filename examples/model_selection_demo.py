"""
Demonstration of intelligent model selection and management system.

This example shows how to use the model selection strategies, caching,
and lifecycle management features implemented in Stream B.
"""

import asyncio
from typing import Dict, Any

from src.orchestrator.models.registry import ModelRegistry
from src.orchestrator.models.selection.strategies import (
    TaskRequirements, 
    TaskBasedStrategy,
    CostAwareStrategy,
    PerformanceBasedStrategy,
    WeightedStrategy,
    FallbackStrategy,
)
from src.orchestrator.models.selection.manager import ModelManager
from src.orchestrator.models.providers.base import ProviderConfig


async def demo_model_selection():
    """Demonstrate intelligent model selection."""
    print("=== Model Selection & Management Demo ===\n")
    
    # Initialize model registry (simulated - in practice this would be configured)
    registry = ModelRegistry()
    
    # Note: In a real implementation, you would configure actual providers:
    # registry.configure_provider("openai", "openai", {"api_key": "your-key"})
    # registry.configure_provider("anthropic", "anthropic", {"api_key": "your-key"}) 
    # registry.configure_provider("local", "local", {"base_path": "/path/to/models"})
    
    print("Registry configured with providers")
    
    # Demo 1: Task-based selection
    print("\n1. Task-Based Model Selection")
    print("-" * 40)
    
    task_strategy = TaskBasedStrategy()
    requirements = TaskRequirements(
        task_type="code_generation",
        context_window=8192,
        required_capabilities=["supports_function_calling"],
        languages=["en"],
    )
    
    print(f"Requirements: {requirements.task_type} task with {requirements.context_window} context window")
    # result = await task_strategy.select_model(registry, requirements)
    # print(f"Selected: {result.model.name} from {result.provider}")
    # print(f"Reason: {result.selection_reason}")
    # print(f"Confidence: {result.confidence_score:.2f}")
    
    # Demo 2: Cost-aware selection
    print("\n2. Cost-Aware Model Selection") 
    print("-" * 40)
    
    cost_strategy = CostAwareStrategy(cost_weight=0.8)  # Heavy cost emphasis
    budget_requirements = TaskRequirements(
        task_type="text_generation",
        budget_limit=0.01,  # $0.01 budget
        budget_period="per-task",
    )
    
    print(f"Budget constraint: ${budget_requirements.budget_limit} {budget_requirements.budget_period}")
    # result = await cost_strategy.select_model(registry, budget_requirements)
    # print(f"Cost-effective choice: {result.model.name}")
    # print(f"Estimated cost: ${result.estimated_cost:.4f}")
    
    # Demo 3: Performance-optimized selection
    print("\n3. Performance-Based Model Selection")
    print("-" * 40)
    
    perf_strategy = PerformanceBasedStrategy(accuracy_weight=0.7, speed_weight=0.3)
    performance_requirements = TaskRequirements(
        task_type="analysis",
        max_latency_ms=2000,  # 2 second max latency
        accuracy_threshold=0.9,  # High accuracy requirement
    )
    
    print(f"Performance needs: accuracy >= {performance_requirements.accuracy_threshold}, latency <= {performance_requirements.max_latency_ms}ms")
    
    # Demo 4: Weighted multi-criteria selection
    print("\n4. Weighted Multi-Criteria Selection")
    print("-" * 40)
    
    weighted_strategy = WeightedStrategy(
        task_weight=0.3,
        cost_weight=0.2, 
        performance_weight=0.3,
        capability_weight=0.2,
    )
    
    balanced_requirements = TaskRequirements(
        task_type="creative_writing",
        preferred_domains=["creative", "writing"],
        budget_limit=0.05,
        languages=["en"],
    )
    
    print("Balanced selection considering task fit, cost, performance, and capabilities")
    
    # Demo 5: Fallback strategy
    print("\n5. Fallback Strategy Chain")
    print("-" * 40)
    
    fallback_strategy = FallbackStrategy(strategies=[
        PerformanceBasedStrategy(accuracy_weight=0.9, speed_weight=0.1),  # Try high accuracy first
        CostAwareStrategy(cost_weight=0.8),                               # Fallback to cost-effective
        TaskBasedStrategy(),                                              # Final fallback to task-based
    ])
    
    print("Strategy chain: Performance → Cost-Aware → Task-Based")
    
    # Demo 6: Model Manager with caching and optimization
    print("\n6. Model Manager with Optimization")
    print("-" * 40)
    
    manager = ModelManager(
        registry=registry,
        selection_strategy=fallback_strategy,
        enable_caching=True,
        enable_pooling=True,
        max_cache_size=1000,
        pool_size=10,
    )
    
    print("Model manager initialized with:")
    print(f"- Caching enabled (max size: 1000)")
    print(f"- Connection pooling enabled (pool size: 10)")
    print(f"- Fallback selection strategy")
    
    manager_info = manager.get_manager_info()
    print(f"Manager status: {manager_info}")
    
    # In a real implementation, you would use the manager like this:
    # 
    # # Intelligent model selection
    # selection = await manager.select_model(requirements)
    # 
    # # Generate with performance optimization
    # response, metadata = await manager.generate_with_model(
    #     model=selection.model,
    #     provider=selection.provider,
    #     prompt="Generate a Python function to calculate fibonacci numbers",
    #     temperature=0.7,
    # )
    # 
    # # Check performance stats
    # stats = await manager.get_model_stats()
    # print(f"Usage statistics: {stats}")
    # 
    # # Optimize performance based on collected data
    # optimization = await manager.optimize_performance()
    # print(f"Optimization suggestions: {optimization['optimizations']}")
    # 
    # # Health check all models
    # health = await manager.health_check()
    # print(f"System health: {health}")
    
    print("\n=== Demo Complete ===")
    print("The model selection and management system provides:")
    print("✓ Intelligent model selection based on task requirements")
    print("✓ Multi-criteria optimization (cost, performance, capabilities)")
    print("✓ Response caching for improved performance")
    print("✓ Connection pooling for efficient resource utilization")
    print("✓ Health monitoring and performance optimization")
    print("✓ Fallback strategies for robustness")
    
    # Cleanup
    await manager.cleanup()


def demo_cache_system():
    """Demonstrate caching system features."""
    from src.orchestrator.models.optimization.caching import ModelResponseCache
    
    print("\n=== Caching System Demo ===")
    
    # Create cache with specific configuration
    cache = ModelResponseCache(
        max_size=100,           # Max 100 entries
        default_ttl=3600.0,     # 1 hour TTL
        max_memory_mb=10,       # 10MB memory limit
    )
    
    print(f"Cache configured: max_size=100, ttl=1h, max_memory=10MB")
    
    # Generate cache keys
    key1 = cache.generate_cache_key("Hello world", temperature=0.7, max_tokens=100)
    key2 = cache.generate_cache_key("Hello world", temperature=0.8, max_tokens=100)  # Different temp
    key3 = cache.generate_cache_key("Hello world", temperature=0.7, max_tokens=100)  # Same as key1
    
    print(f"\nCache key examples:")
    print(f"Prompt 1 (temp=0.7): {key1}")
    print(f"Prompt 1 (temp=0.8): {key2}")  
    print(f"Prompt 1 (temp=0.7): {key3}")
    print(f"Key1 == Key3: {key1 == key3}")  # Should be True
    print(f"Key1 == Key2: {key1 == key2}")  # Should be False
    
    # Cache features demonstrated
    print(f"\nCache features:")
    print(f"✓ Intelligent key generation based on all parameters")
    print(f"✓ LRU eviction policy when cache is full")
    print(f"✓ TTL-based expiration for cache freshness")
    print(f"✓ Memory-based eviction to prevent excessive memory usage")
    print(f"✓ Selection result caching for model choice optimization")
    print(f"✓ Pattern-based cache invalidation")
    print(f"✓ Comprehensive statistics and monitoring")


def demo_pooling_system():
    """Demonstrate connection pooling features."""
    from src.orchestrator.models.optimization.pooling import ConnectionPool
    
    print("\n=== Connection Pooling Demo ===")
    
    # Create pool with specific configuration
    pool = ConnectionPool(
        provider_name="demo_provider",
        min_connections=2,              # Keep at least 2 connections
        max_connections=10,             # Max 10 connections
        max_idle_time=300.0,            # 5 minute idle timeout
        max_uses_per_connection=1000,   # Refresh after 1000 uses
        health_check_interval=60.0,     # Health check every minute
        queue_timeout=30.0,             # 30 second queue timeout
    )
    
    print(f"Connection pool configured:")
    print(f"- Provider: demo_provider")
    print(f"- Connection range: 2-10")
    print(f"- Idle timeout: 5 minutes")
    print(f"- Max uses per connection: 1000")
    print(f"- Health check interval: 1 minute")
    print(f"- Queue timeout: 30 seconds")
    
    pool_info = pool.get_pool_info()
    print(f"\nPool status: {pool_info['initialized']}")
    
    print(f"\nPooling features:")
    print(f"✓ Connection reuse to avoid repeated initialization")
    print(f"✓ Automatic scaling between min/max connections")
    print(f"✓ Request queuing when pool is at capacity")
    print(f"✓ Stale connection cleanup (idle timeout)")
    print(f"✓ Connection refresh (max uses limit)")
    print(f"✓ Health monitoring with periodic checks")
    print(f"✓ Load balancing across available connections")
    print(f"✓ Comprehensive performance statistics")


if __name__ == "__main__":
    # Run the async demo
    asyncio.run(demo_model_selection())
    
    # Run synchronous demos
    demo_cache_system()
    demo_pooling_system()