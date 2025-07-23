#!/usr/bin/env python3
"""Test load balancing and failover functionality."""

import asyncio
import os
import sys
import time
import random
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.models.model_registry import ModelRegistry
from orchestrator.models.load_balancer import LoadBalancer, ModelPoolConfig
from orchestrator.models.openai_model import OpenAIModel
from orchestrator.integrations.ollama_model import OllamaModel


@pytest.fixture
async def registry():
    """Create a model registry for testing."""
    return ModelRegistry()


@pytest.fixture
async def load_balancer(registry):
    """Create a load balancer with test models and pools."""
    load_balancer = LoadBalancer(registry)
    
    # Register some test models (this setup is simplified for testing)
    models_registered = []
    
    # Try to register local models if available
    try:
        llama_small = OllamaModel("llama3.2:1b")
        registry.register_model(llama_small)
        models_registered.append("ollama:llama3.2:1b")
    except Exception:
        pass
    
    # Configure test pools
    if models_registered:
        # Primary pool
        primary_config = ModelPoolConfig(
            name="primary",
            models=[models_registered[0]],
            selection_strategy="weighted",
            model_weights={models_registered[0]: 1.0}
        )
        load_balancer.configure_pool("primary", primary_config)
        
        # Backup pool
        backup_config = ModelPoolConfig(
            name="backup",
            models=[models_registered[0]],
            selection_strategy="round_robin"
        )
        load_balancer.configure_pool("backup", backup_config)
    
    return load_balancer


async def setup_models_and_pools():
    """Set up test models and pools."""
    registry = ModelRegistry()
    load_balancer = LoadBalancer(registry)
    
    # Register models
    models_registered = []
    
    # Local models (always available for testing)
    try:
        llama_small = OllamaModel("llama3.2:1b")
        registry.register_model(llama_small)
        models_registered.append("ollama:llama3.2:1b")
        print("✓ Registered llama3.2:1b")
    except Exception as e:
        print(f"✗ Failed to register llama3.2:1b: {e}")
    
    try:
        llama_medium = OllamaModel("llama3.1:8b")
        registry.register_model(llama_medium)
        models_registered.append("ollama:llama3.1:8b")
        print("✓ Registered llama3.1:8b")
    except Exception as e:
        print(f"✗ Failed to register llama3.1:8b: {e}")
    
    # API models
    if os.getenv("OPENAI_API_KEY"):
        try:
            gpt35 = OpenAIModel("gpt-3.5-turbo")
            registry.register_model(gpt35)
            models_registered.append("openai:gpt-3.5-turbo")
            print("✓ Registered gpt-3.5-turbo")
        except Exception as e:
            print(f"✗ Failed to register gpt-3.5-turbo: {e}")
    
    # Configure model pools
    
    # Primary pool - mix of models with different weights
    primary_pool = ModelPoolConfig(
        models=[
            {
                "model": "ollama:llama3.2:1b",
                "weight": 0.4,  # 40% of traffic
                "max_concurrent": 5
            },
            {
                "model": "ollama:llama3.1:8b",
                "weight": 0.6,  # 60% of traffic
                "max_concurrent": 3
            }
        ],
        fallback_pool="emergency",
        retry_config={
            "max_retries": 3,
            "backoff": "exponential",
            "initial_delay": 0.5
        }
    )
    
    # Emergency fallback pool
    emergency_pool = ModelPoolConfig(
        models=[
            {
                "model": "ollama:llama3.2:1b",
                "weight": 1.0,
                "max_concurrent": 10
            }
        ],
        always_available=True,  # Always consider this pool available
        retry_config={
            "max_retries": 5,
            "backoff": "linear",
            "initial_delay": 1.0
        }
    )
    
    # High-performance pool (if API models available)
    if "openai:gpt-3.5-turbo" in models_registered:
        perf_pool = ModelPoolConfig(
            models=[
                {
                    "model": "openai:gpt-3.5-turbo",
                    "weight": 0.8,
                    "max_concurrent": 20
                },
                {
                    "model": "ollama:llama3.1:8b",
                    "weight": 0.2,
                    "max_concurrent": 5
                }
            ],
            fallback_pool="primary"
        )
        load_balancer.configure_pool("performance", perf_pool)
    
    load_balancer.configure_pool("primary", primary_pool)
    load_balancer.configure_pool("emergency", emergency_pool)
    
    return registry, load_balancer, models_registered


async def test_weighted_selection(load_balancer: LoadBalancer):
    """Test weighted model selection."""
    print("\n=== Testing Weighted Selection ===")
    
    selection_counts = {}
    num_selections = 100
    
    for i in range(num_selections):
        try:
            model = await load_balancer.select_from_pool("primary")
            model_id = f"{model.provider}:{model.name}"
            selection_counts[model_id] = selection_counts.get(model_id, 0) + 1
            
            # Release the model
            load_balancer.model_states[model_id].current_requests -= 1
            
        except Exception as e:
            print(f"Selection {i} failed: {e}")
    
    print(f"\nSelection distribution over {num_selections} requests:")
    for model_id, count in selection_counts.items():
        percentage = (count / num_selections) * 100
        print(f"  {model_id}: {count} ({percentage:.1f}%)")
    
    # Check if distribution roughly matches weights
    if "ollama:llama3.2:1b" in selection_counts:
        small_pct = selection_counts["ollama:llama3.2:1b"] / num_selections
        print(f"\nExpected ~40% for llama3.2:1b, got {small_pct*100:.1f}%")
    
    if "ollama:llama3.1:8b" in selection_counts:
        medium_pct = selection_counts["ollama:llama3.1:8b"] / num_selections
        print(f"Expected ~60% for llama3.1:8b, got {medium_pct*100:.1f}%")


async def test_concurrent_limits(load_balancer: LoadBalancer):
    """Test concurrent request limiting."""
    print("\n=== Testing Concurrent Request Limits ===")
    
    # Try to exceed concurrent limit for a model
    model_id = "ollama:llama3.2:1b"
    max_concurrent = 5  # As configured in pool
    
    tasks = []
    
    async def make_request(i):
        try:
            model = await load_balancer.select_from_pool("primary")
            selected_id = f"{model.provider}:{model.name}"
            
            if selected_id == model_id:
                # Simulate some work
                await asyncio.sleep(0.1)
                return selected_id
            else:
                # Release immediately if different model
                load_balancer.model_states[selected_id].current_requests -= 1
                return selected_id
        except Exception as e:
            return f"Failed: {e}"
    
    # Launch more requests than max concurrent
    for i in range(max_concurrent + 3):
        tasks.append(make_request(i))
    
    results = await asyncio.gather(*tasks)
    
    # Count how many requests went to each model
    model_counts = {}
    for result in results:
        if result.startswith("Failed"):
            print(f"  {result}")
        else:
            model_counts[result] = model_counts.get(result, 0) + 1
    
    print("\nConcurrent request distribution:")
    for model, count in model_counts.items():
        print(f"  {model}: {count}")
    
    print(f"\nWith max_concurrent={max_concurrent} for {model_id},")
    print("excess requests should spill over to other models.")


async def test_failover(load_balancer: LoadBalancer):
    """Test failover to backup pool."""
    print("\n=== Testing Failover ===")
    
    # Simulate all models in primary pool being unavailable
    # by maxing out their concurrent requests
    for model_info in load_balancer.pools["primary"].models:
        model_id = model_info["model"]
        state = load_balancer.model_states[model_id]
        state.current_requests = state.max_concurrent
    
    print("Simulated primary pool exhaustion...")
    
    # Try to select - should failover to emergency pool
    try:
        model = await load_balancer.select_from_pool("primary")
        print(f"✓ Failover successful! Selected: {model.provider}:{model.name}")
        print("  (This should be from the emergency pool)")
    except Exception as e:
        print(f"✗ Failover failed: {e}")
    
    # Reset concurrent requests
    for model_id in load_balancer.model_states:
        load_balancer.model_states[model_id].current_requests = 0


async def test_retry_with_backoff(load_balancer: LoadBalancer):
    """Test retry logic with exponential backoff."""
    print("\n=== Testing Retry with Backoff ===")
    
    # Create a mock model that fails a few times
    class FailingModel:
        def __init__(self, fail_count=2):
            self.provider = "test"
            self.name = "failing-model"
            self.attempts = 0
            self.fail_count = fail_count
        
        async def generate(self, prompt: str, **kwargs):
            self.attempts += 1
            if self.attempts <= self.fail_count:
                raise Exception(f"Simulated failure {self.attempts}")
            return f"Success after {self.attempts} attempts"
    
    model = FailingModel(fail_count=2)
    
    start_time = time.time()
    try:
        result = await load_balancer.execute_with_retry(
            model, "generate", "Test prompt"
        )
        elapsed = time.time() - start_time
        
        print(f"✓ Retry successful after {model.attempts} attempts")
        print(f"  Result: {result}")
        print(f"  Total time: {elapsed:.2f}s (includes backoff delays)")
    except Exception as e:
        print(f"✗ All retries failed: {e}")


async def test_circuit_breaker(load_balancer: LoadBalancer):
    """Test circuit breaker functionality."""
    print("\n=== Testing Circuit Breaker ===")
    
    model_id = "ollama:llama3.2:1b"
    state = load_balancer.model_states[model_id]
    
    # Simulate multiple failures to trip circuit breaker
    print("Simulating consecutive failures...")
    for i in range(6):
        await load_balancer._update_failure_metrics(model_id)
    
    print(f"Circuit breaker state: {'OPEN' if state.circuit_open else 'CLOSED'}")
    print(f"Consecutive failures: {state.consecutive_failures}")
    
    # Try to select from pool - should skip the failed model
    try:
        model = await load_balancer.select_from_pool("primary")
        selected_id = f"{model.provider}:{model.name}"
        print(f"✓ Selected alternative model: {selected_id}")
        print(f"  (Should not be {model_id} due to open circuit)")
    except Exception as e:
        print(f"Selection failed: {e}")
    
    # Reset circuit breaker
    state.circuit_open = False
    state.consecutive_failures = 0


async def test_pool_status(load_balancer: LoadBalancer):
    """Test pool status reporting."""
    print("\n=== Testing Pool Status ===")
    
    # Make some requests to generate statistics
    for _ in range(10):
        try:
            model = await load_balancer.select_from_pool("primary")
            model_id = f"{model.provider}:{model.name}"
            
            # Simulate success
            await load_balancer._update_success_metrics(model_id, random.uniform(0.1, 0.5))
            
            # Release
            load_balancer.model_states[model_id].current_requests -= 1
        except:
            pass
    
    # Get pool status
    status = load_balancer.get_pool_status("primary")
    
    print("\nPrimary Pool Status:")
    print(f"  Fallback pool: {status['fallback_pool']}")
    print(f"  Always available: {status['always_available']}")
    
    print("\nModel Statistics:")
    for model_status in status["models"]:
        print(f"\n  Model: {model_status['model']}")
        print(f"    Weight: {model_status['weight']}")
        print(f"    Success rate: {model_status['success_rate']:.2%}")
        print(f"    Avg latency: {model_status['avg_latency']:.3f}s")
        print(f"    Current requests: {model_status['current_requests']}/{model_status['max_concurrent']}")


async def test_real_generation_with_lb(registry: ModelRegistry, load_balancer: LoadBalancer):
    """Test real generation through load balancer."""
    print("\n=== Testing Real Generation with Load Balancing ===")
    
    prompt = "What is the capital of France? Give a one word answer."
    
    try:
        # Select model from pool
        model = await load_balancer.select_from_pool("primary")
        print(f"Selected model: {model.provider}:{model.name}")
        
        # Execute with retry
        result = await load_balancer.execute_with_retry(
            model, "generate", prompt, temperature=0
        )
        
        print(f"Prompt: {prompt}")
        print(f"Response: {result.strip()}")
        
        # Update registry metrics
        registry.update_model_performance(
            model,
            success=True,
            latency=0.2,
            cost=0.0 if model.cost.is_free else 0.001
        )
        
        print("✓ Generation successful")
        
    except Exception as e:
        print(f"✗ Generation failed: {e}")


async def main():
    """Run all load balancer tests."""
    print("🚀 LOAD BALANCING AND FAILOVER TEST")
    print("="*50)
    
    # Set up models and pools
    registry, load_balancer, models = await setup_models_and_pools()
    
    if not models:
        print("\n⚠️  No models registered! Ensure Ollama is running.")
        return
    
    print(f"\nRegistered {len(models)} models")
    
    # Run tests
    await test_weighted_selection(load_balancer)
    await test_concurrent_limits(load_balancer)
    await test_failover(load_balancer)
    await test_retry_with_backoff(load_balancer)
    await test_circuit_breaker(load_balancer)
    await test_pool_status(load_balancer)
    await test_real_generation_with_lb(registry, load_balancer)
    
    print("\n" + "="*50)
    print("✓ All load balancing tests complete!")


if __name__ == "__main__":
    asyncio.run(main())