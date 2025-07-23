#!/usr/bin/env python3
"""Comprehensive test of intelligent model routing with real API calls."""

import asyncio
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.models.model_registry import ModelRegistry
from orchestrator.models.model_selector import ModelSelector, ModelSelectionCriteria
from orchestrator.models.load_balancer import LoadBalancer, ModelPoolConfig
from orchestrator.models.domain_router import DomainRouter
from orchestrator.models.openai_model import OpenAIModel
from orchestrator.integrations.ollama_model import OllamaModel
from orchestrator.models.anthropic_model import AnthropicModel


async def setup_comprehensive_registry():
    """Set up registry with all available models."""
    registry = ModelRegistry()
    registry.enable_auto_registration()

    models_added = []

    # Local models (Ollama)
    ollama_models = [
        ("llama3.2:1b", ["general", "fast"], 0.7, "fast", 1.0),
        ("llama3.1:8b", ["general", "reasoning"], 0.85, "medium", 8.0),
        ("mistral:7b", ["general", "code"], 0.8, "medium", 7.0),
    ]

    for model_name, domains, accuracy, speed, size in ollama_models:
        try:
            model = OllamaModel(model_name)
            model.capabilities.domains = domains
            model.capabilities.accuracy_score = accuracy
            model.capabilities.speed_rating = speed
            model._expertise = domains
            model._size_billions = size
            registry.register_model(model)
            models_added.append(f"ollama:{model_name}")
            print(f"✓ Registered {model_name}")
        except Exception as e:
            print(f"✗ Failed to register {model_name}: {e}")

    # OpenAI models
    if os.getenv("OPENAI_API_KEY"):
        openai_models = [
            ("gpt-3.5-turbo", ["general", "code", "creative"], 0.85, "fast", True),
            (
                "gpt-4",
                ["general", "reasoning", "code", "technical", "medical", "legal"],
                0.95,
                "medium",
                True,
            ),
        ]

        for model_name, domains, accuracy, speed, is_code_specialized in openai_models:
            try:
                model = OpenAIModel(model_name)
                model.capabilities.domains = domains
                model.capabilities.accuracy_score = accuracy
                model.capabilities.speed_rating = speed
                model.capabilities.code_specialized = is_code_specialized
                model._expertise = domains
                registry.register_model(model)
                models_added.append(f"openai:{model_name}")
                print(f"✓ Registered {model_name}")
            except Exception as e:
                print(f"✗ Failed to register {model_name}: {e}")

    # Anthropic models
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            claude = AnthropicModel("claude-3-sonnet-20240229")
            claude.capabilities.domains = ["general", "reasoning", "code", "creative"]
            claude.capabilities.accuracy_score = 0.9
            claude.capabilities.speed_rating = "medium"
            claude.capabilities.code_specialized = True
            claude._expertise = claude.capabilities.domains
            registry.register_model(claude)
            models_added.append("anthropic:claude-3-sonnet-20240229")
            print("✓ Registered claude-3-sonnet")
        except Exception as e:
            print(f"✗ Failed to register Claude: {e}")

    return registry, models_added


async def test_auto_tag_routing(registry: ModelRegistry):
    """Test AUTO tag model selection with real generation."""
    print("\n=== Testing AUTO Tag Model Selection ===")

    selector = ModelSelector(registry)

    test_cases = [
        {
            "auto_tag": "Select a fast, cost-effective model for simple text generation",
            "prompt": "What is the capital of France?",
            "expected_type": "fast/cheap",
        },
        {
            "auto_tag": "Choose the best model for complex code generation with high accuracy",
            "prompt": "Write a Python function to calculate fibonacci numbers recursively",
            "expected_type": "code/accurate",
        },
        {
            "auto_tag": "Pick a creative model for storytelling",
            "prompt": "Write the opening line of a mystery novel",
            "expected_type": "creative",
        },
        {
            "auto_tag": "Select an accurate model for technical analysis requiring 8k context",
            "prompt": "Explain the concept of quantum entanglement",
            "expected_type": "technical/accurate",
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. AUTO: {test['auto_tag']}")
        print(f"   Prompt: {test['prompt'][:50]}...")

        try:
            # Select model using AUTO tag
            criteria = ModelSelectionCriteria()
            model = await selector.select_model(criteria, test["auto_tag"])

            print(f"   Selected: {model.provider}:{model.name}")
            print(
                f"   Properties: speed={model.capabilities.speed_rating}, "
                + f"accuracy={model.capabilities.accuracy_score}, "
                + f"cost={'free' if model.cost.is_free else 'paid'}"
            )

            # Generate response
            start_time = time.time()
            response = await model.generate(test["prompt"], temperature=0.7, max_tokens=50)
            latency = time.time() - start_time

            print(f"   Response: {response.strip()[:100]}...")
            print(f"   Latency: {latency:.2f}s")

            # Update metrics
            registry.update_model_performance(model, success=True, latency=latency)

        except Exception as e:
            print(f"   ✗ Failed: {e}")


async def test_cost_optimized_routing(registry: ModelRegistry):
    """Test cost-optimized model selection."""
    print("\n=== Testing Cost-Optimized Routing ===")

    selector = ModelSelector(registry)

    # Generate 10 simple prompts
    prompts = [
        "What is 2+2?",
        "Name a color",
        "What day comes after Monday?",
        "Is water wet?",
        "What's the opposite of hot?",
        "Count to 5",
        "Name a fruit",
        "What's 10 minus 3?",
        "Is the sky blue?",
        "What's the first letter of the alphabet?",
    ]

    total_cost = 0.0
    models_used = {}

    for prompt in prompts:
        try:
            # Select cost-optimized model
            criteria = ModelSelectionCriteria(
                prefer_free_models=True,
                selection_strategy="cost_optimized",
                max_cost_per_1k_tokens=0.01,
            )

            model = await selector.select_model(criteria)
            model_key = f"{model.provider}:{model.name}"
            models_used[model_key] = models_used.get(model_key, 0) + 1

            # Generate response
            response = await model.generate(prompt, temperature=0, max_tokens=10)

            # Calculate cost
            if not model.cost.is_free:
                # Estimate tokens
                input_tokens = len(prompt.split()) * 1.5  # Rough estimate
                output_tokens = len(response.split()) * 1.5
                cost = model.cost.calculate_cost(int(input_tokens), int(output_tokens))
                total_cost += cost

        except Exception as e:
            print(f"Failed on '{prompt}': {e}")

    print(f"\nResults for {len(prompts)} prompts:")
    print(f"Total estimated cost: ${total_cost:.4f}")
    print("Models used:")
    for model_key, count in models_used.items():
        print(f"  {model_key}: {count} times")


async def test_domain_specific_generation(registry: ModelRegistry):
    """Test domain-specific routing with real generation."""
    print("\n=== Testing Domain-Specific Generation ===")

    router = DomainRouter(registry)

    domain_prompts = [
        {
            "prompt": "Explain the symptoms and treatment for pneumonia",
            "expected_domain": "medical",
        },
        {"prompt": "Draft a non-disclosure agreement template", "expected_domain": "legal"},
        {
            "prompt": "Write a function to sort an array using quicksort",
            "expected_domain": "technical/code",
        },
        {"prompt": "Compose a haiku about the seasons", "expected_domain": "creative"},
        {"prompt": "Explain photosynthesis to a 10-year-old", "expected_domain": "educational"},
    ]

    for test in domain_prompts:
        print(f"\n--- {test['expected_domain'].upper()} Domain ---")
        print(f"Prompt: {test['prompt']}")

        try:
            # Detect domain
            domains = router.detect_domains(test["prompt"])
            print(f"Detected: {', '.join([f'{d[0]} ({d[1]:.2f})' for d in domains[:2]])}")

            # Route and generate
            model = await router.route_by_domain(test["prompt"])
            print(f"Model: {model.provider}:{model.name}")

            response = await model.generate(test["prompt"], temperature=0.7, max_tokens=100)
            print(f"Response: {response.strip()[:150]}...")

            # Verify domain coverage
            if domains and domains[0][0] in model.capabilities.domains:
                print("✓ Model has appropriate domain expertise")

        except Exception as e:
            print(f"✗ Failed: {e}")


async def test_load_balanced_generation(registry: ModelRegistry):
    """Test load balancing with real concurrent requests."""
    print("\n=== Testing Load-Balanced Generation ===")

    load_balancer = LoadBalancer(registry)

    # Configure pools
    primary_pool = ModelPoolConfig(
        models=[
            {"model": "ollama:llama3.2:1b", "weight": 0.6, "max_concurrent": 3},
            {"model": "ollama:llama3.1:8b", "weight": 0.4, "max_concurrent": 2},
        ]
    )

    # Add API models if available
    api_models = []
    if "openai:gpt-3.5-turbo" in [f"{m.provider}:{m.name}" for m in registry.models.values()]:
        api_models.append({"model": "openai:gpt-3.5-turbo", "weight": 0.7, "max_concurrent": 5})

    if api_models:
        api_pool = ModelPoolConfig(models=api_models, fallback_pool="local")
        load_balancer.configure_pool("api", api_pool)

    load_balancer.configure_pool("local", primary_pool)

    # Launch concurrent requests
    async def make_request(i: int, pool: str):
        try:
            model = await load_balancer.select_from_pool(pool)
            prompt = f"Generate a random number between 1 and 100. (Request {i})"

            result = await load_balancer.execute_with_retry(
                model, "generate", prompt, temperature=1.0, max_tokens=20
            )

            return {
                "request": i,
                "model": f"{model.provider}:{model.name}",
                "response": result.strip()[:50],
            }
        except Exception as e:
            return {"request": i, "error": str(e)}

    # Test with local pool
    print("\nTesting with local models (5 concurrent requests):")
    tasks = [make_request(i, "local") for i in range(5)]
    results = await asyncio.gather(*tasks)

    for result in results:
        if "error" in result:
            print(f"  Request {result['request']}: Failed - {result['error']}")
        else:
            print(f"  Request {result['request']}: {result['model']} -> {result['response']}")

    # Show pool statistics
    stats = load_balancer.get_pool_status("local")
    print("\nLocal Pool Statistics:")
    for model_stat in stats["models"]:
        print(
            f"  {model_stat['model']}: "
            + f"{model_stat['successful_requests']}/{model_stat['total_requests']} successful, "
            + f"avg latency: {model_stat['avg_latency']:.2f}s"
        )


async def test_model_performance_tracking(registry: ModelRegistry):
    """Test model performance tracking over multiple requests."""
    print("\n=== Testing Model Performance Tracking ===")

    selector = ModelSelector(registry)

    # Make multiple requests and track performance
    prompts = [
        "What is machine learning?",
        "Explain recursion in simple terms",
        "What are the primary colors?",
        "How does photosynthesis work?",
        "What is the speed of light?",
    ]

    print("\nMaking requests and tracking performance...")

    for i, prompt in enumerate(prompts):
        try:
            # Select model with balanced strategy
            criteria = ModelSelectionCriteria(selection_strategy="balanced")
            model = await selector.select_model(criteria)

            # Generate and time response
            start = time.time()
            response = await model.generate(prompt, temperature=0.5, max_tokens=50)
            latency = time.time() - start

            # Update metrics
            success = len(response.strip()) > 0
            cost = 0.001 if not model.cost.is_free else 0.0

            registry.update_model_performance(model, success=success, latency=latency, cost=cost)

            print(
                f"{i+1}. {model.provider}:{model.name} - {latency:.2f}s - {'✓' if success else '✗'}"
            )

        except Exception as e:
            print(f"{i+1}. Failed: {e}")

    # Show final statistics
    print("\n=== Final Model Statistics ===")
    stats = registry.get_model_statistics()

    print(f"Total models: {stats['total_models']}")
    print(f"Healthy models: {stats['healthy_models']}")

    print("\nModel Performance:")
    for model_key, perf in stats["selection_stats"]["model_performance"].items():
        if perf["attempts"] > 0:
            print(f"\n{model_key}:")
            print(f"  Attempts: {perf['attempts']}")
            print(f"  Success rate: {perf['success_rate']:.1%}")
            print(f"  Avg reward: {perf['average_reward']:.3f}")


async def test_failover_scenario(registry: ModelRegistry):
    """Test failover when primary models fail."""
    print("\n=== Testing Failover Scenario ===")

    selector = ModelSelector(registry)

    # Try to select a model with impossible requirements first
    print("1. Testing with impossible requirements (should failover):")
    try:
        criteria = ModelSelectionCriteria(
            min_context_window=1000000, min_accuracy_score=0.99  # 1M context (impossible)
        )
        model = await selector.select_model(criteria)
        print(f"   Unexpected success: {model.provider}:{model.name}")
    except Exception as e:
        print(f"   Expected failure: {e}")

    # Now with relaxed requirements
    print("\n2. Testing with relaxed requirements:")
    try:
        criteria = ModelSelectionCriteria(min_context_window=4096, min_accuracy_score=0.7)
        model = await selector.select_model(criteria)
        print(f"   Success: Selected {model.provider}:{model.name}")

        # Test generation
        response = await model.generate("Hello, how are you?", max_tokens=20)
        print(f"   Response: {response.strip()}")

    except Exception as e:
        print(f"   Failed: {e}")


async def save_test_results(results: dict):
    """Save test results to a file."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"test_results_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nTest results saved to {filename}")


async def main():
    """Run comprehensive intelligent routing tests."""
    print("🚀 COMPREHENSIVE INTELLIGENT MODEL ROUTING TEST")
    print("=" * 60)

    # Set up registry
    registry, models = await setup_comprehensive_registry()

    if not models:
        print("\n⚠️  No models registered! Please ensure:")
        print("  - Ollama is running for local models")
        print("  - API keys are set (OPENAI_API_KEY, ANTHROPIC_API_KEY)")
        return

    print(f"\n✓ Registered {len(models)} models")
    print(f"  Models: {', '.join(models)}")

    # Run all tests
    test_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models_available": models,
        "tests_run": [],
    }

    tests = [
        ("AUTO Tag Routing", test_auto_tag_routing),
        ("Cost-Optimized Routing", test_cost_optimized_routing),
        ("Domain-Specific Generation", test_domain_specific_generation),
        ("Load-Balanced Generation", test_load_balanced_generation),
        ("Model Performance Tracking", test_model_performance_tracking),
        ("Failover Scenarios", test_failover_scenario),
    ]

    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        start_time = time.time()

        try:
            await test_func(registry)
            duration = time.time() - start_time
            test_results["tests_run"].append(
                {"name": test_name, "status": "passed", "duration": duration}
            )
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            test_results["tests_run"].append(
                {"name": test_name, "status": "failed", "error": str(e)}
            )

    # Save results
    await save_test_results(test_results)

    print("\n" + "=" * 60)
    print("✓ All tests complete!")
    print(f"  Passed: {sum(1 for t in test_results['tests_run'] if t['status'] == 'passed')}")
    print(f"  Failed: {sum(1 for t in test_results['tests_run'] if t['status'] == 'failed')}")


if __name__ == "__main__":
    asyncio.run(main())
