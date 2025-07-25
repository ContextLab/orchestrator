#!/usr/bin/env python3
"""Test intelligent model routing with real models."""

import os
import pytest

from orchestrator.models.model_registry import ModelRegistry
from orchestrator.models.model_selector import ModelSelector, ModelSelectionCriteria
from orchestrator.models.openai_model import OpenAIModel
from orchestrator.models.anthropic_model import AnthropicModel
from orchestrator.integrations.ollama_model import OllamaModel
from orchestrator.core.model import ModelCost


async def setup_test_registry() -> ModelRegistry:
    """Set up a model registry with test models."""
    registry = ModelRegistry()

    # Register local models (Ollama) - free models
    try:
        # Small fast model
        llama_small = OllamaModel(
            model_name="llama3.2:1b", base_url="http://localhost:11434"
        )
        llama_small.capabilities.domains = ["general"]
        llama_small.capabilities.speed_rating = "fast"
        llama_small.capabilities.accuracy_score = 0.75
        llama_small.cost = ModelCost(is_free=True)
        llama_small._size_billions = 1.0
        registry.register_model(llama_small)
        print("✓ Registered llama3.2:1b")
    except Exception as e:
        print(f"✗ Failed to register llama3.2:1b: {e}")

    try:
        # Medium general model
        llama_medium = OllamaModel(
            model_name="llama3.1:8b", base_url="http://localhost:11434"
        )
        llama_medium.capabilities.domains = ["general", "technical"]
        llama_medium.capabilities.speed_rating = "medium"
        llama_medium.capabilities.accuracy_score = 0.85
        llama_medium.cost = ModelCost(is_free=True)
        llama_medium._size_billions = 8.0
        registry.register_model(llama_medium)
        print("✓ Registered llama3.1:8b")
    except Exception as e:
        print(f"✗ Failed to register llama3.1:8b: {e}")

    # Register API models if keys are available
    if os.getenv("OPENAI_API_KEY"):
        try:
            # GPT-3.5 - fast and cheap
            gpt35 = OpenAIModel("gpt-3.5-turbo")
            registry.register_model(gpt35)
            print("✓ Registered gpt-3.5-turbo")

            # GPT-4 - powerful but expensive
            gpt4 = OpenAIModel("gpt-4")
            registry.register_model(gpt4)
            print("✓ Registered gpt-4")
        except Exception as e:
            print(f"✗ Failed to register OpenAI models: {e}")

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            # Claude models
            claude = AnthropicModel("claude-3-sonnet-20240229")
            claude.capabilities.domains = ["general", "technical", "creative"]
            claude.capabilities.accuracy_score = 0.9
            claude.capabilities.speed_rating = "medium"
            registry.register_model(claude)
            print("✓ Registered claude-3-sonnet")
        except Exception as e:
            print(f"✗ Failed to register Anthropic models: {e}")

    return registry


@pytest.fixture
async def registry():
    """Create test registry fixture."""
    return await setup_test_registry()


@pytest.mark.asyncio
async def test_basic_selection(registry: ModelRegistry):
    """Test basic model selection."""
    print("\n=== Testing Basic Model Selection ===")

    selector = ModelSelector(registry)

    # Test 1: Select a fast model
    print("\n1. Selecting a fast model:")
    criteria = ModelSelectionCriteria(
        speed_preference="fast", selection_strategy="performance_optimized"
    )

    try:
        model = await selector.select_model(criteria)
        print(f"   Selected: {model.provider}:{model.name}")
        print(f"   Speed: {model.capabilities.speed_rating}")
        print(f"   Cost: Free={model.cost.is_free}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 2: Select a free model
    print("\n2. Selecting a free model:")
    criteria = ModelSelectionCriteria(
        prefer_free_models=True, selection_strategy="cost_optimized"
    )

    try:
        model = await selector.select_model(criteria)
        print(f"   Selected: {model.provider}:{model.name}")
        print(f"   Free: {model.cost.is_free}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 3: Select an accurate model
    print("\n3. Selecting an accurate model:")
    criteria = ModelSelectionCriteria(
        min_accuracy_score=0.9, selection_strategy="accuracy_optimized"
    )

    try:
        model = await selector.select_model(criteria)
        print(f"   Selected: {model.provider}:{model.name}")
        print(f"   Accuracy: {model.capabilities.accuracy_score}")
    except Exception as e:
        print(f"   Error: {e}")


@pytest.mark.asyncio
async def test_auto_tag_parsing(registry: ModelRegistry):
    """Test AUTO tag parsing."""
    print("\n=== Testing AUTO Tag Parsing ===")

    selector = ModelSelector(registry)

    # Test various AUTO tags
    auto_tags = [
        "Select a fast model for quick responses",
        "Choose the best model for code generation",
        "Pick a cost-effective model for general chat",
        "Select an accurate model for technical analysis",
        "Choose a model that can handle 32k context",
    ]

    for i, auto_tag in enumerate(auto_tags, 1):
        print(f"\n{i}. AUTO: {auto_tag}")

        try:
            model = await selector.select_model(ModelSelectionCriteria(), auto_tag)
            print(f"   Selected: {model.provider}:{model.name}")
            print(
                f"   Capabilities: speed={model.capabilities.speed_rating}, accuracy={model.capabilities.accuracy_score}"
            )
        except Exception as e:
            print(f"   Error: {e}")


@pytest.mark.asyncio
async def test_capability_matching(registry: ModelRegistry):
    """Test capability-based selection."""
    print("\n=== Testing Capability Matching ===")

    selector = ModelSelector(registry)

    # Test 1: Code-specialized model
    print("\n1. Selecting code-specialized model:")
    criteria = ModelSelectionCriteria(
        required_capabilities=["code"], required_tasks=["code", "generate"]
    )

    try:
        model = await selector.select_model(criteria)
        print(f"   Selected: {model.provider}:{model.name}")
        print(f"   Code specialized: {model.capabilities.code_specialized}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 2: Function calling model
    print("\n2. Selecting model with function calling:")
    criteria = ModelSelectionCriteria(required_capabilities=["tools"])

    try:
        model = await selector.select_model(criteria)
        print(f"   Selected: {model.provider}:{model.name}")
        print(f"   Supports functions: {model.capabilities.supports_function_calling}")
    except Exception as e:
        print(f"   Error: {e}")


@pytest.mark.asyncio
async def test_real_generation(registry: ModelRegistry):
    """Test actual generation with selected models."""
    print("\n=== Testing Real Generation ===")

    selector = ModelSelector(registry)

    # Select a fast free model for testing
    criteria = ModelSelectionCriteria(
        prefer_free_models=True,
        speed_preference="fast",
        selection_strategy="cost_optimized",
    )

    try:
        model = await selector.select_model(criteria)
        print(f"\nSelected model: {model.provider}:{model.name}")

        # Test generation
        prompt = "What is 2+2? Give a one word answer."
        print(f"Prompt: {prompt}")

        response = await model.generate(prompt, temperature=0)
        print(f"Response: {response.strip()}")

        # Update metrics based on success
        registry.update_model_performance(
            model, success=True, latency=0.5, cost=0.0 if model.cost.is_free else 0.001
        )

        print("✓ Generation successful, metrics updated")

    except Exception as e:
        print(f"✗ Generation failed: {e}")


@pytest.mark.asyncio
async def test_cost_calculation(registry: ModelRegistry):
    """Test cost calculation for different models."""
    print("\n=== Testing Cost Calculation ===")

    available_models = await registry.get_available_models()

    for model_key in available_models[:3]:  # Test first 3 models
        # Parse provider and model name correctly
        parts = model_key.split(":", 1)
        if len(parts) == 2:
            provider, model_name = parts
        else:
            provider = ""
            model_name = model_key

        model = registry.get_model(model_name, provider)

        print(f"\nModel: {model_key}")
        print(f"  Free: {model.cost.is_free}")

        if not model.cost.is_free:
            # Calculate cost for 1000 input + 500 output tokens
            cost = model.cost.calculate_cost(1000, 500)
            print(f"  Cost for 1K input + 500 output tokens: ${cost:.4f}")
            print(f"  Input rate: ${model.cost.input_cost_per_1k_tokens}/1K tokens")
            print(f"  Output rate: ${model.cost.output_cost_per_1k_tokens}/1K tokens")


# This file now uses pytest - no main function needed
