#!/usr/bin/env python3
"""Test domain-specific model routing."""

import os
import pytest

from orchestrator.models.model_registry import ModelRegistry
from orchestrator.models.domain_router import DomainRouter, DomainConfig
from orchestrator.models.openai_model import OpenAIModel
from orchestrator.integrations.ollama_model import OllamaModel


async def setup_registry():
    """Set up model registry with domain capabilities."""
    registry = ModelRegistry()

    # Register models with domain expertise
    try:
        # General purpose model
        llama = OllamaModel("llama3.1:8b")
        llama.capabilities.domains = ["general", "educational"]
        llama.capabilities.accuracy_score = 0.85
        llama._expertise = ["general", "educational"]  # For compatibility
        registry.register_model(llama)
        print("✓ Registered llama3.1:8b (general, educational)")
    except Exception as e:
        print(f"✗ Failed to register llama3.1:8b: {e}")

    if os.getenv("OPENAI_API_KEY"):
        try:
            # GPT-3.5 for general/educational
            gpt35 = OpenAIModel("gpt-3.5-turbo")
            gpt35.capabilities.domains = ["general", "educational", "creative"]
            gpt35.capabilities.accuracy_score = 0.85
            gpt35._expertise = [
                "general",
                "educational",
                "creative",
            ]  # For compatibility
            registry.register_model(gpt35)
            print("✓ Registered gpt-3.5-turbo (general, educational, creative)")

            # GPT-4 for advanced domains
            gpt4 = OpenAIModel("gpt-4")
            gpt4.capabilities.domains = [
                "general",
                "technical",
                "medical",
                "legal",
                "scientific",
                "financial",
            ]
            gpt4.capabilities.accuracy_score = 0.95
            gpt4._expertise = [
                "general",
                "technical",
                "medical",
                "legal",
                "scientific",
                "financial",
            ]  # For compatibility
            registry.register_model(gpt4)
            print(
                "✓ Registered gpt-4 (technical, medical, legal, scientific, financial)"
            )
        except Exception as e:
            print(f"✗ Failed to register OpenAI models: {e}")

    return registry


@pytest.fixture
async def registry():
    """Create model registry fixture."""
    return await setup_registry()


@pytest.fixture
async def router(registry):
    """Create domain router fixture."""
    # Create router (it initializes with default domains)
    return DomainRouter(registry)


@pytest.mark.asyncio
async def test_domain_detection(router: DomainRouter):
    """Test domain detection in various texts."""
    print("\n=== Testing Domain Detection ===")

    test_texts = [
        # Medical
        (
            "The patient presented with symptoms of acute respiratory distress. "
            "Initial diagnosis suggests pneumonia, requiring immediate antibiotic treatment.",
            "medical",
        ),
        # Legal
        (
            "According to Section 5.2 of the contract, the liability for damages "
            "shall not exceed the total contract value. This clause is subject to "
            "jurisdiction of the state court.",
            "legal",
        ),
        # Creative
        (
            "Write a short story about a robot who discovers it can feel emotions. "
            "The narrative should explore themes of consciousness and identity.",
            "creative",
        ),
        # Technical
        (
            "We need to implement a microservices architecture with proper API "
            "gateway integration. The system should handle 10k requests per second.",
            "technical",
        ),
        # Scientific
        (
            "Our hypothesis suggests that increased CO2 levels correlate with "
            "temperature rise. The experimental methodology includes controlled "
            "variables and peer-reviewed analysis.",
            "scientific",
        ),
        # Financial
        (
            "The portfolio shows a 15% return on investment this quarter. "
            "Risk assessment indicates moderate exposure to market volatility.",
            "financial",
        ),
        # Educational
        (
            "Can you explain how photosynthesis works? I'm a student trying "
            "to understand the concept for my biology class.",
            "educational",
        ),
        # Multi-domain
        (
            "As a medical researcher, I need to analyze patient data to test "
            "my hypothesis about treatment efficacy. The results will be "
            "published in a peer-reviewed journal.",
            "medical/scientific",
        ),
    ]

    for text, expected in test_texts:
        print(f"\nText: {text[:60]}...")
        print(f"Expected: {expected}")

        detected = router.detect_domains(text)
        if detected:
            print(
                f"Detected: {', '.join([f'{d[0]} ({d[1]:.2f})' for d in detected[:3]])}"
            )
        else:
            print("Detected: None")

    # Test analysis function
    print("\n--- Full Analysis Example ---")
    analysis = router.analyze_text(test_texts[0][0])
    print(f"Text length: {analysis['text_length']}")
    print(f"Primary domain: {analysis['primary_domain']}")
    print(f"All domains: {analysis['detected_domains']}")


@pytest.mark.asyncio
async def test_domain_routing(router: DomainRouter):
    """Test model selection based on domain."""
    print("\n=== Testing Domain-Based Model Selection ===")

    test_cases = [
        {
            "text": "Diagnose the patient's condition based on these symptoms",
            "expected_domain": "medical",
            "expected_accuracy": 0.8,
        },
        {
            "text": "Review this contract for potential legal issues",
            "expected_domain": "legal",
            "expected_accuracy": 0.8,
        },
        {
            "text": "Write a creative story about time travel",
            "expected_domain": "creative",
            "expected_accuracy": 0.8,
        },
        {
            "text": "Explain the water cycle to a 5th grade student",
            "expected_domain": "educational",
            "expected_accuracy": 0.8,
        },
    ]

    for case in test_cases:
        print(f"\nText: {case['text']}")
        print(f"Expected domain: {case['expected_domain']}")

        try:
            # Route by domain
            model = await router.route_by_domain(case["text"])

            print(f"Selected model: {model.provider}:{model.name}")
            print(f"Model domains: {model.capabilities.domains}")
            print(f"Model accuracy: {model.capabilities.accuracy_score}")

            # Check if model meets domain requirements
            if case["expected_domain"] in model.capabilities.domains:
                print("✓ Model has required domain expertise")
            else:
                print("✗ Model lacks required domain expertise")

            if model.capabilities.accuracy_score >= case["expected_accuracy"]:
                print("✓ Model meets accuracy requirement")
            else:
                print("✗ Model below accuracy requirement")

        except Exception as e:
            print(f"✗ Routing failed: {e}")


@pytest.mark.asyncio
async def test_custom_domain(router: DomainRouter):
    """Test registering and using custom domains."""
    print("\n=== Testing Custom Domain Registration ===")

    # Create custom domain for gaming
    gaming_domain = DomainConfig(
        name="gaming",
        keywords=["game", "player", "level", "quest", "boss", "gameplay", "mechanics"],
        patterns=[
            r"\b(game|player|gameplay|mechanic)\b",
            r"\b(level|quest|boss|npc|character)\b",
        ],
        preferred_models=["gpt-4", "claude-3-opus"],
        required_capabilities=["creative", "gaming"],
        min_accuracy_score=0.8,
    )

    # Register the domain
    router.register_domain(gaming_domain)
    print("✓ Registered custom 'gaming' domain")

    # Test detection
    gaming_text = "Design a boss battle for level 5 with unique gameplay mechanics"
    detected = router.detect_domains(gaming_text)

    print(f"\nText: {gaming_text}")
    print(f"Detected domains: {detected}")

    if detected and detected[0][0] == "gaming":
        print("✓ Custom domain detected correctly")
    else:
        print("✗ Custom domain not detected")

    # List all domains
    print(f"\nAll registered domains: {router.list_domains()}")


@pytest.mark.asyncio
async def test_domain_override(router: DomainRouter):
    """Test forcing specific domain selection."""
    print("\n=== Testing Domain Override ===")

    text = "This is a general text without specific domain indicators"

    # Test without override
    print(f"\nText: {text}")
    detected = router.detect_domains(text)
    print(f"Auto-detected domains: {detected[:3] if detected else 'None'}")

    # Test with override
    for domain in ["technical", "creative", "medical"]:
        try:
            model = await router.route_by_domain(text, domain_override=domain)
            print(f"\nForced domain: {domain}")
            print(f"Selected model: {model.provider}:{model.name}")
            print(f"Model domains: {model.capabilities.domains}")
        except Exception as e:
            print(f"\nForced domain: {domain}")
            print(f"✗ Selection failed: {e}")


@pytest.mark.asyncio
async def test_multi_domain_handling(router: DomainRouter):
    """Test handling of multi-domain content."""
    print("\n=== Testing Multi-Domain Content ===")

    # Text that spans multiple domains
    multi_domain_text = """
    As a medical AI researcher, I'm analyzing patient data to validate
    our hypothesis about a new treatment protocol. The results will be
    submitted for peer review and publication in a scientific journal.
    """

    print(f"Multi-domain text: {multi_domain_text.strip()[:100]}...")

    # Detect all domains
    detected = router.detect_domains(multi_domain_text, threshold=0.2)
    print("\nDetected domains:")
    for domain, confidence in detected:
        print(f"  - {domain}: {confidence:.2f}")

    # Route based on primary domain
    try:
        model = await router.route_by_domain(multi_domain_text)
        print(f"\nSelected model: {model.provider}:{model.name}")
        print(f"Model domains: {model.capabilities.domains}")

        # Check coverage
        detected_names = [d[0] for d in detected]
        covered = [d for d in detected_names if d in model.capabilities.domains]
        print(f"Domain coverage: {len(covered)}/{len(detected_names)} domains covered")

    except Exception as e:
        print(f"\n✗ Selection failed: {e}")


@pytest.mark.asyncio
async def test_real_generation_with_domain(
    registry: ModelRegistry, router: DomainRouter
):
    """Test real generation with domain-appropriate model."""
    print("\n=== Testing Real Generation with Domain Routing ===")

    # Different domain prompts
    prompts = [
        {
            "text": "Explain how machine learning works to a beginner",
            "domain": "educational",
        },
        {"text": "Write a haiku about artificial intelligence", "domain": "creative"},
    ]

    for prompt_info in prompts:
        prompt = prompt_info["text"]
        expected_domain = prompt_info["domain"]

        print(f"\nPrompt: {prompt}")
        print(f"Expected domain: {expected_domain}")

        try:
            # Select model based on domain
            model = await router.route_by_domain(prompt)
            print(f"Selected model: {model.provider}:{model.name}")

            # Generate response
            response = await model.generate(prompt, temperature=0.7, max_tokens=100)
            print(f"Response: {response.strip()[:150]}...")

            # Update metrics
            registry.update_model_performance(
                model,
                success=True,
                latency=0.5,
                cost=0.0 if model.cost.is_free else 0.002,
            )

            print("✓ Generation successful")

        except Exception as e:
            print(f"✗ Generation failed: {e}")


# This file now uses pytest - no main function needed
