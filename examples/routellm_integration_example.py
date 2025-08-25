#!/usr/bin/env python3
"""
RouteLLM Integration Example

This example demonstrates how to use RouteLLM integration with the orchestrator
framework for intelligent model routing and cost optimization.
"""

import asyncio
import logging
from typing import Dict, Any

from orchestrator.models.model_registry import ModelRegistry
from orchestrator.models.domain_router import DomainRouter
from orchestrator.models.routellm_integration import (
    RouteLLMConfig,
    FeatureFlags,
    RouterType,
)


# Configure logging to see routing decisions
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def basic_example():
    """Basic RouteLLM integration example."""
    print("=" * 60)
    print("Basic RouteLLM Integration Example")
    print("=" * 60)
    
    # Create model registry (in real usage, this would be properly configured)
    registry = ModelRegistry()
    await registry.discover_models()  # Auto-discover available models
    
    # Create basic RouteLLM configuration
    config = RouteLLMConfig(
        enabled=True,
        router_type=RouterType.MATRIX_FACTORIZATION,
        strong_model="gpt-4-1106-preview",
        weak_model="gpt-3.5-turbo",
        threshold=0.11593,  # Default threshold for balanced routing
        cost_tracking_enabled=True,
    )
    
    # Configure feature flags for gradual rollout
    flags = FeatureFlags()
    flags.enable(FeatureFlags.ROUTELLM_ENABLED)
    flags.enable(FeatureFlags.ROUTELLM_TECHNICAL_DOMAIN)
    
    # Create domain router with RouteLLM integration
    router = DomainRouter(registry, config, flags)
    
    # Check RouteLLM status
    status = router.get_routellm_status()
    print(f"RouteLLM Status:")
    print(f"  Enabled: {status['config_enabled']}")
    print(f"  Feature Flag: {status['feature_flag_enabled']}")
    print(f"  Controller Available: {status['controller_available']}")
    print(f"  Router Type: {status['router_type']}")
    print(f"  Strong Model: {status['strong_model']}")
    print(f"  Weak Model: {status['weak_model']}")
    print()
    
    # Test queries with different complexity levels
    test_queries = [
        "Hello, how are you?",  # Simple query
        "Explain machine learning basics",  # Medium complexity
        "Design a distributed microservices architecture with Kubernetes, implementing service mesh patterns, observability, and automated CI/CD",  # Complex
        "What's the weather like?",  # Simple
        "Implement a neural network from scratch using backpropagation",  # Complex technical
    ]
    
    print("Testing different query complexities:")
    print("-" * 40)
    
    for i, query in enumerate(test_queries, 1):
        try:
            # Analyze the query
            analysis = router.analyze_text(query)
            
            # Route the query (this would normally be async)
            selected_model = await router.route_by_domain(query)
            
            print(f"{i}. Query: \"{query[:50]}{'...' if len(query) > 50 else ''}\"")
            print(f"   Domains: {[d['domain'] for d in analysis['detected_domains']]}")
            
            if analysis.get('routellm_enabled'):
                print(f"   Complexity: {analysis['complexity_score']:.3f}")
                print(f"   RouteLLM Recommendation: {analysis['recommended_routing']}")
                print(f"   Confidence: {analysis['routing_confidence']:.3f}")
            
            print(f"   Selected Model: {selected_model.provider}:{selected_model.name}")
            print()
            
        except Exception as e:
            logger.error(f"Error processing query {i}: {e}")
    
    # Show routing metrics
    metrics = router.get_routing_metrics_summary()
    if "error" not in metrics:
        print("Routing Metrics Summary:")
        print(f"  Total Requests: {metrics['total_requests']}")
        print(f"  Success Rate: {metrics['success_rate']:.1%}")
        print(f"  RouteLLM Usage: {metrics['routellm_usage_rate']:.1%}")
        print(f"  Average Cost: ${metrics['average_cost']:.6f}")
        print(f"  Average Latency: {metrics['average_latency_ms']:.1f}ms")
        print()


async def advanced_configuration_example():
    """Advanced configuration and monitoring example."""
    print("=" * 60)
    print("Advanced Configuration Example")
    print("=" * 60)
    
    registry = ModelRegistry()
    await registry.discover_models()
    
    # Advanced configuration with domain-specific overrides
    config = RouteLLMConfig(
        enabled=True,
        router_type=RouterType.BERT_CLASSIFIER,
        strong_model="gpt-4-turbo",
        weak_model="gpt-3.5-turbo",
        threshold=0.15,  # More conservative default threshold
        
        # Domain-specific routing overrides
        domain_routing_overrides={
            "medical": {
                "threshold": 0.05,  # Very conservative for medical queries
                "strong_model": "gpt-4-1106-preview"  # Use specific model for medical
            },
            "legal": {
                "threshold": 0.08,  # Conservative for legal queries
            },
            "creative": {
                "threshold": 0.3,  # More liberal for creative tasks
            },
            "technical": {
                "threshold": 0.2,  # Moderate for technical queries
            }
        },
        
        # Performance tuning
        timeout_seconds=20.0,
        cost_optimization_target=0.6,  # Target 60% cost reduction
        metrics_retention_days=7,  # Keep metrics for 1 week
    )
    
    # Configure feature flags for specific domains
    flags = FeatureFlags()
    flags.enable(FeatureFlags.ROUTELLM_ENABLED)
    flags.enable(FeatureFlags.ROUTELLM_TECHNICAL_DOMAIN)
    flags.enable(FeatureFlags.ROUTELLM_CREATIVE_DOMAIN)
    # Keep medical and legal domains on traditional routing for safety
    
    router = DomainRouter(registry, config, flags)
    
    print("Advanced Configuration:")
    print(f"  Router Type: {config.router_type.value}")
    print(f"  Default Threshold: {config.threshold}")
    print(f"  Domain Overrides: {len(config.domain_routing_overrides)} configured")
    print(f"  Cost Target: {config.cost_optimization_target:.0%} reduction")
    print()
    
    # Test domain-specific routing
    domain_tests = [
        ("Write a creative story about time travel", "creative"),
        ("Explain the legal implications of AI", "legal"),
        ("Implement a binary search algorithm", "technical"),
        ("Diagnose symptoms of chest pain", "medical"),
    ]
    
    print("Domain-Specific Routing Tests:")
    print("-" * 40)
    
    for query, expected_domain in domain_tests:
        try:
            analysis = router.analyze_text(query)
            selected_model = await router.route_by_domain(query)
            
            primary_domain = analysis['primary_domain']
            domain_enabled = flags.is_domain_enabled(primary_domain) if primary_domain else False
            
            print(f"Query: \"{query}\"")
            print(f"  Expected Domain: {expected_domain}")
            print(f"  Detected Domain: {primary_domain}")
            print(f"  Domain Enabled for RouteLLM: {domain_enabled}")
            
            if analysis.get('routellm_enabled') and domain_enabled:
                print(f"  Complexity: {analysis['complexity_score']:.3f}")
                print(f"  Recommendation: {analysis['recommended_routing']}")
                
                # Check for domain-specific overrides
                override = config.get_domain_override(primary_domain)
                if override:
                    print(f"  Domain Override: {override}")
            
            print(f"  Selected Model: {selected_model.provider}:{selected_model.name}")
            print()
            
        except Exception as e:
            logger.error(f"Error processing domain test: {e}")


async def cost_tracking_example():
    """Cost tracking and reporting example."""
    print("=" * 60)
    print("Cost Tracking and Reporting Example")
    print("=" * 60)
    
    registry = ModelRegistry()
    await registry.discover_models()
    
    config = RouteLLMConfig(
        enabled=True,
        cost_tracking_enabled=True,
        performance_monitoring_enabled=True,
    )
    
    flags = FeatureFlags()
    flags.enable(FeatureFlags.ROUTELLM_ENABLED)
    flags.enable(FeatureFlags.ROUTELLM_TECHNICAL_DOMAIN)
    
    router = DomainRouter(registry, config, flags)
    
    # Simulate multiple requests for cost tracking
    test_requests = [
        "Hello world",
        "Implement quicksort algorithm",
        "What is the capital of France?",
        "Design a scalable database architecture",
        "How to make coffee?",
        "Optimize this SQL query for performance",
        "Simple greeting response",
        "Complex distributed systems design patterns",
    ]
    
    print(f"Processing {len(test_requests)} requests for cost analysis...")
    
    for i, request in enumerate(test_requests, 1):
        try:
            selected_model = await router.route_by_domain(request)
            print(f"  {i}. Processed: \"{request[:30]}...\" -> {selected_model.name}")
        except Exception as e:
            logger.error(f"Error processing request {i}: {e}")
    
    print()
    
    # Generate cost savings report
    report = router.get_cost_savings_report(period_days=1)  # Last day
    
    if report:
        print("Cost Savings Report (Last 24 Hours):")
        print("-" * 40)
        print(f"Total Requests: {report['total_requests']}")
        print(f"RouteLLM Requests: {report['routellm_requests']}")
        print(f"Traditional Requests: {report['traditional_requests']}")
        print()
        print(f"Estimated Costs:")
        print(f"  RouteLLM: ${report.get('routellm_estimated_cost', 0):.6f}")
        print(f"  Traditional: ${report.get('traditional_estimated_cost', 0):.6f}")
        print(f"  Total: ${report.get('total_estimated_cost', 0):.6f}")
        print()
        print(f"Cost Savings: ${report['estimated_savings']:.6f} ({report['savings_percentage']:.1f}%)")
        print(f"Success Rate: {report['success_rate']:.1%}")
        print(f"Average Latency: {report['average_latency_ms']:.1f}ms")
        
        if report.get('average_quality_score'):
            print(f"Average Quality Score: {report['average_quality_score']:.2f}")
        print()
    
    # Get detailed metrics summary
    metrics = router.get_routing_metrics_summary()
    if "error" not in metrics:
        print("Detailed Metrics Summary:")
        print("-" * 40)
        print(f"Total Requests Processed: {metrics['total_requests']}")
        print(f"Overall Success Rate: {metrics['success_rate']:.1%}")
        print(f"RouteLLM Adoption Rate: {metrics['routellm_usage_rate']:.1%}")
        print(f"Average Request Cost: ${metrics['average_cost']:.6f}")
        print(f"Average Routing Latency: {metrics['average_latency_ms']:.1f}ms")


async def feature_flag_management_example():
    """Feature flag management and A/B testing example."""
    print("=" * 60)
    print("Feature Flag Management Example")
    print("=" * 60)
    
    registry = ModelRegistry()
    await registry.discover_models()
    
    config = RouteLLMConfig(enabled=True)
    flags = FeatureFlags()
    
    router = DomainRouter(registry, config, flags)
    
    # Demonstrate gradual rollout strategy
    rollout_phases = [
        {
            "phase": "Phase 1: Technical Domain Only",
            "flags": {FeatureFlags.ROUTELLM_ENABLED: True, FeatureFlags.ROUTELLM_TECHNICAL_DOMAIN: True}
        },
        {
            "phase": "Phase 2: Add Educational Domain", 
            "flags": {FeatureFlags.ROUTELLM_EDUCATIONAL_DOMAIN: True}
        },
        {
            "phase": "Phase 3: Add Creative Domain",
            "flags": {FeatureFlags.ROUTELLM_CREATIVE_DOMAIN: True}
        },
        {
            "phase": "Phase 4: Full Rollout (except high-risk domains)",
            "flags": {
                FeatureFlags.ROUTELLM_SCIENTIFIC_DOMAIN: True,
                FeatureFlags.ROUTELLM_FINANCIAL_DOMAIN: True
            }
        }
    ]
    
    test_queries_by_domain = {
        "technical": "Implement a REST API",
        "educational": "Explain photosynthesis",
        "creative": "Write a short poem",
        "scientific": "Calculate molecular weight",
        "financial": "Analyze stock performance",
        "medical": "Diagnose symptoms",  # High-risk domain
        "legal": "Contract interpretation"  # High-risk domain
    }
    
    for phase_info in rollout_phases:
        print(f"{phase_info['phase']}")
        print("-" * len(phase_info['phase']))
        
        # Update feature flags for this phase
        router.update_feature_flags(phase_info['flags'])
        
        # Test each domain
        for domain, query in test_queries_by_domain.items():
            domain_enabled = flags.is_domain_enabled(domain)
            
            try:
                analysis = router.analyze_text(query)
                selected_model = await router.route_by_domain(query)
                
                status = "🟢 RouteLLM" if (analysis.get('routellm_enabled') and domain_enabled) else "🔴 Traditional"
                
                print(f"  {domain.capitalize():12} -> {status:15} ({selected_model.name})")
                
            except Exception as e:
                print(f"  {domain.capitalize():12} -> ❌ Error: {e}")
        
        print()
    
    # Show final feature flag state
    all_flags = flags.get_all_flags()
    enabled_flags = [flag for flag, enabled in all_flags.items() if enabled]
    
    print("Final Feature Flag State:")
    print("-" * 30)
    for flag in enabled_flags:
        print(f"  ✅ {flag}")
    
    disabled_flags = [flag for flag, enabled in all_flags.items() if not enabled]
    for flag in disabled_flags:
        if "DOMAIN" in flag or flag in [FeatureFlags.ROUTELLM_ENABLED]:
            print(f"  ❌ {flag}")


async def error_handling_example():
    """Error handling and fallback demonstration."""
    print("=" * 60)
    print("Error Handling and Fallback Example")
    print("=" * 60)
    
    registry = ModelRegistry()
    await registry.discover_models()
    
    # Create configuration that might have issues
    config = RouteLLMConfig(
        enabled=True,
        timeout_seconds=1.0,  # Very short timeout to force errors
        max_retry_attempts=1,  # Minimal retries
    )
    
    flags = FeatureFlags()
    flags.enable(FeatureFlags.ROUTELLM_ENABLED)
    
    router = DomainRouter(registry, config, flags)
    
    # Check initial status
    status = router.get_routellm_status()
    print("Initial RouteLLM Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    print()
    
    # Test with various scenarios that might cause fallbacks
    test_scenarios = [
        "Simple query that should work",
        "Complex technical query about distributed systems",
        "Medical diagnosis query that might be sensitive",
        "Legal advice query that might require fallback",
    ]
    
    print("Testing Fallback Scenarios:")
    print("-" * 40)
    
    for i, query in enumerate(test_scenarios, 1):
        try:
            print(f"{i}. Processing: \"{query}\"")
            
            # This will try RouteLLM first, then fallback if needed
            selected_model = await router.route_by_domain(query)
            
            print(f"   ✅ Success: {selected_model.provider}:{selected_model.name}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print()
    
    # Show final metrics including any fallbacks
    metrics = router.get_routing_metrics_summary()
    if "error" not in metrics:
        print("Final Metrics (including fallbacks):")
        print("-" * 40)
        print(f"Total Requests: {metrics['total_requests']}")
        print(f"Success Rate: {metrics['success_rate']:.1%}")
        print(f"RouteLLM Usage Rate: {metrics['routellm_usage_rate']:.1%}")
        
        if metrics['routellm_usage_rate'] < 0.5:
            print("⚠️  High fallback rate detected - check RouteLLM configuration")


async def main():
    """Run all examples."""
    print("RouteLLM Integration Examples")
    print("=" * 60)
    print("This example demonstrates the RouteLLM integration capabilities")
    print("including intelligent routing, cost optimization, and monitoring.")
    print()
    
    examples = [
        ("Basic Integration", basic_example),
        ("Advanced Configuration", advanced_configuration_example),
        ("Cost Tracking", cost_tracking_example),
        ("Feature Flag Management", feature_flag_management_example),
        ("Error Handling", error_handling_example),
    ]
    
    for example_name, example_func in examples:
        try:
            print(f"\n🚀 Running {example_name} Example...")
            await example_func()
            print(f"✅ {example_name} completed successfully\n")
        except Exception as e:
            print(f"❌ {example_name} failed: {e}\n")
            logger.exception(f"Error in {example_name}")
    
    print("=" * 60)
    print("All examples completed!")
    print()
    print("Key Takeaways:")
    print("- RouteLLM integration is completely transparent to existing code")
    print("- Feature flags enable safe gradual rollout")
    print("- Cost tracking provides detailed insights into savings")
    print("- Fallback mechanisms ensure reliability")
    print("- Domain-specific configuration allows fine-tuned control")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())