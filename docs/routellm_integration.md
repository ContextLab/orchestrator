# RouteLLM Integration Documentation

## Overview

The RouteLLM integration provides intelligent model routing and cost optimization capabilities for the orchestrator framework. By leveraging the RouteLLM SDK, the system can achieve 40-85% cost reductions while maintaining 95% GPT-4 performance quality through intelligent query complexity assessment and automatic model routing.

## Features

### Core Capabilities
- **Intelligent Routing**: Automatic query complexity assessment to route between strong and weak models
- **Cost Optimization**: Up to 85% cost reduction while maintaining quality
- **Seamless Integration**: Drop-in enhancement to existing domain router functionality
- **Complete API Compatibility**: Zero breaking changes to existing interfaces
- **Feature Flags**: Safe gradual rollout with per-domain controls
- **Cost Tracking**: Comprehensive monitoring and reporting of routing decisions
- **Fallback Mechanisms**: Robust error handling with automatic fallbacks

### Supported Router Types
- **Matrix Factorization** (mf) - Recommended default router
- **BERT Classifier** (bert) - Text classification-based routing
- **Causal LLM** (causal_llm) - Large language model-based classification
- **Similarity Weighted** (sw_ranking) - Weighted Elo calculation
- **Random** (random) - Random routing for testing

## Installation

RouteLLM integration is included by default with orchestrator. The dependency is automatically installed:

```bash
pip install "routellm[serve,eval]>=0.1.0"
```

## Configuration

### Basic Configuration

```python
from orchestrator.models.routellm_integration import RouteLLMConfig, RouterType, FeatureFlags
from orchestrator.models.domain_router import DomainRouter
from orchestrator.models.model_registry import ModelRegistry

# Create RouteLLM configuration
config = RouteLLMConfig(
    enabled=True,
    router_type=RouterType.MATRIX_FACTORIZATION,
    strong_model="gpt-4-1106-preview",
    weak_model="gpt-3.5-turbo",
    threshold=0.11593,  # Default threshold for 50% routing to GPT-4
    cost_tracking_enabled=True,
    performance_monitoring_enabled=True
)

# Configure feature flags for gradual rollout
flags = FeatureFlags()
flags.enable(FeatureFlags.ROUTELLM_ENABLED)
flags.enable(FeatureFlags.ROUTELLM_TECHNICAL_DOMAIN)  # Start with technical domain only

# Create domain router with RouteLLM integration
registry = ModelRegistry()
router = DomainRouter(registry, config, flags)
```

### Advanced Configuration

```python
from orchestrator.models.routellm_integration import RouteLLMConfig, RouterType

# Advanced configuration with domain-specific overrides
config = RouteLLMConfig(
    enabled=True,
    router_type=RouterType.BERT_CLASSIFIER,
    strong_model="gpt-4-turbo",
    weak_model="gpt-3.5-turbo",
    threshold=0.15,  # Higher threshold = more conservative routing
    
    # Domain-specific routing overrides
    domain_routing_overrides={
        "medical": {
            "threshold": 0.05,  # Lower threshold = prefer strong model for medical
            "strong_model": "gpt-4-medical"  # Domain-specific strong model
        },
        "legal": {
            "threshold": 0.08,  # Conservative routing for legal domain
        },
        "creative": {
            "threshold": 0.25,  # Liberal routing for creative tasks
        }
    },
    
    # Performance tuning
    max_retry_attempts=3,
    timeout_seconds=30.0,
    cost_optimization_target=0.6,  # Target 60% cost reduction
    
    # Monitoring
    metrics_retention_days=90,  # Keep metrics for 90 days
)
```

## Usage Examples

### Basic Usage (Transparent Integration)

The RouteLLM integration is completely transparent to existing code:

```python
# Existing code continues to work unchanged
router = DomainRouter(registry)

# Route as normal - RouteLLM integration happens automatically if enabled
selected_model = await router.route_by_domain("Explain quantum computing concepts")

# The router will:
# 1. Detect domains (e.g., "scientific", "educational") 
# 2. Check if RouteLLM should be used for this domain
# 3. Use RouteLLM to assess query complexity
# 4. Route to appropriate model (strong vs weak)
# 5. Fall back to traditional domain routing if needed
```

### Monitoring and Reporting

```python
# Get RouteLLM integration status
status = router.get_routellm_status()
print(f"RouteLLM enabled: {status['config_enabled']}")
print(f"Controller available: {status['controller_available']}")
print(f"Cost tracking: {status['cost_tracking_enabled']}")

# Get cost savings report
report = router.get_cost_savings_report(period_days=30)
print(f"Total requests: {report['total_requests']}")
print(f"Cost savings: {report['savings_percentage']:.1f}%")
print(f"Estimated savings: ${report['estimated_savings']:.4f}")

# Get routing metrics summary
metrics = router.get_routing_metrics_summary()
print(f"RouteLLM usage rate: {metrics['routellm_usage_rate']:.1%}")
print(f"Success rate: {metrics['success_rate']:.1%}")
print(f"Average latency: {metrics['average_latency_ms']:.1f}ms")
```

### Dynamic Configuration Updates

```python
# Update RouteLLM configuration at runtime
new_config = RouteLLMConfig(
    enabled=True,
    router_type=RouterType.CAUSAL_LLM,  # Switch to different router type
    threshold=0.2  # More conservative routing
)
router.update_routellm_config(new_config)

# Update feature flags for A/B testing
new_flags = {
    FeatureFlags.ROUTELLM_MEDICAL_DOMAIN: True,  # Enable for medical domain
    FeatureFlags.ROUTELLM_LEGAL_DOMAIN: False,  # Disable for legal domain
}
router.update_feature_flags(new_flags)
```

### Text Analysis with RouteLLM Insights

```python
# Analyze text with RouteLLM complexity assessment
analysis = router.analyze_text("Implement a distributed microservices architecture")

print(f"Detected domains: {analysis['detected_domains']}")
print(f"Complexity score: {analysis['complexity_score']:.2f}")
print(f"Recommended routing: {analysis['recommended_routing']}")
print(f"Routing confidence: {analysis['routing_confidence']:.2f}")

# Output example:
# Detected domains: [{'domain': 'technical', 'confidence': 0.95}]
# Complexity score: 0.82
# Recommended routing: strong_model
# Routing confidence: 0.96
```

## Feature Flags for Gradual Rollout

### Global Feature Flags

```python
from orchestrator.models.routellm_integration import FeatureFlags

flags = FeatureFlags()

# Core feature flags
flags.enable(FeatureFlags.ROUTELLM_ENABLED)  # Master switch
flags.enable(FeatureFlags.ROUTELLM_COST_TRACKING)  # Cost tracking
flags.enable(FeatureFlags.ROUTELLM_PERFORMANCE_MONITORING)  # Performance monitoring
```

### Domain-Specific Rollout

```python
# Enable RouteLLM for specific domains gradually
flags.enable(FeatureFlags.ROUTELLM_TECHNICAL_DOMAIN)  # Start with technical
flags.enable(FeatureFlags.ROUTELLM_EDUCATIONAL_DOMAIN)  # Add educational

# Keep high-risk domains on traditional routing initially
assert not flags.is_enabled(FeatureFlags.ROUTELLM_MEDICAL_DOMAIN)
assert not flags.is_enabled(FeatureFlags.ROUTELLM_LEGAL_DOMAIN)

# Check domain-specific enablement
print(f"Technical domain enabled: {flags.is_domain_enabled('technical')}")
print(f"Medical domain enabled: {flags.is_domain_enabled('medical')}")
```

### Experimental Features

```python
# Enable experimental features for testing
flags.enable(FeatureFlags.ROUTELLM_DYNAMIC_THRESHOLD)  # Dynamic threshold adjustment
flags.enable(FeatureFlags.ROUTELLM_A_B_TESTING)  # A/B testing capabilities
flags.enable(FeatureFlags.ROUTELLM_QUALITY_FEEDBACK)  # Quality feedback loop
```

## Cost Tracking and Analytics

### Cost Tracking Configuration

```python
from orchestrator.models.routellm_integration import CostTracker

# Cost tracking is automatic when enabled
router = DomainRouter(registry, RouteLLMConfig(cost_tracking_enabled=True))

# Manually access cost tracker for advanced usage
if router.cost_tracker:
    # Get detailed metrics
    all_metrics = router.cost_tracker.metrics
    for metric in all_metrics[-5:]:  # Last 5 requests
        print(f"Request {metric.tracking_id[:8]}: "
              f"{metric.routing_method} -> {metric.selected_model} "
              f"(${metric.estimated_cost:.4f})")
```

### Cost Savings Reports

```python
# Generate comprehensive cost savings report
report = router.get_cost_savings_report(period_days=7)  # Last 7 days

print("=== Weekly Cost Savings Report ===")
print(f"Period: {report['period_days']} days")
print(f"Total requests: {report['total_requests']}")
print(f"RouteLLM requests: {report['routellm_requests']}")
print(f"Traditional requests: {report['traditional_requests']}")
print()
print(f"Estimated total cost: ${report['total_requests'] * 0.01:.4f}")  # Baseline
print(f"Actual estimated cost: ${report['routellm_estimated_cost'] + report['traditional_estimated_cost']:.4f}")
print(f"Cost savings: ${report['estimated_savings']:.4f} ({report['savings_percentage']:.1f}%)")
print()
print(f"Success rate: {report['success_rate']:.1%}")
print(f"Average latency: {report['average_latency_ms']:.1f}ms")
if report['average_quality_score']:
    print(f"Average quality: {report['average_quality_score']:.2f}")
```

## Error Handling and Fallbacks

The RouteLLM integration includes comprehensive error handling:

### Automatic Fallbacks

1. **Import Failure**: If RouteLLM SDK is not available, falls back to traditional routing
2. **Controller Initialization**: If RouteLLM controller fails to initialize, disables RouteLLM
3. **API Errors**: If RouteLLM API calls fail, falls back to domain selector
4. **Model Unavailable**: If recommended model is unavailable, selects alternative
5. **Timeout**: If routing takes too long, falls back with timeout protection

### Error Tracking

```python
# Check for integration issues
status = router.get_routellm_status()

if not status['controller_available']:
    print(f"RouteLLM issue: {status['controller_error']}")
    print("Falling back to traditional domain routing")

# Monitor fallback usage in cost tracking
report = router.get_cost_savings_report()
fallback_requests = report['traditional_requests']
total_requests = report['total_requests']
fallback_rate = fallback_requests / total_requests if total_requests > 0 else 0

if fallback_rate > 0.1:  # More than 10% fallback
    print(f"Warning: High fallback rate {fallback_rate:.1%}")
```

## Performance Considerations

### Latency Impact

RouteLLM routing adds minimal latency:
- Typical routing decision: 10-50ms
- Complexity assessment: 5-15ms
- Network call to RouteLLM: 20-100ms (cached)
- Total overhead: Usually < 100ms

### Optimization Tips

1. **Use Appropriate Thresholds**: Higher thresholds favor strong models (higher cost, better quality)
2. **Enable Domain-Specific Routing**: Configure different thresholds per domain
3. **Monitor Performance**: Use built-in monitoring to tune thresholds
4. **Gradual Rollout**: Start with low-risk domains and expand gradually

### Caching and Performance

```python
# RouteLLM includes built-in caching for routing decisions
config = RouteLLMConfig(
    enabled=True,
    timeout_seconds=15.0,  # Shorter timeout for faster fallback
    max_retry_attempts=2,  # Fewer retries for faster response
)
```

## Troubleshooting

### Common Issues

1. **RouteLLM Not Available**
   ```python
   status = router.get_routellm_status()
   if not status['controller_available']:
       print("RouteLLM SDK not installed or import failed")
       print("Install with: pip install 'routellm[serve,eval]'")
   ```

2. **High Fallback Rate**
   ```python
   metrics = router.get_routing_metrics_summary()
   if metrics.get('routellm_usage_rate', 0) < 0.5:
       print("Check feature flags and domain enablement")
       print("Verify RouteLLM configuration")
   ```

3. **Cost Tracking Not Working**
   ```python
   if not router.cost_tracker:
       print("Cost tracking disabled in configuration")
       config.cost_tracking_enabled = True
   ```

### Debug Logging

Enable debug logging to troubleshoot routing decisions:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("orchestrator.models.domain_router")
logger.setLevel(logging.DEBUG)

# Now routing decisions will be logged in detail
selected_model = await router.route_by_domain("Test query")
```

## Migration Guide

### Upgrading Existing Code

No code changes are required for basic integration:

```python
# Before (existing code)
router = DomainRouter(registry)
model = await router.route_by_domain(text)

# After (automatically gets RouteLLM benefits when enabled)
router = DomainRouter(registry)  # Same initialization
model = await router.route_by_domain(text)  # Same API call
```

### Enabling RouteLLM for Existing Projects

1. **Add Configuration** (optional):
   ```python
   config = RouteLLMConfig(enabled=True)
   flags = FeatureFlags()
   flags.enable(FeatureFlags.ROUTELLM_ENABLED)
   
   router = DomainRouter(registry, config, flags)
   ```

2. **Enable Gradually**:
   ```python
   # Week 1: Enable for technical domain only
   flags.enable(FeatureFlags.ROUTELLM_TECHNICAL_DOMAIN)
   
   # Week 2: Add educational domain
   flags.enable(FeatureFlags.ROUTELLM_EDUCATIONAL_DOMAIN)
   
   # Week 3: Add creative domain
   flags.enable(FeatureFlags.ROUTELLM_CREATIVE_DOMAIN)
   ```

3. **Monitor and Tune**:
   ```python
   # Check performance weekly
   report = router.get_cost_savings_report(period_days=7)
   if report['savings_percentage'] < 30:
       # Tune threshold for more aggressive routing
       config.threshold = 0.2  # Higher threshold
   ```

## API Reference

### RouteLLMConfig

```python
@dataclass
class RouteLLMConfig:
    enabled: bool = False
    router_type: RouterType = RouterType.MATRIX_FACTORIZATION
    threshold: float = 0.11593
    strong_model: str = "gpt-4-1106-preview"
    weak_model: str = "gpt-3.5-turbo"
    fallback_enabled: bool = True
    max_retry_attempts: int = 3
    timeout_seconds: float = 30.0
    cost_tracking_enabled: bool = True
    cost_optimization_target: float = 0.5
    performance_monitoring_enabled: bool = True
    metrics_retention_days: int = 30
    domain_specific_routing: bool = True
    domain_routing_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
```

### DomainRouter Enhanced Methods

```python
class DomainRouter:
    # Enhanced methods for RouteLLM integration
    def get_routellm_config(self) -> RouteLLMConfig
    def update_routellm_config(self, config: RouteLLMConfig) -> None
    def get_feature_flags(self) -> FeatureFlags
    def update_feature_flags(self, flags: Dict[str, bool]) -> None
    def get_cost_savings_report(self, period_days: int = 30) -> Optional[Dict[str, Any]]
    def get_routing_metrics_summary(self) -> Dict[str, Any]
    def is_routellm_enabled(self) -> bool
    def get_routellm_status(self) -> Dict[str, Any]
    
    # Original methods work unchanged
    async def route_by_domain(self, text: str, base_criteria: Optional[ModelSelectionCriteria] = None, domain_override: Optional[str] = None) -> Model
    def detect_domains(self, text: str, threshold: float = 0.3) -> List[Tuple[str, float]]
    def analyze_text(self, text: str) -> Dict[str, Any]  # Enhanced with RouteLLM insights
```

## Best Practices

### Production Deployment

1. **Start Conservative**: Begin with higher thresholds to ensure quality
2. **Monitor Closely**: Watch cost savings and quality metrics
3. **Gradual Rollout**: Enable domains one by one
4. **Test Thoroughly**: Validate routing decisions with your specific use cases
5. **Plan Fallbacks**: Ensure traditional routing works if RouteLLM fails

### Cost Optimization

1. **Tune Thresholds**: Lower thresholds = more weak model usage = higher savings
2. **Domain-Specific Tuning**: Different thresholds for different domains
3. **Monitor Quality**: Ensure cost savings don't compromise output quality
4. **Regular Review**: Weekly review of cost savings reports

### Quality Assurance

1. **Quality Scoring**: Implement response quality scoring when possible
2. **User Feedback**: Collect user satisfaction metrics
3. **A/B Testing**: Compare RouteLLM vs traditional routing performance
4. **Regular Audits**: Periodically review routing decisions manually

This integration provides a powerful way to optimize model usage costs while maintaining quality, with complete backward compatibility and comprehensive monitoring capabilities.