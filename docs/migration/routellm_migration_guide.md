# RouteLLM Integration Migration Guide

## Overview

This guide provides step-by-step instructions for migrating to RouteLLM integration in the orchestrator framework. RouteLLM enables intelligent model routing with cost optimization, allowing you to achieve significant cost savings while maintaining output quality.

## What is RouteLLM?

RouteLLM is a framework for serving and evaluating LLM routers that automatically route queries to different models based on their complexity and requirements. This integration provides:

- **Cost Optimization**: Automatic routing to cost-effective models for simpler queries
- **Quality Preservation**: Intelligent routing to powerful models for complex tasks  
- **Performance Monitoring**: Comprehensive tracking of routing decisions and cost savings
- **Feature Flag Control**: Safe, gradual rollout with instant rollback capabilities

## Pre-Migration Assessment

### System Requirements

Before starting the migration, ensure your system meets these requirements:

```bash
# Check Python version (3.8+ required)
python --version

# Check if orchestrator is installed and updated
pip show orchestrator-framework

# Check if RouteLLM dependencies are available
pip list | grep routellm
```

### Compatibility Checklist

- [ ] **Python 3.8+**: RouteLLM integration requires Python 3.8 or higher
- [ ] **Model Access**: Verify access to both strong and weak models (e.g., GPT-4 and GPT-3.5)
- [ ] **API Keys**: Ensure valid API keys for model providers
- [ ] **Configuration**: Current model configuration is compatible
- [ ] **Monitoring**: Optional but recommended monitoring system access

### Current State Assessment

Run this assessment script to evaluate your current setup:

```python
# assessment_script.py
from src.orchestrator.models.routellm_integration import RouteLLMConfig, FeatureFlags
import logging

def assess_current_setup():
    """Assess current setup compatibility with RouteLLM integration."""
    assessment = {
        "compatible": True,
        "warnings": [],
        "errors": []
    }
    
    # Check configuration
    try:
        config = RouteLLMConfig()
        print(f"✓ RouteLLMConfig can be instantiated")
    except Exception as e:
        assessment["errors"].append(f"Configuration error: {e}")
        assessment["compatible"] = False
    
    # Check feature flags
    try:
        flags = FeatureFlags()
        print(f"✓ Feature flags system available")
    except Exception as e:
        assessment["errors"].append(f"Feature flags error: {e}")
        assessment["compatible"] = False
    
    return assessment

if __name__ == "__main__":
    result = assess_current_setup()
    print(f"\nAssessment: {'✓ Compatible' if result['compatible'] else '✗ Issues found'}")
    for warning in result["warnings"]:
        print(f"⚠️  {warning}")
    for error in result["errors"]:
        print(f"❌ {error}")
```

## Migration Steps

### Step 1: Install Dependencies

If not already installed, add RouteLLM dependencies:

```bash
# Add to requirements.txt or install directly
pip install routellm  # Add this if not already in dependencies

# Or update orchestrator with RouteLLM support
pip install orchestrator-framework[routellm]
```

### Step 2: Basic Configuration

Create or update your configuration file to enable RouteLLM:

```yaml
# config.yaml
model_config:
  # Your existing model configuration
  default_model: "gpt-3.5-turbo"
  
# Add RouteLLM configuration
routellm:
  enabled: false  # Start with disabled for testing
  router_type: "mf"  # Matrix factorization router
  threshold: 0.11593  # Default threshold
  strong_model: "gpt-4-1106-preview"
  weak_model: "gpt-3.5-turbo"
  
  # Cost optimization settings
  cost_tracking_enabled: true
  cost_optimization_target: 0.5  # Target 50% cost reduction
  
  # Fallback settings
  fallback_enabled: true
  max_retry_attempts: 3
  timeout_seconds: 30.0
```

Or in Python:

```python
from src.orchestrator.models.routellm_integration import RouteLLMConfig, RouterType

# Create configuration
routellm_config = RouteLLMConfig(
    enabled=False,  # Start disabled
    router_type=RouterType.MATRIX_FACTORIZATION,
    threshold=0.11593,
    strong_model="gpt-4-1106-preview",
    weak_model="gpt-3.5-turbo",
    cost_tracking_enabled=True,
    fallback_enabled=True
)
```

### Step 3: Feature Flag Setup

Enable RouteLLM through feature flags for safe rollout:

```python
from src.orchestrator.models.routellm_integration import FeatureFlags

# Initialize feature flags
flags = FeatureFlags()

# Start with minimal enablement
flags.enable(FeatureFlags.ROUTELLM_COST_TRACKING)
flags.enable(FeatureFlags.ROUTELLM_PERFORMANCE_MONITORING)

# Enable for specific domain first (optional)
flags.enable(FeatureFlags.ROUTELLM_TECHNICAL_DOMAIN)

print("Feature flags enabled:", flags.get_all_flags())
```

### Step 4: Testing Phase

Before enabling RouteLLM, test the integration:

```python
# test_routellm.py
import asyncio
from src.orchestrator.models.routellm_integration import (
    RouteLLMConfig, 
    FeatureFlags, 
    CostTracker
)

async def test_routellm_integration():
    """Test RouteLLM integration without enabling it."""
    
    # Initialize components
    config = RouteLLMConfig(enabled=False)  # Keep disabled for testing
    flags = FeatureFlags()
    tracker = CostTracker()
    
    # Test configuration validation
    print("✓ Configuration created successfully")
    
    # Test routing decision simulation
    test_queries = [
        "What is 2+2?",  # Simple query - should route to weak model
        "Explain quantum computing in detail",  # Complex query - should route to strong model
        "Write a Python function to sort a list",  # Medium complexity
    ]
    
    for query in test_queries:
        # Simulate routing decision
        routing_id = tracker.track_routing_decision(
            text=query,
            domains=["technical"],
            routing_method="simulation",
            selected_model=config.weak_model if len(query) < 50 else config.strong_model,
            estimated_cost=0.001 if len(query) < 50 else 0.003
        )
        print(f"Tracked routing decision for query: '{query[:30]}...' ID: {routing_id}")
    
    # Generate test report
    report = tracker.get_cost_savings_report(period_days=1)
    print(f"Test report - Tracked {report.total_requests} requests")

if __name__ == "__main__":
    asyncio.run(test_routellm_integration())
```

### Step 5: Gradual Rollout

Enable RouteLLM gradually using feature flags:

#### Phase 1: Monitoring Only
```python
# Enable monitoring and cost tracking first
flags = FeatureFlags()
flags.enable(FeatureFlags.ROUTELLM_COST_TRACKING)
flags.enable(FeatureFlags.ROUTELLM_PERFORMANCE_MONITORING)

# Update configuration but keep routing disabled
config = RouteLLMConfig(enabled=False, cost_tracking_enabled=True)
```

#### Phase 2: Single Domain
```python
# Enable for one domain (e.g., technical)
flags.enable(FeatureFlags.ROUTELLM_TECHNICAL_DOMAIN)

# Enable RouteLLM with domain restrictions
config = RouteLLMConfig(
    enabled=True,
    domain_specific_routing=True,
    domain_routing_overrides={
        "technical": {"enabled": True}
    }
)
```

#### Phase 3: Full Rollout
```python
# Enable globally after testing
flags.enable(FeatureFlags.ROUTELLM_ENABLED)

# Full configuration
config = RouteLLMConfig(
    enabled=True,
    cost_tracking_enabled=True,
    performance_monitoring_enabled=True
)
```

### Step 6: Monitoring and Validation

Monitor the integration after each rollout phase:

```python
# monitoring_example.py
from src.orchestrator.models.routellm_integration import CostTracker
import asyncio

async def monitor_routellm_performance():
    """Monitor RouteLLM performance and generate reports."""
    
    tracker = CostTracker()
    
    # Get daily report
    daily_report = tracker.get_cost_savings_report(period_days=1)
    print(f"Daily Report:")
    print(f"  Total requests: {daily_report.total_requests}")
    print(f"  RouteLLM requests: {daily_report.routellm_requests}")
    print(f"  Estimated savings: ${daily_report.estimated_savings:.2f}")
    print(f"  Success rate: {daily_report.success_rate:.2%}")
    
    # Get metrics summary
    summary = tracker.get_metrics_summary()
    print(f"\nMetrics Summary:")
    print(f"  Success rate: {summary['success_rate']:.2%}")
    print(f"  RouteLLM usage: {summary['routellm_usage_rate']:.2%}")
    print(f"  Average cost: ${summary['average_cost']:.4f}")
    print(f"  Average latency: {summary['average_latency_ms']:.1f}ms")
    
    return daily_report

if __name__ == "__main__":
    asyncio.run(monitor_routellm_performance())
```

## Configuration Reference

### Core Configuration Options

```python
@dataclass
class RouteLLMConfig:
    # Core routing settings
    enabled: bool = False
    router_type: RouterType = RouterType.MATRIX_FACTORIZATION
    threshold: float = 0.11593
    
    # Model configuration
    strong_model: str = "gpt-4-1106-preview"
    weak_model: str = "gpt-3.5-turbo"
    
    # Reliability settings
    fallback_enabled: bool = True
    max_retry_attempts: int = 3
    timeout_seconds: float = 30.0
    
    # Cost optimization
    cost_tracking_enabled: bool = True
    cost_optimization_target: float = 0.5
    
    # Monitoring
    performance_monitoring_enabled: bool = True
    metrics_retention_days: int = 30
    
    # Domain-specific
    domain_specific_routing: bool = True
    domain_routing_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
```

### Router Types

| Router Type | Description | Use Case |
|-------------|-------------|----------|
| `mf` (Matrix Factorization) | Default, good balance | General purpose |
| `bert` (BERT Classifier) | BERT-based classification | Text classification tasks |
| `causal_llm` | Causal language model routing | Language generation |
| `sw_ranking` | Similarity weighted ranking | Semantic similarity routing |
| `random` | Random routing | Testing/benchmarking |

### Feature Flags Reference

```python
# Core flags
ROUTELLM_ENABLED = "routellm_enabled"
ROUTELLM_COST_TRACKING = "routellm_cost_tracking"
ROUTELLM_PERFORMANCE_MONITORING = "routellm_performance_monitoring"

# Domain-specific flags
ROUTELLM_MEDICAL_DOMAIN = "routellm_medical_domain"
ROUTELLM_LEGAL_DOMAIN = "routellm_legal_domain"
ROUTELLM_TECHNICAL_DOMAIN = "routellm_technical_domain"
ROUTELLM_CREATIVE_DOMAIN = "routellm_creative_domain"
ROUTELLM_SCIENTIFIC_DOMAIN = "routellm_scientific_domain"

# Experimental flags
ROUTELLM_DYNAMIC_THRESHOLD = "routellm_dynamic_threshold"
ROUTELLM_A_B_TESTING = "routellm_a_b_testing"
```

### Environment Variables

```bash
# Core RouteLLM settings
export ROUTELLM_ENABLED=false
export ROUTELLM_ROUTER_TYPE=mf
export ROUTELLM_THRESHOLD=0.11593

# Model settings
export ROUTELLM_STRONG_MODEL=gpt-4-1106-preview  
export ROUTELLM_WEAK_MODEL=gpt-3.5-turbo

# Monitoring settings
export ROUTELLM_COST_TRACKING=true
export ROUTELLM_PERFORMANCE_MONITORING=true
```

## Migration Validation

### Validation Checklist

After migration, verify these items:

- [ ] **Configuration Loading**: RouteLLM configuration loads without errors
- [ ] **Feature Flags**: Feature flag system responds correctly
- [ ] **Model Access**: Both strong and weak models are accessible
- [ ] **Routing Logic**: Routing decisions are made appropriately
- [ ] **Cost Tracking**: Cost metrics are collected accurately
- [ ] **Monitoring**: Performance monitoring is functional
- [ ] **Fallback**: Fallback mechanisms work when needed

### Validation Script

```python
# validation_script.py
import asyncio
from src.orchestrator.models.routellm_integration import (
    RouteLLMConfig, FeatureFlags, CostTracker
)

async def validate_migration():
    """Comprehensive migration validation."""
    
    validation_results = {
        "passed": [],
        "failed": [],
        "warnings": []
    }
    
    # Test 1: Configuration
    try:
        config = RouteLLMConfig()
        validation_results["passed"].append("Configuration initialization")
    except Exception as e:
        validation_results["failed"].append(f"Configuration: {e}")
    
    # Test 2: Feature Flags
    try:
        flags = FeatureFlags()
        flags.enable(FeatureFlags.ROUTELLM_COST_TRACKING)
        assert flags.is_enabled(FeatureFlags.ROUTELLM_COST_TRACKING)
        validation_results["passed"].append("Feature flags system")
    except Exception as e:
        validation_results["failed"].append(f"Feature flags: {e}")
    
    # Test 3: Cost Tracking
    try:
        tracker = CostTracker()
        tracker_id = tracker.track_routing_decision(
            text="test query",
            domains=["test"],
            routing_method="test",
            selected_model="test-model",
            estimated_cost=0.001
        )
        assert tracker_id
        validation_results["passed"].append("Cost tracking")
    except Exception as e:
        validation_results["failed"].append(f"Cost tracking: {e}")
    
    # Display results
    print("Migration Validation Results:")
    print("=" * 40)
    
    if validation_results["passed"]:
        print("✓ PASSED:")
        for test in validation_results["passed"]:
            print(f"  ✓ {test}")
    
    if validation_results["failed"]:
        print("\n❌ FAILED:")
        for test in validation_results["failed"]:
            print(f"  ❌ {test}")
    
    if validation_results["warnings"]:
        print("\n⚠️  WARNINGS:")
        for warning in validation_results["warnings"]:
            print(f"  ⚠️  {warning}")
    
    success = len(validation_results["failed"]) == 0
    print(f"\nOverall: {'✓ SUCCESS' if success else '❌ FAILED'}")
    return success

if __name__ == "__main__":
    asyncio.run(validate_migration())
```

## Rollback Procedures

If issues occur during migration, follow these rollback steps:

### Emergency Rollback

```python
# emergency_rollback.py
from src.orchestrator.models.routellm_integration import FeatureFlags

def emergency_rollback():
    """Immediately disable RouteLLM integration."""
    
    flags = FeatureFlags()
    
    # Disable all RouteLLM features
    flags.disable(FeatureFlags.ROUTELLM_ENABLED)
    
    # Disable domain-specific routing
    for domain in ["MEDICAL", "LEGAL", "TECHNICAL", "CREATIVE", "SCIENTIFIC"]:
        flag_name = f"ROUTELLM_{domain}_DOMAIN"
        if hasattr(FeatureFlags, flag_name):
            flags.disable(getattr(FeatureFlags, flag_name))
    
    print("Emergency rollback completed. RouteLLM disabled.")
    print("Current flags:", flags.get_all_flags())

if __name__ == "__main__":
    emergency_rollback()
```

### Configuration Rollback

```bash
# Backup current configuration
cp config.yaml config.yaml.backup

# Restore previous configuration
git checkout HEAD~1 -- config.yaml

# Or manually remove RouteLLM section
# Edit config.yaml and remove the 'routellm:' section
```

### Gradual Rollback

For gradual rollback, reverse the rollout phases:

1. **Disable global RouteLLM**: `flags.disable(FeatureFlags.ROUTELLM_ENABLED)`
2. **Disable domain-specific routing**: Remove domain flags one by one
3. **Keep monitoring enabled**: Maintain monitoring for analysis
4. **Remove configuration**: Eventually remove RouteLLM configuration

## Troubleshooting

### Common Issues

#### Issue: RouteLLM not routing correctly
```python
# Check configuration
config = RouteLLMConfig()
print("Router string:", config.get_router_model_string())
print("Threshold:", config.threshold)

# Check feature flags
flags = FeatureFlags()
print("RouteLLM enabled:", flags.is_enabled(FeatureFlags.ROUTELLM_ENABLED))
```

#### Issue: High costs/poor routing decisions
```python
# Analyze cost tracking data
tracker = CostTracker()
report = tracker.get_cost_savings_report(period_days=7)

if report.estimated_savings < 0:
    print("⚠️  RouteLLM is increasing costs")
    # Consider adjusting threshold or router type
```

#### Issue: Fallback being used frequently
```python
# Check metrics for fallback usage
summary = tracker.get_metrics_summary()
if summary['success_rate'] < 0.95:
    print("⚠️  High fallback usage detected")
    # Check timeout settings and model availability
```

### Support Resources

- **Configuration Validation**: Use the provided validation scripts
- **Monitoring Dashboard**: Check cost and performance metrics
- **Feature Flags**: Use feature flags for safe experimentation
- **Logging**: Enable detailed logging for debugging

## Expected Benefits

After successful migration, you should see:

- **Cost Reduction**: 20-50% reduction in model costs
- **Performance Maintenance**: Quality maintained for complex tasks
- **Operational Insight**: Detailed routing and cost analytics
- **Flexibility**: Fine-tuned control over routing behavior

## Next Steps

1. **Optimize Thresholds**: Fine-tune routing thresholds based on your use cases
2. **Domain Specialization**: Configure domain-specific routing rules
3. **A/B Testing**: Use experimental flags for testing optimizations
4. **Integration**: Integrate with your monitoring and alerting systems

This migration guide provides a comprehensive approach to adopting RouteLLM integration safely and effectively. Follow the phases carefully and monitor performance at each step to ensure successful adoption.